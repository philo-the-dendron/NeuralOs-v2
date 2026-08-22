//! NIR — the Neuromorphic Intermediate Representation, slice 1:
//! import (JSON container) + honest export for the node subset the
//! substrate can honor exactly: **`Input`, `Linear`, `LIF`, `Output`**.
//!
//! # Provenance (the doctrine: layouts come from the reference VERBATIM)
//!
//! Reference implementation: `neuromorphs/NIR` @
//! `7883c3c85f1be27ed113ccc9e8d6ab47ab541df4` (BSD-3, read 2026-08-21;
//! clone pinned at `nir-ref/`, gitignored). The dict schema implemented
//! here is the reference's `NIRNode::to_dict()` / `dict2NIRNode` shape
//! (`nir/ir/{node,graph,neuron,linear}.py`):
//!
//! ```text
//! { "version": <string>,                     // exactly one version block
//!   "node": { "type": "NIRGraph",
//!             "edges": [[src, dst], ...],     // node-name pairs
//!             "nodes": { <name>: { "type": "LIF"|"Linear"|"Input"|"Output",
//!                                   ...kind fields... },
//!                        ... } } }
//! ```
//!
//! The reference's own file container at this sha is HDF5
//! (`nir/serialization.py`, h5py); the dict schema is
//! container-independent and the JSON encoding of it is what this
//! module reads and writes (the historical NIR container). HDF5 `.nir`
//! file IO is slice-2 work and lands std-side (needs the hdf5 crate).
//!
//! LIF (`nir/ir/neuron.py`): `tau` [s], `r` [Ω], `v_leak` [V],
//! `v_threshold` [V], `v_reset` [V] — per-neuron arrays; `v_reset`
//! may be absent (defaults to zeros, reference `from_dict`
//! semantics). Linear (`nir/ir/linear.py`): `weight` (2-D,
//! `y = W·x`, rows = outputs). Input/Output: `shape`.
//!
//! # The quantization contract (THE design axis — loud by design)
//!
//! NIR numbers are float; the substrate is i16 fixed-point. Every
//! float→i16 hop is recorded in the node's quant record and rendered
//! into the export's `metadata.neuralos` block (`provenance` = source
//! floats, `quant` = the derived integers' scales/errors):
//!
//! - **Linear**: `q = round(w / scale)`, `scale = absmax/32767`
//!   (dequant `w' = q · scale`), `max_abs_err = max|w − w'|`. An
//!   all-zero tensor keeps `scale = 1` and a note.
//! - **LIF**: potentials → voltage quanta (`×1000` on the mV grid,
//!   `×100_000` on centi-mV), `tau` → μs, `r` → MΩ; rounding errors
//!   recorded. **Hard failures, never silent:** `tau ≤ 0`, `tau < dt`,
//!   threshold quantizing to 0, any potential outside the membrane
//!   bounds, `r ≤ 0` or outside u16 MΩ, non-finite numbers anywhere.
//! - `dt` is an explicit import argument (NIR LIF carries no
//!   timestep); the derived integers + `dt_us` live in the record.
//!
//! Export renders the **derived** values (substrate-exact integers
//! converted back to source units exactly, dequantized weights) so
//! importing an export reproduces the identical quantized node — the
//! idempotence gate. Provenance rides `metadata`, which the reference
//! round-trips natively. The scale algebra is exact: the max-|w|
//! element maps to ±32767 by construction, so re-import recovers
//! `scale` bit-exactly and `round(q·scale/scale) = q` (the double
//! rounding error is ≤ 1 ulp ≈ 7e-12 of full scale, five orders below
//! the 0.5 rounding boundary).
//!
//! # `no_std`, zero-alloc, zero new deps
//!
//! Everything is buffer-based, like [`crate::bridge`]: the caller owns
//! all memory. Two-pass protocol — [`nir_scan`] counts,
//! [`NirBuffers`] + [`nir_import`] fills (the import itself walks the
//! document twice: nodes, then edges — edge endpoints may appear
//! before their nodes in key order). During import each Linear's
//! source f64 weights stage into [`NirBuffers::scratch`] (one slot
//! per cell) and quantize into the weight arena, so both must hold
//! the total weight count; the arena keeps exactly the quantized
//! result. Export writes into a caller byte slice. Strings are
//! borrowed from the input (schema strings must be escape-free
//! printable ASCII — loud error otherwise). `f64` appears in the
//! setup path only, never in any per-step hot path.
//!
//! The structured-entry seam: callers that already hold materialized
//! values (HDF5 import, builders) skip the JSON reader and call
//! [`quantize_linear`] / [`quantize_lif`] directly — the same
//! quantization contract, the same errors, arena placement included —
//! or assemble the whole graph in memory with [`NirBuilder`] (std).
//! The printable-ASCII string gate is a property of the JSON
//! container: it fires at read and at [`nir_export`], never on the
//! typed surface.

#![allow(clippy::module_name_repetitions)]
// `tau_s` vs `tau_us`, `r_ohm` vs `r_mohm`: the unit suffixes ARE the
// distinction — domain names, not near-duplicates.
#![allow(clippy::similar_names)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::lif_neuron::{VoltageResolution, MEMBRANE_MV_MAX, MEMBRANE_MV_MIN};
use core::fmt::Write as _;

/// The pinned reference commit this schema derives from (provenance
/// string for exports and reports).
pub const NIR_REF_SHA: &str = "7883c3c85f1be27ed113ccc9e8d6ab47ab541df4";

/// The version string our exports carry (the ONE version block).
/// States the reference sha the schema is pinned to.
pub const EXPORT_VERSION: &str = "nir@7883c3c";

/// i16 weight full-scale used by the Linear quantizer.
const I16_FS: f64 = 32_767.0;

// ---------------------------------------------------------------------------
// Errors — loud, no clamping, no guessing
// ---------------------------------------------------------------------------

/// Everything that can go wrong importing/exporting NIR. Copy +
/// borrowed strings only (no alloc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NirError<'a> {
    /// Malformed JSON at byte `pos`.
    Json(usize),
    /// A string we must read (type/name/version/edge endpoint) carries
    /// a JSON escape or non-ASCII byte — outside the documented subset.
    EscapedOrNonAsciiString(usize),
    /// A node name outside the escape-free printable-ASCII subset the
    /// JSON container can write (carries the name). The
    /// structured-entry builder accepts any `&str`; this gate fires
    /// only at the JSON boundary — [`nir_export`].
    NonAsciiNodeName(&'a str),
    /// A node kind outside the slice-1 subset (includes `Affine`,
    /// `CubaLIF`, `Conv1d`, …). Carries the kind name.
    UnsupportedNodeKind(&'a str),
    /// A required field is missing from a node/graph object.
    MissingField(&'static str),
    /// A field has the wrong JSON shape (e.g. scalar where the
    /// reference emits an array, weight not 2-D).
    BadShape(&'static str),
    /// `tau ≤ 0`, `r ≤ 0`/out of range, or a non-finite number.
    BadNumber(&'static str),
    /// `tau < dt` — the derived decay would be nonsense.
    TauBelowDt,
    /// `v_threshold` quantized to 0 quanta — a deaf neuron.
    ThresholdZero,
    /// A potential quantizes outside `[MEMBRANE_MV_MIN, MEMBRANE_MV_MAX]`.
    PotentialOutOfRange(&'static str),
    /// More nodes/edges/weights than the caller's buffers hold.
    BufferOverflow,
    /// An edge endpoint names no node.
    UnknownEdgeEndpoint(&'a str),
    /// The same edge appears twice (reference `validate_structure`).
    DuplicateEdge,
    /// A node name appears twice in `nodes`.
    DuplicateNodeName,
    /// Topology outside the slice-1 assembly (the format layer still
    /// imports it; only [`NirImport::build_chain_network`] rejects —
    /// e.g. a LIF population whose size ≠ the feeding Linear's
    /// rows).
    UnsupportedTopology(&'static str),
    /// The export byte buffer is too small.
    ExportTooSmall,
}

impl core::fmt::Display for NirError<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Json(p) => write!(f, "malformed JSON at byte {p}"),
            Self::EscapedOrNonAsciiString(p) => write!(
                f,
                "string at byte {p} uses escapes/non-ASCII — outside the documented subset"
            ),
            Self::NonAsciiNodeName(n) => write!(
                f,
                "node name '{n}' outside the printable-ASCII subset the JSON container writes"
            ),
            Self::UnsupportedNodeKind(k) => write!(
                f,
                "node kind '{k}' outside the slice-1 subset (Input/Linear/LIF/Output)"
            ),
            Self::MissingField(n) => write!(f, "missing required field '{n}'"),
            Self::BadShape(n) => write!(f, "field '{n}' has the wrong shape"),
            Self::BadNumber(n) => {
                write!(f, "non-finite or out-of-range value in '{n}'")
            }
            Self::TauBelowDt => write!(f, "tau < dt — derived decay would be nonsense"),
            Self::ThresholdZero => write!(f, "v_threshold quantizes to 0 — a deaf neuron"),
            Self::PotentialOutOfRange(n) => {
                write!(f, "'{n}' quantizes outside the membrane bounds")
            }
            Self::BufferOverflow => write!(f, "caller buffers too small for this graph"),
            Self::UnknownEdgeEndpoint(n) => write!(f, "edge endpoint '{n}' names no node"),
            Self::DuplicateEdge => write!(f, "duplicate edge"),
            Self::DuplicateNodeName => write!(f, "duplicate node name"),
            Self::UnsupportedTopology(n) => {
                write!(f, "topology unsupported by slice 1: {n}")
            }
            Self::ExportTooSmall => write!(f, "export byte buffer too small"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NirError<'_> {}

// ---------------------------------------------------------------------------
// JSON reader — the documented subset, zero alloc, borrowed strings
// ---------------------------------------------------------------------------

/// Maximum container nesting [`Reader::skip_value`] will walk before
/// rejecting the document. The schema's own depth is ≤ 6; 64 leaves
/// generous headroom for foreign `metadata` while bounding the
/// recursion (no stack overflow on adversarial input).
const MAX_SKIP_DEPTH: usize = 64;

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }

    fn ws(&mut self) {
        while let Some(&c) = self.b.get(self.pos) {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.b.get(self.pos).copied()
    }

    fn eat(&mut self, c: u8) -> Result<(), NirError<'static>> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(NirError::Json(self.pos))
        }
    }

    fn eat_lit(&mut self, lit: &str) -> Result<(), NirError<'static>> {
        if self.b[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(NirError::Json(self.pos))
        }
    }

    /// A string we must READ: escape-free printable ASCII, borrowed.
    /// (`\` is 0x5c — printable — so it is checked explicitly first.)
    fn read_string(&mut self) -> Result<&'a str, NirError<'static>> {
        self.eat(b'"')?;
        let start = self.pos;
        loop {
            match self.b.get(self.pos) {
                None => return Err(NirError::Json(self.pos)),
                Some(b'"') => break,
                Some(&c) if c != b'\\' && (0x20..=0x7e).contains(&c) => self.pos += 1,
                Some(_) => return Err(NirError::EscapedOrNonAsciiString(self.pos)),
            }
        }
        let s = core::str::from_utf8(&self.b[start..self.pos])
            .map_err(|_| NirError::EscapedOrNonAsciiString(start))?;
        self.pos += 1; // closing quote
        Ok(s)
    }

    /// A number as f64 (core's correctly-rounded parser — the same
    /// values Python's `float()` produces for the same token).
    /// Grammar note: this accepts Rust's `f64::FromStr` set, a slight
    /// SUPERSET of JSON's (`.5`, `1.`, `+5`, `007` parse here, are
    /// errors in Python's `json`). The reference emitter never
    /// produces these forms and the numeric values agree wherever
    /// both accept — a documented laxity, not an accident.
    fn read_number(&mut self) -> Result<f64, NirError<'static>> {
        self.ws();
        let start = self.pos;
        while let Some(&c) = self.b.get(self.pos) {
            if matches!(c, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if start == self.pos {
            return Err(NirError::Json(self.pos));
        }
        let tok =
            core::str::from_utf8(&self.b[start..self.pos]).map_err(|_| NirError::Json(start))?;
        tok.parse::<f64>().map_err(|_| NirError::Json(start))
    }

    /// Array-element stepper: `first` gates the opening `[`; returns
    /// `true` when an element follows (caller parses it), `false` at
    /// `]`.
    fn array_step(&mut self, first: &mut bool) -> Result<bool, NirError<'static>> {
        if *first {
            self.eat(b'[')?;
            *first = false;
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Ok(false);
            }
            Ok(true)
        } else if self.peek() == Some(b',') {
            self.pos += 1;
            Ok(true)
        } else if self.peek() == Some(b']') {
            self.pos += 1;
            Ok(false)
        } else {
            Err(NirError::Json(self.pos))
        }
    }

    /// Object-key stepper: yields the next key (value left for the
    /// caller to parse/skip), `None` at `}`. Duplicate keys:
    /// scalar fields are last-wins (Python `json.loads` semantics);
    /// repeated container fields (`edges`, `nodes`) are visited in
    /// order and their contents aggregate. Our exports never repeat
    /// a key; the reference emitter neither.
    fn object_step(&mut self, first: &mut bool) -> Result<Option<&'a str>, NirError<'static>> {
        if *first {
            self.eat(b'{')?;
            *first = false;
            if self.peek() == Some(b'}') {
                self.pos += 1;
                return Ok(None);
            }
        } else if self.peek() == Some(b',') {
            self.pos += 1;
        } else if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(None);
        } else {
            return Err(NirError::Json(self.pos));
        }
        let key = self.read_string()?;
        self.eat(b':')?;
        Ok(Some(key))
    }

    /// Skip one value of any shape (unknown fields, `metadata`).
    /// Depth-capped: nesting beyond [`MAX_SKIP_DEPTH`] is rejected as
    /// malformed instead of overflowing the call stack (a 50k-deep
    /// junk array is adversarial input, not a document).
    fn skip_value(&mut self, depth: usize) -> Result<(), NirError<'static>> {
        if depth > MAX_SKIP_DEPTH {
            return Err(NirError::Json(self.pos));
        }
        match self.peek() {
            Some(b'"') => {
                self.pos += 1;
                while let Some(&c) = self.b.get(self.pos) {
                    self.pos += 1;
                    if c == b'\\' {
                        let _ = self.b.get(self.pos).ok_or(NirError::Json(self.pos))?;
                        self.pos += 1;
                    } else if c == b'"' {
                        return Ok(());
                    }
                }
                Err(NirError::Json(self.pos))
            }
            Some(b'{') => {
                let mut first = true;
                while self.object_step(&mut first)?.is_some() {
                    self.skip_value(depth + 1)?;
                }
                Ok(())
            }
            Some(b'[') => {
                let mut first = true;
                while self.array_step(&mut first)? {
                    self.skip_value(depth + 1)?;
                }
                Ok(())
            }
            Some(b't') => self.eat_lit("true"),
            Some(b'f') => self.eat_lit("false"),
            Some(b'n') => self.eat_lit("null"),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.read_number().map(|_| ()),
            _ => Err(NirError::Json(self.pos)),
        }
    }
}

// ---------------------------------------------------------------------------
// Schema types
// ---------------------------------------------------------------------------

/// Node kinds of the slice-1 subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NirNodeKind {
    /// Virtual input plumbing (shape carrier).
    Input,
    /// Virtual output plumbing (shape carrier).
    Output,
    /// `y = W·x` — quantized into the weight arena.
    Linear,
    /// Leaky integrate-and-fire — quantized onto the voltage grid.
    Lif,
}

/// A quantized LIF node: provenance floats + derived substrate
/// integers. Export renders the derived fields; provenance rides
/// `metadata`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NirLif {
    // provenance (source units: s, Ω, V)
    pub tau_s: f64,
    pub r_ohm: f64,
    pub v_leak_v: f64,
    pub v_threshold_v: f64,
    pub v_reset_v: f64,
    pub v_reset_defaulted: bool,
    // derived (substrate units)
    pub tau_us: u32,
    pub resistance_mohm: u16,
    pub capacitance_pf: u16,
    pub capacitance_clamped: bool,
    pub leak_q: i16,
    pub threshold_q: i16,
    pub reset_q: i16,
    // quantization record
    pub tau_err_s: f64,
    pub max_v_err_v: f64,
}

/// A quantized LIF population view: `len` neurons with per-neuron
/// records in the caller's lifs buffer ([`NirBuffers::lifs`] or
/// [`NirImport::lifs`]), starting at `offset`. A length-1
/// population is the slice-1 single neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NirLifPopulation {
    /// First record index in the lifs buffer.
    pub offset: usize,
    /// Population size (the per-neuron param array length).
    pub len: usize,
}

/// Per-neuron source-unit LIF parameters (population form) for the
/// structured-entry builder: every slice must have equal length
/// ≥ 1; `v_reset_v` `None` = the reference's absent-`v_reset`
/// zeros semantics.
#[derive(Debug, Clone, Copy)]
pub struct NirLifParams<'v> {
    pub tau_s: &'v [f64],
    pub r_ohm: &'v [f64],
    pub v_leak_v: &'v [f64],
    pub v_threshold_v: &'v [f64],
    pub v_reset_v: Option<&'v [f64]>,
}

/// A quantized Linear node: weights live in the shared i16 arena.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NirLinear {
    pub rows: usize,
    pub cols: usize,
    /// Arena view `[offset, offset + rows·cols)`, row-major
    /// (`weight[out][in]`, the reference layout).
    pub weight_offset: usize,
    /// Dequant factor: `w' = q · scale`.
    pub scale: f64,
    /// Source tensor max |w| (provenance).
    pub absmax: f64,
    /// `max |w − q·scale|` over the tensor.
    pub max_abs_err: f64,
    pub zero_tensor: bool,
}

/// One imported node. Shapes are `[u32; 4]` + length (≤ 4 dims,
/// reference shapes are 1–4-D). A LIF node carries a population
/// view into the lifs buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NirNode<'a> {
    pub name: &'a str,
    pub kind: NirNodeKind,
    pub shape: [u32; 4],
    pub shape_len: usize,
    pub lif: Option<NirLifPopulation>,
    pub linear: Option<NirLinear>,
}

/// Import notes (loud lossiness — each counted, none silent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum NirNote {
    /// `v_reset` absent — defaulted to 0 V (reference semantics).
    VResetDefaulted = 0,
    /// Derived capacitance exceeded u16 — clamped (informational; the
    /// substrate's leak uses tau, not C).
    CapacitanceClamped,
    /// `tau` lost sub-μs precision in the →μs rounding.
    TauTruncated,
    /// Some potential lost sub-quantum precision.
    PotentialTruncated,
    /// All-zero weight tensor — scale pinned to 1.
    ZeroWeightTensor,
    /// The quantize→dequant round moved some weight (`max_abs_err` > 0).
    QuantizationLoss,
}

/// Number of note kinds ([`NirReport::notes`] length).
pub const NIR_NOTE_KINDS: usize = 6;

/// The loud import report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NirReport {
    pub inputs: usize,
    pub outputs: usize,
    pub linears: usize,
    pub lifs: usize,
    pub edges: usize,
    pub weight_cells: usize,
    pub notes: [usize; NIR_NOTE_KINDS],
}

impl NirReport {
    /// Total noted events (0 = a fully lossless import).
    #[must_use]
    pub fn note_count(&self) -> usize {
        self.notes.iter().sum()
    }

    fn note(&mut self, n: NirNote) {
        self.notes[n as usize] += 1;
    }
}

/// First pass: buffer sizes + the version string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NirScan<'a> {
    pub version: &'a str,
    pub node_count: usize,
    pub edge_count: usize,
    pub weight_cells: usize,
    /// Total per-neuron LIF records (`sum` of the param array
    /// lengths). Foreign `tau`-like arrays on non-LIF nodes may
    /// inflate this bound on malformed documents (overprovision
    /// only, never under).
    pub lif_neurons: usize,
}

/// Caller-owned buffers for [`nir_import`]. `weights`, `lifs` and
/// `scratch` must each hold [`NirScan::weight_cells`],
/// [`NirScan::lif_neurons`], and `weight_cells + 5 × lif_neurons`
/// cells respectively: import stages each Linear's source f64
/// weights and each LIF's five param arrays into `scratch`
/// (transiently) while quantizing into the arena and the lifs
/// buffer, which keep exactly the results.
#[derive(Debug)]
pub struct NirBuffers<'buf, 'a> {
    /// `bufs` borrow lifetime (`'buf`) and json-data lifetime (`'a`).
    pub nodes: &'buf mut [NirNode<'a>],
    pub edges: &'buf mut [(u32, u32)],
    pub weights: &'buf mut [i16],
    /// Quantized LIF records (one per neuron; population views
    /// index into this).
    pub lifs: &'buf mut [NirLif],
    /// Transient f64 staging (import only; one slot per staged
    /// weight cell or LIF param).
    pub scratch: &'buf mut [f64],
}

/// Import options: the two things NIR does not carry that the
/// substrate needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NirImportOptions {
    /// Simulation timestep (μs). Hard requirement: `tau ≥ dt`.
    pub dt_us: u32,
    /// Voltage grid the potentials quantize onto.
    pub resolution: VoltageResolution,
}

impl NirImportOptions {
    /// Options from `(dt, grid)`.
    #[must_use]
    pub const fn new(dt_us: u32, resolution: VoltageResolution) -> Self {
        Self { dt_us, resolution }
    }
}

impl Default for NirImportOptions {
    fn default() -> Self {
        Self::new(1_000, VoltageResolution::Millivolt)
    }
}

// ---------------------------------------------------------------------------
// Quantization (pure, unit-tested below)
// ---------------------------------------------------------------------------

/// `f64::round` (half away from zero) without std — `round`/`floor`/
/// `ceil` are std-only. Truncate-then-compare: `as i64` truncates
/// toward zero (saturating at the extremes), then the exact fraction
/// decides. (The classic add-±0.5-then-truncate idiom MISROUNDS
/// values up to 1 ulp below a half boundary — `0.49999999999999994`
/// would yield 1.0 — and that value is reachable via `r` in MΩ.)
/// Every caller pre-validates finiteness and bounds; saturation at
/// ±`i64::MAX` lands far outside every caller's accepted range and
/// fails their checks loudly.
#[allow(clippy::cast_precision_loss)]
fn round_half_away(x: f64) -> f64 {
    let t = x as i64 as f64;
    let frac = x - t;
    if frac >= 0.5 {
        t + 1.0
    } else if frac <= -0.5 {
        t - 1.0
    } else {
        t
    }
}

/// Quantize one potential (V) onto the grid. Loud on out-of-range.
fn quant_potential(
    v_v: f64,
    field: &'static str,
    scale: i32,
) -> Result<(i16, f64), NirError<'static>> {
    if !v_v.is_finite() {
        return Err(NirError::BadNumber(field));
    }
    let q_f = v_v * 1000.0 * f64::from(scale);
    let q = round_half_away(q_f);
    let lo = f64::from(MEMBRANE_MV_MIN) * f64::from(scale);
    let hi = f64::from(MEMBRANE_MV_MAX) * f64::from(scale);
    if q < lo || q > hi {
        return Err(NirError::PotentialOutOfRange(field));
    }
    let q = q as i16;
    let err = (q_f - f64::from(q)).abs() / (1000.0 * f64::from(scale));
    Ok((q, err))
}

/// Quantize a LIF parameter set (single neuron) onto the substrate
/// grids (μs / MΩ / voltage quanta) — the structured-entry seam's LIF
/// half. Every hard failure of the contract fires here.
///
/// # Errors
///
/// [`NirError::BadNumber`] on non-finite values, `tau ≤ 0`, `r ≤ 0`
/// or out-of-range magnitudes; [`NirError::TauBelowDt`];
/// [`NirError::ThresholdZero`]; [`NirError::PotentialOutOfRange`].
pub fn quantize_lif(
    tau_s: f64,
    r_ohm: f64,
    v_leak_v: f64,
    v_threshold_v: f64,
    v_reset_v: f64,
    v_reset_defaulted: bool,
    opts: NirImportOptions,
) -> Result<NirLif, NirError<'static>> {
    if [tau_s, r_ohm, v_leak_v, v_threshold_v, v_reset_v]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err(NirError::BadNumber("LIF param"));
    }
    if tau_s <= 0.0 {
        return Err(NirError::BadNumber("tau"));
    }
    if r_ohm <= 0.0 {
        return Err(NirError::BadNumber("r"));
    }
    let tau_us_f = tau_s * 1.0e6;
    let tau_us_round = round_half_away(tau_us_f);
    if !(1.0..=f64::from(u32::MAX)).contains(&tau_us_round) {
        return Err(NirError::BadNumber("tau"));
    }
    let tau_us = tau_us_round as u32;
    if f64::from(tau_us) < f64::from(opts.dt_us) {
        return Err(NirError::TauBelowDt);
    }
    let tau_err_s = (tau_us_f - f64::from(tau_us)).abs() * 1.0e-6;

    let r_millions = r_ohm / 1.0e6;
    let r_mohm = round_half_away(r_millions);
    if !(1.0..=f64::from(u16::MAX)).contains(&r_mohm) {
        return Err(NirError::BadNumber("r"));
    }

    let s = opts.resolution.scale();
    let (leak_q, e1) = quant_potential(v_leak_v, "v_leak", s)?;
    let (threshold_q, e2) = quant_potential(v_threshold_v, "v_threshold", s)?;
    if threshold_q == 0 {
        return Err(NirError::ThresholdZero);
    }
    let (reset_q, e3) = quant_potential(v_reset_v, "v_reset", s)?;

    // C[F] = tau/r → pF = tau_s/r · 1e12; clamp is informational
    let c_pf_f = tau_s / r_ohm * 1.0e12;
    let capacitance_clamped = c_pf_f > f64::from(u16::MAX);
    let capacitance_pf = round_half_away(c_pf_f.clamp(0.0, f64::from(u16::MAX))) as u16;

    Ok(NirLif {
        tau_s,
        r_ohm,
        v_leak_v,
        v_threshold_v,
        v_reset_v,
        v_reset_defaulted,
        tau_us,
        resistance_mohm: r_mohm as u16,
        capacitance_pf,
        capacitance_clamped,
        leak_q,
        threshold_q,
        reset_q,
        tau_err_s,
        max_v_err_v: e1.max(e2).max(e3),
    })
}

/// Quantize a materialized Linear weight tensor into the arena at
/// `offset` — the structured-entry seam's Linear half. `values` is
/// row-major `weight[out][in]` (`y = W·x`, rows = outputs), the same
/// contract the JSON importer applies; callers that hold f64 weights
/// (HDF5 import, builders) enter here without a JSON document.
///
/// # Errors
///
/// [`NirError::BadShape("weight")`] unless `values.len()` is exactly
/// `rows·cols` with both nonzero; [`NirError::BufferOverflow`] when
/// the arena cannot hold `offset + rows·cols`; [`NirError::BadNumber`]
/// on any non-finite value or a `scale` that underflows to 0 (denormal
/// `absmax` — the record would lie, per the R3 review finding).
pub fn quantize_linear(
    values: &[f64],
    rows: usize,
    cols: usize,
    arena: &mut [i16],
    offset: usize,
) -> Result<NirLinear, NirError<'static>> {
    if rows == 0 || cols == 0 {
        return Err(NirError::BadShape("weight"));
    }
    let n = rows.checked_mul(cols).ok_or(NirError::BadShape("weight"))?;
    if values.len() != n {
        return Err(NirError::BadShape("weight"));
    }
    let end = offset.checked_add(n).ok_or(NirError::BufferOverflow)?;
    if end > arena.len() {
        return Err(NirError::BufferOverflow);
    }
    let mut absmax = 0.0f64;
    for &v in values {
        if !v.is_finite() {
            return Err(NirError::BadNumber("weight"));
        }
        absmax = absmax.max(v.abs());
    }
    let (scale, zero_tensor) = if absmax == 0.0 {
        (1.0, true)
    } else {
        (absmax / I16_FS, false)
    };
    // Denormal `absmax`: `absmax/32767` underflows to exactly 0.0 —
    // finite, so every prior check passes, but then q·scale = 0 ≠
    // absmax and the quant record would lie (export would silently
    // zero the tensor and break idempotence). Loud, per the contract.
    // (NaN unreachable: absmax is finite and I16_FS is a nonzero
    // constant, so `== 0.0` is exact, not partial-order fuzz.)
    if scale == 0.0 {
        return Err(NirError::BadNumber("weight"));
    }
    let mut max_abs_err = 0.0f64;
    for (k, &v) in values.iter().enumerate() {
        let q = round_half_away(v / scale).clamp(-I16_FS, I16_FS) as i16;
        arena[offset + k] = q;
        max_abs_err = max_abs_err.max((v - f64::from(q) * scale).abs());
    }
    Ok(NirLinear {
        rows,
        cols,
        weight_offset: offset,
        scale,
        absmax,
        max_abs_err,
        zero_tensor,
    })
}

// ---------------------------------------------------------------------------
// Pass 1: scan (counts + version; full shape walk)
// ---------------------------------------------------------------------------

/// Scan a NIR JSON document: counts + version. Walks the full schema
/// shape (so malformed structure fails here); param values are NOT
/// validated — that is import's job.
pub fn nir_scan(json: &[u8]) -> Result<NirScan<'_>, NirError<'_>> {
    let mut r = Reader::new(json);
    let mut version: Option<&str> = None;
    let mut node_count = 0usize;
    let mut edge_count = 0usize;
    let mut weight_cells = 0usize;
    let mut lif_neurons = 0usize;
    let mut saw_node = false;

    let mut first = true;
    while let Some(key) = r.object_step(&mut first)? {
        match key {
            "version" => version = Some(r.read_string()?),
            "node" => {
                saw_node = true;
                scan_graph(
                    &mut r,
                    &mut node_count,
                    &mut edge_count,
                    &mut weight_cells,
                    &mut lif_neurons,
                )?;
            }
            _ => r.skip_value(0)?,
        }
    }
    // strict subset: nothing but whitespace may follow the root object
    // (Python's `json.loads` rejects trailing content; so do we)
    if r.peek().is_some() {
        return Err(NirError::Json(r.pos));
    }
    let version = version.ok_or(NirError::MissingField("version"))?;
    if !saw_node {
        return Err(NirError::MissingField("node"));
    }
    Ok(NirScan {
        version,
        node_count,
        edge_count,
        weight_cells,
        lif_neurons,
    })
}

fn scan_graph(
    r: &mut Reader<'_>,
    node_count: &mut usize,
    edge_count: &mut usize,
    weight_cells: &mut usize,
    lif_neurons: &mut usize,
) -> Result<(), NirError<'static>> {
    let mut first = true;
    while let Some(key) = r.object_step(&mut first)? {
        match key {
            "type" => {
                if r.read_string()? != "NIRGraph" {
                    return Err(NirError::BadShape("node.type"));
                }
            }
            "edges" => {
                let mut efirst = true;
                while r.array_step(&mut efirst)? {
                    let mut pfirst = true;
                    let mut n = 0;
                    while r.array_step(&mut pfirst)? {
                        r.read_string()?;
                        n += 1;
                    }
                    if n != 2 {
                        return Err(NirError::BadShape("edges"));
                    }
                    *edge_count += 1;
                }
            }
            "nodes" => {
                let mut nfirst = true;
                while r.object_step(&mut nfirst)?.is_some() {
                    scan_node(r, weight_cells, lif_neurons)?;
                    *node_count += 1;
                }
            }
            _ => r.skip_value(0)?,
        }
    }
    Ok(())
}

fn scan_node(
    r: &mut Reader<'_>,
    weight_cells: &mut usize,
    lif_neurons: &mut usize,
) -> Result<(), NirError<'static>> {
    // per-node LIF population bookkeeping: the five param arrays
    // must all be present-at-equal-length (v_reset may be absent)
    let mut pop: Option<usize> = None;
    let mut first = true;
    while let Some(key) = r.object_step(&mut first)? {
        match key {
            "type" => {
                r.read_string()?; // validated at import
            }
            "weight" => {
                let mut depth = 0usize;
                count_array(r, &mut depth, weight_cells)?;
            }
            "tau" | "r" | "v_leak" | "v_threshold" | "v_reset" => {
                let n = count_param_array(r)?;
                match pop {
                    None => pop = Some(n),
                    Some(p) if p != n => return Err(NirError::BadShape("LIF param")),
                    Some(_) => {}
                }
            }
            _ => r.skip_value(0)?,
        }
    }
    *lif_neurons += pop.unwrap_or(0);
    Ok(())
}

/// Walk one LIF param array, counting elements (each must be a
/// number — nesting is a wrong-shape rejection here, matching
/// [`count_array`]'s strictness).
fn count_param_array(r: &mut Reader<'_>) -> Result<usize, NirError<'static>> {
    let mut n = 0usize;
    let mut first = true;
    while r.array_step(&mut first)? {
        if r.peek() == Some(b'[') {
            return Err(NirError::BadShape("LIF param"));
        }
        r.read_number()?;
        n += 1;
    }
    Ok(n)
}

/// Recursively walk a weight array, counting leaves. A weight tensor
/// is EXACTLY 2-D of non-empty rows — 1-D, empty, empty-row, and 3-D
/// shapes all fail here (scan's "malformed structure fails here"
/// contract); ragged rows still fail at import (shape is checked,
/// not counted, there).
fn count_array(
    r: &mut Reader<'_>,
    depth: &mut usize,
    leaves: &mut usize,
) -> Result<(), NirError<'static>> {
    if *depth >= 2 {
        // 2-D max (a weight tensor); deeper = wrong shape
        return Err(NirError::BadShape("weight"));
    }
    let mut first = true;
    let mut elems = 0usize;
    let mut nested = false;
    while r.array_step(&mut first)? {
        if r.peek() == Some(b'[') {
            *depth += 1;
            nested = true;
            count_array(r, depth, leaves)?;
            *depth -= 1;
        } else {
            r.read_number()?;
            *leaves += 1;
        }
        elems += 1;
    }
    if elems == 0 {
        // `[]` outer or an empty row `[[]]`
        return Err(NirError::BadShape("weight"));
    }
    if *depth == 0 && !nested {
        // a flat array at the top = 1-D tensor
        return Err(NirError::BadShape("weight"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pass 2: import (nodes walk, then edges walk)
// ---------------------------------------------------------------------------

/// Import a NIR JSON document into caller buffers. Fills `nodes` in
/// document order, `edges` as resolved node-index pairs, and the
/// quantized weights into the arena. On any error nothing is promised
/// about buffer contents.
// two documented walks (nodes, then edges) + their trailing checks;
// the 100-line cap is crossed by the EOF checks alone (network.rs
// precedent for the allow)
#[allow(clippy::too_many_lines)]
pub fn nir_import<'a>(
    json: &'a [u8],
    opts: NirImportOptions,
    bufs: &mut NirBuffers<'_, 'a>,
) -> Result<NirReport, NirError<'a>> {
    let mut report = NirReport::default();

    // walk 1: nodes (+ weights + LIF populations)
    let mut node_count = 0usize;
    let mut weight_fill = 0usize;
    let mut lif_fill = 0usize;
    {
        let mut r = Reader::new(json);
        let mut first = true;
        while let Some(key) = r.object_step(&mut first)? {
            if key == "node" {
                let mut gfirst = true;
                while let Some(gkey) = r.object_step(&mut gfirst)? {
                    if gkey == "nodes" {
                        let mut nfirst = true;
                        while let Some(name) = r.object_step(&mut nfirst)? {
                            if node_count >= bufs.nodes.len() {
                                return Err(NirError::BufferOverflow);
                            }
                            import_node(
                                &mut r,
                                name,
                                opts,
                                bufs,
                                &mut node_count,
                                &mut weight_fill,
                                &mut lif_fill,
                                &mut report,
                            )?;
                        }
                    } else {
                        r.skip_value(0)?;
                    }
                }
            } else {
                r.skip_value(0)?;
            }
        }
        // trailing content after the root is rejected here too
        if r.peek().is_some() {
            return Err(NirError::Json(r.pos));
        }
    }
    report.weight_cells = weight_fill;

    // duplicate node names (the reference keys nodes by name)
    for i in 0..node_count {
        for j in (i + 1)..node_count {
            if bufs.nodes[i].name == bufs.nodes[j].name {
                return Err(NirError::DuplicateNodeName);
            }
        }
    }

    // walk 2: edges (names now resolvable, key order irrelevant)
    let mut edge_count = 0usize;
    {
        let mut r = Reader::new(json);
        let mut first = true;
        while let Some(key) = r.object_step(&mut first)? {
            if key == "node" {
                let mut gfirst = true;
                while let Some(gkey) = r.object_step(&mut gfirst)? {
                    if gkey == "edges" {
                        let mut efirst = true;
                        while r.array_step(&mut efirst)? {
                            if edge_count >= bufs.edges.len() {
                                return Err(NirError::BufferOverflow);
                            }
                            let mut pair = [0u32; 2];
                            let mut pfirst = true;
                            let mut pi = 0;
                            while r.array_step(&mut pfirst)? {
                                if pi >= 2 {
                                    return Err(NirError::BadShape("edges"));
                                }
                                let name = r.read_string()?;
                                let idx = (0..node_count)
                                    .find(|&i| bufs.nodes[i].name == name)
                                    .ok_or(NirError::UnknownEdgeEndpoint(name))?;
                                pair[pi] = idx as u32;
                                pi += 1;
                            }
                            if pi != 2 {
                                return Err(NirError::BadShape("edges"));
                            }
                            bufs.edges[edge_count] = (pair[0], pair[1]);
                            edge_count += 1;
                            report.edges = edge_count;
                        }
                    } else {
                        r.skip_value(0)?;
                    }
                }
            } else {
                r.skip_value(0)?;
            }
        }
        // and here (walk 2 sees the same document)
        if r.peek().is_some() {
            return Err(NirError::Json(r.pos));
        }
    }

    // duplicate edges (reference validate_structure)
    for i in 0..edge_count {
        for j in (i + 1)..edge_count {
            if bufs.edges[i] == bufs.edges[j] {
                return Err(NirError::DuplicateEdge);
            }
        }
    }
    Ok(report)
}

/// Read one LIF param array, staging its values contiguously into
/// `scratch` at `*fill`; returns the `(start, len)` staged range.
fn read_param_array(
    r: &mut Reader<'_>,
    scratch: &mut [f64],
    fill: &mut usize,
    field: &'static str,
) -> Result<(usize, usize), NirError<'static>> {
    let start = *fill;
    let mut first = true;
    while r.array_step(&mut first)? {
        if r.peek() == Some(b'[') {
            return Err(NirError::BadShape(field));
        }
        let v = r.read_number()?;
        if *fill >= scratch.len() {
            return Err(NirError::BufferOverflow);
        }
        scratch[*fill] = v;
        *fill += 1;
    }
    Ok((start, *fill - start))
}

/// LIF tail of [`import_node`]: validate the five staged param
/// ranges (equal lengths, ≥ 1; `v_reset` absent = reference zeros),
/// quantize the population per neuron via [`quantize_lif`] into
/// `bufs.lifs`, note lossiness, and return the population view.
#[allow(clippy::too_many_arguments)]
fn finish_lif_population(
    tau: Option<(usize, usize)>,
    res: Option<(usize, usize)>,
    v_leak: Option<(usize, usize)>,
    v_threshold: Option<(usize, usize)>,
    v_reset: Option<(usize, usize)>,
    bufs: &mut NirBuffers<'_, '_>,
    opts: NirImportOptions,
    report: &mut NirReport,
    lif_fill: &mut usize,
) -> Result<NirLifPopulation, NirError<'static>> {
    let rng = |o: Option<(usize, usize)>, f: &'static str| o.ok_or(NirError::MissingField(f));
    let tau_r = rng(tau, "tau")?;
    let res_r = rng(res, "r")?;
    let leak_r = rng(v_leak, "v_leak")?;
    let thr_r = rng(v_threshold, "v_threshold")?;
    let n = tau_r.1;
    if n == 0 {
        return Err(NirError::BadShape("tau"));
    }
    for (r_, f) in [(res_r, "r"), (leak_r, "v_leak"), (thr_r, "v_threshold")] {
        if r_.1 != n {
            return Err(NirError::BadShape(f));
        }
    }
    if let Some(vr) = v_reset {
        if vr.1 != n {
            return Err(NirError::BadShape("v_reset"));
        }
    } else {
        report.note(NirNote::VResetDefaulted);
    }
    let start = *lif_fill;
    if start + n > bufs.lifs.len() {
        return Err(NirError::BufferOverflow);
    }
    for i in 0..n {
        let v_reset_val = match v_reset {
            Some((s, _)) => bufs.scratch[s + i],
            None => 0.0,
        };
        let lif = quantize_lif(
            bufs.scratch[tau_r.0 + i],
            bufs.scratch[res_r.0 + i],
            bufs.scratch[leak_r.0 + i],
            bufs.scratch[thr_r.0 + i],
            v_reset_val,
            v_reset.is_none(),
            opts,
        )?;
        if lif.tau_err_s > 0.0 {
            report.note(NirNote::TauTruncated);
        }
        if lif.max_v_err_v > 0.0 {
            report.note(NirNote::PotentialTruncated);
        }
        if lif.capacitance_clamped {
            report.note(NirNote::CapacitanceClamped);
        }
        bufs.lifs[start + i] = lif;
    }
    *lif_fill += n;
    Ok(NirLifPopulation {
        offset: start,
        len: n,
    })
}

#[allow(clippy::too_many_arguments)]
fn import_node<'a>(
    r: &mut Reader<'a>,
    name: &'a str,
    opts: NirImportOptions,
    bufs: &mut NirBuffers<'_, 'a>,
    node_count: &mut usize,
    weight_fill: &mut usize,
    lif_fill: &mut usize,
    report: &mut NirReport,
) -> Result<(), NirError<'a>> {
    let idx = *node_count;
    bufs.nodes[idx] = NirNode {
        name,
        kind: NirNodeKind::Input,
        shape: [0; 4],
        shape_len: 0,
        lif: None,
        linear: None,
    };

    let mut kind: Option<&str> = None;
    let mut shape = [0u32; 4];
    let mut shape_len = 0usize;
    // LIF params stage contiguously into scratch (per-node `fill`,
    // freed after quantization); ranges are (start, len) pairs.
    let mut staged = 0usize;
    let mut tau: Option<(usize, usize)> = None;
    let mut res: Option<(usize, usize)> = None;
    let mut v_leak: Option<(usize, usize)> = None;
    let mut v_threshold: Option<(usize, usize)> = None;
    let mut v_reset: Option<(usize, usize)> = None;

    let mut first = true;
    while let Some(key) = r.object_step(&mut first)? {
        match key {
            "type" => kind = Some(r.read_string()?),
            "shape" => {
                let mut sfirst = true;
                while r.array_step(&mut sfirst)? {
                    if shape_len >= 4 {
                        return Err(NirError::BadShape("shape"));
                    }
                    let d = r.read_number()?;
                    if !(0.0..=f64::from(u32::MAX)).contains(&d) {
                        return Err(NirError::BadShape("shape"));
                    }
                    shape[shape_len] = d as u32;
                    shape_len += 1;
                }
            }
            "tau" => tau = Some(read_param_array(r, bufs.scratch, &mut staged, "tau")?),
            "r" => res = Some(read_param_array(r, bufs.scratch, &mut staged, "r")?),
            "v_leak" => v_leak = Some(read_param_array(r, bufs.scratch, &mut staged, "v_leak")?),
            "v_threshold" => {
                v_threshold = Some(read_param_array(r, bufs.scratch, &mut staged, "v_threshold")?);
            }
            "v_reset" => v_reset = Some(read_param_array(r, bufs.scratch, &mut staged, "v_reset")?),
            "weight" => {
                let lin = import_weight(r, bufs, *weight_fill)?;
                if lin.max_abs_err > 0.0 {
                    report.note(NirNote::QuantizationLoss);
                }
                if lin.zero_tensor {
                    report.note(NirNote::ZeroWeightTensor);
                }
                bufs.nodes[idx].linear = Some(lin);
                *weight_fill += lin.rows * lin.cols;
            }
            _ => r.skip_value(0)?, // metadata + unknown fields tolerated
        }
    }

    let kind = kind.ok_or(NirError::MissingField("type"))?;
    bufs.nodes[idx].shape = shape;
    bufs.nodes[idx].shape_len = shape_len;
    bufs.nodes[idx].kind = match kind {
        "Input" => {
            if shape_len == 0 {
                return Err(NirError::MissingField("shape"));
            }
            report.inputs += 1;
            NirNodeKind::Input
        }
        "Output" => {
            if shape_len == 0 {
                return Err(NirError::MissingField("shape"));
            }
            report.outputs += 1;
            NirNodeKind::Output
        }
        "Linear" => {
            if bufs.nodes[idx].linear.is_none() {
                return Err(NirError::MissingField("weight"));
            }
            report.linears += 1;
            NirNodeKind::Linear
        }
        "LIF" => {
            report.lifs += 1;
            bufs.nodes[idx].lif = Some(finish_lif_population(
                tau, res, v_leak, v_threshold, v_reset, bufs, opts, report, lif_fill,
            )?);
            NirNodeKind::Lif
        }
        other => return Err(NirError::UnsupportedNodeKind(other)),
    };
    *node_count += 1;
    Ok(())
}

/// Read a 2-D weight array and quantize it into the arena at
/// `offset`: parse + stage the source f64s into `bufs.scratch`
/// (row-major `weight[out][in]`: `y_o = Σ_i w[o][i] · x_i`), then
/// delegate to [`quantize_linear`] — the shared quantization
/// contract.
fn import_weight(
    r: &mut Reader<'_>,
    bufs: &mut NirBuffers<'_, '_>,
    offset: usize,
) -> Result<NirLinear, NirError<'static>> {
    let mut rows = 0usize;
    let mut cols: Option<usize> = None;
    let mut staged = 0usize; // f64 slots written to scratch

    let mut rfirst = true;
    while r.array_step(&mut rfirst)? {
        if r.peek() != Some(b'[') {
            return Err(NirError::BadShape("weight"));
        }
        // one pass per row: read, stage, count — raggedness compares
        // after the row (buffer contents are unspecified on error)
        let mut cfirst = true;
        let mut rc = 0usize;
        while r.array_step(&mut cfirst)? {
            if r.peek() == Some(b'[') {
                return Err(NirError::BadShape("weight")); // 3-D
            }
            let v = r.read_number()?;
            if staged >= bufs.scratch.len() {
                return Err(NirError::BufferOverflow);
            }
            bufs.scratch[staged] = v;
            staged += 1;
            rc += 1;
        }
        if rc == 0 {
            return Err(NirError::BadShape("weight")); // empty row
        }
        match cols {
            None => cols = Some(rc),
            Some(c) if c != rc => return Err(NirError::BadShape("weight")), // ragged
            Some(_) => {}
        }
        rows += 1;
    }
    let cols = cols.ok_or(NirError::BadShape("weight"))?; // `[]` outer
    quantize_linear(&bufs.scratch[..staged], rows, cols, bufs.weights, offset)
}

// ---------------------------------------------------------------------------
// Export — canonical bytes, derived values rendered, provenance in metadata
// ---------------------------------------------------------------------------

struct ByteWriter<'a> {
    out: &'a mut [u8],
    len: usize,
}

impl ByteWriter<'_> {
    fn push(&mut self, b: u8) -> Result<(), NirError<'static>> {
        if self.len >= self.out.len() {
            return Err(NirError::ExportTooSmall);
        }
        self.out[self.len] = b;
        self.len += 1;
        Ok(())
    }
    fn push_str(&mut self, s: &str) -> Result<(), NirError<'static>> {
        for &b in s.as_bytes() {
            self.push(b)?;
        }
        Ok(())
    }
}

impl core::fmt::Write for ByteWriter<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.push_str(s).map_err(|_| core::fmt::Error)
    }
}

/// Render one f64 canonically: Rust's `Display` (shortest round-trip
/// — deterministic across platforms and releases) — parses back
/// bit-exactly with the same correctly-rounded reader.
fn write_f64(w: &mut ByteWriter<'_>, v: f64) -> Result<(), NirError<'static>> {
    if !v.is_finite() {
        return Err(NirError::BadNumber("export value"));
    }
    write!(w, "{v}").map_err(|_| NirError::ExportTooSmall)
}

fn write_u32(w: &mut ByteWriter<'_>, v: u32) -> Result<(), NirError<'static>> {
    write!(w, "{v}").map_err(|_| NirError::ExportTooSmall)
}

/// The JSON container's string subset: printable ASCII, no escape
/// char, no quote — exactly what [`Reader::read_string`] reads
/// back. The structured-entry builder accepts any `&str`; this
/// rule gates only the JSON boundary (export).
fn name_is_writable(s: &str) -> bool {
    s.bytes()
        .all(|c| c != b'"' && c != b'\\' && (0x20..=0x7e).contains(&c))
}

/// Export an imported graph: the SAME dict schema, canonical bytes —
/// fixed key order, derived (substrate-exact) values in the schema
/// fields, provenance + quant records in `metadata.neuralos`.
/// Returns the byte length written. Importing the export must
/// reproduce the import bit-for-bit (the idempotence gate).
///
/// # Errors
///
/// [`NirError::ExportTooSmall`] when `out` cannot hold the document;
/// [`NirError::NonAsciiNodeName`] when a node name cannot be written
/// escape-free (the builder accepts any `&str`; the container
/// cannot).
pub fn nir_export<'a>(
    nodes: &[NirNode<'a>],
    edges: &[(u32, u32)],
    weights: &[i16],
    lifs: &[NirLif],
    opts: NirImportOptions,
    out: &mut [u8],
) -> Result<usize, NirError<'a>> {
    for node in nodes {
        if !name_is_writable(node.name) {
            return Err(NirError::NonAsciiNodeName(node.name));
        }
    }
    let mut w = ByteWriter { out, len: 0 };
    w.push_str("{\"version\":\"")?;
    w.push_str(EXPORT_VERSION)?;
    w.push_str("\",\"node\":{\"type\":\"NIRGraph\",\"edges\":[")?;
    for (i, (a, b)) in edges.iter().enumerate() {
        if i > 0 {
            w.push(b',')?;
        }
        // edge indices must name nodes — "?" placeholders would be a
        // silent lie in an export document
        let an = nodes
            .get(*a as usize)
            .ok_or(NirError::BadShape("edges"))?
            .name;
        let bn = nodes
            .get(*b as usize)
            .ok_or(NirError::BadShape("edges"))?
            .name;
        w.push_str("[\"")?;
        w.push_str(an)?;
        w.push_str("\",\"")?;
        w.push_str(bn)?;
        w.push_str("\"]")?;
    }
    w.push_str("],\"nodes\":{")?;
    for (i, node) in nodes.iter().enumerate() {
        if i > 0 {
            w.push(b',')?;
        }
        w.push_str("\"")?;
        w.push_str(node.name)?;
        w.push_str("\":")?;
        export_node(&mut w, node, weights, lifs, opts)?;
    }
    w.push_str("}}}")?;
    Ok(w.len)
}

// the per-neuron array rendering (schema fields + metadata) crosses
// the 100-line cap with the population loops alone (nir_import has
// the precedent allow for its EOF checks)
#[allow(clippy::too_many_lines)]
fn export_node(
    w: &mut ByteWriter<'_>,
    node: &NirNode<'_>,
    weights: &[i16],
    lifs: &[NirLif],
    opts: NirImportOptions,
) -> Result<(), NirError<'static>> {
    let scale = opts.resolution.scale();
    let to_v = |q: i16| f64::from(q) / (1000.0 * f64::from(scale));
    match node.kind {
        NirNodeKind::Input | NirNodeKind::Output => {
            let t = if node.kind == NirNodeKind::Input {
                "Input"
            } else {
                "Output"
            };
            w.push_str("{\"type\":\"")?;
            w.push_str(t)?;
            w.push_str("\",\"shape\":[")?;
            for d in 0..node.shape_len {
                if d > 0 {
                    w.push(b',')?;
                }
                write_u32(w, node.shape[d])?;
            }
            w.push_str("]}")?;
        }
        NirNodeKind::Lif => {
            let pop = node.lif.ok_or(NirError::MissingField("lif"))?;
            let rec = |i: usize| lifs.get(pop.offset + i).ok_or(NirError::MissingField("lif"));
            // derived, rendered back in source units EXACTLY as the
            // substrate holds them (tau_us/1e6, r_mohm*1e6, q/1000):
            // re-import reproduces the identical quantized
            // population. Provenance + quant records ride metadata
            // as per-neuron arrays.
            w.push_str("{\"type\":\"LIF\",\"tau\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, f64::from(rec(i)?.tau_us) / 1.0e6)?;
            }
            w.push_str("],\"r\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, f64::from(rec(i)?.resistance_mohm) * 1.0e6)?;
            }
            w.push_str("],\"v_leak\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, to_v(rec(i)?.leak_q))?;
            }
            w.push_str("],\"v_threshold\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, to_v(rec(i)?.threshold_q))?;
            }
            w.push_str("],\"v_reset\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, to_v(rec(i)?.reset_q))?;
            }
            w.push_str("],\"metadata\":{\"neuralos\":{\"provenance\":{\
\"tau_s\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, rec(i)?.tau_s)?;
            }
            w.push_str("],\"r_ohm\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, rec(i)?.r_ohm)?;
            }
            w.push_str("],\"v_leak_v\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, rec(i)?.v_leak_v)?;
            }
            w.push_str("],\"v_threshold_v\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, rec(i)?.v_threshold_v)?;
            }
            w.push_str("],\"v_reset_v\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, rec(i)?.v_reset_v)?;
            }
            w.push_str("],\"v_reset_defaulted\":")?;
            w.push_str(if rec(0)?.v_reset_defaulted { "true" } else { "false" })?;
            w.push_str("},\"quant\":{\"grid\":\"")?;
            w.push_str(match opts.resolution {
                VoltageResolution::Millivolt => "mV",
                VoltageResolution::CentiMillivolt => "cV",
            })?;
            w.push_str("\",\"dt_us\":")?;
            write_u32(w, opts.dt_us)?;
            w.push_str(",\"tau_err_s\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, rec(i)?.tau_err_s)?;
            }
            w.push_str("],\"max_v_err_v\":[")?;
            for i in 0..pop.len {
                if i > 0 {
                    w.push(b',')?;
                }
                write_f64(w, rec(i)?.max_v_err_v)?;
            }
            w.push_str("]}}}}")?; // quant, neuralos, metadata, node
        }
        NirNodeKind::Linear => {
            let lin = node.linear.ok_or(NirError::MissingField("linear"))?;
            // dequantized weights: w' = q·scale — plain NIR a real
            // snnTorch/NIR stack can load; provenance rides metadata.
            w.push_str("{\"type\":\"Linear\",\"weight\":[")?;
            for row in 0..lin.rows {
                if row > 0 {
                    w.push(b',')?;
                }
                w.push(b'[')?;
                for col in 0..lin.cols {
                    if col > 0 {
                        w.push(b',')?;
                    }
                    let q = weights[lin.weight_offset + row * lin.cols + col];
                    write_f64(w, f64::from(q) * lin.scale)?;
                }
                w.push(b']')?;
            }
            w.push_str("],\"metadata\":{\"neuralos\":{\"provenance\":{\
\"absmax\":")?;
            write_f64(w, lin.absmax)?;
            w.push_str("},\"quant\":{\"scale\":")?;
            write_f64(w, lin.scale)?;
            w.push_str(",\"max_abs_err\":")?;
            write_f64(w, lin.max_abs_err)?;
            w.push_str(",\"zero_tensor\":")?;
            w.push_str(if lin.zero_tensor { "true" } else { "false" })?;
            w.push_str(",\"source\":\"")?;
            w.push_str(NIR_REF_SHA)?;
            w.push_str("\"}}}}")?; // quant, neuralos, metadata, node
        }
    }
    Ok(())
}

#[cfg(feature = "std")]
mod std_assembly {
    //! The slice-1 network assembly: the canonical chain
    //! `Input → Linear → LIF → Output` onto a real
    //! [`crate::network::SpikingNeuralNetwork`]. Linear is the input
    //! encoder (`y = W·x` → per-step μA currents, the substrate's only
    //! external-input channel); the LIF node's quantized params are
    //! the neurons. Any other graph shape is a loud
    //! [`NirError::UnsupportedTopology`] naming slice 2.

    use super::{
        NirError, NirImportOptions, NirLif, NirLifParams, NirLinear, NirLifPopulation, NirNode,
        NirNodeKind, NIR_REF_SHA,
    };
    use crate::lif_neuron::{LIFNeuron, NeuronType};
    use crate::network::SpikingNeuralNetwork;

    /// An imported graph in owned buffers (std convenience over the
    /// two-pass buffer API).
    #[derive(Debug)]
    pub struct NirImport<'a> {
        /// Nodes in document order.
        pub nodes: Vec<NirNode<'a>>,
        /// Resolved node-index pairs.
        pub edges: Vec<(u32, u32)>,
        /// Quantized weights (arena layout per [`NirLinear`]).
        pub weights: Vec<i16>,
        /// Quantized LIF records, one per neuron (population views
        /// index into this).
        pub lifs: Vec<NirLif>,
        /// The options the graph was imported under.
        pub opts: NirImportOptions,
        /// Reference provenance.
        pub ref_sha: &'static str,
    }

    impl NirImport<'_> {
        /// Scan + import a JSON document into owned buffers.
        ///
        /// # Errors
        ///
        /// Every [`NirError`] the buffer import can raise.
        pub fn from_json(json: &[u8], opts: NirImportOptions) -> Result<NirImport<'_>, NirError<'_>> {
            let scan = super::nir_scan(json)?;
            let mut nodes: Vec<NirNode<'_>> = vec![
                NirNode {
                    name: "",
                    kind: NirNodeKind::Input,
                    shape: [0; 4],
                    shape_len: 0,
                    lif: None,
                    linear: None,
                };
                scan.node_count
            ];
            let mut edges = vec![(0u32, 0u32); scan.edge_count];
            // scratch contract: arena + f64 scratch each hold the
            // exact weight-cell count (the arena keeps the result)
            let n = scan.weight_cells;
            let mut weights = vec![0i16; n];
            let mut lifs = vec![NirLif::default(); scan.lif_neurons];
            let mut scratch = vec![0f64; scan.weight_cells + 5 * scan.lif_neurons];
            {
                let mut bufs = super::NirBuffers {
                    nodes: &mut nodes,
                    edges: &mut edges,
                    weights: &mut weights,
                    lifs: &mut lifs,
                    scratch: &mut scratch,
                };
                super::nir_import(json, opts, &mut bufs)?;
            }
            Ok(NirImport {
                nodes,
                edges,
                weights,
                lifs,
                opts,
                ref_sha: NIR_REF_SHA,
            })
        }

        /// Assemble the canonical chain into a substrate network:
        /// returns the net (LIF params honored) + the Linear encoder
        /// (rows = neurons, cols = input features).
        ///
        /// # Errors
        ///
        /// [`NirError::UnsupportedTopology`] for anything but
        /// `Input → Linear → LIF → Output`; [`NirError::MissingField`]
        /// if the fixture starved a node.
        #[allow(clippy::missing_panics_doc)] // no panics — flagged for the unwrap on .first()
        pub fn build_chain_network(
            &self,
        ) -> Result<(SpikingNeuralNetwork, ChainEncoder<'_>), NirError<'_>> {
            let mut input_n = None;
            let mut linear_n = None;
            let mut lif_n = None;
            let mut output_n = None;
            for (i, n) in self.nodes.iter().enumerate() {
                match n.kind {
                    NirNodeKind::Input => input_n = input_n.or(Some(i)),
                    NirNodeKind::Linear => linear_n = linear_n.or(Some(i)),
                    NirNodeKind::Lif => lif_n = lif_n.or(Some(i)),
                    NirNodeKind::Output => output_n = output_n.or(Some(i)),
                }
            }
            let inp = input_n.ok_or(NirError::UnsupportedTopology("no Input node"))?;
            let lin = linear_n.ok_or(NirError::UnsupportedTopology("no Linear node"))?;
            let lif = lif_n.ok_or(NirError::UnsupportedTopology("no LIF node"))?;
            let out = output_n.ok_or(NirError::UnsupportedTopology("no Output node"))?;
            if self.nodes.len() != 4 {
                return Err(NirError::UnsupportedTopology(
                    "exactly 4 nodes (Input-Linear-LIF-Output) in slice 1",
                ));
            }
            let chain = [
                (inp as u32, lin as u32),
                (lin as u32, lif as u32),
                (lif as u32, out as u32),
            ];
            if self.edges.as_slice() != chain {
                return Err(NirError::UnsupportedTopology(
                    "edges must be Input->Linear->LIF->Output",
                ));
            }
            let pop = self.nodes[lif].lif.ok_or(NirError::MissingField("lif"))?;
            let linear_rec = self.nodes[lin].linear.ok_or(NirError::MissingField("linear"))?;
            if linear_rec.rows > u16::MAX as usize {
                return Err(NirError::BufferOverflow);
            }
            if self.nodes[inp].shape.first().copied().unwrap_or(0) != linear_rec.cols as u32 {
                return Err(NirError::UnsupportedTopology("Input shape != Linear cols"));
            }
            // reference type-check parity: the LIF population must
            // match the feeding Linear's rows
            if pop.len != linear_rec.rows {
                return Err(NirError::UnsupportedTopology(
                    "LIF population != Linear rows",
                ));
            }

            // one neuron per Linear row, its own quantized params
            let mut neurons = Vec::with_capacity(linear_rec.rows);
            for id in 0..linear_rec.rows {
                let p = self
                    .lifs
                    .get(pop.offset + id)
                    .ok_or(NirError::MissingField("lif"))?;
                let mut n = LIFNeuron::new_with_type_resolution(
                    id as u16,
                    NeuronType::Excitatory,
                    self.opts.resolution,
                );
                n.resting_potential = p.leak_q;
                n.membrane_potential = p.leak_q;
                n.threshold = p.threshold_q;
                n.reset_potential = p.reset_q;
                n.tau_membrane_us = p.tau_us;
                n.tau_refractory_us = 1_000; // NIR LIF has no refractory → minimum
                n.resistance_mohm = p.resistance_mohm;
                n.capacitance_pf = p.capacitance_pf;
                n.noise_amplitude_ua = 0; // import is deterministic
                neurons.push(n);
            }
            let net =
                SpikingNeuralNetwork::from_neurons(neurons, self.opts.dt_us)
                    .map_err(|_| NirError::BufferOverflow)?;
            let encoder = ChainEncoder {
                lin: linear_rec,
                weights: &self.weights,
            };
            Ok((net, encoder))
        }
    }

    /// The Linear half of the chain: feature currents (μA) →
    /// per-neuron currents (μA), saturating i16 — `y = W·x` in
    /// substrate units. Weights are scaled by 1/100 (the substrate's
    /// synapse convention: a weight contributes `w/100` μA per unit
    /// input feature current), documented as the encoder's gain.
    #[derive(Debug)]
    pub struct ChainEncoder<'a> {
        lin: NirLinear,
        weights: &'a [i16],
    }

    impl ChainEncoder<'_> {
        /// Rows (neurons) of the encoder.
        #[must_use]
        pub fn rows(&self) -> usize {
            self.lin.rows
        }

        /// Columns (input features).
        #[must_use]
        pub fn cols(&self) -> usize {
            self.lin.cols
        }

        /// Encode feature currents into per-neuron currents.
        /// i64 accumulator: i32 would overflow (and in release,
        /// silently wrap) at cols ≥ 3 with |w| = |x| = 32767.
        #[must_use]
        pub fn encode(&self, x: &[i16]) -> Vec<i16> {
            let mut out = vec![0i16; self.lin.rows];
            for (r, o) in out.iter_mut().enumerate() {
                let mut acc: i64 = 0;
                for c in 0..self.lin.cols {
                    let w = i64::from(
                        self.weights[self.lin.weight_offset + r * self.lin.cols + c],
                    );
                    acc += w * i64::from(x.get(c).copied().unwrap_or(0));
                }
                acc /= 100;
                *o = acc.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;
            }
            out
        }
    }

    /// Structured-entry graph builder: construct a NIR graph in
    /// memory — no JSON document. `quantize_lif` / `quantize_linear`
    /// are the quantization seam; this is the graph seam over them.
    /// Node names may be any `&str` (the printable-ASCII gate is a
    /// property of the JSON container and fires only at export);
    /// `add_*` return the node index `add_edge` wires together;
    /// [`NirBuilder::build`] runs the import-parity checks
    /// (duplicate names, duplicate edges) and yields a
    /// [`NirImport`] that exports and assembles like any other.
    #[derive(Debug)]
    pub struct NirBuilder<'a> {
        nodes: Vec<NirNode<'a>>,
        edges: Vec<(u32, u32)>,
        weights: Vec<i16>,
        lifs: Vec<NirLif>,
        opts: NirImportOptions,
    }

    impl<'a> NirBuilder<'a> {
        /// An empty graph under the given import options (dt + the
        /// voltage grid the LIF quantizer uses).
        #[must_use]
        pub fn new(opts: NirImportOptions) -> Self {
            Self {
                nodes: Vec::new(),
                edges: Vec::new(),
                weights: Vec::new(),
                lifs: Vec::new(),
                opts,
            }
        }

        /// Add an `Input` node (shape carrier; 1–4 dims).
        ///
        /// # Errors
        ///
        /// [`NirError::BadShape("shape")`] beyond 4 dims.
        pub fn add_input(&mut self, name: &'a str, shape: &[u32]) -> Result<usize, NirError<'a>> {
            self.push_node(name, NirNodeKind::Input, shape)
        }

        /// Add an `Output` node (shape carrier; 1–4 dims).
        ///
        /// # Errors
        ///
        /// [`NirError::BadShape("shape")`] beyond 4 dims.
        pub fn add_output(&mut self, name: &'a str, shape: &[u32]) -> Result<usize, NirError<'a>> {
            self.push_node(name, NirNodeKind::Output, shape)
        }

        /// Add a `LIF` population from per-neuron source-unit
        /// parameters — quantized via [`quantize_lif`](super::quantize_lif)
        /// under the builder's options. All slices must have equal
        /// length ≥ 1; `v_reset_v` `None` = the reference's
        /// absent-`v_reset` zeros semantics.
        ///
        /// Atomic: a failed add (including a mid-population
        /// quantize failure) leaves the builder exactly as it was.
        ///
        /// # Errors
        ///
        /// [`NirError::BadShape("LIF param")`] on length mismatch or
        /// empty arrays; every [`quantize_lif`](super::quantize_lif)
        /// hard failure otherwise.
        pub fn add_lif_population(
            &mut self,
            name: &'a str,
            params: &NirLifParams<'_>,
        ) -> Result<usize, NirError<'a>> {
            let n = params.tau_s.len();
            if n == 0
                || params.r_ohm.len() != n
                || params.v_leak_v.len() != n
                || params.v_threshold_v.len() != n
                || params.v_reset_v.is_some_and(|v| v.len() != n)
            {
                return Err(NirError::BadShape("LIF param"));
            }
            // Quantize the whole population into scratch FIRST: a
            // mid-population failure must leave the builder
            // untouched (no zombie node, no half-appended lif
            // records).
            let mut pop = Vec::with_capacity(n);
            for i in 0..n {
                let v_reset = match params.v_reset_v {
                    Some(v) => v[i],
                    None => 0.0,
                };
                pop.push(super::quantize_lif(
                    params.tau_s[i],
                    params.r_ohm[i],
                    params.v_leak_v[i],
                    params.v_threshold_v[i],
                    v_reset,
                    params.v_reset_v.is_none(),
                    self.opts,
                )?);
            }
            // Past this point nothing can fail (see add_linear).
            let idx = self.push_node(name, NirNodeKind::Lif, &[])?;
            let offset = self.lifs.len();
            self.lifs.extend_from_slice(&pop);
            self.nodes[idx].lif = Some(NirLifPopulation { offset, len: n });
            Ok(idx)
        }

        /// Add a `Linear` node from materialized row-major f64
        /// weights — quantized into the arena via
        /// [`quantize_linear`](super::quantize_linear).
        ///
        /// Atomic: a failed add leaves the builder exactly as it
        /// was — no node, no arena growth.
        ///
        /// # Errors
        ///
        /// Every [`quantize_linear`](super::quantize_linear) failure.
        pub fn add_linear(
            &mut self,
            name: &'a str,
            values: &[f64],
            rows: usize,
            cols: usize,
        ) -> Result<usize, NirError<'a>> {
            // Quantize into scratch FIRST: any failure must leave
            // the builder untouched (no zombie node, no zero tail
            // on the weights arena).
            let n = rows.checked_mul(cols).ok_or(NirError::BadShape("weight"))?;
            let mut scratch = vec![0i16; n];
            let mut lin = super::quantize_linear(values, rows, cols, &mut scratch, 0)?;
            // Past this point nothing can fail: push_node with an
            // empty shape is infallible.
            let idx = self.push_node(name, NirNodeKind::Linear, &[])?;
            lin.weight_offset = self.weights.len();
            self.weights.extend_from_slice(&scratch);
            self.nodes[idx].linear = Some(lin);
            Ok(idx)
        }

        /// Wire `from → to` (indices returned by the `add_*` calls).
        ///
        /// # Errors
        ///
        /// [`NirError::BadShape("edges")`] when an index names no
        /// node (the builder twin of import's endpoint resolution).
        pub fn add_edge(&mut self, from: usize, to: usize) -> Result<(), NirError<'a>> {
            let end = self.nodes.len();
            if from >= end || to >= end {
                return Err(NirError::BadShape("edges"));
            }
            self.edges.push((from as u32, to as u32));
            Ok(())
        }

        /// Finish: import-parity checks (duplicate node names,
        /// duplicate edges — the reference `validate_structure`
        /// semantics) and hand over the graph as a [`NirImport`].
        ///
        /// # Errors
        ///
        /// [`NirError::DuplicateNodeName`], [`NirError::DuplicateEdge`].
        pub fn build(self) -> Result<NirImport<'a>, NirError<'a>> {
            for i in 0..self.nodes.len() {
                for j in (i + 1)..self.nodes.len() {
                    if self.nodes[i].name == self.nodes[j].name {
                        return Err(NirError::DuplicateNodeName);
                    }
                }
            }
            for i in 0..self.edges.len() {
                for j in (i + 1)..self.edges.len() {
                    if self.edges[i] == self.edges[j] {
                        return Err(NirError::DuplicateEdge);
                    }
                }
            }
            Ok(NirImport {
                nodes: self.nodes,
                edges: self.edges,
                weights: self.weights,
                lifs: self.lifs,
                opts: self.opts,
                ref_sha: NIR_REF_SHA,
            })
        }

        fn push_node(
            &mut self,
            name: &'a str,
            kind: NirNodeKind,
            shape: &[u32],
        ) -> Result<usize, NirError<'a>> {
            if shape.len() > 4 {
                return Err(NirError::BadShape("shape"));
            }
            let mut s = [0u32; 4];
            s[..shape.len()].copy_from_slice(shape);
            self.nodes.push(NirNode {
                name,
                kind,
                shape: s,
                shape_len: shape.len(),
                lif: None,
                linear: None,
            });
            Ok(self.nodes.len() - 1)
        }
    }
}

#[cfg(feature = "std")]
pub use std_assembly::{ChainEncoder, NirBuilder, NirImport};

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    const CHAIN: &str = "{\"version\":\"test\",\"node\":{\"type\":\"NIRGraph\",\
\"edges\":[[\"input\",\"linear\"],[\"linear\",\"lif\"],[\"lif\",\"output\"]],\
\"nodes\":{\"input\":{\"type\":\"Input\",\"shape\":[3]},\
\"linear\":{\"type\":\"Linear\",\"weight\":[[0.5,-1.0,0.25],[0.0,0.75,-0.5]]},\
\"lif\":{\"type\":\"LIF\",\"tau\":[0.02,0.02],\"r\":[100000000.0,100000000.0],\
\"v_leak\":[-0.07,-0.07],\"v_threshold\":[-0.055,-0.055],\"v_reset\":[-0.08,-0.08]},\
\"output\":{\"type\":\"Output\",\"shape\":[2]}}}}";

    #[test]
    fn scan_counts_the_chain() {
        let s = nir_scan(CHAIN.as_bytes()).expect("chain scans");
        assert_eq!(s.version, "test");
        assert_eq!(s.node_count, 4);
        assert_eq!(s.edge_count, 3);
        assert_eq!(s.weight_cells, 6);
        assert_eq!(s.lif_neurons, 2);
    }

    #[test]
    fn scan_rejects_malformed() {
        assert!(matches!(
            nir_scan(b"{not json"),
            Err(NirError::Json(_))
        ));
        assert!(matches!(
            nir_scan(b"{}"),
            Err(NirError::MissingField("version"))
        ));
        assert!(matches!(
            nir_scan(b"{\"version\":\"x\"}"),
            Err(NirError::MissingField("node"))
        ));
    }

    #[allow(clippy::type_complexity)]
    fn import_chain(
        opts: NirImportOptions,
    ) -> (Vec<NirNode<'static>>, Vec<(u32, u32)>, Vec<i16>, Vec<NirLif>, NirReport) {
        let scan = nir_scan(CHAIN.as_bytes()).unwrap();
        let mut nodes = vec![NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }; scan.node_count];
        let mut edges = vec![(0u32, 0u32); scan.edge_count];
        let mut weights = vec![0i16; scan.weight_cells];
        let mut lifs = vec![NirLif::default(); scan.lif_neurons];
        let mut scratch = vec![0f64; scan.weight_cells + 5 * scan.lif_neurons];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        let report = nir_import(CHAIN.as_bytes(), opts, &mut bufs).expect("chain imports");
        (nodes, edges, weights, lifs, report)
    }

    #[test]
    fn import_quantizes_the_chain() {
        let (nodes, edges, weights, lifs, report) =
            import_chain(NirImportOptions::default());
        assert_eq!(
            edges,
            vec![(0, 1), (1, 2), (2, 3)],
            "edges resolve by name to document order"
        );
        assert_eq!((report.inputs, report.linears, report.lifs, report.outputs), (1, 1, 1, 1));
        assert_eq!(report.weight_cells, 6);

        let pop = nodes[2].lif.expect("lif present");
        assert_eq!(pop.len, 2, "a 2x3 Linear feeds a 2-neuron population");
        assert_eq!(pop.offset, 0);
        let lif = lifs[pop.offset];
        assert_eq!(lif.tau_us, 20_000);
        assert_eq!(lif.resistance_mohm, 100);
        assert_eq!(lif.leak_q, -70);
        assert_eq!(lif.threshold_q, -55);
        assert_eq!(lif.reset_q, -80);
        assert_eq!(lif.capacitance_pf, 200, "C = tau/r: 0.02/1e8 F = 200 pF");

        let lin = nodes[1].linear.expect("linear present");
        assert_eq!((lin.rows, lin.cols), (2, 3));
        assert!((lin.scale - 1.0 / I16_FS).abs() < 1e-18, "absmax 1.0");
        // q = round(w/scale) = w*32767 exactly for these dyadics
        let q: Vec<i32> = weights.iter().map(|&q| i32::from(q)).collect();
        assert_eq!(
            q,
            vec![16384, -32767, 8192, 0, 24575, -16384]
        );
        // scale = 1/32767 is non-dyadic: dequant error ≤ scale/2 and
        // the loss note fires — loud lossiness doing its job
        assert!(lin.max_abs_err > 0.0 && lin.max_abs_err <= lin.scale / 2.0);
        assert!(report.notes[NirNote::QuantizationLoss as usize] >= 1);
    }

    #[test]
    fn lif_hard_failures() {
        // tau < dt: the whole import must fail loudly
        let opts = NirImportOptions {
            dt_us: 30_000,
            ..NirImportOptions::default()
        };
        let scan = nir_scan(CHAIN.as_bytes()).unwrap();
        let mut nodes = vec![NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }; scan.node_count];
        let mut edges = vec![(0u32, 0u32); scan.edge_count];
        let mut weights = vec![0i16; scan.weight_cells];
        let mut lifs = vec![NirLif::default(); scan.lif_neurons];
        let mut scratch = vec![0f64; scan.weight_cells + 5 * scan.lif_neurons];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        assert_eq!(
            nir_import(CHAIN.as_bytes(), opts, &mut bufs),
            Err(NirError::TauBelowDt)
        );
        // direct quantize_lif contract:
        assert!(matches!(
            quantize_lif(0.0, 1e8, -0.07, -0.055, -0.08, false, NirImportOptions::default()),
            Err(NirError::BadNumber("tau"))
        ));
        assert!(matches!(
            quantize_lif(-0.02, 1e8, -0.07, -0.055, -0.08, false, NirImportOptions::default()),
            Err(NirError::BadNumber("tau"))
        ));
        assert!(matches!(
            quantize_lif(0.02, 0.0, -0.07, -0.055, -0.08, false, NirImportOptions::default()),
            Err(NirError::BadNumber("r"))
        ));
        assert!(matches!(
            quantize_lif(0.02, 1e8, -0.07, -0.0004, -0.08, false, NirImportOptions::default()),
            Err(NirError::ThresholdZero)
        ));
        assert!(matches!(
            quantize_lif(0.02, 1e8, -0.07, -0.055, -0.08, false, NirImportOptions::new(30_000, VoltageResolution::Millivolt)),
            Err(NirError::TauBelowDt)
        ));
        assert!(matches!(
            quantize_lif(0.02, 1e8, -0.07, 0.06, -0.08, false, NirImportOptions::default()),
            Err(NirError::PotentialOutOfRange("v_threshold"))
        ));
        // threshold on the centi grid: -0.0555 V is representable
        let lif = quantize_lif(
            0.02,
            1e8,
            -0.07,
            -0.0555,
            -0.08,
            false,
            NirImportOptions::new(1_000, VoltageResolution::CentiMillivolt),
        )
        .expect("centi grid");
        assert_eq!(lif.threshold_q, -5550);
    }

    #[test]
    fn unknown_kinds_and_affine_reject_loudly() {
        let doc = "{\"version\":\"x\",\"node\":{\"type\":\"NIRGraph\",\"edges\":[],\
\"nodes\":{\"a\":{\"type\":\"Affine\",\"weight\":[[1.0]],\"bias\":[0.5]}}}}";
        let mut nodes = [NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }];
        let mut edges = [];
        let mut weights = [0i16; 8];
        let mut lifs = [];
        let mut scratch = [0f64; 8];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        assert_eq!(
            nir_import(doc.as_bytes(), NirImportOptions::default(), &mut bufs),
            Err(NirError::UnsupportedNodeKind("Affine"))
        );
    }

    #[test]
    fn edges_before_nodes_resolve() {
        // key order edges-then-nodes (our own export order)
        let doc = "{\"version\":\"x\",\"node\":{\"edges\":[[\"a\",\"b\"]],\
\"type\":\"NIRGraph\",\"nodes\":{\"a\":{\"type\":\"Input\",\"shape\":[1]},\
\"b\":{\"type\":\"Output\",\"shape\":[1]}}}}";
        let mut nodes = [NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }; 2];
        let mut edges = [(0u32, 0u32); 1];
        let mut weights = [];
        let mut lifs = [];
        let mut scratch = [];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        nir_import(doc.as_bytes(), NirImportOptions::default(), &mut bufs).expect("resolves");
        assert_eq!(edges[0], (0, 1));
    }

    #[test]
    fn structural_rejections() {
        let mk = |nodes: &str, edges: &str| {
            format!(
                "{{\"version\":\"x\",\"node\":{{\"type\":\"NIRGraph\",\"edges\":{edges},\"nodes\":{nodes}}}}}"
            )
        };
        #[allow(clippy::type_complexity)]
        let cases: Vec<(&str, String, fn(&NirError) -> bool)> = vec![
            (
                "unknown endpoint",
                mk("{\"a\":{\"type\":\"Input\",\"shape\":[1]}}", "[[\"a\",\"zz\"]]"),
                |e| matches!(e, NirError::UnknownEdgeEndpoint("zz")),
            ),
            (
                "duplicate edge",
                mk(
                    "{\"a\":{\"type\":\"Input\",\"shape\":[1]},\"b\":{\"type\":\"Output\",\"shape\":[1]}}",
                    "[[\"a\",\"b\"],[\"a\",\"b\"]]",
                ),
                |e| matches!(e, NirError::DuplicateEdge),
            ),
            (
                "duplicate node name",
                mk(
                    "{\"a\":{\"type\":\"Input\",\"shape\":[1]},\"a\":{\"type\":\"Output\",\"shape\":[1]}}",
                    "[]",
                ),
                |e| matches!(e, NirError::DuplicateNodeName),
            ),
            (
                "escaped name",
                "{\"version\":\"x\",\"node\":{\"type\":\"NIRGraph\",\"edges\":[],\"nodes\":{\"a\\u0041\":{\"type\":\"Input\",\"shape\":[1]}}}}".to_string(),
                |e| matches!(e, NirError::EscapedOrNonAsciiString(_)),
            ),
            (
                "3d weight",
                mk(
                    "{\"a\":{\"type\":\"Linear\",\"weight\":[[[1.0]]]}}",
                    "[]",
                ),
                |e| matches!(e, NirError::BadShape("weight")),
            ),
            (
                "ragged weight",
                mk(
                    "{\"a\":{\"type\":\"Linear\",\"weight\":[[1.0],[1.0,2.0]]}}",
                    "[]",
                ),
                |e| matches!(e, NirError::BadShape("weight")),
            ),
        ];
        for (label, doc, check) in cases {
            let mut nodes = [NirNode {
                name: "",
                kind: NirNodeKind::Input,
                shape: [0; 4],
                shape_len: 0,
                lif: None,
                linear: None,
            }; 4];
            let mut edges = [(0u32, 0u32); 4];
            let mut weights = [0i16; 64];
            let mut lifs = [];
            let mut scratch = [0f64; 64];
            let mut bufs = NirBuffers {
                nodes: &mut nodes,
                edges: &mut edges,
                weights: &mut weights,
                lifs: &mut lifs,
                scratch: &mut scratch,
            };
            let err = nir_import(doc.as_bytes(), NirImportOptions::default(), &mut bufs)
                .expect_err(label);
            assert!(check(&err), "{label}: got {err:?}");
        }
    }

    #[test]
    fn buffer_overflow_is_loud() {
        // node capacity: the nodes check fires before weights are read
        let mut nodes = [NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }];
        let mut edges = [];
        let mut weights = [];
        let mut lifs = [];
        let mut scratch = [];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        assert_eq!(
            nir_import(CHAIN.as_bytes(), NirImportOptions::default(), &mut bufs),
            Err(NirError::BufferOverflow)
        );
        // arena too small: staging completes (scratch fits the 6
        // cells), the quantizer refuses — offset+6 > 2
        let mut nodes = [NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }; 4];
        let mut edges = [(0u32, 0u32); 3];
        let mut weights = [0i16; 2];
        let mut lifs = [NirLif::default(); 1];
        let mut scratch = [0f64; 11]; // 6 weight + 5 LIF staging
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        assert_eq!(
            nir_import(CHAIN.as_bytes(), NirImportOptions::default(), &mut bufs),
            Err(NirError::BufferOverflow)
        );
        // scratch too small: the staging loop refuses mid-tensor
        let mut nodes = [NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }; 4];
        let mut edges = [(0u32, 0u32); 3];
        let mut weights = [0i16; 64];
        let mut lifs = [NirLif::default(); 1];
        let mut scratch = [0f64; 2];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        assert_eq!(
            nir_import(CHAIN.as_bytes(), NirImportOptions::default(), &mut bufs),
            Err(NirError::BufferOverflow)
        );
    }

    #[test]
    fn export_then_import_is_idempotent() {
        let opts = NirImportOptions::default();
        let (nodes, edges, weights, lifs, _report) = import_chain(opts);

        let mut out = [0u8; 2048];
        let n = nir_export(&nodes, &edges, &weights, &lifs, opts, &mut out).expect("exports");
        let exported = core::str::from_utf8(&out[..n]).expect("utf8");

        // byte-stable: second export identical
        let mut out2 = [0u8; 2048];
        let n2 = nir_export(&nodes, &edges, &weights, &lifs, opts, &mut out2).unwrap();
        assert_eq!(&out[..n], &out2[..n2], "export is byte-stable");

        // re-import reproduces the graph bit-for-bit
        let scan2 = nir_scan(exported.as_bytes()).unwrap();
        assert_eq!(scan2.version, EXPORT_VERSION);
        let mut nodes2 = vec![NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }; scan2.node_count];
        let mut edges2 = vec![(0u32, 0u32); scan2.edge_count];
        let mut weights2 = vec![0i16; scan2.weight_cells];
        let mut lifs2 = vec![NirLif::default(); scan2.lif_neurons];
        let mut scratch2 = vec![0f64; scan2.weight_cells + 5 * scan2.lif_neurons];
        let mut bufs2 = NirBuffers {
            nodes: &mut nodes2,
            edges: &mut edges2,
            weights: &mut weights2,
            lifs: &mut lifs2,
            scratch: &mut scratch2,
        };
        let report2 = nir_import(exported.as_bytes(), opts, &mut bufs2).expect("re-imports");
        assert_eq!(edges2, edges);
        assert_eq!(weights2, weights, "quantized weights identical");
        // The CONTRACT is quantized-state identity. The re-import's
        // notes MAY grow: the dequantized max (|q|·scale in f64) can
        // re-derive a scale 1 ulp off, which the lossy-tensor note
        // honestly records while the state stays identical.
        let _ = report2;
        let pop1 = nodes[2].lif.unwrap();
        let pop2 = nodes2[2].lif.unwrap();
        let lif1 = lifs[pop1.offset];
        let lif2 = lifs2[pop2.offset];
        assert_eq!(lif2, lif1, "the whole quantized record survives the round trip");
        let linear_out = nodes2[1].linear.unwrap();
        assert_eq!(linear_out.scale, nodes[1].linear.unwrap().scale, "scale recovered exactly");
        // provenance rode metadata: the ORIGINAL source floats survive
        assert_eq!(lif2.tau_s, lif1.tau_s);
    }

    #[test]
    fn exported_json_is_valid_shape() {
        let opts = NirImportOptions::default();
        let (nodes, edges, weights, lifs, _) = import_chain(opts);
        let mut out = [0u8; 2048];
        let n = nir_export(&nodes, &edges, &weights, &lifs, opts, &mut out).unwrap();
        let s = core::str::from_utf8(&out[..n]).unwrap();
        assert!(s.starts_with("{\"version\":\"nir@7883c3c\",\"node\":{\"type\":\"NIRGraph\""));
        assert!(s.contains("\"metadata\":{\"neuralos\":{\"provenance\":{\"absmax\":1}"));
        assert!(s.ends_with("}}}"));
        // one version block only
        assert_eq!(s.matches("\"version\"").count(), 1);
        // balanced braces
        let mut depth = 0i32;
        for c in s.chars() {
            if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
            }
        }
        assert_eq!(depth, 0, "braces balance");
    }

    #[test]
    fn zero_tensor_and_lossy_notes() {
        let doc = "{\"version\":\"x\",\"node\":{\"type\":\"NIRGraph\",\"edges\":[],\
\"nodes\":{\"a\":{\"type\":\"Linear\",\"weight\":[[0.0,0.0],[0.0,0.0]]}}}}";
        let mut nodes = [NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }];
        let mut edges = [];
        let mut weights = [0i16; 64];
        let mut lifs = [];
        let mut scratch = [0f64; 64];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        let rep = nir_import(doc.as_bytes(), NirImportOptions::default(), &mut bufs).unwrap();
        assert!(rep.notes[NirNote::ZeroWeightTensor as usize] >= 1);
        assert_eq!(nodes[0].linear.unwrap().scale, 1.0);

        // a lossy tensor (0.1 is not dyadic) is noted
        let doc2 = doc.replace("[[0.0,0.0],[0.0,0.0]]", "[[0.1,0.3]]");
        let mut bufs2 = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        let rep2 = nir_import(doc2.as_bytes(), NirImportOptions::default(), &mut bufs2).unwrap();
        assert!(rep2.notes[NirNote::QuantizationLoss as usize] >= 1);
    }

    #[test]
    fn v_reset_defaults_to_zero_with_note() {
        // reference from_dict semantics: absent v_reset = zeros
        let doc = "{\"version\":\"x\",\"node\":{\"type\":\"NIRGraph\",\"edges\":[],\
\"nodes\":{\"a\":{\"type\":\"LIF\",\"tau\":[0.02],\"r\":[100000000.0],\
\"v_leak\":[-0.07],\"v_threshold\":[-0.055]}}}}";
        let mut nodes = [NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }];
        let mut edges = [];
        let mut weights = [];
        let mut lifs = [NirLif::default(); 1];
        let mut scratch = [0f64; 5];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        let rep = nir_import(doc.as_bytes(), NirImportOptions::default(), &mut bufs).unwrap();
        assert!(rep.notes[NirNote::VResetDefaulted as usize] >= 1);
        let pop = nodes[0].lif.unwrap();
        assert_eq!(pop.len, 1);
        let lif = lifs[pop.offset];
        assert_eq!(lif.reset_q, 0);
        assert!(lif.v_reset_defaulted);
    }

    // ---- fresh-eyes review (R9): adversarial pins for each fix ----

    #[test]
    fn skip_value_is_depth_capped() {
        let nest = |n: usize| format!("{}0{}", "[".repeat(n), "]".repeat(n));
        // deep junk at top level — a Json error, not a stack overflow
        let doc = format!("{{\"version\":\"x\",\"junk\":{}}}", nest(200));
        assert!(matches!(nir_scan(doc.as_bytes()), Err(NirError::Json(_))));
        // deep junk inside a node (the metadata path)
        let doc2 = format!(
            "{{\"version\":\"x\",\"node\":{{\"type\":\"NIRGraph\",\"edges\":[],\
             \"nodes\":{{\"a\":{{\"type\":\"Input\",\"shape\":[1],\"deep\":{}}}}}}}}}",
            nest(200)
        );
        assert!(matches!(nir_scan(doc2.as_bytes()), Err(NirError::Json(_))));
        // generous-but-legal nesting still passes (metadata headroom)
        let ok = format!(
            "{{\"version\":\"x\",\"node\":{{\"type\":\"NIRGraph\",\"edges\":[],\"nodes\":{{}}}},\"junk\":{}}}",
            nest(40)
        );
        assert!(nir_scan(ok.as_bytes()).is_ok());
    }

    #[test]
    fn trailing_content_after_root_is_rejected() {
        let mut b = CHAIN.as_bytes().to_vec();
        b.extend_from_slice(b"garbage");
        assert!(matches!(nir_scan(&b), Err(NirError::Json(_))));
        // even well-formed extra JSON after the root
        let mut b2 = CHAIN.as_bytes().to_vec();
        b2.extend_from_slice(b" {}");
        assert!(matches!(nir_scan(&b2), Err(NirError::Json(_))));
        // trailing whitespace is fine (the fixtures end with \n)
        let mut b3 = CHAIN.as_bytes().to_vec();
        b3.extend_from_slice(b" \n\t");
        assert!(nir_scan(&b3).is_ok());
        // the buffer-import walks enforce it too
        let doc = b"{\"version\":\"x\",\"node\":{\"type\":\"NIRGraph\",\"edges\":[],\
\"nodes\":{\"a\":{\"type\":\"Input\",\"shape\":[1]}}}} x";
        let mut nodes = [NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }];
        let mut edges = [];
        let mut weights = [];
        let mut lifs = [];
        let mut scratch = [];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
            lifs: &mut lifs,
            scratch: &mut scratch,
        };
        assert!(matches!(
            nir_import(doc, NirImportOptions::default(), &mut bufs),
            Err(NirError::Json(_))
        ));
    }

    #[test]
    fn denormal_absmax_scale_is_a_loud_error() {
        // absmax/32767 underflows to exactly 0.0: the record would
        // lie (q·scale = 0 ≠ absmax) and export would silently zero
        // the tensor, breaking idempotence
        for tok in ["5e-324", "1e-320", "2.47e-321"] {
            let doc = format!(
                "{{\"version\":\"x\",\"node\":{{\"type\":\"NIRGraph\",\"edges\":[],\
                 \"nodes\":{{\"a\":{{\"type\":\"Linear\",\"weight\":[[{tok}]]}}}}}}}}"
            );
            let mut nodes = [NirNode {
                name: "",
                kind: NirNodeKind::Input,
                shape: [0; 4],
                shape_len: 0,
                lif: None,
                linear: None,
            }];
            let mut edges = [];
            let mut weights = [0i16; 8];
            let mut lifs = [];
            let mut scratch = [0f64; 8];
            let mut bufs = NirBuffers {
                nodes: &mut nodes,
                edges: &mut edges,
                weights: &mut weights,
                lifs: &mut lifs,
                scratch: &mut scratch,
            };
            assert_eq!(
                nir_import(doc.as_bytes(), NirImportOptions::default(), &mut bufs),
                Err(NirError::BadNumber("weight")),
                "{tok}"
            );
        }
    }

    #[test]
    fn round_half_away_pins_half_boundaries() {
        // exact halves round away from zero
        assert_eq!(round_half_away(0.5), 1.0);
        assert_eq!(round_half_away(-0.5), -1.0);
        assert_eq!(round_half_away(2.5), 3.0);
        assert_eq!(round_half_away(-2.5), -3.0);
        // one ulp BELOW a half must round down — the classic
        // add-±0.5-then-truncate idiom rounded these UP (review
        // finding: 0.49999999999999994 + 0.5 == 1.0 in f64)
        let below = |x: f64| f64::from_bits(x.to_bits() - 1);
        assert_eq!(round_half_away(below(0.5)), 0.0);
        assert_eq!(round_half_away(below(1.5)), 1.0);
        assert_eq!(round_half_away(below(2.5)), 2.0);
        assert_eq!(round_half_away(below(10.5)), 10.0);
        // reachability: r one ulp below 0.5 MΩ is a loud BadNumber,
        // not a silent 1 MΩ
        assert!(matches!(
            quantize_lif(
                0.02,
                499_999.999_999_999_94,
                -0.07,
                -0.055,
                -0.08,
                false,
                NirImportOptions::default()
            ),
            Err(NirError::BadNumber("r"))
        ));
    }

    #[test]
    fn lif_population_semantics() {
        // mismatched param lengths are structural: scan rejects
        let mk_lif = |tau: &str, r: &str| {
            format!(
                "{{\"version\":\"x\",\"node\":{{\"type\":\"NIRGraph\",\"edges\":[],\
                 \"nodes\":{{\"a\":{{\"type\":\"LIF\",\"tau\":{tau},\"r\":{r},\
                 \"v_leak\":[-0.07,-0.07],\"v_threshold\":[-0.055,-0.055],\
                 \"v_reset\":[-0.08,-0.08]}}}}}}}}"
            )
        };
        assert!(matches!(
            nir_scan(mk_lif("[0.02,0.03]", "[1e8]").as_bytes()),
            Err(NirError::BadShape("LIF param"))
        ));
        // a coherent 2-neuron population scans and counts
        let ok = mk_lif("[0.02,0.03]", "[1e8,1e8]");
        let s = nir_scan(ok.as_bytes()).expect("population scans");
        assert_eq!(s.lif_neurons, 2);

        // assembly parity (reference type-check): population size
        // must equal the feeding Linear's rows
        let mut bld = NirBuilder::new(NirImportOptions::default());
        let inp = bld.add_input("i", &[3]).expect("input");
        let lin = bld
            .add_linear("l", &[1.0, 1.0, 1.0], 1, 3)
            .expect("linear");
        let lif = bld
            .add_lif_population(
                "n",
                &NirLifParams {
                    tau_s: &[0.02, 0.03],
                    r_ohm: &[1e8, 1e8],
                    v_leak_v: &[-0.07, -0.07],
                    v_threshold_v: &[-0.055, -0.055],
                    v_reset_v: Some(&[-0.08, -0.08]),
                },
            )
            .expect("lif");
        let out = bld.add_output("o", &[1]).expect("output");
        for (a, c) in [(inp, lin), (lin, lif), (lif, out)] {
            bld.add_edge(a, c).expect("edge");
        }
        let g = bld.build().expect("builds — the format layer permits it");
        assert!(matches!(
            g.build_chain_network(),
            Err(NirError::UnsupportedTopology("LIF population != Linear rows"))
        ));
    }

    #[test]
    fn scan_rejects_non_2d_weight_shapes() {
        let mk = |weight: &str| {
            format!(
                "{{\"version\":\"x\",\"node\":{{\"type\":\"NIRGraph\",\"edges\":[],\
                 \"nodes\":{{\"a\":{{\"type\":\"Linear\",\"weight\":{weight}}}}}}}}}"
            )
        };
        // 1-D, empty outer, empty row, empty+full rows, 3-D — all at
        // SCAN now (the "malformed structure fails here" contract;
        // previously 1-D/empty slipped through to import)
        for w in ["[1.0, 2.0]", "[]", "[[]]", "[[],[1.0]]", "[[[1.0]]]"] {
            assert!(
                matches!(nir_scan(mk(w).as_bytes()), Err(NirError::BadShape("weight"))),
                "scan must reject {w}"
            );
        }
        // 2-D non-empty scans; raggedness remains import's check
        assert!(nir_scan(mk("[[1.0],[1.0,2.0]]").as_bytes()).is_ok());
    }

    #[test]
    fn export_rejects_dangling_edge_indices() {
        let (nodes, _edges, weights, lifs, _report) = import_chain(NirImportOptions::default());
        let mut out = [0u8; 512];
        assert_eq!(
            nir_export(
                &nodes,
                &[(0, 99)],
                &weights,
                &lifs,
                NirImportOptions::default(),
                &mut out
            ),
            Err(NirError::BadShape("edges")),
            "edge indices must name nodes — no silent \"?\" placeholders"
        );
    }

    #[test]
    fn encoder_saturates_instead_of_wrapping() {
        // all-max weights × all-max inputs: 3·32767²/100 = 32212250
        // clamps to 32767 — the i32 accumulator produced a NEGATIVE
        // output here before the fix (release mode: a silent wrap)
        let doc = "{\"version\":\"x\",\"node\":{\"type\":\"NIRGraph\",\"edges\":[\
[\"i\",\"l\"],[\"l\",\"n\"],[\"n\",\"o\"]],\"nodes\":{\"i\":{\"type\":\"Input\",\"shape\":[3]},\
\"l\":{\"type\":\"Linear\",\"weight\":[[1.0,1.0,1.0]]},\
\"n\":{\"type\":\"LIF\",\"tau\":[0.02],\"r\":[100000000.0],\"v_leak\":[-0.07],\
\"v_threshold\":[-0.055],\"v_reset\":[-0.08]},\"o\":{\"type\":\"Output\",\"shape\":[1]}}}}";
        let g =
            NirImport::from_json(doc.as_bytes(), NirImportOptions::default()).expect("imports");
        let (_net, enc) = g.build_chain_network().expect("canonical chain");
        assert_eq!(enc.encode(&[32767, 32767, 32767]), vec![32767]);
        // negative saturation lands on i16::MIN (-32768)
        assert_eq!(enc.encode(&[-32767, -32767, -32767]), vec![-32768]);
        // arithmetic sanity: w·x/100 truncates
        assert_eq!(enc.encode(&[1, 0, 0]), vec![327]);
    }

    // ---- structured-entry seam: quantize_linear / quantize_lif ----

    #[test]
    fn quantize_linear_dyadic_vector_is_exact() {
        // the gate vector, direct: [0.5,-1,0.25] at absmax 1.0
        let vals = [0.5, -1.0, 0.25];
        let mut arena = [0i16; 8];
        let lin = quantize_linear(&vals, 1, 3, &mut arena, 0).expect("quantizes");
        assert_eq!((lin.rows, lin.cols, lin.weight_offset), (1, 3, 0));
        assert!((lin.scale - 1.0 / I16_FS).abs() < 1e-18);
        assert_eq!(&arena[..3], &[16384, -32767, 8192]);
        assert!(
            lin.max_abs_err > 0.0 && lin.max_abs_err <= lin.scale / 2.0,
            "scale 1/32767 is non-dyadic: bounded loss, recorded"
        );
        // offset placement: the record views exactly the slice it wrote
        let lin2 = quantize_linear(&vals, 1, 3, &mut arena, 3).expect("quantizes");
        assert_eq!(lin2.weight_offset, 3);
        assert_eq!(&arena[3..6], &[16384, -32767, 8192]);
        assert_eq!(lin2.scale, lin.scale, "same tensor, same scale");
        // absmax 32767 → scale exactly 1.0: integers are lossless
        let lin3 = quantize_linear(&[32767.0, -16384.0, 0.0], 1, 3, &mut arena, 0)
            .expect("quantizes");
        assert_eq!(lin3.scale, 1.0);
        assert_eq!(lin3.max_abs_err, 0.0);
        assert_eq!(&arena[..3], &[32767, -16384, 0]);
    }

    #[test]
    fn quantize_linear_zero_tensor_and_full_scale() {
        let mut arena = [7i16; 4];
        let lin = quantize_linear(&[0.0; 4], 2, 2, &mut arena, 0).expect("quantizes");
        assert!(lin.zero_tensor);
        assert_eq!(lin.scale, 1.0);
        assert_eq!(lin.max_abs_err, 0.0);
        assert_eq!(&arena[..4], &[0, 0, 0, 0]);
        // the max-|w| element maps to ±32767 by construction
        // (1.0/(3/32767) = 10922.33 — far from the .5 boundary)
        let lin2 = quantize_linear(&[3.0, -3.0, 1.0], 1, 3, &mut arena, 0).expect("quantizes");
        assert!(!lin2.zero_tensor);
        assert_eq!(lin2.absmax, 3.0);
        assert_eq!(&arena[..3], &[32767, -32767, 10922]);
    }

    #[test]
    fn quantize_linear_denormal_absmax_is_loud() {
        // absmax/32767 underflows to 0.0: the record would lie
        // (q·scale = 0 ≠ absmax) and export would zero the tensor
        for v in [5e-324, 1e-320, 2.47e-321] {
            let mut arena = [0i16; 4];
            assert_eq!(
                quantize_linear(&[v], 1, 1, &mut arena, 0),
                Err(NirError::BadNumber("weight")),
                "{v}"
            );
        }
    }

    #[test]
    fn quantize_linear_rejects_loudly() {
        let mut arena = [0i16; 8];
        // non-finite (`1e400` reaches this same door via JSON parse)
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                quantize_linear(&[1.0, v], 1, 2, &mut arena, 0),
                Err(NirError::BadNumber("weight"))
            );
        }
        // len mismatch / zero dims
        assert_eq!(
            quantize_linear(&[1.0], 1, 2, &mut arena, 0),
            Err(NirError::BadShape("weight"))
        );
        assert_eq!(
            quantize_linear(&[], 0, 3, &mut arena, 0),
            Err(NirError::BadShape("weight"))
        );
        assert_eq!(
            quantize_linear(&[1.0], 1, 0, &mut arena, 0),
            Err(NirError::BadShape("weight"))
        );
        // arena bounds (including offset arithmetic overflow)
        assert_eq!(
            quantize_linear(&[1.0; 6], 2, 3, &mut arena, 4),
            Err(NirError::BufferOverflow)
        );
        assert_eq!(
            quantize_linear(&[1.0; 6], 2, 3, &mut arena, usize::MAX - 2),
            Err(NirError::BufferOverflow)
        );
    }

    use proptest::prelude::*;

    proptest! {
        /// The R9 200k-fuzz invariants, property-pinned: bounded
        /// dequant error, full-scale max element, and
        /// re-quantize(dequant) == q — the core of export
        /// idempotence.
        #[test]
        fn prop_quantize_linear_round_trip(
            rows in 1usize..=4,
            cols in 1usize..=5,
            values in proptest::collection::vec(-1024.0f64..1024.0, 20usize..=20),
        ) {
            let n = rows * cols;
            let values = &values[..n];
            let mut arena = [0i16; 32];
            let lin = quantize_linear(values, rows, cols, &mut arena, 0)
                .expect("finite values quantize");
            prop_assert!(lin.max_abs_err <= lin.scale * (0.5 + 1e-9));
            if lin.zero_tensor {
                prop_assert_eq!(lin.scale, 1.0);
            } else {
                prop_assert!(arena[..n].iter().any(|&q| q.abs() == 32767));
            }
            let deq: Vec<f64> = arena[..n].iter().map(|&q| f64::from(q) * lin.scale).collect();
            let mut arena2 = [0i16; 32];
            quantize_linear(&deq, rows, cols, &mut arena2, 0).expect("re-quantizes");
            prop_assert_eq!(&arena2[..n], &arena[..n]);
        }
    }

    // ---- structured-entry seam: the typed builder ----

    fn lif_params_single() -> NirLifParams<'static> {
        NirLifParams {
            tau_s: &[0.02],
            r_ohm: &[1e8],
            v_leak_v: &[-0.07],
            v_threshold_v: &[-0.055],
            v_reset_v: Some(&[-0.08]),
        }
    }

    fn builder_chain() -> NirImport<'static> {
        let mut bld = NirBuilder::new(NirImportOptions::default());
        let inp = bld.add_input("input", &[3]).expect("input");
        let lin = bld
            .add_linear(
                "linear",
                &[0.5, -1.0, 0.25, 0.0, 0.75, -0.5],
                2,
                3,
            )
            .expect("linear");
        let lif = bld
            .add_lif_population(
                "lif",
                &NirLifParams {
                    tau_s: &[0.02, 0.02],
                    r_ohm: &[1e8, 1e8],
                    v_leak_v: &[-0.07, -0.07],
                    v_threshold_v: &[-0.055, -0.055],
                    v_reset_v: Some(&[-0.08, -0.08]),
                },
            )
            .expect("lif");
        let out = bld.add_output("output", &[2]).expect("output");
        for (a, c) in [(inp, lin), (lin, lif), (lif, out)] {
            bld.add_edge(a, c).expect("edge");
        }
        bld.build().expect("builds")
    }

    #[test]
    fn builder_matches_the_json_path_exactly() {
        let built = builder_chain();
        let (jn, je, jw, jl, _rep) = import_chain(NirImportOptions::default());
        // same quantized state, node for node
        assert_eq!(built.edges, je);
        assert_eq!(built.weights, jw);
        assert_eq!(built.lifs, jl, "every LIF record quantized identically");
        for (b, j) in built.nodes.iter().zip(jn.iter()) {
            assert_eq!(b.name, j.name);
            assert_eq!(b.kind, j.kind);
            assert_eq!((b.shape, b.shape_len), (j.shape, j.shape_len));
            assert_eq!(b.lif, j.lif, "population views identical");
            assert_eq!(b.linear, j.linear, "Linear quantized identically");
        }
    }

    #[test]
    fn builder_chain_assembles_and_exports() {
        let g = builder_chain();
        let (mut net, enc) = g.build_chain_network().expect("canonical chain");
        assert_eq!((net.neuron_count(), enc.rows(), enc.cols()), (2, 2, 3));
        let spikes: usize = (0..100)
            .map(|_| net.step(&enc.encode(&[400, 0, 0])).unwrap().len())
            .sum();
        assert!(spikes > 0, "the built chain fires");

        // export → JSON boundary → re-import: state-identical
        let mut out = vec![0u8; 2048];
        let n = nir_export(&g.nodes, &g.edges, &g.weights, &g.lifs, g.opts, &mut out)
            .expect("exports");
        let g2 = NirImport::from_json(&out[..n], g.opts).expect("re-imports");
        assert_eq!(g2.weights, g.weights);
        assert_eq!(g2.edges, g.edges);
        assert_eq!(g2.lifs, g.lifs, "every quantized record survives");
    }

    #[test]
    fn ascii_gate_fires_at_export_only() {
        // a non-ASCII name is FINE on the typed surface…
        let mut bld = NirBuilder::new(NirImportOptions::default());
        let inp = bld.add_input("entrée", &[1]).expect("input");
        let lif = bld
            .add_lif_population("lif", &lif_params_single())
            .expect("lif");
        bld.add_edge(inp, lif).expect("edge");
        let g = bld.build().expect("builds — no gate on the typed surface");
        // …and the LIF quantized fine
        let pop = g.nodes[1].lif.unwrap();
        assert_eq!(g.lifs[pop.offset].tau_us, 20_000);
        // …but the JSON container cannot write it
        let mut out = vec![0u8; 2048];
        assert_eq!(
            nir_export(&g.nodes, &g.edges, &g.weights, &g.lifs, g.opts, &mut out),
            Err(NirError::NonAsciiNodeName("entrée"))
        );
        // quotes and escapes are unwritable too (would corrupt the
        // container); the plain-ASCII builder chain exports fine
        let mut q = NirBuilder::new(NirImportOptions::default());
        let qi = q.add_input("a\"b", &[1]).expect("input");
        let qn = q
            .add_lif_population("lif", &lif_params_single())
            .expect("lif");
        q.add_edge(qi, qn).expect("edge");
        let gq = q.build().expect("builds");
        assert_eq!(
            nir_export(&gq.nodes, &gq.edges, &gq.weights, &gq.lifs, gq.opts, &mut out),
            Err(NirError::NonAsciiNodeName("a\"b"))
        );
        let ok = builder_chain();
        assert!(nir_export(&ok.nodes, &ok.edges, &ok.weights, &ok.lifs, ok.opts, &mut out).is_ok());
    }

    #[test]
    fn builder_parity_checks_and_rejections() {
        let mut b = NirBuilder::new(NirImportOptions::default());
        let i = b.add_input("a", &[1]).expect("input");
        let o = b.add_output("a", &[1]).expect("output");
        b.add_edge(i, o).expect("edge");
        assert!(matches!(b.build(), Err(NirError::DuplicateNodeName)));

        let mut b2 = NirBuilder::new(NirImportOptions::default());
        let i2 = b2.add_input("a", &[1]).expect("input");
        let o2 = b2.add_output("b", &[1]).expect("output");
        b2.add_edge(i2, o2).expect("edge");
        // shape > 4 dims and edge indices naming no node
        assert_eq!(
            b2.add_input("c", &[1, 2, 3, 4, 5]),
            Err(NirError::BadShape("shape"))
        );
        assert_eq!(b2.add_edge(0, 99), Err(NirError::BadShape("edges")));
        // LIF hard failures propagate from quantize_lif
        assert_eq!(
            b2.add_lif_population(
                "x",
                &NirLifParams {
                    tau_s: &[-0.02],
                    r_ohm: &[1e8],
                    v_leak_v: &[-0.07],
                    v_threshold_v: &[-0.055],
                    v_reset_v: Some(&[-0.08]),
                }
            ),
            Err(NirError::BadNumber("tau"))
        );
        // linear shape mismatch propagates from quantize_linear
        assert_eq!(
            b2.add_linear("y", &[1.0], 2, 3),
            Err(NirError::BadShape("weight"))
        );
        // duplicate edge fires at build (the reference's
        // validate_structure parity)
        b2.add_edge(i2, o2).expect("edge");
        assert!(matches!(b2.build(), Err(NirError::DuplicateEdge)));
    }

    #[test]
    fn failed_adds_leave_no_zombie_state() {
        let mut b = NirBuilder::new(NirImportOptions::default());
        let inp = b.add_input("input", &[3]).expect("input");
        // (a) failed add_linear: NaN weight — quantize_linear
        //     hard-fails AFTER the old push-first shape would have
        //     created the node and zero-extended the arena
        assert_eq!(
            b.add_linear("zombie", &[0.5, f64::NAN, 0.25], 1, 3),
            Err(NirError::BadNumber("weight"))
        );
        // (b) failed add_lif_population: neuron 2's tau ≤ 0 fails
        //     mid-population — neuron 1 quantized fine, the exact
        //     half-appended shape the old push-first code produced
        assert_eq!(
            b.add_lif_population(
                "zombie-lif",
                &NirLifParams {
                    tau_s: &[0.02, -0.02],
                    r_ohm: &[1e8, 1e8],
                    v_leak_v: &[-0.07, -0.07],
                    v_threshold_v: &[-0.055, -0.055],
                    v_reset_v: Some(&[-0.08, -0.08]),
                },
            ),
            Err(NirError::BadNumber("tau"))
        );
        // The builder is still usable and carries no zombie: the
        // identical clean chain (builder_chain's graph) builds to
        // exactly the same state as if the failures never happened.
        let lin = b
            .add_linear("linear", &[0.5, -1.0, 0.25, 0.0, 0.75, -0.5], 2, 3)
            .expect("linear");
        let lif = b
            .add_lif_population(
                "lif",
                &NirLifParams {
                    tau_s: &[0.02, 0.02],
                    r_ohm: &[1e8, 1e8],
                    v_leak_v: &[-0.07, -0.07],
                    v_threshold_v: &[-0.055, -0.055],
                    v_reset_v: Some(&[-0.08, -0.08]),
                },
            )
            .expect("lif");
        let out = b.add_output("output", &[2]).expect("output");
        for (a, c) in [(inp, lin), (lin, lif), (lif, out)] {
            b.add_edge(a, c).expect("edge");
        }
        let g = b.build().expect("builds");
        let clean = builder_chain();
        // no zombie nodes (count + no zombie names)
        assert_eq!(g.nodes.len(), clean.nodes.len());
        assert!(
            g.nodes
                .iter()
                .all(|n| n.name != "zombie" && n.name != "zombie-lif")
        );
        // no arena tail from the failed linear, no half-appended
        // lif records from the failed population
        assert_eq!(g.weights, clean.weights);
        assert_eq!(g.lifs, clean.lifs);
        assert_eq!(g.edges, clean.edges);
    }
}
