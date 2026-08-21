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
//! before their nodes in key order). During import the weight arena is
//! used as scratch (each f64 staged as two i16 slots), so it must hold
//! **2× the total weight count**; the quantized result occupies the
//! first half. Export writes into a caller byte slice. Strings are
//! borrowed from the input (schema strings must be escape-free
//! printable ASCII — loud error otherwise). `f64` appears in the
//! setup path only, never in any per-step hot path.

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
    /// imports it; only [`NirImport::build_chain_network`] rejects,
    /// plus LIF param arrays longer than 1 — per-neuron expansion is
    /// slice 2).
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
    /// last-wins (Python `json.loads` semantics).
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
    fn skip_value(&mut self) -> Result<(), NirError<'static>> {
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
                    self.skip_value()?;
                }
                Ok(())
            }
            Some(b'[') => {
                let mut first = true;
                while self.array_step(&mut first)? {
                    self.skip_value()?;
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
/// reference shapes are 1–4-D).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NirNode<'a> {
    pub name: &'a str,
    pub kind: NirNodeKind,
    pub shape: [u32; 4],
    pub shape_len: usize,
    pub lif: Option<NirLif>,
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
}

/// Caller-owned buffers for [`nir_import`]. The weight arena is used
/// as scratch during import (each f64 staged as four i16 slots), so
/// it must hold **4× [`NirScan::weight_cells`]**; the quantized
/// result occupies the first `weight_cells` slots.
#[derive(Debug)]
pub struct NirBuffers<'buf, 'a> {
    /// `bufs` borrow lifetime (`'buf`) and json-data lifetime (`'a`).
    pub nodes: &'buf mut [NirNode<'a>],
    pub edges: &'buf mut [(u32, u32)],
    pub weights: &'buf mut [i16],
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

/// `f64::round` (half away from zero) without std — `round`/`trunc`
/// are std-only. `as i64` truncates toward zero (saturating); every
/// caller pre-validates finiteness and bounds (`|x| ≤ u32::MAX` here,
/// exactly representable in both directions).
#[allow(clippy::cast_precision_loss)]
fn round_half_away(x: f64) -> f64 {
    let t = (x + f64::copysign(0.5, x)) as i64;
    t as f64
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

/// Quantize a LIF parameter set (single neuron). Every hard failure
/// of the contract fires here.
fn quant_lif(
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
    let mut saw_node = false;

    let mut first = true;
    while let Some(key) = r.object_step(&mut first)? {
        match key {
            "version" => version = Some(r.read_string()?),
            "node" => {
                saw_node = true;
                scan_graph(&mut r, &mut node_count, &mut edge_count, &mut weight_cells)?;
            }
            _ => r.skip_value()?,
        }
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
    })
}

fn scan_graph(
    r: &mut Reader<'_>,
    node_count: &mut usize,
    edge_count: &mut usize,
    weight_cells: &mut usize,
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
                    scan_node(r, weight_cells)?;
                    *node_count += 1;
                }
            }
            _ => r.skip_value()?,
        }
    }
    Ok(())
}

fn scan_node(r: &mut Reader<'_>, weight_cells: &mut usize) -> Result<(), NirError<'static>> {
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
            _ => r.skip_value()?,
        }
    }
    Ok(())
}

/// Recursively walk an array, counting leaves.
fn count_array(
    r: &mut Reader<'_>,
    depth: &mut usize,
    leaves: &mut usize,
) -> Result<(), NirError<'static>> {
    if *depth >= 3 {
        // 2-D max (a weight tensor); deeper = wrong shape
        return Err(NirError::BadShape("weight"));
    }
    let mut first = true;
    while r.array_step(&mut first)? {
        if r.peek() == Some(b'[') {
            *depth += 1;
            count_array(r, depth, leaves)?;
            *depth -= 1;
        } else {
            r.read_number()?;
            *leaves += 1;
        }
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
pub fn nir_import<'a>(
    json: &'a [u8],
    opts: NirImportOptions,
    bufs: &mut NirBuffers<'_, 'a>,
) -> Result<NirReport, NirError<'a>> {
    let mut report = NirReport::default();

    // walk 1: nodes (+ weights)
    let mut node_count = 0usize;
    let mut weight_fill = 0usize;
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
                                &mut report,
                            )?;
                        }
                    } else {
                        r.skip_value()?;
                    }
                }
            } else {
                r.skip_value()?;
            }
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
                        r.skip_value()?;
                    }
                }
            } else {
                r.skip_value()?;
            }
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

/// LIF tail of [`import_node`]: default `v_reset`, quantize, note.
fn finish_lif(
    tau: Option<f64>,
    res: Option<f64>,
    v_leak: Option<f64>,
    v_threshold: Option<f64>,
    v_reset: Option<f64>,
    opts: NirImportOptions,
    report: &mut NirReport,
) -> Result<NirLif, NirError<'static>> {
    let v_reset_val = v_reset.unwrap_or_else(|| {
        report.note(NirNote::VResetDefaulted);
        0.0
    });
    let lif = quant_lif(
        tau.ok_or(NirError::MissingField("tau"))?,
        res.ok_or(NirError::MissingField("r"))?,
        v_leak.ok_or(NirError::MissingField("v_leak"))?,
        v_threshold.ok_or(NirError::MissingField("v_threshold"))?,
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
    Ok(lif)
}

/// The reference emits LIF params as per-neuron arrays; slice 1
/// supports length-1 (one LIF node = one neuron). Length ≠ 1 is a
/// loud error naming what slice 2 brings.
fn read_len1_array(r: &mut Reader<'_>, field: &'static str) -> Result<f64, NirError<'static>> {
    let mut first = true;
    let mut count = 0;
    let mut val = 0.0f64;
    while r.array_step(&mut first)? {
        val = r.read_number()?;
        count += 1;
    }
    match count {
        1 => Ok(val),
        0 => Err(NirError::BadShape(field)),
        _ => Err(NirError::UnsupportedTopology(
            "LIF param arrays longer than 1 (per-neuron expansion is slice 2)",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn import_node<'a>(
    r: &mut Reader<'a>,
    name: &'a str,
    opts: NirImportOptions,
    bufs: &mut NirBuffers<'_, 'a>,
    node_count: &mut usize,
    weight_fill: &mut usize,
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
    let mut tau: Option<f64> = None;
    let mut res: Option<f64> = None;
    let mut v_leak: Option<f64> = None;
    let mut v_threshold: Option<f64> = None;
    let mut v_reset: Option<f64> = None;

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
            "tau" => tau = Some(read_len1_array(r, "tau")?),
            "r" => res = Some(read_len1_array(r, "r")?),
            "v_leak" => v_leak = Some(read_len1_array(r, "v_leak")?),
            "v_threshold" => v_threshold = Some(read_len1_array(r, "v_threshold")?),
            "v_reset" => v_reset = Some(read_len1_array(r, "v_reset")?),
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
            _ => r.skip_value()?, // metadata + unknown fields tolerated
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
            bufs.nodes[idx].lif = Some(finish_lif(
                tau, res, v_leak, v_threshold, v_reset, opts, report,
            )?);
            NirNodeKind::Lif
        }
        other => return Err(NirError::UnsupportedNodeKind(other)),
    };
    *node_count += 1;
    Ok(())
}

/// Read a 2-D weight array, quantizing into the arena at `offset`
/// (staging f64 bits in the arena's scratch half first). Row-major
/// `weight[out][in]`: `y_o = Σ_i w[o][i] · x_i`.
fn import_weight(
    r: &mut Reader<'_>,
    bufs: &mut NirBuffers<'_, '_>,
    offset: usize,
) -> Result<NirLinear, NirError<'static>> {
    let mut rows = 0usize;
    let mut cols: Option<usize> = None;
    let mut staged = 0usize; // f64 slots written (2 i16 each)

    let mut rfirst = true;
    while r.array_step(&mut rfirst)? {
        if r.peek() != Some(b'[') {
            return Err(NirError::BadShape("weight"));
        }
        let row_start = r.pos;
        // count + validate the row
        let mut cfirst = true;
        let mut rc = 0usize;
        while r.array_step(&mut cfirst)? {
            if r.peek() == Some(b'[') {
                return Err(NirError::BadShape("weight"));
            }
            r.read_number()?;
            rc += 1;
        }
        if rc == 0 {
            return Err(NirError::BadShape("weight"));
        }
        match cols {
            None => cols = Some(rc),
            Some(c) if c != rc => return Err(NirError::BadShape("weight")),
            Some(_) => {}
        }
        rows += 1;
        // re-parse the row from its start, staging the f64 bits
        let mut sub = Reader { b: r.b, pos: row_start };
        let mut sfirst = true;
        while sub.array_step(&mut sfirst)? {
            let v = sub.read_number()?;
            if offset + staged + 4 > bufs.weights.len() {
                return Err(NirError::BufferOverflow);
            }
            let bits = v.to_bits();
            for k in 0..4 {
                bufs.weights[offset + staged + k] =
                    ((bits >> (16 * k)) & 0xffff) as u16 as i16;
            }
            staged += 4;
        }
    }
    let cols = cols.ok_or(NirError::BadShape("weight"))?;
    let n = rows * cols;

    // quantize from the staged bits
    // reassemble the staged 64-bit pattern (4 x u16 lanes, little-endian)
    let staged_val = |bufs: &NirBuffers<'_, '_>, k: usize| -> f64 {
        let mut bits: u64 = 0;
        for lane in 0..4 {
            bits |= u64::from(bufs.weights[offset + 4 * k + lane] as u16) << (16 * lane);
        }
        f64::from_bits(bits)
    };
    let mut absmax = 0.0f64;
    for k in 0..n {
        let v = staged_val(bufs, k);
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
    let mut max_abs_err = 0.0f64;
    for k in 0..n {
        let v = staged_val(bufs, k);
        let q = round_half_away(v / scale).clamp(-I16_FS, I16_FS) as i16;
        bufs.weights[offset + k] = q;
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

/// Export an imported graph: the SAME dict schema, canonical bytes —
/// fixed key order, derived (substrate-exact) values in the schema
/// fields, provenance + quant records in `metadata.neuralos`.
/// Returns the byte length written. Importing the export must
/// reproduce the import bit-for-bit (the idempotence gate).
///
/// # Errors
///
/// [`NirError::ExportTooSmall`] when `out` cannot hold the document.
pub fn nir_export(
    nodes: &[NirNode<'_>],
    edges: &[(u32, u32)],
    weights: &[i16],
    opts: NirImportOptions,
    out: &mut [u8],
) -> Result<usize, NirError<'static>> {
    let mut w = ByteWriter { out, len: 0 };
    w.push_str("{\"version\":\"")?;
    w.push_str(EXPORT_VERSION)?;
    w.push_str("\",\"node\":{\"type\":\"NIRGraph\",\"edges\":[")?;
    for (i, (a, b)) in edges.iter().enumerate() {
        if i > 0 {
            w.push(b',')?;
        }
        let an = nodes.get(*a as usize).map_or("?", |n| n.name);
        let bn = nodes.get(*b as usize).map_or("?", |n| n.name);
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
        export_node(&mut w, node, weights, opts)?;
    }
    w.push_str("}}}")?;
    Ok(w.len)
}

fn export_node(
    w: &mut ByteWriter<'_>,
    node: &NirNode<'_>,
    weights: &[i16],
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
            let lif = node.lif.ok_or(NirError::MissingField("lif"))?;
            // derived, rendered back in source units EXACTLY as the
            // substrate holds them (tau_us/1e6, r_mohm*1e6, q/1000):
            // re-import reproduces the identical quantized node.
            let tau_s = f64::from(lif.tau_us) / 1.0e6;
            let r_ohm = f64::from(lif.resistance_mohm) * 1.0e6;
            w.push_str("{\"type\":\"LIF\",\"tau\":[")?;
            write_f64(w, tau_s)?;
            w.push_str("],\"r\":[")?;
            write_f64(w, r_ohm)?;
            w.push_str("],\"v_leak\":[")?;
            write_f64(w, to_v(lif.leak_q))?;
            w.push_str("],\"v_threshold\":[")?;
            write_f64(w, to_v(lif.threshold_q))?;
            w.push_str("],\"v_reset\":[")?;
            write_f64(w, to_v(lif.reset_q))?;
            w.push_str("],\"metadata\":{\"neuralos\":{\"provenance\":{\
\"tau_s\":")?;
            write_f64(w, lif.tau_s)?;
            w.push_str(",\"r_ohm\":")?;
            write_f64(w, lif.r_ohm)?;
            w.push_str(",\"v_leak_v\":")?;
            write_f64(w, lif.v_leak_v)?;
            w.push_str(",\"v_threshold_v\":")?;
            write_f64(w, lif.v_threshold_v)?;
            w.push_str(",\"v_reset_v\":")?;
            write_f64(w, lif.v_reset_v)?;
            w.push_str(",\"v_reset_defaulted\":")?;
            w.push_str(if lif.v_reset_defaulted { "true" } else { "false" })?;
            w.push_str("},\"quant\":{\"grid\":\"")?;
            w.push_str(match opts.resolution {
                VoltageResolution::Millivolt => "mV",
                VoltageResolution::CentiMillivolt => "cV",
            })?;
            w.push_str("\",\"dt_us\":")?;
            write_u32(w, opts.dt_us)?;
            w.push_str(",\"tau_us\":")?;
            write_u32(w, lif.tau_us)?;
            w.push_str(",\"tau_err_s\":")?;
            write_f64(w, lif.tau_err_s)?;
            w.push_str(",\"max_v_err_v\":")?;
            write_f64(w, lif.max_v_err_v)?;
            w.push_str("}}}}")?; // quant, neuralos, metadata, node
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

    use super::{NirError, NirImportOptions, NirLinear, NirNode, NirNodeKind, NIR_REF_SHA};
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
            // scratch contract: 2× weights during import
            let mut weights = vec![0i16; scan.weight_cells.saturating_mul(4)];
            {
                let mut bufs = super::NirBuffers {
                    nodes: &mut nodes,
                    edges: &mut edges,
                    weights: &mut weights,
                };
                super::nir_import(json, opts, &mut bufs)?;
            }
            weights.truncate(scan.weight_cells);
            Ok(NirImport {
                nodes,
                edges,
                weights,
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
            let lif_params = self.nodes[lif].lif.ok_or(NirError::MissingField("lif"))?;
            let linear_rec = self.nodes[lin].linear.ok_or(NirError::MissingField("linear"))?;
            if linear_rec.rows > u16::MAX as usize {
                return Err(NirError::BufferOverflow);
            }
            if self.nodes[inp].shape.first().copied().unwrap_or(0) != linear_rec.cols as u32 {
                return Err(NirError::UnsupportedTopology("Input shape != Linear cols"));
            }

            // one neuron per Linear row, LIF params honored
            let mut neurons = Vec::with_capacity(linear_rec.rows);
            for id in 0..linear_rec.rows {
                let mut n = LIFNeuron::new_with_type_resolution(
                    id as u16,
                    NeuronType::Excitatory,
                    self.opts.resolution,
                );
                n.resting_potential = lif_params.leak_q;
                n.membrane_potential = lif_params.leak_q;
                n.threshold = lif_params.threshold_q;
                n.reset_potential = lif_params.reset_q;
                n.tau_membrane_us = lif_params.tau_us;
                n.tau_refractory_us = 1_000; // NIR LIF has no refractory → minimum
                n.resistance_mohm = lif_params.resistance_mohm;
                n.capacitance_pf = lif_params.capacitance_pf;
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
        #[must_use]
        pub fn encode(&self, x: &[i16]) -> Vec<i16> {
            let mut out = vec![0i16; self.lin.rows];
            for (r, o) in out.iter_mut().enumerate() {
                let mut acc: i32 = 0;
                for c in 0..self.lin.cols {
                    let w = i32::from(
                        self.weights[self.lin.weight_offset + r * self.lin.cols + c],
                    );
                    acc += w * i32::from(x.get(c).copied().unwrap_or(0));
                }
                acc /= 100;
                *o = acc.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
            }
            out
        }
    }
}

#[cfg(feature = "std")]
pub use std_assembly::{ChainEncoder, NirImport};

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    const CHAIN: &str = "{\"version\":\"test\",\"node\":{\"type\":\"NIRGraph\",\
\"edges\":[[\"input\",\"linear\"],[\"linear\",\"lif\"],[\"lif\",\"output\"]],\
\"nodes\":{\"input\":{\"type\":\"Input\",\"shape\":[3]},\
\"linear\":{\"type\":\"Linear\",\"weight\":[[0.5,-1.0,0.25],[0.0,0.75,-0.5]]},\
\"lif\":{\"type\":\"LIF\",\"tau\":[0.02],\"r\":[100000000.0],\
\"v_leak\":[-0.07],\"v_threshold\":[-0.055],\"v_reset\":[-0.08]},\
\"output\":{\"type\":\"Output\",\"shape\":[2]}}}}";

    #[test]
    fn scan_counts_the_chain() {
        let s = nir_scan(CHAIN.as_bytes()).expect("chain scans");
        assert_eq!(s.version, "test");
        assert_eq!(s.node_count, 4);
        assert_eq!(s.edge_count, 3);
        assert_eq!(s.weight_cells, 6);
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
    fn import_chain(opts: NirImportOptions) -> (Vec<NirNode<'static>>, Vec<(u32, u32)>, Vec<i16>, NirReport) {
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
        let mut weights = vec![0i16; scan.weight_cells * 4];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
        };
        let report = nir_import(CHAIN.as_bytes(), opts, &mut bufs).expect("chain imports");
        weights.truncate(scan.weight_cells);
        (nodes, edges, weights, report)
    }

    #[test]
    fn import_quantizes_the_chain() {
        let (nodes, edges, weights, report) =
            import_chain(NirImportOptions::default());
        assert_eq!(
            edges,
            vec![(0, 1), (1, 2), (2, 3)],
            "edges resolve by name to document order"
        );
        assert_eq!((report.inputs, report.linears, report.lifs, report.outputs), (1, 1, 1, 1));
        assert_eq!(report.weight_cells, 6);

        let lif = nodes[2].lif.expect("lif present");
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
        let mut weights = vec![0i16; scan.weight_cells * 4];
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
        };
        assert_eq!(
            nir_import(CHAIN.as_bytes(), opts, &mut bufs),
            Err(NirError::TauBelowDt)
        );
        // direct quant_lif contract:
        assert!(matches!(
            quant_lif(0.0, 1e8, -0.07, -0.055, -0.08, false, NirImportOptions::default()),
            Err(NirError::BadNumber("tau"))
        ));
        assert!(matches!(
            quant_lif(-0.02, 1e8, -0.07, -0.055, -0.08, false, NirImportOptions::default()),
            Err(NirError::BadNumber("tau"))
        ));
        assert!(matches!(
            quant_lif(0.02, 0.0, -0.07, -0.055, -0.08, false, NirImportOptions::default()),
            Err(NirError::BadNumber("r"))
        ));
        assert!(matches!(
            quant_lif(0.02, 1e8, -0.07, -0.0004, -0.08, false, NirImportOptions::default()),
            Err(NirError::ThresholdZero)
        ));
        assert!(matches!(
            quant_lif(0.02, 1e8, -0.07, -0.055, -0.08, false, NirImportOptions::new(30_000, VoltageResolution::Millivolt)),
            Err(NirError::TauBelowDt)
        ));
        assert!(matches!(
            quant_lif(0.02, 1e8, -0.07, 0.06, -0.08, false, NirImportOptions::default()),
            Err(NirError::PotentialOutOfRange("v_threshold"))
        ));
        // threshold on the centi grid: -0.0555 V is representable
        let lif = quant_lif(
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
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
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
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
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
            let mut bufs = NirBuffers {
                nodes: &mut nodes,
                edges: &mut edges,
                weights: &mut weights,
            };
            let err = nir_import(doc.as_bytes(), NirImportOptions::default(), &mut bufs)
                .expect_err(label);
            assert!(check(&err), "{label}: got {err:?}");
        }
    }

    #[test]
    fn buffer_overflow_is_loud() {
        let mut nodes = [NirNode {
            name: "",
            kind: NirNodeKind::Input,
            shape: [0; 4],
            shape_len: 0,
            lif: None,
            linear: None,
        }];
        let mut edges = [];
        let mut weights = [0i16; 2]; // chain needs 24 (scratch)
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
        };
        assert_eq!(
            nir_import(CHAIN.as_bytes(), NirImportOptions::default(), &mut bufs),
            Err(NirError::BufferOverflow)
        );
    }

    #[test]
    fn export_then_import_is_idempotent() {
        let opts = NirImportOptions::default();
        let (nodes, edges, weights, _report) = import_chain(opts);

        let mut out = [0u8; 2048];
        let n = nir_export(&nodes, &edges, &weights, opts, &mut out).expect("exports");
        let exported = core::str::from_utf8(&out[..n]).expect("utf8");

        // byte-stable: second export identical
        let mut out2 = [0u8; 2048];
        let n2 = nir_export(&nodes, &edges, &weights, opts, &mut out2).unwrap();
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
        let mut weights2 = vec![0i16; scan2.weight_cells * 4];
        let mut bufs2 = NirBuffers {
            nodes: &mut nodes2,
            edges: &mut edges2,
            weights: &mut weights2,
        };
        let report2 = nir_import(exported.as_bytes(), opts, &mut bufs2).expect("re-imports");
        weights2.truncate(scan2.weight_cells);
        assert_eq!(edges2, edges);
        assert_eq!(weights2, weights, "quantized weights identical");
        // The CONTRACT is quantized-state identity. The re-import's
        // notes MAY grow: the dequantized max (|q|·scale in f64) can
        // re-derive a scale 1 ulp off, which the lossy-tensor note
        // honestly records while the state stays identical.
        let _ = report2;
        let lif1 = nodes[2].lif.unwrap();
        let lif2 = nodes2[2].lif.unwrap();
        assert_eq!((lif2.tau_us, lif2.threshold_q, lif2.leak_q, lif2.reset_q, lif2.resistance_mohm),
                   (lif1.tau_us, lif1.threshold_q, lif1.leak_q, lif1.reset_q, lif1.resistance_mohm));
        let linear_out = nodes2[1].linear.unwrap();
        assert_eq!(linear_out.scale, nodes[1].linear.unwrap().scale, "scale recovered exactly");
        // provenance rode metadata: the ORIGINAL source floats survive
        assert_eq!(lif2.tau_s, lif1.tau_s);
    }

    #[test]
    fn exported_json_is_valid_shape() {
        let opts = NirImportOptions::default();
        let (nodes, edges, weights, _) = import_chain(opts);
        let mut out = [0u8; 2048];
        let n = nir_export(&nodes, &edges, &weights, opts, &mut out).unwrap();
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
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
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
        let mut bufs = NirBuffers {
            nodes: &mut nodes,
            edges: &mut edges,
            weights: &mut weights,
        };
        let rep = nir_import(doc.as_bytes(), NirImportOptions::default(), &mut bufs).unwrap();
        assert!(rep.notes[NirNote::VResetDefaulted as usize] >= 1);
        let lif = nodes[0].lif.unwrap();
        assert_eq!(lif.reset_q, 0);
        assert!(lif.v_reset_defaulted);
    }
}
