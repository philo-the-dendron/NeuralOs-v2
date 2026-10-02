//! neuralos-nir2json — the inbound bridge: a stranger's NIR `.nir`
//! (HDF5) file converted into the JSON schema `neuralos-snn`'s
//! [`nir_import`] consumes.
//!
//! # Design (single-writer by construction)
//!
//! The tool never writes JSON itself: HDF5 is read via `hdf5-pure`
//! (pure Rust, no C toolchain: `cargo tree -e normal` lists it), the typed
//! values feed snn's own `NirBuilder` (the structured-entry seam —
//! THE quantizer), and snn's `nir_export` renders the canonical
//! bytes. Schema and quantization live in one Rust source; the
//! dual-implementation drift class dies by construction.
//!
//! # Layout (pinned from the reference's own emission; the same map
//! rt's `nir_hdf5.rs` carries)
//!
//! ```text
//! version                      scalar vlen-string dataset
//! node/                        group, type "NIRGraph"
//!     type                     scalar vlen-string
//!     nodes/<name>/            one group per node
//!         type                 scalar vlen-string (Input|LIF|Linear|Output|…)
//!         <arrays>             gzip (deflate) datasets: f64 params,
//!                              i64 shapes, 2-D f64 weights
//!     edges/                   (N,2) vlen-string dataset, UNCOMPRESSED
//!     metadata/                group, only when non-empty (ignored here)
//! ```
//!
//! # Filter census (pre-read, per dataset — stated policy)
//!
//! The reference's emissions carry exactly no filter (strings, edges)
//! or deflate (arrays). This reader accepts those two and rejects
//! EVERYTHING else loudly BY NAME (`lzf`, `szip`, …) — a filter we
//! cannot decode is a silent-corruption hazard, not an inconvenience.
//!
//! # f32 handling
//!
//! Stranger files (snnTorch exports default to fp32) carrying F32
//! datasets are widened to f64 bit-exactly and the fact is STAMPED in
//! a sidecar (`<out>.meta.json` — a file-level audit annotation; snn
//! never sees it, and any re-export drops it naturally: no durability
//! illusion). rt cannot serve as an f32 oracle — it hard-rejects
//! non-f64 — so the tool's test oracle compares the f32-twin output
//! against the f64 origin at f32-precision tolerance.
//!
//! # Out-of-subset honesty
//!
//! The supported node kinds are exactly Input, LIF, Linear, Output, and
//! under `--sim-units` Affine, rewritten as a Linear and, for a bias not
//! all zero, an input of its own ([`convert_file_opts`]). Anything else
//! is refused loudly with the node's name and kind — a recorded result,
//! never a partial conversion.
//!
//! # Freeze
//!
//! [`freeze`] builds the converted graph with the library and writes the
//! arrays a `FixedNetwork` steps, one module, with the trace of its drive
//! on the host (README § Freeze; `--freeze` on the CLI).
//!
//! [`nir_import`]: neuralos_snn::nir::nir_import

use std::fmt;
use std::path::Path;

use hdf5_pure::{DType, Dataset, File, Group, VlenStringReadOptions};
use neuralos_snn::FixedSynapse;
use neuralos_snn::fixed::freeze as freezer;
use neuralos_snn::nir::{
    NirBuilder, NirError, NirGraphEncoder, NirImport, NirImportOptions, NirLifParams, NirNodeKind,
    SIM_CURRENT_QUANTA, Thousandths, nir_export,
};
use neuralos_snn::trace::{self, Kind, Rows, row};

/// Tool version (sidecar stamp).
pub const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// 1.0, a bias input's value from its start and each feature's in a run
/// of the drive that gives none.
const ONE: Thousandths = Thousandths(1_000);

/// The time step under `--sim-units` when `--dt` is not given, µs:
/// snnTorch's exporter writes `tau = dt / (1 - beta)` with `dt = 1e-4`
/// fixed, and a `.nir` file carries no step. The CLI's default, and the
/// step of this crate's tests of simulation units.
pub const SIM_DT_US: u32 = 100;

/// The deflate filter id (H5Z_FILTER_DEFLATE) — the only compression
/// the census admits.
const FILTER_DEFLATE: u16 = 1;

/// Everything that can stop a conversion, each nameable in one line.
#[derive(Debug)]
pub enum ConvertError {
    /// The file could not be opened or parsed as HDF5.
    Open(String),
    /// The NIR layout contract is violated (missing/`broken group).
    Layout(String),
    /// Filter-census rejection: the named dataset carries a filter we
    /// refuse to decode.
    Filter {
        dataset: String,
        id: u16,
        name: String,
    },
    /// Out-of-subset node kind — named, never partial. `Affine` is one
    /// in native units only.
    UnsupportedNode { node: String, kind: String },
    /// Under `--sim-units`, an `Affine` whose bias has no place: it
    /// feeds a node that is not a LIF, a LIF that no Input reaches or
    /// that the Inputs reach at two depths, or LIFs at two depths
    /// ([`convert_file_opts`]).
    Bias { node: String, why: String },
    /// Simulation-unit LIF parameters detected without the transform
    /// flag: the DOUBLE wall, named together (r fires first in
    /// quantize_lif's check order, but the dimensionless voltages
    /// also refuse natively — e.g. v_th 0.1 read as 0.1 V is 100 mV,
    /// beyond the +50 mV membrane).
    SimUnits { node: String, r_ohm: f64 },
    /// A dataset's dtype/shape is not what its node kind requires.
    BadData { dataset: String, what: String },
    /// snn's builder/exporter refused the graph (quantization bounds,
    /// edges, duplicates, non-ASCII names) — mapped with the failing
    /// stage named and the error rendered (NirError borrows node
    /// names; the string severs the lifetime at this boundary).
    Snn { stage: &'static str, msg: String },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(m) => write!(f, "cannot open/parse HDF5: {m}"),
            Self::Layout(m) => write!(f, "NIR layout violation: {m}"),
            Self::Filter { dataset, id, name } => write!(
                f,
                "filter census REFUSES dataset '{dataset}': filter {name} (id {id}) — \
                 policy accepts only none or deflate (gzip); re-emit uncompressed or gzip"
            ),
            Self::UnsupportedNode { node, kind } if kind == "Affine" => write!(
                f,
                "node '{node}' is kind 'Affine' — in native units outside the supported subset \
                 (Input, LIF, Linear, Output); `--sim-units` converts it, its bias an input of its own"
            ),
            Self::UnsupportedNode { node, kind } => write!(
                f,
                "node '{node}' is kind '{kind}' — outside the supported subset \
                 (Input, LIF, Linear, Output); refusing loudly rather than partially converting"
            ),
            Self::Bias { node, why } => write!(
                f,
                "node '{node}' is an Affine whose bias has no place: {why}"
            ),
            Self::SimUnits { node, r_ohm } => write!(
                f,
                "node '{node}' carries SIMULATION-UNIT LIF parameters (r = {r_ohm} Ω < 1 MΩ; \
                 and the dimensionless voltages would also exceed the substrate's \
                 [−100, +50] mV membrane read natively as volts) — the substrate quantizes \
                 biological scale (MΩ, mV). Re-run with `--sim-units`: the stamped \
                 convention transform (a voltage scale per node, true-scale weights, centi grid)"
            ),
            Self::BadData { dataset, what } => write!(f, "dataset '{dataset}': {what}"),
            Self::Snn { stage, msg } => write!(f, "snn {stage}: {msg}"),
        }
    }
}

/// The file-level audit annotation (sidecar `<out>.meta.json`).
#[derive(Debug, Clone)]
pub struct Stamp {
    /// The file's own `version` string (recorded, never parsed for
    /// behavior — emitters ship many NIR-lib versions).
    pub nir_version: String,
    /// (node name, kind) in document order.
    pub node_census: Vec<(String, String)>,
    /// Datasets widened f32→f64 (bit-exact); empty on f64 files.
    pub f32_datasets: Vec<String>,
    /// The import options used (snn defaults unless overridden).
    pub dt_us: u32,
    /// The voltage grid name (EFFECTIVE — centi when sim-units).
    pub resolution: &'static str,
    /// The sim-unit convention transform was applied (stamped — an
    /// interpretive act, never silent).
    pub sim_units: bool,
    /// Under `--sim-units`, each transformed LIF node's voltage scale,
    /// mV per simulation unit, in document order (stamped, as the
    /// transform is).
    pub volt_scale: Vec<(String, f64)>,
    /// Under `--sim-units`, each bias input the conversion added, in the
    /// document order of their Affines.
    pub bias: Vec<Bias>,
}

/// A bias input `--sim-units` adds for an `Affine` whose bias is not all
/// zero ([`convert_file_opts`]): an `Input` of one feature, named the
/// Affine's name and `/bias`, into a one-column Linear, named the
/// Affine's name and `/b`, which holds the bias and feeds each LIF the
/// Affine feeds. A `.nir` file names its nodes by HDF5 link names, which
/// never hold a `/`, so neither name meets one of the file's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bias {
    /// The `Input` node's name, the Affine's and `/bias`.
    pub input: String,
    /// The first step its value is 1, before it 0: the depth of the LIFs
    /// its Affine feeds, the count of spike edges between them and an
    /// Input, since a spike reaches the next population one step late.
    pub start: u32,
}

/// A converted graph's Inputs in the order `NirGraphEncoder::encode`
/// reads them, the graph's node order: each the graph's own, or a
/// [`Bias`] input with its start. [`freeze`] drives a graph through it,
/// and so does this crate's parity test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inputs {
    /// Per Input, its features and, for a bias input, its start.
    slots: Vec<(usize, Option<u32>)>,
}

impl Inputs {
    /// The Inputs `enc` reads, `graph`'s in node order, with each of
    /// `bias` among them.
    ///
    /// # Errors
    ///
    /// The name of a bias input that is not an Input of one feature in
    /// `graph`.
    pub fn new(
        graph: &NirImport<'_>,
        enc: &NirGraphEncoder,
        bias: &[Bias],
    ) -> Result<Self, String> {
        // the encoder numbers the Inputs in node order, as here
        let names: Vec<&str> = graph
            .nodes
            .iter()
            .filter(|n| n.kind == NirNodeKind::Input)
            .map(|n| n.name)
            .collect();
        let slots: Vec<(usize, Option<u32>)> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let start = bias.iter().find(|b| b.input == *name).map(|b| b.start);
                (enc.input_features(i), start)
            })
            .collect();
        for b in bias {
            match names.iter().position(|n| *n == b.input) {
                Some(i) if slots[i].0 == 1 => {}
                _ => return Err(b.input.clone()),
            }
        }
        Ok(Self { slots })
    }

    /// The features of the graph's own Inputs: what a drive gives.
    #[must_use]
    pub fn features(&self) -> usize {
        self.slots
            .iter()
            .filter(|(_, start)| start.is_none())
            .map(|(features, _)| features)
            .sum()
    }

    /// The steps a bias input starts on, ascending, each once.
    #[must_use]
    pub fn starts(&self) -> Vec<u32> {
        let mut starts: Vec<u32> = self.slots.iter().filter_map(|&(_, s)| s).collect();
        starts.sort_unstable();
        starts.dedup();
        starts
    }

    /// Every Input's values at `step`, in the encoder's order: the graph's
    /// own from `own`, one value per feature in Input order, and each
    /// bias input's 0 before its start and 1 from it.
    ///
    /// # Panics
    ///
    /// When `own` holds fewer values than [`Self::features`].
    #[must_use]
    pub fn at(&self, step: u32, own: &[Thousandths]) -> Vec<Vec<Thousandths>> {
        let mut rest = own;
        self.slots
            .iter()
            .map(|&(features, start)| match start {
                Some(s) => vec![if step >= s { ONE } else { Thousandths(0) }; features],
                None => {
                    let (this, others) = rest.split_at(features);
                    rest = others;
                    this.to_vec()
                }
            })
            .collect()
    }
}

/// A completed conversion: canonical JSON + its stamp.
#[derive(Debug)]
pub struct Converted {
    pub json: Vec<u8>,
    pub stamp: Stamp,
}

fn filter_name(id: u16) -> String {
    match id {
        32004 | 32000 => "lzf".into(),
        4 => "szip".into(),
        2 => "shuffle".into(),
        3 => "fletcher32".into(),
        5 => "nbit".into(),
        6 => "scaleoffset".into(),
        other => format!("unknown-filter-{other}"),
    }
}

/// One dataset's census entry: path (for error naming) + filters.
fn census_dataset(g: &Group, base: &str, name: &str, out: &mut Vec<(String, Vec<u16>)>) {
    if let Ok(ds) = g.dataset(name) {
        out.push((format!("{base}/{name}"), ds.filters()));
    }
}

/// The pre-read filter census over every dataset the decode pass
/// would touch. Policy: each dataset carries NO filter or deflate
/// only; anything else is rejected by name.
fn census(node: &Group) -> Result<(), ConvertError> {
    let mut entries: Vec<(String, Vec<u16>)> = Vec::new();
    census_dataset(node, "node", "type", &mut entries);
    census_dataset(node, "node", "edges", &mut entries);
    if let Ok(nodes) = node.group("nodes") {
        let names = nodes.groups().map_err(bad_groups("node/nodes".into()))?;
        for name in names {
            let g = nodes
                .group(&name)
                .map_err(bad_groups(format!("node/nodes/{name}")))?;
            for d in g
                .datasets()
                .map_err(bad_groups(format!("node/nodes/{name}")))?
            {
                census_dataset(&g, &format!("node/nodes/{name}"), &d, &mut entries);
            }
        }
    }
    for (dataset, filters) in entries {
        let bad = filters.iter().copied().find(|&id| id != FILTER_DEFLATE);
        if let Some(id) = bad {
            return Err(ConvertError::Filter {
                dataset,
                id,
                name: filter_name(id),
            });
        }
    }
    Ok(())
}

fn bad_groups(what: String) -> impl Fn(hdf5_pure::Error) -> ConvertError {
    move |e| ConvertError::Layout(format!("cannot walk {what}: {e}"))
}

fn read_str(ds: &Dataset) -> Result<String, ConvertError> {
    let v = ds
        .read_vlen_strings(VlenStringReadOptions::default())
        .map_err(|e| ConvertError::Open(format!("vlen string read: {e}")))?;
    v.into_iter()
        .next()
        .ok_or_else(|| ConvertError::Layout("empty string dataset".into()))
}

/// Read one numeric 1-D param dataset, widening F32→f64 bit-exactly
/// (stamped). F64 passes through; anything else is refused loudly.
fn read_param_f64(
    ds: &Dataset,
    path: &str,
    stamps: &mut Vec<String>,
) -> Result<Vec<f64>, ConvertError> {
    match ds
        .dtype()
        .map_err(|e| ConvertError::Open(format!("dtype: {e}")))?
    {
        DType::F64 => ds
            .read_f64()
            .map_err(|e| ConvertError::Open(format!("f64 read: {e}"))),
        DType::F32 => {
            let v = ds
                .read_f32()
                .map_err(|e| ConvertError::Open(format!("f32 read: {e}")))?;
            stamps.push(path.to_string());
            Ok(v.into_iter().map(f64::from).collect())
        }
        other => Err(ConvertError::BadData {
            dataset: path.into(),
            what: format!("param dtype {other:?} — expected float64 (or float32, widened)"),
        }),
    }
}

fn shape_u32(ds: &Dataset, path: &str) -> Result<Vec<u32>, ConvertError> {
    match ds
        .dtype()
        .map_err(|e| ConvertError::Open(format!("dtype: {e}")))?
    {
        DType::I64 => ds
            .read_i64()
            .map_err(|e| ConvertError::Open(format!("i64 read: {e}")))?,
        other => {
            return Err(ConvertError::BadData {
                dataset: path.into(),
                what: format!("shape dtype {other:?} — expected int64"),
            });
        }
    }
    .into_iter()
    .map(|d: i64| {
        u32::try_from(d).map_err(|_| ConvertError::BadData {
            dataset: path.into(),
            what: format!("shape dim {d} outside u32"),
        })
    })
    .collect()
}

/// A Linear's or an Affine's `weight`: its values, rows and columns.
fn read_weight(
    g: &Group,
    name: &str,
    stamps: &mut Vec<String>,
) -> Result<(Vec<f64>, usize, usize), ConvertError> {
    let path = format!("node/nodes/{name}/weight");
    let ds = g
        .dataset("weight")
        .map_err(|e| ConvertError::Layout(format!("weight: {e}")))?;
    let shape = ds
        .shape()
        .map_err(|e| ConvertError::Open(format!("shape: {e}")))?;
    if shape.len() != 2 {
        return Err(ConvertError::BadData {
            dataset: path,
            what: format!("{}-D weight — expected 2-D", shape.len()),
        });
    }
    let vals = read_param_f64(&ds, &path, stamps)?;
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    if vals.len() != rows * cols {
        return Err(ConvertError::BadData {
            dataset: path,
            what: format!("{} values != {rows}×{cols}", vals.len()),
        });
    }
    Ok((vals, rows, cols))
}

/// Under `--sim-units`, the step an `Affine`'s bias starts on: the depth
/// of the LIFs it feeds, `None` when it feeds nothing. Refused by name
/// ([`ConvertError::Bias`]) when it feeds anything but a LIF, a LIF with
/// no one depth, or LIFs at two depths, since its one bias input starts
/// on one step. `kinds` and `edges` are the file's, by slot.
fn bias_start(
    names: &[String],
    kinds: &[&str],
    edges: &[(usize, usize)],
    affine: usize,
) -> Result<Option<u32>, ConvertError> {
    // a LIF it feeds and that LIF's depth, which every other must share
    let mut start: Option<(usize, u32)> = None;
    for &(_, t) in edges.iter().filter(|&&(a, _)| a == affine) {
        let why = match kinds[t] {
            "LIF" => match (depth(kinds, edges, t), start) {
                (Ok(d), Some((lif, s))) if d != s => format!(
                    "the LIFs '{}' and '{}' it feeds sit at depths {s} and {d}, and a bias starts at one",
                    names[lif], names[t]
                ),
                (Ok(d), _) => {
                    start = Some((t, d));
                    continue;
                }
                (Err(NoDepth::Unreached), _) => format!(
                    "no Input reaches the LIF '{}' it feeds, and a bias starts at its depth",
                    names[t]
                ),
                (Err(NoDepth::Two), _) => format!(
                    "the Inputs reach the LIF '{}' it feeds at two depths, and a bias starts at one",
                    names[t]
                ),
            },
            "Output" => format!(
                "it feeds the Output '{}', and a bias drives a LIF",
                names[t]
            ),
            kind => format!(
                "it feeds the {kind} '{}', and a bias drives a LIF directly",
                names[t]
            ),
        };
        return Err(ConvertError::Bias {
            node: names[affine].clone(),
            why,
        });
    }
    Ok(start.map(|(_, d)| d))
}

/// Why a LIF has no depth.
#[derive(Debug, PartialEq, Eq)]
enum NoDepth {
    /// No path from an Input reaches it.
    Unreached,
    /// Two paths from the Inputs cross a different count of LIFs, a loop
    /// among them.
    Two,
}

/// A LIF's depth: the spike edges between it and an Input, one count on
/// every path from the Inputs, since each LIF on the way delays a spike
/// by one step. `kinds` and `edges` by slot.
fn depth(kinds: &[&str], edges: &[(usize, usize)], lif: usize) -> Result<u32, NoDepth> {
    let n = kinds.len();
    // the nodes with a path to the LIF, the LIF among them
    let mut feeds = vec![false; n];
    feeds[lif] = true;
    let mut work = vec![lif];
    while let Some(v) = work.pop() {
        for &(a, b) in edges {
            if b == v && !feeds[a] {
                feeds[a] = true;
                work.push(a);
            }
        }
    }
    // their depths from the Inputs among them: an edge adds 1 after a
    // LIF and 0 after any other node, and a node two edges reach must be
    // reached at one depth
    let mut at: Vec<Option<u32>> = vec![None; n];
    let mut work: Vec<(usize, u32)> = Vec::new();
    for i in (0..n).filter(|&i| feeds[i] && kinds[i] == "Input") {
        at[i] = Some(0);
        work.push((i, 0));
    }
    while let Some((v, at_v)) = work.pop() {
        let d = at_v + u32::from(kinds[v] == "LIF");
        for &(a, b) in edges {
            if a == v && feeds[b] {
                match at[b] {
                    None => {
                        at[b] = Some(d);
                        work.push((b, d));
                    }
                    Some(e) if e != d => return Err(NoDepth::Two),
                    Some(_) => {}
                }
            }
        }
    }
    at[lif].ok_or(NoDepth::Unreached)
}

/// Convert one `.nir` file into the snn JSON schema (native units —
/// biological scale: MΩ, V).
///
/// # Errors
///
/// Every [`ConvertError`] — each is a named, one-line refusal.
///
/// # Panics
///
/// Never on stranger input (all decode paths are checked); the export
/// buffer growth loop terminates at 64 MiB + data scale.
pub fn convert_file(path: &Path, opts: NirImportOptions) -> Result<Converted, ConvertError> {
    convert_file_opts(path, opts, false)
}

/// Convert with the `--sim-units` convention transform available
/// (amended B-brief, 2026-08-24). When `sim_units` is set, each LIF
/// node whose `r` is under 1 MΩ is read in the ecosystem's
/// simulation-unit convention:
///
/// - a voltage scale per node, `V = min(10 / largest |threshold|,
///   45 / largest |potential|)` mV per unit, stamped: the node's
///   largest threshold is 1,000 quanta, fewer only where a potential
///   past 4.5 times it lowers `V`, its others in proportion, and the
///   −100 mV floor ten times it or more down, under the membrane's
///   +50 mV ceiling;
/// - the potentials × `V`, read as mV, on the CENTI GRID (forced);
/// - `r × V × 1000 / SIM_CURRENT_QUANTA` MΩ, so the product `r·I` keeps
///   the source's scale under the library's true-scale weights;
/// - the import in simulation units, [`effective_options`]: true-scale
///   weights, NIR's `v > v_threshold`, no refractory period;
/// - each `Affine`, `y = W·x + b` (snnTorch's export of a biased
///   `nn.Linear`), rewritten: `W` a Linear under the Affine's name, and a
///   bias not all zero a [`Bias`] input into a one-column Linear holding
///   `b`, into each LIF the Affine feeds; an all-zero bias adds nothing.
///   A bias input's value is 0 before its `start`, the LIFs' depth, and
///   1 from it: a spike reaches the next population one step late, so a
///   population one spike edge from an Input takes its bias from step 1.
///
/// The transform is an INTERPRETIVE ACT (a convention assumption about
/// what the stranger's dimensionless numbers mean) — hence opt-in,
/// sidecar-stamped, never silent.
///
/// # Errors
///
/// Every [`ConvertError`]; with `sim_units == false`, a detected
/// sim-unit file yields [`ConvertError::SimUnits`] naming BOTH walls
/// and the flag. Detection is `r < 1e6 Ω` — the only population that
/// cannot import natively (the marginal [0.5, 1) MΩ band rounds to
/// the 1 MΩ floor natively; detection deliberately refuses the
/// ambiguous band rather than guessing). An `Affine` is
/// [`ConvertError::UnsupportedNode`] in native units, and under
/// `--sim-units` [`ConvertError::Bias`] when its bias feeds anything but
/// a LIF, a LIF that no Input reaches or that the Inputs reach at two
/// depths (a loop among them), or LIFs at two depths; a bias whose
/// length is not `W`'s rows is [`ConvertError::BadData`].
pub fn convert_file_opts(
    path: &Path,
    opts: NirImportOptions,
    sim_units: bool,
) -> Result<Converted, ConvertError> {
    // Effective options: the transform forces the centi grid.
    let effective = effective_options(opts, sim_units);
    let f = File::open(path).map_err(|e| ConvertError::Open(format!("{}: {e}", path.display())))?;
    let root = f.root();

    // version (recorded, never behavior-switching)
    let nir_version = read_str(
        &root
            .dataset("version")
            .map_err(|e| ConvertError::Layout(format!("version: {e}")))?,
    )?;

    // node group + NIRGraph contract
    let node = root
        .group("node")
        .map_err(|e| ConvertError::Layout(format!("node/: {e}")))?;
    let graph_type = read_str(
        &node
            .dataset("type")
            .map_err(|e| ConvertError::Layout(format!("node/type: {e}")))?,
    )?;
    if graph_type != "NIRGraph" {
        return Err(ConvertError::Layout(format!(
            "node/type is {graph_type:?} — expected \"NIRGraph\""
        )));
    }

    // PRE-READ census — nothing decodes before every filter is admitted
    census(&node)?;

    let nodes = node
        .group("nodes")
        .map_err(|e| ConvertError::Layout(format!("node/nodes/: {e}")))?;
    let node_names = nodes.groups().map_err(bad_groups("node/nodes".into()))?;

    let mut builder = NirBuilder::new(effective);
    let mut stamp = Stamp {
        nir_version,
        node_census: Vec::new(),
        f32_datasets: Vec::new(),
        dt_us: opts.dt_us,
        resolution: resolution_name(effective),
        sim_units,
        volt_scale: Vec::new(),
        bias: Vec::new(),
    };

    // Owned name storage: NirNode borrows &'a str for the builder's
    // lifetime — collect ALL names first so no push mutates the Vec
    // while the builder holds borrows into it. That includes the two
    // names a bias input takes, made from each node's name: only an
    // Affine's are used.
    let owned: Vec<String> = node_names.clone();
    let bias_names: Vec<[String; 2]> = node_names
        .iter()
        .map(|n| [format!("{n}/bias"), format!("{n}/b")])
        .collect();
    // each Affine whose bias is not all zero: its slot and its bias
    let mut biased: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut index: std::collections::HashMap<String, usize> =
        std::collections::HashMap::with_capacity(node_names.len());

    for (slot, name) in node_names.iter().enumerate() {
        let borrowed: &str = owned[slot].as_str();
        let g = nodes
            .group(name)
            .map_err(bad_groups(format!("node/nodes/{name}")))?;
        let ty = read_str(
            &g.dataset("type")
                .map_err(|e| ConvertError::Layout(format!("type: {e}")))?,
        )?;
        match ty.as_str() {
            "Input" | "Output" => {
                let ds = g
                    .dataset("shape")
                    .map_err(|e| ConvertError::Layout(format!("shape: {e}")))?;
                let sh = shape_u32(&ds, &format!("node/nodes/{name}/shape"))?;
                let idx = if ty == "Input" {
                    builder
                        .add_input(borrowed, &sh)
                        .map_err(|e| ConvertError::Snn {
                            stage: "add_input",
                            msg: e.to_string(),
                        })
                } else {
                    builder
                        .add_output(borrowed, &sh)
                        .map_err(|e| ConvertError::Snn {
                            stage: "add_output",
                            msg: e.to_string(),
                        })
                }?;
                index.insert(name.clone(), idx);
            }
            "LIF" => {
                let mut get = |field: &str| -> Result<Vec<f64>, ConvertError> {
                    let ds = g
                        .dataset(field)
                        .map_err(|e| ConvertError::Layout(format!("LIF {field}: {e}")))?;
                    read_param_f64(
                        &ds,
                        &format!("node/nodes/{name}/{field}"),
                        &mut stamp.f32_datasets,
                    )
                };
                let tau = get("tau")?;
                let r = get("r")?;
                let v_leak = get("v_leak")?;
                let v_threshold = get("v_threshold")?;
                // absent v_reset = the reference's zeros semantics (None)
                let v_reset = match g.dataset("v_reset") {
                    Ok(ds) => Some(read_param_f64(
                        &ds,
                        &format!("node/nodes/{name}/v_reset"),
                        &mut stamp.f32_datasets,
                    )?),
                    Err(_) => None,
                };
                // Sim-unit detection (the ratified cutoff): without the
                // flag, refuse naming BOTH walls + the flag; with it,
                // apply the transform this function's doc lists.
                let (r, v_leak, v_threshold, v_reset) = if r.iter().any(|&x| x < 1e6) {
                    if !sim_units {
                        return Err(ConvertError::SimUnits {
                            node: name.clone(),
                            r_ohm: r[0],
                        });
                    }
                    let v = volt_scale(&v_leak, &v_threshold, v_reset.as_deref());
                    stamp.volt_scale.push((name.clone(), v));
                    let t = |p: Vec<f64>| p.into_iter().map(|x| x * 1e-3 * v).collect();
                    // r·I keeps the source's scale: the library's weights
                    // are SIM_CURRENT_QUANTA quanta per unit
                    let per_unit = 1e9 * v / f64::from(SIM_CURRENT_QUANTA);
                    let r: Vec<f64> = r.into_iter().map(|x| x * per_unit).collect();
                    (r, t(v_leak), t(v_threshold), v_reset.map(t))
                } else {
                    (r, v_leak, v_threshold, v_reset)
                };
                let params = NirLifParams {
                    tau_s: &tau,
                    r_ohm: &r,
                    v_leak_v: &v_leak,
                    v_threshold_v: &v_threshold,
                    v_reset_v: v_reset.as_deref(),
                };
                let idx = builder.add_lif_population(borrowed, &params).map_err(|e| {
                    ConvertError::Snn {
                        stage: "add_lif_population",
                        msg: format!("[node {name}] {e}"),
                    }
                })?;
                index.insert(name.clone(), idx);
            }
            "Linear" => {
                let (vals, rows, cols) = read_weight(&g, name, &mut stamp.f32_datasets)?;
                let idx = builder
                    .add_linear(borrowed, &vals, rows, cols)
                    .map_err(|e| ConvertError::Snn {
                        stage: "add_linear",
                        msg: e.to_string(),
                    })?;
                index.insert(name.clone(), idx);
            }
            // W a Linear under the Affine's name; its bias waits for the
            // edges, which place it
            "Affine" if sim_units => {
                let (vals, rows, cols) = read_weight(&g, name, &mut stamp.f32_datasets)?;
                let path = format!("node/nodes/{name}/bias");
                let ds = g
                    .dataset("bias")
                    .map_err(|e| ConvertError::Layout(format!("Affine bias: {e}")))?;
                let bias = read_param_f64(&ds, &path, &mut stamp.f32_datasets)?;
                if bias.len() != rows {
                    return Err(ConvertError::BadData {
                        dataset: path,
                        what: format!("{} values != the weight's {rows} rows", bias.len()),
                    });
                }
                let idx = builder
                    .add_linear(borrowed, &vals, rows, cols)
                    .map_err(|e| ConvertError::Snn {
                        stage: "add_linear",
                        msg: e.to_string(),
                    })?;
                index.insert(name.clone(), idx);
                if bias.iter().any(|&b| b != 0.0) {
                    biased.push((slot, bias));
                }
            }
            other => {
                return Err(ConvertError::UnsupportedNode {
                    node: name.clone(),
                    kind: other.into(),
                });
            }
        }
        stamp.node_census.push((name.clone(), ty));
    }

    // edges: (N,2) vlen strings → index pairs
    let edges_ds = node
        .dataset("edges")
        .map_err(|e| ConvertError::Layout(format!("node/edges: {e}")))?;
    let flat = edges_ds
        .read_vlen_strings(VlenStringReadOptions::default())
        .map_err(|e| ConvertError::Open(format!("edges read: {e}")))?;
    if flat.len() % 2 != 0 {
        return Err(ConvertError::Layout(format!(
            "edges: odd string count {} — expected (N,2)",
            flat.len()
        )));
    }
    let add_edge = |builder: &mut NirBuilder<'_>, a: usize, b: usize| {
        builder.add_edge(a, b).map_err(|e| ConvertError::Snn {
            stage: "add_edge",
            msg: e.to_string(),
        })
    };
    // the builder numbers the file's nodes as it adds them, in document
    // order, so a node's index is its slot
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(flat.len() / 2);
    for pair in flat.as_chunks::<2>().0 {
        let resolve = |n: &str| {
            index
                .get(n)
                .copied()
                .ok_or_else(|| ConvertError::Layout(format!("edge names unknown node {n:?}")))
        };
        let (a, b) = (resolve(&pair[0])?, resolve(&pair[1])?);
        add_edge(&mut builder, a, b)?;
        edges.push((a, b));
    }

    // each bias not all zero: an Input of one feature into a one-column
    // Linear into each LIF its Affine feeds, after the file's nodes
    let kinds: Vec<&str> = stamp.node_census.iter().map(|(_, k)| k.as_str()).collect();
    for (slot, bias) in &biased {
        let Some(start) = bias_start(&node_names, &kinds, &edges, *slot)? else {
            continue; // the Affine feeds nothing, so neither does its bias
        };
        let [input_name, linear_name] = &bias_names[*slot];
        let input = builder
            .add_input(input_name, &[1])
            .map_err(|e| ConvertError::Snn {
                stage: "add_input",
                msg: e.to_string(),
            })?;
        let linear = builder
            .add_linear(linear_name, bias, bias.len(), 1)
            .map_err(|e| ConvertError::Snn {
                stage: "add_linear",
                msg: e.to_string(),
            })?;
        add_edge(&mut builder, input, linear)?;
        for &(_, lif) in edges.iter().filter(|&&(a, _)| a == *slot) {
            add_edge(&mut builder, linear, lif)?;
        }
        stamp.bias.push(Bias {
            input: input_name.clone(),
            start,
        });
    }

    let graph = builder.build().map_err(|e| ConvertError::Snn {
        stage: "build",
        msg: e.to_string(),
    })?;

    // Export with bounded buffer growth (start at data scale, ×4, four tries).
    let w = graph.weights.len();
    let l = graph.lifs.len();
    let mut cap = 1usize << 16;
    let mut scale = cap.max(w.saturating_mul(40)).max(l.saturating_mul(300));
    loop {
        let mut buf = vec![0u8; scale];
        match nir_export(
            &graph.nodes,
            &graph.edges,
            &graph.weights,
            &graph.lifs,
            graph.opts,
            &mut buf,
        ) {
            Ok(n) => {
                buf.truncate(n);
                return Ok(Converted { json: buf, stamp });
            }
            Err(NirError::ExportTooSmall) => {
                if scale > (1 << 30) {
                    return Err(ConvertError::Layout(format!(
                        "export exceeds 1 GiB (weights {w}, lifs {l}) — refusing"
                    )));
                }
                scale = scale.saturating_mul(4).max(cap);
                cap = scale;
            }
            Err(e) => {
                return Err(ConvertError::Snn {
                    stage: "nir_export",
                    msg: e.to_string(),
                });
            }
        }
    }
}

/// The import options a conversion runs under: `opts`, or under
/// `--sim-units` the same time step in simulation units,
/// [`NirImportOptions::sim_units`]: the centi-mV grid, true-scale
/// weights, NIR's firing rule, no refractory. `--freeze` builds the
/// network under the same.
#[must_use]
pub fn effective_options(opts: NirImportOptions, sim_units: bool) -> NirImportOptions {
    if sim_units {
        NirImportOptions::sim_units(opts.dt_us)
    } else {
        opts
    }
}

/// One LIF node's voltage scale under `--sim-units`, mV per simulation
/// unit of potential: `V = min(10 / max|threshold|, 45 / max|potential|)`.
/// With `V` from the threshold, the node's largest threshold is 1,000
/// quanta on the centi-mV grid and the −100 mV floor ten times it
/// below 0, whatever the threshold; its other thresholds scale with it.
/// `V` is lowered only where the node's largest potential would pass
/// 45 mV, under the membrane's +50 mV ceiling, and then the largest
/// threshold is fewer quanta and the floor further down. `r`'s range
/// is not a cap: an `r` that the scale takes past 65,535 MΩ is refused
/// by name.
fn volt_scale(v_leak: &[f64], v_threshold: &[f64], v_reset: Option<&[f64]>) -> f64 {
    let largest = |xs: &[f64]| xs.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let threshold = largest(v_threshold);
    let potential = largest(v_leak)
        .max(threshold)
        .max(v_reset.map_or(0.0, largest));
    // a term over 0 is +inf, which leaves the other term to decide
    let v = (10.0 / threshold).min(45.0 / potential);
    // both at 0: a zero threshold, which the import refuses by name
    // (ThresholdZero) whatever finite V it gets
    if v.is_finite() { v } else { 1.0 }
}

/// What `--freeze` writes (README § Freeze): one module of arrays and the
/// trace of its drive on the host.
#[derive(Debug)]
pub struct Frozen {
    /// One `pub mod`: the arrays a `FixedNetwork<N, S>` steps, written by
    /// the library's freezer (`neuralos_snn::fixed::freeze::module`).
    pub module: String,
    /// `neuralos-trace v1`: the module's header line, then one row per
    /// step, each written by `neuralos_snn::trace::row`.
    pub trace: String,
    /// The network's neurons.
    pub neurons: usize,
    /// The network's synapses.
    pub synapses: usize,
    /// What the library noted as it assembled the graph.
    pub assembly: Assembly,
}

/// The library's assembly report (`neuralos_snn::nir::NirAssemblyReport`),
/// owned: `--freeze` prints its notes and writes all of it in the
/// sidecar (README § Freeze).
#[derive(Debug, Clone, PartialEq)]
pub struct Assembly {
    /// Neurons, every population's.
    pub neurons: usize,
    /// Synapses, the LIF→LIF pairs' and the spiking Linear edges'.
    pub synapses: usize,
    /// Input nodes.
    pub inputs: usize,
    /// Linear nodes that feed a LIF, from an Input or a LIF.
    pub drive_linears: usize,
    /// Quantized encoder matrices.
    pub stages: usize,
    /// Each stage composed from two tensors or more, root first.
    pub fused: Vec<Fused>,
    /// The undriven LIF populations and Linear encoders: no input reaches
    /// them, or, for an encoder, it reaches no LIF.
    pub undriven: Vec<String>,
    /// The gain note: in native units, more than one drive Linear, each
    /// scaled by its own absmax.
    pub multi_linear_gain: bool,
    /// Plasticity frozen at assembly.
    pub plasticity_frozen: bool,
}

/// One fused stage: its chain of Linear nodes, root first, and each
/// tensor's quantization scale in chain order.
#[derive(Debug, Clone, PartialEq)]
pub struct Fused {
    /// The composed nodes' names, root first, target last.
    pub chain: Vec<String>,
    /// Each component tensor's quantization scale, in `chain` order.
    pub scales: Vec<f64>,
}

/// Everything that stops `--freeze`, each nameable in one line.
#[derive(Debug)]
pub enum FreezeError {
    /// The converted JSON does not import: this tool wrote it, so the
    /// fault is the tool's, not the stranger's.
    Import(String),
    /// The graph does not assemble: the library's named refusal
    /// (`build_network`), e.g. a readout to Output or no LIF at all.
    Assembly(String),
    /// More neurons than a `u16` id holds.
    TooManyNeurons(usize),
    /// Plasticity on: a `FixedNetwork` has none (never, from NIR:
    /// `build_network` freezes it).
    Plasticity,
    /// A run of the drive does not give one value per input feature;
    /// `run` counts from 1.
    Input {
        run: usize,
        given: usize,
        features: usize,
    },
    /// The drive holds no run, a run of 0 steps, or more steps than a
    /// trace counts (`u32`).
    Drive(&'static str),
    /// `name` is not a [`module_name`], which `freeze` needs: the
    /// module is named `name`, and the trace's case name is `name` with
    /// `_` made `-`. Never from the CLI, which passes `module_name`'s
    /// output.
    Name(String),
    /// A bias input that is not an Input of one feature in the graph:
    /// `bias` is not the conversion's that wrote the JSON. Never from the
    /// CLI, which passes the conversion's own.
    Bias(String),
}

impl fmt::Display for FreezeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Import(m) => write!(f, "the converted JSON does not import: {m}"),
            Self::Assembly(m) => write!(f, "the graph does not assemble: {m}"),
            Self::TooManyNeurons(n) => write!(
                f,
                "{n} neurons: a FixedNetwork neuron id is a u16, at most 65,535"
            ),
            Self::Plasticity => write!(f, "plasticity is on: a FixedNetwork has none"),
            Self::Input {
                run,
                given,
                features,
            } => write!(
                f,
                "the drive's run {run} gives {given} value{}; the graph has {features} input feature{}, one value each",
                if *given == 1 { "" } else { "s" },
                if *features == 1 { "" } else { "s" },
            ),
            Self::Drive(why) => write!(f, "the drive holds {why}"),
            Self::Name(n) => write!(
                f,
                "{n:?} is not a module name: lowercase ASCII letters, digits and _, a letter or _ first, not _ alone, not a Rust keyword"
            ),
            Self::Bias(n) => write!(
                f,
                "the bias input {n:?} is not an Input of one feature in the graph"
            ),
        }
    }
}

/// Rust's keywords, strict and reserved: no module takes their name.
const KEYWORDS: &[&str] = &[
    "abstract", "as", "async", "await", "become", "box", "break", "const", "continue", "crate",
    "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if", "impl",
    "in", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub", "ref",
    "return", "self", "static", "struct", "super", "trait", "true", "try", "type", "typeof",
    "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
];

/// The module name `--freeze` gives a file stem: the stem lowercased,
/// every character but an ASCII letter, digit or `_` made `_`. `None`
/// when that is no identifier: empty, a digit first, `_` alone, or a
/// keyword.
#[must_use]
pub fn module_name(stem: &str) -> Option<String> {
    let name: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let starts_well = name.starts_with(|c: char| c.is_ascii_lowercase() || c == '_');
    (starts_well && name != "_" && !KEYWORDS.contains(&name.as_str())).then_some(name)
}

/// Freeze a converted graph: import `json` under `opts` (the
/// conversion's own, [`effective_options`]), build it (`build_network`),
/// and write one module named `name` (a [`module_name`]) holding its
/// arrays, `kind=stranger` in its header, and its drive: each run of
/// `drive`, a step count and one value per feature of the graph's own
/// Inputs, in Input order, in [`Thousandths`] of a unit (`None` meaning
/// 1.0 for every feature), as the currents the graph's encoder gives
/// for it. Each bias input of `bias`, the conversion's [`Stamp::bias`],
/// is 0 before its start and 1 from it ([`Inputs`]), so a run splits
/// where one starts inside it. With the module, the trace of that drive
/// on the host, and what the library noted as it assembled the graph.
///
/// The trace is the std network's, plasticity off, the network
/// `FixedNetwork::try_from` would convert, each row written by
/// `neuralos_snn::trace::row`. The fixed step equals the std step on
/// every network `try_from` converts (the library's trace tests), and
/// this crate's test builds the module and steps it to the same rows.
///
/// # Errors
///
/// Every [`FreezeError`].
///
/// # Panics
///
/// Never, and not by luck: an assembled network steps, a `String` takes
/// every write, and neither the header line nor the library's freezer
/// refuses what it is handed here. Each refusal is ruled out at its
/// source: plasticity on, refused above; a CSR that no longer delivers
/// its synapses — `build_network` finalizes once, after its last edge;
/// a clock not at 0, and a neuron that is not at rest — the module is
/// written before the stepping loop below, and `build_network` builds
/// every membrane at its leak; neurons on two grids — `build_network`
/// builds every neuron on the options' grid; a case name off the rule —
/// `name` is a [`module_name`] (anything else is refused first, as
/// [`FreezeError::Name`]), which with `_` made `-` is lowercase
/// letters, digits and `-`; an `id` that is not the neuron's position —
/// `build_network` numbers the neurons as it pushes them; and a chain
/// that does not rebuild its neuron, which would be a defect of the
/// library, not of a stranger's graph. The drive it is given is the
/// encoder's output, one current per neuron, and [`Inputs::at`] is given
/// one value per feature of the graph's own Inputs, refused above
/// otherwise.
pub fn freeze(
    json: &[u8],
    opts: NirImportOptions,
    name: &str,
    drive: &[(u32, Option<&[Thousandths]>)],
    bias: &[Bias],
) -> Result<Frozen, FreezeError> {
    if module_name(name).as_deref() != Some(name) {
        return Err(FreezeError::Name(name.to_string()));
    }
    let graph = NirImport::from_json(json, opts).map_err(|e| FreezeError::Import(e.to_string()))?;
    // one LIF record per neuron; a FixedNetwork's neuron ids are u16
    if graph.lifs.len() > usize::from(u16::MAX) {
        return Err(FreezeError::TooManyNeurons(graph.lifs.len()));
    }
    let (mut net, enc, report) = graph
        .build_network()
        .map_err(|e| FreezeError::Assembly(e.to_string()))?;
    if net.plasticity_enabled() {
        return Err(FreezeError::Plasticity);
    }

    let inputs = Inputs::new(&graph, &enc, bias).map_err(FreezeError::Bias)?;

    if drive.is_empty() {
        return Err(FreezeError::Drive("no run"));
    }
    let features = inputs.features();
    let starts = inputs.starts();
    let mut steps = 0u32;
    let mut runs: Vec<(u32, Vec<i16>)> = Vec::with_capacity(drive.len());
    for (run, &(count, input)) in (1..).zip(drive) {
        if count == 0 {
            return Err(FreezeError::Drive("a run of 0 steps"));
        }
        let first = steps;
        steps = steps
            .checked_add(count)
            .ok_or(FreezeError::Drive("more steps than a trace counts"))?;
        let values = match input {
            Some(v) if v.len() != features => {
                return Err(FreezeError::Input {
                    run,
                    given: v.len(),
                    features,
                });
            }
            Some(v) => v.to_vec(),
            None => vec![ONE; features],
        };
        // the run splits where a bias input starts inside it
        let mut from = first;
        let ends = starts.iter().copied().filter(|&s| s > first && s < steps);
        for to in ends.chain([steps]) {
            let per_input = inputs.at(from, &values);
            let slices: Vec<&[Thousandths]> = per_input.iter().map(Vec::as_slice).collect();
            runs.push((to - from, enc.encode(&slices)));
            from = to;
        }
    }

    // build_network builds the network at opts.dt_us, every neuron on
    // opts.resolution, plasticity off (refused above)
    let case = name.replace('_', "-");
    let header = trace::header(&net, &case, Kind::Stranger, steps, Rows::All)
        .expect("one grid, the options', and a case name from module_name");
    let synapses = FixedSynapse::from_network(&net).len();
    let module = freezer::module(&net, &case, Kind::Stranger, steps, Rows::All, &runs);

    let mut trace = format!("{header}\n");
    let mut fired = vec![false; net.neurons().len()];
    let mut step = 0u32;
    for (count, currents) in &runs {
        for _ in 0..*count {
            let time_us = net.current_time_us();
            let spikes = net.step(currents).expect("an assembled network steps");
            fired.fill(false);
            for spike in spikes {
                fired[usize::from(spike.neuron_id)] = true;
            }
            row(&mut trace, step, time_us, &fired, net.neurons())
                .expect("a String takes every write");
            step += 1;
        }
    }
    let assembly = Assembly {
        neurons: report.neurons,
        synapses: report.synapses,
        inputs: report.inputs,
        drive_linears: report.drive_linears,
        stages: report.stages,
        fused: report
            .fused
            .iter()
            .map(|f| Fused {
                chain: f.chain.iter().map(|n| (*n).to_string()).collect(),
                scales: f.scales.clone(),
            })
            .collect(),
        undriven: report.undriven.iter().map(|n| (*n).to_string()).collect(),
        multi_linear_gain: report.multi_linear_gain,
        plasticity_frozen: report.plasticity_frozen,
    };
    Ok(Frozen {
        module,
        trace,
        neurons: net.neurons().len(),
        synapses,
        assembly,
    })
}

fn resolution_name(opts: NirImportOptions) -> &'static str {
    match opts.resolution {
        neuralos_snn::VoltageResolution::Millivolt => "mv",
        neuralos_snn::VoltageResolution::CentiMillivolt => "centi-mv",
    }
}

/// Render the sidecar stamp document (hand-built — the tool writes
/// exactly two small documents, this and the export's canonical bytes;
/// no serializer dependency). With `--freeze`'s [`Assembly`], the
/// document carries it too, as `assembly`.
#[must_use]
pub fn stamp_json(s: &Stamp, source: &Path, assembly: Option<&Assembly>) -> String {
    let mut out = String::from("{\"tool\":\"neuralos-nir2json\",\"version\":\"");
    out.push_str(TOOL_VERSION);
    out.push_str("\",\"source\":\"");
    out.push_str(&json_escape(&source.display().to_string()));
    out.push_str("\",\"nir_version\":\"");
    out.push_str(&json_escape(&s.nir_version));
    out.push_str("\",\"dt_us\":");
    out.push_str(&s.dt_us.to_string());
    out.push_str(",\"resolution\":\"");
    out.push_str(&json_escape(s.resolution));
    out.push_str("\",\"sim_units\":");
    out.push_str(if s.sim_units { "true" } else { "false" });
    if s.sim_units {
        out.push_str(",\"current_quanta\":");
        out.push_str(&SIM_CURRENT_QUANTA.to_string());
        out.push_str(",\"volt_scale\":{");
        for (i, (name, v)) in s.volt_scale.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"{}\":{v}", json_escape(name)));
        }
        out.push_str("},\"bias\":{");
        for (i, b) in s.bias.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"{}\":{}", json_escape(&b.input), b.start));
        }
        out.push('}');
    }
    out.push_str(",\"f32_widened\":[");
    for (i, d) in s.f32_datasets.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("\"{}\"", json_escape(d)));
    }
    out.push_str("],\"nodes\":{");
    for (i, (name, kind)) in s.node_census.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "\"{}\":\"{}\"",
            json_escape(name),
            json_escape(kind)
        ));
    }
    out.push('}');
    if let Some(a) = assembly {
        out.push_str(&assembly_json(a));
    }
    out.push('}');
    out
}

/// A string as a JSON string's contents: `"`, `\` and the control
/// characters escaped. The export refuses a node name that holds any of
/// them (`NirError::NonAsciiNodeName`), so no conversion writes such a
/// name into the sidecar; the file's own version string and a file path
/// can hold one, and a caller of [`stamp_json`] can pass anything.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out
}

/// The sidecar's `assembly` member, its leading comma included.
fn assembly_json(a: &Assembly) -> String {
    let names = |xs: &[String]| {
        xs.iter()
            .map(|x| format!("\"{}\"", json_escape(x)))
            .collect::<Vec<_>>()
            .join(",")
    };
    let fused = a
        .fused
        .iter()
        .map(|f| {
            let scales: Vec<String> = f.scales.iter().map(f64::to_string).collect();
            format!(
                "{{\"chain\":[{}],\"scales\":[{}]}}",
                names(&f.chain),
                scales.join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        ",\"assembly\":{{\"neurons\":{},\"synapses\":{},\"inputs\":{},\"drive_linears\":{},\
         \"stages\":{},\"fused\":[{fused}],\"undriven\":[{}],\"multi_linear_gain\":{},\
         \"plasticity_frozen\":{}}}",
        a.neurons,
        a.synapses,
        a.inputs,
        a.drive_linears,
        a.stages,
        names(&a.undriven),
        a.multi_linear_gain,
        a.plasticity_frozen
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO: Thousandths = Thousandths(0);

    fn fixture(name: &str) -> std::path::PathBuf {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        assert!(p.exists(), "fixture missing: {}", p.display());
        p
    }

    fn rt_fixture(name: &str) -> std::path::PathBuf {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../neuralos-rt/tests/nir_fixtures")
            .join(name);
        assert!(p.exists(), "rt fixture missing: {}", p.display());
        p
    }

    #[test]
    fn chain_population_converts_and_imports() {
        let c = convert_file(
            &rt_fixture("chain_population.nir"),
            NirImportOptions::default(),
        )
        .expect("converts");
        // the money assertion: the published crate imports what we wrote
        let g = neuralos_snn::nir::NirImport::from_json(&c.json, NirImportOptions::default())
            .expect("snn imports the tool's output");
        assert_eq!(g.nodes.len(), 4);
        assert_eq!(g.edges.len(), 3);
        assert!(c.stamp.f32_datasets.is_empty());
        assert_eq!(c.stamp.nir_version, "1.0.9.dev1+g7883c3c85");
    }

    #[test]
    fn lzf_is_refused_by_name() {
        let err = convert_file(&fixture("neg_filter_lzf.nir"), NirImportOptions::default())
            .expect_err("lzf must be refused");
        match &err {
            ConvertError::Filter { name, .. } => assert_eq!(name, "lzf"),
            other => panic!("expected Filter, got {other:?}"),
        }
        assert!(
            err.to_string().contains("lzf"),
            "message names the filter: {err}"
        );
    }

    #[test]
    fn out_of_subset_kind_is_named() {
        let err = convert_file(
            &fixture("community/lif_norse.nir"),
            NirImportOptions::default(),
        )
        .expect_err("Affine is refused in native units");
        match &err {
            ConvertError::UnsupportedNode { kind, .. } => assert_eq!(kind, "Affine"),
            other => panic!("expected UnsupportedNode, got {other:?}"),
        }
        assert_eq!(
            err.to_string(),
            "node '0' is kind 'Affine' — in native units outside the supported subset \
             (Input, LIF, Linear, Output); `--sim-units` converts it, its bias an input of its own"
        );
        // any other kind, the message as before
        assert_eq!(
            ConvertError::UnsupportedNode {
                node: "n".into(),
                kind: "CubaLIF".into(),
            }
            .to_string(),
            "node 'n' is kind 'CubaLIF' — outside the supported subset \
             (Input, LIF, Linear, Output); refusing loudly rather than partially converting"
        );
    }

    /// Two biased Affines under `--sim-units`
    /// (`tools/gen_nir2json_fixtures.py`): each `W` a Linear under its
    /// Affine's name, each bias an Input of one feature into a one-column
    /// Linear into the LIF it feeds, from that LIF's depth, after the
    /// file's nodes and edges.
    #[test]
    fn an_affine_converts_under_sim_units_its_bias_an_input_of_its_own() {
        use neuralos_snn::nir::NirNodeKind::{Input, Lif, Linear, Output};
        let (c, opts) = sim_conversion("affine_two_layer.nir");
        assert_eq!(
            c.stamp.bias,
            [
                Bias {
                    input: "a/bias".into(),
                    start: 0,
                },
                Bias {
                    input: "c/bias".into(),
                    start: 1,
                },
            ]
        );
        for d in ["node/nodes/a/bias", "node/nodes/c/bias"] {
            assert!(
                c.stamp.f32_datasets.iter().any(|x| x == d),
                "{d} widened: {:?}",
                c.stamp.f32_datasets
            );
        }
        let g = NirImport::from_json(&c.json, opts).expect("imports");
        let nodes: Vec<_> = g.nodes.iter().map(|n| (n.name, n.kind)).collect();
        assert_eq!(
            nodes,
            [
                ("a", Linear),
                ("c", Linear),
                ("input", Input),
                ("l1", Lif),
                ("l2", Lif),
                ("output", Output),
                ("a/bias", Input),
                ("a/b", Linear),
                ("c/bias", Input),
                ("c/b", Linear),
            ]
        );
        let name = |i: u32| g.nodes[i as usize].name;
        let edges: Vec<_> = g.edges.iter().map(|&(a, b)| (name(a), name(b))).collect();
        assert_eq!(
            edges,
            [
                ("input", "a"),
                ("a", "l1"),
                ("l1", "c"),
                ("c", "l2"),
                ("l2", "output"),
                ("a/bias", "a/b"),
                ("a/b", "l1"),
                ("c/bias", "c/b"),
                ("c/b", "l2"),
            ]
        );
        let (_, enc, report) = g.build_network().expect("assembles");
        assert_eq!((report.neurons, report.inputs), (5, 3));
        // the biases alone, `round(b · 1,000)` quanta a neuron
        assert_eq!(
            enc.encode(&[&[ZERO, ZERO], &[ONE], &[ONE]]),
            [50, -20, 0, 30, -10]
        );
        // `W` alone, into l1; l2's `W` is a spike edge, not a drive
        assert_eq!(
            enc.encode(&[&[ONE, ZERO], &[ZERO], &[ZERO]]),
            [500, 125, -500, 0, 0]
        );
    }

    #[test]
    fn an_affine_s_bias_of_the_wrong_length_is_bad_data() {
        let opts = NirImportOptions {
            dt_us: SIM_DT_US,
            ..NirImportOptions::default()
        };
        match convert_file_opts(&fixture("affine_bias_length.nir"), opts, true) {
            Err(ConvertError::BadData { dataset, what }) => {
                assert_eq!(dataset, "node/nodes/c/bias");
                assert_eq!(what, "3 values != the weight's 2 rows");
            }
            other => panic!("expected BadData, got {other:?}"),
        }
    }

    /// A weight of one dimension is refused by name, never read as `W`'s
    /// rows by its columns (`tools/gen_nir2json_fixtures.py`).
    #[test]
    fn a_weight_that_is_not_2_d_is_bad_data() {
        match convert_file(&fixture("weight_1d.nir"), NirImportOptions::default()) {
            Err(ConvertError::BadData { dataset, what }) => {
                assert_eq!(dataset, "node/nodes/linear/weight");
                assert_eq!(what, "1-D weight — expected 2-D");
            }
            other => panic!("expected BadData, got {other:?}"),
        }
    }

    /// The two-layer Affine graph and a third Affine, `d`, that feeds
    /// nothing (`tools/gen_nir2json_fixtures.py`): its `W` a Linear under
    /// its name, and no input for its bias, which would drive nothing.
    #[test]
    fn an_affine_that_feeds_nothing_adds_no_bias_input() {
        use neuralos_snn::nir::NirNodeKind::Linear;
        let (c, opts) = sim_conversion("affine_dangling.nir");
        let bias: Vec<_> = c
            .stamp
            .bias
            .iter()
            .map(|b| (b.input.as_str(), b.start))
            .collect();
        assert_eq!(bias, [("a/bias", 0), ("c/bias", 1)]);
        let g = NirImport::from_json(&c.json, opts).expect("imports");
        let d: Vec<_> = g
            .nodes
            .iter()
            .filter(|n| n.name.starts_with('d'))
            .map(|n| (n.name, n.kind))
            .collect();
        assert_eq!(d, [("d", Linear)]);
    }

    /// Node names by slot, for graphs given by kinds and edges.
    fn slots(n: usize) -> Vec<String> {
        (0..n).map(|i| i.to_string()).collect()
    }

    #[test]
    fn a_bias_starts_at_the_depth_of_the_lifs_it_feeds() {
        let start = |kinds: &[&str], edges: &[(usize, usize)], affine: usize| {
            bias_start(&slots(kinds.len()), kinds, edges, affine).expect("placed")
        };
        // input → a → lif → b → lif → c → lif: a bias at each depth
        let kinds = ["Input", "Affine", "LIF", "Affine", "LIF", "Affine", "LIF"];
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)];
        assert_eq!(
            [1, 3, 5].map(|a| start(&kinds, &edges, a)),
            [Some(0), Some(1), Some(2)]
        );
        // a direct LIF→LIF edge delays a spike too: input → x → lif →
        // lif → a → lif
        let kinds = ["Input", "Linear", "LIF", "LIF", "Affine", "LIF"];
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
        assert_eq!(start(&kinds, &edges, 4), Some(2));
        // two paths at one depth: input → x → a and input → y → a → lif
        let kinds = ["Input", "Linear", "Linear", "Affine", "LIF"];
        let edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)];
        assert_eq!(start(&kinds, &edges, 3), Some(0));
        // a skip after the bias's LIF puts a later LIF at two depths, and
        // that LIF is not the bias's: input → a → lif → lif → output, and
        // the first lif → lif → the second
        let kinds = ["Input", "Affine", "LIF", "LIF", "LIF", "Output"];
        let edges = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 3), (3, 5)];
        assert_eq!(start(&kinds, &edges, 1), Some(0));
        // an Affine into two LIFs at one depth: one start
        assert_eq!(
            start(
                &["Input", "Affine", "LIF", "LIF"],
                &[(0, 1), (1, 2), (1, 3)],
                1
            ),
            Some(0)
        );
        // an Affine that feeds nothing places no bias
        assert_eq!(start(&["Input", "Affine"], &[(0, 1)], 1), None);
    }

    #[test]
    fn a_bias_with_no_place_is_refused_by_name() {
        let refusal = |kinds: &[&str], edges: &[(usize, usize)], affine: usize| match bias_start(
            &slots(kinds.len()),
            kinds,
            edges,
            affine,
        ) {
            Err(ConvertError::Bias { node, why }) => {
                assert_eq!(node, affine.to_string());
                why
            }
            other => panic!("expected Bias, got {other:?}"),
        };
        let chain = [(0, 1), (1, 2), (2, 3), (3, 4)];
        assert_eq!(
            refusal(&["Input", "Linear", "LIF", "Affine", "Output"], &chain, 3),
            "it feeds the Output '4', and a bias drives a LIF"
        );
        assert_eq!(
            refusal(&["Input", "Affine", "Linear", "LIF"], &chain[..3], 1),
            "it feeds the Linear '2', and a bias drives a LIF directly"
        );
        // the LIF at depth 1 through lif 2, and at depth 0 through 5
        let kinds = ["Input", "Linear", "LIF", "Affine", "LIF", "Linear"];
        let two = "the Inputs reach the LIF '4' it feeds at two depths, and a bias starts at one";
        assert_eq!(
            refusal(&kinds, &[&chain[..], &[(0, 5), (5, 4)]].concat(), 3),
            two
        );
        // the LIF in a loop: depth 1, then 3 around it
        assert_eq!(
            refusal(&kinds, &[&chain[..], &[(4, 5), (5, 2)]].concat(), 3),
            two
        );
        // lif 3 → a → lif 5, and no Input reaches lif 3
        assert_eq!(
            refusal(
                &["Input", "Linear", "LIF", "LIF", "Affine", "LIF"],
                &[(0, 1), (1, 2), (3, 4), (4, 5)],
                4
            ),
            "no Input reaches the LIF '5' it feeds, and a bias starts at its depth"
        );
        // a loop back into the Input: the LIF at depth 0, and then the
        // Input at depth 1
        assert_eq!(
            refusal(&["Input", "Affine", "LIF"], &[(0, 1), (1, 2), (2, 0)], 1),
            "the Inputs reach the LIF '2' it feeds at two depths, and a bias starts at one"
        );
        // a LIF first does not place a bias that feeds an Output too
        assert_eq!(
            refusal(
                &["Input", "Affine", "LIF", "Output"],
                &[(0, 1), (1, 2), (1, 3)],
                1
            ),
            "it feeds the Output '3', and a bias drives a LIF"
        );
        // an Affine no Input reaches, into lif 2 at depth 0 and lif 4 at
        // depth 1: its one bias input has no one start, whichever edge
        // the file names first
        let kinds = ["Input", "Linear", "LIF", "Linear", "LIF", "Affine"];
        for (affine_edges, lifs) in [
            (
                [(5, 2), (5, 4)],
                "the LIFs '2' and '4' it feeds sit at depths 0 and 1",
            ),
            (
                [(5, 4), (5, 2)],
                "the LIFs '4' and '2' it feeds sit at depths 1 and 0",
            ),
        ] {
            assert_eq!(
                refusal(&kinds, &[&chain[..], &affine_edges[..]].concat(), 5),
                format!("{lifs}, and a bias starts at one")
            );
        }
        assert_eq!(
            ConvertError::Bias {
                node: "2".into(),
                why: "it feeds the Output 'out', and a bias drives a LIF".into(),
            }
            .to_string(),
            "node '2' is an Affine whose bias has no place: it feeds the Output 'out', \
             and a bias drives a LIF"
        );
    }

    #[test]
    fn module_name_is_a_snake_case_identifier_or_none() {
        assert_eq!(
            module_name("two_lif_neurons").as_deref(),
            Some("two_lif_neurons")
        );
        assert_eq!(
            module_name("snnTorch-two.layer").as_deref(),
            Some("snntorch_two_layer")
        );
        assert_eq!(module_name("_x").as_deref(), Some("_x"));
        for bad in ["", "_", "2layer", "type", "mod", "crate"] {
            assert_eq!(module_name(bad), None, "{bad:?}");
        }
    }

    /// `freeze` refuses a `name` that is not a `module_name` before it
    /// reads the graph (the JSON here is not JSON), so the header
    /// line's case-name rule never meets it; a `module_name` goes on to
    /// the import.
    #[test]
    fn freeze_refuses_a_name_that_is_not_a_module_name_first() {
        for bad in ["Graph", "3chain", "loop", "_", "", "two-layer"] {
            match freeze(
                b"not json",
                NirImportOptions::default(),
                bad,
                &[(1, None)],
                &[],
            ) {
                Err(FreezeError::Name(n)) => assert_eq!(n, bad),
                Err(e) => panic!("{bad:?}: refused as {e:?}, not by its name"),
                Ok(_) => panic!("{bad:?}: frozen"),
            }
        }
        assert!(
            FreezeError::Name("Graph".to_string())
                .to_string()
                .contains("not a module name")
        );
        assert!(matches!(
            freeze(
                b"not json",
                NirImportOptions::default(),
                "graph",
                &[(1, None)],
                &[]
            ),
            Err(FreezeError::Import(_))
        ));
    }

    #[test]
    fn the_voltage_scale_takes_the_smaller_of_its_two_terms() {
        // the threshold's, 10 / 2, where the range's is 45 / 2
        assert_eq!(volt_scale(&[0.0], &[2.0], Some(&[0.0])), 5.0);
        // the range's, 45 / 5: a reset five thresholds down
        assert_eq!(volt_scale(&[0.0], &[1.0], Some(&[-5.0])), 9.0);
        // the range's, 45 / 6: a leak six thresholds up
        assert_eq!(volt_scale(&[6.0], &[1.0], None), 7.5);
    }

    #[test]
    fn the_voltage_scale_reads_the_node_s_largest_values() {
        // the threshold's, 10 / 2: the largest threshold, the middle one
        assert_eq!(volt_scale(&[0.0; 3], &[1.0, 2.0, 0.5], None), 5.0);
        // the range's, 45 / 5: the largest potential, the middle reset
        assert_eq!(
            volt_scale(&[0.0; 3], &[1.0; 3], Some(&[0.0, -5.0, 0.0])),
            9.0
        );
        // the range's, 45 / 6: the largest potential, the middle leak
        assert_eq!(volt_scale(&[0.0, 6.0, 0.0], &[1.0; 3], None), 7.5);
    }

    #[test]
    fn stamp_renders_valid_json() {
        let c = convert_file(
            &rt_fixture("chain_population.nir"),
            NirImportOptions::default(),
        )
        .expect("converts");
        let s = stamp_json(&c.stamp, std::path::Path::new("x.nir"), None);
        serde_json::from_str::<serde_json::Value>(&s).expect("sidecar is valid JSON");
        assert!(s.contains("\"f32_widened\":[]"));
    }

    /// A graph converted under `--sim-units` at the CLI's default step,
    /// and the options `--freeze` builds it under.
    fn sim_conversion(name: &str) -> (Converted, NirImportOptions) {
        let opts = NirImportOptions {
            dt_us: SIM_DT_US,
            ..NirImportOptions::default()
        };
        let c = convert_file_opts(&fixture(name), opts, true).expect("converts under --sim-units");
        (c, effective_options(opts, true))
    }

    #[test]
    fn the_voltage_scale_s_range_term_binds_under_one() {
        // the range's, 45 / 0.9, under the threshold's, 10 / 0.1: a
        // largest potential under 1 decides too
        assert_eq!(volt_scale(&[0.9], &[0.1], None), 50.0);
    }

    #[test]
    fn a_native_sidecar_carries_no_simulation_keys() {
        let c = convert_file(
            &rt_fixture("chain_population.nir"),
            NirImportOptions::default(),
        )
        .expect("converts");
        let s = stamp_json(&c.stamp, std::path::Path::new("x.nir"), None);
        assert!(s.contains("\"sim_units\":false"), "{s}");
        assert!(!s.contains("current_quanta"), "{s}");
        assert!(!s.contains("volt_scale"), "{s}");
        assert!(!s.contains("\"bias\""), "{s}");
    }

    #[test]
    fn the_sidecar_escapes_every_name() {
        let odd = "a\\b\"c\u{1}d";
        let stamp = Stamp {
            nir_version: odd.to_string(),
            node_census: vec![(odd.to_string(), odd.to_string())],
            f32_datasets: vec![odd.to_string()],
            dt_us: SIM_DT_US,
            resolution: odd,
            sim_units: true,
            volt_scale: vec![(odd.to_string(), 10.0)],
            bias: vec![Bias {
                input: odd.to_string(),
                start: 3,
            }],
        };
        let assembly = Assembly {
            neurons: 1,
            synapses: 0,
            inputs: 1,
            drive_linears: 1,
            stages: 1,
            fused: vec![Fused {
                chain: vec![odd.to_string()],
                scales: vec![0.5],
            }],
            undriven: vec![odd.to_string()],
            multi_linear_gain: false,
            plasticity_frozen: true,
        };
        let s = stamp_json(&stamp, std::path::Path::new(odd), Some(&assembly));
        let v: serde_json::Value = serde_json::from_str(&s).expect("the sidecar is valid JSON");
        assert_eq!(v["source"], odd);
        assert_eq!(v["nir_version"], odd);
        assert_eq!(v["resolution"], odd);
        assert_eq!(v["volt_scale"][odd], 10.0);
        assert_eq!(v["bias"][odd], 3);
        assert_eq!(v["f32_widened"][0], odd);
        assert_eq!(v["nodes"][odd], odd);
        assert_eq!(v["assembly"]["fused"][0]["chain"][0], odd);
        assert_eq!(v["assembly"]["undriven"][0], odd);
    }

    /// The escape stops below the space: 0x1f is written `\u001f`, and
    /// 0x20, a space, is kept.
    #[test]
    fn the_escape_stops_below_the_space() {
        assert_eq!(json_escape("a\u{1f} b"), "a\\u001f b");
    }

    #[test]
    fn freeze_steps_each_run_of_its_drive() {
        let (c, opts) = sim_conversion("community/snnTorch_two_layer.nir");
        let on = freeze(&c.json, opts, "graph", &[(8, Some(&[ONE]))], &[]).expect("freezes");
        let split =
            freeze(&c.json, opts, "graph", &[(4, Some(&[ONE])), (4, None)], &[]).expect("freezes");
        assert_eq!(split.trace, on.trace, "one input in two runs, one trace");
        assert_eq!(split.trace.lines().count(), 9, "the header and 8 rows");
        let off = freeze(
            &c.json,
            opts,
            "graph",
            &[(4, Some(&[ONE])), (4, Some(&[ZERO]))],
            &[],
        )
        .expect("freezes");
        assert!(
            off.module
                .contains("        (4, [1000, 0]),\n        (4, [0, 0]),\n")
        );
        assert!(off.module.contains("pub const STEPS: u32 = 8;"));
        let first = |t: &str| t.lines().take(5).map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(
            first(&off.trace),
            first(&on.trace),
            "the runs step in order: the header, then the first run's 4 rows"
        );
        assert_ne!(off.trace, on.trace, "the second run drives nothing");
    }

    /// The two biased Affines' graph: its Inputs in the encoder's order,
    /// the graph's own two features, then `a/bias` from step 0 and
    /// `c/bias` from step 1.
    #[test]
    fn the_inputs_are_the_graph_s_own_then_each_bias_input_from_its_start() {
        let (c, opts) = sim_conversion("affine_two_layer.nir");
        let g = NirImport::from_json(&c.json, opts).expect("imports");
        let (_, enc, _) = g.build_network().expect("assembles");
        let inputs = Inputs::new(&g, &enc, &c.stamp.bias).expect("the conversion's bias");
        assert_eq!(
            inputs.features(),
            2,
            "the graph's own, the bias inputs aside"
        );
        assert_eq!(inputs.starts(), [0, 1]);
        assert_eq!(
            inputs.at(0, &[ONE, ZERO]),
            [vec![ONE, ZERO], vec![ONE], vec![ZERO]]
        );
        assert_eq!(
            inputs.at(1, &[ZERO, ONE]),
            [vec![ZERO, ONE], vec![ONE], vec![ONE]]
        );
        // each start once, ascending, whatever the bias inputs' order: the
        // conversion lists them in the document order of their Affines,
        // not by depth
        let starts = |a_start: u32, c_start: u32| {
            let bias = [
                Bias {
                    input: "a/bias".into(),
                    start: a_start,
                },
                Bias {
                    input: "c/bias".into(),
                    start: c_start,
                },
            ];
            Inputs::new(&g, &enc, &bias)
                .expect("the graph's bias inputs")
                .starts()
        };
        assert_eq!(starts(2, 1), [1, 2]);
        assert_eq!(starts(1, 1), [1]);
        // a bias input the graph does not hold, or not of one feature
        let bias = |input: &str| {
            Inputs::new(
                &g,
                &enc,
                &[Bias {
                    input: input.into(),
                    start: 0,
                }],
            )
        };
        assert_eq!(bias("x/bias"), Err("x/bias".to_string()));
        assert_eq!(bias("input"), Err("input".to_string()));
        match freeze(
            &c.json,
            opts,
            "graph",
            &[(1, None)],
            &[Bias {
                input: "x/bias".into(),
                start: 0,
            }],
        ) {
            Err(e @ FreezeError::Bias(_)) => assert_eq!(
                e.to_string(),
                "the bias input \"x/bias\" is not an Input of one feature in the graph"
            ),
            other => panic!("expected Bias, got {other:?}"),
        }
    }

    /// A graph with two Inputs of its own, the library's merge fixture:
    /// `at` gives each Input its own values, in Input order.
    #[test]
    fn the_inputs_split_a_drive_in_input_order() {
        let merge = include_bytes!("../../neuralos-snn/tests/nir_fixtures/merge.json");
        let g = NirImport::from_json(merge, NirImportOptions::default()).expect("imports");
        let (_, enc, _) = g.build_network().expect("assembles");
        let inputs = Inputs::new(&g, &enc, &[]).expect("no bias input");
        assert_eq!((inputs.features(), inputs.starts()), (4, vec![]));
        let own = [1, 2, 3, 4].map(Thousandths);
        assert_eq!(inputs.at(0, &own), [own[..2].to_vec(), own[2..].to_vec()]);
    }

    /// `freeze` drives each bias input 0 before its start and 1 from
    /// it: the one run of three steps splits at `c/bias`'s start, and a
    /// run that ends there needs no split. The drive gives the graph's own
    /// two features alone.
    #[test]
    fn freeze_drives_each_bias_input_from_its_start() {
        let (c, opts) = sim_conversion("affine_two_layer.nir");
        let bias = &c.stamp.bias;
        let f = freeze(&c.json, opts, "graph", &[(3, None)], bias).expect("freezes");
        // W·(1, 1) + b into l1, and from step 1 c's bias into l2
        assert!(
            f.module.contains(
                "        (1, [300, 855, -250, 0, 0]),\n        (2, [300, 855, -250, 30, -10]),\n"
            ),
            "{}",
            f.module
        );
        let f = freeze(
            &c.json,
            opts,
            "graph",
            &[(1, Some(&[ONE, ZERO])), (4, Some(&[ZERO, ONE]))],
            bias,
        )
        .expect("freezes");
        assert!(
            f.module.contains(
                "        (1, [550, 105, -500, 0, 0]),\n        (4, [-200, 730, 250, 30, -10]),\n"
            ),
            "{}",
            f.module
        );
        // bias inputs given in any order, a start shared: the run splits
        // at each start inside it, once, in order
        let module = |a_start: u32, c_start: u32| {
            let bias = [
                Bias {
                    input: "a/bias".into(),
                    start: a_start,
                },
                Bias {
                    input: "c/bias".into(),
                    start: c_start,
                },
            ];
            freeze(&c.json, opts, "graph", &[(3, None)], &bias)
                .expect("freezes")
                .module
        };
        let m = module(2, 1);
        assert!(
            m.contains(concat!(
                "        (1, [250, 875, -250, 0, 0]),\n",
                "        (1, [250, 875, -250, 30, -10]),\n",
                "        (1, [300, 855, -250, 30, -10]),\n",
            )),
            "{m}"
        );
        let m = module(1, 1);
        assert!(
            m.contains(concat!(
                "        (1, [250, 875, -250, 0, 0]),\n",
                "        (2, [300, 855, -250, 30, -10]),\n",
            )),
            "{m}"
        );
        assert!(matches!(
            freeze(
                &c.json,
                opts,
                "graph",
                &[(1, Some(&[ONE, ZERO, ONE]))],
                bias
            ),
            Err(FreezeError::Input {
                run: 1,
                given: 3,
                features: 2
            })
        ));
    }

    #[test]
    fn freeze_refuses_a_drive_it_cannot_step_by_name() {
        let (c, opts) = sim_conversion("community/snnTorch_two_layer.nir");
        let refusal = |drive: &[(u32, Option<&[Thousandths]>)]| {
            freeze(&c.json, opts, "graph", drive, &[])
                .map(|_| ())
                .expect_err("refused")
                .to_string()
        };
        assert_eq!(refusal(&[]), "the drive holds no run");
        assert_eq!(
            refusal(&[(1, None), (0, None)]),
            "the drive holds a run of 0 steps"
        );
        assert_eq!(
            refusal(&[(u32::MAX, None), (1, None)]),
            "the drive holds more steps than a trace counts"
        );
        assert_eq!(
            refusal(&[(1, None), (1, Some(&[ONE, ONE]))]),
            "the drive's run 2 gives 2 values; the graph has 1 input feature, one value each"
        );
        // each count the other way: one value, two features
        assert_eq!(
            FreezeError::Input {
                run: 1,
                given: 1,
                features: 2,
            }
            .to_string(),
            "the drive's run 1 gives 1 value; the graph has 2 input features, one value each"
        );
    }

    #[test]
    fn freeze_returns_the_assembly_report_the_sidecar_carries() {
        let (c, opts) = sim_conversion("community/snnTorch_two_layer.nir");
        let f = freeze(&c.json, opts, "graph", &[(1, None)], &[]).expect("freezes");
        assert_eq!(
            f.assembly,
            Assembly {
                neurons: 2,
                synapses: 1,
                inputs: 1,
                drive_linears: 2,
                stages: 1,
                fused: vec![],
                undriven: vec![],
                multi_linear_gain: false,
                plasticity_frozen: true,
            }
        );
        let s = stamp_json(&c.stamp, std::path::Path::new("x.nir"), Some(&f.assembly));
        let v: serde_json::Value = serde_json::from_str(&s).expect("the sidecar is valid JSON");
        assert_eq!(v["assembly"]["drive_linears"], 2);
        assert_eq!(v["assembly"]["multi_linear_gain"], false);
        let bare = stamp_json(&c.stamp, std::path::Path::new("x.nir"), None);
        assert!(!bare.contains("assembly"), "{bare}");
    }

    /// The branch graph of the library's `nir_assembly_gate` (its gate
    /// 1): one Input and two stages, one fused from two tensors, and in
    /// native units the gain note. Its `inputs` and `stages` differ and
    /// its fused stage carries its scales, so the report is held whole,
    /// as `freeze` copies it and as the sidecar writes it.
    #[test]
    fn freeze_and_the_sidecar_carry_a_two_stage_report_whole() {
        let branch = include_bytes!("../../neuralos-snn/tests/nir_fixtures/branch.json");
        let f = freeze(
            branch,
            NirImportOptions::default(),
            "branch",
            &[(1, None)],
            &[],
        )
        .expect("freezes");
        // each tensor's absmax is 1.0, so each scale is 1 / 32767
        let scale = 1.0 / 32767.0;
        assert_eq!(
            f.assembly,
            Assembly {
                neurons: 4,
                synapses: 0,
                inputs: 1,
                drive_linears: 2,
                stages: 2,
                fused: vec![Fused {
                    chain: vec!["l1".to_string(), "l2".to_string()],
                    scales: vec![scale, scale],
                }],
                undriven: vec![],
                multi_linear_gain: true,
                plasticity_frozen: true,
            }
        );
        // the stamp is not under test here, the report is
        let stamp = Stamp {
            nir_version: "1.0.0".to_string(),
            node_census: vec![],
            f32_datasets: vec![],
            dt_us: 1_000,
            resolution: "mv",
            sim_units: false,
            volt_scale: vec![],
            bias: vec![],
        };
        let s = stamp_json(
            &stamp,
            std::path::Path::new("branch.json"),
            Some(&f.assembly),
        );
        let v: serde_json::Value = serde_json::from_str(&s).expect("the sidecar is valid JSON");
        assert_eq!(
            v["assembly"],
            serde_json::json!({
                "neurons": 4,
                "synapses": 0,
                "inputs": 1,
                "drive_linears": 2,
                "stages": 2,
                "fused": [{"chain": ["l1", "l2"], "scales": [scale, scale]}],
                "undriven": [],
                "multi_linear_gain": true,
                "plasticity_frozen": true
            })
        );
    }
}
