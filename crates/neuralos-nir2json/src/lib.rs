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
//! The supported node kinds are exactly Input, LIF, Linear, Output.
//! Anything else is refused loudly with the node's name and kind —
//! a recorded result, never a partial conversion.
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
    NirBuilder, NirError, NirImport, NirImportOptions, NirLifParams, SIM_CURRENT_QUANTA, nir_export,
};
use neuralos_snn::trace::{self, Kind, Rows, row};

/// Tool version (sidecar stamp).
pub const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");

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
    /// Out-of-subset node kind — named, never partial.
    UnsupportedNode { node: String, kind: String },
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
            Self::UnsupportedNode { node, kind } => write!(
                f,
                "node '{node}' is kind '{kind}' — outside the supported subset \
                 (Input, LIF, Linear, Output); refusing loudly rather than partially converting"
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
///   weights, NIR's `v > v_threshold`, no refractory period.
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
/// ambiguous band rather than guessing).
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
    };

    // Owned name storage: NirNode borrows &'a str for the builder's
    // lifetime — collect ALL names first so no push mutates the Vec
    // while the builder holds borrows into it.
    let owned: Vec<String> = node_names.clone();
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
                let ds = g
                    .dataset("weight")
                    .map_err(|e| ConvertError::Layout(format!("Linear weight: {e}")))?;
                let shape = ds
                    .shape()
                    .map_err(|e| ConvertError::Open(format!("shape: {e}")))?;
                if shape.len() != 2 {
                    return Err(ConvertError::BadData {
                        dataset: format!("node/nodes/{name}/weight"),
                        what: format!("{}-D weight — expected 2-D", shape.len()),
                    });
                }
                let vals = read_param_f64(
                    &ds,
                    &format!("node/nodes/{name}/weight"),
                    &mut stamp.f32_datasets,
                )?;
                let (rows, cols) = (shape[0] as usize, shape[1] as usize);
                if vals.len() != rows * cols {
                    return Err(ConvertError::BadData {
                        dataset: format!("node/nodes/{name}/weight"),
                        what: format!("{} values != {rows}×{cols}", vals.len()),
                    });
                }
                let idx = builder
                    .add_linear(borrowed, &vals, rows, cols)
                    .map_err(|e| ConvertError::Snn {
                        stage: "add_linear",
                        msg: e.to_string(),
                    })?;
                index.insert(name.clone(), idx);
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
    for pair in flat.as_chunks::<2>().0 {
        let resolve = |n: &str| {
            index
                .get(n)
                .copied()
                .ok_or_else(|| ConvertError::Layout(format!("edge names unknown node {n:?}")))
        };
        let (a, b) = (resolve(&pair[0])?, resolve(&pair[1])?);
        builder.add_edge(a, b).map_err(|e| ConvertError::Snn {
            stage: "add_edge",
            msg: e.to_string(),
        })?;
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
    /// The LIF populations and Linear encoders no input reaches.
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
/// `drive`, a step count and one value per input feature in Input order
/// (`None` meaning 1 for every feature), as the currents the graph's
/// encoder gives for it. With it, the trace of that drive on the host,
/// and what the library noted as it assembled the graph.
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
/// encoder's output, one current per neuron.
pub fn freeze(
    json: &[u8],
    opts: NirImportOptions,
    name: &str,
    drive: &[(u32, Option<&[i16]>)],
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

    if drive.is_empty() {
        return Err(FreezeError::Drive("no run"));
    }
    let features: usize = (0..enc.input_count()).map(|i| enc.input_features(i)).sum();
    let mut steps = 0u32;
    let mut runs: Vec<(u32, Vec<i16>)> = Vec::with_capacity(drive.len());
    for (run, &(count, input)) in (1..).zip(drive) {
        if count == 0 {
            return Err(FreezeError::Drive("a run of 0 steps"));
        }
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
            None => vec![1; features],
        };
        let mut per_input: Vec<&[i16]> = Vec::with_capacity(enc.input_count());
        let mut rest = values.as_slice();
        for i in 0..enc.input_count() {
            let (this, others) = rest.split_at(enc.input_features(i));
            per_input.push(this);
            rest = others;
        }
        runs.push((count, enc.encode(&per_input)));
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
        .expect_err("Affine must be refused");
        match &err {
            ConvertError::UnsupportedNode { kind, .. } => assert_eq!(kind, "Affine"),
            other => panic!("expected UnsupportedNode, got {other:?}"),
        }
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
            match freeze(b"not json", NirImportOptions::default(), bad, &[(1, None)]) {
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
                &[(1, None)]
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
        let on = freeze(&c.json, opts, "graph", &[(8, Some(&[1]))]).expect("freezes");
        let split = freeze(&c.json, opts, "graph", &[(4, Some(&[1])), (4, None)]).expect("freezes");
        assert_eq!(split.trace, on.trace, "one input in two runs, one trace");
        assert_eq!(split.trace.lines().count(), 9, "the header and 8 rows");
        let off =
            freeze(&c.json, opts, "graph", &[(4, Some(&[1])), (4, Some(&[0]))]).expect("freezes");
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

    #[test]
    fn freeze_refuses_a_drive_it_cannot_step_by_name() {
        let (c, opts) = sim_conversion("community/snnTorch_two_layer.nir");
        let refusal = |drive: &[(u32, Option<&[i16]>)]| {
            freeze(&c.json, opts, "graph", drive)
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
            refusal(&[(1, None), (1, Some(&[1, 1]))]),
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
        let f = freeze(&c.json, opts, "graph", &[(1, None)]).expect("freezes");
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
        let f =
            freeze(branch, NirImportOptions::default(), "branch", &[(1, None)]).expect("freezes");
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
