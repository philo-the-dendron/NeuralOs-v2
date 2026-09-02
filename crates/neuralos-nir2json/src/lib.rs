//! neuralos-nir2json — the inbound bridge: a stranger's NIR `.nir`
//! (HDF5) file converted into the JSON schema `neuralos-snn`'s
//! [`nir_import`] consumes.
//!
//! # Design (single-writer by construction)
//!
//! The tool never writes JSON itself: HDF5 is read via `hdf5-pure`
//! (pure Rust — byteorder + miniz_oxide, no C toolchain), the typed
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
//! [`nir_import`]: neuralos_snn::nir::nir_import

use std::fmt;
use std::path::Path;

use hdf5_pure::{DType, Dataset, File, Group, VlenStringReadOptions};
use neuralos_snn::nir::{NirBuilder, NirError, NirImportOptions, NirLifParams, nir_export};

/// Tool version (sidecar stamp).
pub const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");

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
                 biological scale (MΩ, mV). Re-run with `--sim-units`: the stamped, exact \
                 convention transform (r × 1000 → MΩ, voltages read as mV, centi grid forced)"
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
/// (amended B-brief, 2026-08-24). When `sim_units` is set, LIF
/// parameters are interpreted in the ecosystem's simulation-unit
/// convention via the THREE-element exact transform:
///
/// - `r × 1000` — dimensionless/Ω → MΩ (r = 1.0 → 1000 MΩ; the
///   substrate's coupling is the product `r·I` only, so the dynamics
///   are preserved when the stranger drives their dimensionless
///   currents as µA numerics);
/// - voltages read AS mV (numerically unchanged; the native reading
///   of e.g. `v_threshold = 0.1` as 0.1 V = 100 mV exceeds the
///   [−100, +50] mV membrane — the wall is DOUBLE: r fires first in
///   quantize_lif's check order, voltages refuse right behind it);
/// - CENTI GRID FORCED — on the default mV grid, the 0.1-threshold
///   family (rockpool, norse, small-β snnTorch) dies at ThresholdZero
///   (0.1 mV quantum rounds to 0); centi (scale 100) gives 10 quanta.
///
/// The transform is an INTERPRETIVE ACT (declaring dimensionless =
/// mV/µA-numeric is a convention assumption about the stranger's
/// intent) — hence opt-in, sidecar-stamped, never silent.
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
    let effective = if sim_units {
        neuralos_snn::nir::NirImportOptions::new(
            opts.dt_us,
            neuralos_snn::VoltageResolution::CentiMillivolt,
        )
    } else {
        opts
    };
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
                // apply the three-element transform.
                let (r, v_leak, v_threshold, v_reset) = if r.iter().any(|&x| x < 1e6) {
                    if !sim_units {
                        return Err(ConvertError::SimUnits {
                            node: name.clone(),
                            r_ohm: r[0],
                        });
                    }
                    let t = |v: Vec<f64>| v.into_iter().map(|x| x * 1e-3).collect();
                    let r: Vec<f64> = r.into_iter().map(|x| x * 1e9).collect();
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
    for pair in flat.chunks_exact(2) {
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

fn resolution_name(opts: NirImportOptions) -> &'static str {
    match opts.resolution {
        neuralos_snn::VoltageResolution::Millivolt => "mv",
        neuralos_snn::VoltageResolution::CentiMillivolt => "centi-mv",
    }
}

/// Render the sidecar stamp document (hand-built — the tool writes
/// exactly two small documents, this and the export's canonical bytes;
/// no serializer dependency).
#[must_use]
pub fn stamp_json(s: &Stamp, source: &Path) -> String {
    let mut out = String::from("{\"tool\":\"neuralos-nir2json\",\"version\":\"");
    out.push_str(TOOL_VERSION);
    out.push_str("\",\"source\":\"");
    out.push_str(
        &source
            .display()
            .to_string()
            .replace('\\', "\\\\")
            .replace('"', "\\\""),
    );
    out.push_str("\",\"nir_version\":\"");
    out.push_str(&s.nir_version.replace('\\', "\\\\").replace('"', "\\\""));
    out.push_str("\",\"dt_us\":");
    out.push_str(&s.dt_us.to_string());
    out.push_str(",\"resolution\":\"");
    out.push_str(s.resolution);
    out.push_str("\",\"sim_units\":");
    out.push_str(if s.sim_units { "true" } else { "false" });
    out.push_str(",\"f32_widened\":[");
    for (i, d) in s.f32_datasets.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("\"{}\"", d.replace('"', "\\\"")));
    }
    out.push_str("],\"nodes\":{");
    for (i, (name, kind)) in s.node_census.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("\"{}\":\"{}\"", name.replace('"', "\\\""), kind));
    }
    out.push_str("}}");
    out
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
    fn stamp_renders_valid_json() {
        let c = convert_file(
            &rt_fixture("chain_population.nir"),
            NirImportOptions::default(),
        )
        .expect("converts");
        let s = stamp_json(&c.stamp, std::path::Path::new("x.nir"));
        serde_json::from_str::<serde_json::Value>(&s).expect("sidecar is valid JSON");
        assert!(s.contains("\"f32_widened\":[]"));
    }
}
