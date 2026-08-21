//! NIR HDF5 container (NIR slice 2, feature `hdf5`) — the reference's
//! own `.nir` file format, read through the structured-entry seam.
//!
//! # Layout (pinned verbatim from the reference's own emission)
//!
//! `neuromorphs/NIR` @ `7883c3c` `serialization.write()` emits:
//!
//! ```text
//! version                      scalar vlen-string dataset (any string;
//!                              the pinned install reports
//!                              "1.0.9.dev1+g7883c3c85")
//! node/                        group, type "NIRGraph"
//!     type                     scalar vlen-string, "NIRGraph"
//!     nodes/<name>/            one group per node
//!         type                 scalar vlen-string
//!         <arrays>             gzip (deflate) datasets: f64 params,
//!                              i64 shapes, 2-D f64 weights
//!     edges/                   (N,2) vlen-string dataset, UNCOMPRESSED
//!                              (the reference's write falls through to
//!                              the no-compression branch for lists)
//!     metadata/                group, only when non-empty
//! ```
//!
//! Probe findings of record (2026-08-21, `.nirenv` h5py 3.16.0 +
//! pinned clone): the reference **cannot emit an empty-edges file** —
//! `NIRGraph` construction auto-wires graph-level
//! `input_<n>`/`<n>_output` junction nodes into `edges=[]`. Our own
//! writer emits a 0-row edges dataset for a zero-edge graph and the
//! reader accepts both shapes. SZIP is **not practically emittable**
//! by the pinned toolchain (libaec rejects every legal pixels-per-block
//! against NIR-scale chunk geometry) — the census still rejects it by
//! name if a foreign file presents it.
//!
//! # Filter census (pre-read, per dataset — stated honesty)
//!
//! Every dataset is censused via its creation property list BEFORE any
//! data read. The reference's emissions carry exactly: no filter
//! (strings, edges) or deflate/gzip id 1 (arrays). Our census accepts
//! those two and rejects EVERYTHING else loudly, naming the filter —
//! `lzf` (id 32000) and `szip` (id 4) are documented-legal in the
//! reference's `write(compression=...)`, and we still reject them: a
//! filter we cannot decode is a silent-corruption hazard, not an
//! inconvenience. This rejection is a stated policy, not an
//! HDF5 limitation.
//!
//! # Plugin path
//!
//! The vendored HDF5 build's compiled-in plugin directory never
//! exists; [`ensure_plugin_dir`] structurally points
//! `HDF5_PLUGIN_PATH` at the repo's `tools/hdf5-plugins/` (path
//! baked in by `build.rs`) before any file operation.

use std::collections::HashMap;
use std::fmt;
use std::os::raw::{c_char, c_int, c_uint};
use std::path::{Path, PathBuf};

use hdf5::types::VarLenUnicode;
use hdf5::{Dataset, File as H5File, Group};
use hdf5_sys::h5d::H5Dget_create_plist;
use hdf5_sys::h5p::{H5Pget_filter2, H5Pget_nfilters, H5Pclose};
use neuralos_snn::nir::{
    NirBuilder, NirImport, NirImportOptions, NirLifParams, NirError,
};

/// HDF5's built-in deflate filter id (`H5Z_FILTER_DEFLATE`).
const FILTER_DEFLATE: c_int = 1;

/// Everything that can go wrong on the HDF5 side of a `.nir` read or
/// write. The named boundary between this container module and the
/// [`neuralos_snn::nir`] seam: container failures stay here, seam
/// failures ride [`NirHdfError::Seam`] with the underlying
/// [`neuralos_snn::nir::NirError`] rendered in full.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NirHdfError {
    /// The file could not be opened or its root layout is not a NIR
    /// container (missing `version`, `node`, `node/nodes`, `node/edges`).
    Open(String),
    /// A dataset read failed (I/O or HDF5-level error).
    Read(String),
    /// A dataset's shape/dtype is not what the reference schema emits.
    Shape(String),
    /// A vlen-string dataset could not be read or decoded.
    Strings(String),
    /// The pre-read filter census rejected a dataset: it carries a
    /// filter outside the {none, deflate} policy. Names the dataset
    /// and the filter — loud on purpose.
    Filter { dataset: String, filter: String },
    /// The neuralos-snn seam (quantization, graph validation) rejected
    /// the decoded container. Carries the seam error rendered in full.
    Seam(String),
}

impl fmt::Display for NirHdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(m) => write!(f, "nir hdf5 open/layout: {m}"),
            Self::Read(m) => write!(f, "nir hdf5 read: {m}"),
            Self::Shape(m) => write!(f, "nir hdf5 shape: {m}"),
            Self::Strings(m) => write!(f, "nir hdf5 strings: {m}"),
            Self::Filter { dataset, filter } => write!(
                f,
                "nir hdf5 filter census: dataset '{dataset}' carries filter \
                 '{filter}' — outside the {{none, deflate}} policy (lzf/szip \
                 are legal in the reference ecosystem and rejected here on \
                 purpose: an undecodable filter is a corruption hazard)"
            ),
            Self::Seam(m) => write!(f, "nir hdf5 seam: {m}"),
        }
    }
}

impl std::error::Error for NirHdfError {}

impl From<neuralos_snn::nir::NirError<'_>> for NirHdfError {
    fn from(e: neuralos_snn::nir::NirError<'_>) -> Self {
        Self::Seam(e.to_string())
    }
}

impl From<hdf5::Error> for NirHdfError {
    fn from(e: hdf5::Error) -> Self {
        Self::Read(e.to_string())
    }
}

/// Point `HDF5_PLUGIN_PATH` at a directory that exists, before any
/// HDF5 file operation.
///
/// The vendored build's compiled-in default (`/usr/local/hdf5/lib/plugin`)
/// is absent on dev boxes and CI (ISA R7 finding), and the library wants
/// the variable to name an existing directory. Policy: an existing
/// `HDF5_PLUGIN_PATH` wins (the operator knows something we don't);
/// otherwise the repo's `tools/hdf5-plugins/` — the path `build.rs`
/// bakes in as `NEURALOS_HDF5_PLUGIN_DIR` — is created if absent and
/// installed. Structural, never README-only.
///
/// # Panics
///
/// Never; a failure to create the directory leaves the variable set to
/// the intended path and HDF5 will complain loudly if it cares.
pub fn ensure_plugin_dir() -> PathBuf {
    static INIT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    if let Some(p) = std::env::var_os("HDF5_PLUGIN_PATH") {
        return PathBuf::from(p);
    }
    INIT.get_or_init(|| {
        let dir = PathBuf::from(env!("NEURALOS_HDF5_PLUGIN_DIR"));
        let _ = std::fs::create_dir_all(&dir);
        // SAFETY: exactly once per process (OnceLock), before any HDF5
        // object exists — the documented way to point the vendored
        // build at a plugin dir without touching the compiled-in one.
        std::env::set_var("HDF5_PLUGIN_PATH", &dir);
        dir
    })
    .clone()
}

/// Census one dataset's creation property list BEFORE any data read.
///
/// Accepts: zero filters, or deflate (id 1) only. Rejects everything
/// else — including shuffle/fletcher32 layered on top of deflate —
/// naming the offending filter. Returns the filter list length on
/// success (diagnostics).
///
/// # Errors
///
/// [`NirHdfError::Filter`] naming the dataset and the first
/// out-of-policy filter; [`NirHdfError::Read`] if the property list
/// cannot be inspected.
pub fn census_dataset(ds: &Dataset, dataset: &str) -> Result<usize, NirHdfError> {
    // the raw FFI bypasses the hdf5 crate's per-call global lock — take
    // it ourselves (parallel tests found the race: SIGSEGV under
    // default threading without this)
    hdf5::sync::sync(move || {
        // SAFETY: H5Dget_create_plist on a live dataset id returns a
        // property-list id we own and close below; H5Pget_nfilters /
        // H5Pget_filter2 only read it into caller-sized buffers.
        let plist = unsafe { H5Dget_create_plist(ds.id()) };
        if plist < 0 {
            return Err(NirHdfError::Read(format!(
                "H5Dget_create_plist failed for '{dataset}'"
            )));
        }
        let nfilters = unsafe { H5Pget_nfilters(plist) };
        let mut verdict = Ok(nfilters.max(0) as usize);
        for i in 0..nfilters.max(0) {
            let mut flags: c_uint = 0;
            let mut nvals: usize = 0;
            let mut cd_values = [0u32; 16];
            let mut name_buf = [0u8; 64];
            // SAFETY: buffers above are sized to match the FFI contract;
            // filter_config is optional (null).
            let filter_id = unsafe {
                H5Pget_filter2(
                    plist,
                    i as c_uint,
                    &mut flags,
                    &mut nvals,
                    cd_values.as_mut_ptr(),
                    name_buf.len(),
                    name_buf.as_mut_ptr().cast::<c_char>(),
                    std::ptr::null_mut(),
                )
            };
            let name_end = name_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_buf.len());
            let filter_name = String::from_utf8_lossy(&name_buf[..name_end]).into_owned();
            if filter_id != FILTER_DEFLATE {
                let filter = if filter_name.is_empty() {
                    format!("id {filter_id}")
                } else {
                    filter_name
                };
                verdict = Err(NirHdfError::Filter {
                    dataset: dataset.to_owned(),
                    filter,
                });
                break;
            }
        }
        // SAFETY: closing the property-list id we own; after this no
        // captured buffer is used.
        unsafe { H5Pclose(plist) };
        verdict
    })
}

// ---------------------------------------------------------------------------
// The owned container document (names live here; the seam borrows it)
// ---------------------------------------------------------------------------

/// One decoded node of a `.nir` file, in the file's (alphabetical)
/// member order. Names are owned here and borrowed by
/// [`NirHdfDoc::import`].
#[derive(Debug, Clone, PartialEq)]
pub struct NirHdfNode {
    /// The HDF5 group name (the NIR node name).
    pub name: String,
    /// The node's `type` string, decoded.
    pub kind: NirHdfNodeKind,
    /// `Input`/`Output` shape, dimensions in file order.
    pub shape: Vec<u32>,
    /// `Linear` weights, row-major, `rows × cols`.
    pub weight: Vec<f64>,
    /// `Linear` rows (outputs).
    pub rows: usize,
    /// `Linear` cols (inputs).
    pub cols: usize,
    /// `LIF` per-neuron source-unit parameters.
    pub lif: NirHdfLifParams,
}

/// The four node kinds the slice-1 seam assembles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NirHdfNodeKind {
    Input,
    Output,
    Linear,
    Lif,
}

/// `LIF` per-neuron arrays as the file stores them (source units;
/// quantization is the seam's job). `v_reset_v` `None` = dataset
/// absent = the reference's zeros-on-read semantics.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NirHdfLifParams {
    /// Time constants, seconds.
    pub tau_s: Vec<f64>,
    /// Resistances, ohms.
    pub r_ohm: Vec<f64>,
    /// Leak potentials, volts.
    pub v_leak_v: Vec<f64>,
    /// Thresholds, volts.
    pub v_threshold_v: Vec<f64>,
    /// Reset potentials, volts — `None` when the dataset is absent.
    pub v_reset_v: Option<Vec<f64>>,
}

/// A decoded `.nir` container: everything the reference's `write()`
/// emits, metadata excepted (provenance rides the JSON export channel;
/// HDF5 idempotence is semantic — see the writer's doc).
#[derive(Debug, Clone, PartialEq)]
pub struct NirHdfDoc {
    /// The `version` dataset verbatim (any string — JSON-reader parity;
    /// the pinned install reports `1.0.9.dev1+g7883c3c85`).
    pub version: String,
    /// Nodes in the file's member order (alphabetical link order in
    /// practice — HDF5 b-trees; edges re-resolve by name, so order
    /// carries no semantics).
    pub nodes: Vec<NirHdfNode>,
    /// Edges as endpoint-name pairs, file order.
    pub edges: Vec<(String, String)>,
}

/// Read a `.nir` file: layout walk, filter census per dataset (BEFORE
/// any data read), typed decode into an owned
/// [`NirHdfDoc`]. Quantization does not happen here —
/// [`NirHdfDoc::import`] hands the typed values to the seam.
///
/// # Errors
///
/// [`NirHdfError::Open`] for a non-NIR layout;
/// [`NirHdfError::Filter`] when the census rejects a dataset;
/// [`NirHdfError::Shape`]/[`NirHdfError::Strings`] for schema-shaped
/// reads; [`NirHdfError::Seam`] for an unknown node kind.
pub fn nir_hdf5_read(path: &Path) -> Result<NirHdfDoc, NirHdfError> {
    ensure_plugin_dir();
    let file = H5File::open(path)
        .map_err(|e| NirHdfError::Open(format!("{}: {e}", path.display())))?;

    // root layout: version dataset + node group typed NIRGraph
    let version_ds = file
        .dataset("version")
        .map_err(|e| NirHdfError::Open(format!("missing root 'version': {e}")))?;
    let version = read_string_scalar(&version_ds, "version")?;
    let node = file
        .group("node")
        .map_err(|e| NirHdfError::Open(format!("missing root group 'node': {e}")))?;
    let node_type_ds = node
        .dataset("type")
        .map_err(|e| NirHdfError::Open(format!("missing 'node/type': {e}")))?;
    let node_type = read_string_scalar(&node_type_ds, "node/type")?;
    if node_type != "NIRGraph" {
        return Err(NirHdfError::Open(format!(
            "node/type is '{node_type}', not NIRGraph"
        )));
    }
    let nodes_group = node
        .group("nodes")
        .map_err(|e| NirHdfError::Open(format!("missing 'node/nodes' group: {e}")))?;
    if !node.link_exists("edges") {
        return Err(NirHdfError::Open(
            "missing 'node/edges' dataset (the reference always emits it)".into(),
        ));
    }
    let edges_ds = node.dataset("edges").map_err(|e| {
        NirHdfError::Open(format!("'node/edges' is not a dataset: {e}"))
    })?;

    // nodes, alphabetical member order
    let names = nodes_group
        .member_names()
        .map_err(|e| NirHdfError::Open(format!("cannot list 'node/nodes': {e}")))?;
    let mut nodes = Vec::with_capacity(names.len());
    for name in names {
        nodes.push(read_node(&nodes_group, &name)?);
    }

    let edges = read_edges(&edges_ds)?;
    Ok(NirHdfDoc {
        version,
        nodes,
        edges,
    })
}

/// Census + read one scalar vlen-string dataset.
fn read_string_scalar(ds: &Dataset, path: &str) -> Result<String, NirHdfError> {
    census_dataset(ds, path)?;
    if !ds.shape().is_empty() {
        return Err(NirHdfError::Strings(format!(
            "'{path}' must be a scalar string (shape {:?})",
            ds.shape()
        )));
    }
    let s: VarLenUnicode = ds
        .read_scalar()
        .map_err(|e| NirHdfError::Strings(format!("'{path}': {e}")))?;
    Ok(s.as_str().to_owned())
}

/// Census + read one 1-D f64 dataset (exact dtype enforced — HDF5
/// would otherwise silently convert integers to f64).
fn read_f64_1d(ds: &Dataset, path: &str) -> Result<Vec<f64>, NirHdfError> {
    census_dataset(ds, path)?;
    if !ds.dtype().map_err(|e| NirHdfError::Shape(format!("'{path}': {e}")))?.is::<f64>() {
        return Err(NirHdfError::Shape(format!(
            "'{path}' must be float64 (the reference dtype)"
        )));
    }
    if ds.shape().len() != 1 {
        return Err(NirHdfError::Shape(format!(
            "'{path}' must be 1-D (shape {:?})",
            ds.shape()
        )));
    }
    let arr = ds
        .read_1d::<f64>()
        .map_err(|e| NirHdfError::Shape(format!("'{path}': {e}")))?;
    Ok(arr.to_vec())
}

/// Census + read one 1-D i64 dataset (the `shape` fields).
fn read_shape_1d(ds: &Dataset, path: &str) -> Result<Vec<u32>, NirHdfError> {
    census_dataset(ds, path)?;
    if !ds.dtype().map_err(|e| NirHdfError::Shape(format!("'{path}': {e}")))?.is::<i64>() {
        return Err(NirHdfError::Shape(format!(
            "'{path}' must be int64 (the reference dtype)"
        )));
    }
    let arr = ds
        .read_1d::<i64>()
        .map_err(|e| NirHdfError::Shape(format!("'{path}': {e}")))?;
    let mut out = Vec::with_capacity(arr.len());
    for &d in arr.iter() {
        out.push(u32::try_from(d).map_err(|_| {
            NirHdfError::Shape(format!("'{path}' dimension {d} is not a valid u32"))
        })?);
    }
    Ok(out)
}

/// The seam's MissingField error, rendered with the HDF5 path.
fn missing(name: &str, field: &'static str) -> NirHdfError {
    NirHdfError::Open(format!(
        "'node/nodes/{name}/{field}': {}",
        NirError::MissingField(field)
    ))
}

/// Read one node group: census every dataset, decode by `type`.
fn read_node(nodes_group: &Group, name: &str) -> Result<NirHdfNode, NirHdfError> {
    let group = nodes_group.group(name).map_err(|e| {
        NirHdfError::Open(format!("'node/nodes/{name}' is not a group: {e}"))
    })?;
    let type_ds = group
        .dataset("type")
        .map_err(|e| NirHdfError::Open(format!("'node/nodes/{name}/type': {e}")))?;
    let path = |field: &str| format!("node/nodes/{name}/{field}");
    let node_type = read_string_scalar(&type_ds, &path("type"))?;
    match node_type.as_str() {
        "Input" => {
            let ds = group.dataset("shape").map_err(|_| missing(name, "shape"))?;
            Ok(NirHdfNode {
                name: name.to_owned(),
                kind: NirHdfNodeKind::Input,
                shape: read_shape_1d(&ds, &path("shape"))?,
                weight: Vec::new(),
                rows: 0,
                cols: 0,
                lif: NirHdfLifParams::default(),
            })
        }
        "Output" => {
            let ds = group.dataset("shape").map_err(|_| missing(name, "shape"))?;
            Ok(NirHdfNode {
                name: name.to_owned(),
                kind: NirHdfNodeKind::Output,
                shape: read_shape_1d(&ds, &path("shape"))?,
                weight: Vec::new(),
                rows: 0,
                cols: 0,
                lif: NirHdfLifParams::default(),
            })
        }
        "Linear" => {
            let ds = group.dataset("weight").map_err(|_| missing(name, "weight"))?;
            census_dataset(&ds, &path("weight"))?;
            if !ds
                .dtype()
                .map_err(|e| NirHdfError::Shape(format!("'{}': {e}", path("weight"))))?
                .is::<f64>()
            {
                return Err(NirHdfError::Shape(format!(
                    "'{}' must be float64",
                    path("weight")
                )));
            }
            if ds.shape().len() != 2 {
                return Err(NirHdfError::Shape(format!(
                    "'{}' must be 2-D (shape {:?})",
                    path("weight"),
                    ds.shape()
                )));
            }
            let arr = ds
                .read_2d::<f64>()
                .map_err(|e| NirHdfError::Shape(format!("'{}': {e}", path("weight"))))?;
            let rows = arr.nrows();
            let cols = arr.ncols();
            Ok(NirHdfNode {
                name: name.to_owned(),
                kind: NirHdfNodeKind::Linear,
                shape: Vec::new(),
                weight: arr.iter().copied().collect(),
                rows,
                cols,
                lif: NirHdfLifParams::default(),
            })
        }
        "LIF" => {
            let lif = NirHdfLifParams {
                tau_s: read_f64_1d(
                    &group.dataset("tau").map_err(|_| missing(name, "tau"))?,
                    &path("tau"),
                )?,
                r_ohm: read_f64_1d(
                    &group.dataset("r").map_err(|_| missing(name, "r"))?,
                    &path("r"),
                )?,
                v_leak_v: read_f64_1d(
                    &group
                        .dataset("v_leak")
                        .map_err(|_| missing(name, "v_leak"))?,
                    &path("v_leak"),
                )?,
                v_threshold_v: read_f64_1d(
                    &group
                        .dataset("v_threshold")
                        .map_err(|_| missing(name, "v_threshold"))?,
                    &path("v_threshold"),
                )?,
                v_reset_v: if group.link_exists("v_reset") {
                    Some(read_f64_1d(
                        &group
                            .dataset("v_reset")
                            .map_err(|_| missing(name, "v_reset"))?,
                        &path("v_reset"),
                    )?)
                } else {
                    None // the reference's absent-v_reset zeros semantics
                },
            };
            Ok(NirHdfNode {
                name: name.to_owned(),
                kind: NirHdfNodeKind::Lif,
                shape: Vec::new(),
                weight: Vec::new(),
                rows: 0,
                cols: 0,
                lif,
            })
        }
        other => Err(NirHdfError::Seam(
            NirError::UnsupportedNodeKind(other).to_string(),
        )),
    }
}

/// Census + read the `node/edges` dataset: (N,2) vlen strings
/// (uncompressed in every reference emission). A 0-row or (0,)-shaped
/// dataset is a legal empty edge set (our writer's zero-edge emission;
/// the reference itself cannot emit one — probe finding).
fn read_edges(ds: &Dataset) -> Result<Vec<(String, String)>, NirHdfError> {
    census_dataset(ds, "node/edges")?;
    match ds.shape().as_slice() {
        // empty edge set: our 0-row zero-edge emission (the reference
        // itself cannot emit one — probe finding)
        [0, 2] | [0] => Ok(Vec::new()),
        // the reference's (N,2) emission
        [_, 2] => {
            let arr = ds
                .read_2d::<VarLenUnicode>()
                .map_err(|e| NirHdfError::Strings(format!("'node/edges': {e}")))?;
            Ok(arr
                .outer_iter()
                .map(|row| (row[0].as_str().to_owned(), row[1].as_str().to_owned()))
                .collect())
        }
        shape => Err(NirHdfError::Shape(format!(
            "'node/edges' must be (N,2) strings (shape {shape:?})"
        ))),
    }
}

impl NirHdfDoc {
    /// Hand the typed document to the structured-entry seam
    /// ([`NirBuilder`]): the same `quantize_lif`/`quantize_linear`
    /// contract, records, and errors as the JSON path. Node names
    /// borrow from `self`.
    ///
    /// # Errors
    ///
    /// [`NirHdfError::Seam`] carrying every
    /// [`NirError`](neuralos_snn::nir::NirError) the builder can raise
    /// (quantization hard failures, unknown edge endpoints, duplicate
    /// names/edges, population length mismatches).
    pub fn import(&self, opts: NirImportOptions) -> Result<NirImport<'_>, NirHdfError> {
        let mut bld = NirBuilder::new(opts);
        let mut index: HashMap<&str, usize> = HashMap::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let idx = match node.kind {
                NirHdfNodeKind::Input => bld.add_input(&node.name, &node.shape),
                NirHdfNodeKind::Output => bld.add_output(&node.name, &node.shape),
                NirHdfNodeKind::Linear => {
                    bld.add_linear(&node.name, &node.weight, node.rows, node.cols)
                }
                NirHdfNodeKind::Lif => {
                    let params = NirLifParams {
                        tau_s: &node.lif.tau_s,
                        r_ohm: &node.lif.r_ohm,
                        v_leak_v: &node.lif.v_leak_v,
                        v_threshold_v: &node.lif.v_threshold_v,
                        v_reset_v: node.lif.v_reset_v.as_deref(),
                    };
                    bld.add_lif_population(&node.name, &params)
                }
            }
            .map_err(NirHdfError::from)?;
            index.insert(&node.name, idx);
        }
        for (from, to) in &self.edges {
            let resolve = |name: &str| {
                index
                    .get(name)
                    .copied()
                    .ok_or_else(|| NirHdfError::from(NirError::UnknownEdgeEndpoint(name)))
            };
            let (f, t) = (resolve(from)?, resolve(to)?);
            bld.add_edge(f, t).map_err(NirHdfError::from)?;
        }
        bld.build().map_err(NirHdfError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::{census_dataset, ensure_plugin_dir, NirHdfError};
    use hdf5::types::VarLenUnicode;
    use hdf5::{File, H5Type};

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("neuralos-rt-nir-hdf5-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn census_passes_uncompressed_and_deflate() {
        ensure_plugin_dir();
        let path = scratch("census_ok.h5");
        let f = File::create(&path).unwrap();
        let data = [0.25f64, -1.0, 0.5];
        f.new_dataset_builder()
            .with_data(&data)
            .create("plain")
            .unwrap();
        f.new_dataset_builder()
            .with_data(&data)
            .deflate(4)
            .create("gzipped")
            .unwrap();
        // the reference layout: scalar vlen strings carry no filter
        // (VarLenUnicode is constructed via FromStr — the hdf5 crate's
        // only safe constructor)
        let s: VarLenUnicode = "LIF".parse().unwrap();
        f.new_dataset_builder()
            .with_data(&s)
            .create("a_string")
            .unwrap();
        f.close().unwrap();

        let f = File::open(&path).unwrap();
        assert_eq!(census_dataset(&f.dataset("plain").unwrap(), "plain"), Ok(0));
        assert_eq!(
            census_dataset(&f.dataset("gzipped").unwrap(), "gzipped"),
            Ok(1)
        );
        assert_eq!(
            census_dataset(&f.dataset("a_string").unwrap(), "a_string"),
            Ok(0)
        );
    }

    #[test]
    fn census_rejects_shuffle_loudly_by_name() {
        ensure_plugin_dir();
        let path = scratch("census_shuffle.h5");
        let f = File::create(&path).unwrap();
        let data = [1.0f64, 2.0, 3.0, 4.0];
        f.new_dataset_builder()
            .with_data(&data)
            .shuffle()
            .create("shuffled")
            .unwrap();
        f.close().unwrap();

        let f = File::open(&path).unwrap();
        let err = census_dataset(&f.dataset("shuffled").unwrap(), "node/nodes/x/tau")
            .expect_err("shuffle must be censused out");
        assert_eq!(
            err,
            NirHdfError::Filter {
                dataset: "node/nodes/x/tau".into(),
                filter: "shuffle".into()
            },
            "rejection must carry dataset path + filter name"
        );
    }

    #[test]
    fn census_rejects_deflate_plus_fletcher32_as_a_named_filter() {
        ensure_plugin_dir();
        let path = scratch("census_fletcher.h5");
        let f = File::create(&path).unwrap();
        let data = [1.0f64, 2.0, 3.0, 4.0];
        f.new_dataset_builder()
            .with_data(&data)
            .deflate(4)
            .fletcher32()
            .create("layered")
            .unwrap();
        f.close().unwrap();

        let f = File::open(&path).unwrap();
        let err = census_dataset(&f.dataset("layered").unwrap(), "weight")
            .expect_err("layered filters must be censused out");
        match err {
            NirHdfError::Filter { dataset, filter } => {
                assert_eq!(dataset, "weight");
                // first out-of-policy filter wins; deflate(1) passes,
                // fletcher32(3) is the named rejection
                assert_eq!(filter, "fletcher32");
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn filter_policy_is_documented_by_the_classifier_boundaries() {
        // ids of record: deflate=1 (accepted), shuffle=2, fletcher32=3,
        // szip=4, nbit=5, scaleoffset=6, lzf=32000 (h5py's registered
        // user filter). The classifier is `!= 1 → reject`, pinned by
        // the two file-backed tests above; this test pins the id
        // constants it classifies against.
        assert_eq!(super::FILTER_DEFLATE, 1);
    }

    #[test]
    fn plugin_dir_is_set_to_an_existing_directory() {
        let dir = ensure_plugin_dir();
        assert!(
            std::env::var_os("HDF5_PLUGIN_PATH").is_some(),
            "HDF5_PLUGIN_PATH must be set after ensure_plugin_dir"
        );
        assert!(dir.is_dir(), "{} must exist", dir.display());
    }

    #[test]
    fn error_display_names_the_filter_and_the_policy() {
        let e = NirHdfError::Filter {
            dataset: "node/nodes/lif/tau".into(),
            filter: "lzf".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("node/nodes/lif/tau") && msg.contains("lzf"));
        assert!(msg.contains("deflate"), "the policy must be stated in the error");
    }

    #[test]
    fn seam_error_carries_the_seam_rendering() {
        let seam = neuralos_snn::nir::NirError::TauBelowDt;
        let e: NirHdfError = seam.into();
        let NirHdfError::Seam(msg) = e else {
            panic!("seam error must map to Seam, got {e:?}");
        };
        assert!(msg.starts_with("tau < dt"), "full seam rendering rides along: {msg}");
    }

    // silence dead-code lint for H5Type import used only via macro path
    #[test]
    fn vlen_unicode_is_h5type() {
        fn assert_h5type<T: H5Type>() {}
        assert_h5type::<VarLenUnicode>();
    }
}
