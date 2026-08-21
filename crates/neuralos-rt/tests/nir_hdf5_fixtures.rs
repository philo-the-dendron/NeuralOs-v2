//! HDF5 `.nir` fixture tests (feature `hdf5`) — the reference's own
//! `write()` emissions, same doctrine as the snn JSON fixtures.
//!
//! The population chain reuses `chain_population.json`'s exact values,
//! so the frozen quanta are cross-container comparable: the JSON path
//! and the HDF5 path must produce the SAME records for the same
//! source values.

#![cfg(feature = "hdf5")]

use std::path::PathBuf;

use neuralos_rt::nir_hdf5::{
    nir_hdf5_read, NirHdfError, NirHdfNodeKind,
};
use neuralos_snn::nir::{NirImport, NirImportOptions, NirLif};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/nir_fixtures")
        .join(name)
}

/// The frozen population quanta (identical to the JSON fixture pins):
/// neuron 0 (τ20ms, R100MΩ, −70/−55/−80 mV) and neuron 1
/// (τ30ms, R200MΩ, −65/−50/−75 mV).
const EXPECTED_LIFS: [(u32, i16, i16, i16, u16, u16); 2] = [
    (20_000, -70, -55, -80, 100, 200),
    (30_000, -65, -50, -75, 200, 150),
];

const EXPECTED_WEIGHTS: [i16; 6] = [16384, -32767, 8192, -8192, 32767, 16384];

fn find<'a>(g: &'a NirImport<'a>, name: &str) -> &'a neuralos_snn::nir::NirNode<'a> {
    g.nodes.iter().find(|n| n.name == name).expect("node")
}

#[test]
fn reference_population_chain_imports_per_neuron() {
    let doc = nir_hdf5_read(&fixture("chain_population.nir")).expect("reference file reads");
    assert!(
        doc.version.starts_with("1.0"),
        "version dataset: {}",
        doc.version
    );
    // alphabetical link order — order carries no semantics
    assert_eq!(
        doc.nodes.iter().map(|n| (n.name.as_str(), n.kind)).collect::<Vec<_>>(),
        vec![
            ("input", NirHdfNodeKind::Input),
            ("lif", NirHdfNodeKind::Lif),
            ("linear", NirHdfNodeKind::Linear),
            ("output", NirHdfNodeKind::Output),
        ]
    );
    assert_eq!(
        doc.edges,
        vec![
            ("input".to_owned(), "linear".to_owned()),
            ("linear".to_owned(), "lif".to_owned()),
            ("lif".to_owned(), "output".to_owned()),
        ]
    );

    let g = doc.import(NirImportOptions::default()).expect("imports");
    let lin = find(&g, "linear").linear.expect("linear");
    assert_eq!(&g.weights[lin.weight_offset..lin.weight_offset + 6], &EXPECTED_WEIGHTS);

    let lif_pop = find(&g, "lif").lif.expect("lif population");
    assert_eq!(lif_pop.len, 2);
    let lifs: Vec<NirLif> = (0..2).map(|i| g.lifs[lif_pop.offset + i]).collect();
    for (got, (tau, leak, thr, reset, r, c)) in lifs.iter().zip(EXPECTED_LIFS) {
        assert_eq!(
            (got.tau_us, got.leak_q, got.threshold_q, got.reset_q),
            (tau, leak, thr, reset),
            "per-neuron quanta"
        );
        assert_eq!((got.resistance_mohm, got.capacitance_pf), (r, c));
    }
}

#[test]
fn hdf5_matches_the_json_path_record_for_record() {
    // cross-container parity: same source values through the reference's
    // JSON shape (to_dict, hand-emitted as chain_population.json) and
    // its HDF5 write() must yield identical quantized state
    let h5_doc = nir_hdf5_read(&fixture("chain_population.nir")).expect("read");
    let h5 = h5_doc.import(NirImportOptions::default()).expect("import");
    let json_doc = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../neuralos-snn/tests/nir_fixtures/chain_population.json"),
    )
    .expect("json fixture");
    let js = NirImport::from_json(json_doc.as_bytes(), NirImportOptions::default())
        .expect("json import");

    for name in ["input", "linear", "lif", "output"] {
        let (h, j) = (find(&h5, name), find(&js, name));
        assert_eq!(h.kind, j.kind, "{name}: kind");
        if let Some(hl) = h.linear {
            let jl = j.linear.expect("linear");
            assert_eq!((hl.rows, hl.cols), (jl.rows, jl.cols));
            assert_eq!(
                &h5.weights[hl.weight_offset..hl.weight_offset + hl.rows * hl.cols],
                &js.weights[jl.weight_offset..jl.weight_offset + jl.rows * jl.cols],
                "{name}: weights"
            );
        }
        if let Some(hp) = h.lif {
            let jp = j.lif.expect("lif");
            assert_eq!(hp.len, jp.len);
            for k in 0..hp.len {
                assert_eq!(h5.lifs[hp.offset + k], js.lifs[jp.offset + k], "{name}[{k}]");
            }
        }
    }
    // edges as name pairs
    fn pair<'a>(g: &NirImport<'a>) -> Vec<(&'a str, &'a str)> {
        g.edges
            .iter()
            .map(|&(a, b)| (g.nodes[a as usize].name, g.nodes[b as usize].name))
            .collect()
    }
    assert_eq!(pair(&h5), pair(&js));
}

#[test]
fn uncompressed_variant_imports_identically() {
    let a = nir_hdf5_read(&fixture("chain_population.nir")).expect("gzip");
    let b = nir_hdf5_read(&fixture("chain_population_uncompressed.nir")).expect("none");
    assert_eq!(a, b, "compression changes bytes, never records");
}

#[test]
fn lzf_fixture_rejects_at_the_census_before_any_read() {
    let err = nir_hdf5_read(&fixture("neg_filter_lzf.nir")).expect_err("censused out");
    assert_eq!(
        err,
        NirHdfError::Filter {
            // first array in the alphabetical walk (input's shape) —
            // the census fires before ANY data read
            dataset: "node/nodes/input/shape".into(),
            filter: "lzf".into(),
        }
    );
}

#[test]
fn lzf_rejection_names_the_policy() {
    let err = nir_hdf5_read(&fixture("neg_filter_lzf.nir")).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("lzf") && msg.contains("deflate"), "{msg}");
}

// ---- hand-built container negatives (one error class each) ---------

mod hand {
    use super::*;
    use hdf5::types::VarLenUnicode;
    use hdf5::File;
    use ndarray::Array2;

    pub fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("neuralos-rt-nir-hdf5-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    pub fn vlen(s: &str) -> VarLenUnicode {
        s.parse().unwrap()
    }

    /// Scalar vlen-string dataset, the reference's `create_dataset`
    /// shape for `version`/`type` (with_data would infer an array).
    /// Takes `&Group` — `File` derefs to one.
    pub fn write_str_scalar(parent: &hdf5::Group, name: &str, s: &str) {
        let ds = parent
            .new_dataset_builder()
            .empty::<VarLenUnicode>()
            .shape(())
            .create(name)
            .unwrap();
        ds.write_scalar(&vlen(s)).unwrap();
    }

    /// Minimal NIR container skeleton; `nodes_spec` builds node groups.
    pub fn write_container(
        path: &std::path::Path,
        root_type: Option<&str>,
        with_version: bool,
        edges: &[(&str, &str)],
        nodes_spec: &[(&str, &str)], // (name, type) — arrays added by caller hook
    ) {
        let f = File::create(path).unwrap();
        if with_version {
            write_str_scalar(&f, "version", "1.0.0");
        }
        let node = f.create_group("node").unwrap();
        write_str_scalar(&node, "type", root_type.unwrap_or("NIRGraph"));
        let nodes = node.create_group("nodes").unwrap();
        for (name, kind) in nodes_spec {
            let g = nodes.create_group(name).unwrap();
            write_str_scalar(&g, "type", kind);
        }
        let arr = Array2::from_shape_fn(
            (edges.len(), 2),
            |(i, j)| if j == 0 { vlen(edges[i].0) } else { vlen(edges[i].1) },
        );
        node.new_dataset_builder()
            .with_data(&arr)
            .create("edges")
            .unwrap();
        f.close().unwrap();
    }

    pub fn add_lif_arrays(
        path: &std::path::Path,
        node_name: &str,
        tau: &[f64],
        r: &[f64],
    ) {
        let f = File::open_rw(path).unwrap();
        let g = f
            .group(&format!("node/nodes/{node_name}"))
            .unwrap();
        g.new_dataset_builder()
            .with_data(tau)
            .deflate(4)
            .create("tau")
            .unwrap();
        g.new_dataset_builder()
            .with_data(r)
            .deflate(4)
            .create("r")
            .unwrap();
        g.new_dataset_builder()
            .with_data(&[-0.07f64])
            .deflate(4)
            .create("v_leak")
            .unwrap();
        g.new_dataset_builder()
            .with_data(&[-0.055f64])
            .deflate(4)
            .create("v_threshold")
            .unwrap();
        f.close().unwrap();
    }
}

#[test]
fn missing_version_is_an_open_error() {
    let p = hand::scratch("neg_no_version.h5");
    hand::write_container(&p, Some("NIRGraph"), false, &[("a", "b")], &[("a", "Input")]);
    let err = nir_hdf5_read(&p).expect_err("no version");
    match err {
        NirHdfError::Open(m) => assert!(m.contains("version"), "{m}"),
        other => panic!("wrong error: {other:?}"),
    }
}

#[test]
fn wrong_root_type_is_an_open_error() {
    let p = hand::scratch("neg_root_type.h5");
    hand::write_container(&p, Some("Delay"), true, &[], &[("a", "Input")]);
    let err = nir_hdf5_read(&p).expect_err("root type");
    match err {
        NirHdfError::Open(m) => assert!(m.contains("NIRGraph") && m.contains("Delay"), "{m}"),
        other => panic!("wrong error: {other:?}"),
    }
}

#[test]
fn unknown_node_kind_is_a_seam_error() {
    let p = hand::scratch("neg_cuba.h5");
    hand::write_container(&p, Some("NIRGraph"), true, &[], &[("a", "CubaLIF")]);
    let err = nir_hdf5_read(&p).expect_err("CubaLIF");
    match err {
        NirHdfError::Seam(m) => assert!(m.contains("CubaLIF"), "{m}"),
        other => panic!("wrong error: {other:?}"),
    }
}

#[test]
fn ghost_edge_endpoint_is_a_seam_error() {
    let p = hand::scratch("neg_ghost_edge.h5");
    hand::write_container(
        &p,
        Some("NIRGraph"),
        true,
        &[("input", "ghost")],
        &[("input", "Input")],
    );
    // the input node needs its shape dataset for import to reach edges
    let f = hdf5::File::open_rw(&p).unwrap();
    f.group("node/nodes/input")
        .unwrap()
        .new_dataset_builder()
        .with_data(&[3i64])
        .deflate(4)
        .create("shape")
        .unwrap();
    f.close().unwrap();
    let doc = nir_hdf5_read(&p).expect("reads");
    let err = doc.import(NirImportOptions::default()).expect_err("ghost");
    match err {
        NirHdfError::Seam(m) => assert!(m.contains("ghost"), "{m}"),
        other => panic!("wrong error: {other:?}"),
    }
}

#[test]
fn lif_param_length_mismatch_is_a_seam_badshape() {
    let p = hand::scratch("neg_param_len.h5");
    hand::write_container(&p, Some("NIRGraph"), true, &[], &[("l", "LIF")]);
    hand::add_lif_arrays(&p, "l", &[0.02, 0.03], &[1e8]);
    let doc = nir_hdf5_read(&p).expect("reads at container level");
    let err = doc.import(NirImportOptions::default()).expect_err("param len");
    match err {
        NirHdfError::Seam(m) => assert!(m.contains("LIF param"), "{m}"),
        other => panic!("wrong error: {other:?}"),
    }
}

#[test]
fn integer_weight_dataset_is_a_shape_error() {
    let p = hand::scratch("neg_int_weight.h5");
    hand::write_container(&p, Some("NIRGraph"), true, &[], &[("w", "Linear")]);
    let f = hdf5::File::open_rw(&p).unwrap();
    f.group("node/nodes/w")
        .unwrap()
        .new_dataset_builder()
        .with_data(&ndarray::Array2::from_shape_fn((1, 3), |(i, j)| (i * 3 + j) as i64))
        .deflate(4)
        .create("weight")
        .unwrap();
    f.close().unwrap();
    let err = nir_hdf5_read(&p).expect_err("int weight");
    match err {
        NirHdfError::Shape(m) => assert!(m.contains("float64"), "{m}"),
        other => panic!("wrong error: {other:?}"),
    }
}
