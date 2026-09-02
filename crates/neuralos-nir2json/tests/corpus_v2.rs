//! Corpus v2 — the amended spike-gate battery (ISA 2026-08-23).
//!
//! a. f32 cast-down twins: the tool widens to f64 internally; oracle =
//!    compare the twin's converted values against the f64 origin at
//!    f32-precision tolerance (rt CANNOT oracle f32 — it hard-rejects
//!    non-f64, code-pinned in nir_hdf5.rs).
//! b. lzf refusal by name (unit-tested in lib; re-asserted through the
//!    CLI binary here — exit 2, message names lzf).
//! c. Large-graph JSON-blowup check: a 1024×1024 Linear converts in
//!    bounded time with a well-formed result.
//! d. Stranger emissions (the SEALED nomination): full path for the
//!    in-subset trio; loud named rejection for the wall probes.

use std::path::Path;
use std::process::Command;

use neuralos_nir2json::{ConvertError, convert_file, convert_file_opts};
use neuralos_snn::nir::NirImportOptions;

fn fixture(name: &str) -> std::path::PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    assert!(p.exists(), "fixture missing: {}", p.display());
    p
}

fn rt_fixture(name: &str) -> std::path::PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../neuralos-rt/tests/nir_fixtures")
        .join(name);
    assert!(p.exists(), "rt fixture missing: {}", p.display());
    p
}

/// CI-portable scratch path (no /tmp/opencode — that's one harness's
/// local convention): the system temp dir + a test-unique name.
/// Stale files from an earlier same-pid run are swept best-effort.
fn tmp(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("nir2json-{name}-{}", std::process::id()));
    let _ = std::fs::remove_file(format!("{}.meta.json", p.display()));
    let _ = std::fs::remove_file(&p);
    p
}

/// Recursively compare two JSON documents at f32-precision tolerance:
/// numbers must satisfy |a−b| ≤ 8 · f32::EPSILON · max(1, |a|, |b|)
/// (f32 widening error is ≤ 1 ulp; the factor covers the dequantized
/// product paths). Everything else must be equal.
fn assert_close(a: &serde_json::Value, b: &serde_json::Value, path: &str) {
    match (a, b) {
        (serde_json::Value::Number(x), serde_json::Value::Number(y)) => {
            let (x, y) = (x.as_f64().expect("finite"), y.as_f64().expect("finite"));
            let tol = 8.0 * f32::EPSILON as f64 * x.abs().max(y.abs()).max(1.0);
            assert!(
                (x - y).abs() <= tol,
                "{path}: {x} vs {y} exceeds f32 tolerance {tol}"
            );
        }
        (serde_json::Value::Object(m), serde_json::Value::Object(n)) => {
            assert_eq!(m.len(), n.len(), "{path}: key count");
            for (k, v) in m {
                assert_close(
                    v,
                    n.get(k).unwrap_or_else(|| panic!("{path}/{k} missing")),
                    &format!("{path}/{k}"),
                );
            }
        }
        (serde_json::Value::Array(x), serde_json::Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: len");
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                assert_close(u, v, &format!("{path}[{i}]"));
            }
        }
        _ => assert_eq!(a, b, "{path}: structural mismatch"),
    }
}

// ---- (a) f32 twins -------------------------------------------------------

#[test]
fn f32_twin_matches_f64_origin_within_f32_tolerance() {
    for name in ["chain_population", "merge"] {
        let origin = convert_file(
            &rt_fixture(&format!("{name}.nir")),
            NirImportOptions::default(),
        )
        .expect("f64 origin converts");
        let twin = convert_file(
            &fixture(&format!("{name}_f32.nir")),
            NirImportOptions::default(),
        )
        .expect("f32 twin converts");
        assert!(
            !twin.stamp.f32_datasets.is_empty(),
            "{name}: the twin must actually carry f32 datasets (else the test tests nothing)"
        );
        let o: serde_json::Value = serde_json::from_slice(&origin.json).unwrap();
        let t: serde_json::Value = serde_json::from_slice(&twin.json).unwrap();
        // metadata blocks carry provenance/quant records whose floats
        // also ride the tolerance — compare the WHOLE document.
        assert_close(&t, &o, name);
    }
}

#[test]
fn f32_twin_still_imports_into_snn() {
    let twin = convert_file(
        &fixture("chain_population_f32.nir"),
        NirImportOptions::default(),
    )
    .expect("converts");
    neuralos_snn::nir::NirImport::from_json(&twin.json, NirImportOptions::default())
        .expect("the widened document is first-class import material");
}

// ---- (b) lzf refusal through the binary ----------------------------------

#[test]
fn binary_refuses_lzf_with_named_exit_2() {
    let out = Command::new(env!("CARGO_BIN_EXE_neuralos-nir2json"))
        .arg(fixture("neg_filter_lzf.nir"))
        .arg(tmp("lzf-refusal"))
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(2), "exit 2 on refusal");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("lzf"), "message names the filter: {stderr}");
}

// ---- (c) large-graph blowup check ----------------------------------------

#[test]
fn big_linear_converts_bounded() {
    let t0 = std::time::Instant::now();
    let c = convert_file(&fixture("big_linear.nir"), NirImportOptions::default())
        .expect("the 1M-weight graph converts");
    let elapsed = t0.elapsed();
    // 1024×1024 f64 weights + one Input + one Output. The honest
    // bound: generous (this box also runs a live burn), but finite —
    // a blowup (quadratic buffer growth, OOM loop) cannot hide under it.
    assert!(elapsed.as_secs() < 120, "bounded time: {elapsed:?}");
    assert!(
        c.json.len() > 1_000_000,
        "the JSON really carries ~1M weight cells ({} B)",
        c.json.len()
    );
    let v: serde_json::Value = serde_json::from_slice(&c.json).expect("well-formed");
    assert!(
        v["node"]["nodes"]["linear"]["weight"].is_array(),
        "linear weights present"
    );
    // and it imports — the full path, at scale
    neuralos_snn::nir::NirImport::from_json(&c.json, NirImportOptions::default())
        .expect("imports at scale");
}

// ---- (d) the SEALED stranger nomination ----------------------------------
//
// EMPIRICAL OUTCOME (the corpus-v2 finding of record): the paper
// corpus's LIF files carry the simulation-unit convention (r = 1–24 Ω)
// — below the substrate's biological floor (r ≥ 1 MΩ). snnTorch
// itself exports r = 1.0 (its LIF is dimensionless). The wall is the
// convention boundary, recorded via named rejections below; the full
// path is carried by the pre-authorized fallback emission (ladder
// rung ii): a Linear-only snnTorch head — no LIF, no r, REAL f32
// weights (torch's default), the audience's own emitter pipeline.

#[test]
fn stranger_smoke_two_lif_neurons_is_detected_sim_units_both_walls() {
    // Without the flag: the DOUBLE wall, named together — r fires
    // first in quantize_lif's order, but the message must carry the
    // voltage wall too (the fact-check finding).
    let err = convert_file(
        &fixture("community/two_lif_neurons.nir"),
        NirImportOptions::default(),
    )
    .expect_err("sim-unit file refused natively");
    match &err {
        ConvertError::SimUnits { node, r_ohm } => {
            assert_eq!(node, "lif1");
            assert!((r_ohm - 1.0).abs() < 1e-12);
        }
        other => panic!("expected SimUnits, got {other:?}"),
    }
    let msg = err.to_string();
    assert!(msg.contains("--sim-units"), "names the flag: {msg}");
    assert!(
        msg.contains('Ω') || msg.contains("MΩ"),
        "names the r wall: {msg}"
    );
    assert!(msg.contains("mV"), "names the voltage wall: {msg}");
}

#[test]
fn stranger_smoke_two_lif_with_sim_units_pins_discrete_centi() {
    // THE transform acceptance: DISCRETE output pins on the centi
    // grid, per the amended B-brief. two_lif: lif1 (tau 10 ms, r 1.0,
    // leak 1.2, th 1.0, v_reset ABSENT → defaulted zeros) and lif2
    // (tau 10 ms, r 1.0, leak 0.0, th 20.0).
    let c = convert_file_opts(
        &fixture("community/two_lif_neurons.nir"),
        NirImportOptions::default(),
        true,
    )
    .expect("the transform converts the smoke file");
    assert!(c.stamp.sim_units);
    assert_eq!(c.stamp.resolution, "centi-mv", "centi grid FORCED");
    // centi opts ride the document; import under them for the pins
    let g = neuralos_snn::nir::NirImport::from_json(
        &c.json,
        neuralos_snn::nir::NirImportOptions::new(
            1_000,
            neuralos_snn::VoltageResolution::CentiMillivolt,
        ),
    )
    .expect("imports under centi");
    assert_eq!(
        g.lifs.len(),
        2,
        "lif1, lif2 (groups() order: input, lif1, lif2, …)"
    );
    let l1 = &g.lifs[0];
    assert_eq!(l1.tau_us, 10_000, "10 ms");
    assert_eq!(l1.resistance_mohm, 1_000, "r 1.0 × 1000 → 1000 MΩ");
    assert_eq!(l1.leak_q, 120, "1.2 mV → 120 centi quanta");
    assert_eq!(l1.threshold_q, 100, "1.0 mV → 100 centi quanta");
    assert_eq!(l1.reset_q, 0, "v_reset absent → zeros semantics");
    // v_reset_defaulted: the in-memory flag does not survive snn's
    // JSON round trip (spine behavior); the DURABLE pin is the
    // provenance block — assert it on the artifact itself.
    let doc: serde_json::Value = serde_json::from_slice(&c.json).unwrap();
    assert_eq!(
        doc["node"]["nodes"]["lif1"]["metadata"]["neuralos"]["provenance"]["v_reset_defaulted"],
        serde_json::Value::Bool(true),
        "defaulted-ness round-trips in provenance"
    );
    // THE TONIC WITNESS (source quirk carried faithfully, not fixed):
    // leak 1.2 mV > threshold 1.0 mV — this neuron spikes at rest by
    // construction of the source model. Ours must say so too.
    assert!(
        l1.leak_q > l1.threshold_q,
        "lif1 tonically spikes at rest (source property, faithfully imported)"
    );
    let l2 = &g.lifs[1];
    assert_eq!(l2.tau_us, 10_000);
    assert_eq!(l2.resistance_mohm, 1_000);
    assert_eq!(l2.leak_q, 0);
    assert_eq!(l2.threshold_q, 2_000, "20 mV → 2000 centi quanta");
}

#[test]
fn stranger_emitter_skew_rockpool_transform_pins_the_f32_path() {
    // rockpool: f32 datasets (real emitter float32) + the 0.1-threshold
    // family — the file that DIES at ThresholdZero on the default grid
    // and lives on centi. Both stamps in one stranger.
    let err = convert_file(
        &fixture("community/lif_rockpool.nir"),
        NirImportOptions::default(),
    )
    .expect_err("detected sim-units");
    assert!(matches!(err, ConvertError::SimUnits { .. }));

    let c = convert_file_opts(
        &fixture("community/lif_rockpool.nir"),
        NirImportOptions::default(),
        true,
    )
    .expect("the transform converts rockpool");
    assert!(
        c.stamp.f32_datasets.iter().any(|d| d.ends_with("/tau")),
        "f32 widening stamped alongside sim-units: {:?}",
        c.stamp.f32_datasets
    );
    let g = neuralos_snn::nir::NirImport::from_json(
        &c.json,
        neuralos_snn::nir::NirImportOptions::new(
            1_000,
            neuralos_snn::VoltageResolution::CentiMillivolt,
        ),
    )
    .expect("imports");
    let lif = &g.lifs[0];
    assert_eq!(lif.tau_us, 2_500, "2.5 ms");
    assert_eq!(
        lif.resistance_mohm, 24_020,
        "24.019737 (f32) × 1000 → 24020 MΩ"
    );
    assert_eq!(
        lif.threshold_q, 10,
        "0.1 mV → 10 centi quanta (ThresholdZero on mV grid)"
    );
    assert_eq!(lif.leak_q, 0);
}

#[test]
fn stranger_fallback_snntorch_head_completes_full_path_f32() {
    // The audience's emitter, the audience's dtype: torch exports f32
    // weights by default — this is the REAL f32 stranger case.
    let c = convert_file(
        &fixture("community/snnTorch_linear_head.nir"),
        NirImportOptions::default(),
    )
    .expect("the snnTorch head completes the full path");
    assert!(
        c.stamp.f32_datasets.iter().any(|d| d.ends_with("/weight")),
        "the f32 widening is stamped: {:?}",
        c.stamp.f32_datasets
    );
    let g = neuralos_snn::nir::NirImport::from_json(&c.json, NirImportOptions::default())
        .expect("snn imports the snnTorch emission");
    assert_eq!(g.nodes.len(), 3, "Input, Linear, Output");
    assert_eq!(g.edges.len(), 2);
    assert_eq!(c.stamp.nir_version, "1.0.8");
}

#[test]
fn stranger_emitter_skew_norse_is_a_named_wall() {
    // lif_norse carries an Affine node (Linear+bias) — outside the
    // four-kind subset. The loud named rejection IS the recorded
    // result (anti-circularity leg: the emission is genuinely not
    // ours, and its refusal is honest).
    let err = convert_file(
        &fixture("community/lif_norse.nir"),
        NirImportOptions::default(),
    )
    .expect_err("Affine is out of subset");
    match &err {
        ConvertError::UnsupportedNode { node, kind } => {
            assert_eq!(kind, "Affine");
            assert!(!node.is_empty(), "the node is named");
        }
        other => panic!("expected UnsupportedNode, got {other:?}"),
    }
}

#[test]
fn stranger_wall_probes_are_all_named_rejections() {
    for name in [
        "cnn_sinabs.nir",
        "braille_noDelay_bias_zero.nir",
        "braille_noDelay_noBias_subtract.nir",
        "braille_noDelay_bias_zero_subgraph.nir",
        "braille_noDelay_noBias_subtract_subgraph.nir",
    ] {
        let err = convert_file(
            &fixture(&format!("community/{name}")),
            NirImportOptions::default(),
        )
        .err()
        .unwrap_or_else(|| panic!("{name}: expected a refusal"));
        match &err {
            ConvertError::UnsupportedNode { kind, .. } => {
                // recorded, not asserted-to-a-fixed-kind: the probes
                // carry several exotic kinds (Conv, Sum, RNN, …) —
                // what matters is the refusal names one, loudly.
                assert!(!kind.is_empty(), "{name}: kind named");
                eprintln!("wall probe {name}: refused on kind '{kind}' (recorded result)");
            }
            other => panic!("{name}: expected UnsupportedNode, got {other:?}"),
        }
    }
}

#[test]
fn binary_end_to_end_smoke_happy_and_sad() {
    let e2e = tmp("e2e.json");
    let out = Command::new(env!("CARGO_BIN_EXE_neuralos-nir2json"))
        .arg(fixture("community/snnTorch_linear_head.nir"))
        .arg(&e2e)
        .output()
        .expect("binary runs");
    assert_eq!(
        out.status.code(),
        Some(0),
        "happy path exits 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // the transform through the binary: two_lif + --sim-units exits 0
    // and stamps the sidecar
    let sim_path = tmp("sim.json");
    let sim = Command::new(env!("CARGO_BIN_EXE_neuralos-nir2json"))
        .arg("--sim-units")
        .arg(fixture("community/two_lif_neurons.nir"))
        .arg(&sim_path)
        .output()
        .expect("binary runs");
    assert_eq!(
        sim.status.code(),
        Some(0),
        "sim-units happy path exits 0: {}",
        String::from_utf8_lossy(&sim.stderr)
    );
    let sidecar_path = std::path::PathBuf::from(format!("{}.meta.json", sim_path.display()));
    let sidecar = std::fs::read_to_string(&sidecar_path).unwrap();
    assert!(sidecar.contains("\"sim_units\":true"), "stamped: {sidecar}");
    assert!(
        sidecar.contains("\"resolution\":\"centi-mv\""),
        "centi recorded: {sidecar}"
    );

    // without the flag: exit 2, message names the flag + both walls
    let refused = tmp("should-not-exist.json");
    let sad = Command::new(env!("CARGO_BIN_EXE_neuralos-nir2json"))
        .arg(fixture("community/two_lif_neurons.nir"))
        .arg(&refused)
        .output()
        .expect("binary runs");
    assert_eq!(sad.status.code(), Some(2), "sim-unit refusal exits 2");
    let stderr = String::from_utf8_lossy(&sad.stderr);
    assert!(
        stderr.contains("--sim-units"),
        "suggests the flag: {stderr}"
    );
    assert!(stderr.contains("mV"), "names the voltage wall: {stderr}");
    assert!(!refused.exists(), "no partial output");
    assert!(e2e.exists());
    assert!(
        e2e.exists() && std::path::PathBuf::from(format!("{}.meta.json", e2e.display())).exists(),
        "sidecar written"
    );
    assert!(
        !std::path::PathBuf::from(format!("{}.meta.json", refused.display())).exists(),
        "no partial sidecar"
    );
}
