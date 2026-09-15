//! `--freeze` on the reference witness, `two_lif_neurons.nir` under
//! `--sim-units`, with the default drive (feature 1, 150 steps). The
//! library's trace case `two-lif-neurons` runs the same graph on the same
//! drive, so the tool's trace is that file line for line and its module
//! the case's frozen module, the header's kind aside. The module then
//! compiles in a scratch crate against the `no_std` library, and
//! `FixedNetwork`, stepping its arrays, prints the tool's trace. A graph
//! that does not assemble is refused by name, and nothing is written.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A path from the repository root.
fn repo(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

/// A scratch directory under the target dir, empty, with a `src/`.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("src")).expect("the scratch directory");
    dir
}

/// The first line where two texts part, or `None`.
fn first_difference(have: &str, want: &str) -> Option<(usize, String, String)> {
    let (mut a, mut b) = (have.lines(), want.lines());
    for n in 1.. {
        match (a.next(), b.next()) {
            (None, None) => return None,
            (x, y) if x != y => {
                return Some((n, x.unwrap_or("<end>").into(), y.unwrap_or("<end>").into()));
            }
            _ => {}
        }
    }
    unreachable!()
}

/// The scratch crate's program: the frozen module stepped by
/// `FixedNetwork`, its header and rows written by the library's writer,
/// as the firmware steps a frozen case.
const MAIN: &str = r#"use neuralos_snn::fixed::row;
use neuralos_snn::FixedNetwork;

include!("two_lif_neurons.rs");

fn main() {
    use two_lif_neurons as c;
    let mut net = FixedNetwork::<{ c::N }, { c::S }>::new(c::NEURONS, c::SYNAPSES, c::DT_US);
    let mut fired = [false; c::N];
    let mut text = format!("{}\n", c::HEADER);
    let mut step = 0u32;
    for &(count, input) in c::DRIVE {
        for _ in 0..count {
            let time_us = net.time_us();
            net.step(&input, &mut fired);
            if !c::SPIKES_ONLY || fired.contains(&true) {
                row(&mut text, step, time_us, &fired, net.neurons()).expect("a String takes every write");
            }
            step += 1;
        }
    }
    assert_eq!(step, c::STEPS, "the drive covers the run");
    print!("{text}");
}
"#;

#[test]
fn freeze_writes_the_witness_as_the_library_freezes_it() {
    let dir = scratch("freeze-two-lif");
    let module = dir.join("src/two_lif_neurons.rs");
    let out = Command::new(env!("CARGO_BIN_EXE_neuralos-nir2json"))
        .arg("--sim-units")
        .arg("--freeze")
        .arg(&module)
        .arg(repo(
            "crates/neuralos-nir2json/tests/fixtures/community/two_lif_neurons.nir",
        ))
        .arg(dir.join("two_lif_neurons.json"))
        .output()
        .expect("the binary runs");
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // the trace: the case's file, line for line, under a stranger's kind
    let trace = fs::read_to_string(dir.join("src/two_lif_neurons.trace")).expect("the trace");
    let case = fs::read_to_string(repo(
        "crates/neuralos-snn/tests/traces/two-lif-neurons.trace",
    ))
    .expect("the case's trace");
    let want = case.replacen("kind=regression", "kind=stranger", 1);
    if let Some((n, have, want)) = first_difference(&trace, &want) {
        panic!(
            "the tool's trace parts from two-lif-neurons.trace at line {n}:\n  tool: {have}\n  case: {want}"
        );
    }

    // the module: the case's frozen module, one writer, the kind aside
    let frozen =
        fs::read_to_string(repo("crates/neuralos-snn/tests/traces/frozen.rs")).expect("frozen.rs");
    let start = frozen
        .find("pub mod two_lif_neurons {")
        .expect("the frozen case");
    let end = start + frozen[start..].find("\n}\n").expect("its end") + "\n}\n".len();
    let written = fs::read_to_string(&module).expect("the module");
    assert_eq!(
        written,
        frozen[start..end].replacen("kind=regression", "kind=stranger", 1),
        "the tool's module is the frozen case's"
    );

    // the scratch build: the module compiles against the no_std library,
    // and FixedNetwork steps its arrays to the tool's trace
    fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"freeze-scratch\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
             [workspace]\n\n[dependencies]\nneuralos-snn = {{ path = {:?}, default-features = false }}\n",
            repo("crates/neuralos-snn").display().to_string()
        ),
    )
    .expect("the scratch manifest");
    fs::write(dir.join("src/main.rs"), MAIN).expect("the scratch program");
    let run = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--offline", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env(
            "CARGO_TARGET_DIR",
            Path::new(env!("CARGO_TARGET_TMPDIR")).join("freeze-scratch-target"),
        )
        .output()
        .expect("cargo runs");
    assert!(
        run.status.success(),
        "the scratch build: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stepped = String::from_utf8(run.stdout).expect("utf-8");
    if let Some((n, have, want)) = first_difference(&stepped, &trace) {
        panic!(
            "FixedNetwork on the module's arrays parts from the tool's trace at line {n}:\n  fixed: {have}\n  tool:  {want}"
        );
    }
}

#[test]
fn freeze_refuses_by_name_and_writes_nothing() {
    let dir = scratch("freeze-refusals");
    let run = |nir: &str, extra: &[&str], name: &str| {
        Command::new(env!("CARGO_BIN_EXE_neuralos-nir2json"))
            .args(extra)
            .arg("--freeze")
            .arg(dir.join(format!("{name}.rs")))
            .arg(repo(&format!(
                "crates/neuralos-nir2json/tests/fixtures/community/{nir}"
            )))
            .arg(dir.join(format!("{name}.json")))
            .output()
            .expect("the binary runs")
    };

    // no LIF at all: the library's named refusal, exit 2
    let out = run("snnTorch_linear_head.nir", &[], "head");
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("encoder-only"), "names why: {stderr}");

    // one input feature, two values: a usage error, exit 1
    let out = run(
        "two_lif_neurons.nir",
        &["--sim-units", "--input", "1,2"],
        "two",
    );
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--input gives 2 values"), "{stderr}");

    for name in ["head", "two"] {
        for ext in ["rs", "trace", "json", "json.meta.json"] {
            let path = dir.join(format!("{name}.{ext}"));
            assert!(!path.exists(), "nothing written: {}", path.display());
        }
    }
}
