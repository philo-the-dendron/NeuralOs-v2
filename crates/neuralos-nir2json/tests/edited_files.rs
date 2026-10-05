//! Files one byte edit away from a fixture, from the `.nir` probe of
//! 2026-10-05: each is made here from its seed, so no edited file is
//! tracked, and the edit is asserted on the seed's bytes first, so a
//! regenerated seed fails by name instead of testing nothing. Each runs
//! as the CLI would, natively and under `--sim-units`.

use std::path::{Path, PathBuf};

use neuralos_nir2json::{ConvertError, SIM_DT_US, convert_file_opts};
use neuralos_snn::nir::NirImportOptions;

/// The seed `seed`, its bytes at `at` turned from `old` into `new`,
/// written under the target dir as `name`.
fn edited(seed: &str, at: usize, old: &[u8], new: &[u8], name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(seed);
    let mut bytes = std::fs::read(&path).expect("the seed");
    assert_eq!(
        bytes.get(at..at + old.len()),
        Some(old),
        "{seed} at {at}: not the seed this edit was made on"
    );
    bytes[at..at + old.len()].copy_from_slice(new);
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&out, bytes).expect("the edited file");
    out
}

/// Both conversions of `path`: native, then under `--sim-units` at the
/// CLI's step.
fn both(path: &Path) -> [Result<neuralos_nir2json::Converted, ConvertError>; 2] {
    let sim = NirImportOptions {
        dt_us: SIM_DT_US,
        ..NirImportOptions::default()
    };
    [
        convert_file_opts(path, NirImportOptions::default(), false),
        convert_file_opts(path, sim, true),
    ]
}

#[test]
fn an_input_or_an_output_whose_shape_is_empty_is_refused_at_its_add() {
    // `dims[0]` of `node/nodes/in1/shape`, then of `node/nodes/out/shape`,
    // 1 → 0: the dataset reads as no value, and the JSON the converter
    // wrote held `"shape":[]`, which the library refuses
    for (at, name, add) in [
        (8456, "in1_shape_empty.nir", "add_input"),
        (39376, "out_shape_empty.nir", "add_output"),
    ] {
        let f = edited("merge_f32.nir", at, &[1], &[0], name);
        for r in both(&f) {
            match r {
                Err(ConvertError::Snn { stage, msg }) if stage == add => {
                    assert!(msg.contains("'shape'"), "{name}: {msg}");
                }
                other => panic!("{name}: {other:?}"),
            }
        }
    }
}
