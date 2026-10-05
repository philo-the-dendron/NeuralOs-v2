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

/// An edit of a seed: the seed, the offset, the old and the new bytes,
/// and the dataset whose dims it overstates.
type Row = (
    &'static str,
    usize,
    &'static [u8],
    &'static [u8],
    &'static str,
);

/// Both conversions of `path`, native, then under `--sim-units` at the
/// CLI's step: the JSON's length, or the refusal.
fn both(path: &Path) -> [Result<usize, ConvertError>; 2] {
    let sim = NirImportOptions {
        dt_us: SIM_DT_US,
        ..NirImportOptions::default()
    };
    [
        convert_file_opts(path, NirImportOptions::default(), false),
        convert_file_opts(path, sim, true),
    ]
    .map(|r| r.map(|c| c.json.len()))
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

#[test]
fn dims_a_file_overstates_are_refused_before_any_read() {
    // the probe's files, one a read: an Input's shape; a LIF parameter,
    // f32, widened; a weight, 8 s to refuse at `main`; an Affine's
    // bias; a threshold whose count wrapped and converted at `main`.
    // Then by hand: the edges; a weight whose dims' product passes
    // u64, 2^32 by 2^32; a `tau` whose dims, u64's largest, bring the
    // sum past u64
    let rows: [Row; 8] = [
        ("merge_f32.nir", 14179, &[0], &[5], "node/nodes/in2/shape"),
        (
            "chain_population_f32.nir",
            21822,
            &[0, 0, 2, 0],
            &[0xff; 4],
            "node/nodes/lif/v_reset",
        ),
        (
            "parity/fan_in_4_3_2_b0.9.nir",
            14066,
            &[0],
            &[0xd3],
            "node/nodes/0/weight",
        ),
        (
            "affine_bias_length.nir",
            28054,
            &[0, 0, 3, 0, 0, 0, 0, 0],
            &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f],
            "node/nodes/c/bias",
        ),
        (
            "parity/fan_in_4_3_2_b0.9.nir",
            25687,
            &[0],
            &[0x40],
            "node/nodes/1/v_threshold",
        ),
        ("merge_f32.nir", 1868, &[0], &[1], "node/edges"),
        (
            "merge_f32.nir",
            18448,
            &[2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "node/nodes/la/weight",
        ),
        (
            "merge_f32.nir",
            28928,
            &[2, 0, 0, 0, 0, 0, 0, 0],
            &[0xff; 8],
            "node/nodes/lif/tau",
        ),
    ];
    for (seed, at, old, new, dataset) in rows {
        let name = format!("{}_{at}.nir", seed.replace('/', "_"));
        let f = edited(seed, at, old, new, &name);
        for r in both(&f) {
            match r {
                Err(e) => {
                    assert!(
                        matches!(&e, ConvertError::TooLarge { dataset: d, .. } if d == dataset),
                        "{name}: {e:?}"
                    );
                    // the message names the dataset and the bound
                    let m = e.to_string();
                    assert!(m.contains(&format!("'{dataset}'")), "{name}: {m}");
                    assert!(m.contains("4,194,304"), "{name}: {m}");
                }
                Ok(len) => panic!("{name}: converted, {len} bytes"),
            }
        }
    }
}

/// A refusal in our words: its kind and our message up to the reason
/// hdf5-pure gives, so a new hdf5-pure wording moves no test.
fn refusal(e: &ConvertError) -> String {
    match e {
        ConvertError::Open(m) => format!("open {}", m.split(':').next().unwrap_or("")),
        ConvertError::Layout(m) => format!("layout {}", m.split(": ").next().unwrap_or("")),
        ConvertError::UnexpectedKey { path, .. } => format!("key {path}"),
        other => format!("{other:?}"),
    }
}

#[test]
fn files_the_count_cannot_read_are_refused_as_before() {
    // what the count cannot open, it skips, and the census or the
    // decode refuses the file by its own name, as at `main`: a dims
    // hdf5-pure cannot parse; the file's `node` and the graph's `nodes`
    // renamed; a node's header, a dataset's header and a node's heap
    // that hdf5-pure cannot read
    let rows: [(usize, &[u8], &[u8], &str); 6] = [
        (8448, &[1], &[9], "open i64 read"),
        (723, b"e", b"f", "key nodf"),
        (1436, b"s", b"z", "key node/nodez"),
        (7392, &[1], &[9], "layout cannot walk node/nodes"),
        (8424, &[1], &[9], "layout cannot walk node/nodes/in1"),
        (7979, b"P", b"Q", "layout cannot walk node/nodes/in1"),
    ];
    for (at, old, new, want) in rows {
        let name = format!("unreadable_{at}.nir");
        let f = edited("merge_f32.nir", at, old, new, &name);
        for r in both(&f) {
            match r {
                Err(e) => assert_eq!(refusal(&e), want, "{name}: {e}"),
                Ok(len) => panic!("{name}: converted, {len} bytes"),
            }
        }
    }
}
