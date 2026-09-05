//! Step-5 CALIBRATION — the lesion/graft positive control
//! (`evidence/step5-calibration/PREREG.md` is the spec; this file is its
//! instrument).
//!
//! The question the control answers (PREREG §1): could the frozen step-5
//! readout have seen a STRUCTURED, function-bearing change in the surgery
//! window, if one had been there? Two arm classes go through the SAME
//! splice machinery as every step-5 arm, each against its own dose-matched
//! null family:
//!
//! - **LESION-h**, h ∈ {0,1,2,3} — window rows `[128h, 128h+128)` × cols
//!   `[0,512)` set to 0. One head's q-projection over the window's input
//!   dims: coherent loss of function, diff classes `(−1→0)`/`(+1→0)` only.
//! - **GRAFT-k**, k ∈ {1,18,35} — window rows `[0,128)` × cols `[0,512)`
//!   replaced by the same rows/cols of `blk.k.attn_q.weight`. Functional
//!   weights in the wrong layer: any of the six off-diagonal classes.
//!
//! Two modes, and the split is the point:
//!
//! - `--measure` WRITES NOTHING. It decodes the base window and each graft
//!   source, and prints every arm file's changed-cell count, its full
//!   per-class composition, and the PREREG §4 range check
//!   ([30,000, 120,000] changed cells). Its output fills §4's table, and
//!   the principal's stamp waits on that table.
//! - `--generate` writes the seven arm files and their 5 nulls each
//!   (`models/cal-*.gguf`) plus `evidence/step5-calibration/arms.txt`. It
//!   REFUSES to run while PREREG §9 reads `_unstamped_` — the stamp is a
//!   mechanical precondition here, not an operator convention (§7 step 3).
//!
//! Seeds come from `evidence/step5-calibration/seeds.txt` by decade
//! (§7: LESION-0 301–310 … GRAFT-35 361–370), first five per arm at N=5,
//! last five reserved for escalation. Never minted at run time.
//!
//! Usage:
//!   cargo run -p neuralos-rt --release --example step5_calibration -- --measure
//!   cargo run -p neuralos-rt --release --example step5_calibration -- --generate
//!
//! Splice/null/guard logic is `harness::{splice_and_verify,
//! dose_matched_null, assert_unbanked}` verbatim — the step-5 family's own
//! implementation, not a second one (no new example by cloning).

use neuralos_rt::harness::{
    assert_unbanked, decode_slice, dose_matched_null, splice_and_verify, tix, ExperimentParams,
};
use neuralos_rt::{GgufFile, GGML_TYPE_Q2_0};
use neuralos_snn::Trit;
use std::collections::BTreeMap;

const BASE: &str = "models/Ternary-Bonsai-4B-Q2_0.gguf";
/// PREP.md pin, re-verified before anything is decoded (PREREG §2).
const BASE_SHA: &str = "4e0bf8b737b0431552f8c2c97695ab7c0cb214c94bcdeb4f5f267e67ddf28b8b";
const PREREG_FILE: &str = "evidence/step5-calibration/PREREG.md";
const SEEDS_FILE: &str = "evidence/step5-calibration/seeds.txt";
const ARMS_FILE: &str = "evidence/step5-calibration/arms.txt";

/// PREREG §4 dose range: outside it an arm is NOT judged (dose no longer
/// comparable to the step-5 record) and §6's denominator adjusts.
const DOSE_MIN: u64 = 30_000;
const DOSE_MAX: u64 = 120_000;
/// N=5 at first pass; the decade's last five are the §7 escalation reserve.
const NULLS_PER_ARM: usize = 5;
/// 4B head width: 4096 output rows / 32 heads.
const HEAD_ROWS: usize = 128;
/// Arm-file count of §3 — 4 lesions + 3 grafts.
const ARM_COUNT: usize = 7;

/// Graft sources, in the §7 seed-decade order. `&'static str` because
/// `ExperimentParams::tensor` is one: the three layers are pre-registered
/// constants, so no name is ever built at run time.
const GRAFT_LAYERS: [(usize, &str); 3] = [
    (1, "blk.1.attn_q.weight"),
    (18, "blk.18.attn_q.weight"),
    (35, "blk.35.attn_q.weight"),
];
const LESION_HEADS: [usize; 4] = [0, 1, 2, 3];

/// Trit bucket names for composition lines (index = `harness::tix`).
const TNAME: [&str; 3] = ["-1", "0", "+1"];

/// sha of a file through the system tool — the `hybrid_invivo` pattern
/// verbatim (the pins of record were produced by `sha256sum`).
fn sha256_of(f: &str) -> String {
    let o = std::process::Command::new("sha256sum")
        .arg(f)
        .output()
        .expect("sha256sum runs");
    String::from_utf8_lossy(&o.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string()
}

/// PREREG §2: the base is re-verified against the PREP.md pin BEFORE it is
/// decoded, in both modes. A drifted base voids everything downstream.
fn verify_base_sha() {
    let sha = sha256_of(BASE);
    assert_eq!(
        sha, BASE_SHA,
        "base {BASE} sha {sha} != PREP.md pin {BASE_SHA} — the base drifted; \
         calibration is void until that is explained"
    );
    println!("base    : {BASE}");
    println!("          sha {sha} == PREREG §2 pin : PASS");
}

/// PREREG §7 step 3: no arm file exists before the stamp. Enforced here
/// rather than trusted to the operator — `--generate` is the irreversible
/// half of this tool.
fn assert_stamped() {
    let text = std::fs::read_to_string(PREREG_FILE)
        .unwrap_or_else(|e| panic!("cannot read {PREREG_FILE}: {e}"));
    let stamp = text
        .split("## 9. Stamp")
        .nth(1)
        .unwrap_or_else(|| panic!("{PREREG_FILE}: § 9 Stamp section not found"))
        .trim();
    assert!(
        !stamp.is_empty() && !stamp.contains("_unstamped_"),
        "{PREREG_FILE} §9 is UNSTAMPED — refusing to generate arm files \
         (PREREG §7 step 3: the principal's stamp precedes --generate)"
    );
    println!("stamp   : {PREREG_FILE} §9 carries a stamp : PASS");
}

/// Confirm every graft source carries the host tensor's exact q2_0 layout
/// BEFORE a single byte of it is read (PREREG §3: a graft is only
/// dose-comparable if the source has the same shape and encoding).
/// Loud on any mismatch — never a silent skip, never a reshape.
fn assert_graft_layouts(p: &ExperimentParams) {
    let buf = std::fs::read(BASE).unwrap_or_else(|e| panic!("cannot read {BASE}: {e}"));
    let f = GgufFile::parse(&buf).expect("GGUF container must parse");
    let find = |name: &str| {
        f.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("tensor {name} not found in {BASE}"))
    };
    let host = find(p.tensor);
    assert_eq!(host.ty, GGML_TYPE_Q2_0, "host tensor must be q2_0");
    println!(
        "layout  : host {} — ty q2_0 · dims {:?} · {} B",
        p.tensor,
        host.dims,
        p.tensor_bytes()
    );
    for (k, name) in GRAFT_LAYERS {
        let t = find(name);
        assert_eq!(
            t.ty, host.ty,
            "graft source {name} is ggml type {} but host {} is {} (q2_0) — \
             layouts differ, REFUSING to read it",
            t.ty, p.tensor, host.ty
        );
        assert_eq!(
            t.dims, host.dims,
            "graft source {name} dims {:?} != host {} dims {:?} — \
             shapes differ, REFUSING to read it",
            t.dims, p.tensor, host.dims
        );
        let data = f.tensor_data(t).expect("graft tensor window in bounds");
        assert_eq!(
            data.len(),
            p.tensor_bytes(),
            "graft source {name} byte window {} != {} (rows × row_bytes) — \
             layout is not the host's, REFUSING to read it",
            data.len(),
            p.tensor_bytes()
        );
        println!(
            "          graft k={k} {name} — ty q2_0 · dims {:?} · {} B : MATCHES HOST",
            t.dims,
            data.len()
        );
    }
}

/// Decode the §2 window (rows 0..512 × cols 0..512) of an arbitrary layer's
/// `attn_q`, through `harness::decode_slice` — the same strict decode path
/// (type / dims / byte-size asserts) the host window goes through. Only the
/// tensor name moves.
fn decode_layer_window(tensor: &'static str, p: &ExperimentParams) -> Vec<Trit> {
    let mut q = p.clone();
    q.tensor = tensor;
    decode_slice(BASE, &q)
}

/// LESION-h: head h's rows of the window, zeroed across all 512 input cols.
fn lesion_patch(base: &[Trit], h: usize, n: usize) -> Vec<Trit> {
    let mut out = base.to_vec();
    let lo = h * HEAD_ROWS * n;
    let hi = lo + HEAD_ROWS * n;
    assert!(
        hi <= out.len(),
        "lesion head {h} outside the {n}×{n} window"
    );
    out[lo..hi].fill(Trit::Zero);
    out
}

/// GRAFT-k: the window's first head rows replaced by layer k's same rows.
fn graft_patch(base: &[Trit], src: &[Trit], n: usize) -> Vec<Trit> {
    assert_eq!(base.len(), src.len(), "graft source window size mismatch");
    let mut out = base.to_vec();
    let cells = HEAD_ROWS * n;
    out[..cells].copy_from_slice(&src[..cells]);
    out
}

/// Changed-cell count and the full per-class composition of `arm` vs `base`.
fn composition(base: &[Trit], arm: &[Trit]) -> (u64, BTreeMap<(usize, usize), u64>) {
    let mut classes: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut cells = 0u64;
    for (b, a) in base.iter().zip(arm.iter()) {
        if a != b {
            *classes.entry((tix(*b), tix(*a))).or_default() += 1;
            cells += 1;
        }
    }
    (cells, classes)
}

fn fmt_comp(classes: &BTreeMap<(usize, usize), u64>) -> String {
    classes
        .iter()
        .map(|((f, t), n)| format!("({}→{} : {n})", TNAME[*f], TNAME[*t]))
        .collect::<Vec<_>>()
        .join(" ")
}

/// PREREG §7 seed decades: one per arm file, arm order = LESION-0..3 then
/// GRAFT-1/18/35, ranges 301–310 … 361–370. Selection is BY POSITION and
/// the map is asserted, the `harness::decade_for` contract shape — that
/// function's ranges are step-5's (201–250) and do not apply here.
fn decade_for_arm(seeds: &[u64], arm_idx: usize) -> Vec<u64> {
    assert!(
        arm_idx < ARM_COUNT,
        "arm index {arm_idx} outside the {ARM_COUNT} pre-registered arms"
    );
    assert!(
        seeds.len() >= 10 * (arm_idx + 1),
        "{SEEDS_FILE} carries only {} entries — arm {arm_idx} needs {}",
        seeds.len(),
        10 * (arm_idx + 1)
    );
    let decade: Vec<u64> = seeds.iter().copied().skip(10 * arm_idx).take(10).collect();
    let (lo, hi) = (301 + 10 * arm_idx as u64, 310 + 10 * arm_idx as u64);
    assert!(
        decade.iter().all(|&s| (lo..=hi).contains(&s)),
        "seed map violation: arm {arm_idx} must use {lo}–{hi}, got {decade:?}"
    );
    decade
}

/// Read the pre-committed seeds. Integer lines only; comments skipped.
fn load_seeds() -> Vec<u64> {
    let text = std::fs::read_to_string(SEEDS_FILE).unwrap_or_else(|e| {
        panic!("cannot read {SEEDS_FILE}: {e} — seeds are pre-committed, never minted")
    });
    let seeds: Vec<u64> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            l.parse::<u64>()
                .unwrap_or_else(|e| panic!("{SEEDS_FILE}: non-integer seed line {l:?}: {e}"))
        })
        .collect();
    assert!(
        seeds.len() >= 10 * ARM_COUNT,
        "{SEEDS_FILE}: need ≥{} seeds (one decade per arm file, §7), found {}",
        10 * ARM_COUNT,
        seeds.len()
    );
    seeds
}

/// The seven arm files of §3, built in §7 seed order: their names and their
/// full patched windows. Built once, used by both modes — `--measure` and
/// `--generate` must never disagree about what an arm IS.
fn build_arms(base: &[Trit], p: &ExperimentParams) -> Vec<(String, Vec<Trit>)> {
    let n = p.n;
    let mut arms: Vec<(String, Vec<Trit>)> = Vec::with_capacity(ARM_COUNT);
    for h in LESION_HEADS {
        arms.push((format!("lesion-h{h}"), lesion_patch(base, h, n)));
    }
    for (k, tensor) in GRAFT_LAYERS {
        let src = decode_layer_window(tensor, p);
        arms.push((format!("graft-k{k}"), graft_patch(base, &src, n)));
    }
    assert_eq!(arms.len(), ARM_COUNT, "§3 defines exactly {ARM_COUNT} arms");
    arms
}

fn in_range(cells: u64) -> bool {
    (DOSE_MIN..=DOSE_MAX).contains(&cells)
}

/// `--measure` — dose accounting for PREREG §4. Writes nothing.
fn measure() {
    let p = ExperimentParams::default();
    println!("=== step-5 calibration: --measure (PREREG §4 dose accounting; WRITES NOTHING) ===");
    verify_base_sha();
    assert_graft_layouts(&p);
    let base = decode_slice(BASE, &p);
    let census = base.iter().fold([0u64; 3], |mut c, t| {
        c[tix(*t)] += 1;
        c
    });
    println!(
        "window  : {} rows 0..{} × cols 0..{} — census (-1/0/+1) {:?} of {} cells",
        p.tensor,
        p.n,
        p.n,
        census,
        base.len()
    );
    println!("range   : PREREG §4 check is [{DOSE_MIN}, {DOSE_MAX}] changed cells\n");

    let arms = build_arms(&base, &p);
    let mut rows: Vec<(String, u64, String, bool)> = Vec::with_capacity(ARM_COUNT);
    for (name, patch) in &arms {
        let (cells, classes) = composition(&base, patch);
        let ok = in_range(cells);
        println!(
            "{name}: {cells} cells · {} · {}",
            fmt_comp(&classes),
            if ok {
                "INSIDE range"
            } else {
                "OUTSIDE range — NOT judged (§4)"
            }
        );
        rows.push((name.clone(), cells, fmt_comp(&classes), ok));
    }

    // The §4 table, ready to paste (the document is filled from this run,
    // never from a recollection of it).
    println!("\n--- PREREG §4 table ---");
    println!("| Arm | changed cells | composition (from→to : n) |");
    println!("|---|---|---|");
    for (name, cells, comp, _) in &rows {
        let label = if let Some(h) = name.strip_prefix("lesion-h") {
            format!("LESION-{h}")
        } else if let Some(k) = name.strip_prefix("graft-k") {
            format!("GRAFT-{k}")
        } else {
            name.clone()
        };
        println!("| {label} | {cells} | {comp} |");
    }
    let excluded: Vec<&str> = rows.iter().filter(|r| !r.3).map(|r| r.0.as_str()).collect();
    println!(
        "\nrange check: {}/{} arms inside [{DOSE_MIN}, {DOSE_MAX}]",
        ARM_COUNT - excluded.len(),
        ARM_COUNT
    );
    if excluded.is_empty() {
        println!("exclusions : none — all seven arms are judgeable at this dose");
    } else {
        println!("exclusions : {excluded:?} — NOT judged; §6 denominators adjust, record in §4");
    }
    println!("\nmeasure done — nothing written. §4 is filled from this output, then the stamp.");
}

/// `--generate` — the seven arm files + 5 nulls each + arms.txt.
/// Refuses while §9 is unstamped.
fn generate() {
    let p = ExperimentParams::default();
    println!("=== step-5 calibration: --generate (PREREG §7 step 4) ===");
    verify_base_sha();
    assert_stamped();
    assert_graft_layouts(&p);
    let base = decode_slice(BASE, &p);
    let seeds = load_seeds();
    let arms = build_arms(&base, &p);

    let mut written: Vec<String> = Vec::with_capacity(ARM_COUNT);
    for (idx, (name, patch)) in arms.iter().enumerate() {
        let (cells, classes) = composition(&base, patch);
        if !in_range(cells) {
            println!(
                "{name}: {cells} cells OUTSIDE [{DOSE_MIN}, {DOSE_MAX}] — EXCLUDED, not generated \
                 (§4); §6 denominator adjusts"
            );
            continue;
        }
        let out = format!("models/cal-{name}.gguf");
        assert_unbanked(&out);
        splice_and_verify(BASE, &out, patch, Some(&base), &p);
        println!(
            "{name}: {cells} cells · {} · S2 clean → {out}",
            fmt_comp(&classes)
        );

        let decade = decade_for_arm(&seeds, idx);
        for &seed in decade.iter().take(NULLS_PER_ARM) {
            let null = dose_matched_null(&base, patch, seed);
            let nout = format!("models/cal-{name}-s{seed}.gguf");
            assert_unbanked(&nout);
            splice_and_verify(BASE, &nout, &null, Some(&base), &p);
            let changed = null.iter().zip(base.iter()).filter(|(a, b)| a != b).count() as u64;
            assert_eq!(
                changed, cells,
                "null s{seed}: dose {changed} != arm {cells}"
            );
            println!("  null s{seed}: {changed} cells (exact dose) · S2 clean → {nout}");
        }
        written.push(name.clone());
    }

    let mut text = String::from(
        "# Step-5 calibration arm files — written by step5_calibration --generate.\n\
         # One arm name per line, in PREREG §7 seed-decade order. The aggregator\n\
         # reads THIS file; no arm list is ever hardcoded (§7 step 6).\n",
    );
    for name in &written {
        text.push_str(name);
        text.push('\n');
    }
    std::fs::write(ARMS_FILE, &text).unwrap_or_else(|e| panic!("cannot write {ARMS_FILE}: {e}"));
    println!(
        "\narms.txt: {} arm(s) → {ARMS_FILE}\ngenerate done — judge chain next (PREREG §7 step 5).",
        written.len()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--measure") if args.len() == 1 => measure(),
        Some("--generate") if args.len() == 1 => generate(),
        _ => {
            eprintln!("usage: step5_calibration --measure | --generate");
            std::process::exit(2);
        }
    }
}
