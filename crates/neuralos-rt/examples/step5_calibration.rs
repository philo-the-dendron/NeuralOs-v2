//! Step-5 CALIBRATION — the lesion/graft positive control
//! (`evidence/step5-calibration/PREREG.md` v2 is the spec; this file is
//! its instrument).
//!
//! The question (PREREG §1): could the frozen step-5 readout have seen a
//! STRUCTURED, function-bearing change in the surgery window, if one had
//! been there? Eight arm files go through the SAME splice machinery as
//! every step-5 arm, each against its own dose-matched null families:
//!
//! - **LESION-h**, h ∈ {0,1,2,3} — window rows `[128h, 128h+128)` × cols
//!   `[0,512)` set to 0. One head's q-projection over the window's input
//!   dims: coherent, localized loss of function; classes `(−1→0)`/`(+1→0)`.
//! - **GRAFT-k**, k ∈ {1,18,35} — window rows `[0,128)` × cols `[0,512)`
//!   replaced by the same rows/cols of `blk.k.attn_q.weight`. Trained
//!   weights in the wrong layer; any of the six off-diagonal classes.
//! - **IDENTITY** — the base slice spliced onto the base through the same
//!   pipeline. 0 changed cells, sha ≡ base: the contamination tripwire.
//!
//! Null families (§3): **SCAT** shuffles the arm's diff over the whole
//! 512×512 window (step 5's own construction); **LOCAL** shuffles it
//! inside the arm's own 128×512 block and embeds it there, separating
//! structure from concentration. Grafts get both — LOCAL is their primary
//! band, SCAT is reported alongside. Lesions get SCAT only: a lesion zeroes
//! every nonzero cell of its block, so a local null IS the lesion (§3).
//!
//! Two modes, and the split is the point:
//!
//! - `--measure` WRITES NOTHING. It decodes the base window and the three
//!   graft sources and prints every arm file's changed-cell count, full
//!   per-class composition, census-predicted dose and the §4 ±5% sanity
//!   check. Its output fills §4; the stamp waits on that table.
//! - `--generate` writes the eight arm files, their null families and
//!   `arms.txt`. It REFUSES while §10 reads `_unstamped_` — the stamp is a
//!   mechanical precondition here, not an operator convention (§7 step 4).
//!
//! Seeds come from `evidence/step5-calibration/seeds.txt` by decade (§7:
//! SCAT 301–370, LOCAL 371–400), first five per family, the rest reserved
//! and unused (there is no escalation, §3). Never minted at run time.
//!
//! Usage:
//!   cargo run -p neuralos-rt --release --example step5_calibration -- --measure
//!   cargo run -p neuralos-rt --release --example step5_calibration -- --generate
//!
//! Splice, null and guard logic are `harness::{splice_and_verify,
//! dose_matched_null, assert_unbanked}` verbatim — the step-5 family's own
//! implementation, not a second one (no new example by cloning).

use neuralos_rt::harness::{
    assert_unbanked, decode_slice, dose_matched_null, splice_and_verify, tix, ExperimentParams,
};
use neuralos_rt::{GgufFile, GGML_TYPE_Q2_0};
use neuralos_snn::Trit;
use std::collections::BTreeMap;

const BASE: &str = "models/Ternary-Bonsai-4B-Q2_0.gguf";
/// PREP.md pin, re-verified before `--measure` and before `--generate`
/// (PREREG §2; a mismatch is a §7 kill criterion).
const BASE_SHA: &str = "4e0bf8b737b0431552f8c2c97695ab7c0cb214c94bcdeb4f5f267e67ddf28b8b";
const PREREG_FILE: &str = "evidence/step5-calibration/PREREG.md";
const SEEDS_FILE: &str = "evidence/step5-calibration/seeds.txt";
const ARMS_FILE: &str = "evidence/step5-calibration/arms.txt";

/// §4 sanity check: every measured dose within ±5% of its census
/// prediction, IDENTITY exactly 0. A miss means the wrong tensor, layout
/// or window was read — kill (§7).
const DOSE_TOLERANCE: f64 = 0.05;
/// Five per family, no escalation (§3: both metrics are one-sided, more
/// nulls could only demote a band).
const NULLS_PER_FAMILY: usize = 5;
/// 4B head width: 4096 output rows / 32 heads.
const HEAD_ROWS: usize = 128;
/// §7 seed decades, in order: 10 slots of 10, first 301–310, last 391–400.
const SEED_SLOTS: usize = 10;
const SEED_BASE: u64 = 301;

/// Graft sources, in §7 seed order. `&'static str` because
/// `ExperimentParams::tensor` is one: the three layers are pre-registered
/// constants, so no tensor name is ever built at run time.
const GRAFT_LAYERS: [(usize, &str); 3] = [
    (1, "blk.1.attn_q.weight"),
    (18, "blk.18.attn_q.weight"),
    (35, "blk.35.attn_q.weight"),
];
const LESION_HEADS: [usize; 4] = [0, 1, 2, 3];

/// Trit bucket names for composition lines (index = `harness::tix`).
const TNAME: [&str; 3] = ["-1", "0", "+1"];

/// Which dose-matched null construction a family uses (§3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    /// Whole-window shuffle — step 5's own null.
    Scat,
    /// Shuffle inside the arm's own 128×512 block, embedded there.
    Local,
}

impl Family {
    fn label(self) -> &'static str {
        match self {
            Family::Scat => "scat",
            Family::Local => "local",
        }
    }
}

/// One arm file: its name, its patched window, its census prediction, and
/// the seed slot of each null family it carries (§7 decade order).
struct Arm {
    name: String,
    patch: Vec<Trit>,
    /// Census-predicted changed cells (§4). IDENTITY predicts exactly 0.
    predicted: f64,
    /// (family, seed slot) — LESION: SCAT only. GRAFT: SCAT + LOCAL.
    /// IDENTITY: none (it is a tripwire, not a judged comparison).
    families: Vec<(Family, usize)>,
    /// Is this the IDENTITY tripwire (exact-zero dose, sha ≡ base)?
    identity: bool,
}

/// sha of a file through the system tool — the `hybrid_invivo` pattern
/// verbatim (every pin of record was produced by `sha256sum`).
fn sha256_of(f: &str) -> String {
    let o = std::process::Command::new("sha256sum")
        .arg(f)
        .output()
        .expect("sha256sum runs");
    // A missing file makes sha256sum exit nonzero with empty stdout; the
    // `hybrid_invivo` form would return "" and the caller would report a
    // drifted base. Fail on the real cause instead.
    assert!(
        o.status.success(),
        "sha256sum {f} failed ({}): {}",
        o.status,
        String::from_utf8_lossy(&o.stderr).trim()
    );
    let sha = String::from_utf8_lossy(&o.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();
    assert_eq!(sha.len(), 64, "sha256sum {f}: unexpected output {sha:?}");
    sha
}

/// PREREG §2: the base is re-verified against the PREP.md pin BEFORE it is
/// decoded, in both modes. A drifted base voids everything downstream (§7).
fn verify_base_sha() {
    let sha = sha256_of(BASE);
    assert_eq!(
        sha, BASE_SHA,
        "base {BASE} sha {sha} != §2 pin {BASE_SHA} — KILL (§7): the base drifted"
    );
    println!("base    : {BASE}");
    println!("          sha {sha} == §2 pin : PASS");
}

/// PREREG §7 step 4: no arm file exists before the stamp. Enforced here
/// rather than trusted to the operator — `--generate` is the irreversible
/// half of this tool.
fn assert_stamped() {
    let text = std::fs::read_to_string(PREREG_FILE)
        .unwrap_or_else(|e| panic!("cannot read {PREREG_FILE}: {e}"));
    let stamp = text
        .split("## 10. Stamp")
        .nth(1)
        .unwrap_or_else(|| panic!("{PREREG_FILE}: § 10 Stamp section not found"))
        .trim();
    assert!(
        !stamp.is_empty() && !stamp.contains("_unstamped_"),
        "{PREREG_FILE} §10 is UNSTAMPED — refusing to generate arm files \
         (§7 step 4: the principal's stamp precedes --generate)"
    );
    println!("stamp   : {PREREG_FILE} §10 carries a stamp : PASS");
}

/// Confirm every graft source carries the host tensor's exact q2_0 layout
/// BEFORE a single byte of it is read (§3; a mismatch is a §7 kill
/// criterion). Loud on any difference — never a silent skip, never a
/// reshape.
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
             layouts differ, KILL (§7), REFUSING to read it",
            t.ty, p.tensor, host.ty
        );
        assert_eq!(
            t.dims, host.dims,
            "graft source {name} dims {:?} != host {} dims {:?} — \
             shapes differ, KILL (§7), REFUSING to read it",
            t.dims, p.tensor, host.dims
        );
        let data = f.tensor_data(t).expect("graft tensor window in bounds");
        assert_eq!(
            data.len(),
            p.tensor_bytes(),
            "graft source {name} byte window {} != {} (rows × row_bytes) — \
             layout is not the host's, KILL (§7), REFUSING to read it",
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
/// `attn_q` through `harness::decode_slice` — the same strict decode path
/// (type / dims / byte-size asserts) the host window goes through. Only the
/// tensor name moves.
fn decode_layer_window(tensor: &'static str, p: &ExperimentParams) -> Vec<Trit> {
    let mut q = p.clone();
    q.tensor = tensor;
    decode_slice(BASE, &q)
}

/// Class mix (−1/0/+1 fractions) of a trit run.
fn mix(cells: &[Trit]) -> [f64; 3] {
    let mut c = [0u64; 3];
    for t in cells {
        c[tix(*t)] += 1;
    }
    let n = cells.len() as f64;
    [c[0] as f64 / n, c[1] as f64 / n, c[2] as f64 / n]
}

/// LESION-h: head h's rows of the window, zeroed across all 512 input cols.
fn lesion_patch(base: &[Trit], h: usize, n: usize) -> Vec<Trit> {
    let mut out = base.to_vec();
    let (lo, hi) = (h * HEAD_ROWS * n, (h + 1) * HEAD_ROWS * n);
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
    if classes.is_empty() {
        return "(none)".to_string();
    }
    classes
        .iter()
        .map(|((f, t), n)| format!("({}→{} : {n})", TNAME[*f], TNAME[*t]))
        .collect::<Vec<_>>()
        .join(" ")
}

/// PREREG §7 seed decades: slot i is 301+10i … 310+10i. Selection is BY
/// POSITION and the map is asserted — `harness::decade_for` cannot serve
/// these (it asserts r ∈ 0..=4 and the 201–250 range), so this is the
/// generator's own selector with its own range assert (reviewer N1).
fn decade(seeds: &[u64], slot: usize) -> Vec<u64> {
    assert!(
        slot < SEED_SLOTS,
        "seed slot {slot} outside the {SEED_SLOTS} pre-registered decades"
    );
    assert!(
        seeds.len() >= 10 * (slot + 1),
        "{SEEDS_FILE} carries only {} entries — slot {slot} needs {}",
        seeds.len(),
        10 * (slot + 1)
    );
    let d: Vec<u64> = seeds.iter().copied().skip(10 * slot).take(10).collect();
    let lo = SEED_BASE + 10 * slot as u64;
    let hi = lo + 9;
    assert!(
        d.iter().all(|&s| (lo..=hi).contains(&s)),
        "seed map violation: slot {slot} must use {lo}–{hi}, got {d:?}"
    );
    d
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
        seeds.len() >= 10 * SEED_SLOTS,
        "{SEEDS_FILE}: need ≥{} seeds (one decade per family, §7), found {}",
        10 * SEED_SLOTS,
        seeds.len()
    );
    seeds
}

/// The eight arm files of §3, built in §7 seed-decade order, with their
/// census predictions (§4). Built once and used by both modes —
/// `--measure` and `--generate` must never disagree about what an arm IS.
///
/// Predictions: a LESION zeroes its block, so its dose is the window's
/// nonzero fraction × the block (§4's `0.6345 × 65,536` form). A GRAFT
/// overwrites its block with another layer's, so its dose is
/// `block × (1 − Σ_i p_i·q_i)` over the two blocks' ACTUAL class mixes
/// (§4's `1 − Σ p_i²` is that formula when the mixes coincide).
fn build_arms(base: &[Trit], p: &ExperimentParams) -> Vec<Arm> {
    let n = p.n;
    let block = HEAD_ROWS * n;
    let window_mix = mix(base);
    let mut arms: Vec<Arm> = Vec::with_capacity(LESION_HEADS.len() + GRAFT_LAYERS.len() + 1);

    for (i, h) in LESION_HEADS.into_iter().enumerate() {
        arms.push(Arm {
            name: format!("lesion-h{h}"),
            patch: lesion_patch(base, h, n),
            predicted: (1.0 - window_mix[1]) * block as f64,
            families: vec![(Family::Scat, i)],
            identity: false,
        });
    }
    let base_block_mix = mix(&base[..block]);
    for (i, (k, tensor)) in GRAFT_LAYERS.into_iter().enumerate() {
        let src = decode_layer_window(tensor, p);
        let src_mix = mix(&src[..block]);
        let same: f64 = (0..3).map(|c| base_block_mix[c] * src_mix[c]).sum();
        arms.push(Arm {
            name: format!("graft-k{k}"),
            patch: graft_patch(base, &src, n),
            predicted: (1.0 - same) * block as f64,
            families: vec![
                (Family::Scat, LESION_HEADS.len() + i),
                (Family::Local, LESION_HEADS.len() + GRAFT_LAYERS.len() + i),
            ],
            identity: false,
        });
    }
    arms.push(Arm {
        name: "identity".to_string(),
        patch: base.to_vec(),
        predicted: 0.0,
        families: Vec::new(),
        identity: true,
    });
    arms
}

/// LOCAL-family feasibility (§3): a block-local null must place the arm's
/// whole diff inside the arm's own 128×512 block, drawing each class from
/// that block's OWN src-class cells. `dose_matched_null` asserts the exact
/// dose and panics if a pool runs short — but that panic would land
/// mid-`--generate`, after gigabytes are written. Checked here instead, in
/// the mode that writes nothing.
fn local_feasible(base_block: &[Trit], arm_block: &[Trit]) -> (bool, String) {
    let mut demand = [0u64; 3];
    let mut avail = [0u64; 3];
    for (b, a) in base_block.iter().zip(arm_block.iter()) {
        avail[tix(*b)] += 1;
        if a != b {
            demand[tix(*b)] += 1;
        }
    }
    let ok = (0..3).all(|c| demand[c] <= avail[c]);
    let detail = (0..3)
        .map(|c| format!("{}: {}/{}", TNAME[c], demand[c], avail[c]))
        .collect::<Vec<_>>()
        .join(" · ");
    (
        ok,
        format!(
            "local pools (demand/available) {detail} : {}",
            if ok {
                "FEASIBLE"
            } else {
                "SHORT — LOCAL family impossible"
            }
        ),
    )
}

/// §4 sanity check: IDENTITY must measure exactly 0; every other arm must
/// land within ±5% of its census prediction. Returns (ok, rendered check).
fn sanity(measured: u64, predicted: f64, identity: bool) -> (bool, String) {
    if identity {
        let ok = measured == 0;
        return (
            ok,
            format!(
                "predicted 0 · {}",
                if ok {
                    "EXACT : PASS"
                } else {
                    "NONZERO : KILL (§7)"
                }
            ),
        );
    }
    let dev = (measured as f64 - predicted) / predicted;
    let ok = dev.abs() <= DOSE_TOLERANCE;
    (
        ok,
        format!(
            "predicted {predicted:.0} · dev {:+.2}% · {}",
            dev * 100.0,
            if ok {
                "within ±5% : PASS"
            } else {
                "outside ±5% : KILL (§7)"
            }
        ),
    )
}

/// `--measure` — dose accounting for PREREG §4. Writes nothing.
fn measure() {
    let p = ExperimentParams::default();
    let block = HEAD_ROWS * p.n;
    println!("=== step-5 calibration: --measure (PREREG §4 dose accounting; WRITES NOTHING) ===");
    verify_base_sha();
    assert_graft_layouts(&p);
    let base = decode_slice(BASE, &p);
    let census = base.iter().fold([0u64; 3], |mut c, t| {
        c[tix(*t)] += 1;
        c
    });
    println!(
        "window  : {} rows 0..{} × cols 0..{} — census (-1/0/+1) {census:?} of {} cells · nonzero {:.4}",
        p.tensor,
        p.n,
        p.n,
        base.len(),
        1.0 - census[1] as f64 / base.len() as f64
    );
    println!(
        "sanity  : §4 check is ±{:.0}% of the census prediction (IDENTITY exactly 0)\n",
        DOSE_TOLERANCE * 100.0
    );

    let arms = build_arms(&base, &p);
    let mut rows: Vec<(String, u64, String, String, bool)> = Vec::with_capacity(arms.len());
    for arm in &arms {
        let (cells, classes) = composition(&base, &arm.patch);
        let (ok, check) = sanity(cells, arm.predicted, arm.identity);
        let fams: Vec<&str> = arm.families.iter().map(|(f, _)| f.label()).collect();
        println!(
            "{}: {cells} cells · {check} · families [{}]",
            arm.name,
            fams.join(", ")
        );
        println!("    composition {}", fmt_comp(&classes));
        if arm.families.iter().any(|(f, _)| *f == Family::Local) {
            let (feasible, detail) = local_feasible(&base[..block], &arm.patch[..block]);
            println!("    {detail}");
            assert!(
                feasible,
                "{}: the LOCAL family cannot be built inside the block — spec/instrument \
                 disagreement, KILL before any write",
                arm.name
            );
        }
        rows.push((
            arm.name.clone(),
            cells,
            format!("{:.0}", arm.predicted),
            fmt_comp(&classes),
            ok,
        ));
    }

    // The §4 table, ready to paste (the document is filled from this run,
    // never from a recollection of it).
    println!("\n--- PREREG §4 table ---");
    println!("| Arm | predicted | measured | composition (from→to : n) |");
    println!("|---|---|---|---|");
    for (name, cells, pred, comp, _) in &rows {
        let label = if let Some(h) = name.strip_prefix("lesion-h") {
            format!("LESION-{h}")
        } else if let Some(k) = name.strip_prefix("graft-k") {
            format!("GRAFT-{k}")
        } else {
            name.to_uppercase()
        };
        println!("| {label} | {pred} | {cells} | {comp} |");
    }
    let failed: Vec<&str> = rows.iter().filter(|r| !r.4).map(|r| r.0.as_str()).collect();
    if failed.is_empty() {
        println!(
            "\nsanity check: {}/{} arms PASS — no §7 kill criterion fired",
            rows.len(),
            rows.len()
        );
    } else {
        println!("\nsanity check: KILL (§7) — {failed:?} outside their prediction");
    }
    println!("\nmeasure done — nothing written. §4 is filled from this output, then the stamp.");
}

/// `--generate` — the eight arm files, their null families, and arms.txt.
/// Refuses while §10 is unstamped.
fn generate() {
    let p = ExperimentParams::default();
    let block = HEAD_ROWS * p.n;
    println!("=== step-5 calibration: --generate (PREREG §7 step 5) ===");
    verify_base_sha();
    assert_stamped();
    assert_graft_layouts(&p);
    let base = decode_slice(BASE, &p);
    let seeds = load_seeds();
    let arms = build_arms(&base, &p);

    let mut manifest: Vec<String> = Vec::with_capacity(arms.len());
    for arm in &arms {
        let (cells, classes) = composition(&base, &arm.patch);
        let (ok, check) = sanity(cells, arm.predicted, arm.identity);
        assert!(
            ok,
            "{}: {cells} cells — {check}; §7 kill criterion, nothing is written",
            arm.name
        );
        let out = format!("models/cal-{}.gguf", arm.name);
        assert_unbanked(&out);
        splice_and_verify(BASE, &out, &arm.patch, Some(&base), &p);
        println!(
            "{}: {cells} cells · {} · S2 clean → {out}",
            arm.name,
            fmt_comp(&classes)
        );
        if arm.identity {
            // §7 kill criterion: the tripwire's file must BE the base.
            let (sha, base_sha) = (sha256_of(&out), sha256_of(BASE));
            assert_eq!(
                sha, base_sha,
                "identity export sha {sha} != base {base_sha} — KILL (§7): the pipeline is not transparent"
            );
            println!("  identity: sha == base ({base_sha:.16}…) : PASS");
        }

        let mut fams: Vec<String> = Vec::new();
        for &(family, slot) in &arm.families {
            for &seed in decade(&seeds, slot).iter().take(NULLS_PER_FAMILY) {
                let null = match family {
                    Family::Scat => dose_matched_null(&base, &arm.patch, seed),
                    Family::Local => {
                        // Shuffle inside the arm's own 128×512 block, then
                        // embed at that block: same count, same composition,
                        // positions random INSIDE the footprint (§3).
                        let shuffled = dose_matched_null(&base[..block], &arm.patch[..block], seed);
                        let mut full = base.clone();
                        full[..block].copy_from_slice(&shuffled);
                        full
                    }
                };
                let nout = format!("models/cal-{}-{}-s{seed}.gguf", arm.name, family.label());
                assert_unbanked(&nout);
                splice_and_verify(BASE, &nout, &null, Some(&base), &p);
                let changed = null.iter().zip(base.iter()).filter(|(a, b)| a != b).count() as u64;
                assert_eq!(
                    changed,
                    cells,
                    "{} {} s{seed}: dose {changed} != arm {cells}",
                    arm.name,
                    family.label()
                );
                println!(
                    "  {} s{seed}: {changed} cells (exact dose) · S2 clean → {nout}",
                    family.label()
                );
            }
            fams.push(family.label().to_string());
        }
        manifest.push(
            format!("{} {}", arm.name, fams.join(" "))
                .trim_end()
                .to_string(),
        );
    }

    let mut text = String::from(
        "# Step-5 calibration arm files — written by step5_calibration --generate.\n\
         # One line per arm: <arm-stem> [family …], in PREREG §7 decade order.\n\
         # Files are models/cal-<arm>.gguf and models/cal-<arm>-<family>-s<seed>.gguf.\n\
         # The aggregator reads THIS file; no arm list is ever hardcoded (§7 step 7).\n",
    );
    for line in &manifest {
        text.push_str(line);
        text.push('\n');
    }
    std::fs::write(ARMS_FILE, &text).unwrap_or_else(|e| panic!("cannot write {ARMS_FILE}: {e}"));
    println!(
        "\narms.txt: {} arm(s) → {ARMS_FILE}\ngenerate done — judge chain next (§7 step 6).",
        manifest.len()
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
