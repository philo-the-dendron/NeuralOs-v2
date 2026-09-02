//! Session I ladder rung 3 v2 — THE NULL FAMILY: dose-matched primary ×10
//! + position-shuffle ×3, both derived from the H2 TERMINAL PATCH.
//!
//! Pre-registered (ISA sI amendment 2026-08-20, committed before the stress
//! chain's results were read): the PRIMARY family carries EXACTLY the H2
//! patch's changed-cell count (87,119), composition from the H2 terminal
//! diff (decode the H2 patched GGUF's slice vs the original). The earlier
//! census-transition family (~178k cells) is the STRESS ARM, report-only.
//!
//! - **dose-<seed>** (×10, seeds 1–10): exactly `H2_CELLS` changed
//!   cells, per-class composition matching the H2 terminal diff,
//!   placed uniformly over cells whose source value matches the
//!   from-class (seeded FY shuffle). (An escalation family, seeds
//!   11–20, was pre-registered but never generated — the
//!   adjudication made it moot: 8/10 ≫ the 1/10 bar,
//!   `evidence/session-i-primary/README.md`.)
//! - **shuffle-<seed>** (×3) — AS BUILT (v3 amendment): the REAL
//!   changed-cell set, exact dose, values REFLECTED where legal
//!   (−1→0 becomes −1→+1 · +1→0 becomes +1→−1 · 0→+1 unreflectable
//!   and held); seed variation via a random ~10% hold-out. The
//!   pre-registered v2 design (uniform value shuffle, re-draw on
//!   collision) is provably impossible for this patch — unique-flow
//!   theorem in the ISA. A shrunk dose ABORTS loudly, never
//!   silently. Pre-stated bar: "reproduces the signature" = ≥10/12
//!   p3 flips OR the p3 step-1 knife-edge crossing.
//!
//! Usage: `cargo run -p neuralos-rt --release --example null_patches --
//! [orig.gguf] [h2-patched.gguf]` — writes models/null-dose-<s>.gguf and
//! models/null-flip-<s>.gguf, all S2-asserted.

use neuralos_rt::harness::{decode_slice, splice_and_verify, tix, xorshift64, ExperimentParams};
use neuralos_snn::Trit;

#[allow(non_snake_case)]
fn main() {
    let p = ExperimentParams::default();
    let N = p.n;
    let orig_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Ternary-Bonsai-4B-Q2_0.gguf".into());
    let h2_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/Ternary-Bonsai-4B-Q2_0-invivo.gguf".into());
    println!("=== Null family v2: dose-matched ×10 + shuffle ×3 (from the H2 terminal patch) ===");
    println!("orig    : {orig_path}");
    println!("H2      : {h2_path}");

    let src = decode_slice(&orig_path, &p);
    let h2 = decode_slice(&h2_path, &p);

    // ----- The H2 terminal diff: exact changed-cell set + composition -----
    let changed_idx: Vec<usize> = (0..N * N).filter(|&i| h2[i] != src[i]).collect();
    let h2_cells = changed_idx.len();
    // per (from,to) class composition of the terminal diff
    let mut comp: [u64; 9] = [0; 9]; // (from*3 + to), values 0..2 minus/zero/one
    for &i in &changed_idx {
        comp[tix(src[i]) * 3 + tix(h2[i])] += 1;
    }
    println!("H2 diff : {h2_cells} changed cells (pre-registered 87,119)");
    assert_eq!(
        h2_cells, 87_119,
        "the H2 terminal diff must carry exactly the recorded cell count"
    );
    println!(
        "comp    : −1→0 {} · 0→−1 {} · 0→+1 {} · +1→0 {} · (others {})",
        comp[1],
        comp[3],
        comp[5],
        comp[7],
        comp.iter()
            .enumerate()
            .filter(|(k, _)| {
                !matches!(*k, 1 | 3 | 5 | 7) // off-diagonal minus identity mappings
            })
            .count()
    );

    // ----- The surgery unit (shared harness splice_and_verify; S2
    // re-read on EVERY file — the R4-extracted core) -----
    let do_surgery = |patched: &[Trit], out: &str| {
        splice_and_verify(&orig_path, out, patched, None, &p);
    };

    // ----- PRIMARY: dose-matched ×10 (exact H2 composition, uniform cells) -----
    // class list: ALL nonzero (from,to) classes of the terminal diff —
    // the exact-dose assert caught that crossovers (−1→+1 etc., ~21k
    // cells) exist beyond the four census classes; dose-matching must
    // carry the FULL composition or the dose silently shrinks.
    let tr_of = |i: usize| -> Trit {
        match i {
            0 => Trit::MinusOne,
            1 => Trit::Zero,
            _ => Trit::One,
        }
    };
    let classes: Vec<(Trit, Trit, u64)> = (0..9)
        .filter(|&k| comp[k] > 0 && k / 3 != k % 3)
        .map(|k| (tr_of(k / 3), tr_of(k % 3), comp[k]))
        .collect();
    println!(
        "classes : {} (full composition incl. crossovers)",
        classes.len()
    );
    for seed in 1..=10u64 {
        let mut rng = 0xD05E_0000_0000_0001_u64 ^ seed;
        let mut patched = src.clone();
        // changed-once flag: pools draw from SRC-class cells that are
        // still unchanged — a cell consumed by one class can never be
        // redrawn by another (the double-draw bug the exact-dose assert
        // caught: 66,161 ≠ 87,119).
        let mut used = vec![false; N * N];
        for (from, to, count) in &classes {
            let mut idxs: Vec<usize> = (0..N * N)
                .filter(|&i| !used[i] && src[i] == *from)
                .collect();
            for i in (1..idxs.len()).rev() {
                rng = xorshift64(rng);
                let j = (rng % (i as u64 + 1)) as usize;
                idxs.swap(i, j);
            }
            let take = (*count).min(idxs.len() as u64) as usize;
            for &cell in idxs.iter().take(take) {
                patched[cell] = *to;
                used[cell] = true;
            }
        }
        let changed = patched.iter().zip(&src).filter(|(a, b)| a != b).count();
        assert!(
            changed == h2_cells,
            "dose-{seed}: changed {changed} != target {h2_cells} — dose must match EXACTLY"
        );
        let out = format!("models/null-dose-{seed}.gguf");
        do_surgery(&patched, &out);
        println!("dose-{seed}: {changed} cells (exact) · S2 clean → {out}");
    }

    // ----- VALUE-FLIP ×3 (v3 amendment: the uniform shuffle is provably
    // impossible for this patch — unique-flow theorem in the ISA). Real
    // changed positions, exact dose, values REFLECTED where legal:
    // −1→0 becomes −1→+1 · +1→0 becomes +1→−1 · 0→+1 unreflectable
    // (no legal alternative at a 0-source under no-same-source) and held.
    // Seed variation enters via a random hold-out split: each seed holds
    // out a different random ~10% of flippable cells (kept at H2 values)
    // so the three family members differ while every member stays exact-
    // dose and legal.
    for seed in 1..=3u64 {
        let mut rng = 0xF11_0000_0000_0003_u64.wrapping_add(seed);
        let mut patched = src.clone();
        let mut flipped = 0usize;
        let mut held = 0usize;
        for &i in &changed_idx {
            let (from, to) = (src[i], h2[i]);
            let flipped_val = match (from, to) {
                (Trit::MinusOne, Trit::Zero) => Some(Trit::One),
                (Trit::One, Trit::Zero) => Some(Trit::MinusOne),
                _ => None,
            };
            match flipped_val {
                Some(v) => {
                    rng = xorshift64(rng);
                    if rng % 10 != 0 {
                        patched[i] = v;
                        flipped += 1;
                    } else {
                        patched[i] = to;
                        held += 1;
                    }
                }
                None => {
                    patched[i] = to;
                    held += 1;
                }
            }
        }
        let changed = patched.iter().zip(&src).filter(|(a, b)| a != b).count();
        assert_eq!(changed, h2_cells, "flip-{seed}: dose exact");
        let out = format!("models/null-flip-{seed}.gguf");
        do_surgery(&patched, &out);
        println!(
            "flip-{seed}: {changed} cells (exact; {flipped} reflected · {held} held) · S2 clean → {out}"
        );
    }
    println!("done — judge chains next (queued behind the stress arm).");
}
