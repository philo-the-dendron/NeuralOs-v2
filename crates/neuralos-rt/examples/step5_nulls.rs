//! Step-5 NULL generator — the per-replicate shuffled-drift families
//! (PREREG §3/§4, evidence/step5-readout/PREREG.md).
//!
//! Session-I's `null_patches` generated ONE family from the H2 terminal
//! diff with hardcoded seeds. Step 5 needs a family per ON replicate:
//! dose-matched shuffles of THAT replicate's own terminal diff (the
//! adapted file vs the original — positions AND values from the
//! artifact, never re-run), seeded from the pre-committed
//! `evidence/step5-readout/null_seeds.txt` — NEVER minted at run time.
//!
//! Seed map (null_seeds.txt): 201–210 → r0 · 211–220 → r1 · 221–230 →
//! r2 · 231–240 → r3 · 241–250 → r4 (the ESCALATION decades — windows
//! 3/4, ruled 2026-08-26 after the n=3 MIXED verdict, before any
//! escalation arm ran; seeds extended to 250 by the same ruling).
//!
//! The shuffle algorithm is `harness::dose_matched_null` — the
//! session-I PRIMARY-family logic extracted verbatim (its exact-dose +
//! full-composition contract, crossovers included, asserted inside).
//! For seeds 1..=10 against the H2 diff it reproduces the banked
//! `null-dose-{s}` family byte-identically (the r4-closeout
//! regeneration proof is that function's pin).
//!
//! Usage:
//!   cargo run -p neuralos-rt --release --example step5_nulls -- --replicate 0
//!   # optional: --orig <base.gguf> --on <that replicate's ON export>
//!
//! Writes `models/null-r{r}-s{seed}.gguf` (×10, S2-asserted, unbanked-
//! guarded). Requires the ON export to exist and differ from base.

use neuralos_rt::harness::{
    assert_unbanked, decode_slice, dose_matched_null, splice_and_verify, tix, ExperimentParams,
};

const SEEDS_FILE: &str = "evidence/step5-readout/null_seeds.txt";

/// Parse null_seeds.txt: integer lines only, comments (#) skipped.
/// Returns the seeds in file order (ascending of record).
fn load_seeds() -> Vec<u64> {
    let text = std::fs::read_to_string(SEEDS_FILE).unwrap_or_else(|e| {
        panic!("cannot read {SEEDS_FILE}: {e} — seeds are pre-committed, never minted")
    });
    let seeds: Vec<u64> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            l.parse::<u64>()
                .unwrap_or_else(|e| panic!("{SEEDS_FILE}: non-integer seed line {l:?}: {e}"))
        })
        .collect();
    assert!(
        seeds.len() >= 50,
        "{SEEDS_FILE}: need ≥50 seeds (201–250: main + escalation decades), found {}",
        seeds.len()
    );
    seeds
}

fn main() {
    #[allow(non_snake_case)]
    let p = ExperimentParams::default();
    #[allow(non_snake_case)]
    let N = p.n;
    let mut replicate: Option<usize> = None;
    let mut orig_path = "models/Ternary-Bonsai-4B-Q2_0.gguf".to_string();
    let mut on_path: Option<String> = None;
    {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0usize;
        while i < argv.len() {
            match argv[i].as_str() {
                "--replicate" => {
                    replicate = Some(
                        argv.get(i + 1)
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or_else(|| panic!("--replicate expects 0|1|2|3|4")),
                    );
                    i += 2;
                }
                "--orig" => {
                    orig_path = argv.get(i + 1).expect("--orig <path>").clone();
                    i += 2;
                }
                "--on" => {
                    on_path = Some(argv.get(i + 1).expect("--on <path>").clone());
                    i += 2;
                }
                other => panic!("unknown arg {other:?} — expected --replicate/--orig/--on"),
            }
        }
    }
    let r = replicate.expect("--replicate <0..=4> is required (PREREG §4 + escalation amendment)");
    assert!(
        (0..=4).contains(&r),
        "replicate r ∈ 0..=4 (r3/r4 are the pre-authorized escalation windows, PREREG §5 ladder)"
    );
    let on_path =
        on_path.unwrap_or_else(|| format!("models/Ternary-Bonsai-4B-Q2_0-invivo-r{r}.gguf"));

    // Seeds: the replicate's decade, from the file only. 201–210 → r0…
    // (the decade selection + full seed-map contract lives in
    // harness::decade_for — test-pinned for all five replicates)
    let seeds = load_seeds();
    let decade = neuralos_rt::harness::decade_for(&seeds, r);

    println!("=== step-5 nulls: replicate r{r} — dose-matched shuffled-drift ×10 ===");
    println!("orig    : {orig_path}");
    println!("ON      : {on_path}");
    println!("seeds   : {decade:?} (from {SEEDS_FILE}; escalation decades 231–250 per the 2026-08-26 ruling)");

    let src = decode_slice(&orig_path, &p);
    let on = decode_slice(&on_path, &p);
    let diff_cells: usize = (0..N * N).filter(|&i| on[i] != src[i]).count();
    assert!(
        diff_cells > 0,
        "ON export is byte-identical to base in the slice — nothing to shuffle (arm void?)"
    );
    // Composition of record (printed; exact-dose + full-composition are
    // asserted INSIDE dose_matched_null).
    let mut comp_names = Vec::new();
    for i in 0..N * N {
        if on[i] != src[i] {
            comp_names.push((tix(src[i]), tix(on[i])));
        }
    }
    let mut classes: std::collections::BTreeMap<(usize, usize), u64> = Default::default();
    for k in comp_names {
        *classes.entry(k).or_default() += 1;
    }
    println!("ON diff : {diff_cells} cells; composition {classes:?}");

    for &seed in &decade {
        let patched = dose_matched_null(&src, &on, seed);
        let out = format!("models/null-r{r}-s{seed}.gguf");
        assert_unbanked(&out);
        splice_and_verify(&orig_path, &out, &patched, None, &p);
        let changed = patched.iter().zip(&src).filter(|(a, b)| a != b).count();
        println!("null-r{r}-s{seed}: {changed} cells (exact dose) · S2 clean → {out}");
    }
    println!("done — judge chains next (tools/run_prompts.sh, single-run per null).");
}
