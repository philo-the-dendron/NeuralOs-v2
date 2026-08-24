//! Step-5 readout aggregator — the mechanical verdict table (PREREG §5).
//!
//! Two modes:
//!
//! - `--calibrate` : the calibration gate from banked logs (loud d1/d3
//!   classify, quiet d7/loop silent, M3 reproduces the P3′ provenance
//!   pins). Zero new judge runs; exits nonzero on any gate miss.
//! - `<dir>`       : aggregate a burn-window evidence root. Expected
//!   layout under `<dir>` (PREREG §8): `on-r{0,1,2}/`, `null-r{r}-s*`
//!   (the seeded shuffled-drift family), `domain/`, `free-ck*/` — each
//!   judge-leg dir carrying `p{0..4}_run1.log` (+ `_run2.log` for
//!   double-run arms, asserted byte-identical) and `p{0..4}_run1.err`
//!   dumps. Replicates aggregate only when their full null family is
//!   present; partial roots report partial, loudly.
//!
//! The verdict is COMPUTED from the PREREG §5 bands (library
//! `step5_band`); no hand-counted verdicts anywhere. Tier-3 outcome
//! rule: ≥2/3 SEPARATED = demonstrated; ≤1/3 = rests evidenced;
//! 0/3 null-consistent = clean null. Delta-zero publishable.
//!
//! Usage:
//!   cargo run -p neuralos-rt --release --example step5_aggregate -- --calibrate
//!   cargo run -p neuralos-rt --release --example step5_aggregate -- evidence/step5-readout/burn

use neuralos_rt::judge::{
    parse_dump_file, step5_band, step5_classify, step5_continuation, step5_max_margin_delta,
    step5_read_dir, Step5Band, Step5FileReadout,
};
use std::path::{Path, PathBuf};
use std::process::exit;

fn base_dir(root: &Path) -> PathBuf {
    let d = root.join("base");
    if d.join("p0_run1.log").exists() {
        return d;
    }
    // Not every burn layout re-runs base legs: the frozen base
    // continuations live in the library; dumps default to the banked
    // f-judge legs (the in-tree base-side convention, PREREG §5).
    // Dual-prefix: examples run from the crate dir AND the repo root.
    ["evidence/session-f-judge", "../../evidence/session-f-judge"]
        .iter()
        .map(PathBuf::from)
        .find(|d| d.join("p0_run1.log").exists())
        .unwrap_or_else(|| PathBuf::from("evidence/session-f-judge"))
}

/// Max M3 over the five prompts for one judged dir (None = no
/// knife-edge steps anywhere — recorded, never zero).
fn m3_of(dir: &Path, base_root: &Path) -> Option<f64> {
    let mut max = f64::NEG_INFINITY;
    for p in 0..5 {
        let base = parse_dump_file(&base_root.join(format!("p{p}_run1.err")).to_string_lossy());
        let cand = parse_dump_file(&dir.join(format!("p{p}_run1.err")).to_string_lossy());
        if let Some(d) = step5_max_margin_delta(&base, &cand) {
            max = max.max(d);
        }
    }
    (max != f64::NEG_INFINITY).then_some(max)
}

fn assert_double_run(dir: &Path) {
    for p in 0..5 {
        let r1 = dir.join(format!("p{p}_run1.log"));
        let r2 = dir.join(format!("p{p}_run2.log"));
        if r2.exists() {
            let (a, b) = (
                std::fs::read(&r1).unwrap_or_else(|e| panic!("read {}: {e}", r1.display())),
                std::fs::read(&r2).unwrap_or_else(|e| panic!("read {}: {e}", r2.display())),
            );
            assert!(a == b, "{}: run1 != run2 (double-run determinism)", dir.display());
        }
    }
}

fn calibrate() {
    let root = |p: &str| {
        ["evidence/", "../../evidence/"]
            .iter()
            .map(|d| format!("{d}{p}"))
            .find(|f| Path::new(f).exists())
            .unwrap_or_else(|| panic!("banked artifact {p} not found"))
    };
    println!("== step-5 calibration gate (banked logs, mechanical parity) ==");
    let mut ok = true;

    // Loud probes: classification must be byte-exact.
    for (dir, want) in [
        ("session-i-primary/null-d1", "p3:B1 p4:B4"),
        ("session-i-primary/null-d3", "p2:B2b p3:B1"),
    ] {
        let ro = step5_read_dir(&root(dir));
        let got: Vec<String> = ro
            .flips
            .iter()
            .map(|(p, s)| {
                let c = match step5_classify(*p, s) {
                    neuralos_rt::judge::Step5Destination::Basin(id) => id.to_string(),
                    _ => "NOVEL".to_string(),
                };
                format!("p{p}:{c}")
            })
            .collect();
        let got = got.join(" ");
        let pass = got == want && ro.voids.is_empty();
        println!("  loud  {dir}: [{got}] want [{want}] : {}", if pass { "PASS" } else { "FAIL" });
        ok &= pass;
    }

    // Quiet probes: zero flips, zero voids.
    {
        let dir = "session-i-primary/null-d7";
        let ro = step5_read_dir(&root(dir));
        let pass = ro.flips.is_empty() && ro.voids.is_empty();
        println!("  quiet {dir}: {} flips : {}", ro.flips.len(), if pass { "PASS" } else { "FAIL" });
        ok &= pass;
    }
    for p in 0..5 {
        let text = std::fs::read_to_string(root(&format!("session-f-judge/p{p}_run1.log"))).unwrap();
        let pass = step5_classify(p, step5_continuation(p, &text).unwrap())
            == neuralos_rt::judge::Step5Destination::Identical;
        println!("  quiet loop p{p}: base-identical : {}", if pass { "PASS" } else { "FAIL" });
        ok &= pass;
    }
    // (M3 provenance pins are test-enforced in judge.rs —
    // step5_calibration_gate_passes_on_banked_logs.)

    println!(
        "calibration: {}",
        if ok { "GATE PASS — arms may be trusted" } else { "GATE FAIL — stop, fix, re-gate" }
    );
    if !ok {
        exit(1);
    }
}

fn aggregate(root: &Path) {
    println!("== step-5 aggregation over {} ==", root.display());
    let base = base_dir(root);

    let mut verdict: Vec<Step5Band> = Vec::new();
    for r in 0..3 {
        let on_dir = root.join(format!("on-r{r}"));
        if !on_dir.exists() {
            println!("  on-r{r}: ABSENT (partial root)");
            continue;
        }
        assert_double_run(&on_dir);
        let on = step5_read_dir(&on_dir.to_string_lossy());
        if !on.voids.is_empty() {
            println!("  on-r{r}: VOIDS {:#?} — void protocol §6", on.voids);
        }
        let mut nulls: Vec<Step5FileReadout> = Vec::new();
        let mut null_m3s: Vec<Option<f64>> = Vec::new();
        for e in std::fs::read_dir(root).unwrap().flatten() {
            let d = e.path();
            let name = d.file_name().unwrap().to_string_lossy().into_owned();
            if name.starts_with(&format!("null-r{r}-")) && d.is_dir() {
                let ro = step5_read_dir(&d.to_string_lossy());
                if !ro.voids.is_empty() {
                    println!("  {name}: VOIDS {:#?} — excluded (void protocol)", ro.voids);
                    continue;
                }
                null_m3s.push(m3_of(&d, &base));
                nulls.push(ro);
            }
        }
        if nulls.is_empty() {
            println!("  on-r{r}: null family ABSENT — replicate not adjudicable yet");
            continue;
        }
        let on_m3 = m3_of(&on_dir, &base);
        let band = step5_band(&on, on_m3, &nulls, &null_m3s);
        let null_m3_max = null_m3s.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  on-r{r}: {} flips · M3 {:?} · nulls {} · M3max {} → {:?}",
            on.flips.len(),
            on_m3,
            nulls.len(),
            if null_m3_max == f64::NEG_INFINITY { "n/a".to_string() } else { format!("{null_m3_max:.4}") },
            band
        );
        verdict.push(band);
    }

    // The outcome line is only defined on a complete set (PREREG §1);
    // the tool must not be quotable on incomplete data.
    if verdict.len() < 3 {
        println!(
            "\nverdict: PARTIAL ROOT ({}/3 replicates) — no outcome quoted on incomplete data",
            verdict.len()
        );
        return;
    }

    let sep = verdict.iter().filter(|b| **b == Step5Band::Separated).count();
    let mixed = verdict.iter().filter(|b| **b == Step5Band::Mixed).count();
    let outcome = match (verdict.len(), sep, mixed) {
        (0, _, _) => "no complete replicates yet".to_string(),
        (_, s, _) if s >= 2 => "TIER 3 DEMONSTRATED".to_string(),
        (_, 0, 0) => "CLEAN NULL — delta-zero publishable".to_string(),
        (_, s, m) if s <= 1 && m == 0 => "RESTS EVIDENCED".to_string(),
        (_, _, m) if m > 0 => "MIXED present — escalation ladder §5 (one, pre-authorized)".to_string(),
        _ => "see PREREG §1 bands".to_string(),
    };
    println!("\nverdict: {sep}/{} SEPARATED · {mixed} MIXED → {outcome}", verdict.len());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--calibrate") => calibrate(),
        Some(dir) => aggregate(Path::new(dir)),
        None => {
            eprintln!("usage: step5_aggregate --calibrate | <burn-root>");
            exit(1);
        }
    }
}
