//! Step-5 readout aggregator — the mechanical verdict table (PREREG §5).
//!
//! Two modes:
//!
//! - `--calibrate` : the calibration gate from banked logs (loud d1/d3
//!   classify, quiet d7/loop silent, M3 reproduces the P3′ provenance
//!   pins). Zero new judge runs; exits nonzero on any gate miss.
//! - `--poscontrol <root>` : the step-5 CALIBRATION positive control
//!   (`evidence/step5-calibration/PREREG.md` v2) — lesion/graft arms
//!   banded against their own null families and the §6 verdict. A
//!   DIFFERENT thing from `--calibrate` above, which is the banked-log
//!   parity gate of the instrument; the names are kept unmistakable
//!   on purpose (reviewer B5).
//! - `<dir>`       : aggregate a burn-window evidence root. Expected
//!   layout under `<dir>` (PREREG §8): `on-r{0..4}/`, `null-r{r}-s*`
//!   (the seeded shuffled-drift family), `domain/`, `free-ck*/` — each
//!   judge-leg dir carrying `p{0..4}_run1.log` (+ `_run2.log` for
//!   double-run arms, asserted byte-identical) and `p{0..4}_run1.err`
//!   dumps. Replicates aggregate only when their full null family is
//!   present; partial roots report partial, loudly.
//!
//! The verdict is COMPUTED from the PREREG §5 bands (library
//! `step5_band`); no hand-counted verdicts anywhere. Tier-3 outcome
//! rule (n=5, post-escalation): s ≥ 2/5 SEPARATED = demonstrated;
//! s ≤ 1 with 0 MIXED = rests evidenced; 0/5 with 0 MIXED = clean
//! null, delta-zero publishable; s ≤ 1 with MIXED present = NO
//! OUTCOME QUOTABLE — the one pre-authorized escalation is spent
//! (PRE-REGISTRATION-UNDEFINED; ISA ruling 2026-08-27).
//!
//! Usage:
//!   cargo run -p neuralos-rt --release --example step5_aggregate -- --calibrate
//!   cargo run -p neuralos-rt --release --example step5_aggregate -- evidence/step5-readout/burn

use neuralos_rt::harness::{decode_slice, ExperimentParams};
use neuralos_rt::judge::{
    parse_dump_file, poscontrol_verdict, step5_band, step5_classify, step5_continuation,
    step5_max_margin_delta, step5_read_dir, PosControlVerdict, Step5Band, Step5FileReadout,
};
use std::path::{Path, PathBuf};
use std::process::exit;

/// FREE-arm pre-judge re-verification (PREREG §3): the banked H2b
/// checkpoints decoded against the base — cell counts MUST match the
/// H2-banked values (61,210 / 71,381 / 80,391) before judging.
fn verify_free() {
    let p = ExperimentParams::default();
    let n = p.n;
    let orig = decode_slice("models/Ternary-Bonsai-4B-Q2_0.gguf", &p);
    for (ck, want) in [(400usize, 61_210u64), (800, 71_381), (1200, 80_391)] {
        let path = format!("models/Ternary-Bonsai-4B-Q2_0-invivo-ck{ck}.gguf");
        let trits = decode_slice(&path, &p);
        let cells = (0..n * n).filter(|&i| trits[i] != orig[i]).count() as u64;
        assert_eq!(cells, want, "ck{ck}: decoded {cells} changed cells != H2-banked {want} — artifact drifted, FREE arm voids");
        println!("  ck{ck}: {cells} cells == banked {want} : PASS");
    }
    println!("verify-free: ALL ck cell counts re-verified — FREE arm may be judged");
}

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
            assert!(
                a == b,
                "{}: run1 != run2 (double-run determinism)",
                dir.display()
            );
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
        println!(
            "  loud  {dir}: [{got}] want [{want}] : {}",
            if pass { "PASS" } else { "FAIL" }
        );
        ok &= pass;
    }

    // Quiet probes: zero flips, zero voids.
    {
        let dir = "session-i-primary/null-d7";
        let ro = step5_read_dir(&root(dir));
        let pass = ro.flips.is_empty() && ro.voids.is_empty();
        println!(
            "  quiet {dir}: {} flips : {}",
            ro.flips.len(),
            if pass { "PASS" } else { "FAIL" }
        );
        ok &= pass;
    }
    for p in 0..5 {
        let text =
            std::fs::read_to_string(root(&format!("session-f-judge/p{p}_run1.log"))).unwrap();
        let pass = step5_classify(p, step5_continuation(p, &text).unwrap())
            == neuralos_rt::judge::Step5Destination::Identical;
        println!(
            "  quiet loop p{p}: base-identical : {}",
            if pass { "PASS" } else { "FAIL" }
        );
        ok &= pass;
    }
    // (M3 provenance pins are test-enforced in judge.rs —
    // step5_calibration_gate_passes_on_banked_logs.)

    println!(
        "calibration: {}",
        if ok {
            "GATE PASS — arms may be trusted"
        } else {
            "GATE FAIL — stop, fix, re-gate"
        }
    );
    if !ok {
        exit(1);
    }
}

fn aggregate(root: &Path) {
    println!("== step-5 aggregation over {} ==", root.display());
    let base = base_dir(root);

    let mut verdict: Vec<Step5Band> = Vec::new();
    // 0..5 post-escalation (2026-08-26 ruling): r3/r4 are first-class
    // replicates. The n=3 verdict of 2026-08-26 06:11Z stands banked
    // in the log's history; this loop reading 5 is what makes the
    // re-verdict actually see the escalation arms (a hardcoded 0..3
    // would have silently skipped them — caught at build review).
    for r in 0..5 {
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
        let null_m3_max = null_m3s
            .iter()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  on-r{r}: {} flips · M3 {:?} · nulls {} · M3max {} → {:?}",
            on.flips.len(),
            on_m3,
            nulls.len(),
            if null_m3_max == f64::NEG_INFINITY {
                "n/a".to_string()
            } else {
                format!("{null_m3_max:.4}")
            },
            band
        );
        verdict.push(band);
    }

    // The outcome line is only defined on a complete set (PREREG §1,
    // n=5 post-escalation ruling); the tool must not be quotable on
    // incomplete data.
    if verdict.len() < 5 {
        println!(
            "\nverdict: PARTIAL ROOT ({}/5 replicates) — no outcome quoted on incomplete data",
            verdict.len()
        );
        return;
    }

    let sep = verdict
        .iter()
        .filter(|b| **b == Step5Band::Separated)
        .count();
    let mixed = verdict.iter().filter(|b| **b == Step5Band::Mixed).count();
    let outcome = match (verdict.len(), sep, mixed) {
        (0, _, _) => "no complete replicates yet".to_string(),
        (_, s, _) if s >= 2 => "TIER 3 DEMONSTRATED".to_string(),
        (_, 0, 0) => "CLEAN NULL — delta-zero publishable".to_string(),
        (_, s, m) if s <= 1 && m == 0 => "RESTS EVIDENCED".to_string(),
        // The one pre-authorized escalation (§5) is SPENT — the n=5
        // ruling (ISA 2026-08-27) declared the old directive here a
        // stale string not to be quoted. On s<=1 with MIXED present the
        // ratified arms define no outcome: refuse to quote one, add no
        // threshold. The gap itself is a defect the step-8
        // pre-registration must close (its constraint 3).
        (_, _, m) if m > 0 => {
            "NO OUTCOME QUOTABLE — s<=1 with MIXED present and the one pre-authorized \
             escalation SPENT (PRE-REGISTRATION-UNDEFINED; ISA ruling 2026-08-27)"
                .to_string()
        }
        _ => "see PREREG §1 bands".to_string(),
    };
    println!(
        "\nverdict: {sep}/{} SEPARATED · {mixed} MIXED → {outcome}",
        verdict.len()
    );
}

/// Null families are exactly five files (PREREG §5): a PRIMARY family
/// reduced below five by the void protocol voids its ARM, because a
/// smaller family lowers the M3 max and biases toward the positive
/// verdict. A depleted graft SCAT family degrades only the SCAT band.
const NULLS_PER_FAMILY: usize = 5;

/// `arms.txt` lives beside PREREG.md, one level above the burn root, and
/// is written by `step5_calibration --generate` (§7 step 5). The arm list
/// is read from it and never hardcoded here.
fn arms_file(root: &Path) -> PathBuf {
    if let Some(f) = root
        .parent()
        .map(|d| d.join("arms.txt"))
        .filter(|f| f.exists())
    {
        return f;
    }
    // Dual-prefix: examples run from the crate dir AND the repo root.
    [
        "evidence/step5-calibration/arms.txt",
        "../../evidence/step5-calibration/arms.txt",
    ]
    .iter()
    .map(PathBuf::from)
    .find(|f| f.exists())
    .unwrap_or_else(|| {
        panic!(
            "arms.txt not found beside {} nor under evidence/step5-calibration/ — \
             step5_calibration --generate writes it (§7 step 5)",
            root.display()
        )
    })
}

fn band_label(b: &Step5Band) -> &'static str {
    match b {
        Step5Band::Separated => "SEPARATED",
        Step5Band::Mixed => "MIXED",
        Step5Band::NullConsistent => "NULL-CONSISTENT",
    }
}

/// One arm file banded against ONE of its null families, with the §5 M3
/// convention applied: when the ARM has no knife-edge step, or when the
/// FAMILY has none, the max is undefined and M3 does NOT fire. Both cases
/// are forced by handing `step5_band` an absent arm M3 — the band function
/// itself stays step 5's, unmodified, so M2 is computed by one
/// implementation only.
///
/// `None` = the family is not usable (fewer than five non-void members).
fn family_band(
    root: &Path,
    base: &Path,
    stem: &str,
    family: &str,
    arm: &Step5FileReadout,
    arm_m3: Option<f64>,
) -> Option<(Step5Band, &'static str)> {
    let prefix = format!("cal-{stem}-{family}-s");
    let mut nulls: Vec<Step5FileReadout> = Vec::new();
    let mut m3s: Vec<Option<f64>> = Vec::new();
    let mut found = 0usize;
    let mut entries: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", root.display()))
        .flatten()
        .map(|e| e.path())
        .collect();
    entries.sort();
    for d in entries {
        let name = d
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !d.is_dir() || !name.starts_with(&prefix) {
            continue;
        }
        found += 1;
        let ro = step5_read_dir(&d.to_string_lossy());
        if !ro.voids.is_empty() {
            println!(
                "    {name}: VOIDS {:?} — excluded (void protocol §7)",
                ro.voids
            );
            continue;
        }
        m3s.push(m3_of(&d, base));
        nulls.push(ro);
    }
    if nulls.len() != NULLS_PER_FAMILY {
        println!(
            "    {family}: {} of {NULLS_PER_FAMILY} usable ({found} found) — family incomplete (§5)",
            nulls.len()
        );
        return None;
    }
    // §5 M3 convention (reviewers N5 and D3), conservative in both
    // directions: the §6 tree is monotone in Ls and Gs, so a non-firing
    // M3 can only move a verdict down the ladder.
    let (effective_m3, note) = if arm_m3.is_none() {
        (None, "m3: arm-empty")
    } else if m3s.iter().all(Option::is_none) {
        (None, "m3: family-empty")
    } else {
        (arm_m3, "m3: compared")
    };
    Some((step5_band(arm, effective_m3, &nulls, &m3s), note))
}

/// The step-5 CALIBRATION positive control: bands per arm file against its
/// PRIMARY null family (LESION → scat, GRAFT → local), the secondary band
/// alongside, and the §6 verdict computed by `judge::poscontrol_verdict` —
/// the rule lives in the library with its exhaustive truth-table test,
/// never inline here.
fn poscontrol(root: &Path) {
    println!(
        "== step-5 positive control (PREREG §5/§6) over {} ==",
        root.display()
    );
    let base = base_dir(root);
    let af = arms_file(root);
    println!("arms    : {}", af.display());
    let text =
        std::fs::read_to_string(&af).unwrap_or_else(|e| panic!("read {}: {e}", af.display()));

    let mut partial = false;
    let (mut lj, mut ls, mut lq) = (0usize, 0usize, 0usize);
    let (mut gj, mut gs, mut gq, mut gscat) = (0usize, 0usize, 0usize, 0usize);
    // The global print rule (§6, reviewer D2): every graft file SEPARATED
    // while the verdict is not CALIBRATED is printed verbatim, under every
    // rule, never promoted.
    let mut graft_separated: Vec<String> = Vec::new();
    let mut scat_bands: Vec<String> = Vec::new();

    for line in text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let mut it = line.split_whitespace();
        let stem = it.next().expect("a non-empty line carries an arm stem");
        // `<family>*` marks the family §6 decides on. The generator writes
        // it because the generator knows what an arm is; the reader never
        // infers a primary from the arm's name.
        let tokens: Vec<&str> = it.collect();
        let families: Vec<&str> = tokens.iter().map(|f| f.trim_end_matches('*')).collect();
        let primaries: Vec<&str> = tokens
            .iter()
            .filter(|f| f.ends_with('*'))
            .map(|f| f.trim_end_matches('*'))
            .collect();
        assert!(
            primaries.len() == usize::from(!families.is_empty()),
            "{}: {stem} must mark exactly one primary family (none for the tripwire), got {tokens:?}",
            af.display()
        );
        let dir = root.join(format!("cal-{stem}"));
        if !dir.exists() {
            println!("  {stem}: ABSENT — partial root");
            partial = true;
            continue;
        }
        assert_double_run(&dir);
        let arm = step5_read_dir(&dir.to_string_lossy());
        if !arm.voids.is_empty() {
            println!(
                "  {stem}: VOIDS {:?} — arm voided (void protocol §7)",
                arm.voids
            );
            continue;
        }

        // IDENTITY is a tripwire, not a banded comparison: any flip is a
        // kill criterion (§7), not a result.
        if stem == "identity" {
            let ok = arm.flips.is_empty();
            println!(
                "  identity: {} flips : {}",
                arm.flips.len(),
                if ok {
                    "PASS — no contamination"
                } else {
                    "KILL (§7) — the tripwire flipped"
                }
            );
            assert!(ok, "identity arm flipped {:?} — KILL (§7)", arm.flips);
            continue;
        }

        // The arm CLASS still comes from the stem — `lesion-h{h}` and
        // `graft-k{k}` are the §7 filenames, and §6 counts the two classes
        // separately. The PRIMARY family no longer does.
        let is_lesion = stem.starts_with("lesion-");
        assert!(
            is_lesion || stem.starts_with("graft-"),
            "{}: unknown arm stem {stem:?} — arms.txt is written by --generate",
            af.display()
        );
        let primary = primaries[0];
        assert!(
            families.contains(&primary),
            "{stem}: primary {primary:?} is not among its families {families:?}"
        );

        let arm_m3 = m3_of(&dir, &base);
        let quiet = arm.flips.is_empty();
        let Some((band, note)) = family_band(root, &base, stem, primary, &arm, arm_m3) else {
            println!("  {stem}: PRIMARY family {primary} unusable — arm VOIDED (§5)");
            continue;
        };
        println!(
            "  {stem}: {} flips · {note} · vs {primary} → {}{}",
            arm.flips.len(),
            band_label(&band),
            if quiet { " · QUIET" } else { "" }
        );
        // The secondary family is reported and never enters §6 (§3), but
        // for grafts its band rides in the verdict string.
        for family in families.iter().filter(|f| **f != primary) {
            match family_band(root, &base, stem, family, &arm, arm_m3) {
                Some((b, n)) => {
                    println!(
                        "    {family}: {} · {n} (reported, not banded into §6)",
                        band_label(&b)
                    );
                    if !is_lesion && *family == "scat" {
                        if b == Step5Band::Separated && !quiet {
                            gscat += 1;
                        }
                        scat_bands.push(format!("{stem} {}", band_label(&b)));
                    }
                }
                None => {
                    println!("    {family}: not usable (reported only, §6 unaffected)");
                    if !is_lesion && *family == "scat" {
                        scat_bands.push(format!("{stem} DEPLETED"));
                    }
                }
            }
        }

        let separated = band == Step5Band::Separated && !quiet;
        if is_lesion {
            lj += 1;
            ls += usize::from(separated);
            lq += usize::from(quiet);
        } else {
            gj += 1;
            gs += usize::from(separated);
            gq += usize::from(quiet);
            if separated {
                graft_separated.push(stem.to_string());
            }
        }
    }

    if partial {
        println!("\nverdict: PARTIAL ROOT — arms absent; no verdict quoted on incomplete data");
        return;
    }
    let out = poscontrol_verdict(lj, ls, lq, gj, gs, gq);
    println!(
        "\nverdict: {} (graft: {gs}/{gj} SEPARATED vs LOCAL, {gscat}/{gj} SEPARATED vs SCAT)",
        out.verdict.label()
    );
    // Every number §1's licence templates read, printed as numbers.
    println!("counts : Ls={ls} Lj={lj} Lq={lq} · Gs={gs} Gj={gj} Gq={gq} · Gscat={gscat}");
    println!("rule   : §6 rule {}", out.rule);
    if !scat_bands.is_empty() {
        println!("scat   : {}", scat_bands.join(" · "));
    }
    if out.rule == 3 {
        println!(
            "note   : {gq} of {gj} graft files produced no flips; content-sensitivity is untested \
             on those and rests on the remaining {}",
            gj - gq
        );
    }
    if out.verdict != PosControlVerdict::Calibrated && !graft_separated.is_empty() {
        println!(
            "anomaly: graft file(s) {graft_separated:?} SEPARATED vs LOCAL while the verdict is \
             {} — printed verbatim, never promoted (§6 global print rule)",
            out.verdict.label()
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--calibrate") => calibrate(),
        Some("--verify-free") => verify_free(),
        Some("--poscontrol") => {
            let root = args.get(2).unwrap_or_else(|| {
                eprintln!("usage: step5_aggregate --poscontrol <burn-root>");
                exit(1)
            });
            poscontrol(Path::new(root));
        }
        Some(dir) => aggregate(Path::new(dir)),
        None => {
            eprintln!(
                "usage: step5_aggregate --calibrate | --verify-free | --poscontrol <root> | <burn-root>"
            );
            exit(1);
        }
    }
}
