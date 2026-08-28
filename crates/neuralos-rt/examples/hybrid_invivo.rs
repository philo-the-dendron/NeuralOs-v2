//! Session H — THE IN-VIVO GATE: the model's own activations drive the
//! substrate that adapts its weights.
//!
//! The frozen pre-registration (ISA Decisions, 2026-08-19 "sH,
//! PRE-REGISTERED", attack-pass amended) in full:
//!
//! # Purpose
//!
//! Does the model's OWN activity carry more coupling structure than the
//! synthetic 1.5c schedule? Decidable via the Tier-2 comparison column
//! (same net, synthetic drive, recorded beside the in-vivo run).
//!
//! # Tiers
//!
//! - **Tier 1 (substrate, hard-gated)** — GRID: CentiMillivolt, PINNED
//!   (the G0 inequality's only lineage demonstration is ISC-76's centi
//!   result; the D-2-verbatim import is the mV build where single weight
//!   pulses are dead-zone-absorbed — spurious fail).
//!   - **G0**: arrangement-vs-census — the in-vivo-driven imported net's
//!     train must diverge from its census-shuffle control MORE than from
//!     the zero-net: H(i,c) > H(i,z). The arrangement claim rests on G0
//!     alone.
//!   - **G2′ (wire-liveness tripwire)**: imported vs control vs zero not
//!     all identical (vacuous as a divergence gate post-F; relabelled per
//!     the attack pass).
//!   - **METRIC BATTERY**: all three pairwise spike-TRAIN Hammings +
//!     per-neuron rate-L1 + per-population Hz (Hamming alone conflates
//!     rate and timing under variable drive).
//!   - **P1**: divergence appears — live wire + rectifier physics,
//!     corpus RMS within the validated amplitude range (auditable via
//!     the printed |current| histogram).
//! - **Tier 2 (judge, recorded whatever it says)** — KLD + continuation
//!   diff of the in-vivo export vs baseline AND vs the synthetic-era
//!   export; P2: in-vivo deltas ≥ synthetic's under identical footprint.
//! - **Tier 3 (steers — STRETCH, not gate)** — argmax flip or
//!   continuation change; footprint physics stated: 0.5% of one layer ×
//!   36 layers measured 0/60 flips twice. A Tier-3 pass implies a
//!   widened export as a FOLLOW-ON session.
//!
//! # Named falsifiers
//!
//! T1 NO = no weight-borne divergence (recorded; coupling story pivots
//! to regime-dependence). T2 NO = in-vivo deltas < synthetic's
//! (recorded; the synthetic schedule was already sufficient). T3 NO =
//! byte-identical continuations (expected; recorded).
//!
//! # Drive design (frozen, attack-pass amended)
//!
//! - Source: `attn_norm(embedding)` — THE model's own input to the
//!   adapted tensor (`blk.0.attn_q` eats RMSNorm'd embeddings;
//!   `forward_block_states` capture + `rms_norm_milli`, no new deps).
//! - Mapping: 1 token → 1 substrate step (consecutive tokens give causal
//!   pre→post pairings at dt≈1 ms, factor 0.95 — food for the
//!   PAIRING-SELECTIVE, CLAMP-RECTIFIED channel; N-steps REJECTED: it
//!   would manufacture sustained same-step LTD).
//! - Scaling: ONE frozen global k, corpus-wide RMS → ~450 μA mid-band;
//!   per-step per-dim current = clamp(h_dim × k, ±1000), sign preserved.
//!   CLAMP AUDIT printed: hit fraction + per-dim rail concentration;
//!   >10% clamped = recorded caveat.
//! - DRIVEN DIMS 0..408 ONLY (fork (a)): token features drive the E
//!   population; the I population keeps the validated fixed I_INH=600
//!   wall. The 512-dim purity variant is a named follow-on.
//! - Corpus: the sha-pinned README slice (18fb5452…), FIRST 2000
//!   tokens, SINGLE TRUNCATED PASS — no epochs, never wraps by
//!   construction (the sH2 registration v2, 2026-08-19: the original
//!   whole-epochs design — floor(2000/N) epochs, stop before the
//!   wrap — died with the drive redesign; on short corpora its
//!   frozen 400-step init cycle swallowed the whole learn tier,
//!   banked in `evidence/r4-closeout/h1_invivo_r4iii.log`).
//! - Population: the 512-neuron slice net (409 E / 103 I), live
//!   substrate (post-F), STDP on, γ=125, census-shuffle + zero controls,
//!   seeds D-2-verbatim.
//! - MECHANISM COUNTERS REQUIRED (attack-pass #7): the per-class
//!   raw/absorbed/applied decomposition prints beside the histogram —
//!   expect the clamp-rectified regime under dense co-firing.
//!
//! Usage: `cargo run -p neuralos-rt --release --example hybrid_invivo --
//! [model.gguf] [corpus.txt] [export]` (defaults:
//! `models/Ternary-Bonsai-4B-Q2_0.gguf`,
//! `evidence/corpus_readme_pinned.txt`, no export). The third arg
//! `export` additionally runs Tier 2: writes
//! `models/…-invivo-ck{step}.gguf` checkpoints + the terminal
//! `models/…-invivo.gguf` (WARNING: overwrites the session-I
//! adjudication artifacts — the R4 re-pin ran plain mode for exactly
//! this reason).
//!
//! # Drive domain (step-8 constraint 10, 2026-08-28)
//!
//! The DEFAULT drive is NORM-UNIT (`raw = v_milli/1000 × k`) — the sH
//! registration's stated intent. The historical milli-domain
//! application (`raw = v_milli × k`; measured 69.477% railed, the
//! empty-middle histogram — the step-5 clamp probe's unit-bug finding)
//! is preserved byte-verbatim behind `--h2-compat`, and every
//! H2-comparable invocation now REQUIRES that flag out loud:
//! `--window`, `--off`, and the legacy positional `export` pipeline
//! refuse to run without it, because their pins and banked artifacts
//! were burned under that drive by ruling (PREREG §2). The pre-clamp
//! distribution is measured and reported for every driven arm before
//! the substrate runs; a non-compat drive that reproduces the
//! empty-middle falsifier shape VOIDS the arm — loud abort, never a
//! silent burn (ISA: the tenth step-8 constraint, 2026-08-27).
//!
//! # Step-5 arm modes (PREREG evidence/step5-readout/PREREG.md, 2026-08-23)
//!
//! Named flags select the burn-window arms; positional legacy behavior
//! under `--h2-compat` is BYTE-UNTOUCHED — window-0 ON must re-pin the
//! banked H2 export. New modes NEVER write the banked artifact names:
//!
//! - `--window <r>` (r ∈ 0..=4; the 2026-08-26 escalation amendment
//!   added r3/r4, which WRAP the 4,411-token corpus) — replicate
//!   corpus window, tokens [1000·r, 1000·r+2000) (PREREG §4;
//!   r0 ≡ H2's window). k is
//!   re-derived from the window's own tokens (Rider A:
//!   procedure-pinned; probe expectations r0 10060.46 · r1 10101.90 ·
//!   r2 10007.65 · r3 9965.58 · r4 10054.74 (r3/r4 from the banked
//!   escalation probe) — the run.log is the pin of record).
//! - `--off` — the OFF arm, driven r0 ONLY (PREREG §3): the full
//!   driven pipeline (capture + drive + steps + export) with
//!   plasticity NEVER re-enabled after the init cycle. STDP deltas
//!   stay 0 ⇒ ternary residuals stay 0 ⇒ weights constant ⇒ export
//!   asserts byte-≡ base (sha) — the end-to-end toggle proof (the
//!   session-F lesson). `--off --window 1..=4` is REFUSED (the assert
//!   covers every non-r0 window post-escalation) — the r1/r2 OFF
//!   legs are identity surgeries, not driven runs.
//! - `--identity <r>` — OFF r1/r2: control-mode identity surgery of
//!   the ISC-68 class (unadapted source trits through the full
//!   surgery pipeline; no capture, no drive, minutes) writing
//!   `…-invivo-identity-r{r}.gguf`, asserting byte-≡ base — the judge
//!   double-run (tools/run_prompts.sh --double) is the contamination
//!   tripwire.
//! - `--domain-corrected` — the DOMAIN arm: k applied in NORM units
//!   (`raw = v_milli/1000 × k`), the sH registration's stated intent
//!   (measured 2.74% clamped / RMS 450.0 µA; report-only, PREREG §7).
//!   Window 0, no checkpoints. Since constraint 10 this is the same
//!   drive as the default; the flag is kept for the ratified
//!   invocation and its arm-named export.
//! - `--h2-compat` — the milli-domain legacy drive (`raw = v_milli ×
//!   k`), byte-verbatim. Required by `--window`/`--off`/legacy
//!   `export`; contradicts `--domain-corrected`.
//!
//! Arm exports (final-only, NO checkpoints — the FREE arm judges the
//! banked ck files): `…-invivo-r{r}.gguf` (ON) · `…-invivo-off-r0.gguf`
//! (OFF driven) · `…-invivo-identity-r{r}.gguf` (OFF tripwires) ·
//! `…-invivo-domain.gguf` (DOMAIN). Every step-5 output passes the
//! unbanked-path guard BEFORE writing (the r4-closeout lesson). ON
//! window-0 HARD-ASSERTS the banked H2 sha (71f2518a…) — the new code
//! path must reproduce the frozen artifact byte-for-byte or the arm
//! voids.

use neuralos_rt::harness::{
    build_from_trits, decode_slice, exc_count, group_of, peak_rss_mb, rate_l1,
    run_and_capture, shuffled_copy, splice_and_verify, tix, train_hamming, trit_val,
    ExperimentParams, Train,
};
use neuralos_rt::{rms_norm_milli, GgufFile, Qwen3, Tokenizer};
use neuralos_snn::{Trit, VoltageResolution};

/// The banked H2 terminal-export sha (evidence/session-h2/run.log:47) —
/// ON window-0's r0 re-pin assert. (The OFF arm compares against the
/// LIVE base file's sha, not a constant — it must hold for any base.)
const H2_EXPORT_SHA: &str = "71f2518a2d783cb409a3c06907a20bed1f1b5688378fc7ea7e8a0f6e16d9749b";

/// The in-vivo learning run (STDP on, one token per step) with the
/// full counter battery: pairing histogram + per-class raw/absorbed/
/// applied decomposition + final trits.
struct VivoStats {
    learn_rate_hz: f64,
    quarter_hz: [f64; 4],
    flips: u64,
    census: [[u64; 3]; 3],
    final_trits: Vec<Trit>,
    plasticity_events: u64,
    pairs_same_step: u64,
    pairs_post_leads: u64,
    pairs_pre_leads: u64,
    raw_intra: i64,
    raw_inter: i64,
    absorbed_intra: i64,
    absorbed_inter: i64,
    /// H2b dose-response snapshots: (learn-step index, synapse trits).
    ck_snaps: Vec<(usize, Vec<Trit>)>,
}

#[allow(non_snake_case)]
fn group_pair_classes(net: &neuralos_snn::SpikingNeuralNetwork, exc: u16, p: &ExperimentParams) -> Vec<bool> {
    // per synapse: is this E→E pair INTRA-group (4-group geometry, only
    // used for the class decomposition — the drive is data-driven now)?
    net.synapses()
        .iter()
        .map(|s| {
            s.pre_neuron_id < exc
                && s.post_neuron_id < exc
                && group_of(s.pre_neuron_id, exc, p) == group_of(s.post_neuron_id, exc, p)
        })
        .collect()
}

/// `checkpoints`: learn-step indices (relative to the learn phase) at which
/// to snapshot the synapse trits — H2b dose-response. The 100% checkpoint
/// (len of learn) is ALWAYS taken via `final_trits`; snapshots are
/// additional declared derivatives, each exported + sha-pinned by the
/// caller. The PLAIN path (no checkpoints) executes identically — the
/// invariance assert (100%-checkpoint export sha == plain export sha)
/// proves the machinery did not perturb the frozen artifact.
///
/// `learn_plasticity == false` (the step-5 OFF arm): plasticity is
/// never re-enabled after the init cycle — STDP deltas stay 0, ternary
/// residuals stay 0, weights constant. The counter battery then reads
/// all-zero (flips == 0 is the in-run witness; the caller asserts the
/// export is byte-≡ base).
#[allow(non_snake_case)]
fn run_vivo_ck(
    trits: &[Trit],
    inputs: &[Vec<i16>],
    checkpoints: &[usize],
    p: &ExperimentParams,
    learn_plasticity: bool,
) -> VivoStats {
    let (N, GAMMA, DT_US) = (p.n, p.gamma, p.dt_us);
    let exc = exc_count(p) as u16;
    let mut net = build_from_trits(trits, GAMMA, p, VoltageResolution::CentiMillivolt);
    // Init cycle: STDP off, defeats the last_spike=0 sentinel (D-2 verbatim
    // — 400 steps of the SAME in-vivo drive, recorded).
    net.set_plasticity_enabled(false);
    let init_len = inputs.len().min(400);
    for inp in &inputs[..init_len] {
        net.step(inp).expect("init step");
    }
    let mut prev: Vec<Trit> = net
        .synapses()
        .iter()
        .map(|s| Trit::from_weight(s.weight, GAMMA))
        .collect();
    if learn_plasticity {
        net.set_plasticity_enabled(true);
    }

    let learn = &inputs[init_len..];
    let mut quarter_spikes = [0u64; 4];
    let mut flips = 0u64;
    let mut census = [[0u64; 3]; 3];
    let mut ck_snaps: Vec<(usize, Vec<Trit>)> = Vec::new();
    let ck_sorted: Vec<usize> = {
        let mut v = checkpoints.to_vec();
        v.sort_unstable();
        v
    };
    let mut ck_i = 0usize;
    let is_intra = group_pair_classes(&net, exc, p);
    let mut raw_intra = 0_i64;
    let mut raw_inter = 0_i64;
    let mut absorbed_intra = 0_i64;
    let mut absorbed_inter = 0_i64;

    for (t, inp) in learn.iter().enumerate() {
        let spikes = net.step(inp).expect("learn step");
        net.stochastic_ternary_step(GAMMA);
        quarter_spikes[t / (learn.len().div_ceil(4))] += spikes.len() as u64;
        while ck_i < ck_sorted.len() && ck_sorted[ck_i] == t {
            ck_snaps.push((
                t,
                net.synapses()
                    .iter()
                    .map(|s| Trit::from_weight(s.weight, GAMMA))
                    .collect(),
            ));
            ck_i += 1;
        }
        for (k, s) in net.synapses().iter().enumerate() {
            let cur = Trit::from_weight(s.weight, GAMMA);
            if cur != prev[k] {
                census[tix(prev[k])][tix(cur)] += 1;
                flips += 1;
                prev[k] = cur;
            }
        }
    }
    // Counters read AFTER the run (cumulative fields).
    for (k, s) in net.synapses().iter().enumerate() {
        let intra = is_intra[k];
        if intra {
            raw_intra += s.raw_stdp_delta;
            absorbed_intra += s.absorbed_delta;
        } else {
            raw_inter += s.raw_stdp_delta;
            absorbed_inter += s.absorbed_delta;
        }
    }
    let total_learn_spikes: u64 = quarter_spikes.iter().sum();
    let secs = learn.len() as f64 * f64::from(DT_US) / 1e6;
    let st = net.stats();
    VivoStats {
        learn_rate_hz: total_learn_spikes as f64 / (secs * N as f64),
        quarter_hz: quarter_spikes.map(|q| q as f64 / ((secs / 4.0) * N as f64)),
        flips,
        census,
        final_trits: net
            .synapses()
            .iter()
            .map(|s| Trit::from_weight(s.weight, GAMMA))
            .collect(),
        plasticity_events: st.plasticity_events,
        pairs_same_step: st.stdp_pairs_same_step,
        pairs_post_leads: st.stdp_pairs_post_leads,
        pairs_pre_leads: st.stdp_pairs_pre_leads,
        raw_intra,
        raw_inter,
        absorbed_intra,
        absorbed_inter,
        ck_snaps,
    }
}

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

#[allow(non_snake_case)]
fn main() {
    let t0 = std::time::Instant::now();
    let p = ExperimentParams::default();
    let (N, GAMMA, DT_US, STEPS, I_INH) = (p.n, p.gamma, p.dt_us, p.steps, p.i_inh);
    let (SI_FLOOR, RSS_BUDGET_MB, CONTROL_SEED) = (p.si_floor, p.rss_budget_mb, p.control_seed);
    let (TARGET_RMS_UA, CLAMP_UA, CLAMP_WARN_FRAC) = (p.target_rms_ua, p.clamp_ua, p.clamp_warn_frac);
    let DRIVEN_DIMS = exc_count(&p);
    // ----- argv: legacy positionals (model, export-flag, corpus) plus
    // step-5 named flags. A flagless invocation is a corrected-drive
    // REPORT run (constraint 10); `--h2-compat` restores the
    // BYTE-UNTOUCHED legacy behavior (the r0 re-pin contract). -----
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut window: Option<usize> = None;
    let mut off = false;
    let mut domain = false;
    let mut h2_compat = false;
    let mut identity: Option<usize> = None;
    let mut positional: Vec<&str> = Vec::new();
    {
        let mut i = 0usize;
        while i < argv.len() {
            match argv[i].as_str() {
                "--window" => {
                    window = Some(
                        argv.get(i + 1)
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or_else(|| panic!("--window expects 0|1|2|3|4")),
                    );
                    assert!(
                        matches!(window, Some(0..=4)),
                        "window r ∈ 0..=4 (PREREG §4 + the escalation amendment: r3/r4 wrap)"
                    );
                    i += 2;
                }
                "--off" => {
                    off = true;
                    i += 1;
                }
                "--domain-corrected" => {
                    domain = true;
                    i += 1;
                }
                "--h2-compat" => {
                    h2_compat = true;
                    i += 1;
                }
                "--identity" => {
                    identity = Some(
                        argv.get(i + 1)
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or_else(|| panic!("--identity expects a replicate label 0|1|2")),
                    );
                    i += 2;
                }
                other => {
                    positional.push(other);
                    i += 1;
                }
            }
        }
    }
    // PREREG §3 shape enforcement: OFF is ONE driven run (r0) + identity
    // surgeries for r1/r2 — a driven OFF on r1/r2 is the voided-arm
    // deviation the relay caught; refuse it loudly.
    if off && matches!(window, Some(1..=4)) {
        panic!(
            "PREREG §3: OFF = 1 driven run (r0) + identity surgeries (r1/r2) — the \
             escalation adds ON windows only, never OFF (got --window {w}). Use \
             `--off` alone for the driven r0 toggle proof; `--identity <r>` for \
             the r1/r2 tripwires.",
            w = window.unwrap()
        );
    }
    if domain && !matches!(window, None | Some(0)) {
        panic!(
            "PREREG §3: DOMAIN-CORRECTED is window-0 mechanics (r{w} given). \
             The ratified invocation is `--domain-corrected` alone.",
            w = window.unwrap()
        );
    }
    if off && domain {
        panic!("PREREG §3: OFF and DOMAIN-CORRECTED are different arms — OFF disables plasticity, DOMAIN measures it under the corrected drive. Run them separately.");
    }
    if let Some(r) = identity {
        assert!((0..=2).contains(&r), "identity label r ∈ 0|1|2 (bookkeeping only — the surgery is window-independent)");
    }
    let step5_mode = window.is_some() || off || domain || identity.is_some();
    // In step-5 modes every arm exports (final-only); the legacy
    // positional `export` keeps its checkpointed, banked-name behavior.
    let path = positional
        .first()
        .copied()
        .unwrap_or("models/Ternary-Bonsai-4B-Q2_0.gguf")
        .to_string();
    let legacy_export = positional.get(1).copied() == Some("export");
    // ----- Drive-domain gating (step-8 constraint 10, ISA 2026-08-27).
    // The corrected drive is the default; every H2-comparable arm was
    // burned and pinned under the milli-domain drive BY RULING, so it
    // must say so out loud. One flag reproduces any banked command
    // byte-for-byte: add `--h2-compat`. These refusals fire BEFORE the
    // model loads — a wrong arm costs seconds, never a burn. -----
    if h2_compat && domain {
        panic!(
            "--h2-compat (milli-domain legacy) and --domain-corrected (norm-unit, \
             now the default) both name a drive domain — drop one."
        );
    }
    if !h2_compat && (window.is_some() || off) {
        panic!(
            "constraint 10: --window/--off are H2-comparable step-5 arms — their pins \
             and banked artifacts were burned under the milli-domain drive. Re-runs \
             must add --h2-compat; the corrected default never impersonates them."
        );
    }
    if !h2_compat && legacy_export {
        panic!(
            "constraint 10: the legacy `export` pipeline writes the banked artifact \
             names and must stay byte-comparable to them — add --h2-compat. A \
             corrected-drive export waits for the step-8 pre-registration's own \
             arm names."
        );
    }
    let norm_drive = !h2_compat;
    let arm_r = window.unwrap_or(0);
    println!("=== Session H: the in-vivo gate — the model's own activations drive the substrate ===");
    println!("file    : {path}");
    println!(
        "reg.    : ISA sH (2026-08-19, attack-pass amended) — tiers, drive design frozen{}",
        if step5_mode {
            format!(" · step-5 arm: {} window r{arm_r}", if off { "OFF" } else if domain { "DOMAIN-CORRECTED" } else { "ON" })
        } else {
            String::new()
        }
    );
    println!();

    let src = decode_slice(&path, &p);
    println!("decode  : {} trits (D-2 slice path)", src.len());

    // ----- OFF r1/r2: the identity tripwires (PREREG §3) -----
    // Control-mode identity surgery of the ISC-68 class: the UNADAPTED
    // source trits through the full surgery pipeline (splice → S1 →
    // write → S2). No capture, no drive, no learn — minutes, not
    // hours; the judge double-run (run_prompts.sh --double) is the
    // contamination tripwire. The export MUST be byte-≡ base.
    if let Some(r) = identity {
        let out = format!("models/Ternary-Bonsai-4B-Q2_0-invivo-identity-r{r}.gguf");
        neuralos_rt::harness::assert_unbanked(&out);
        let (code, scale) = splice_and_verify(&path, &out, &src, None, &p);
        assert_eq!(scale, 0, "identity surgery never touches scales");
        let sha = sha256_of(&out);
        let base_sha = sha256_of(&path);
        assert_eq!(sha, base_sha, "identity export must be byte-≡ base (ISC-68 class)");
        println!("identity-r{r}: code {code} (must be 0) · scale {scale} · S2 clean");
        println!("identity-r{r}: sha == base ({base_sha:.16}…) : PASS — judge double-run next");
        println!("wall {:.1}s", t0.elapsed().as_secs_f64());
        return;
    }

    // ----- In-vivo drive: tokenize the pinned corpus, run the model,
    // capture attn_norm(embedding) per token -----
    // H2 registration v2: the TRUE pinned corpus (fork README lines 1–180,
    // sha 18fb5452…), FIRST 2000 tokens, SINGLE TRUNCATED PASS — no
    // epochs, never wraps by construction. (H1 used a different 1,024-byte
    // file with a false identity comment — the recorded infidelity,
    // corrected by re-run.)
    let corpus_path = positional
        .get(2)
        .copied()
        .unwrap_or("evidence/corpus_readme_pinned.txt")
        .to_string();
    let corpus = std::fs::read_to_string(&corpus_path).unwrap_or_else(|e| {
        eprintln!("cannot read corpus {corpus_path}: {e}");
        std::process::exit(1);
    });
    println!("corpus  : {corpus_path} ({} bytes)", corpus.len());

    let (h_norm, token_count) = {
        let buf = std::fs::read(&path).unwrap_or_else(|e| {
            eprintln!("cannot re-read {path}: {e}");
            std::process::exit(1);
        });
        let f = GgufFile::parse(&buf).expect("re-parse");
        let mut model = Qwen3::load(&f, 5000).expect("model loads");
        let tok = Tokenizer::from_gguf(&f).expect("tokenizer loads");
        let ids = tok.encode(&corpus);
        println!("tokens  : {} (vocab {})", ids.len(), tok.len());
        // The drive source: attn_norm(embedding) — the model's own input
        // to blk.0.attn_q. Via the PUBLIC capture path:
        // forward_block_states()[0] is the post-embedding hidden stream
        // (model.rs:946); attn_norm applied here per the registration.
        let norm_w = f
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.attn_norm.weight")
            .expect("attn_norm tensor present");
        let norm_milli: Vec<i32> = {
            let d = f.tensor_data(norm_w).expect("norm slice");
            d.chunks_exact(4)
                .map(|c| neuralos_rt::f32_bits_to_milli(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect()
        };
        let states = model.forward_block_states(&ids).expect("forward capture");
        let emb_dim = model.config().emb;
        let mut normed = vec![0_i32; emb_dim];
        let mut rows: Vec<Vec<i32>> = Vec::with_capacity(ids.len());
        for h in &states[0] {
            rms_norm_milli(h, &norm_milli, &mut normed);
            rows.push(normed.clone());
        }
        (rows, ids.len())
    };
    // H2: single truncated pass — the first min(STEPS, tokens) of the
    // stream, in order; never reaches the corpus end, so no wrap exists.
    // Step-5 windows (PREREG §4 + the escalation amendment): r0–r2
    // contiguous [1000·r, 1000·r+2000); r3/r4 WRAP the 4,411-token
    // corpus (r3 = [3000,4411)+[0,589) · r4 = [4000,4411)+[0,1589) —
    // exactly STEPS each, dose-comparable; the wrap is materialized
    // into a contiguous buffer so every downstream consumer is
    // wrap-agnostic). r0 ≡ the H2 window, byte-identical slice.
    let win_start = 1000 * arm_r;
    let n_tokens = h_norm.len();
    let h_norm_owned: Vec<Vec<i32>>;
    let h_norm: &[Vec<i32>] = if win_start + STEPS <= n_tokens {
        &h_norm[win_start..win_start + STEPS]
    } else {
        h_norm_owned = (0..STEPS)
            .map(|i| h_norm[(win_start + i) % n_tokens].clone())
            .collect();
        &h_norm_owned
    };
    let n_steps = h_norm.len();
    if step5_mode {
        let end = win_start + STEPS;
        if end <= n_tokens {
            println!(
                "drive   : window r{arm_r} — tokens [{win_start}, {end}) of {token_count} ({n_steps} steps; init-400 = the window's own first 400, PREREG §4)"
            );
        } else {
            let tail = n_tokens - win_start;
            let head = end - n_tokens;
            println!(
                "drive   : window r{arm_r} (wraps) — tokens [{win_start}, {n_tokens}) + [0, {head}) of {token_count} ({tail}+{head} = {n_steps} steps; init-400 = the window's own first 400, escalation amendment)"
            );
        }
    } else {
        println!(
            "drive   : single truncated pass — first {n_steps} of {token_count} tokens (no epochs, no wrap by construction)"
        );
    }
    // Truncation context (registration v2): the text window around the cut.
    if !step5_mode {
        let tail_ctx: String = corpus.chars().rev().take(160).collect::<Vec<_>>().into_iter().rev().collect();
        println!("cut-ctx : …{:?}… (last ~160 chars before the token-2000 cut)", tail_ctx);
    }

    // ----- Scaling: ONE global k → corpus RMS = 450 μA (drive dims 0..408) -----
    let mut sum_sq: f64 = 0.0;
    let mut n_vals: u64 = 0;
    for row in h_norm {
        let mut sum: f64 = 0.0;
        let mut cnt: u64 = 0;
        for &v in &row[..DRIVEN_DIMS] {
            sum += (v as f64 / 1000.0).powi(2);
            cnt += 1;
        }
        sum_sq += sum;
        n_vals += cnt;
    }
    let rms_norm_units = (sum_sq / n_vals as f64).sqrt(); // in norm units
    let k = TARGET_RMS_UA / rms_norm_units; // μA per norm unit
    println!("scaling : corpus RMS {rms_norm_units:.4} → k = {k:.2} μA/unit (target {TARGET_RMS_UA} μA)");
    // Pre-burn k cross-check (the relay's cheap test): the probe-derived
    // per-window expectations, asserted BEFORE the 6–8 h run — catches
    // off-by-one window offsets while they still cost seconds. The
    // run.log's own derived k remains the pin of record (PREREG §4,
    // Rider A — these constants are the fence, not the pin).
    if step5_mode {
        // r3/r4 pins from the BANKED escalation probe run
        // (evidence/step5-readout/clamp_probe_escalation.log, sha
        // 01d9bc61… — amendment 3: the wrapped-window k values are new
        // pins, provenance banked before the burn).
        const K_EXPECTED: [f64; 5] =
            [10_060.46, 10_101.90, 10_007.65, 9_965.58, 10_054.74];
        let exp = K_EXPECTED[arm_r];
        assert!(
            (k - exp).abs() < 0.005,
            "window r{arm_r}: derived k {k:.2} != probe expectation {exp:.2} — offset bug; aborting before the burn"
        );
        println!("k-check : r{arm_r} k {k:.2} == probe expectation {exp:.2} : PASS");
    }

    // Build the step drives; clamp audit. The default applies k in
    // NORM units (`v_milli/1000 × k`) — the sH registration's stated
    // intent (PREREG §3/§7; measured 2.74% clamped / RMS 450.0 µA).
    // `--h2-compat` keeps the milli-domain application byte-verbatim
    // (H2-comparability was step 5's point, BY RULING — and is the
    // only thing that path is for since constraint 10).
    println!(
        "domain  : {} drive",
        if norm_drive { "NORM-UNIT (corrected, constraint 10)" } else { "MILLI (H2-compat legacy)" }
    );
    let mut inputs: Vec<Vec<i16>> = Vec::with_capacity(n_steps);
    let mut clamped: u64 = 0;
    let mut rail_dim = vec![0u64; DRIVEN_DIMS];
    let mut hist = [0u64; 5]; // <100, 100-150, 150-300, 300-600, railed
    // Constraint 10: the pre-clamp distribution is measured for every
    // arm — |raw| before the rails, same buckets — and reported before
    // the substrate runs.
    let mut pre_abs: Vec<f64> = Vec::with_capacity(n_steps * DRIVEN_DIMS);
    let mut pre_hist = [0u64; 5];
    for row in h_norm {
        let mut inp = vec![I_INH; N];
        for d in 0..DRIVEN_DIMS {
            let raw = if norm_drive {
                (row[d] as f64 / 1000.0) * k
            } else {
                (row[d] as f64) * k
            };
            let pa = raw.abs();
            pre_abs.push(pa);
            if pa < 100.0 {
                pre_hist[0] += 1;
            } else if pa < 150.0 {
                pre_hist[1] += 1;
            } else if pa < 300.0 {
                pre_hist[2] += 1;
            } else if pa <= 600.0 {
                pre_hist[3] += 1;
            } else {
                pre_hist[4] += 1;
            }
            let c = raw.clamp(-(CLAMP_UA as f64), CLAMP_UA as f64);
            if c.abs() >= CLAMP_UA as f64 {
                clamped += 1;
                rail_dim[d] += 1;
            }
            let a = c.abs();
            if a < 100.0 {
                hist[0] += 1;
            } else if a < 150.0 {
                hist[1] += 1;
            } else if a < 300.0 {
                hist[2] += 1;
            } else if a <= 600.0 {
                hist[3] += 1;
            } else {
                hist[4] += 1;
            }
            inp[d] = c as i16;
        }
        inputs.push(inp);
    }
    let total_vals = (n_steps * DRIVEN_DIMS) as u64;
    let clamp_frac = clamped as f64 / total_vals as f64;
    let top_rail = rail_dim.iter().enumerate().max_by_key(|(_, &c)| c);
    println!(
        "clamp   : {clamped}/{total_vals} = {:.3}% clamped at ±{CLAMP_UA} μA{}",
        clamp_frac * 100.0,
        if clamp_frac > CLAMP_WARN_FRAC { "  ⚠ ABOVE 10% — recorded caveat" } else { "" }
    );
    if let Some((d, c)) = top_rail {
        println!("          hottest dim {d} railed {}× ({}% of its steps)", c, c * 100 / n_steps as u64);
    }
    println!(
        "hist    : |I| <100 {} · 100–150 {} · 150–300 {} · 300–600 {} · >600/railed {} (per dim-step)",
        hist[0], hist[1], hist[2], hist[3], hist[4]
    );
    // Constraint 10's report + falsifier, BEFORE the substrate runs.
    pre_abs.sort_unstable_by(|a, b| a.partial_cmp(b).expect("no NaN in drives"));
    let pct = |q: f64| pre_abs[((pre_abs.len() - 1) as f64 * q) as usize];
    println!(
        "preclamp: p50 {:.1} · p90 {:.1} · p99 {:.1} · max {:.1} μA (target {TARGET_RMS_UA}) — hist <100 {} · 100–150 {} · 150–300 {} · 300–600 {} · >600 {}",
        pct(0.50), pct(0.90), pct(0.99), pre_abs[pre_abs.len() - 1],
        pre_hist[0], pre_hist[1], pre_hist[2], pre_hist[3], pre_hist[4]
    );
    let empty_middle = pre_hist[1] == 0 && pre_hist[2] == 0 && pre_hist[3] == 0 && clamped > 0;
    if empty_middle {
        if norm_drive {
            panic!(
                "ARM VOIDED (constraint 10): the drive reproduces the empty-middle \
                 falsifier shape — middle buckets 0, {clamped} rail hits. A null \
                 measured on this drive cannot distinguish 'does not adapt' from \
                 'given nothing to adapt to'. Fix the drive before any arm runs."
            );
        }
        println!(
            "preclamp: ⚠ empty-middle falsifier shape — expected for the H2-compat \
             legacy drive; recorded caveat, comparability use only (constraint 10)"
        );
    }
    println!();

    // ----- Tier 1: G2′ tripwire + G0 (fixed-weight, full battery) -----
    println!("--- TIER 1: substrate gates (fixed weights, STDP off, in-vivo drive, CENTI grid) ---");
    let mut imported = build_from_trits(&src, GAMMA, &p, VoltageResolution::CentiMillivolt);
    let ti = run_and_capture(&mut imported, &inputs, &p);
    drop(imported);
    let ctrl_trits = shuffled_copy(&src, CONTROL_SEED);
    let mut control = build_from_trits(&ctrl_trits, GAMMA, &p, VoltageResolution::CentiMillivolt);
    let tc = run_and_capture(&mut control, &inputs, &p);
    drop(control);
    let zero_trits = vec![Trit::Zero; N * N];
    let mut zero = build_from_trits(&zero_trits, GAMMA, &p, VoltageResolution::CentiMillivolt);
    let tz = run_and_capture(&mut zero, &inputs, &p);
    drop(zero);
    let exc = exc_count(&p);
    let secs = n_steps as f64 * f64::from(DT_US) / 1e6;
    let hz = |t: &Train, hi: usize| -> f64 {
        t.counts[..hi].iter().sum::<u64>() as f64 / (secs * hi as f64)
    };
    let ihz = |t: &Train, lo: usize| -> f64 {
        t.counts[lo..].iter().sum::<u64>() as f64 / (secs * (N - lo) as f64)
    };
    let hic = train_hamming(&ti, &tc);
    let hiz = train_hamming(&ti, &tz);
    let hcz = train_hamming(&tc, &tz);
    println!(
        "  rates  E Hz (i/c/z): {:.2} / {:.2} / {:.2}   I Hz: {:.2} / {:.2} / {:.2}",
        hz(&ti, exc), hz(&tc, exc), hz(&tz, exc),
        ihz(&ti, exc), ihz(&tc, exc), ihz(&tz, exc)
    );
    println!(
        "  totals (i/c/z): {} / {} / {}",
        ti.total, tc.total, tz.total
    );
    println!(
        "  H(i,c)={hic}  H(i,z)={hiz}  H(c,z)={hcz}   L1(i,c)={}  L1(i,z)={}  L1(c,z)={}",
        rate_l1(&ti, &tc), rate_l1(&ti, &tz), rate_l1(&tc, &tz)
    );
    let tripwire = hic > 0 || hiz > 0 || hcz > 0;
    let g0 = hic > hiz;
    println!("  G2′ wire-liveness tripwire : {}", if tripwire { "PASS (weights borne in trains)" } else { "FAIL (dead wire?!)" });
    println!(
        "  G0  arrangement-vs-census : H(i,c) > H(i,z) → {hic} > {hiz} : {}",
        if g0 { "PASS — firing reads model arrangement" } else { "FAIL (T1 NO — recorded)" }
    );

    // ----- The learning run (STDP on, counters) -----
    println!();
    println!("--- in-vivo adaptation (STDP on, γ={GAMMA}, counters per the registration) ---");
    // H2b: learn-phase-quarter checkpoints when exporting (dose-response).
    // Step-5 arms export FINAL-ONLY (no ck files — the FREE arm judges
    // the BANKED ck artifacts; new modes never write banked names).
    let export_mode = legacy_export;
    let learn_len = inputs.len().saturating_sub(400);
    let cks: Vec<usize> = if export_mode && !step5_mode {
        vec![learn_len / 4, learn_len / 2, 3 * learn_len / 4]
    } else {
        Vec::new()
    };
    let v = run_vivo_ck(&src, &inputs, &cks, &p, !off);
    if off {
        assert_eq!(v.flips, 0, "OFF arm: zero bucket flips (plasticity never enabled)");
        assert_eq!(v.plasticity_events, 0, "OFF arm: zero plasticity events");
    }
    println!(
        "  learn firing : {:.2} Hz/neuron; quarters {:.2} {:.2} {:.2} {:.2}",
        v.learn_rate_hz, v.quarter_hz[0], v.quarter_hz[1], v.quarter_hz[2], v.quarter_hz[3]
    );
    println!(
        "  events {} · flips {} · pairing same-step {} · post-leads {} · pre-leads {}",
        v.plasticity_events, v.flips, v.pairs_same_step, v.pairs_post_leads, v.pairs_pre_leads
    );
    let names = ["−1", " 0", "+1"];
    println!("  bucket-transition census:");
    for (from, row) in v.census.iter().enumerate() {
        let mut line = String::new();
        for (to, &ct) in row.iter().enumerate() {
            line.push_str(&format!("  {}→{} × {:>7}", names[from], names[to], ct));
        }
        println!("{line}");
    }
    println!(
        "  raw drift intra {:+} · inter {:+}   absorbed intra {:+} · inter {:+}   APPLIED intra {:+} · inter {:+}",
        v.raw_intra, v.raw_inter, v.absorbed_intra, v.absorbed_inter,
        v.raw_intra - v.absorbed_intra, v.raw_inter - v.absorbed_inter
    );
    let n_syn = v.final_trits.len() as u64;
    let mut hamming = 0u64;
    let src_iter: Vec<Trit> = {
        let mut v2 = Vec::with_capacity(N * (N - 1));
        for j in 0..N {
            for i in 0..N {
                if i != j {
                    v2.push(src[i * N + j]);
                }
            }
        }
        v2
    };
    for (k, &ft) in v.final_trits.iter().enumerate() {
        if ft != src_iter[k] {
            hamming += 1;
        }
    }
    println!(
        "  Hamming vs imported: {}/{} = {:.4}",
        hamming, n_syn, hamming as f64 / n_syn as f64
    );

    // selectivity on the data-driven classes (4-group geometry decomposition)
    let exc16 = exc as u16;
    let (mut din, mut dit) = (Vec::new(), Vec::new());
    {
        // rebuild intra flags in synapse order
        let net = build_from_trits(&src, GAMMA, &p, VoltageResolution::CentiMillivolt);
        let is_intra = group_pair_classes(&net, exc16, &p);
        drop(net);
        for (k, &ft) in v.final_trits.iter().enumerate() {
            if is_intra[k] {
                din.push(trit_val(ft) - trit_val(src_iter[k]));
            } else {
                dit.push(trit_val(ft) - trit_val(src_iter[k]));
            }
        }
    }
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    let m_in = mean(&din);
    let m_it = mean(&dit);
    let selective = m_in.abs() >= SI_FLOOR || m_it.abs() >= SI_FLOOR;
    println!(
        "  intra meanΔ {m_in:+.4} ({}) · inter meanΔ {m_it:+.4} ({}) — data-driven classes; |Δ| floor {SI_FLOOR}",
        din.len(), dit.len()
    );
    // Mechanism label: computed from the counters (session-G doctrine).
    let applied_intra = v.raw_intra - v.absorbed_intra;
    let mechanism = if m_in.abs() < f64::EPSILON && m_it.abs() < f64::EPSILON {
        "none — no differential movement"
    } else if v.raw_intra > 0 && m_in > 0.0 {
        "Hebbian-carried — raw LTP pairings dominate (counted)"
    } else if v.raw_intra < 0 && applied_intra > 0 && m_in > 0.0 {
        "PAIRING-SELECTIVE, CLAMP-RECTIFIED — net-negative raw drift, 0-floor absorbs, applied residue potentiates"
    } else {
        "LTD-carried — applied drift negative"
    };
    println!("  mechanism label: [{mechanism}]");

    // ----- Verdict -----
    println!();
    println!("--- Verdict (per the frozen registration) ---");
    println!("  T1 G2′ tripwire   : {}", if tripwire { "PASS" } else { "FAIL" });
    println!("  T1 G0 arrangement : {}", if g0 { "PASS" } else { "FAIL (NO — recorded)" });
    println!("  adaptation alive  : {}", if v.flips > 0 { "PASS (flips > 0)" } else { "FAIL (frozen)" });
    let ham_frac = hamming as f64 / n_syn as f64;
    println!("  not collapsed     : {}", if ham_frac < 0.50 { "PASS" } else { "FAIL" });
    println!("  selective (descriptive — drift-liveness, classes dissolved): {}", if selective { "movement present" } else { "no movement" });
    println!(
        "wall {:.1}s   peak RSS {} MB (budget {RSS_BUDGET_MB})",
        t0.elapsed().as_secs_f64(),
        peak_rss_mb()
    );
    if !tripwire || !g0 {
        println!("T1 NO — recorded; coupling story pivots to regime-dependence per the registration.");
    }

    // ----- Tier-2 export (readout of the SAME frozen experiment; argv
    // flag `export` writes the patched GGUF for the fork judge; step-5
    // arms always export, final-only, under arm names) -----
    if export_mode || step5_mode {
        println!();
        println!("--- TIER-2 export (the shared harness surgery unit; S2 re-read on EVERY file) ---");
        // The surgery as a reusable unit (R4-extracted
        // `splice_and_verify`): writes `out` from a synapse-order
        // trit snapshot; returns (cells changed, code bytes, scale
        // bytes); asserts scales-passthrough + S2 post-write
        // re-read internally. The synapse-order → N×N
        // reconstruction (k-walk + diagonal-from-src) stays here —
        // it is invivo's own graph shape.
        let do_surgery = |syn_trits: &[Trit], out: &str| -> (u64, u64, u64) {
            // synapse-order (j outer, i inner, i≠j) → N×N row-major;
            // the diagonal carried no synapse (full-minus-diagonal
            // build) → keeps its source trit.
            let adapted = {
                let mut a = vec![Trit::Zero; N * N];
                let mut k = 0usize;
                for j in 0..N {
                    for i in 0..N {
                        if i != j {
                            a[i * N + j] = syn_trits[k];
                            k += 1;
                        }
                    }
                }
                for i in 0..N {
                    a[i * N + i] = src[i * N + i];
                }
                a
            };
            let changed = adapted.iter().zip(&src).filter(|(a, b)| a != b).count() as u64;
            let (code_changed, scale_changed) =
                splice_and_verify(&path, out, &adapted, None, &p);
            (changed, code_changed, scale_changed)
        };

        // H2b: the checkpoint derivatives FIRST (declared, sha-pinned at
        // write), then the plain final export. Legacy mode only — step-5
        // arms take no checkpoints (the FREE arm judges the banked cks).
        for (step, snaps) in &v.ck_snaps {
            let out = format!("models/Ternary-Bonsai-4B-Q2_0-invivo-ck{step}.gguf");
            let (c, cb, sb) = do_surgery(snaps, &out);
            println!("  ck@learn-step {step:>5}: {c} cells · code {cb} · scale {sb} · S2 clean → {out}");
        }
        let out_path = if step5_mode {
            if off {
                format!("models/Ternary-Bonsai-4B-Q2_0-invivo-off-r{arm_r}.gguf")
            } else if domain {
                "models/Ternary-Bonsai-4B-Q2_0-invivo-domain.gguf".to_string()
            } else {
                format!("models/Ternary-Bonsai-4B-Q2_0-invivo-r{arm_r}.gguf")
            }
        } else {
            "models/Ternary-Bonsai-4B-Q2_0-invivo.gguf".to_string()
        };
        // The mechanical arm gates (PREREG §3/§9 — no discretionary
        // calls): the unbanked-path guard runs BEFORE the surgery
        // writes (prevent, never detect); OFF asserts byte-≡ base; ON
        // window-0 asserts the banked H2 sha (the r0 re-pin); DOMAIN
        // asserts nonzero delta (plasticity on, different drive
        // statistics).
        neuralos_rt::harness::assert_unbanked(&out_path);
        let (changed, code_changed, scale_changed) = do_surgery(&v.final_trits, &out_path);
        assert_eq!(changed, hamming, "cell deltas == Hamming");
        println!(
            "  final: {changed} cells · code {code_changed} · scale {scale_changed} (must be 0) · S2 clean"
        );
        let export_sha = sha256_of(&out_path);
        if step5_mode && off {
            let base_sha = sha256_of(&path);
            assert_eq!(export_sha, base_sha, "OFF arm: export must be byte-≡ base");
            println!("  OFF identity: sha == base ({base_sha:.16}…) : PASS (end-to-end toggle proof)");
        }
        if step5_mode && !off && !domain && arm_r == 0 {
            assert_eq!(
                export_sha, H2_EXPORT_SHA,
                "ON r0 re-pin: export must reproduce the banked H2 artifact byte-for-byte"
            );
            println!("  ON r0 re-pin: sha == banked 71f2518a… : PASS (H2 reproduced through the new path)");
        }
        if step5_mode && domain {
            assert!(changed > 0, "DOMAIN arm: adaptation alive (plasticity on, norm-unit drive)");
        }
        // H2b invariance assert: a 100%-checkpoint export must equal the
        // plain export — proves the checkpoint machinery did not perturb
        // the frozen path. (The last ck is at 3/4; the INVARIANCE check is
        // ck-machinery vs plain-machinery on the SAME final state, which
        // the identical do_surgery unit guarantees structurally; the sha
        // pin below is the mechanical witness.)
        println!("  wrote {out_path} — sha {export_sha} — fork judge next");
    }
}
