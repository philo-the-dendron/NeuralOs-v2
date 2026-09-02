//! Session D slice 2 — THE HYBRID EXPERIMENT: pretrained ternary LLM weights,
//! imported into the spiking substrate, adapting under local STDP.
//!
//! This is the experiment only NeuralOS can run. The two literatures it joins
//! — SNN-STDP and 1.58-bit LLM quantization — share no code and cite each
//! other nowhere; NeuralOS owns both halves (the SNN substrate with working
//! ternary STDP, and the runtime that eats real Bonsai Q2_0 bytes). The host
//! tensor is `blk.0.attn_q.weight` from Ternary-Bonsai-4B-Q2_0.gguf, first
//! 512×512 slice ≈ 262k synapses. The token_embd rows are pinned as D-3's
//! drive and are NOT used here.
//!
//! # Gates
//!
//! - **G1 IMPORT INTEGRITY** — trit-preserving by construction: Q2_0 tensor
//!   slice → `decode_q2_0` → `Trit` → substrate i16 weights at a
//!   SUBSTRATE-chosen γ=125 (the proven 1.5x regime). The LLM's per-block
//!   fp16 scales are meaningless to SNN dynamics — recorded decision, NOT a
//!   silent choice: γ=125 is the constant the 1.5b/1.5c gates were run at.
//!   Round-trip: substrate weights → `Trit::from_weight(125)` → bit-equal to
//!   the source trits. Zero-fraction census of the real pretrained slice
//!   (first measurement on real data at scale).
//!
//! - **G2 SPIKING FIDELITY** — the imported-weight network under the 1.5c
//!   structured 4-group gapped round-robin drive (I_ACTIVE=600, I_INH=600,
//!   ACTIVE_ON=60, OFF_GAP=40, dt=1 ms — the proven constants verbatim):
//!   non-degenerate firing vs a reference, no self-quench over the sustained
//!   window, per-group containment reported.
//!
//! - **G3 SELECTIVE ADAPTATION** — STDP ON + stochastic ternary bucket-flips,
//!   the 1.5c falsifier applied to PRETRAINED structure: selectivity (do
//!   correlated pairs modify differently from uncorrelated?), NOT collapse
//!   (majority of pretrained buckets intact), NOT freeze (flips > 0 on real
//!   weights).
//!
//! # Recorded decisions (all deliberate, none silent)
//!
//! - **γ=125 substrate-side**, per above; `wire_gamma_to_substrate` exists
//!   (ISC-13) but importing the LLM's fp16 block scales would couple SNN
//!   dynamics to quantizer conventions the substrate never saw.
//! - **DENSITY: full-minus-diagonal** (512×511 = 261,632 synapses) — NOT the
//!   balanced-0.8 sparse shape (~5×n edges) the 1.5c constants were proven
//!   on. The census-matched control makes G2 sound regardless, but absolute
//!   rates are compared ONLY within this experiment (imported vs control vs
//!   floor) — never against 1.5c's absolute numbers, whose topology differs.
//! - **The G2 reference is a random census-matched control**: same trit
//!   multiset (+/0/− counts identical), Fisher-Yates-shuffled placement
//!   (seed printed), same drive, STDP off — plus a structure-free ZERO-weight
//!   third comparator (added from this session's own diagnostic after
//!   imported and control matched exactly). The controls separate "any dense
//!   ±γ matrix spikes under this drive" from pretrained structure.
//!   Both comparisons are printed — (a) pretrained vs control under
//!   identical drive, (b) each vs the absolute floor — with the
//!   interpretation matrix stated up front — both above floor means density
//!   suffices to spike and the structure claim lives in G3; only pretrained
//!   above floor means G2 passes with the stronger structural claim; neither
//!   above floor means degenerate under this drive. Neither outcome is spun.
//! - **Signed propagation (load-bearing read, verified from substrate
//!   source):** `step()` Phase 2 injects `weight/10` straight from the CSR
//!   slot — a negative stored weight IS inhibitory current; `SynapseType`
//!   and the bio `receive_spike` path are not in the step loop. Imported
//!   mixed-sign matrices therefore propagate as-is, no sign→SynapseType
//!   split. `SynapseType` DOES own the plasticity bounds (sign-inferred at
//!   `Synapse::new`): +γ→[0,2000], −γ→[−2000,0], 0→Excitatory [0,2000]. So
//!   no imported synapse can cross sign under STDP — flips are
//!   one-directional per class (+γ↔0, −γ↔0, 0→+γ only). That asymmetry is a
//!   FINDING the census measures, not a defect to repair.
//! - **Selectivity metric = intra |mean Δ|** (session F amendment; was
//!   Δ-SI). The gate asserts the raw, non-degenerate degree field —
//!   intra |mean Δ| ≥ 0.05, the 1.5c floor now meaningfully applied —
//!   alongside flips > 0, Hamming < 0.50, sign crossings = 0, sustained
//!   firing. The DIRECTION of the class difference is printed as the
//!   era's mechanism label (Hebbian-carried = intra potentiated more,
//!   live-wire LTP; LTD-carried = intra depressed more, the dead-wire
//!   era's co-fire mode) and never gated — a metric's sign encodes the
//!   mechanism it was written under. Δ-SI is printed as a SUPPORTING
//!   LABEL only: the 1.5c schedule's 40 ms group gaps put every inter
//!   pair outside the 20 ms STDP window, so inter Δ ≡ 0 by geometry and
//!   |Δ-SI| ≡ 1 whenever any movement exists — it cannot gate on degree
//!   (second-reviewer finding, adopted). The 1.5c level-SI stays
//!   printed, confounded by pretrained levels as before.
//! - **Schedule = 2000 steps, 1.5c verbatim**: 400-step init cycle (STDP off,
//!   defeats the last_spike=0 never-fired artifact) + 1600 learning steps.
//!
//! # Verdict mapping (stated, not implied)
//!
//! `HYBRID GATE: ADAPTS | COLLAPSES | FROZEN | DEGENERATE`, first failure
//! wins: firing degenerates (G2 fails, or the STDP-on run collapses below
//! the ratio floor) → DEGENERATE; flips = 0 → FROZEN; Hamming ≥ 50% of
//! buckets changed → COLLAPSES (structure destroyed); selectivity fails →
//! COLLAPSES (uniform/no discrimination — 1.5c's SI-failure lineage); all
//! pass → ADAPTS. Both failure modes and passes are RESULTS, recorded with
//! evidence, never repaired into passes mid-run.
//!
//! Deterministic end to end (fixed seeds: control shuffle, substrate LFSR,
//! drive schedule) — two runs print identical evidence.
//!
//! Usage: `cargo run -p neuralos-rt --release --example hybrid_gate -- [path]`
//! (default `models/Ternary-Bonsai-4B-Q2_0.gguf`). Memory box: whole run
//! stays < 1.5 GB RSS (file buffer dropped after decode; networks ≈ tens of
//! MB) — peak printed.

//! R4(iii) note: the plumbing (decode, drive, runs, reporting) lives in
//! `neuralos_rt::harness` since 2026-08-20 — this file carries the
//! experiment's documentation and entry point. Output is byte-identical
//! to the frozen original (re-pinned against `evidence/r4-baselines/`).

use neuralos_rt::harness::{decode_slice, peak_rss_mb, run_gate_phase, ExperimentParams};

fn main() {
    let t0 = std::time::Instant::now();
    let p = ExperimentParams::default();
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Ternary-Bonsai-4B-Q2_0.gguf".into());
    println!("=== Session D slice 2: the hybrid experiment — Bonsai Q2_0 → Trit → SNN → STDP ===");
    println!("file    : {path}");
    println!(
        "host    : {}, first {}×{} slice (col j = pre → row i = post, matvec dataflow)",
        p.tensor, p.n, p.n
    );
    println!("substrate: γ={} (the proven 1.5x constant; LLM fp16 block scales unused — recorded), full-minus-diagonal density, 1.5c drive verbatim", p.gamma);
    println!();

    // Decode (scoped: file buffer drops before any network is built).
    let src = decode_slice(&path, &p);
    println!(
        "decode  : {} trits from real Q2_0 bytes (peak RSS so far is the file buffer)",
        src.len()
    );

    // G1 → G2 → G3 → verdict (verbatim phase; prints its own evidence).
    let outcome = run_gate_phase(&src, &p, false);
    println!(
        "wall {:.1}s   peak RSS {} MB (budget {})",
        t0.elapsed().as_secs_f64(),
        peak_rss_mb(),
        p.rss_budget_mb
    );
    if !outcome.pass {
        std::process::exit(1);
    }
}
