//! Session E stage 1 — THE AMPLITUDE SWEEP: at what drive amplitude do the
//! pretrained weights begin to shape firing?
//!
//! The stage-0 deep-dive finding this experiment falsifies or confirms: at
//! I_ACTIVE=600 μA the excitatory threshold margin is ~450 μA while a
//! recurrent ±12 μA (weight/10) pulse is integer-mV-quantization-absorbed
//! on the E climb — an 8.6–11.2σ event needed to gate a spike, which is why
//! imported / census-matched control / zero-weight nets all fired exactly
//! 35,157 spikes (D-2 + session E, recorded). The weight→firing channel is
//! closed at 600 μA. This sweep walks the amplitude DOWN to find where it
//! opens — the curve is itself a publishable figure.
//!
//! # Design (locked pre-run; single variable)
//!
//! - Grid: I_ACTIVE ∈ {600, 450, 300, 240, 200, 170, 150, 125, 100} μA.
//!   The E-neuron threshold-equivalent current is ~150 μA (V_ss = −70 +
//!   I/10 mV; threshold −55 mV) — at and below it, recurrent input is the
//!   only road to a spike, which is the channel we are trying to open.
//! - **I_INH stays 600** (unscaled): single variable = I_ACTIVE. The
//!   inhibitory population keeps firing ~125 Hz THROUGH the imported
//!   weights — a weight-informed background the E population sits in.
//! - Everything else D-2 verbatim: N=512 slice of blk.0.attn_q, γ=125,
//!   full-minus-diagonal, 1.5c schedule (4 groups, 60/40), 2000 steps,
//!   STDP OFF (fixed weights — this is a firing experiment, not learning).
//!
//! # Attribution (why any divergence here is weight-borne)
//!
//! The three nets at one amplitude share the drive schedule, the noise
//! realization (substrate noise is seeded id⊕time — weight-independent),
//! and the census (imported vs its shuffle; zero is the structure-free
//! floor). ONLY the weights differ. Identical weights ⇒ identical trains;
//! any (step, neuron) event difference is weight-borne by construction.
//!
//! # Metrics (recorded, not just totals — the D-2 lesson)
//!
//! - Per-population rates (E Hz/neuron, I Hz/neuron) + totals per net.
//! - **Spike-TRAIN Hamming** per pair (imported↔control, imported↔zero,
//!   control↔zero): count of (step, neuron) events present in exactly one
//!   train. D-2's 35,157×3 was a total-count check; trains are the record.
//! - Per-neuron rate-vector L1 per pair (a neuron-resolved view).
//!
//! # Pre-registered prediction + criterion (written before any run)
//!
//! - At 600 μA the trains are IDENTICAL (the recorded 35,157 ×3).
//! - **A\* = the HIGHEST amplitude with any pairwise train Hamming > 0.**
//! - Prediction: onset ≤ 300 μA — mechanism: recurrent σ ≈ 40–52 μA vs
//!   margin I_ACTIVE − 150 μA; at 170 μA the margin (~20 μA) is inside the
//!   fluctuation, at 240 μA (~90 μA) marginal, at 300+ absorbed. Divergence
//!   should GROW as amplitude drops.
//! - Expected companion regimes, recorded as findings not failures: near/
//!   below 150 μA the E population may go silent from drive alone (absent
//!   recurrent bootstrap); I-rate may sag if the E background dies.
//! - Honest NO: zero divergence at EVERY amplitude with sane rates ⇒ the
//!   weight→firing channel does not open by amplitude alone; recorded, and
//!   the coupling redesign conversation reopens with the curve attached.
//!
//! Usage: `cargo run -p neuralos-rt --release --example hybrid_sweep --
//! [model.gguf]` (default `models/Ternary-Bonsai-4B-Q2_0.gguf`).
//! Memory box: one file buffer + three 512 nets + three 128 KB trains —
//! well under the D-2 single-buffer budget; peak printed.

//! R4(iii) note: decode, drive, nets, and the ladder live in
//! `neuralos_rt::harness::run_amplitude_sweep` since 2026-08-20 — this
//! file carries the design of record and the entry point. Output is
//! byte-identical to the frozen original (re-pinned against
//! `evidence/r4-baselines/sweep_mv_run1.log`).

use neuralos_rt::harness::{run_amplitude_sweep, ExperimentParams};
use neuralos_snn::VoltageResolution;

fn main() {
    let p = ExperimentParams::default();
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "models/Ternary-Bonsai-4B-Q2_0.gguf".into()
    });
    println!("=== Session E stage 1: the amplitude sweep — where do weights begin to shape firing? ===");
    println!("file    : {path}");
    println!(
        "slice   : {}, first {}×{}, γ={}, full-minus-diagonal, STDP OFF (fixed weights)",
        p.tensor, p.n, p.n, p.gamma
    );
    println!("drive   : 1.5c schedule verbatim; I_INH={} FIXED; I_ACTIVE swept over {:?}", p.i_inh, p.amplitudes);
    println!("nets    : imported / census-matched control (seed {:#x}) / zero — shared drive + noise; only weights differ", p.control_seed);
    println!("metric  : spike-TRAIN Hamming per pair (not totals) + per-neuron rate L1 + per-population Hz");
    println!();

    run_amplitude_sweep(&path, &p, VoltageResolution::Millivolt);
}
