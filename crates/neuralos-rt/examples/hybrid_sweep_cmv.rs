//! Session E stage 1c — THE FINER-RULER SWEEP: same amplitude grid, same
//! nets, same instrument as stage 1 — on the CENTI-MILLIVOLT grid.
//!
//! Stage 1's honest NO (recorded): on the 1 mV voltage grid the three nets'
//! spike trains were identical at EVERY amplitude — delta_v truncation makes
//! the ±12 μA recurrent pulses arithmetically invisible (a steady current
//! below ~200 μA at rest moves the membrane exactly zero, forever; even
//! 160 μA — above the ~150 μA threshold current — is blind from rest). The
//! channel was never closed by amplitude; it was closed by the RULER.
//!
//! This sweep re-runs the frozen stage-1 protocol verbatim with one change:
//! every neuron is constructed on VoltageResolution::CentiMillivolt
//! (0.01 mV quanta; dead zone ≈ 2 μA; the ±12 μA pulses become 6-quanta
//! membrane motion). The mV-grid lineage numbers (35,157 etc.) are history;
//! centi mode has its OWN pinned state — totals are expected to shift
//! slightly (the drive currents themselves now integrate without mV
//! truncation, shifting ISIs by a step here and there), and that is
//! expected behavior, recorded, not divergence-from-lineage.
//!
//! # Design (locked pre-run; single variable vs stage 1 = the voltage grid)
//!
//! - Grid: I_ACTIVE ∈ {600, 450, 300, 240, 200, 170, 150, 125, 100} μA —
//!   the stage-1 grid verbatim, for the paired-figure comparison.
//! - I_INH stays 600 (unscaled), as in stage 1.
//! - Everything else D-2 verbatim: N=512 slice of blk.0.attn_q, γ=125,
//!   full-minus-diagonal, 1.5c schedule (4 groups, 60/40), 2000 steps,
//!   STDP OFF (fixed weights — a firing experiment, not learning).
//!
//! # Attribution (why any divergence here is weight-borne)
//!
//! The three nets at one amplitude share the drive schedule, the noise
//! realization (substrate noise is seeded id⊕time — weight-independent),
//! and the census (imported vs its shuffle; zero is the structure-free
//! floor). ONLY the weights differ. On the mV grid that difference could
//! not reach a spike decision; on the centi grid it can. Any (step,
//! neuron) event difference between trains is weight-borne by construction.
//!
//! # Metrics (same instrument as stage 1)
//!
//! - Per-population rates (E Hz/neuron, I Hz/neuron) + totals per net.
//! - **Spike-TRAIN Hamming** per pair + per-neuron rate-vector L1 per pair.
//!
//! # Pre-registered prediction + criterion (written before any run)
//!
//! - **A\* = the HIGHEST amplitude with any pairwise train Hamming > 0.**
//! - Prediction: **divergence at 600 μA already** — mechanism: a ±12 μA
//!   pulse is now 6 quanta vs E-climb steps of ~350 quanta; ~17.5
//!   concurrent pulses give per-step σ ≈ 25 quanta (~7% of a climb step),
//!   so spike TIMING shifts by a step accumulate over the ~10-step climb
//!   → train divergence grows with the number of active presynaptic E
//!   (drive-dominated rates unchanged to first order).
//! - Companion expectations: totals shift from 35,157 (centi pinned
//!   state); I-rate ≈ 125 Hz at high amplitude; below the stage-1 cliff
//!   the E population may still be silent (no bootstrap) — divergence
//!   requires SOME firing.
//! - Honest NO: zero divergence at every amplitude even on the centi
//!   grid ⇒ the ruler wasn't the blocker either; recorded, and the
//!   coupling fork drops to in-vivo drive / balanced background.
//!
//! Usage: `cargo run -p neuralos-rt --release --example hybrid_sweep_cmv
//! -- [model.gguf]` (default `models/Ternary-Bonsai-4B-Q2_0.gguf`).
//! Memory box: one file buffer + three 512 nets + three 128 KB trains —
//! well under the D-2 single-buffer budget; peak printed.

//! R4(iii) note: decode, drive, nets, and the ladder live in
//! `neuralos_rt::harness::run_amplitude_sweep` since 2026-08-20 — this
//! file carries the design of record and the entry point. Output is
//! byte-identical to the frozen original (re-pinned against
//! `evidence/r4-baselines/sweep_cmv_run1.log`).

use neuralos_rt::harness::{run_amplitude_sweep, ExperimentParams};
use neuralos_snn::VoltageResolution;

fn main() {
    let p = ExperimentParams::default();
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "models/Ternary-Bonsai-4B-Q2_0.gguf".into()
    });
    println!("=== Session E stage 1c: the finer-ruler sweep — centi-mV grid, same protocol ===");
    println!("file    : {path}");
    println!(
        "slice   : {}, first {}×{}, γ={}, full-minus-diagonal, STDP OFF (fixed weights)",
        p.tensor, p.n, p.n, p.gamma
    );
    println!("grid    : VoltageResolution::CentiMillivolt (0.01 mV quanta, dead zone ≈ 2 μA)");
    println!("drive   : 1.5c schedule verbatim; I_INH={} FIXED; I_ACTIVE swept over {:?}", p.i_inh, p.amplitudes);
    println!("nets    : imported / census-matched control (seed {:#x}) / zero — shared drive + noise; only weights differ", p.control_seed);
    println!("metric  : spike-TRAIN Hamming per pair (not totals) + per-neuron rate L1 + per-population Hz");
    println!("note    : centi pinned state — totals shift from the mV lineage (35,157) by design");
    println!();

    run_amplitude_sweep(&path, &p, VoltageResolution::CentiMillivolt);
}
