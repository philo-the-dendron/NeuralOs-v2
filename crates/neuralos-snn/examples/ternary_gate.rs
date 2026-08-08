//! Ternary-bridge gate — Stage 1 (deterministic) + Stage 1.5b (stochastic).
//!
//! Question: does an SNN whose synapse weights are *constrained* to
//! `{-γ, 0, +γ}` still spike and learn comparably to the i16 baseline?
//!
//! Runs three regimes on identical balanced 128-neuron networks over the same
//! 300-step / drive=600 μA run, plus a fixed-weight spiking-fidelity phase:
//!   * **Phase A — spiking fidelity** (STDP off): does ternary propagation
//!     sustain firing like the i16 baseline?
//!   * **Phase B — learning fidelity** (STDP on), three regimes:
//!     - **(i)** i16 baseline — free drift (weights move freely).
//!     - **(ii)** ternary + deterministic per-step re-projection (Stage 1) —
//!       the ruled-out baseline; STDP deltas (±5) can't cross the γ/2 ≈ 62
//!       boundary → 0 flips (frozen).
//!     - **(iii)** ternary + stochastic bucket-flips (Stage 1.5b) — THIS is
//!       what we're measuring. Each STDP event does a Bernoulli(∝|δ|) draw to
//!       flip one bucket in the delta's direction.
//!
//! The numbers this prints ARE the gate evidence. Gate = YES iff (iii) shows
//! nontrivial bucket movement (vs (ii)'s 0 flips) AND spiking stays sane
//! (non-collapsed vs baseline). The YES/NO call is recorded in the commit
//! message and `docs/VISION.md`.

use neuralos_snn::{NetworkTopology, SpikingNeuralNetwork, Trit};

const NEURONS: u16 = 128;
const DT_US: u32 = 1000; // 1 ms sim per step
const DRIVE_UA: i16 = 600; // sustains firing in the balanced net (per main.rs default)
const STEPS: usize = 300;

fn build() -> SpikingNeuralNetwork {
    let mut net = SpikingNeuralNetwork::new(NEURONS, DT_US, NetworkTopology::default())
        .expect("balanced 128-neuron net must construct");
    net.build_topology().expect("topology must build");
    net
}

fn inputs() -> Vec<i16> {
    vec![DRIVE_UA; NEURONS as usize]
}

/// Run `steps` with STDP off; return (firing_rate_hz, total_spikes, synapse_count).
fn run_spiking(net: &mut SpikingNeuralNetwork, steps: usize, inputs: &[i16]) -> (f64, u64, u32) {
    net.set_plasticity_enabled(false);
    for _ in 0..steps {
        let _ = net.step(inputs).expect("step");
    }
    let s = net.stats();
    (s.firing_rate_hz, s.total_spikes, net.synapse_count())
}

/// Run `steps` with STDP on; return (plasticity_events, #weights_changed, mean|Δweight|).
fn run_learning_baseline(
    net: &mut SpikingNeuralNetwork,
    steps: usize,
    inputs: &[i16],
) -> (u64, u64, f64) {
    let initial: Vec<i16> = net.synapses().iter().map(|s| s.weight).collect();
    net.set_plasticity_enabled(true);
    for _ in 0..steps {
        let _ = net.step(inputs).expect("step");
    }
    let plasticity = net.stats().plasticity_events;
    let mut changed = 0u64;
    let mut sum_abs_delta = 0i64;
    for (s, &init) in net.synapses().iter().zip(initial.iter()) {
        if s.weight != init {
            changed += 1;
            sum_abs_delta += i64::from((s.weight - init).abs());
        }
    }
    let mean_delta = if changed > 0 {
        sum_abs_delta as f64 / changed as f64
    } else {
        0.0
    };
    (plasticity, changed, mean_delta)
}

/// Run `steps` with STDP on + per-step re-projection onto {-γ,0,+γ} (Stage 1).
/// Returns (plasticity_events, bucket_flips, final {minus,zero,plus} counts).
fn run_learning_ternary(
    net: &mut SpikingNeuralNetwork,
    gamma: i16,
    steps: usize,
    inputs: &[i16],
) -> (u64, u64, (u64, u64, u64)) {
    let mut prev_buckets: Vec<Trit> = net
        .synapses()
        .iter()
        .map(|s| Trit::from_weight(s.weight, gamma))
        .collect();
    net.set_plasticity_enabled(true);
    let mut bucket_flips = 0u64;
    for _ in 0..steps {
        let _ = net.step(inputs).expect("step");
        net.reproject_ternary(gamma);
        for (i, s) in net.synapses().iter().enumerate() {
            let cur = Trit::from_weight(s.weight, gamma);
            if cur != prev_buckets[i] {
                bucket_flips += 1;
                prev_buckets[i] = cur;
            }
        }
    }
    let plasticity = net.stats().plasticity_events;
    let dist = bucket_distribution(net, gamma);
    (plasticity, bucket_flips, dist)
}

/// Run `steps` with STDP on + stochastic bucket-flips (Stage 1.5b).
/// Returns (plasticity_events, bucket_flips, distribution, firing_rate_hz, total_spikes).
fn run_learning_stochastic(
    net: &mut SpikingNeuralNetwork,
    gamma: i16,
    steps: usize,
    inputs: &[i16],
) -> (u64, u64, (u64, u64, u64), f64, u64) {
    let mut prev_buckets: Vec<Trit> = net
        .synapses()
        .iter()
        .map(|s| Trit::from_weight(s.weight, gamma))
        .collect();
    net.set_plasticity_enabled(true);
    let mut bucket_flips = 0u64;
    for _ in 0..steps {
        let _ = net.step(inputs).expect("step");
        net.stochastic_ternary_step(gamma);
        for (i, s) in net.synapses().iter().enumerate() {
            let cur = Trit::from_weight(s.weight, gamma);
            if cur != prev_buckets[i] {
                bucket_flips += 1;
                prev_buckets[i] = cur;
            }
        }
    }
    let plasticity = net.stats().plasticity_events;
    let rate = net.stats().firing_rate_hz;
    let spikes = net.stats().total_spikes;
    let dist = bucket_distribution(net, gamma);
    (plasticity, bucket_flips, dist, rate, spikes)
}

/// Count the final ternary bucket distribution.
fn bucket_distribution(net: &SpikingNeuralNetwork, gamma: i16) -> (u64, u64, u64) {
    let mut minus = 0u64;
    let mut zero = 0u64;
    let mut plus = 0u64;
    for s in net.synapses() {
        match Trit::from_weight(s.weight, gamma) {
            Trit::MinusOne => minus += 1,
            Trit::Zero => zero += 1,
            Trit::One => plus += 1,
        }
    }
    (minus, zero, plus)
}

fn main() {
    let inputs = inputs();
    println!("=== Ternary-bridge gate: Stage 1 (deterministic) + Stage 1.5b (stochastic) ===");
    println!(
        "network: balanced E/I, {NEURONS} neurons, dt={DT_US}μs, drive=±{DRIVE_UA}μA, {STEPS} steps/phase"
    );
    println!();

    // ---- Phase A: spiking fidelity (STDP off, fixed weights) ----
    let (base_rate, base_spikes, n_syn) = run_spiking(&mut build(), STEPS, &inputs);

    let mut tern = build();
    let gamma = tern.ternarize_weights();
    let (tern_rate, tern_spikes, _) = run_spiking(&mut tern, STEPS, &inputs);

    println!("--- Phase A: spiking fidelity (STDP off, fixed weights) ---");
    println!("γ (mean|w|)               : {gamma}");
    println!("synapse count             : {n_syn}");
    println!(
        "baseline i16  firing rate : {:.2} Hz/neuron  ({base_spikes} spikes)",
        base_rate
    );
    println!(
        "ternary       firing rate : {:.2} Hz/neuron  ({tern_spikes} spikes)",
        tern_rate
    );
    if base_rate > 0.0 {
        println!("ternary / baseline        : {:.2}×", tern_rate / base_rate);
    }
    println!();

    // ---- Phase B: learning fidelity (STDP on) — three regimes ----

    // (i) i16 baseline — free drift
    let (base_plast, base_changed, base_mean_delta) =
        run_learning_baseline(&mut build(), STEPS, &inputs);

    // (ii) ternary + deterministic re-projection (Stage 1)
    let mut tern_det = build();
    let gamma_det = tern_det.ternarize_weights();
    let (det_plast, det_flips, (det_minus, det_zero, det_plus)) =
        run_learning_ternary(&mut tern_det, gamma_det, STEPS, &inputs);

    // (iii) ternary + stochastic flips (Stage 1.5b)
    let mut tern_sto = build();
    let gamma_sto = tern_sto.ternarize_weights();
    let (sto_plast, sto_flips, (sto_minus, sto_zero, sto_plus), sto_rate, sto_spikes) =
        run_learning_stochastic(&mut tern_sto, gamma_sto, STEPS, &inputs);

    println!("--- Phase B: learning fidelity (STDP on) ---");
    println!();
    println!("  (i) i16 baseline — free drift");
    println!("      plasticity events       : {base_plast}");
    println!("      weights changed         : {base_changed} / {n_syn}");
    println!(
        "      mean |Δweight| (changed): {:.2}",
        base_mean_delta
    );
    println!();
    println!("  (ii) ternary + deterministic re-projection (Stage 1)");
    println!("      plasticity events       : {det_plast}");
    println!("      bucket flips            : {det_flips}");
    println!(
        "      final distribution      : −γ={det_minus}, 0={det_zero}, +γ={det_plus}"
    );
    println!();
    println!("  (iii) ternary + stochastic flips (Stage 1.5b) ← measuring this");
    println!("      plasticity events       : {sto_plast}");
    println!("      bucket flips            : {sto_flips}");
    println!(
        "      final distribution      : −γ={sto_minus}, 0={sto_zero}, +γ={sto_plus}"
    );
    if sto_plast > 0 {
        println!(
            "      flip rate               : {:.4} flips/event",
            sto_flips as f64 / sto_plast as f64
        );
    }
    println!();

    // ---- Phase C: spiking under stochastic learning (non-collapse check) ----
    println!("--- Phase C: spiking under stochastic learning (non-collapse check) ---");
    println!(
        "stochastic ternary  firing rate : {:.2} Hz/neuron  ({sto_spikes} spikes)",
        sto_rate
    );
    println!(
        "baseline i16 (fixed) firing rate : {:.2} Hz/neuron  ({base_spikes} spikes)",
        base_rate
    );
    if base_rate > 0.0 {
        println!(
            "stochastic / baseline           : {:.2}×",
            sto_rate / base_rate
        );
    }
    println!();

    // ---- Verdict context ----
    println!("--- Verdict context ---");
    let spikes_ok = base_rate > 0.0 && sto_rate / base_rate > 0.1;
    let movement_ok = sto_flips > 0 && det_flips == 0;
    println!(
        "bucket movement : {} stochastic flips vs {} deterministic flips → {}",
        sto_flips,
        det_flips,
        if movement_ok { "NONTRIVIAL ✓" } else { "FROZEN ✗" }
    );
    println!(
        "spiking sanity  : {:.2}× baseline → {}",
        if base_rate > 0.0 { sto_rate / base_rate } else { 0.0 },
        if spikes_ok { "NON-COLLAPSED ✓" } else { "COLLAPSED ✗" }
    );
    let gate_pass = movement_ok && spikes_ok;
    println!();
    println!(
        "GATE: {} — {}",
        if gate_pass { "YES" } else { "NO" },
        if gate_pass {
            "stochastic ternary STDP learns (nonzero flips) and spiking stays sane"
        } else {
            "learning frozen or spiking collapsed — bridge stays paused"
        }
    );
}
