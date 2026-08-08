//! Ternary-bridge Stage 1 gate — the minimal falsifier.
//!
//! Question: does an SNN whose synapse weights are *constrained* to
//! `{-γ, 0, +γ}` still spike and learn comparably to the i16 baseline?
//!
//! Runs two phases on identical balanced 128-neuron networks:
//!   * **Phase A — spiking fidelity** (STDP off, fixed weights): does ternary
//!     propagation sustain firing like the i16 baseline?
//!   * **Phase B — learning fidelity** (STDP on): under per-step re-projection
//!     onto the ternary grid, do weights still move (bucket flips), vs the
//!     free i16 drift in the baseline?
//!
//! The numbers this prints ARE the gate evidence. The YES/NO call lives in the
//! commit message. A clean NO (ternary freezes learning) is a valid outcome
//! that stops the bridge per `docs/VISION.md`.

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
fn run_learning_baseline(net: &mut SpikingNeuralNetwork, steps: usize, inputs: &[i16]) -> (u64, u64, f64) {
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

/// Run `steps` with STDP on + per-step re-projection onto {-γ,0,+γ}.
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
        // Count ternary-state transitions vs the previous step's buckets.
        for (i, s) in net.synapses().iter().enumerate() {
            let cur = Trit::from_weight(s.weight, gamma);
            if cur != prev_buckets[i] {
                bucket_flips += 1;
                prev_buckets[i] = cur;
            }
        }
    }
    let plasticity = net.stats().plasticity_events;
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
    (plasticity, bucket_flips, (minus, zero, plus))
}

fn main() {
    let inputs = inputs();
    println!("=== Ternary-bridge Stage 1 gate ===");
    println!("network: balanced E/I, {NEURONS} neurons, dt={DT_US}μs, drive=±{DRIVE_UA}μA, {STEPS} steps/phase");
    println!();

    // ---- Phase A: spiking fidelity (STDP off) ----
    let (base_rate, base_spikes, n_syn) = run_spiking(&mut build(), STEPS, &inputs);

    let mut tern = build();
    let gamma = tern.ternarize_weights();
    let (tern_rate, tern_spikes, _) = run_spiking(&mut tern, STEPS, &inputs);

    println!("--- Phase A: spiking fidelity (STDP off, fixed weights) ---");
    println!("γ (mean|w|)               : {gamma}");
    println!("synapse count             : {n_syn}");
    println!("baseline i16  firing rate : {:.2} Hz/neuron  ({base_spikes} spikes)", base_rate);
    println!("ternary       firing rate : {:.2} Hz/neuron  ({tern_spikes} spikes)", tern_rate);
    if base_rate > 0.0 {
        println!("ternary / baseline        : {:.2}×", tern_rate / base_rate);
    }
    println!();

    // ---- Phase B: learning fidelity (STDP on) ----
    let (base_plast, base_changed, base_mean_delta) =
        run_learning_baseline(&mut build(), STEPS, &inputs);

    let mut tern_l = build();
    let gamma_l = tern_l.ternarize_weights();
    let (tern_plast, tern_flips, (minus, zero, plus)) =
        run_learning_ternary(&mut tern_l, gamma_l, STEPS, &inputs);

    println!("--- Phase B: learning fidelity (STDP on) ---");
    println!("baseline i16  plasticity events       : {base_plast}");
    println!("baseline i16  weights changed         : {base_changed} / {n_syn}");
    println!("baseline i16  mean |Δweight| (changed): {:.2}", base_mean_delta);
    println!("ternary       plasticity events       : {tern_plast}");
    println!("ternary       bucket flips            : {tern_flips}");
    println!(
        "ternary       final distribution      : −γ={minus}, 0={zero}, +γ={plus}",
    );
    println!();

    // ---- Verdict context (the call itself is made in the commit message) ----
    println!("--- Verdict context ---");
    let rate_ratio = if base_rate > 0.0 { tern_rate / base_rate } else { 0.0 };
    println!(
        "spiking  : ternary fires {:.2}× baseline ({:.2} vs {:.2} Hz/neuron)",
        rate_ratio, tern_rate, base_rate
    );
    let learning_ratio = if base_plast > 0 {
        tern_flips as f64 / base_plast as f64
    } else {
        0.0
    };
    println!(
        "learning : {} bucket flips / {} plasticity events ({:.4} flip/event)",
        tern_flips, tern_plast, learning_ratio
    );
    println!();
    println!("Gate rule: YES requires spiking non-collapsed AND meaningful bucket");
    println!("movement. NO (frozen learning or collapsed spiking) stops the bridge.");
}
