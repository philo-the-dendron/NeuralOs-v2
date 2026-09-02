//! Ternary-bridge gate — Stage 3: the shared kernel + hybrid net.
//!
//! Question: does the union compose — can an SNN layer and a dense
//! LLM-style layer share ONE ternary matmul kernel and compute something
//! coherent together?
//!
//! Pipeline (the composition IS the claim):
//!
//!   structured drive → ternary SNN (fixed γ, STDP off) → per-neuron spike
//!   counts → Q15 absmax activations (integer BitNet-style) → dense 4×128
//!   ternary classifier — whose weights ARRIVE as BitNet `i2_s` wire bytes
//!   and reach the kernel only through `repack_i2s_to_kernel`.
//!
//! The dense weights are **constructed, not trained** (+1 on the target
//! group's excitatory neurons, −1 on other E neurons, 0 on inhibitory).
//! Stage 3's claim is composition — wire format → kernel → coherent
//! output — not dense-layer learning (that was never on trial; SNN-side
//! learning closed at Stage 1.5d). If the gate passes, Stage 4's
//! pure-Rust ternary-LLM runtime is earned.
//!
//! Input paradigm is the proven 1.5c setup (gapped 4-group round-robin,
//! tonic inhibitory drive); chance = 25%, gate = 4/4 groups classified
//! with printed margins. The numbers this prints ARE the gate evidence.

use neuralos_snn::{
    absmax_normalize_q15, encode_i2_s, repack_i2s_to_kernel, ternary_matvec, NetworkTopology,
    SpikingNeuralNetwork, Trit,
};

const NEURONS: u16 = 128;
const GROUPS: u16 = 4;
const DT_US: u32 = 1000;
const EXCITATORY_RATIO: f64 = 0.8;
const ACTIVE_ON: usize = 60;
const OFF_GAP: usize = 40;
const I_ACTIVE: i16 = 600;
const I_IDLE: i16 = 0;
const I_INH: i16 = 600; // tonic drive on inhibitory neurons (1.5c constant)

fn exc_count() -> u16 {
    (f64::from(NEURONS) * EXCITATORY_RATIO) as u16
}

fn group_of(neuron_id: u16, exc: u16) -> u16 {
    let g = (u32::from(neuron_id) * u32::from(GROUPS) / u32::from(exc)) as u16;
    g.min(GROUPS.saturating_sub(1))
}

/// Input vector for `active_on` steps of driving group `g`.
fn drive_inputs(g: u16) -> Vec<i16> {
    let exc = exc_count();
    let mut inp = vec![I_INH; NEURONS as usize];
    for n in 0..exc {
        inp[n as usize] = if group_of(n, exc) == g {
            I_ACTIVE
        } else {
            I_IDLE
        };
    }
    inp
}

fn idle_inputs() -> Vec<i16> {
    vec![I_INH; NEURONS as usize]
}

/// The dense layer as raw trits: row g = +1 on group-g E neurons, −1 on
/// other E neurons, 0 on inhibitory. All three codes exercised by design.
fn dense_trits() -> Vec<Vec<Trit>> {
    let exc = exc_count();
    (0..GROUPS)
        .map(|g| {
            (0..NEURONS)
                .map(|n| {
                    if n >= exc {
                        Trit::Zero
                    } else if group_of(n, exc) == g {
                        Trit::One
                    } else {
                        Trit::MinusOne
                    }
                })
                .collect()
        })
        .collect()
}

fn main() {
    println!("=== Stage 3 gate — the shared kernel + hybrid net ===");

    // --- Dense weights: trits → i2_s WIRE bytes → kernel compute packing.
    // The classifier reaches the kernel ONLY through the wire path.
    let rows = dense_trits();
    let scale_bits: u32 = 0x3F80_0000; // f32 1.0 — the layer's γ, carried as bits
    let mut wire = [0_u8; 64]; // 128 trits → 32 packed + 32 tail, per row
    let mut kernel_w = [0_u8; 4 * 32];
    for (j, row) in rows.iter().enumerate() {
        encode_i2_s(row, scale_bits, &mut wire).expect("encode i2_s");
        repack_i2s_to_kernel(&wire, row.len(), &mut kernel_w[j * 32..(j + 1) * 32])
            .expect("repack wire→kernel");
    }
    println!("  dense layer: 4x128 ternary, entered as i2_s wire bytes, repacked to kernel layout");

    // --- SNN layer: balanced 128-neuron net, ternary fixed weights, STDP off.
    let mut net = SpikingNeuralNetwork::new(NEURONS, DT_US, NetworkTopology::default())
        .expect("net must construct");
    net.build_topology().expect("topology must build");
    let gamma = net.ternarize_weights();
    net.set_plasticity_enabled(false);
    println!(
        "  SNN layer   : 128 neurons, ternary weights at gamma = {gamma}, STDP off (transducer)"
    );

    // Settle one full cycle of silence so the init state is steady.
    let idle = idle_inputs();
    for _ in 0..((ACTIVE_ON + OFF_GAP) * GROUPS as usize) {
        net.step(&idle).expect("settle step");
    }

    // --- Trials: drive each group, count spikes, classify through the kernel.
    let mut correct = 0_usize;
    for g in 0..GROUPS {
        let gap = idle_inputs(); // clone-free: drive_inputs builds fresh each call
        for _ in 0..OFF_GAP {
            net.step(&gap).expect("gap step");
        }
        let inp = drive_inputs(g);
        let mut counts = vec![0_i16; NEURONS as usize];
        for _ in 0..ACTIVE_ON {
            for spike in net.step(&inp).expect("active step") {
                counts[spike.neuron_id as usize] += 1;
            }
        }

        // Substrate-side evidence: driven group vs the rest.
        let exc = exc_count();
        let (mut on_sum, mut on_n, mut off_sum) = (0_u32, 0_u32, 0_u32);
        for n in 0..exc {
            if group_of(n, exc) == g {
                on_sum += u32::from(counts[n as usize].unsigned_abs());
                on_n += 1;
            } else {
                off_sum += u32::from(counts[n as usize].unsigned_abs());
            }
        }
        let on_mean = f64::from(on_sum) / f64::from(on_n.max(1));
        let off_mean = f64::from(off_sum) / f64::from((u32::from(exc) - on_n).max(1));

        // Kernel-side: counts → Q15 absmax → shared matvec.
        let mut acts = vec![0_i16; NEURONS as usize];
        let absmax = absmax_normalize_q15(&counts, &mut acts);
        let mut out = [0_i32; 4];
        ternary_matvec(&kernel_w, &acts, 4, &mut out).expect("kernel matvec");
        let arg = out
            .iter()
            .enumerate()
            .max_by_key(|(j, v)| (*v, 3 - *j as i32))
            .map(|(j, _)| j)
            .unwrap_or(0);
        let runner_up = out
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != arg)
            .map(|(_, v)| *v)
            .max()
            .unwrap_or(0);
        let margin = out[arg] - runner_up;
        let ok = arg == g as usize;
        correct += ok as usize;
        println!(
            "  trial g={g}: driven-group mean {on_mean:.2} spikes/neuron vs others {off_mean:.2} | absmax {absmax} | logits {out:?} -> argmax {arg} {} (margin {margin})",
            if ok { "OK" } else { "WRONG" }
        );
    }

    println!();
    if correct == GROUPS as usize {
        println!("STAGE 3 GATE: YES — the union composes: SNN spikes -> shared ternary kernel -> coherent 4/4 classification (chance 25%)");
    } else {
        println!("STAGE 3 GATE: NO — {correct}/4 classified");
        std::process::exit(1);
    }
}
