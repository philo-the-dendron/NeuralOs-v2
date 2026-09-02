//! NeuralOS v2 — SNN Showcase Demo
//!
//! Creates a balanced E/I spiking neural network, runs a simulation, and prints
//! a spike raster + learning statistics. This is the RVO Ottawa hallway artifact.
//!
//! Run: `cargo run --example demo`
//!
//! Everything here uses real computation — no hardcoded results, no fake numbers.

use neuralos_snn::{NetworkTopology, NeuronType, SpikingNeuralNetwork};

// --- Demo configuration ---
const NEURON_COUNT: u16 = 50; // Smaller network = denser activity per neuron
const TIMESTEP_US: u32 = 1000; // 1 ms per step
const SIMULATION_STEPS: usize = 200; // 200 ms simulated
const INPUT_CURRENT_UA: i16 = 800; // Strong drive — overcomes E/I balance
const RASTER_STRIDE: usize = 2; // Show every 2nd step for density

fn main() {
    print_header();
    run_demo();
    print_footer();
}

fn print_header() {
    println!();
    println!("  ╔═══════════════════════════════════════════════════════════╗");
    println!("  ║           NeuralOS v2 — SNN Showcase Demo                 ║");
    println!("  ║   no_std · i16 fixed-point · Rust · RISC-V-ready          ║");
    println!("  ╚═══════════════════════════════════════════════════════════╝");
    println!();
}

fn run_demo() {
    // --- Phase 1: Create network ---
    println!("┌─ Network setup ────────────────────────────────────────────┐");
    let mut net = SpikingNeuralNetwork::new(
        NEURON_COUNT,
        TIMESTEP_US,
        NetworkTopology::Balanced {
            excitatory_ratio: 0.8,
        },
    )
    .expect("network creation");

    let exc_count = net
        .neurons()
        .iter()
        .filter(|n| n.neuron_type == NeuronType::Excitatory)
        .count();
    let inh_count = NEURON_COUNT as usize - exc_count;
    println!(
        "│ {} neurons ({} E, {} I) — biological 80/20 ratio         │",
        NEURON_COUNT, exc_count, inh_count
    );
    println!(
        "│ Timestep: {}μs — {} ms simulated                          │",
        TIMESTEP_US, SIMULATION_STEPS
    );

    net.build_topology().expect("topology build");
    println!(
        "│ Balanced E/I topology — {} synapses wired                 │",
        net.synapse_count()
    );
    println!("└────────────────────────────────────────────────────────────┘");
    println!();

    // --- Phase 2: Snapshot initial state ---
    let initial_weights: Vec<i16> = net.synapses().iter().map(|s| s.weight).collect();

    // --- Phase 3: Run simulation ---
    println!(
        "┌─ Simulation: {} steps, {}μA input ────────────────────────┐",
        SIMULATION_STEPS, INPUT_CURRENT_UA
    );
    let inputs = vec![INPUT_CURRENT_UA; NEURON_COUNT as usize];

    // Collect per-step data for visualization.
    let mut spikes_per_step: Vec<u32> = Vec::with_capacity(SIMULATION_STEPS);
    let mut spike_raster: Vec<Vec<u16>> = Vec::with_capacity(SIMULATION_STEPS);

    for step in 0..SIMULATION_STEPS {
        let spikes = net.step(&inputs).expect("step");
        spikes_per_step.push(spikes.len() as u32);
        spike_raster.push(spikes.iter().map(|s| s.neuron_id).collect());

        if (step + 1) % 50 == 0 {
            let stats = net.stats();
            println!(
                "│ step {:>3} — spikes this step: {:>2} | total: {:>5} | rate: {:.1} Hz │",
                step + 1,
                spikes.len(),
                stats.total_spikes,
                stats.firing_rate_hz
            );
        }
    }
    println!("└────────────────────────────────────────────────────────────┘");
    println!();

    // --- Phase 4: Print spike raster ---
    print_spike_raster(&spike_raster, NEURON_COUNT);

    // --- Phase 5: Print population activity sparkline ---
    print_activity_sparkline(&spikes_per_step);

    // --- Phase 6: Final statistics ---
    print_final_stats(&net);

    // --- Phase 7: Weight evolution (STDP learning evidence) ---
    let final_weights: Vec<i16> = net.synapses().iter().map(|s| s.weight).collect();
    let changed = initial_weights
        .iter()
        .zip(final_weights.iter())
        .filter(|(a, b)| a != b)
        .count();

    println!("┌─ Weight evolution (STDP learning) ─────────────────────────┐");
    let (im, is) = weight_stats(&initial_weights);
    let (fm, fs) = weight_stats(&final_weights);
    println!(
        "│ Mean weight:   {:>7.1} → {:>7.1}                       │",
        im, fm
    );
    println!(
        "│ Std deviation: {:>7.1} → {:>7.1}                       │",
        is, fs
    );
    println!(
        "│ Synapses changed: {} / {} ({:.0}%)                     │",
        changed,
        final_weights.len(),
        (changed as f64 / final_weights.len().max(1) as f64) * 100.0
    );
    println!("└────────────────────────────────────────────────────────────┘");
    println!();
}

fn print_spike_raster(raster: &[Vec<u16>], neuron_count: u16) {
    let n = neuron_count as usize;
    println!(
        "┌─ Spike raster ({} neurons, sampled every {} steps) ─────────┐",
        n, RASTER_STRIDE
    );

    // Header: tens digit row and ones digit row.
    let tens: String = (0..n)
        .map(|i| {
            if i % 10 == 0 {
                char::from_digit((i / 10) as u32 % 10, 10).unwrap_or(' ')
            } else {
                ' '
            }
        })
        .collect();
    let ones: String = (0..n)
        .map(|i| char::from_digit((i % 10) as u32, 10).unwrap_or(' '))
        .collect();
    println!("│        │{}│", tens);
    println!("│  step  │{}│", ones);

    for (step_idx, fired_ids) in raster.iter().enumerate() {
        if step_idx % RASTER_STRIDE != 0 {
            continue;
        }
        let mut row = vec!['·'; n];
        for &id in fired_ids {
            if (id as usize) < n {
                row[id as usize] = '█';
            }
        }
        let line: String = row.iter().collect();
        println!("│ {:>5}  │{}│", step_idx, line);
    }
    println!("└────────────────────────────────────────────────────────────┘");
    println!();
}

fn print_activity_sparkline(spikes_per_step: &[u32]) {
    let max_spikes = *spikes_per_step.iter().max().unwrap_or(&1) as f64;
    if max_spikes == 0.0 {
        println!("┌─ Population activity ──────────────────────────────────────┐");
        println!("│ (no activity detected)                                     │");
        println!("└────────────────────────────────────────────────────────────┘");
        println!();
        return;
    }

    println!("┌─ Population activity (spikes per step) ────────────────────┐");
    // ASCII bar chart: one character per step, height = spike count scaled to 10.
    let height = 8;
    for row in (1..=height).rev() {
        let threshold = max_spikes * row as f64 / height as f64;
        let line: String = spikes_per_step
            .iter()
            .map(|&s| {
                if s as f64 >= threshold {
                    '▆'
                } else if s as f64 >= threshold * 0.5 {
                    '▃'
                } else {
                    ' '
                }
            })
            .collect();
        let label = (max_spikes * row as f64 / height as f64).ceil() as u32;
        println!("│ {:>2} │{}│", label, line);
    }
    println!("│    └{}", "─".repeat(spikes_per_step.len()));
    println!("│      steps 0→{}", spikes_per_step.len());
    println!("└────────────────────────────────────────────────────────────┘");
    println!();
}

fn print_final_stats(net: &SpikingNeuralNetwork) {
    let s = net.stats();
    println!("┌─ Final statistics ──────────────────────────────────────────┐");
    println!(
        "│ Total spikes:           {:>8}                         │",
        s.total_spikes
    );
    println!(
        "│ Plasticity events:      {:>8}                         │",
        s.plasticity_events
    );
    println!(
        "│ Synapses:               {:>8}                         │",
        s.total_synapses
    );
    println!(
        "│ Avg firing rate:        {:>8.2} Hz                    │",
        s.firing_rate_hz
    );
    println!(
        "│ Avg membrane potential: {:>8.2} mV                    │",
        s.avg_membrane_potential_mv
    );
    println!(
        "│ Simulation time:        {:>8} μs ({:.1} ms)             │",
        net.current_time_us(),
        net.current_time_us() as f64 / 1000.0
    );
    println!("└────────────────────────────────────────────────────────────┘");
    println!();
}

fn weight_stats(weights: &[i16]) -> (f64, f64) {
    if weights.is_empty() {
        return (0.0, 0.0);
    }
    let n = weights.len() as f64;
    let mean = weights.iter().map(|&w| w as f64).sum::<f64>() / n;
    let variance = weights
        .iter()
        .map(|&w| (w as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, variance.sqrt())
}

fn print_footer() {
    println!("┌─ About this library ───────────────────────────────────────┐");
    println!("│ Every neuron uses i16 fixed-point math (no FPU required).  │");
    println!("│ Every synapse uses fixed-point conductance (no allocator).  │");
    println!("│ The hot-path primitives compile to no_std (verified).       │");
    println!("│                                                             │");
    println!("│ This same code runs on:                                     │");
    println!("│   x86_64 desktop (this demo)                                │");
    println!("│   ARM64 server                                              │");
    println!("│   RISC-V QEMU (riscv64gc)                                   │");
    println!("│   ESP32-C3 bare metal (via esp-rs)                          │");
    println!("│                                                             │");
    println!("│ No OpenAI. No Anthropic. No cloud. No telemetry.            │");
    println!("│ Own your compute.                                           │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Source: https://gitea.com/Caramoussin/NeuralOs-v2");
    println!("  License: AGPL-3.0-or-later");
    println!();
}
