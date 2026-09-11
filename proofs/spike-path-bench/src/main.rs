//! Spike-path timing on the host, through the public API only.
//!
//! Two arms, same loop:
//! - `forced`: threshold at the membrane floor, refractory period 0,
//!   constant input — every step fires, so every step records a spike
//!   in the history. After the 64th spike the alpha.5 history shifted
//!   64 entries per step (`heapless::Vec::remove(0)`); the alpha.6 ring
//!   overwrites one slot.
//! - `control`: no input, threshold untouched — no step fires; the same
//!   loop minus the spike path.
//!
//! Output: one line per repetition (`arm rep steps spikes ns_per_step`),
//! then `min` and `median` per arm. Numbers only; the comparison across
//! the two trees is made in the evidence README, not here.

use neuralos_snn::LIFNeuron;
use std::hint::black_box;
use std::time::Instant;

const STEPS: u32 = 10_000_000;
const REPS: usize = 10;
const DT_US: u32 = 1_000;

fn run(arm: &str, input_ua: i16, force: bool) -> Vec<f64> {
    let mut ns_per_step = Vec::with_capacity(REPS);
    for rep in 1..=REPS {
        let mut n = LIFNeuron::new(1);
        n.noise_amplitude_ua = 0;
        if force {
            n.threshold = neuralos_snn::MEMBRANE_MV_MIN;
            n.tau_refractory_us = 0;
        }
        let mut t: u32 = 0;
        let mut spikes: u64 = 0;
        let start = Instant::now();
        for _ in 0..STEPS {
            t = t.wrapping_add(DT_US);
            spikes += u64::from(n.integrate_and_fire(black_box(input_ua), DT_US, t));
        }
        let elapsed = start.elapsed();
        black_box(&n);
        let ns = elapsed.as_nanos() as f64 / f64::from(STEPS);
        println!("arm={arm} rep={rep} steps={STEPS} spikes={spikes} ns_per_step={ns:.3}");
        ns_per_step.push(ns);
    }
    ns_per_step
}

fn summary(arm: &str, mut v: Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let median = (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0;
    println!(
        "arm={arm} min_ns_per_step={:.3} median_ns_per_step={median:.3}",
        v[0]
    );
}

fn main() {
    println!("spike-path-bench steps={STEPS} reps={REPS} dt_us={DT_US} profile=release");
    let forced = run("forced", 1_000, true);
    let control = run("control", 0, false);
    summary("forced", forced);
    summary("control", control);
}
