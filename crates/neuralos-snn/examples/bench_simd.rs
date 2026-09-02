//! bench_simd — measure AVX2 vs scalar batch LIF throughput on THIS CPU.
//!
//! Run: `cargo run --example bench_simd --features simd --release`
//! (release is mandatory — debug SIMD is meaningless.)
//!
//! Prints ns per integration step and the speedup ratio for N = {64, 256, 1024}.
//! The number that matters: is AVX2 faster than scalar here, and by how much?

use neuralos_snn::simd::{
    detect_simd_support, dt_over_tau, integrate_batch_scalar, integrate_lif_batch, SimdSupport,
};

const ITERS: usize = 20_000;

fn main() {
    // Force benchmark-mode defaults (like the demo does).
    println!();
    println!("  ╔═══════════════════════════════════════════════════════════╗");
    println!(" ║           neuralos-snn — SIMD vs Scalar Benchmark          ║");
    println!(" ║   i16 fixed-point LIF batch integration                    ║");
    println!(" ╚═══════════════════════════════════════════════════════════╝");
    println!();

    let support = detect_simd_support();
    println!("  runtime SIMD support: {support:?}");
    if !matches!(support, SimdSupport::Avx2) {
        println!("  ⚠ AVX2 not available — scalar will run for both; speedup ≈ 1.0x.");
    }
    println!();

    let dtot = dt_over_tau(1000, 20_000); // 1 ms step, 20 ms tau — typical cortical.

    println!("  ┌────────┬───────────────┬───────────────┬───────────┐");
    println!("  │   N    │ scalar ns/step│ simd   ns/step│  speedup  │");
    println!("  ├────────┼───────────────┼───────────────┼───────────┤");

    for &n in &[64_usize, 256, 1024] {
        let (s_ns, x_ns) = bench(n, dtot);
        let ratio = s_ns / x_ns;
        println!(
            "  │ {:>6} │ {:>13.1} │ {:>13.1} │ {:>7.2}x  │",
            n, s_ns, x_ns, ratio
        );
    }
    println!("  └────────┴───────────────┴───────────────┴───────────┘");
    println!();
    println!(
        "  {} iterations per measurement · release build · {:?}",
        ITERS, support
    );
    println!();
}

/// Time scalar and dispatched (AVX2-or-scalar) over `iters` steps of `n` neurons.
/// Returns (scalar ns/step, dispatched ns/step).
fn bench(n: usize, dtot: i32) -> (f64, f64) {
    // Identical starting state for both runs.
    let mut s = make_inputs(n);
    let mut x = make_inputs(n);
    let mut spikes_s = vec![false; n];
    let mut spikes_x = vec![false; n];

    // Warm up (fill caches, branch-predict).
    for _ in 0..100 {
        integrate_batch_scalar(
            &mut s.membrane,
            &s.resting,
            &s.current,
            &s.resistance,
            &s.threshold,
            dtot,
            &mut spikes_s,
        );
        integrate_lif_batch(
            &mut x.membrane,
            &x.resting,
            &x.current,
            &x.resistance,
            &x.threshold,
            dtot,
            &mut spikes_x,
        );
    }
    // Reset to deterministic starting state for the measured runs.
    s = make_inputs(n);
    x = make_inputs(n);

    // Scalar timing.
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        integrate_batch_scalar(
            &mut s.membrane,
            &s.resting,
            &s.current,
            &s.resistance,
            &s.threshold,
            dtot,
            &mut spikes_s,
        );
    }
    let scalar_dur = t0.elapsed();

    // Dispatched (AVX2 when available) timing.
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        integrate_lif_batch(
            &mut x.membrane,
            &x.resting,
            &x.current,
            &x.resistance,
            &x.threshold,
            dtot,
            &mut spikes_x,
        );
    }
    let simd_dur = t0.elapsed();

    let s_ns = scalar_dur.as_nanos() as f64 / ITERS as f64;
    let x_ns = simd_dur.as_nanos() as f64 / ITERS as f64;
    (s_ns, x_ns)
}

/// SoA batch inputs for one benchmark run.
struct Inputs {
    membrane: Vec<i16>,
    resting: Vec<i16>,
    current: Vec<i16>,
    resistance: Vec<i16>,
    threshold: Vec<i16>,
}

fn make_inputs(n: usize) -> Inputs {
    Inputs {
        membrane: vec![-70i16; n],
        resting: vec![-70i16; n],
        current: vec![200i16; n],
        resistance: vec![100i16; n],
        threshold: vec![-55i16; n],
    }
}
