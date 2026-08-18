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

use neuralos_rt::{GgufFile, GGML_TYPE_Q2_0};
use neuralos_snn::{decode_q2_0, NetworkTopology, SpikingNeuralNetwork, Trit};

// ----- Geometry (D-2 verbatim) -----
const N: usize = 512; // slice side: 512 output rows × 512 input cols of attn_q
const TENSOR: &str = "blk.0.attn_q.weight";
const MODEL_COLS: usize = 2560; // 4B config: attn_q input width (emb)
const MODEL_ROWS: usize = 4096; // attn_q output width (32 heads × 128)
const ROW_BYTES: usize = (MODEL_COLS / 128) * 34; // 20 blocks × 34 B = 680
/// Substrate γ — the proven 1.5x regime constant (recorded decision).
const GAMMA: i16 = 125;

// ----- Drive (1.5c constants verbatim; I_ACTIVE is the swept variable) -----
const DT_US: u32 = 1000;
const EXCITATORY_RATIO: f64 = 0.8;
const GROUPS: u16 = 4;
const ACTIVE_ON: u32 = 60;
const OFF_GAP: u32 = 40;
const I_IDLE: i16 = 0;
/// Inhibitory drive — FIXED while I_ACTIVE sweeps (single-variable design).
const I_INH: i16 = 600;
const STEPS: usize = 2000;

/// The sweep grid — down to 100 μA, below the ~150 μA E-threshold.
const AMPLITUDES: [i16; 9] = [600, 450, 300, 240, 200, 170, 150, 125, 100];

/// Fisher-Yates seed for the census-matched control (printed in evidence).
const CONTROL_SEED: u64 = 0x5EED_C0DE_0000_0002;
/// Memory budget (the D-2 single-buffer box — this run stays under it).
const RSS_BUDGET_MB: u64 = 1536;

fn exc_count() -> usize {
    ((N as f64) * EXCITATORY_RATIO) as usize // 409
}

fn group_of(neuron_id: u16, exc: u16) -> u16 {
    let g = (u32::from(neuron_id) * u32::from(GROUPS) / u32::from(exc)) as u16;
    g.min(GROUPS.saturating_sub(1))
}

/// Which group is driven at `step` (GROUPS = silent gap) — the 1.5c schedule.
fn active_group_at(step: usize) -> u16 {
    let slot_len = ACTIVE_ON + OFF_GAP;
    let cycle = slot_len * u32::from(GROUPS);
    let within = (step as u32) % cycle;
    let slot = within / slot_len;
    if slot < u32::from(GROUPS) && within % slot_len < ACTIVE_ON {
        slot as u16
    } else {
        GROUPS
    }
}

fn make_inputs(i_active: i16) -> Vec<Vec<i16>> {
    let exc = exc_count() as u16;
    let mut inputs = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let active = active_group_at(step);
        let mut inp = vec![I_INH; N];
        for n in 0..exc {
            inp[n as usize] = if group_of(n, exc) == active {
                i_active
            } else {
                I_IDLE
            };
        }
        inputs.push(inp);
    }
    inputs
}

/// Peak RSS (VmHWM) in MB from /proc/self/status.
fn peak_rss_mb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()))
        })
        .map(|kb| kb / 1024)
        .unwrap_or(0)
}

fn xorshift64(state: u64) -> u64 {
    let mut x = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Census-matched control: same trit multiset, shuffled placement.
fn shuffled_copy(src: &[Trit]) -> Vec<Trit> {
    let mut v = src.to_vec();
    let mut rng = CONTROL_SEED;
    for i in (1..v.len()).rev() {
        rng = xorshift64(rng);
        let j = (rng % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// Build the substrate network from a trit matrix (D-2 build, verbatim).
fn build_from_trits(trits: &[Trit]) -> SpikingNeuralNetwork {
    let mut net = SpikingNeuralNetwork::new(
        N as u16,
        DT_US,
        NetworkTopology::Random { connectivity: 0.0 },
    )
    .expect("512-neuron net must construct");
    net.build_topology()
        .expect("zero-connectivity build (empty wiring) must succeed");
    for j in 0..N {
        for i in 0..N {
            if i != j {
                let w = trits[i * N + j].to_weight(GAMMA);
                net.add_synapse(j as u16, i as u16, w)
                    .expect("in-bounds, non-self edge");
            }
        }
    }
    net.finalize_synapses();
    net
}

/// One fixed-weight run's full record: the spike TRAIN as a (step, neuron)
/// bitset (step-major, 8 words per 512-neuron step) + per-neuron counts.
struct Train {
    words: Vec<u64>,
    counts: Vec<u64>,
    total: u64,
}

fn run_and_capture(net: &mut SpikingNeuralNetwork, inputs: &[Vec<i16>]) -> Train {
    net.set_plasticity_enabled(false);
    const WORDS_PER_STEP: usize = N / 64; // 512 neurons → 8 words
    let mut words = vec![0u64; STEPS * WORDS_PER_STEP];
    let mut counts = vec![0u64; N];
    let mut total = 0u64;
    for (t, inp) in inputs.iter().enumerate() {
        let spikes = net.step(inp).expect("step");
        for sp in &spikes {
            let n = sp.neuron_id as usize;
            words[t * WORDS_PER_STEP + n / 64] |= 1u64 << (n % 64);
            counts[n] += 1;
            total += 1;
        }
    }
    Train { words, counts, total }
}

/// (step, neuron) events present in exactly one train.
fn train_hamming(a: &Train, b: &Train) -> u64 {
    a.words
        .iter()
        .zip(&b.words)
        .map(|(x, y)| (x ^ y).count_ones() as u64)
        .sum()
}

/// Per-neuron rate-vector L1 (spike-count Manhattan distance).
fn rate_l1(a: &Train, b: &Train) -> u64 {
    a.counts
        .iter()
        .zip(&b.counts)
        .map(|(x, y)| x.abs_diff(*y))
        .sum()
}

fn main() {
    let t0 = std::time::Instant::now();
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "models/Ternary-Bonsai-4B-Q2_0.gguf".into()
    });
    println!("=== Session E stage 1: the amplitude sweep — where do weights begin to shape firing? ===");
    println!("file    : {path}");
    println!("slice   : {TENSOR}, first {N}×{N}, γ={GAMMA}, full-minus-diagonal, STDP OFF (fixed weights)");
    println!("drive   : 1.5c schedule verbatim; I_INH={I_INH} FIXED; I_ACTIVE swept over {:?}", AMPLITUDES);
    println!("nets    : imported / census-matched control (seed {CONTROL_SEED:#x}) / zero — shared drive + noise; only weights differ");
    println!("metric  : spike-TRAIN Hamming per pair (not totals) + per-neuron rate L1 + per-population Hz");
    println!();

    // ----- Decode (D-2 path, verbatim) -----
    let src: Vec<Trit> = {
        let buf = std::fs::read(&path).unwrap_or_else(|e| {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(1);
        });
        let f = GgufFile::parse(&buf).expect("GGUF container must parse");
        let info = f
            .tensors
            .iter()
            .find(|t| t.name == TENSOR)
            .unwrap_or_else(|| panic!("tensor {TENSOR} not found"));
        assert_eq!(info.ty, GGML_TYPE_Q2_0, "host tensor must be q2_0");
        assert_eq!(
            info.dims,
            vec![MODEL_COLS as u64, MODEL_ROWS as u64],
            "attn_q is [2560,4096] on the 4B"
        );
        let data = f.tensor_data(info).expect("tensor slice in bounds");
        assert_eq!(data.len(), MODEL_ROWS * ROW_BYTES);
        let mut out = Vec::with_capacity(N * N);
        let mut row_trits = vec![Trit::Zero; N];
        let mut scales = [0u16; N / 128];
        for r in 0..N {
            decode_q2_0(
                &data[r * ROW_BYTES..r * ROW_BYTES + N / 128 * 34],
                &mut row_trits,
                &mut scales,
            )
            .expect("real q2_0 bytes must decode");
            out.extend_from_slice(&row_trits);
        }
        out
    };
    let exc = exc_count();
    let secs = STEPS as f64 * f64::from(DT_US) / 1e6;

    println!("amp(μA) | E Hz (imp/ctl/zero)      | I Hz (imp/ctl/zero) | totals (i/c/z)   | H(i,c)  H(i,z)  H(c,z) | L1(i,c) L1(i,z) L1(c,z)");
    println!("--------+---------------------------+---------------------+------------------+-------------------------+------------------------");

    let mut a_star: Option<i16> = None;
    let mut first_divergence = String::new();
    for &amp in AMPLITUDES.iter() {
        let inputs = make_inputs(amp);
        let mut imported = build_from_trits(&src);
        let mut control = build_from_trits(&shuffled_copy(&src));
        let mut zero = build_from_trits(&[Trit::Zero; N * N]);
        let ti = run_and_capture(&mut imported, &inputs);
        let tc = run_and_capture(&mut control, &inputs);
        let tz = run_and_capture(&mut zero, &inputs);
        drop(imported);
        drop(control);
        drop(zero);

        let hz = |t: &Train, pop_hi: usize| -> f64 {
            let n: u64 = t.counts[..pop_hi].iter().sum();
            n as f64 / (secs * pop_hi as f64)
        };
        let ihz = |t: &Train, pop_lo: usize| -> f64 {
            let n: u64 = t.counts[pop_lo..].iter().sum();
            n as f64 / (secs * (N - pop_lo) as f64)
        };
        let hic = train_hamming(&ti, &tc);
        let hiz = train_hamming(&ti, &tz);
        let hcz = train_hamming(&tc, &tz);
        println!(
            "{amp:>7} | {:+.2} {:+.2} {:+.2} | {:+.2} {:+.2} {:+.2} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}",
            hz(&ti, exc),
            hz(&tc, exc),
            hz(&tz, exc),
            ihz(&ti, exc),
            ihz(&tc, exc),
            ihz(&tz, exc),
            ti.total,
            tc.total,
            tz.total,
            hic,
            hiz,
            hcz,
            rate_l1(&ti, &tc),
            rate_l1(&ti, &tz),
            rate_l1(&tc, &tz),
        );
        if (hic > 0 || hiz > 0 || hcz > 0) && a_star.is_none() {
            a_star = Some(amp);
            first_divergence =
                format!("H(i,c)={hic} H(i,z)={hiz} H(c,z)={hcz} L1(i,c)={} L1(i,z)={} L1(c,z)={}",
                    rate_l1(&ti, &tc), rate_l1(&ti, &tz), rate_l1(&tc, &tz));
        }
    }

    println!();
    println!("--- Verdict (criterion pre-registered: A* = highest amplitude with any pairwise train Hamming > 0) ---");
    match a_star {
        Some(a) => {
            println!("A* = {a} μA — the weight→firing channel OPENS at this amplitude.");
            println!("first-divergence row: {first_divergence}");
        }
        None => {
            println!("NO DIVERGENCE at any amplitude (with rates as printed above).");
            println!("The weight→firing channel does not open by amplitude alone — honest NO,");
            println!("recorded; the coupling redesign conversation reopens with this curve.");
        }
    }
    println!(
        "wall {:.1}s   peak RSS {} MB (budget {RSS_BUDGET_MB})",
        t0.elapsed().as_secs_f64(),
        peak_rss_mb()
    );
}
