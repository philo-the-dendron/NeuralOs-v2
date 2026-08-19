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

use neuralos_rt::{GgufFile, GGML_TYPE_Q2_0};
use neuralos_snn::{decode_q2_0, NetworkTopology, SpikingNeuralNetwork, SynapseType, Trit};

// ----- Geometry -----
const N: usize = 512; // slice side: 512 output rows × 512 input cols of attn_q
const TENSOR: &str = "blk.0.attn_q.weight";
const MODEL_COLS: usize = 2560; // 4B config: attn_q input width (emb)
const MODEL_ROWS: usize = 4096; // attn_q output width (32 heads × 128)
const ROW_BYTES: usize = (MODEL_COLS / 128) * 34; // 20 blocks × 34 B = 680
/// Substrate γ — the proven 1.5x regime constant (recorded decision).
const GAMMA: i16 = 125;

// ----- Drive (1.5c constants, verbatim) -----
const DT_US: u32 = 1000;
const EXCITATORY_RATIO: f64 = 0.8; // neuron-type split, as in 1.5c
const GROUPS: u16 = 4;
const ACTIVE_ON: u32 = 60;
const OFF_GAP: u32 = 40;
const I_ACTIVE: i16 = 600;
const I_IDLE: i16 = 0;
const I_INH: i16 = 600;
const INIT_STEPS: usize = ((ACTIVE_ON + OFF_GAP) * GROUPS as u32) as usize; // 400
const STEPS: usize = 2000;

// ----- Floors / bounds (thresholds printed with every check) -----
/// Δ-SI floor — the 1.5c ternary selectivity floor, applied to Δ-SI.
const SI_FLOOR: f64 = 0.05;
/// Firing ratio floor vs the census-matched fixed-weight control (1.5c's
/// SPIKE_RATIO_FLOOR).
const SPIKE_RATIO_FLOOR: f64 = 0.10;
/// Absolute floor (Hz/neuron) for "not degenerate" — one spike per neuron
/// per 10 s; below this a 512-net under 600 μA group drive is silent for all
/// intents. Recorded decision (the mission's floor is ratio-vs-reference;
/// this is the absolute companion the control reporting pins ask for).
const SPIKE_ABS_FLOOR_HZ: f64 = 0.10;
/// Hamming bound: majority of pretrained buckets must survive (mission:
/// "majority of pretrained buckets intact").
const HAMMING_BOUND: f64 = 0.50;
/// Fisher-Yates seed for the census-matched control (printed in evidence).
const CONTROL_SEED: u64 = 0x5EED_C0DE_0000_0002;
/// Memory budget for the whole run (the mission box).
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

fn make_inputs() -> Vec<Vec<i16>> {
    let exc = exc_count() as u16;
    let mut inputs = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let active = active_group_at(step);
        let mut inp = vec![I_INH; N];
        for n in 0..exc {
            inp[n as usize] = if group_of(n, exc) == active {
                I_ACTIVE
            } else {
                I_IDLE
            };
        }
        inputs.push(inp);
    }
    inputs
}

/// Peak RSS (VmHWM) in MB from /proc/self/status — the memory-box evidence.
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

/// Build the substrate network from a trit matrix (row-major N×N, weight
/// col→row: pre = column j, post = row i — matvec dataflow, recorded).
/// Full-minus-diagonal: 512×511 = 261,632 synapses, added pre-major (sorted)
/// then finalized through the public external-wiring path (session D-2's
/// substrate addition — the reverse CSR the LTP pass needs).
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

struct FixedStats {
    rate_hz: f64,
    quarter_hz: [f64; 4],
    own_active_hz: f64,
    own_idle_hz: f64,
    total_spikes: u64,
}

/// Fixed-weight run (STDP off). Reports over the full window, plus per-quarter
/// rates (the self-quench evidence) and per-group containment.
fn run_fixed(net: &mut SpikingNeuralNetwork, inputs: &[Vec<i16>]) -> FixedStats {
    let exc = exc_count() as u16;
    net.set_plasticity_enabled(false);
    let mut quarter_spikes = [0u64; 4];
    let mut a_sp = 0u64;
    let mut a_st = 0u64;
    let mut i_sp = 0u64;
    let mut i_st = 0u64;
    for (t, inp) in inputs.iter().enumerate() {
        let active = active_group_at(t);
        let spikes = net.step(inp).expect("step");
        quarter_spikes[t / (STEPS / 4)] += spikes.len() as u64;
        for sp in &spikes {
            if sp.neuron_id < exc {
                if group_of(sp.neuron_id, exc) == active {
                    a_sp += 1;
                } else {
                    i_sp += 1;
                }
            }
        }
    }
    // Own-active/idle exposure: over the whole schedule each group is active
    // for the same number of steps; count exposure exactly anyway.
    for t in 0..inputs.len() {
        let active = active_group_at(t);
        for n in 0..exc {
            if group_of(n, exc) == active {
                a_st += 1;
            } else {
                i_st += 1;
            }
        }
    }
    let total: u64 = quarter_spikes.iter().sum();
    let secs_total = STEPS as f64 * f64::from(DT_US) / 1e6;
    let q_secs = secs_total / 4.0;
    FixedStats {
        rate_hz: total as f64 / (secs_total * N as f64),
        quarter_hz: quarter_spikes.map(|q| q as f64 / (q_secs * N as f64)),
        own_active_hz: if a_st > 0 {
            a_sp as f64 / (a_st as f64) * 1000.0
        } else {
            0.0
        },
        own_idle_hz: if i_st > 0 {
            i_sp as f64 / (i_st as f64) * 1000.0
        } else {
            0.0
        },
        total_spikes: total,
    }
}

/// Trit bucket as −1/0/+1 for means and deltas.
fn trit_val(t: Trit) -> f64 {
    match t {
        Trit::MinusOne => -1.0,
        Trit::Zero => 0.0,
        Trit::One => 1.0,
    }
}

struct HybridStats {
    learn_rate_hz: f64,
    quarter_hz: [f64; 4],
    own_active_hz: f64,
    own_idle_hz: f64,
    flips: u64,
    /// transitions[from][to], indices: 0=−1, 1=0, 2=+1.
    census: [[u64; 3]; 3],
    final_trits: Vec<Trit>,
    plasticity_events: u64,
    /// In-window STDP pairing histogram (session F): the Hebbian-
    /// attribution evidence — same_step (co-fire tie-break, LTD),
    /// post_leads (LTD), pre_leads (LTP).
    pairs_same_step: u64,
    pairs_post_leads: u64,
    pairs_pre_leads: u64,
    /// Per-class (E→E intra/inter) cumulative RAW STDP deltas and the
    /// clamp-absorbed remainders (session G): decides whether the realized
    /// bucket movement was pairing-driven or machinery-driven.
    raw_intra: i64,
    raw_inter: i64,
    absorbed_intra: i64,
    absorbed_inter: i64,
    n_intra: u64,
    n_inter: u64,
    cofire_intra: f64,
    cofire_inter: f64,
}

/// The G3 learning run: init cycle (STDP off) → learning (STDP on + stochastic
/// ternary bucket-flips at γ=125). Tracks the bucket-transition census, firing
/// sanity, and E→E co-firing structure over the learning phase.
fn run_hybrid(trits: &[Trit], inputs: &[Vec<i16>]) -> HybridStats {
    let exc = exc_count() as u16;
    let mut net = build_from_trits(trits);
    net.set_plasticity_enabled(false);
    for inp in &inputs[..INIT_STEPS] {
        net.step(inp).expect("init step");
    }
    let mut prev: Vec<Trit> = net
        .synapses()
        .iter()
        .map(|s| Trit::from_weight(s.weight, GAMMA))
        .collect();
    net.set_plasticity_enabled(true);

    let learn = &inputs[INIT_STEPS..];
    let learn_words = learn.len().div_ceil(64);
    // Per-E-neuron firing bitset over the learning phase (co-fire evidence).
    let mut fired: Vec<Vec<u64>> = vec![vec![0u64; learn_words]; exc as usize];
    let mut quarter_spikes = [0u64; 4];
    let mut a_sp = 0u64;
    let mut a_st = 0u64;
    let mut i_sp = 0u64;
    let mut i_st = 0u64;
    let mut flips = 0u64;
    let mut census = [[0u64; 3]; 3];
    let tix = |t: Trit| -> usize {
        match t {
            Trit::MinusOne => 0,
            Trit::Zero => 1,
            Trit::One => 2,
        }
    };
    for (t, inp) in learn.iter().enumerate() {
        let active = active_group_at(INIT_STEPS + t);
        let spikes = net.step(inp).expect("learn step");
        net.stochastic_ternary_step(GAMMA);
        quarter_spikes[t / (learn.len() / 4)] += spikes.len() as u64;
        for sp in &spikes {
            if sp.neuron_id < exc {
                let n = sp.neuron_id as usize;
                fired[n][t / 64] |= 1u64 << (t % 64);
                if group_of(sp.neuron_id, exc) == active {
                    a_sp += 1;
                } else {
                    i_sp += 1;
                }
            }
        }
        for (k, s) in net.synapses().iter().enumerate() {
            let cur = Trit::from_weight(s.weight, GAMMA);
            if cur != prev[k] {
                census[tix(prev[k])][tix(cur)] += 1;
                flips += 1;
                prev[k] = cur;
            }
        }
    }
    for t in 0..learn.len() {
        let active = active_group_at(INIT_STEPS + t);
        for n in 0..exc {
            if group_of(n, exc) == active {
                a_st += 1;
            } else {
                i_st += 1;
            }
        }
    }

    // E→E co-firing: mean same-step co-fire rate for intra vs inter pairs.
    let (mut isum, mut inum, mut esum, mut enum_) = (0.0_f64, 0u64, 0.0_f64, 0u64);
    for pre in 0..exc {
        for post in 0..exc {
            if pre == post {
                continue;
            }
            let mut both = 0u32;
            for (wa, wb) in fired[pre as usize].iter().zip(&fired[post as usize]) {
                both += (wa & wb).count_ones();
            }
            let rate = f64::from(both) / learn.len() as f64;
            if group_of(pre, exc) == group_of(post, exc) {
                isum += rate;
                inum += 1;
            } else {
                esum += rate;
                enum_ += 1;
            }
        }
    }

    // Session G mechanism counters: per-class raw STDP drift + clamp
    // absorption, read from the synapses themselves (E→E pairs only,
    // matching the selectivity classes).
    let (mut raw_intra, mut raw_inter) = (0_i64, 0_i64);
    let (mut absorbed_intra, mut absorbed_inter) = (0_i64, 0_i64);
    let (mut n_intra, mut n_inter) = (0_u32, 0_u32);
    for s in net.synapses() {
        if s.pre_neuron_id < exc && s.post_neuron_id < exc {
            if group_of(s.pre_neuron_id, exc) == group_of(s.post_neuron_id, exc) {
                raw_intra += s.raw_stdp_delta;
                absorbed_intra += s.absorbed_delta;
                n_intra += 1;
            } else {
                raw_inter += s.raw_stdp_delta;
                absorbed_inter += s.absorbed_delta;
                n_inter += 1;
            }
        }
    }

    let total_learn_spikes: u64 = quarter_spikes.iter().sum();
    let secs = learn.len() as f64 * f64::from(DT_US) / 1e6;
    HybridStats {
        learn_rate_hz: total_learn_spikes as f64 / (secs * N as f64),
        quarter_hz: quarter_spikes.map(|q| q as f64 / ((secs / 4.0) * N as f64)),
        own_active_hz: if a_st > 0 {
            a_sp as f64 / (a_st as f64) * 1000.0
        } else {
            0.0
        },
        own_idle_hz: if i_st > 0 {
            i_sp as f64 / (i_st as f64) * 1000.0
        } else {
            0.0
        },
        flips,
        census,
        final_trits: net
            .synapses()
            .iter()
            .map(|s| Trit::from_weight(s.weight, GAMMA))
            .collect(),
        plasticity_events: net.stats().plasticity_events,
        pairs_same_step: net.stats().stdp_pairs_same_step,
        pairs_post_leads: net.stats().stdp_pairs_post_leads,
        pairs_pre_leads: net.stats().stdp_pairs_pre_leads,
        raw_intra,
        raw_inter,
        absorbed_intra,
        absorbed_inter,
        n_intra: n_intra as u64,
        n_inter: n_inter as u64,
        cofire_intra: if inum > 0 { isum / inum as f64 } else { 0.0 },
        cofire_inter: if enum_ > 0 { esum / enum_ as f64 } else { 0.0 },
    }
}

fn main() {
    let t0 = std::time::Instant::now();
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "models/Ternary-Bonsai-4B-Q2_0.gguf".into()
    });
    println!("=== Session D slice 2: the hybrid experiment — Bonsai Q2_0 → Trit → SNN → STDP ===");
    println!("file    : {path}");
    println!("host    : {TENSOR}, first {N}×{N} slice (col j = pre → row i = post, matvec dataflow)");
    println!("substrate: γ={GAMMA} (the proven 1.5x constant; LLM fp16 block scales unused — recorded), full-minus-diagonal density, 1.5c drive verbatim");
    println!();

    // ----- Decode (scoped: file buffer drops before any network is built) -----
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
        // 4B's attn_q is genuinely non-square: [in=2560, out=4096=32 heads ×
        // 128] — ISC-52's config finding. The slice is the first 512 OUTPUT
        // rows × first 512 INPUT cols.
        assert_eq!(
            info.dims,
            vec![MODEL_COLS as u64, MODEL_ROWS as u64],
            "attn_q is [2560,4096] on the 4B"
        );
        let data = f.tensor_data(info).expect("tensor slice in bounds");
        assert_eq!(
            data.len(),
            MODEL_ROWS * ROW_BYTES,
            "tensor byte size = 4096 rows × 680 B"
        );
        let mut out = Vec::with_capacity(N * N);
        let mut row_trits = vec![Trit::Zero; N];
        let mut scales = [0u16; N / 128];
        for r in 0..N {
            decode_q2_0(
                &data[r * ROW_BYTES..r * ROW_BYTES + N / 128 * 34],
                &mut row_trits,
                &mut scales,
            )
            .expect("real q2_0 bytes must decode (code 3 would be loud)");
            out.extend_from_slice(&row_trits);
        }
        out
    };
    println!("decode  : {} trits from real Q2_0 bytes (peak RSS so far is the file buffer)", src.len());

    // ----- G1: census + round-trip on the imported substrate weights -----
    println!();
    println!("--- G1: IMPORT INTEGRITY (trit-preserving by construction) ---");
    let (mut plus, mut zero, mut minus) = (0u64, 0u64, 0u64);
    for t in &src {
        match t {
            Trit::One => plus += 1,
            Trit::Zero => zero += 1,
            Trit::MinusOne => minus += 1,
        }
    }
    let total = (N * N) as u64;
    println!(
        "  census (real pretrained slice, first measurement): +1 × {plus}   0 × {zero}   −1 × {minus}   (of {total})"
    );
    println!(
        "  zero fraction: {:.4}   nonzero: {:.4}",
        zero as f64 / total as f64,
        (plus + minus) as f64 / total as f64
    );

    let mut imported = build_from_trits(&src);
    assert_eq!(imported.synapse_count() as u64, total - N as u64);
    // Sign-asymmetry census: SynapseType owns PLASTICITY BOUNDS (sign-inferred
    // at construction); the substrate type-sign invariant is intentionally
    // not imposed on imported weights — propagation reads the stored sign.
    let (mut exc_t, mut inh_t) = (0u64, 0u64);
    for s in imported.synapses() {
        if s.synapse_type == SynapseType::Excitatory {
            exc_t += 1;
        } else {
            inh_t += 1;
        }
    }
    println!(
        "  substrate bounds census: Excitatory-clamped (0,+γ) × {exc_t}   Inhibitory-clamped (−γ) × {inh_t}"
    );
    println!(
        "  → one-directional flips per class: +γ↔0, −γ↔0, 0→+γ only (no sign crossing — measured in G3)"
    );

    let mut mismatch = 0u64;
    let mut k = 0usize;
    for j in 0..N {
        for i in 0..N {
            if i == j {
                continue;
            }
            let got = Trit::from_weight(imported.synapses()[k].weight, GAMMA);
            if got != src[i * N + j] {
                mismatch += 1;
            }
            k += 1;
        }
    }
    let g1_pass = mismatch == 0;
    println!(
        "  round-trip: substrate weight → Trit::from_weight({GAMMA}) vs source — {mismatch} mismatches / {k} synapses"
    );
    println!("  [G1: {}]", if g1_pass { "PASS — import is trit-exact" } else { "FAIL" });
    println!();

    // ----- G2: spiking fidelity, imported vs census-matched control -----
    println!("--- G2: SPIKING FIDELITY (fixed weights, STDP off, identical 1.5c drive) ---");
    println!(
        "  DENSITY NOTE: full-minus-diagonal ({} synapses), not the balanced-0.8 sparse shape",
        total - N as u64
    );
    println!("                the 1.5c constants were proven on — absolute rates compare ONLY");
    println!("                within this experiment (imported vs control vs floor).");
    println!(
        "  control: census-matched shuffle (same +/0/− multiset), seed {CONTROL_SEED:#x}, STDP off"
    );
    println!("  interpretation matrix (stated up front):");
    println!("    both above floor      → density suffices to spike; the structure claim lives in G3");
    println!("    only pretrained above → G2 passes with the stronger structural claim");
    println!("    neither above floor   → degenerate under this drive");
    let imported_stats = run_fixed(&mut imported, &make_inputs());
    let ctrl_trits = shuffled_copy(&src);
    let mut control = build_from_trits(&ctrl_trits);
    let control_stats = run_fixed(&mut control, &make_inputs());
    // Structure-free third comparator (added from the session's own
    // diagnostic: imported and control matched EXACTLY, and a zero-weight
    // net reproduces both bit-for-bit — the drive-dominated mechanism below).
    let zero_trits = vec![Trit::Zero; N * N];
    let mut zeronet = build_from_trits(&zero_trits);
    let zero_stats = run_fixed(&mut zeronet, &make_inputs());
    println!(
        "  imported (pretrained): {:.2} Hz/neuron ({} spikes)",
        imported_stats.rate_hz, imported_stats.total_spikes
    );
    println!(
        "  control (random)     : {:.2} Hz/neuron ({} spikes)",
        control_stats.rate_hz, control_stats.total_spikes
    );
    println!(
        "  zero-w (no structure): {:.2} Hz/neuron ({} spikes)",
        zero_stats.rate_hz, zero_stats.total_spikes
    );
    println!("  absolute floor       : {SPIKE_ABS_FLOOR_HZ:.2} Hz/neuron");
    let imported_above = imported_stats.rate_hz >= SPIKE_ABS_FLOOR_HZ;
    let control_above = control_stats.rate_hz >= SPIKE_ABS_FLOOR_HZ;
    let ratio = if control_stats.rate_hz > 0.0 {
        imported_stats.rate_hz / control_stats.rate_hz
    } else {
        f64::INFINITY
    };
    println!(
        "  (a) pretrained vs control : {:.2}× (ratio floor {SPIKE_RATIO_FLOOR:.2}×)",
        ratio
    );
    println!(
        "  (b) imported vs floor : {}   control vs floor: {}",
        if imported_above { "PASS" } else { "FAIL" },
        if control_above { "PASS" } else { "FAIL" }
    );
    let ratio_ok = control_stats.rate_hz <= 0.0 || ratio >= SPIKE_RATIO_FLOOR;
    let observed = if imported_above && control_above {
        "BOTH above floor — density suffices to spike; the structure claim lives in G3"
    } else if imported_above {
        "ONLY pretrained above floor — G2 passes with the stronger structural claim"
    } else {
        "NEITHER above floor — degenerate under this drive"
    };
    println!("  [observed: {observed}]");
    if imported_stats.total_spikes == control_stats.total_spikes
        && imported_stats.total_spikes == zero_stats.total_spikes
    {
        println!("  mechanism: all three comparators identical → this drive regime is");
        println!("             DRIVE-DOMINATED — recurrent ±12 μA (weight/10) currents never");
        println!("             gate a spike decision at I_ACTIVE=600, so G2 verifies");
        println!("             non-degeneracy + sustain but cannot discriminate structure;");
        println!("             the structure claim lives in G3 (STDP reads weights pairwise).");
    }
    println!(
        "  quench (imported, Hz/neuron by quarter): {:.2} {:.2} {:.2} {:.2} — last quarter {}",
        imported_stats.quarter_hz[0],
        imported_stats.quarter_hz[1],
        imported_stats.quarter_hz[2],
        imported_stats.quarter_hz[3],
        if imported_stats.quarter_hz[3] > 0.0 { "> 0 PASS (no self-quench)" } else { "= 0 FAIL (quench)" }
    );
    println!(
        "  containment (imported): own-active {:.1} Hz vs own-idle {:.1} Hz ({:.1}×)",
        imported_stats.own_active_hz,
        imported_stats.own_idle_hz,
        if imported_stats.own_idle_hz > 0.0 {
            imported_stats.own_active_hz / imported_stats.own_idle_hz
        } else {
            f64::INFINITY
        }
    );
    println!(
        "  containment (control) : own-active {:.1} Hz vs own-idle {:.1} Hz ({:.1}×)",
        control_stats.own_active_hz,
        control_stats.own_idle_hz,
        if control_stats.own_idle_hz > 0.0 {
            control_stats.own_active_hz / control_stats.own_idle_hz
        } else {
            f64::INFINITY
        }
    );
    let g2_pass = imported_above && ratio_ok && imported_stats.quarter_hz[3] > 0.0;
    println!(
        "  [G2: {} — imported ≥ floor: {}, ≥ {:.2}× control: {}, sustained: {}]",
        if g2_pass { "PASS" } else { "FAIL" },
        if imported_above { "yes" } else { "no" },
        SPIKE_RATIO_FLOOR,
        if ratio_ok { "yes" } else { "no" },
        if imported_stats.quarter_hz[3] > 0.0 { "yes" } else { "no" },
    );
    println!();

    // ----- G3: selective adaptation on pretrained structure -----
    println!("--- G3: SELECTIVE ADAPTATION (STDP on + stochastic flips, γ={GAMMA}, 1.5c schedule) ---");
    let h = run_hybrid(&src, &make_inputs());
    println!(
        "  input structure (learn phase): co-fire intra={:.4} inter={:.4} ({:.1}×); drive containment own-active {:.1} Hz vs own-idle {:.1} Hz",
        h.cofire_intra,
        h.cofire_inter,
        if h.cofire_inter > 0.0 { h.cofire_intra / h.cofire_inter } else { f64::INFINITY },
        h.own_active_hz,
        h.own_idle_hz
    );
    println!(
        "  firing (learn phase)  : {:.2} Hz/neuron (vs control {:.2} — floor {:.2}×); quarters {:.2} {:.2} {:.2} {:.2}",
        h.learn_rate_hz,
        control_stats.rate_hz,
        SPIKE_RATIO_FLOOR,
        h.quarter_hz[0],
        h.quarter_hz[1],
        h.quarter_hz[2],
        h.quarter_hz[3]
    );
    println!(
        "  plasticity events     : {}   bucket flips: {} (freeze if 0)",
        h.plasticity_events, h.flips
    );
    println!(
        "  STDP pairing histogram: same-step {} · post-leads {} · pre-leads {} (in-window; the pre-leads share is the Hebbian evidence)",
        h.pairs_same_step, h.pairs_post_leads, h.pairs_pre_leads
    );
    println!(
        "  raw STDP drift (E→E): intra {:+} over {} syns (mean {:+.4}/syn) · inter {:+} over {} — the pairing-driven sum BEFORE flips/clamps",
        h.raw_intra,
        h.n_intra,
        h.raw_intra as f64 / h.n_intra as f64,
        h.raw_inter,
        h.n_inter
    );
    println!(
        "  clamp-absorbed     : intra {:+} · inter {:+} — bounds-asymmetry evidence",
        h.absorbed_intra, h.absorbed_inter
    );

    // Bucket-transition census + Hamming vs the imported original.
    let src_iter_order: Vec<Trit> = {
        // synapse insertion order = pre-major (j outer, i inner), same as build.
        let mut v = Vec::with_capacity(N * (N - 1));
        for j in 0..N {
            for i in 0..N {
                if i != j {
                    v.push(src[i * N + j]);
                }
            }
        }
        v
    };
    let tix = |t: Trit| -> usize {
        match t {
            Trit::MinusOne => 0,
            Trit::Zero => 1,
            Trit::One => 2,
        }
    };
    let names = ["−1", " 0", "+1"];
    println!("  bucket-transition census (learn phase):");
    let mut impossible = 0u64;
    for (from, row) in h.census.iter().enumerate() {
        let mut line = String::new();
        for (to, &ct) in row.iter().enumerate() {
            line.push_str(&format!("  {}→{} × {:>7}", names[from], names[to], ct));
            // The bounds asymmetry forbids sign crossing: +1→−1, −1→+1.
            if (from == 2 && to == 0) || (from == 0 && to == 2) {
                impossible += ct;
            }
        }
        println!("{line}");
    }
    let mut hamming = 0u64;
    let mut retained = [0u64; 3];
    let mut class_n = [0u64; 3];
    for (k, &final_t) in h.final_trits.iter().enumerate() {
        let s0 = src_iter_order[k];
        class_n[tix(s0)] += 1;
        if final_t == s0 {
            retained[tix(s0)] += 1;
        } else {
            hamming += 1;
        }
    }
    let n_syn = src_iter_order.len() as u64;
    let hamming_frac = hamming as f64 / n_syn as f64;
    println!(
        "  Hamming vs imported   : {}/{} = {:.4} changed buckets (bound < {HAMMING_BOUND:.2} — majority intact)",
        hamming,
        n_syn,
        hamming_frac
    );
    println!(
        "  retention by source class: −1 {:.1}%   0 {:.1}%   +1 {:.1}% intact",
        retained[0] as f64 / class_n[0] as f64 * 100.0,
        retained[1] as f64 / class_n[1] as f64 * 100.0,
        retained[2] as f64 / class_n[2] as f64 * 100.0
    );
    if impossible == 0 {
        println!("  sign-crossing transitions (+1↔−1): 0 — the bounds asymmetry held exactly");
    } else {
        println!("  sign-crossing transitions (+1↔−1): {impossible} — bounds violated (BUG)");
    }

    // Selectivity: Δ-SI (gate-bearing) + 1.5c level-SI (supporting).
    let exc = exc_count() as u16;
    let mut d_intra = Vec::new();
    let mut d_inter = Vec::new();
    let mut lvl_intra = Vec::new();
    let mut lvl_inter = Vec::new();
    let mut k = 0usize;
    for j in 0..N {
        for i in 0..N {
            if i == j {
                continue;
            }
            let (pre, post) = (j as u16, i as u16);
            if pre < exc && post < exc {
                let delta = trit_val(h.final_trits[k]) - trit_val(src_iter_order[k]);
                if group_of(pre, exc) == group_of(post, exc) {
                    d_intra.push(delta);
                    lvl_intra.push(trit_val(h.final_trits[k]));
                } else {
                    d_inter.push(delta);
                    lvl_inter.push(trit_val(h.final_trits[k]));
                }
            }
            k += 1;
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let din = mean(&d_intra);
    let dit = mean(&d_inter);
    let d_denom = dit.abs() + din.abs();
    let d_si = if d_denom > f64::EPSILON {
        (dit - din) / d_denom
    } else {
        0.0
    };
    let lin = mean(&lvl_intra);
    let lit = mean(&lvl_inter);
    let l_denom = lit + lin;
    let l_si = if l_denom.abs() > f64::EPSILON {
        (lit - lin) / l_denom
    } else {
        0.0
    };
    println!(
        "  E→E map: {} intra / {} inter pairs",
        d_intra.len(),
        d_inter.len()
    );
    // Session F criterion (amended): the GATE is the raw, non-degenerate
    // field — intra |mean Δ| (degree of discrimination). The DIRECTION is
    // the era's mechanism label, printed, never gated. Δ-SI is demoted to
    // a supporting label: the 1.5c schedule's 40 ms group gaps put every
    // inter pair outside the 20 ms STDP window, so inter Δ ≡ 0 by geometry
    // in every era and |Δ-SI| ≡ 1 whenever any movement exists — it cannot
    // gate on degree. (Second-reviewer finding, adopted.)
    // Session G amendment: the label is COMPUTED from the counters, not
    // inferred from the sign. Measured decomposition (live-wire D-2): raw
    // intra drift is NET NEGATIVE (−739,295; LTD events outnumber LTP),
    // the E-class 0-floor absorbs −839,029 of it, and the APPLIED residue
    // is positive (+99,734) — the class-differential is timing-driven
    // (inter pairs never pair: 40 ms gaps), the DIRECTION is bounds-driven.
    // Simple "Hebbian-carried" was an inference the counters refute.
    let applied_intra = h.raw_intra - h.absorbed_intra;
    let mechanism = if din.abs() < f64::EPSILON {
        "none — no differential movement between classes"
    } else if h.raw_intra > 0 && din > 0.0 {
        "Hebbian-carried — raw LTP pairings dominate intra drift (counted)"
    } else if h.raw_intra < 0 && applied_intra > 0 && din > 0.0 {
        "PAIRING-SELECTIVE, CLAMP-RECTIFIED — intra co-firing drives a net-NEGATIVE raw drift; the 0-floor absorbs the LTD and the applied residue potentiates (class-differential timing-driven; direction bounds-driven)"
    } else {
        "LTD-carried — intra depressed more (net applied drift negative)"
    };
    println!(
        "  mean Δ (final − imported): intra {din:+.4}   inter {dit:+.4}"
    );
    println!("  mechanism label : [{mechanism}]");
    println!(
        "  intra |mean Δ| (GATE) : {:.4}   (floor {SI_FLOOR:.2} — the non-degenerate degree of discrimination)",
        din.abs()
    );
    println!(
        "  Δ-SI (label)    : {d_si:+.4}   (supporting only: inter Δ ≡ 0 by schedule geometry — 40 ms gaps vs 20 ms window — so |Δ-SI| ≡ 1 whenever movement exists)"
    );
    println!(
        "  level-SI (1.5c formula, confounded by pretrained levels — supporting): {:.4} (intra {lin:+.3} / inter {lit:+.3} final mean trit)",
        l_si
    );

    // ----- Verdict -----
    let firing_ok = h.learn_rate_hz >= SPIKE_RATIO_FLOOR * control_stats.rate_hz.max(0.0)
        && h.learn_rate_hz > 0.0;
    let not_frozen = h.flips > 0;
    let not_collapsed = hamming_frac < HAMMING_BOUND;
    let selective = din.abs() >= SI_FLOOR;
    println!();
    println!("--- Verdict ---");
    println!("  G1 import trit-exact        : {}", if g1_pass { "PASS" } else { "FAIL" });
    println!("  G2 non-degenerate sustained : {}", if g2_pass { "PASS" } else { "FAIL" });
    println!("  firing under STDP sustained : {}", if firing_ok { "PASS" } else { "FAIL" });
    println!("  not frozen (flips > 0)      : {}", if not_frozen { "PASS" } else { "FAIL" });
    println!(
        "  not collapsed (Hamming < {:.2}) : {}",
        HAMMING_BOUND,
        if not_collapsed { "PASS" } else { "FAIL" }
    );
    println!(
        "  selective (intra |Δ| ≥ {:.2}) : {}",
        SI_FLOOR,
        if selective { "PASS" } else { "FAIL" }
    );
    let verdict = if !g1_pass {
        "DEGENERATE — import is not trit-exact (G1 failed; nothing downstream is meaningful)"
    } else if !g2_pass || !firing_ok {
        if !g2_pass {
            "DEGENERATE — imported-weight network does not fire non-degenerately under the 1.5c drive (G2 failed)"
        } else {
            "DEGENERATE — firing collapsed below the ratio floor during adaptation (STDP-on run)"
        }
    } else if !not_frozen {
        "FROZEN — zero bucket flips on real pretrained weights (no adaptation without... more than local STDP)"
    } else if !not_collapsed {
        "COLLAPSES — STDP destroyed the majority of pretrained buckets (Hamming ≥ bound)"
    } else if !selective {
        "COLLAPSES — uniform/no selectivity: correlated pairs did not modify differently from uncorrelated (intra |mean Δ| below floor)"
    } else {
        "ADAPTS — pretrained structure survives AND discriminates under local STDP"
    };
    println!();
    println!("HYBRID GATE: {verdict}");
    let pass = g1_pass && g2_pass && firing_ok && not_frozen && not_collapsed && selective;
    println!(
        "wall {:.1}s   peak RSS {} MB (budget {RSS_BUDGET_MB})",
        t0.elapsed().as_secs_f64(),
        peak_rss_mb()
    );
    if !pass {
        std::process::exit(1);
    }
}
