//! Stage 1.5c — ternary STDP selectivity under structured (correlated) input.
//!
//! Stage 1.5b proved the stochastic-flip MECHANISM works (nonzero flips, non-
//! collapsed) but under *uniform synchronous* drive — a degenerate one-
//! directional collapse. That was movement, not learning. The open question
//! this example answers: does ternary STDP DISCRIMINATE under structured input
//! (selectively weaken by correlation), or does it always collapse regardless
//! of structure? This gates whether the Stage 2 format bridge is worth building.
//!
//! # Result: YES (ternary discriminates, at parity with i16)
//!
//! Under correlated group-structured input, i16 E→E synapses split cleanly:
//! intra-group (correlated) pairs depress to 0, inter-group (uncorrelated)
//! pairs stay at their initial value (Selectivity Index = 1.0). Ternary-
//! stochastic reproduces that differential at parity (SI = 1.0): 100% of intra
//! synapses stochastically flip +γ→0 while 100% of inter synapses stay at +γ,
//! and spiking stays non-collapsed. Ternary learns selectively, not just moves.
//!
//! # Substrate note (Stage 1.5d)
//!
//! This experiment was re-run after Stage 1.5d corrected the substrate to a
//! *full pairwise* STDP rule (the missing post-firing LTP half was added; see
//! `network::update_plasticity`'s two disjoint passes). The earlier 1.5c run
//! reported SI = 0.985 on a structurally LTD-only rule; on the full biphasic
//! rule the gap closes to SI = 1.000 (the 0.8% stochastic-noise residual is
//! gone, with more plasticity events landing every intra synapse). The i16
//! control is unchanged (SI = 1.000 in both runs). The selectivity YES is not a
//! half-STDP artifact — it survives the full rule.
//!
//! # Setup
//!
//! Excitatory neurons are split into `GROUPS` groups. One group at a time is
//! driven (sustained `I_ACTIVE`) for `ACTIVE_ON` ms, then an `OFF_GAP` ms
//! silent gap precedes the next group (round-robin). Intra-group pairs are
//! co-active throughout their window (high-rate correlated firing → frequent
//! same-step co-fires); inter-group pairs are never co-active AND adjacent
//! windows are gap-separated (> the STDP window), so inter LTD/LTP ≈ 0. The same
//! deterministic schedule is replayed across regimes. An init cycle (STDP off)
//! runs first to defeat the `last_spike_time_us = 0` ("never fired") artifact.
//!
//! # Honest scope of the selectivity signal
//!
//! Under *this* synchronous-drive regime the intra-group discrimination is
//! carried by the LTD branch (correlated co-fire → same-step tie-break →
//! depression = anti-Hebbian decorrelation), not by Hebbian LTP: the same-step
//! tie-break biases coincidences to LTD, and the gap places inter pairs outside
//! the window in both directions. The full rule is genuinely biphasic (LTP fires
//! across the network — the example's rule-direction check shows nonzero `up`),
//! but the *selective* signal on E→E is LTD-carried. A pre-before-post sequence
//! regime would exercise LTP-driven selectivity directly; that is a different
//! experiment, out of scope for this diagnostic.
//!
//! # Balanced inhibition
//!
//! With `I_INH = 0`, recurrent E→E spreads activity to every group synchronously
//! and the input structure is destroyed; a tonic inhibitory drive is needed so
//! only the actively-driven group fires. This is reported below as a
//! drive-selectivity check (own-active rate vs own-idle rate).
//!
//! # The gate (falsifier)
//!
//! - ESSENTIAL CONTROL: i16 baseline MUST discriminate (intra ≠ inter). If
//!   not, the input has no learnable structure.
//! - YES: ternary shows real selectivity (comparable SI to i16) AND spiking
//!   stays sane.
//! - NO: ternary collapses uniformly despite i16 discriminating.
//! - MARGINAL: report the numbers honestly. No mid-experiment rescues.

use neuralos_snn::{NetworkTopology, SpikingNeuralNetwork, Trit};

const NEURONS: u16 = 128;
const EXCITATORY_RATIO: f64 = 0.8;
const DT_US: u32 = 1000; // 1 ms / step
const GROUPS: u16 = 4;
/// Each group is continuously driven (active) for `ACTIVE_ON` steps; then an
/// `OFF_GAP`-step silent gap precedes the next group (round-robin). The gap is
/// the selectivity guard: without it, adjacent groups in the rotation are
/// back-to-back, so when group B's pre fires at the very start of its window,
/// group A's post fired at the end of the previous window only 1–10 ms ago —
/// inside the STDP window — applying spurious boundary LTD to adjacent inter
/// pairs. `OFF_GAP` > 20 ms (the STDP window) defeats this.
const ACTIVE_ON: u32 = 60;
const OFF_GAP: u32 = 40;
const I_ACTIVE: i16 = 600; // sustained drive to the one active group
const I_IDLE: i16 = 0; // excitatory neurons otherwise
const I_INH: i16 = 600; // tonic inhibitory drive — without it recurrent E→E
                        // spreads activity to every group synchronously and destroys the structure.
/// Init phase = one full rotation cycle, STDP OFF. Defeats the
/// `last_spike_time_us = 0` ("never fired") artifact.
const INIT_STEPS: usize = ((ACTIVE_ON + OFF_GAP) * GROUPS as u32) as usize; // one cycle
const STEPS: usize = 2000;

/// i16 selectivity index floor for the control to count as "discriminates".
const CONTROL_SI_FLOOR: f64 = 0.10;
/// Ternary SI floor for a YES (must also share the i16 sign and stay spiking).
const TERNARY_SI_FLOOR: f64 = 0.05;
/// Spiking ratio floor (vs fixed-weight reference) for non-collapse.
const SPIKE_RATIO_FLOOR: f64 = 0.10;

fn exc_count() -> u16 {
    (f64::from(NEURONS) * EXCITATORY_RATIO) as u16
}

fn group_of(neuron_id: u16, exc: u16) -> u16 {
    let g = (u32::from(neuron_id) * u32::from(GROUPS) / u32::from(exc)) as u16;
    g.min(GROUPS.saturating_sub(1))
}

fn build() -> SpikingNeuralNetwork {
    let mut net = SpikingNeuralNetwork::new(NEURONS, DT_US, NetworkTopology::default())
        .expect("balanced 128-neuron net must construct");
    net.build_topology().expect("topology must build");
    net
}

/// Pre-generate the per-step input-current vectors. **Gapped rotating drive**:
/// one group active (sustained `I_ACTIVE`) for `ACTIVE_ON` steps, then an
/// `OFF_GAP`-step silent gap, then the next group (round-robin). One
/// deterministic schedule, replayed across regimes for fairness.
fn make_inputs() -> (Vec<Vec<i16>>, Vec<u16>) {
    let exc = exc_count();
    let slot_len = ACTIVE_ON + OFF_GAP;
    let cycle = slot_len * u32::from(GROUPS);
    let mut inputs = Vec::with_capacity(STEPS);
    let mut active_log = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let within = (step as u32) % cycle;
        let slot = within / slot_len;
        let within_slot = within % slot_len;
        let (active_group, mask) = if slot < u32::from(GROUPS) && within_slot < ACTIVE_ON {
            let g = slot as u16;
            (g, 1u16 << g)
        } else {
            (GROUPS, 0u16) // silent gap
        };
        let mut inp = vec![I_INH; NEURONS as usize];
        for n in 0..exc {
            inp[n as usize] = if group_of(n, exc) == active_group {
                I_ACTIVE
            } else {
                I_IDLE
            };
        }
        inputs.push(inp);
        active_log.push(mask);
    }
    (inputs, active_log)
}

/// E→E synapse map: `(synapse_index, pre, post)` plus intra/inter index split.
/// Topology is deterministic (fixed seed), so this is captured once and reused.
struct EeMap {
    all: Vec<(usize, u16, u16)>,
    intra: Vec<usize>,
    inter: Vec<usize>,
}

fn ee_map() -> EeMap {
    let net = build();
    let exc = exc_count();
    let mut all = Vec::new();
    let mut intra = Vec::new();
    let mut inter = Vec::new();
    for (idx, s) in net.synapses().iter().enumerate() {
        if s.pre_neuron_id < exc && s.post_neuron_id < exc {
            all.push((idx, s.pre_neuron_id, s.post_neuron_id));
            if group_of(s.pre_neuron_id, exc) == group_of(s.post_neuron_id, exc) {
                intra.push(idx);
            } else {
                inter.push(idx);
            }
        }
    }
    EeMap { all, intra, inter }
}

fn mean_f64(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

fn selectivity_index(mean_intra: f64, mean_inter: f64) -> f64 {
    let denom = mean_inter + mean_intra;
    if denom.abs() < f64::EPSILON {
        0.0
    } else {
        (mean_inter - mean_intra) / denom
    }
}

/// Fixed-weight spiking reference (STDP off): the non-collapse baseline rate,
/// measured over the learning-phase slice.
fn run_fixed_reference(inputs: &[Vec<i16>]) -> f64 {
    let mut net = build();
    net.set_plasticity_enabled(false);
    for inp in &inputs[..INIT_STEPS.min(inputs.len())] {
        let _ = net.step(inp).expect("step");
    }
    let before = net.stats().total_spikes;
    for inp in &inputs[INIT_STEPS..] {
        let _ = net.step(inp).expect("step");
    }
    let learn_spikes = net.stats().total_spikes - before;
    let learn_steps = (inputs.len() - INIT_STEPS) as f64;
    let secs = learn_steps * f64::from(DT_US) / 1_000_000.0;
    learn_spikes as f64 / (secs * f64::from(NEURONS))
}

/// i16 free-drift learning run. Init cycle (STDP off) → learning phase (STDP on).
/// Returns final weights, learning-phase firing rate, total spikes, and the
/// per-step spike matrix (for the co-firing structure check).
fn run_i16(inputs: &[Vec<i16>]) -> (Vec<i16>, f64, u64, Vec<Vec<u8>>) {
    let mut net = build();
    net.set_plasticity_enabled(false);
    for inp in &inputs[..INIT_STEPS] {
        let _ = net.step(inp).expect("init step");
    }
    net.set_plasticity_enabled(true);
    let learn = &inputs[INIT_STEPS..];
    let mut spike_matrix = vec![vec![0u8; NEURONS as usize]; learn.len()];
    let before = net.stats().total_spikes;
    for (t, inp) in learn.iter().enumerate() {
        let spikes = net.step(inp).expect("step");
        for sp in &spikes {
            spike_matrix[t][sp.neuron_id as usize] = 1;
        }
    }
    let final_w: Vec<i16> = net.synapses().iter().map(|s| s.weight).collect();
    let learn_spikes = net.stats().total_spikes - before;
    let secs = learn.len() as f64 * f64::from(DT_US) / 1_000_000.0;
    let rate = learn_spikes as f64 / (secs * f64::from(NEURONS));
    (final_w, rate, learn_spikes, spike_matrix)
}

/// Ternary stochastic learning run. Init cycle (STDP off) → ternarize →
/// learning phase (STDP on + stochastic bucket-flips). Returns final weights,
/// γ, firing rate, total spikes, and bucket flips (all over the learning phase).
fn run_ternary(inputs: &[Vec<i16>]) -> (Vec<i16>, i16, f64, u64, u64) {
    let mut net = build();
    net.set_plasticity_enabled(false);
    for inp in &inputs[..INIT_STEPS] {
        let _ = net.step(inp).expect("init step");
    }
    let gamma = net.ternarize_weights();
    let mut prev: Vec<Trit> = net
        .synapses()
        .iter()
        .map(|s| Trit::from_weight(s.weight, gamma))
        .collect();
    net.set_plasticity_enabled(true);
    let learn = &inputs[INIT_STEPS..];
    let before = net.stats().total_spikes;
    let mut flips = 0u64;
    for inp in learn {
        let _ = net.step(inp).expect("step");
        net.stochastic_ternary_step(gamma);
        for (i, s) in net.synapses().iter().enumerate() {
            let cur = Trit::from_weight(s.weight, gamma);
            if cur != prev[i] {
                flips += 1;
                prev[i] = cur;
            }
        }
    }
    let final_w: Vec<i16> = net.synapses().iter().map(|s| s.weight).collect();
    let learn_spikes = net.stats().total_spikes - before;
    let secs = learn.len() as f64 * f64::from(DT_US) / 1_000_000.0;
    let rate = learn_spikes as f64 / (secs * f64::from(NEURONS));
    (final_w, gamma, rate, learn_spikes, flips)
}

/// Empirical co-firing rates over E→E synapses: (intra_mean, inter_mean).
fn cofire(spike_matrix: &[Vec<u8>], ee: &[(usize, u16, u16)]) -> (f64, f64) {
    let exc = exc_count();
    let steps = spike_matrix.len();
    let (mut isum, mut inum, mut esum, mut enum_) = (0.0_f64, 0u32, 0.0_f64, 0u32);
    for &(_idx, pre, post) in ee {
        let mut both = 0u32;
        for step in spike_matrix {
            if step[pre as usize] == 1 && step[post as usize] == 1 {
                both += 1;
            }
        }
        let rate = both as f64 / steps as f64;
        if group_of(pre, exc) == group_of(post, exc) {
            isum += rate;
            inum += 1;
        } else {
            esum += rate;
            enum_ += 1;
        }
    }
    (
        if inum > 0 { isum / inum as f64 } else { 0.0 },
        if enum_ > 0 { esum / enum_ as f64 } else { 0.0 },
    )
}

/// Drive selectivity: own-active vs own-idle firing rate (proves the drive
/// produces group-selective firing, not synchronous global firing).
fn drive_selectivity(spike_matrix: &[Vec<u8>], learn_log: &[u16], exc: u16) -> (f64, f64) {
    let (mut a_sp, mut a_st, mut i_sp, mut i_st) = (0u64, 0u64, 0u64, 0u64);
    for (t, mask) in learn_log.iter().enumerate() {
        let active_g = mask.trailing_zeros() as u16;
        for n in 0..exc {
            if group_of(n, exc) == active_g {
                a_st += 1;
                a_sp += u64::from(spike_matrix[t][n as usize]);
            } else {
                i_st += 1;
                i_sp += u64::from(spike_matrix[t][n as usize]);
            }
        }
    }
    (
        a_sp as f64 / a_st as f64 * 1000.0,
        i_sp as f64 / i_st as f64 * 1000.0,
    )
}

fn main() {
    let (inputs, active_log) = make_inputs();
    let ee = ee_map();
    let exc = exc_count();

    println!("=== Stage 1.5c: ternary STDP selectivity under structured (correlated) input ===");
    println!(
        "network : balanced E/I, {NEURONS} neurons ({exc} E / {} I), dt={DT_US}μs",
        NEURONS - exc
    );
    println!("input   : {GROUPS} groups over excitatory neurons; gapped rotating drive");
    println!(
        "          active {ACTIVE_ON} ms then {OFF_GAP} ms silent gap (cycle {cycle} ms), I_ACTIVE={I_ACTIVE}μA, I_IDLE={I_IDLE}μA, I_INH={I_INH}μA; {STEPS} steps",
        cycle = (ACTIVE_ON + OFF_GAP) * u32::from(GROUPS)
    );
    println!(
        "          init cycle {INIT_STEPS} steps (STDP off) + {} learning steps (STDP on)",
        STEPS - INIT_STEPS
    );
    println!(
        "E→E map : {} synapses — {} intra-group, {} inter-group",
        ee.all.len(),
        ee.intra.len(),
        ee.inter.len()
    );
    println!();

    let ref_rate = run_fixed_reference(&inputs);
    let (i16_w, i16_rate, i16_spikes, spikes) = run_i16(&inputs);

    // Rule-direction check (depression-only claim): count weights that moved up.
    let net0 = build();
    let init_w: Vec<i16> = net0.synapses().iter().map(|s| s.weight).collect();
    let (mut up, mut down, mut same) = (0u32, 0u32, 0u32);
    for (a, b) in init_w.iter().zip(i16_w.iter()) {
        if b > a {
            up += 1;
        } else if b < a {
            down += 1;
        } else {
            same += 1;
        }
    }

    let (co_intra, co_inter) = cofire(&spikes, &ee.all);
    let learn_log = &active_log[INIT_STEPS..];
    let (ra, ri) = drive_selectivity(&spikes, learn_log, exc);

    let i16_intra = mean_f64(
        &ee.intra
            .iter()
            .map(|&i| i16_w[i] as f64)
            .collect::<Vec<_>>(),
    );
    let i16_inter = mean_f64(
        &ee.inter
            .iter()
            .map(|&i| i16_w[i] as f64)
            .collect::<Vec<_>>(),
    );
    let i16_si = selectivity_index(i16_intra, i16_inter);

    println!("--- Input structure (empirical, from i16 run) ---");
    println!(
        "  co-fire rate       : intra={:.4}   inter={:.4}   ratio {:.1}×",
        co_intra,
        co_inter,
        if co_inter > 0.0 {
            co_intra / co_inter
        } else {
            f64::INFINITY
        }
    );
    println!(
        "  drive selectivity  : own-active Rₐ={:.1} Hz   own-idle Rᵢ={:.1} Hz   ratio {:.1}×",
        ra,
        ri,
        if ri > 0.0 { ra / ri } else { f64::INFINITY }
    );
    println!();
    println!(
        "--- Rule-direction check (i16, all {} synapses) ---",
        init_w.len()
    );
    println!("  weights up / down / unchanged : {up} / {down} / {same}  (depression-only if up≈0)");
    println!();
    println!("--- Essential CONTROL: i16 baseline selectivity (E→E) ---");
    println!(
        "  mean weight : intra={:.2}   inter={:.2}   (both started at 80)",
        i16_intra, i16_inter
    );
    println!("  Δ(inter−intra) : {:.2}", i16_inter - i16_intra);
    println!(
        "  Selectivity Index SI : {:.4}   (floor {CONTROL_SI_FLOOR})",
        i16_si
    );
    let control_ok = i16_si > CONTROL_SI_FLOOR && i16_inter > i16_intra;
    println!(
        "  [CONTROL: {}]",
        if control_ok {
            "PASS — input is learnable"
        } else {
            "FAIL — input lacks learnable structure"
        }
    );
    println!();

    let (tern_w, gamma, tern_rate, tern_spikes, flips) = run_ternary(&inputs);
    let tern_intra = mean_f64(
        &ee.intra
            .iter()
            .map(|&i| tern_w[i] as f64)
            .collect::<Vec<_>>(),
    );
    let tern_inter = mean_f64(
        &ee.inter
            .iter()
            .map(|&i| tern_w[i] as f64)
            .collect::<Vec<_>>(),
    );
    let tern_si = selectivity_index(tern_intra, tern_inter);
    let frac_plus = |idxs: &[usize]| -> f64 {
        if idxs.is_empty() {
            return 0.0;
        }
        idxs.iter().filter(|&&i| tern_w[i] >= gamma).count() as f64 / idxs.len() as f64
    };
    let tern_intra_plus = frac_plus(&ee.intra);
    let tern_inter_plus = frac_plus(&ee.inter);

    println!("--- Ternary stochastic selectivity (γ={gamma}) ---");
    println!(
        "  mean weight : intra={:.2}   inter={:.2}   (excitatory range {{0,+γ}}={{0,{gamma}}})",
        tern_intra, tern_inter
    );
    println!(
        "  frac at +γ  : intra={:.1}%   inter={:.1}%",
        tern_intra_plus * 100.0,
        tern_inter_plus * 100.0
    );
    println!("  Δ(inter−intra) : {:.2}", tern_inter - tern_intra);
    println!(
        "  Selectivity Index SI : {:.4}   (floor {TERNARY_SI_FLOOR})",
        tern_si
    );
    println!("  bucket flips : {flips}");
    println!();

    let tern_ratio = if ref_rate > 0.0 {
        tern_rate / ref_rate
    } else {
        0.0
    };
    let i16_ratio = if ref_rate > 0.0 {
        i16_rate / ref_rate
    } else {
        0.0
    };
    println!("--- Spiking sanity (non-collapse vs fixed-weight reference) ---");
    println!("  reference (fixed) : {:.2} Hz/neuron", ref_rate);
    println!(
        "  i16 learning      : {:.2} Hz/neuron  ({i16_spikes} spikes, {:.2}× ref)",
        i16_rate, i16_ratio
    );
    println!(
        "  ternary learning  : {:.2} Hz/neuron  ({tern_spikes} spikes, {:.2}× ref)",
        tern_rate, tern_ratio
    );
    println!();

    let tern_selective = tern_si > TERNARY_SI_FLOOR && tern_inter > tern_intra;
    let spiking_ok = tern_ratio > SPIKE_RATIO_FLOOR;
    println!("--- Verdict ---");
    println!(
        "  control (i16 discriminates) : {}",
        if control_ok { "YES" } else { "NO" }
    );
    println!(
        "  ternary selective           : {}",
        if tern_selective { "YES" } else { "NO" }
    );
    println!(
        "  spiking non-collapsed       : {}",
        if spiking_ok { "YES" } else { "NO" }
    );
    println!();

    let gate: String;
    if !control_ok {
        gate = "NO (INVALID) — i16 control failed to discriminate; input lacks learnable structure"
            .to_string();
    } else if tern_selective && spiking_ok {
        gate = format!(
            "YES — ternary discriminates by correlation (SI={:.3} vs i16 SI={:.3}) and stays non-collapsed",
            tern_si, i16_si
        );
    } else if control_ok && !tern_selective {
        gate = format!(
            "NO — ternary collapses uniformly (SI={:.3}) despite i16 discriminating (SI={:.3}); too coarse to learn selectively",
            tern_si, i16_si
        );
    } else {
        gate = format!(
            "MARGINAL — ternary SI={:.3} vs i16 SI={:.3}, spiking {:.2}× ref; partial/weak discrimination",
            tern_si, i16_si, tern_ratio
        );
    }
    println!("GATE: {gate}");
}
