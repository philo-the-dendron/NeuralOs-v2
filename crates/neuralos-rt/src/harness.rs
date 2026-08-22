#![allow(non_snake_case)]
//! The frozen-example experiment harness (R4(iii) extraction, 2026-08-20).
//!
//! (Uppercase locals below are deliberate: the moved bodies stay verbatim
//! against their original `const` names — the re-pin diff is the proof.)
//!
//! The six bridge examples (`hybrid_{gate,loop,invivo,sweep,sweep_cmv}`,
//! `null_patches`) grew ~1,864 duplicated lines under the freeze-source
//! doctrine. Per the autopsy amendment ("freeze evidence, never source"),
//! the shared plumbing now lives here and the examples carry only their
//! unique experiment logic. Every function is a VERBATIM move from
//! `hybrid_gate.rs` (canonical) or its siblings — the R4(iv) re-pin
//! against `evidence/r4-baselines/` proves bit-identity at default
//! parameters.
//!
//! Provenance discipline: every field of [`ExperimentParams`] carries its
//! value's origin (ISA session or "unrecorded" where the audits found no
//! decision entry — honest labels, values untouched: this rung
//! externalizes, it does not re-tune).

use neuralos_snn::{
    decode_q2_0, encode_q2_0, NetworkTopology, SpikingNeuralNetwork, SynapseType, Trit,
    VoltageResolution,
};

use crate::{GgufFile, GGML_TYPE_Q2_0};

/// The recorded experiment parameters of the hybrid family.
///
/// Defaults are the values every baseline in `evidence/r4-baselines/` ran
/// at. The sweep family varies `i_active` per rung (via
/// [`Self::amplitudes`]); the gate/loop/invivo family uses the 1.5c
/// constants verbatim.
#[derive(Debug, Clone)]
pub struct ExperimentParams {
    /// Slice side: 512 output rows × 512 input cols of `attn_q`.
    /// Provenance: session D-2 design (ISA 2026-08-17).
    pub n: usize,
    /// Host tensor name. Provenance: session D-2 (the q2_0 seam, D-1).
    pub tensor: &'static str,
    /// 4B config: attn_q input width (emb). Provenance: ISC-52 config pin.
    pub model_cols: usize,
    /// attn_q output width (32 heads × 128). Provenance: ISC-52.
    pub model_rows: usize,
    /// Substrate γ — the proven 1.5x regime constant. Provenance:
    /// recorded decision, sessions 1.5b/c/d and D-2 (the LLM's fp16 block
    /// scales are meaningless to SNN dynamics; γ=125 is what the gates
    /// ran at).
    pub gamma: i16,
    /// Integration step (μs). Provenance: 1.5c constants, verbatim.
    pub dt_us: u32,
    /// Neuron-type split, as in 1.5c. Provenance: 1.5c constants.
    pub excitatory_ratio: f64,
    /// Structured-drive group count. Provenance: 1.5c constants.
    pub groups: u16,
    /// Group active window (ms). Provenance: 1.5c constants.
    pub active_on: u32,
    /// Inter-group silent gap (ms) — defeats spurious boundary LTD.
    /// Provenance: 1.5c constants.
    pub off_gap: u32,
    /// Drive amplitude for the active group (μA). Provenance: 1.5c/D-2.
    pub i_active: i16,
    /// Idle excitatory input (μA). Provenance: 1.5c constants.
    pub i_idle: i16,
    /// Inhibitory background (μA) — "the validated wall". Provenance:
    /// 1.5c/D-2.
    pub i_inh: i16,
    /// Init-cycle steps (STDP off; defeats the last_spike=0 sentinel).
    /// Provenance: D-2 verbatim = (60+40)×4 = 400.
    pub init_steps: usize,
    /// Total schedule steps. Provenance: D-2 verbatim (400 init + 1600
    /// learn).
    pub steps: usize,
    /// Selectivity degree floor (intra |mean Δ| ≥). Provenance: session F
    /// criterion amendment — 1.5c's SI floor re-applied to the raw field.
    pub si_floor: f64,
    /// Firing ratio floor vs the census-matched control. Provenance: 1.5c
    /// SPIKE_RATIO_FLOOR.
    pub spike_ratio_floor: f64,
    /// Absolute floor (Hz/neuron) for "not degenerate". Provenance:
    /// recorded decision, session D-2.
    pub spike_abs_floor_hz: f64,
    /// Hamming bound: majority of pretrained buckets must survive.
    /// Provenance: D-2 mission language.
    pub hamming_bound: f64,
    /// Fisher-Yates seed for the census-matched control. Provenance:
    /// paper appendix "Seeds of record".
    pub control_seed: u64,
    /// Memory budget for a single-buffer run (MB). Provenance: D-2
    /// mission box.
    pub rss_budget_mb: u64,
    /// Amplitude ladder of the sweep family (μA). Provenance: session E
    /// stage 1 registration (grid 600→100).
    pub amplitudes: [i16; 9],
    /// In-vivo drive scaling target (μA RMS over driven dims).
    /// Provenance: ISA sH registration (session H).
    pub target_rms_ua: f64,
    /// In-vivo clamp rail (μA). Provenance: ISA sH registration.
    pub clamp_ua: i32,
    /// In-vivo clamp warning fraction (the recorded 69.8% caveat rides
    /// above it). Provenance: ISA sH.
    pub clamp_warn_frac: f64,
}

impl Default for ExperimentParams {
    fn default() -> Self {
        Self {
            n: 512,
            tensor: "blk.0.attn_q.weight",
            model_cols: 2560,
            model_rows: 4096,
            gamma: 125,
            dt_us: 1000,
            excitatory_ratio: 0.8,
            groups: 4,
            active_on: 60,
            off_gap: 40,
            i_active: 600,
            i_idle: 0,
            i_inh: 600,
            init_steps: 400,
            steps: 2000,
            si_floor: 0.05,
            spike_ratio_floor: 0.10,
            spike_abs_floor_hz: 0.10,
            hamming_bound: 0.50,
            control_seed: 0x5EED_C0DE_0000_0002,
            rss_budget_mb: 1536,
            amplitudes: [600, 450, 300, 240, 200, 170, 150, 125, 100],
            target_rms_ua: 450.0,
            clamp_ua: 1000,
            clamp_warn_frac: 0.10,
        }
    }
}

impl ExperimentParams {
    /// Row byte size: `(model_cols/128)·34` = 20 blocks × 34 B = 680.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        (self.model_cols / 128) * 34
    }
    /// Loop surgery geometry: the first 4 blocks (512 of 2560 input cols)
    /// of each row — `(n/128)·34` = 136 B.
    #[must_use]
    pub fn chunk_bytes(&self) -> usize {
        (self.n / 128) * 34
    }
    /// Whole-tensor byte size, computed from dims (never inferred from
    /// slice lengths — the parser infers ends from the NEXT tensor's
    /// offset).
    #[must_use]
    pub fn tensor_bytes(&self) -> usize {
        self.model_rows * self.row_bytes()
    }
}

/// Decode the `n×n` slice from a GGUF (original or patched) — the strict
/// D-2 path: type, dims, and byte-size asserts before any trit moves.
/// Verbatim from `hybrid_gate.rs` (canonical).
///
/// # Panics
///
/// Panics on container/type/dims/size mismatch (loud, never clamped).
pub fn decode_slice(path: &str, p: &ExperimentParams) -> Vec<Trit> {
    let N = p.n;
    let TENSOR = p.tensor;
    let buf = std::fs::read(path).unwrap_or_else(|e| {
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
        vec![p.model_cols as u64, p.model_rows as u64],
        "attn_q is [2560,4096] on the 4B"
    );
    let data = f.tensor_data(info).expect("tensor slice in bounds");
    assert_eq!(
        data.len(),
        p.tensor_bytes(),
        "tensor byte size = 4096 rows × 680 B"
    );
    let mut out = Vec::with_capacity(N * N);
    let mut row_trits = vec![Trit::Zero; N];
    let mut scales = vec![0u16; N / 128];
    for r in 0..N {
        decode_q2_0(
            &data[r * p.row_bytes()..r * p.row_bytes() + N / 128 * 34],
            &mut row_trits,
            &mut scales,
        )
        .expect("real q2_0 bytes must decode (code 3 would be loud)");
        out.extend_from_slice(&row_trits);
    }
    out
}

/// Absolute byte offset of the host tensor's data window inside a parsed
/// buffer (surgery anchor). Asserts type and dims first.
///
/// # Panics
///
/// Panics if the tensor is absent or not the expected q2_0 geometry.
pub fn tensor_abs(f: &GgufFile, p: &ExperimentParams) -> usize {
    let TENSOR = p.tensor;
    let info = f
        .tensors
        .iter()
        .find(|t| t.name == TENSOR)
        .unwrap_or_else(|| panic!("tensor {TENSOR} not found"));
    assert_eq!(info.ty, GGML_TYPE_Q2_0);
    assert_eq!(info.dims, vec![p.model_cols as u64, p.model_rows as u64]);
    (f.data_start + info.offset) as usize
}

/// Splice an n×n row-major trit patch over the surgery window of a
/// parsed GGUF buffer — the shared surgery core (R4-style
/// extraction; was inlined ×3 in hybrid_loop / hybrid_invivo /
/// null_patches).
///
/// Per output row: decode the original chunk (keeping the model's
/// own fp16 scale bits — magnitudes stay the model's, recorded
/// decision), re-encode the patch row against those scales, count
/// changed bytes split code-vs-scale by the q2_0 block layout
/// (bytes 0..2 of each 34-byte block are scales), splice into
/// `buf`. Scale-byte changes are asserted ZERO here.
///
/// `expect_src: Some(src)` additionally asserts every decoded
/// original row equals `src` (the loop's chunk==slice codec
/// transparency check); `None` skips it (invivo/null_patches never
/// asserted it).
///
/// Returns `(code_bytes_changed, scale_bytes_changed)`; the second
/// is always 0 (else the assert fired).
///
/// # Panics
///
/// GGUF/codec invariant failure, any scale-byte change, tensor
/// window outside the buffer, or (with `expect_src`) any original
/// chunk that differs from `src`.
pub fn splice_trits(
    buf: &mut [u8],
    patch: &[Trit],
    expect_src: Option<&[Trit]>,
    p: &ExperimentParams,
) -> (u64, u64) {
    let N = p.n;
    let ROW_BYTES = p.row_bytes();
    let CHUNK_BYTES = p.chunk_bytes();
    let f2 = GgufFile::parse(buf).expect("re-parse");
    let abs = tensor_abs(&f2, p);
    assert!(abs + p.tensor_bytes() <= buf.len(), "tensor window inside file");
    let mut row_orig = vec![Trit::Zero; N];
    let mut scales = vec![0u16; N / 128];
    let mut enc = vec![0u8; CHUNK_BYTES];
    let mut code_changed = 0u64;
    let mut scale_changed = 0u64;
    for r in 0..N {
        let off = abs + r * ROW_BYTES;
        decode_q2_0(&buf[off..off + CHUNK_BYTES], &mut row_orig, &mut scales)
            .expect("original chunk decodes");
        if let Some(src) = expect_src {
            assert_eq!(
                &row_orig[..],
                &src[r * N..(r + 1) * N],
                "row {r}: chunk == decoded slice"
            );
        }
        encode_q2_0(&patch[r * N..(r + 1) * N], &scales, &mut enc).expect("encode patch row");
        for (b, (&old, &new)) in buf[off..off + CHUNK_BYTES]
            .iter()
            .zip(enc.iter())
            .enumerate()
        {
            if old != new {
                if b % 34 < 2 {
                    scale_changed += 1;
                } else {
                    code_changed += 1;
                }
            }
        }
        buf[off..off + CHUNK_BYTES].copy_from_slice(&enc);
    }
    assert_eq!(scale_changed, 0, "scale bytes must pass through");
    (code_changed, scale_changed)
}

/// S2 disk round-trip: re-read a written file, parse it, and prove
/// every trit of the surgery window decodes back to `patch`.
///
/// # Panics
///
/// Unreadable file, GGUF parse failure, codec failure, or any
/// post-write decode mismatch.
pub fn verify_disk_roundtrip(out: &str, patch: &[Trit], p: &ExperimentParams) {
    let N = p.n;
    let ROW_BYTES = p.row_bytes();
    let CHUNK_BYTES = p.chunk_bytes();
    let check = std::fs::read(out).expect("re-read patched");
    let f3 = GgufFile::parse(&check).expect("patched file parses");
    let abs3 = tensor_abs(&f3, p);
    let mut rt = vec![Trit::Zero; N];
    let mut sc = vec![0u16; N / 128];
    let mut mism = 0u64;
    for r in 0..N {
        let off = abs3 + r * ROW_BYTES;
        decode_q2_0(&check[off..off + CHUNK_BYTES], &mut rt, &mut sc)
            .expect("patched chunk decodes post-write");
        for c in 0..N {
            if rt[c] != patch[r * N + c] {
                mism += 1;
            }
        }
    }
    assert_eq!(mism, 0, "S2: post-write decode != patch");
}

/// Read `path`, splice `patch` in, write `out`, S2-verify — the
/// one-call form for callers with no work between splice and write
/// (null_patches, hybrid_invivo's per-snapshot exports). Returns
/// `(code_bytes_changed, scale_bytes_changed)`.
///
/// # Panics
///
/// Everything [`splice_trits`] and [`verify_disk_roundtrip`] panic
/// on, plus unreadable base / unwritable out.
pub fn splice_and_verify(
    path: &str,
    out: &str,
    patch: &[Trit],
    expect_src: Option<&[Trit]>,
    p: &ExperimentParams,
) -> (u64, u64) {
    let mut buf = std::fs::read(path).expect("read base");
    let counts = splice_trits(&mut buf, patch, expect_src, p);
    std::fs::write(out, &buf).expect("write patched");
    verify_disk_roundtrip(out, patch, p);
    counts
}

/// Excitatory population size (truncating, matching `build_balanced`'s
/// partition — bit-identical at non-integer `n·ratio` boundaries).
#[must_use]
pub fn exc_count(p: &ExperimentParams) -> usize {
    ((p.n as f64) * p.excitatory_ratio) as usize // 409
}

/// Group membership of an excitatory neuron (4-group geometry).
#[must_use]
pub fn group_of(neuron_id: u16, exc: u16, p: &ExperimentParams) -> u16 {
    let g = (u32::from(neuron_id) * u32::from(p.groups) / u32::from(exc)) as u16;
    g.min(p.groups.saturating_sub(1))
}

/// Which group is driven at `step` (GROUPS = silent gap) — the 1.5c
/// schedule.
#[must_use]
fn active_group_at(step: usize, p: &ExperimentParams) -> u16 {
    let slot_len = p.active_on + p.off_gap;
    let cycle = slot_len * u32::from(p.groups);
    let within = (step as u32) % cycle;
    let slot = within / slot_len;
    if slot < u32::from(p.groups) && within % slot_len < p.active_on {
        slot as u16
    } else {
        p.groups
    }
}

/// The structured-drive input schedule (one `Vec<i16>` per step).
#[must_use]
fn make_inputs(i_active: i16, p: &ExperimentParams) -> Vec<Vec<i16>> {
    let (N, STEPS) = (p.n, p.steps);
    let exc = exc_count(p) as u16;
    let mut inputs = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let active = active_group_at(step, p);
        let mut inp = vec![p.i_inh; N];
        for n in 0..exc {
            inp[n as usize] = if group_of(n, exc, p) == active {
                i_active
            } else {
                p.i_idle
            };
        }
        inputs.push(inp);
    }
    inputs
}

/// Peak RSS (VmHWM) in MB from /proc/self/status — the memory-box
/// evidence.
#[must_use]
pub fn peak_rss_mb() -> u64 {
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

/// The experiment-family PRNG (seeded, deterministic).
#[must_use]
pub fn xorshift64(state: u64) -> u64 {
    let mut x = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Census-matched control: same trit multiset, Fisher-Yates-shuffled
/// placement (seed printed in evidence).
#[must_use]
pub fn shuffled_copy(src: &[Trit], seed: u64) -> Vec<Trit> {
    let mut v = src.to_vec();
    let mut rng = seed;
    for i in (1..v.len()).rev() {
        rng = xorshift64(rng);
        let j = (rng % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// Trit bucket as −1/0/+1 for means and deltas.
#[must_use]
pub fn trit_val(t: Trit) -> f64 {
    match t {
        Trit::MinusOne => -1.0,
        Trit::Zero => 0.0,
        Trit::One => 1.0,
    }
}

/// Trit bucket as census index: 0=−1, 1=0, 2=+1.
#[must_use]
pub fn tix(t: Trit) -> usize {
    match t {
        Trit::MinusOne => 0,
        Trit::Zero => 1,
        Trit::One => 2,
    }
}

/// Build the substrate network from a trit matrix (row-major N×N, weight
/// col→row: pre = column j, post = row i — matvec dataflow, recorded).
/// Full-minus-diagonal: 512×511 = 261,632 synapses, added pre-major
/// (sorted) then finalized through the public external-wiring path
/// (session D-2's substrate addition — the reverse CSR the LTP pass
/// needs). Verbatim from `hybrid_gate.rs`.
///
/// # Panics
///
/// Panics on construction/build/edge failure (all statically in-bounds at
/// the recorded geometry).
pub fn build_from_trits(
    trits: &[Trit],
    gamma: i16,
    p: &ExperimentParams,
    resolution: VoltageResolution,
) -> SpikingNeuralNetwork {
    let N = p.n;
    let mut net = SpikingNeuralNetwork::new_with_voltage_resolution(
        N as u16,
        p.dt_us,
        NetworkTopology::Random { connectivity: 0.0 },
        resolution,
    )
    .expect("512-neuron net must construct");
    net.build_topology()
        .expect("zero-connectivity build (empty wiring) must succeed");
    for j in 0..N {
        for i in 0..N {
            if i != j {
                let w = trits[i * N + j].to_weight(gamma);
                net.add_synapse(j as u16, i as u16, w)
                    .expect("in-bounds, non-self edge");
            }
        }
    }
    net.finalize_synapses();
    net
}

/// Fixed-weight run (STDP off). Reports over the full window, plus
/// per-quarter rates (the self-quench evidence) and per-group
/// containment. Verbatim from `hybrid_gate.rs`.
fn run_fixed(
    net: &mut SpikingNeuralNetwork,
    inputs: &[Vec<i16>],
    p: &ExperimentParams,
) -> FixedStats {
    let (N, STEPS) = (p.n, p.steps);
    let exc = exc_count(p) as u16;
    net.set_plasticity_enabled(false);
    let mut quarter_spikes = [0u64; 4];
    let mut a_sp = 0u64;
    let mut a_st = 0u64;
    let mut i_sp = 0u64;
    let mut i_st = 0u64;
    for (t, inp) in inputs.iter().enumerate() {
        let active = active_group_at(t, p);
        let spikes = net.step(inp).expect("step");
        quarter_spikes[t / (STEPS / 4)] += spikes.len() as u64;
        for sp in &spikes {
            if sp.neuron_id < exc {
                if group_of(sp.neuron_id, exc, p) == active {
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
        let active = active_group_at(t, p);
        for n in 0..exc {
            if group_of(n, exc, p) == active {
                a_st += 1;
            } else {
                i_st += 1;
            }
        }
    }
    let total: u64 = quarter_spikes.iter().sum();
    let secs_total = STEPS as f64 * f64::from(p.dt_us) / 1e6;
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

/// Fixed-weight run statistics (G2 / sweep evidence).
#[derive(Debug, Clone, Copy)]
pub struct FixedStats {
    /// Whole-window rate (Hz/neuron).
    pub rate_hz: f64,
    /// Per-quarter rates (the self-quench evidence).
    pub quarter_hz: [f64; 4],
    /// Containment: rate while the neuron's own group is driven.
    pub own_active_hz: f64,
    /// Containment: rate while idling.
    pub own_idle_hz: f64,
    /// Whole-window spike total.
    pub total_spikes: u64,
}

/// Hybrid (STDP-on) run statistics (G3 evidence + session F/G counters).
#[derive(Debug, Clone)]
pub struct HybridStats {
    pub learn_rate_hz: f64,
    pub quarter_hz: [f64; 4],
    pub own_active_hz: f64,
    pub own_idle_hz: f64,
    pub flips: u64,
    /// transitions[from][to], indices: 0=−1, 1=0, 2=+1.
    pub census: [[u64; 3]; 3],
    pub final_trits: Vec<Trit>,
    pub plasticity_events: u64,
    /// In-window STDP pairing histogram (session F): the Hebbian-
    /// attribution evidence — same_step (co-fire tie-break, LTD),
    /// post_leads (LTD), pre_leads (LTP).
    pub pairs_same_step: u64,
    pub pairs_post_leads: u64,
    pub pairs_pre_leads: u64,
    /// Per-class (E→E intra/inter) cumulative RAW STDP deltas and the
    /// clamp-absorbed remainders (session G): decides whether the realized
    /// bucket movement was pairing-driven or machinery-driven.
    pub raw_intra: i64,
    pub raw_inter: i64,
    pub absorbed_intra: i64,
    pub absorbed_inter: i64,
    pub n_intra: u64,
    pub n_inter: u64,
    pub cofire_intra: f64,
    pub cofire_inter: f64,
}

/// The G3 learning run: init cycle (STDP off) → learning (STDP on +
/// stochastic ternary bucket-flips at γ). Tracks the bucket-transition
/// census, firing sanity, and E→E co-firing structure over the learning
/// phase. Verbatim from `hybrid_gate.rs`.
fn run_hybrid(trits: &[Trit], inputs: &[Vec<i16>], p: &ExperimentParams) -> HybridStats {
    let (N, GAMMA, INIT_STEPS) = (p.n, p.gamma, p.init_steps);
    let exc = exc_count(p) as u16;
    let mut net = build_from_trits(trits, GAMMA, p, VoltageResolution::Millivolt);
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
    for (t, inp) in learn.iter().enumerate() {
        let active = active_group_at(INIT_STEPS + t, p);
        let spikes = net.step(inp).expect("learn step");
        net.stochastic_ternary_step(GAMMA);
        quarter_spikes[t / (learn.len() / 4)] += spikes.len() as u64;
        for sp in &spikes {
            if sp.neuron_id < exc {
                let n = sp.neuron_id as usize;
                fired[n][t / 64] |= 1u64 << (t % 64);
                if group_of(sp.neuron_id, exc, p) == active {
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
        let active = active_group_at(INIT_STEPS + t, p);
        for n in 0..exc {
            if group_of(n, exc, p) == active {
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
            if group_of(pre, exc, p) == group_of(post, exc, p) {
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
            if group_of(s.pre_neuron_id, exc, p) == group_of(s.post_neuron_id, exc, p) {
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
    let secs = learn.len() as f64 * f64::from(p.dt_us) / 1e6;
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

/// Spike train bitset (step-major, 8 words per 512-neuron step) + counts.
/// Verbatim from `hybrid_sweep.rs`.
pub struct Train {
    /// Step-major firing words (WORDS_PER_STEP per step).
    pub words: Vec<u64>,
    /// Per-neuron spike counts.
    pub counts: Vec<u64>,
    /// Total spikes.
    pub total: u64,
}

/// Fixed-weight run captured as a spike TRAIN (the sweep/invivo
/// instrument — totals cannot see timing divergence). Verbatim from
/// `hybrid_sweep.rs`.
pub fn run_and_capture(
    net: &mut SpikingNeuralNetwork,
    inputs: &[Vec<i16>],
    p: &ExperimentParams,
) -> Train {
    let N = p.n;
    net.set_plasticity_enabled(false);
    let words_per_step = N / 64; // 512 neurons → 8 words
    let mut words = vec![0u64; inputs.len() * words_per_step];
    let mut counts = vec![0u64; N];
    let mut total = 0u64;
    for (t, inp) in inputs.iter().enumerate() {
        let spikes = net.step(inp).expect("step");
        for sp in &spikes {
            let n = sp.neuron_id as usize;
            words[t * words_per_step + n / 64] |= 1u64 << (n % 64);
            counts[n] += 1;
            total += 1;
        }
    }
    Train { words, counts, total }
}

/// (step, neuron) events present in exactly one train.
#[must_use]
pub fn train_hamming(a: &Train, b: &Train) -> u64 {
    a.words
        .iter()
        .zip(&b.words)
        .map(|(x, y)| (x ^ y).count_ones() as u64)
        .sum()
}

/// Per-neuron rate-vector L1 (spike-count Manhattan distance).
#[must_use]
pub fn rate_l1(a: &Train, b: &Train) -> u64 {
    a.counts
        .iter()
        .zip(&b.counts)
        .map(|(x, y)| x.abs_diff(*y))
        .sum()
}

/// Everything the gate verdict (and `hybrid_loop`'s D-2 precondition
/// asserts + phase-2 surgery) needs from a gate-phase run.
pub struct GateOutcome {
    /// G1 import trit-exact.
    pub g1_pass: bool,
    /// G2 non-degenerate sustained.
    pub g2_pass: bool,
    /// Firing under STDP sustained.
    pub firing_ok: bool,
    /// Bucket flips > 0.
    pub not_frozen: bool,
    /// Hamming below bound.
    pub not_collapsed: bool,
    /// Intra |mean Δ| ≥ floor.
    pub selective: bool,
    /// All gates.
    pub pass: bool,
    /// The verdict string of record.
    pub verdict: &'static str,
    /// The three G2 comparators (kept alive: `hybrid_loop` drops them
    /// before its containment re-read).
    pub imported: SpikingNeuralNetwork,
    /// Census-matched control net.
    pub control: SpikingNeuralNetwork,
    /// Zero-weight comparator net.
    pub zeronet: SpikingNeuralNetwork,
    /// G2 imported-run stats.
    pub imported_stats: FixedStats,
    /// G2 control-run stats.
    pub control_stats: FixedStats,
    /// G2 zero-weight-run stats.
    pub zero_stats: FixedStats,
    /// The G3 learning run.
    pub hybrid: HybridStats,
    /// Changed-bucket count vs the imported original.
    pub hamming: u64,
    /// Changed-bucket fraction.
    pub hamming_frac: f64,
    /// Intra-class mean Δ (the gate-bearing degree field).
    pub d_intra: f64,
    /// Inter-class mean Δ (≡ 0 by schedule geometry).
    pub d_inter: f64,
}

/// The hybrid-gate phase: G1 import integrity → G2 spiking fidelity →
/// G3 selective adaptation → verdict. Verbatim from `hybrid_gate.rs`'s
/// main body (printing included); `hybrid_loop` runs the identical phase
/// with `phase1 = true` (two label strings differ; everything else is
/// byte-identical output).
///
/// `phase1` selects the loop's labels: verdict header reads
/// "--- Verdict (phase 1, D-2 gates) ---" and the verdict line reads
/// "HYBRID GATE (phase 1): …".
///
/// # Panics
///
/// Panics on any internal precondition (same asserts as the frozen
/// examples).
pub fn run_gate_phase(src: &[Trit], p: &ExperimentParams, phase1: bool) -> GateOutcome {
    let (N, GAMMA) = (p.n, p.gamma);
    let CONTROL_SEED = p.control_seed;
    let SPIKE_ABS_FLOOR_HZ = p.spike_abs_floor_hz;
    let SPIKE_RATIO_FLOOR = p.spike_ratio_floor;
    let HAMMING_BOUND = p.hamming_bound;
    let SI_FLOOR = p.si_floor;

    // ----- G1: census + round-trip on the imported substrate weights -----
    println!();
    println!("--- G1: IMPORT INTEGRITY (trit-preserving by construction) ---");
    let (mut plus, mut zero, mut minus) = (0u64, 0u64, 0u64);
    for t in src {
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

    let mut imported = build_from_trits(src, GAMMA, p, VoltageResolution::Millivolt);
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
    let imported_stats = run_fixed(&mut imported, &make_inputs(p.i_active, p), p);
    let ctrl_trits = shuffled_copy(src, CONTROL_SEED);
    let mut control = build_from_trits(&ctrl_trits, GAMMA, p, VoltageResolution::Millivolt);
    let control_stats = run_fixed(&mut control, &make_inputs(p.i_active, p), p);
    // Structure-free third comparator (added from the session's own
    // diagnostic: imported and control matched EXACTLY, and a zero-weight
    // net reproduces both bit-for-bit — the drive-dominated mechanism below).
    let zero_trits = vec![Trit::Zero; N * N];
    let mut zeronet = build_from_trits(&zero_trits, GAMMA, p, VoltageResolution::Millivolt);
    let zero_stats = run_fixed(&mut zeronet, &make_inputs(p.i_active, p), p);
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
    let h = run_hybrid(src, &make_inputs(p.i_active, p), p);
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
        "  STDP pairing histogram: same-step {} · post-leads {} · pre-leads {} (in-window; per-class raw/applied decomposition below decides the mechanism)",
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
    let exc = exc_count(p) as u16;
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
                if group_of(pre, exc, p) == group_of(post, exc, p) {
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
    if phase1 {
        println!("--- Verdict (phase 1, D-2 gates) ---");
    } else {
        println!("--- Verdict ---");
    }
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
    if phase1 {
        println!("HYBRID GATE (phase 1): {verdict}");
    } else {
        println!("HYBRID GATE: {verdict}");
    }
    let pass = g1_pass && g2_pass && firing_ok && not_frozen && not_collapsed && selective;
    GateOutcome {
        g1_pass,
        g2_pass,
        firing_ok,
        not_frozen,
        not_collapsed,
        selective,
        pass,
        verdict,
        imported,
        control,
        zeronet,
        imported_stats,
        control_stats,
        zero_stats,
        hybrid: h,
        hamming,
        hamming_frac,
        d_intra: din,
        d_inter: dit,
    }
}

/// The amplitude sweep (session E stage 1 / stage 1c instrument): decode
/// → ladder over [`ExperimentParams::amplitudes`] → verdict. The two
/// sweep examples are this function on the mV and centi-mV grids
/// respectively. Verbatim from `hybrid_sweep.rs`'s main body.
///
/// # Panics
///
/// Panics on decode failure (via [`decode_slice`]).
pub fn run_amplitude_sweep(
    path: &str,
    p: &ExperimentParams,
    resolution: VoltageResolution,
) {
    let t0 = std::time::Instant::now();
    let (N, GAMMA, STEPS) = (p.n, p.gamma, p.steps);
    let CONTROL_SEED = p.control_seed;
    let RSS_BUDGET_MB = p.rss_budget_mb;
    let AMPLITUDES = &p.amplitudes;

    // ----- Decode (D-2 path, verbatim) -----
    let src = decode_slice(path, p);
    let exc = exc_count(p);
    let secs = STEPS as f64 * f64::from(p.dt_us) / 1e6;

    println!("amp(μA) | E Hz (imp/ctl/zero)      | I Hz (imp/ctl/zero) | totals (i/c/z)   | H(i,c)  H(i,z)  H(c,z) | L1(i,c) L1(i,z) L1(c,z)");
    println!("--------+---------------------------+---------------------+------------------+-------------------------+------------------------");

    let mut a_star: Option<i16> = None;
    let mut first_divergence = String::new();
    for &amp in AMPLITUDES.iter() {
        let inputs = make_inputs(amp, p);
        let zero_trits = vec![Trit::Zero; N * N];
        let mut imported = build_from_trits(&src, GAMMA, p, resolution);
        let ctrl_trits = shuffled_copy(&src, CONTROL_SEED);
        let mut control = build_from_trits(&ctrl_trits, GAMMA, p, resolution);
        let mut zero = build_from_trits(&zero_trits, GAMMA, p, resolution);
        let ti = run_and_capture(&mut imported, &inputs, p);
        let tc = run_and_capture(&mut control, &inputs, p);
        let tz = run_and_capture(&mut zero, &inputs, p);
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
            match resolution {
                VoltageResolution::CentiMillivolt => println!("A* = {a} μA — on the centi grid the weight→firing channel OPENS at this amplitude."),
                VoltageResolution::Millivolt => println!("A* = {a} μA — the weight→firing channel OPENS at this amplitude."),
            }
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

/// The R7-named gap, closed: harness machinery had zero direct unit
/// tests (its pins were the example re-runs). These exercise the
/// shared surgery unit on a synthetic GGUF-shaped buffer — offline,
/// no model file needed.
#[cfg(test)]
mod tests {
    use super::*;

    /// Small surgery geometry: n=128 (chunk = one 34 B q2_0 block),
    /// model 256×128 — tensor window 128 rows × 68 B.
    fn small_params() -> ExperimentParams {
        ExperimentParams {
            n: 128,
            model_cols: 256,
            model_rows: 128,
            ..ExperimentParams::default()
        }
    }

    /// Minimal GGUF writer: header + one Q2_0 tensor, no KVs
    /// (default alignment 32). Spec type codes: UINT32=4.
    fn gguf_one_tensor(p: &ExperimentParams, data: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&0u64.to_le_bytes()); // kv count
        // tensor info: name, 2 dims, type, offset
        let name = p.tensor;
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&(p.model_cols as u64).to_le_bytes());
        b.extend_from_slice(&(p.model_rows as u64).to_le_bytes());
        b.extend_from_slice(&crate::GGML_TYPE_Q2_0.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // offset
        // pad to 32, then data
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend_from_slice(data);
        b
    }

    /// Original tensor data: every row's surgery chunk encodes 128
    /// `Zero` trits at scale fp16 1.0; the row tail beyond the chunk
    /// (cols 128..256) is a 0xA5 sentinel — the splice must never
    /// touch it.
    fn original_data(p: &ExperimentParams) -> Vec<u8> {
        let chunk = p.chunk_bytes();
        let row_bytes = p.row_bytes();
        let mut data = Vec::with_capacity(p.tensor_bytes());
        for _ in 0..p.n {
            let mut chunk_buf = vec![0u8; chunk];
            let zeros = vec![Trit::Zero; p.n];
            let scales = vec![0x3C00u16; p.n / 128]; // fp16 1.0
            encode_q2_0(&zeros, &scales, &mut chunk_buf).expect("encode zeros");
            data.extend_from_slice(&chunk_buf);
            data.extend(std::iter::repeat_n(0xA5u8, row_bytes - chunk));
        }
        // rows n..model_rows: entire rows of sentinel
        data.extend(std::iter::repeat_n(
            0xA5u8,
            (p.model_rows - p.n) * row_bytes,
        ));
        data
    }

    #[test]
    fn tix_maps_minus_zero_plus() {
        assert_eq!(
            (tix(Trit::MinusOne), tix(Trit::Zero), tix(Trit::One)),
            (0, 1, 2)
        );
    }

    #[test]
    fn splice_and_verify_round_trip_on_synthetic_gguf() {
        let p = small_params();
        let (n, chunk, row_bytes) = (p.n, p.chunk_bytes(), p.row_bytes());
        let mut buf = gguf_one_tensor(&p, &original_data(&p));
        let before = buf.clone();

        // src = what the file decodes to (all Zero, scales 1.0)
        let f = GgufFile::parse(&buf).expect("synthetic parses");
        let abs = tensor_abs(&f, &p);
        let mut src = vec![Trit::Zero; n * n];
        let mut scales = vec![0u16; n / 128];
        for r in 0..n {
            decode_q2_0(&buf[abs + r * row_bytes..][..chunk], &mut src[r * n..(r + 1) * n], &mut scales)
                .expect("decode row");
        }
        assert!(src.iter().all(|&t| t == Trit::Zero));
        assert!(scales.iter().all(|&s| s == 0x3C00));

        // patch: three flips in three different rows
        let mut patch = src.clone();
        patch[0] = Trit::One;
        patch[5 * n + 127] = Trit::MinusOne;
        patch[n * n - 1] = Trit::One;

        // control property first: splicing src over src changes nothing
        let (c0, s0) = splice_trits(&mut buf, &src, Some(&src), &p);
        assert_eq!((c0, s0), (0, 0), "identity splice must be byte-neutral");
        assert_eq!(buf, before, "identity splice left the buffer identical");

        // the real splice: code bytes change, scale bytes never
        let (code, scale) = splice_trits(&mut buf, &patch, Some(&src), &p);
        assert_eq!(scale, 0);
        assert!(code > 0, "three flips must change code bytes");
        // containment: only chunk-region bytes moved
        for (pos, (a, b)) in buf.iter().zip(before.iter()).enumerate() {
            if a != b {
                let rel = pos - abs;
                assert!(
                    rel / row_bytes < n && rel % row_bytes < chunk,
                    "byte changed outside a declared chunk at rel {rel}"
                );
            }
        }

        // write + S2 disk round-trip
        let dir = std::path::Path::new("/tmp/opencode");
        std::fs::create_dir_all(dir).expect("temp dir");
        let out = dir.join("harness_splice_test.gguf");
        std::fs::write(&out, &buf).expect("write");
        verify_disk_roundtrip(out.to_str().unwrap(), &patch, &p);
        std::fs::remove_file(&out).expect("cleanup");

        // the one-call form agrees on the same inputs
        let base_path = dir.join("harness_splice_base.gguf");
        std::fs::write(&base_path, &before).expect("write base");
        let (code2, scale2) =
            splice_and_verify(base_path.to_str().unwrap(), out.to_str().unwrap(), &patch, Some(&src), &p);
        assert_eq!((code2, scale2), (code, scale));
        std::fs::remove_file(&base_path).expect("cleanup base");
        std::fs::remove_file(&out).expect("cleanup out");
    }

    #[test]
    fn splice_rejects_src_mismatch_loudly() {
        let p = small_params();
        let buf = gguf_one_tensor(&p, &original_data(&p));
        let mut buf = buf;
        // expect_src that lies: claims the file held One trits
        let lie = vec![Trit::One; p.n * p.n];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            splice_trits(&mut buf, &lie, Some(&lie), &p);
        }));
        assert!(result.is_err(), "the chunk==slice assert must fire on a lying src");
    }
}
