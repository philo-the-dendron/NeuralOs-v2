# NeuralOS v2 — Vision

> The open `no_std` spiking-neural substrate for RISC-V edge silicon —
> and the lab bench to watch it think.

*Status: living document. Updated 2026-08-08. Grounded in the 2026-08-05/06
landscape + research findings (`docs/landscape/SUMMARY.md`,
`docs/RESEARCH_FINDINGS.md`).*

---

## What NeuralOS v2 is

A sovereignty stack in pure Rust, organized around one defensible artifact: a
`no_std`, i16 fixed-point spiking-neural-network library (`neuralos-snn`)
designed to run on RISC-V edge silicon (ESP32-C3, QEMU, HiFive). Around that
spine: a visualizer that makes the substrate legible, and a forward research
direction — the ternary bridge — that connects spiking nets to the 1-bit-LLM
wave.

This is **not** an "AI OS" (the v0.1 pretense, archived). It is a library with
a microscope on top, sitting in a documented vacuum at the coordinates the
hardware frontier needs.

## Why it's defensible (the evidence)

- **Intel archived Lava on 2026-05-13.** The reference i16 fixed-point SNN
  framework is gone. There is a vacuum where a flagship used to be — and
  NeuralOS can position as its spiritual successor.
- **The SNN-on-RISC-V frontier is real and active, but software-starved.**
  SpikeStream (PULP/Bologna-ETH, 4.4× speedup), FeNN, SNAP-V ("bridges
  large-scale neuromorphic hardware and practical small-scale embedded
  systems"), Kraken SoC, Wenquying-22, IzhiRISC-V — a 2024–2026 burst of
  hardware work with no mature open software library to run on it.
- **The `no_std`/embedded SNN slot is vacant across all languages.** 27 SNN
  frameworks listed in the open-neuromorphic guide; every one Python/C++/JAX
  except one low-quality Rust entry. Nobody owns this.
- **The i16 fixed-point design is externally validated.** IEEE 2025,
  *"Full-Integer SNN Inference with RISC-V ISA"* — its thesis is that
  float-free integer SNN is *required* for FPU-less edge silicon. NeuralOS
  built that by instinct; the field is now publishing papers confirming it.

## The three chapters

### 1. The substrate (now)

`neuralos-snn` — published on crates.io, `no_std`, i16 fixed-point, LIF + STDP
+ 4 topologies + CSR sparse matrix + AVX2 kernel. The spine and the crown
jewel. Every line advances it.

**Near-term:** NIR import/export (instant interop with the snnTorch /
SpikingJelly communities — the #1 ecosystem recommendation), port the
lock-free + SIMD primitives from the v0.1 archive, land the ternary weight
type (on-ramp to the bridge).

### 2. The forward chapter — the ternary bridge (research, honestly labeled)

The brain (SNN) and BitNet b1.58 (1-bit LLM) independently arrived at the same
weight alphabet: `{-1, 0, +1}`. The two literatures don't cite each other and
share no code. The bridge is a unified runtime where a ternary SNN layer and a
ternary LLM layer share the same weights, the same format, and the same
add / subtract / skip compute primitive. NeuralOS already computes in that
regime — i16 fixed-point is exactly ternary-compatible.

**Gated, de-risked path** — each stage ships standalone value and clears a
yes/no feasibility gate before the next begins. If a gate fails, you stop with
real artifacts already shipped, not a dead-end:

| Stage | What | Standalone value (if you stop here) | The gate |
|---|---|---|---|
| **1. Ternary SNN** | `Trit` weight type `{-1,0,+1}` + scale | a more efficient + more biologically-plausible SNN variant | does it still spike + learn (STDP) comparably to i16? **✓ Spiking: YES (1.00× baseline). Learning: Stage 1 deterministic NO → Stage 1.5b stochastic YES (bridge reopened) → Stage 1.5c structured selectivity YES → Stage 1.5d full pairwise STDP YES (missing LTP half added + CSR sync bug fixed; selectivity re-confirmed at SI 1.000 vs i16 1.000 under structured input, rule now bidirectional). Stage 2 is now firmly earned on the corrected substrate (see below).** |
| **2. Format bridge** | ternary format spec; BitNet-compatible **export**, Prism `Q1_0` **import** | NeuralOS speaks the lingua franca of both fields | can we round-trip a ternary tensor? **✓ YES (2026-08-15, see below): `i2_s` round-trip bit-exact, `q1_0`/`q2_0` import exact, `docs/TERNARY_FORMAT.md` is the spec.** |
| **3. Shared kernel** | one `no_std` ternary matmul; a tiny hybrid net (SNN layer + dense-LLM-style layer) | a reusable Rust ternary kernel + a showable hybrid demo | does the union compose — compute something coherent? **✓ YES (2026-08-15, see below): 4/4 classification through one kernel, weights arriving as `i2_s` wire bytes.** |
| **4. Full Rust ternary-LLM** | extend/replace candle's quantized kernels to run a Bonsai `Q1_0` model in pure Rust | the Rust answer to `bitnet.cpp` — sovereignty-grade local AI | gated on Stage 3's proof; multi-session research **— CLOSED 2026-08-16, gate verdict: **NO (3/5 strict prompts; see session 4 below) — tokenizer + incremental decode + deterministic generation all work end-to-end; the 1-bit 1.7B failed two word-sequence continuations. Bridge stops with shipped artifacts per gate doctrine; merge/unmerge call is the principal's** |

**Format decision — deferred.** Stage 1 uses plain `{-1,0,+1}` + scale (zero
wire-format lock-in while we don't yet know the ternary regime works for our
SNN). If the gate passes, Stage 2 commits to **BitNet `Round()` native +
Prism `Q1_0` import** — because BitNet gives a *mechanical* bridge (real
weights flow between SNN and LLM), whereas standard TWN would be conceptual
only (shared alphabet, incompatible encoding, isolated from the models
actually shipping). **Stage 2 honored this commitment (see its section
below).**

### Stage 2 — RUN 2026-08-15, result: YES (the format bridge)

The gate question: *can we round-trip a ternary tensor?* Answered by
`examples/ternary_format_gate.rs`: a 256-trit tensor encodes to BitNet
`i2_s` and decodes back **bit-exact** (trits and f32 scale bits), a
two-block Prism `q1_0` stream imports with exact sign bits and per-block
scales, and a Prism `q2_0` block imports with exact 2-bit codes — with
impossible code-3 input rejected loudly, never clamped.

**Layouts pinned verbatim from reference source code** (fetched the same
session, not from blogs or model cards):

- **BitNet `i2_s`** (`microsoft/BitNet`
  `utils/convert-hf-to-gguf-bitnet.py::quantize_to_i2_s`): 2-bit codes
  `{0,1,2}` in a **transposed** 4-lane packing — element `i` at byte
  `(i/128)·32 + (i%32)`, shift `6 − 2·((i%128)/32)` — plus a 32-byte tail
  with the LE f32 scale. Scale semantics: BitNet-Round `γ = mean|w|`, the
  same convention as our `trit::tensor_scale`. Export **and** import.
- **Prism `q1_0`** (`PrismML-Eng/llama.cpp` `block_q1_0`): per 128 weights,
  LE fp16 scale (`γ = mean|w|` — the fork uses BitNet's convention) + 16
  sign bytes, LSB-first. Binary `{−γ, +γ}`, no zero state — imports
  losslessly into ternary; export does not exist (would silently map zeros
  to `+γ`).
- **Prism `q2_0`** (same fork, `block_q2_0`): per 128 weights, LE fp16 scale
  (`max|w|`, TWN-style) + 32 bytes of LSB-first 2-bit lanes (34 B/block —
  the session-D re-pin; see the session D slice 1 section for how the
  first real file caught our original 18 B/64 w spec being wrong). Code
  `11` decodes to `+2·d` in the reference dequantizer but cannot be
  emitted by its quantizer; we reject it loudly.

**Honest findings:**

1. **The "Q2_0_g128" label in our research docs was RIGHT by accident —
   and our "correction" of it was wrong.** Stage 2 pinned q2_0 as
   group 64 / 18 B, claiming the fork's C source defined `QK2_0 = 64`.
   Session D (2026-08-17) proved the source says **128** — 34 B per
   block — when the first real q2_0 file failed every tensor against
   the old arithmetic. The pin was re-derived from the fork's
   `ggml-common.h` + the file's own bytes; self-consistent hand test
   vectors had kept the misreading invisible (recorded as an ISA
   Learning — a codec with no real artifact to eat stays
   unverified-by-artifact).
2. **`i2_s` requires `n % 128 == 0`, not merely `% 4`.** The reference
   truncates output at `n/4` bytes; with the transposed packing, any
   non-multiple of 128 leaves live elements in truncated bytes — silently
   dropped. A permissive codec would be silently lossy; ours refuses
   (`BadLength`).
3. **The scale conventions differ between formats** (`mean|w|` for `i2_s`
   and `q1_0`, `max|w|` for `q2_0`) — the spec names which one each layout
   carries; treating them as interchangeable would corrupt magnitudes.

**What ships:** `neuralos_snn::bridge` — `encode_i2_s`/`decode_i2_s`
(bit-exact inverses), `decode_q1_0`/`decode_q2_0` (import-only, loud
errors), `half_to_f32_bits` (integer fp16→f32 widening, property-tested
against float math on all 65 536 halves) and `half_to_milli` (fixed-point
scale view) — all buffer-based, zero-alloc, integer-only, `no_std`.
`docs/TERNARY_FORMAT.md` is the spec, with worked byte examples byte-equal
to the crate's test vectors. Thirteen new tests (unit + property +
known-vector), gate example, and the discipline gates green.

**Deferred (fog, kept honest):** how an imported fp16 γ maps into the i16
`SCALE=1000` weight domain when feeding `SpikingNeuralNetwork` — a Stage 3
concern (the milli view exists; the policy doesn't). NativeTernary stays an
unimplemented import path until something real emits it.

### Stage 1 — RUN 2026-08-08, result: NO on learning (deterministic regime)

The minimal gate ran (`crates/neuralos-snn/examples/ternary_gate.rs`): balanced
128-neuron net, BitNet-Round γ = mean|w| = 125, per-step re-projection onto
`{-γ, 0, +γ}`, STDP on, 300 ms sim per phase. The result split clean:

- **Spiking: YES.** Ternary fires **1.00× the i16 baseline** (86.15 Hz/neuron,
  identical 3308 spikes). The ±γ weights propagate ~12.5 μA/spike, comparable
  to the baseline {8, 15, 20, 12}; the ternary representation does **not**
  collapse dynamics.
- **Learning: NO — frozen.** Over 28 199 plasticity events, the ternary net
  produced **0 bucket flips** (final state frozen at {−γ=258, 0=0, +γ=811}).
  The i16 baseline, under identical conditions, drifts all 1069 weights with
  mean |Δw| = 92.

**Diagnosis — precise, not fundamental.** STDP deltas top out at ±5
(a₊=50, lr=100, SCALE=1000), while the ternary bucket boundary sits at
γ/2 ≈ 62 — a ~12× gap. No single-step delta can cross the boundary, so
per-step re-projection snaps every weight back to its starting bucket. The
ternary *representation* is sound; the *learning rule × quantizer* pair is
the bottleneck.

**Per the gate rule, the bridge paused here under the deterministic regime.**
Stage 2 (format bridge) was not earned until Stage 1.5b (below) reopened
learning. The `Trit` type + ternarizer ship anyway — they are real, tested
artifacts. `network::tests::ternary_gate_stage1_deterministic_is_frozen`
pins the negative result as a canary: deterministic per-step re-projection
is a ruled-out baseline, kept in history.

**Literature check (2026-08-08).** The NO above is consistent with the field,
not anomalous. The literature is unanimous that ternary/binary networks do
*not* learn under deterministic per-step re-projection of small gradients —
that is a known-dead regime. Real ternary/binary training (TWN, Li 2016
`1605.04711`; the BitNet lineage `2504.12285` et al.) keeps latent
full-precision weights and re-quantizes on a schedule, never per-step. So
Stage 1 rules out *one regime* (the strictest possible), not ternary SNN
learning itself. The original NO stands as a ruled-out baseline — do not
re-test deterministic per-step re-projection.

**Stage 1.5 reopen paths — literature-backed.** Four mechanisms exist; two
fit our constraints (`no_std`, local STDP, i16, online — no global backprop):

| Mechanism | Fits local STDP? | Status / Precedent (verified 2026-08-08) |
|---|---|---|
| **1.5a — Latent i16 accumulation + periodic re-quantize** (STDP deltas build in i16; re-quantize every N steps, not every step) | yes | NOT started. TWN `1605.04711`, BitNet `2504.12285`, Rathi-Panda-Roy `1710.04734`. Safe fallback if 1.5b's YES doesn't hold under harder tests. |
| **1.5b — Stochastic bucket-flips** (each STDP event does a Bernoulli draw to flip the ternary bucket, rate ∝ STDP signal; bypasses the boundary problem; uses our existing LFSR, no shadow state) | yes — hardware-native | **✓ DONE — YES (see below).** Wu-Saxena `1801.02797`, Mohan `2103.01271`, Camuñas-Mesa `2209.06068`, `ReStoCNet` `1902.04161` |
| Surrogate gradient / STE | no — needs global backprop | N/A. Eshraghian `2202.07221` |
| Multi-step quantized STDP rule | needs rule redesign | N/A. Liu `2306.07712` |

### Stage 1.5b — RUN 2026-08-08, result: YES (bridge reopened)

The stochastic bucket-flip rule ran in the extended gate
(`crates/neuralos-snn/examples/ternary_gate.rs`): same balanced 128-neuron
net, γ = 125, STDP on, 300 ms. Three regimes compared:

- **(i) i16 baseline** — free drift: all 1069 weights move, mean |Δw| = 92.
- **(ii) Ternary + deterministic re-projection (Stage 1)** — **0 flips**
  (frozen, the ruled-out baseline).
- **(iii) Ternary + stochastic flips (1.5b)** — **802 bucket flips** over
  28 199 plasticity events (0.028 flips/event). Final distribution shifted
  from {−γ=258, 0=0, +γ=811} to {−γ=258, 0=802, +γ=9} — the synchronous
  drive produces predominantly LTD, weakening excitatory synapses (+γ → 0)
  while inhibitory (−γ) saturates at its min. A biologically plausible
  desynchronizing signal.

**Spiking under stochastic learning: 1.00× baseline** (86.15 Hz/neuron,
3308 spikes) — non-collapsed. The external drive (600 μA) sustains firing
even with ~75% of excitatory synapses weakened to 0.

**Gate verdict: YES.** Nontrivial bucket movement (802 vs 0) AND spiking
non-collapsed (1.00×). The bridge is reopened. Stage 2 becomes
**conditionally motivated**, pending the structured-input selectivity test and
any substrate-level correctness issues discovered downstream.

**What ships:** `trit::stochastic_ternary_flip` (pure `no_std` Bernoulli
flip, LFSR-driven, integer-only), `SpikingNeuralNetwork::stochastic_ternary_step`
(the per-step plasticity path), property tests (P-range, sign-correctness,
ternary-output invariant), and `ternary_gate_stage1_5b_stochastic_unfreezes_learning`
(the new canary — asserts nonzero flips; replaces Stage 1's frozen
canary as the bridge's living tripwire).

**Caveat — honest scope.** The 802 flips are predominantly LTD-driven
(synchronous drive → all neurons fire together → post-before-pre → LTD).
This proves the stochastic mechanism *works* (nonzero movement, non-collapsed)
but not yet that it *learns something useful* (e.g., discriminates patterns).
That is Stage 1.5c's concern, answered below. The 1.5a latent-accumulation
fallback remains unstarted as insurance.

### Stage 1.5c — RUN 2026-08-08, result: YES (ternary discriminates by correlation)

The test 1.5b couldn't address: does ternary STDP *discriminate* under structured
input, or always collapse regardless of structure? Ran in
`examples/ternary_selectivity.rs`. Excitatory neurons split into 4 groups; one
group driven at a time (sustained 600 μA) on a gapped round-robin schedule
(60 ms active, 40 ms silent gap — the gap defeats spurious boundary LTD between
adjacent groups). Intra-group pairs co-fire (correlated); inter-group pairs never
do (dt ≫ STDP window). An init cycle (STDP off) defeats the
`last_spike_time_us = 0` ("never fired") artifact. The i16 baseline is the
essential control.

- **i16 control: PASS (SI = 1.000).** Intra-group E→E mean → 0.00 (co-fire
  LTD), inter-group mean → 80.00 (unchanged). All 373 inter synapses stayed at
  their initial value; the input's structure is genuinely learnable.
- **Ternary stochastic: YES (SI = 0.985).** Intra mean → 0.95 (99.2% of
  synapses flipped +γ→0), inter mean → 125.00 (100% at +γ), 431 bucket flips,
  spiking 34.36 Hz/neuron (non-collapsed, 1.00× fixed-weight reference).

**Gate verdict: YES.** Ternary reproduces the i16 differential almost exactly
under structured input — it learns selectively, not just moves. The slight gap
(0.985 vs 1.000) is the stochastic flip mechanism's inherent noise (0.8% of
intra synapses retained +γ). This established structured ternary selectivity,
but the later Stage 1.5d rerun was still required after discovering that the
substrate was structurally LTD-only and missing the post-firing LTP half.
Stage 2 (format bridge) is strongly motivated.

**Two honest findings about the rule (documented, not worked around):**

1. **The STDP rule was depression-only at the time of 1.5c — RESOLVED in 1.5d.**
   `update_plasticity` computed `dt = pre_time − post_time`, always ≥ +1 (co-fires
   tie-break to `dt = +1` → LTD; non-co-fires use post's `last_spike ≤ pre_time` →
   `dt ≥ 0` → LTD). LTP (`dt < 0`) never fired — verified empirically (0 of 1069
   weights increased). So 1.5c selectivity was *differential depression*
   (correlated pairs depress more), not Hebbian potentiation. A valid selectivity
   test, and the diagnosis triggered the 1.5d fix: the missing post-firing LTP
   half is now implemented (see Stage 1.5d below), and the selectivity YES
   survives — in fact strengthens to SI 1.000 — on the full biphasic rule.

2. **CSR unsorted-insertion bug (fixed).** Designing this experiment exposed a
   latent defect: `SparseSynapseMatrix`'s incremental `row_ptrs` only grouped
   edges correctly for *sorted* insertion, but `build_balanced` inserts in
   arbitrary presynaptic order — so `connections(pre)` returned slices with the
   right *count* but the wrong *members*, corrupting propagation targeting and
   STDP synapse selection. `finalize()` is now a real counting-sort CSR build and
   `build_topology` calls it; two regression tests pin the invariant. (1.5b's
   qualitative YES — nonzero flips vs Stage 1's 0 — is unaffected: the mechanism
   still moves weights; the bug scrambled *which* synapses, not *whether* they
   moved. Both canaries still pass.)

**What ships:** `examples/ternary_selectivity.rs` (the diagnostic), the CSR fix
+ regression tests, and `network::tests::ternary_gate_stage1_5c_selectivity_under_structured_input`
(the new canary — asserts both the i16 control discriminates AND ternary
preserves the differential with nonzero flips; replaces nothing, joins the
1.5b canary as the bridge's living selectivity tripwire).

### Stage 1.5d — RUN 2026-08-08, result: YES (substrate corrected; selectivity re-confirmed)

1.5c's YES came with a precise caveat: the STDP orchestration was *structurally
LTD-only* (the pre-firing path set `dt ≥ 0` always, so the LTP branch of
`calculate_weight_change` was unreachable). That made 1.5c's selectivity a
differential-depression result, not a Hebbian one — a valid falsifier, but on a
half-rule. Before Stage 2, the missing LTP half was added and the gate re-run.

**Two substrate fixes:**

1. **Full pairwise STDP (the missing LTP half).** `update_plasticity` now runs
   two disjoint passes per step. The existing **LTD pass** (pre-firing events,
   post-before-pre, `dt ≥ 0`) is unchanged — including the a9a2679 same-step
   tie-break to `dt = +1`. A new **LTP pass** iterates each firing postsynaptic
   neuron's *incoming* synapses (via a new reverse CSR,
   `SparseSynapseMatrix::incoming`) and pairs the post spike with a presynaptic
   partner that fired earlier in a previous step (`dt = pre.last_spike − post_time < 0` → LTP). The passes are disjoint per synapse per step: a same-step
   co-fire is handled by LTD and explicitly skipped by LTP (`pre didn't fire this
   step` guard), so the a9a2679 invariant holds and there is no double-counting.

2. **`set_weight` inverse-permutation fix (a latent bd5b098 bug, separate from
   the CSR-insertion bug 1.5c already fixed).** The counting-sort `finalize()`
   reorders the `weights[]` array by `pre_id`, but `set_weight(syn_idx)` indexed
   `weights[syn_idx]` directly — so after `finalize`, plasticity deltas were
   written to the *wrong* CSR slots, desynchronizing the transmission weights
   from `synapses[].weight` (316 of ~1069 slots mismatched after 50 steps, empirically). The fix routes `set_weight` through an inverse permutation
   (`weight_index_of`) built in `finalize`, keeping the write O(1) and correct.
   1.5c's qualitative YES survived this because the external drive dominated and
   the ternary regime reads `synapses[].weight` (correct), not the CSR, for its
   flip decisions — but propagation ran on corrupted weights, so the fix is
   necessary for honest dynamics and for LTP deltas to actually reach neurons.

**Re-run on the corrected substrate** (`examples/ternary_selectivity.rs`):

- **Bidirectionality confirmed.** Under sustained drive, the i16 rule now moves
  weights in both directions: **306 up / 390 down / 373 unchanged** (through
  1.5c `up` was 0 — depression-only). The structural diagnosis is resolved.
- **i16 control: PASS (SI = 1.000).** Intra E→E mean → 0.00, inter → 80.00 —
  identical to 1.5c. The input's structure is genuinely learnable.
- **Ternary stochastic: YES (SI = 1.000, up from 1.5c's 0.985).** Intra → 0.00
  (100% of synapses flipped +γ→0), inter → 125.00 (100% at +γ), 2059 bucket
  flips, spiking 34.36 Hz/neuron (1.00× fixed-weight reference, non-collapsed).
  The 0.8% stochastic-noise gap of 1.5c is gone.

**Gate verdict: YES — strengthened.** Full pairwise STDP works (LTP and LTD both
reachable in orchestration, verified by tests), the i16 control discriminates,
and ternary-stochastic still discriminates with sane spiking — now at parity
with the i16 SI rather than 0.015 below it. The 1.5c selectivity YES was **not**
a half-STDP artifact; it survives the full biphasic rule. The 1.5b "degenerate
LTD-driven collapse" caveat is also resolved: under uniform drive the stochastic
distribution is now biphasic ({+γ:271, 0:613, −γ:185} with 3587 flips), not a
one-directional +γ→0 collapse.

**Honest scope note.** Under *this* synchronous-drive regime, the intra-group
selectivity signal is still carried by the LTD branch (correlated pairs co-fire →
same-step tie-break → depression, an anti-Hebbian decorrelation), not by Hebbian
LTP: the same-step tie-break biases coincidences to LTD, and the gapped schedule
places inter-group pairs outside the STDP window in *both* directions. LTP is
active across the network (306 of 1069 weights increase) but exercises mainly the
non-selective classes (inhibitory, E→I) and within-group temporal sequences. A
regime that drives clean pre-before-post *sequences* rather than synchronous
co-firing would exercise LTP-driven (Hebbian) selectivity more directly — that is
a different experiment, out of scope for this substrate-correction session.

**What ships:** the post-firing LTP pass + reverse CSR (`SparseSynapseMatrix::incoming` / `IncomingIter`), the `weight_index_of` inverse-permutation fix,
and six regression tests pinning both halves — `ltp_post_firing_strengthens_synapse_when_pre_fired_earlier` (focused LTP proof), `ltp_pass_does_not_double_count_same_step_cofire` (a9a2679 preserved under full STDP), `ltd_pre_after_post_still_depresses_under_full_stdp` (LTD half intact), `full_stdp_is_bidirectional_in_orchestration` (orchestration-level up>0 ∧ down>0), `csr_weight_index_of_keeps_slots_synced_in_multi_synapse_net` (the inverse-perm fix + reverse-CSR consistency), and `reverse_csr_incoming_lists_correct_edges_after_finalize`. The 1.5b and 1.5c canaries both still pass unchanged.

**The destination — a pure-Rust ternary-LLM runtime.** Stages 3 and 4 are
where the bridge earns its ambition. Stage 3 proves the union mechanically: a
shared `no_std` ternary matmul primitive that an SNN layer and a dense
LLM-style layer both call, composed in one tiny hybrid net. **Stage 4 is the
artifact worth desiring** — a `no_std`-friendly Rust runtime that loads and
runs Bonsai-style `Q1_0` ternary models the way `bitnet.cpp` does in C++:
sovereignty-grade local AI with no black-box runtime, sharing the SNN's
arithmetic natively. If the ternary SNN + the format bridge prove out, this is
the build that turns NeuralOS from a library into a stack — a Rust answer to
1-bit inference that runs on hardware you own. It is the multi-session
research bet, explicitly gated on Stage 3's proof that the union composes. We
don't start it until the gate earns it; we don't pretend it's smaller than it
is.

### Stage 3 — RUN 2026-08-15, result: YES (the shared kernel composes)

The gate question: *does the union compose — compute something coherent?*
Answered by `examples/ternary_hybrid_gate.rs`: a ternary SNN layer and a
dense LLM-style layer, both computing through **one** `no_std` ternary
matmul kernel, classify which of 4 neuron groups was driven — **4/4
correct** (chance 25%), margins ~10⁶, driven-group spiking 4–5
spikes/neuron vs 0.00 elsewhere (the 1.5c tonic-inhibition containment
holds with fixed ternary weights, STDP off).

**The composition pipeline (the claim itself):**

1. **SNN layer** — balanced 128-neuron net, `ternarize_weights` at γ=125,
   plasticity off: a pure transducer (drive → spike counts).
2. **Activation quant** — per-trial spike counts → Q15 i16 via integer
   per-vector absmax (`kernel::absmax_normalize_q15`) — the integer analog
   of BitNet's per-token activation normalization.
3. **Dense layer enters as wire bytes** — its 4×128 ternary weights are
   `encode_i2_s`-ed (Stage 2's BitNet wire format) and reach the kernel
   only through `bridge::repack_i2s_to_kernel` (wire transposed → compute
   sequential, element-wise bit surgery, zero intermediate buffer). The
   classifier never touches the kernel through a shortcut.
4. **One kernel** — `kernel::ternary_matvec`: sequential 2-bit packed
   trits × Q15 activations → i32, `|acc| ≤ n·32767` documented,
   property-tested against an unpacked scalar reference.

**Two formats, two roles (the Stage-2 fog items, resolved):**

- **Wire format** = BitNet `i2_s` — what crosses systems (Stage 2's codec).
- **Compute format** = sequential 2-bit packing — what the hot loop wants.
  `repack_i2s_to_kernel` is the seam between them.
- The imported-γ policy: `bridge::wire_gamma_to_substrate(milli)` maps a
  wire fp16 γ into the i16 substrate through `synapse::SCALE` (now `pub`,
  coupling pinned by test) — one saturating formula, one home.

**Honest scope:** the dense weights are **constructed, not trained**
(+1 on the target group, −1 on other E neurons, 0 on inhibitory — all
three codes exercised). Stage 3's claim is composition, not dense-layer
learning; SNN-side learning closed at 1.5d. A trained dense layer (e.g.,
STDP-style updates on the classifier) is future work, honestly labeled.

**What ships:** `kernel` module (`pack_trits` / `unpack_trit` /
`ternary_matvec` / `absmax_normalize_q15` — all buffer-based, zero-alloc,
integer-only, `no_std`), `bridge::{repack_i2s_to_kernel,
wire_gamma_to_substrate}`, `pub synapse::SCALE`, 14 new tests (known
vectors, error paths, property round-trips vs scalar reference), and the
gate example. Discipline gates green.

### Stage 4 — OPENED 2026-08-15 (multi-session; session 1 shipped)

The destination run: a pure-Rust runtime that loads and runs Bonsai `Q1_0`
models the way `bitnet.cpp` does in C++. Work lives on
`stage4-ternary-runtime` (pushed to both remotes); `main` receives only
honest, green-gated milestones.

**Session 1 — the container is real.** The first session shipped
`crates/neuralos-rt`: a buffer-based GGUF parser whose layout is pinned
verbatim from the Prism fork's own `gguf.h`/`gguf.cpp` (v3 container, 13
value types, flat arrays only, pow2 alignment default 32) plus the fork's
tensor-type numbers (`GGML_TYPE_Q1_0 = 41`, `GGML_TYPE_Q2_0 = 42`). The
real `Bonsai-1.7B-Q1_0.gguf` (248 MB, from HF `prism-ml/Bonsai-1.7B-gguf`,
gitignored) parses clean: **310 tensors (197 `q1_0` + 113 `f32`), qwen3
architecture, all data slices in-bounds, every `q1_0` tensor byte-exact
vs its dims** — and `token_embd.weight`'s first real block decodes through
the Stage-2 codec (fp16 scale 0x26f0 ≈ 27 milli, signs +65/−63 — genuine
embedding data, not zeros). `examples/bonsai_probe.rs` is the falsifier
run.

**Session-1 decisions (ISA):** from-scratch, no candle — the Q1_0 path is
already ours; revisit trigger recorded if the f32 op surface proves too
costly. Fog for sessions 2+: Q1_0 compute path (decode+existing kernel vs
fused binary kernel), f32-from-scratch vs candle-for-ops, tokenizer (Qwen
BPE from embedded data), and the gate's measurable bar.

**Session 2 — first real compute through real weights.** Shipped
`rt::{q1_0, norm}`: a per-block **`q1_0_matvec`** (sign-bit partial sums ×
per-block γ in the milli domain, property-tested against a decode
reference), embedding materialization (`q1_0_row_to_milli`), and an
**integer RMSNorm** (exact Newton `isqrt`, milli-domain, f32 norm weights
converted at the load edge via `f32_bits_to_milli`).
`examples/bonsai_forward.rs` runs the first-layer slice on the real
model: token id → 2048-dim embedding (±γ per block) → `blk.0.attn_norm`
RMSNorm → Q/K/V projections through `q1_0_matvec` on the real q1_0
tensors — all integer, `FORWARD: OK` across four tokens (Q absmax
~80–90k milli, all stages bounded and nonzero).

**Session-2 decisions (ISA):** Q1_0 compute path = per-block decode-matvec
now; fused/LUT kernel deferred until profiling shows need (fog №1
resolved). The f32-vs-candle activation question narrowed: this session's
entire path is integer. Model facts pinned for sessions 3+: qwen3, 28
blocks, 16 Q / 8 KV heads (GQA), head_dim 128, FFN 6144, rms eps 1e-6,
gpt2 tokenizer.

**Session 3 — the full forward pass.** Shipped `rt::{math, model}` and
`examples/bonsai_full.rs`: the complete 28-block Qwen3 forward on the
real model — per-block QKV (session-2 matvec), per-head q/k RMSNorm, YaRN
RoPE (pinned verbatim from the fork's `ops.cpp`: ramp + corr_dim + mscale,
factor 4 / base 1e6 / orig 8192), GQA 16Q/8KV attention with an
**integer softmax** (Q12 exp2-table, max-subtract, exact-sum), SiLU-gated
FFN, final RMSNorm, and tied-embedding logits over the full 151 669
vocab. **`FULL: OK`** — 4 tokens × 28 blocks in 14.2 s (release, 2-core),
residual stream healthy at every block (nonzero, off the rails), logits
in 0.8 s, top-5 ids = the digit tokens (structured, not collapsed; real
coherence judgment waits for the tokenizer, session 4). The entire
compute path is integer (milli/Q12); f64 appears only at the load edge
(norm weights, rope/exp2 tables — the pinned doctrine).

**Session-3 catches (honest):** the f64-reference doctrine paid twice —
it caught a real bug in the integer sigmoid rounding (an extra +Q12 in
the numerator) and a ×1000 unit slip in a *reference* test (the
implementation was right; the reference was wrong — recorded as the
inverse of the session-2 failure mode).

**Session 3.5 — adversarial review (10 agents, full arc).** Before
session 4 leans on this base, the whole bridge arc (rt + snn
bridge/kernel) went through a red-team pass. Found and fixed, severity
order: the **YaRN ramp window sat one octave high** (the pinned formula
takes the ELEMENT index i0 = 2·pair; the code fed the pair index —
interpolation band [34,68] instead of [17,34]; all three test
derivations shared the slip, so tests stayed green — fixed + window
pairs pinned); the **softmax exact-sum invariant was false at n=4**
(concrete counterexample `[0, −1386, −1386, −7624]` summed to 4097;
rewritten floor + largest-remainder, now exact for every length);
**`f32_bits_to_milli` was broken across an entire exponent decade**
(shift ≥ 64 panics/masks; at shift 64 it returned −2³¹ for a positive
input); a **fp16 −inf scale panicked** on `−(i32::MIN)` in embedding
decode (now saturates per the documented policy); the **65-token
prompt panic** (fixed [i32;64] score arrays) and **out-of-vocab token
panic** (both now loud `ModelError`s); `matvec_scaled`'s error contract
holes; round-half-away drift at three sites (one shared, f64-tested
`div_round_half_away` helper replaces all inline copies); loader
shape-blindness (dims now validated — transposed tensors reject);
config KVs cross-checked at load; `absmax_normalize_q15` returns u16
(32768 no longer wraps); golden non-symmetric i2_s/q1_0/q2_0 vectors
pin lane AND byte order (the old period-8 vector was lane-blind); all
65 536 fp16→milli conversions pinned exhaustively. Hardened:
HashSet dup-scan (O(n²) DoS), u32 gate compares, per-layer residual
deltas + soundness rail in bonsai_full (a dead attention layer can no
longer pass), `rt` marked `publish = false`. Post-fix run: all 28
layers alive (deltas 40k→4.2M milli), residual absmax 17.9M under the
66.6M norm-soundness rail, top-5 shifted as expected (true milli
logits now), 14.6 s forward. Deferred (named ISA tasks): `#[non_exhaustive]`
+ const-hoisting (alpha.2 checklist), gguf lazy arrays, reference-logit
equivalence test (session-4 pre-gate).

**Session 4 — tokenize → generate → the GATE (verdict: NO, 3/5).** The
finish line ran: `rt::token` loads the gpt2 byte-level BPE tokenizer
from the GGUF's embedded metadata (151 669 tokens, 151 387 merges) with
the Qwen2 pre-tokenizer as a hand-rolled scanner pinned verbatim from
the fork's own `unicode_regex_split_custom_qwen2` — no new dependencies
(the `regex` crate cannot even express the pattern's `(?!\S)`
lookahead). The forward was refactored alongside (not inside):
`Session`/`prefill`/`step` give an append-only KV cache whose outputs
are **bit-identical** to the reviewed full forward — proven at
tolerance 0 on a nonzero synthetic model (CI) and on the real file.
`argmax_logit` + a deterministic greedy loop complete generation:
**0.22 tok/s decode, 0.26–0.29 tok/s prefill** (release, 2-core),
residuals 18–29 M under the 66.6 M soundness rail.

**The gate** (`examples/bonsai_generate.rs`, 5 strict prompts fixed
before any run + 1 structural chat demonstrator via the embedded
template): `STAGE 4 GATE: NO` — 3/5 strict.

- **PASS** — `"1 2 3 4 5 6 7"` → `" 8 9 10 11 1"` (digit-pattern
  continuation).
- **PASS** — `"10 11 12 13"` → `" 14 15 16 17"` (multi-digit carry).
- **FAIL** — `"one two three four"` → `"-digit numbers. The problem is
  that the numbers are not unique"` (coherent text about numbers, not
  the sequence).
- **FAIL** — `"Monday Tuesday Wednesday"` → `": 10:00 AM\nWednesday: "`
  (schedule-listing continuation).
- **PASS** — `"The capital of France is"` → `" Paris, which is the
  capital of France…"` (factual recall — the honestly-at-risk one).
- **Chat demonstrator** (structural, never verdict-bearing):
  "Sure! Here's how you can count from 1 to" — coherent, cleanly
  formatted, pre-closed `<think>` handled.

**What the NO means (honest reading):** the runtime is proven
end-to-end — container, tokenizer, template, incremental integer
forward, greedy decode all work, deterministically, on real weights;
the chat demonstrator's coherence is the strongest single piece of
evidence that the stack is sound. What failed is the *model's
capability* on two word-sequence prompts — exactly the failure mode a
1-bit 1.7B can afford. Per gate doctrine the bridge stops here with
shipped artifacts; no prompt-fishing was done (the set and expected
strings were fixed pre-run, and the failures are the result).
Session C's first act is named in the ISA: pin the fork's reference
logits for a fixed prompt against ours, so any future quality claim
sits on an external anchor before the next capability judgment.

**Session-4 catches (honest):** the plan's `\p{N}{1,3}` digit grouping
was GPT-4 recall, not Qwen2 — the fork's source says `\p{N}` (one
digit per piece), caught by fetching the reference before writing the
scanner. Qwen's digit splitting (`" 8"` = two tokens) is why strict
prompts judge decoded-TEXT prefixes, never ids. The Bonsai chat
template inserts no default system prompt (unlike official Qwen3) —
read from the file, fragments asserted.

**Session C-pre — the NO attributed (2026-08-16).** The held merge's
blocking question — is the 3/5 our runtime's fault or the model's? —
was answered by building the reference runtime itself
(PrismML-Eng/llama.cpp @ `9ca265a`, CPU-only scratch build) and
running the same five frozen prompts, greedy forced by construction:
**the reference fails the same two prompts and passes the same
three.** "one two three four" → " four four the first part…" and
"Monday Tuesday Wednesday" → ": 10:00 AM - 12" — while the two digit
prompts come out **byte-identical** to ours (" 8 9 10 11 1", "
14 15 16 17", 24 greedy steps of agreement). At the verdict step the
reference's own distribution denies the expected tokens the top:
" five" sits at rank 4 (10.517) under its argmax " four" (11.486),
and " Thursday" is outside its top-10 — under greedy decoding, the
failures are the 1-bit 1.7B's, not the runtime's. The verdict line
is untouched; the merge case went to the principal with the
evidence. Two honest counter-findings ride along as Session-C scope:
our tokenizer splits "France" ([Fr, ance]) where the reference BPE
reaches the single token, and our logits track the reference at the
argmax level on verdict steps but drift well beyond rounding inside
the top-10 elsewhere (top-logit deltas up to ~5.4, later-step flips
on near-ties) — measured for the first time, not fixed; the Session
C delta redteam owns both.

**Session C-core — the drift killed (2026-08-16).** Both C-pre
counter-findings root-caused and fixed, with the runtime's fidelity
pushed past the reference-comparison bar. The instrument that did it:
an f64 mini-forward written from scratch in the harness (real units
end-to-end) — which matched the FORK's logits to ±0.03 and thereby
became the independent witness the integer path had never had.

- **Tokenizer (stale-rank BPE):** a heap entry pushed for a symbol
  pair can survive both liveness checks after one symbol GROWS via
  later merges; our pop re-looked-up the CURRENT pair and fired it at
  the stale entry's rank position — out of merge order. On " France"
  that produced [" F", "rance"] where the fork reaches the single
  token ĠFrance (9625). Fix: validate the popped entry against its
  push-time rank (the fork's text-equality check). CI regression trap
  (synthetic wrong-rank shape) + real-file pins.
- **Score scale (the big one):** the attention dot product lives in
  milli² (real×10⁶); milli scores need dot × 88.3883/10⁶ — the code
  divided by 10³, making **every score 1000× too large and saturating
  the integer softmax into a hard argmax.** Every attention head lost
  its secondary context mass; the f64 microscope measured the
  injection at **15.5% after block 0's attention alone** (1.1% at the
  embedding, then flat). The session-3 unit test had agreed with the
  bug because its f64 reference re-encoded the same wrong chain —
  circular green, now recorded as an ISA Learning.
- **Exact-γ (kept):** q1_0 matvec now applies block scales at
  fp16-exact precision (integer mantissa × 2^shift); the milli grid
  had been quantizing the model's γ ≈ 0.02–0.09 by 0.4–1.9% per
  block. Measured NOT the drift driver — kept because it is strictly
  more faithful.

**After (teacher-forced fork comparison, all three prompts):** argmax
agreement **36/36 steps**, top-10 overlap 9–10/10, max |Δtop| 0.597
(was 5.407); our France step-0 top logit 18.38 vs fork 18.38 (was
12.98); per-block error ≤1.1% and decaying. **The frozen gate re-run:
3/5 strict — verdict unchanged — but our continuations are now
byte-identical to the reference's greedy** (" Paris, and the capital
of Spain is Madrid…", " four four the first part…", ": 10:00 AM -
12"): the faithful-runtime state, where the runtime reproduces the
model's failures as faithfully as its successes. Merge call to the
principal with the full evidence.

**Session 4B — the runtime generalizes; gate on Bonsai-4B: NO 4/5
(2026-08-16).** The principal's call: attempt the strict YES on the
bigger tier, same frozen prompts, same verdict logic. First the
config diff, pinned from the file's own KVs before any code: 36
blocks, emb 2560, 32/8 heads, head_dim **128 unchanged**, FFN 9728,
vocab **151 669 unchanged** (not Qwen3-4B-upstream's 151 936) — and
two silent breakers a constants-only port would have hit:
`rope.freq_base` is **5e6 on 4B** (1e6 on 1.7B), and the 4B's
embedding slice carries 24 B of GGUF alignment padding the exact-size
rule rejected. `rt::model` is now config-driven (`ModelConfig` from
the fork's own keys, required-loud), the attention score scale
derives from head_dim and is pinned to reproduce the 1.7B constants
bit-identically, and the refactor is regression-proven: the 1.7B gate
byte-identical on all 43 verdict-bearing lines, the real-file
incremental≡forward exact test green in both profiles, and an
e1821be worktree witness pinning the full-forward path (which also
exposed that the old recorded top-5 was a pre-C-core artifact —
stale record, not drift). On 4B: probe/forward/full all green
(36/36 layers alive, residual 10.96 M of the 60.0 M derived rail),
teacher-forced drift 35/36 argmax with the France/tokenizer second
witness clean (12/12, max |Δtop| 0.064), and **THE GATE: NO — 4/5**:
p0/p1/p2/p4 pass with continuations **byte-identical to the fork's**
— including "one two three four" → " five six seven eight…", the
1.7B's failure prompt — and p3 ("…Thursday") fails on BOTH runtimes
(fork: ", June 12, 2018,"; the divergence inside the continuation is
one measured 0.1-logit near-tie). Per-model record: **1.7B NO 3/5,
4B NO 4/5, both fork-attributed, four-of-four matching continuations
byte-exact on the 4B.** The strict YES did not arrive; the 8B-or-stop
call is the principal's. Peak RSS 1.17 GB, decode 0.04–0.08 tok/s
(release, 4-core shared).

**Session 4C coda — p3 disambiguated: quantization severity
(2026-08-17).** The principal approved a fork-side ladder to answer
WHY p3 fails at 4B (greedy forced, raw top-10 dumps, our runtime
untouched). Rung (b) first: the F16 base — 8 GB mmap streamed through
3.5 Gi of RAM — puts " Thursday" TOP-1 at +3.05 logits. Rung (a):
ternary Q2_0 puts it TOP-1 at +3.08, and its step-0 top-10 is
IDENTICAL to the F16's in order, max |Δ| 0.029 — 2-bit ternary is
near-lossless against its own base on this prompt. Q1_0 is where the
weekday chain dies (" Thursday" outside the top-10). The
prompt-shape hypothesis is falsified — the base completes the bare
prompt decisively. And the chat template turns out to bury this
knowledge class at EVERY width (void-check passed: specials as
control ids): Q1_0 answers "It seems like your message might be a
typo or incomplete." — and so does the F16 base ("It" top-1, no
" Thursday" in its top-10, top-3 order identical to Q1_0's) — the
framing itself suppresses completion-mode. So **4/5 is the 1-bit 4B's
honest cap, not the family's**: the knowledge is
present at F16 and Q2_0, and p2's " five" margin triples from +0.93
to +2.97 at 2-bit. The 8B-under-1-bit question is now a grounded
capacity bet, with a cheap sibling on the table — the frozen prompts
on Q2_0-4B would very likely read 5/5. Both calls are the
principal's.

**Session D slice 1 — Q2_0 native: the family's first YES
(2026-08-17).** The principal called the 4C coda's cheap sibling: run
the frozen prompts on Ternary-Bonsai-4B-Q2_0 through OUR runtime. The
slice gave the runtime native ternary weights end-to-end — and the
probe-first discipline paid for the whole session before any compute
existed: **the first probe run refuted our own Stage-2 q2_0 pin** (all
253 q2_0 tensors failed the documented 64 w / 18 B arithmetic; the
fork's real `block_q2_0` is **128 w / 34 B** — fp16 `max|w|` + 32
LSB-first code bytes — double-witnessed by the file's byte arithmetic
and HF's own metadata; the 2026-08-15 "correction" had it backwards,
and self-derived test vectors had kept the error invisible because
nothing had ever eaten a real q2_0 byte). Re-pinned from source,
`decode_q2_0` finally eats real file bytes, and the compute stack
ships: `rt::q2_0` (fp16-exact scales, loud code 3, branch-free inner
loop — 193 s → 21.6 s on the 4B full forward with bit-identical
output) plus per-tensor type routing in the model (`QuantData`; Q1_0
files hit the exact same functions as before — 1.7B gate byte-diff
43/43, both Q1_0 suites green). **THE GATE: YES — 5/5**, the first
YES in the family's record: " Thursday04/05/2018 " — the weekday
chain 1-bit destroyed, byte-identical to the fork's greedy down to
the trailing space — plus p2 " five…", p0/p1/p4 fork-byte-identical.
Drift vs the 4C fork anchors (teacher-forced, all 12 steps): p3
argmax 12/12 with max |Δtop| 0.289; p2 11/12 with one measured
0.34-logit near-tie flip — inside the C-core bar, and tighter than
the 4B Q1_0's own drift. Q2_0 residuals run 6–9% of the soundness
rail (calmer than Q1_0 — real ternary zeros shrink the stream).
**Per-model record: 1.7B-Q1_0 NO 3/5 · 4B-Q1_0 NO 4/5 · 4B-Q2_0
YES 5/5** — three honest rows, all fork-attributed, all in the
paper. This slice is the ternary seam Session D slice 2 (Bonsai
weights → Trit → SNN → STDP, the hybrid experiment) builds on.

**Session D slice 2 — the hybrid seam: HYBRID GATE: ADAPTS
(2026-08-17).** For the first time, pretrained LLM structure entered
the SNN substrate and local plasticity read it. The experiment
(frozen `examples/hybrid_gate.rs`): decode 262,144 trits from the
real `blk.0.attn_q.weight` Q2_0 bytes (first 512×512 slice),
map onto the proven substrate at γ=125 (the LLM's fp16 block scales
are meaningless to SNN dynamics — recorded decision), wire
full-minus-diagonal (261,632 synapses) through the new public
`finalize_synapses` CSR path, run the 1.5c drive verbatim with STDP
on. Three gates, stated up front: **G1 import integrity** —
trit-exact by construction, 0/261,632 round-trip mismatches, first
real-data zero fraction 0.3655. **G2 spiking fidelity** — honest
negative-space result: imported, census-matched random control, and
zero-weight comparator all spike IDENTICALLY (35,157 spikes,
34.33 Hz vs 0.10 floor) — this drive regime is drive-dominated, so
G2 carries non-degeneracy + sustain + containment, and the structure
claim is explicitly deferred to G3 rather than smuggled into G2.
**G3 selective adaptation** — the claim that matters: intra-assembly
synapses depress −0.3133 milli-mean while inter stay at +0.0000
exactly (**Δ-SI 1.0000**), 321,571 bucket flips (alive, not
frozen), Hamming 21.79% < 50% (majority intact, not collapsed), and
zero sign crossings — the SynapseType bounds asymmetry held exactly.
**Verdict: pretrained structure survives AND discriminates under
local STDP** — deterministic, 20.5 s, 1050 MB. The arc's stakes in
one line: a foreign model's weights, imported bit-exactly, are not
just RUN by our substrate (slice 1) but LEARNED ON by it (slice 2).
(Session recovered after a mid-session machine death — code and
verdicts were in git, evidence regenerated by deterministic re-run,
number-for-number identical.)

**Session E — the loop-closer: LOOP GATE: CLOSED (2026-08-18).** The
arc's stake, realized: for the first time, a shipped quantized LLM's
weights made the FULL round trip — imported bit-exactly into the
spiking substrate (D-2), adapted under local backprop-free STDP,
re-encoded through the format bridge (`encode_q2_0`, new: Q2_0 is a
two-way format), and RUN by the foreign runtime with MEASURABLE
behavior change. The evidence chain, every link mechanical: the
surgery asserts D-2's recorded adapted state as a precondition
(Δ-SI 1.0000 et al. reproduced before any write); exports at the
file's original fp16 scales (structure adapted, magnitudes the
model's own); splices 512 disjoint 136-byte chunks with containment
proven (outside = 0, scales = 0); the written file re-parses and
decodes back to EXACTLY the adapted slice; both patched-file
production and fork runs are double-run deterministic; and the
judge's baseline reproduces the 4C anchors number-for-number before
the patched file is ever loaded. Result: **60/60 judged steps moved
(max |Δ| 0.534), top-10 membership shifted, 0/60 argmax flips** —
the honest physics of patching 0.5% of one layer's weights. The
claim is capability — local plasticity, applied to a real model,
visible in foreign tooling — not quality. What this opens: the
follow-on experiments the loop makes possible (more slices, whole
tensors, structured adaptation targets, measured-improvement
attempts) all now stand on a closed, verified circuit.

**Session E stage 0 — the honest instrument (2026-08-18).** The
deep-dive the principal's instinct prompted found the flaw worth
finding: the adaptation was drive-informed, not model-informed — the
substrate's firing never depended on the imported weights (the
recurrent ±12 μA channel was quantization-absorbed 8.6σ deep under
the 600 μA drive margin), so STDP sculpted the model's weights to
OUR drive's correlation structure, and the judge saw directionless
noise at the logits. Rather than scale a blind signal (the widened
slice is withdrawn), the loop gets the redesign it earned: a
measured-identity control (surgery on unadapted trits reproduces the
file byte-for-byte — attribution is now airtight), a
per-token-distribution judge (the fork's KLD instrument,
deterministic, sha-pinned corpora — generic text reads noise, the
model's own continuations read a small consistently-signed shift
with zero argmax movement), and a staged reopening of the channel:
amplitude sweep (down to 100 μA — below the excitatory threshold,
where recurrent input is the only road to a spike), then in-vivo
drive (the model's own layer-0 activations as the input the SNN
adapts to), scale last. The claim tracks the evidence: capability
closed the loop; model-informed coupling is what opens it wider.

**Remaining sessions (honest slice):** attention + full forward
(**✓ session 3**) → adversarial review of the whole arc (**✓ session
3.5**) → tokenizer + generation = the Stage 4 gate (**✓ session 4 —
verdict NO, recorded**) → Session C-pre (**✓ 2026-08-16 — the NO
attributed: the reference fork fails the same two prompts, passes
the same three, digit prompts byte-identical; tokenizer "France"
split + logit-drift findings recorded for the delta redteam**) →
Session C-core (**✓ 2026-08-16 — both findings fixed; drift past the
bar; gate re-run fork-byte-identical at 3/5; runtime fidelity
demonstrated at logit AND generation level**) → Session 4B (**✓
2026-08-16 — runtime generalized to Bonsai-4B, config-driven +
regression-proven; gate on 4B: NO 4/5, fork-attributed with four
byte-identical continuations; per-model record stands) → Session 4C
(**✓ 2026-08-17 — p3 coda: quantization severity; F16 base and Q2_0
both TOP-1 " Thursday" (+3.05/+3.08, top-10s identical within
0.029), 1-bit outside the top-10; prompt-shape falsified; chat shape
buries the weekday chain at every width — even the base clarifies
("It seems like…"); 8B/Q2_0 calls grounded for the
  principal**) → Session D slice 1 (**✓ 2026-08-17 — Q2_0 native in
  the runtime: Stage-2 q2_0 pin caught wrong by the probe and
  re-pinned from source (128 w / 34 B); rt::q2_0 + QuantData type
  routing with Q1_0 structural identity byte-diff-proven; GATE on
  Ternary-Bonsai-4B-Q2_0: **YES 5/5, the family's first YES** —
  " Thursday04/05/2018 " fork-byte-identical; drift p3 12/12
  argmax |Δtop| ≤ 0.289 vs the 4C anchors; per-model record
   1.7B-Q1_0 NO 3/5 · 4B-Q1_0 NO 4/5 · 4B-Q2_0 YES 5/5**) → Session D
   slice 2 (**✓ 2026-08-17 — the hybrid seam: finalize_synapses
   public CSR path + the frozen hybrid_gate experiment; G1 trit-exact
   import (0/261,632), G2 drive-dominated (honest negative space),
   G3 Δ-SI 1.0000 with zero sign crossings — **HYBRID GATE:
   ADAPTS**; deterministic; recovered post-crash from git +
   regenerated evidence**) → Session E (**✓ 2026-08-18 — the
   loop-closer: encode_q2_0 makes Q2_0 two-way; surgery with asserted
   D-2 preconditions, containment + disk round-trip proven; fork
   judge rebuilt at the pinned commit, baseline reproducing the 4C
   anchors; 60/60 steps moved, 0/60 argmax flips — **LOOP GATE:
   CLOSED**, capability-not-quality) → Session C
  (delta redteam on the new code, alpha.2 republish checklist — now
  carrying the q2_0 codec fix as a must-ship, gguf
  lazy arrays, contiguity validation; merge/no-merge of the branch and
  the 8B-vs-stop call are the principal's).

### 3. The lab bench (visible)



The Slint visualizer — a live microscope onto the substrate (spike raster +
synaptic weight heatmap + STDP learning toggle). Not a product; the proof the
library is alive, and the thing you show a researcher or a collaborator.

## What it honestly isn't

- **Not the next Urbit, not a product, not a company** — at least not from the
  substrate alone. The realistic audience today is SNN-on-RISC-V researchers
  (a few hundred worldwide) + curious Rust/embedded hackers. A fine foundation
  for a respected library and research artifact — not, by itself, demand.
- **Sovereignty is tone, not niche.** Urbit's 12-year arc shows conviction
  sustains but doesn't scale. Use it as the *why* wrapping the spine, not as
  the value prop.
- **Distro (Debian "Prime AI") and the crypto layer are seasoning**, not the
  differentiator. They support the ethos; they're not why anyone reaches for
  the library. The research-summarizer app stays demoted (commodity territory,
  re-scoped to the SNN visualizer in `3f7b8a0`).

## Realistic near-term path

1. **Finish the substrate:** NIR import/export, lock-free + SIMD ports,
   ternary Stage 1 + 1.5b + 1.5c + 1.5d (✓ done — ternary SNN spikes 1.00× baseline,
   learns via stochastic bucket-flips over a *full pairwise* STDP rule (LTP + LTD
   both reachable), and discriminates by correlation under structured input
   (SI 1.000 vs i16 1.000); bridge open to Stage 2)
   + Stage 2 format bridge (✓ done 2026-08-15 — `i2_s` round-trip bit-exact,
   `q1_0`/`q2_0` import exact; `docs/TERNARY_FORMAT.md` is the spec; gate YES)
   + Stage 3 shared kernel (✓ done 2026-08-15 — one `no_std` ternary matmul
   serves SNN + dense layers in a 4/4 hybrid gate; wire→compute repack seam;
   gate YES). **Stage 4 (pure-Rust ternary-LLM runtime) is now the earned,
   explicitly multi-session research step.**
2. **Deploy on QEMU RISC-V (`riscv64gc`)** — prove the `no_std` sovereignty
   claim with a real artifact; then ESP32-C3 silicon when budget allows.
3. **Position publicly as Lava's spiritual successor**; cite the
   Neuromorphic-OS paper (arXiv:2603.26722) and the Full-Integer-SNN paper
   (IEEE 2025) as academic validation of the framing + design.
4. **Reach out to ETH Zürich Benini / PULP** (Kraken, SpikeStream). They have
   the hardware; NeuralOS brings the open software library. The natural
   collaboration that turns a solo project into a referenced one.

## Relation to prior docs

This refines `docs/ARCHITECTURE.md`'s 4-target framing: the substrate is
elevated to the spine; the distro and research-app are demoted to seasoning.
Not a contradiction — a sharpening based on the landscape evidence. The
roadmap (`docs/ROADMAP.md`) phases still apply; this document is the *why*
that orders them.
