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
| **2. Format bridge** | ternary format spec; BitNet-compatible **export**, Prism `Q1_0` **import** | NeuralOS speaks the lingua franca of both fields | can we round-trip a ternary tensor? |
| **3. Shared kernel** | one `no_std` ternary matmul; a tiny hybrid net (SNN layer + dense-LLM-style layer) | a reusable Rust ternary kernel + a showable hybrid demo | does the union compose — compute something coherent? |
| **4. Full Rust ternary-LLM** | extend/replace candle's quantized kernels to run a Bonsai `Q1_0` model in pure Rust | the Rust answer to `bitnet.cpp` — sovereignty-grade local AI | gated on Stage 3's proof; multi-session research |

**Format decision — deferred.** Stage 1 uses plain `{-1,0,+1}` + scale (zero
wire-format lock-in while we don't yet know the ternary regime works for our
SNN). If the gate passes, Stage 2 commits to **BitNet `Round()` native +
Prism `Q1_0` import** — because BitNet gives a *mechanical* bridge (real
weights flow between SNN and LLM), whereas standard TWN would be conceptual
only (shared alphabet, incompatible encoding, isolated from the models
actually shipping).

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
   (SI 1.000 vs i16 1.000); bridge open to Stage 2).
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
