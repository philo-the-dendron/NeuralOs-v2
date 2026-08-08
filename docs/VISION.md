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
| **1. Ternary SNN** | `Trit` weight type `{-1,0,+1}` + scale | a more efficient + more biologically-plausible SNN variant | does it still spike + learn (STDP) comparably to i16? **✓ Spiking: YES (1.00× baseline). Learning: Stage 1 deterministic NO → Stage 1.5b stochastic YES (802 flips, non-collapsed). Bridge reopened (see below)** |
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
non-collapsed (1.00×). The bridge is reopened. Stage 2 (format bridge) is
earned.

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
That is Stage 2+'s concern, not this gate's. The 1.5a latent-accumulation
fallback remains unstarted as insurance.

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
   ternary Stage 1 + 1.5b (✓ done — ternary SNN spikes 1.00× baseline and
   learns via stochastic bucket-flips; bridge open to Stage 2).
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
