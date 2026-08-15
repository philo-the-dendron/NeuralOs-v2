# NeuralOS v2 — Roadmap

> Execution order for the active repo.
> `docs/VISION.md` is the north star; this document is the shipping sequence.

*Updated 2026-08-08 to match the active workspace state and the current visualizer-first framing.*

## Priority order

| # | Component | Status |
|---|---|---|
| **1** | `neuralos-snn` — `no_std` SNN substrate | Active spine of the repo. Core library, tests, topology/plasticity work, and `no_std` discipline come first. |
| **2** | `neuralos-app` — Slint visualizer / lab bench | Active visible artifact. Runs the library live and makes spikes / weights / learning legible. |
| **3** | Ternary bridge — Stage 1 | Planned research direction, gated by substrate health. Starts with ternary SNN only, not full runtime ambition. |
| 4 | RISC-V deployment proof | Important proof of the `no_std` claim: QEMU first, real silicon after. |
| 5 | Distro / crypto / broader sovereignty stack | Still relevant to the overall ethos, but no longer the center of gravity of this repo slice. |

## Current validated state

The workspace currently validates the following successfully:

```bash
cargo test -p neuralos-snn
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

These checks describe the current repo state more accurately than the older phase notes from the summarizer era.

## The active phases

### Phase 1 — Substrate hardening

Goal: make `neuralos-snn` the strongest artifact in the workspace.

Current shape of the substrate:

- `no_std`-by-default library
- i16 fixed-point hot path
- LIF neurons
- STDP rule
- network orchestration
- 4 topology builders
- CSR sparse synapse matrix
- optional AVX2 SIMD path

Near-term work inside this phase:

| Work item | Why it matters |
|---|---|
| NIR import/export | Interop with adjacent SNN ecosystems; easiest high-value bridge outward |
| Lock-free ports from v0.1 archive | Throughput and future concurrency experiments |
| SIMD follow-up / hardening | Keep the performance path honest and test-backed |
| Additional regression/property tests | Protect invariants around plasticity, propagation, and topology behavior |
| `no_std` discipline checks | Preserve the embedded / RISC-V posture |

### Phase 2 — Lab bench / visualizer

Goal: make the substrate visible, debuggable, and showable.

The app is now the live SNN visualizer, not the old research summarizer. Its role is to make the library legible.

Current visualizer responsibilities:

- spike raster
- synaptic weight heatmap
- STDP learning toggle
- sustained-firing mode for visible activity
- worker-thread simulation + Slint UI thread bridge

Near-term work inside this phase:

| Work item | Why it matters |
|---|---|
| UI/UX refinement for controls and stats | Better debugging and demonstration value |
| More observability (rates, activity summaries, weight drift) | Makes learning dynamics easier to understand |
| Smoke validation / render verification workflow | Keeps the visualizer trustworthy as a demo artifact |
| Stability around threading and shutdown behavior | Protects the hardest-won lesson in the app layer |

### Phase 3 — Ternary bridge, Stage 1 → 2 (both gated, both passed)

Goal: test the smallest meaningful research gates without overcommitting.

**Stage 1 (ternary SNN) — PASSED** after the 1.5b/1.5c/1.5d reopen cycle:

- `Trit`-style weight type `{-1, 0, +1}` plus scale — shipped
- SNN adapted to that representation — spikes 1.00× baseline, learns via
  stochastic bucket-flips over full pairwise STDP, discriminates by
  correlation (SI 1.000 = i16 parity)
- deterministic per-step re-projection ruled out (Stage 1's honest NO)

**Stage 2 (format bridge) — PASSED 2026-08-15:**

- `docs/TERNARY_FORMAT.md` — the wire-format spec, layouts pinned verbatim
  from microsoft/BitNet and PrismML-Eng/llama.cpp source
- `neuralos_snn::bridge` — `i2_s` encode/decode (bit-exact round-trip),
  `q1_0`/`q2_0` import (loud errors, no silent clamping), integer-only
  fp16 scale plumbing; `no_std`, zero-alloc
- `examples/ternary_format_gate.rs` — the gate: **YES**

Gate question for the phase:

> Does a ternary SNN remain mechanically useful enough to justify the rest
> of the bridge? **Answer so far: YES, twice.**

Stage 3 (shared `no_std` ternary kernel) is the next earned step; Stage 4
(Rust ternary-LLM runtime) stays gated on Stage 3's proof.

### Phase 4 — RISC-V deployment proof

Goal: make the `no_std` claim concrete.

Order of attack:

1. QEMU `riscv64gc`
2. run `neuralos-snn` in that environment
3. real silicon when budget/hardware permits

Deliverables for this phase:

- a reproducible QEMU path
- successful `neuralos-snn` execution in a RISC-V setting
- documentation showing what is truly `no_std`-portable vs host-only

### Phase 5 — Bridge stages 2+

These are explicitly gated by the success of Phase 3.

Potential later work:

- ternary format bridge
- shared `no_std` ternary kernel
- tiny hybrid SNN/LLM proof
- only then, possibly, a larger ternary runtime effort

This is research territory and should stay honestly labeled as such.

## What is no longer the center of gravity

The following areas still matter, but they are no longer the active core of the repo roadmap:

- Debian distro work
- crypto wrapper work
- broader “sovereignty stack” seasoning
- old summarizer-app plans

They can return later, but they should not displace the substrate + visualizer + gated bridge sequence.

## Relation to v0.1 and prior roadmap language

The archived v0.1 repo remains useful as:

- a mine for tested ports
- a source of audit findings
- a reminder of what not to reintroduce

This roadmap intentionally drops the old summarizer-first execution order. The current repo and `docs/VISION.md` make the real structure clear:

1. substrate
2. lab bench
3. gated ternary research

## Practical next moves

If working from this roadmap today, the best near-term sequence is:

1. continue hardening `neuralos-snn`
2. improve the visualizer as the lab bench over that library
3. only then begin ternary Stage 1 experiments
4. prove the `no_std` claim on QEMU RISC-V
