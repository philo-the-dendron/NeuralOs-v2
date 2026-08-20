# NeuralOS v2 — Vision

> The open `no_std` spiking-neural substrate for RISC-V edge silicon —
> and the lab bench to watch it think.

*Status: north star. Rewritten 2026-08-20 (autopsy rung R2) back to a
one-page direction document — the 2026-08-08→20 bridge saga now lives in
`docs/RESEARCH_LOG.md`; the shipping sequence lives in `docs/ROADMAP.md`.*

---

## What NeuralOS v2 is

A sovereignty stack in pure Rust, organized around one defensible
artifact: a `no_std`, i16 fixed-point spiking-neural-network library
(`neuralos-snn`) designed to run on RISC-V edge silicon (ESP32-C3, QEMU,
HiFive). Around that spine: a visualizer that makes the substrate
legible, and a research runtime (`neuralos-rt`, not published) that
proved the ternary bridge — a closed, honestly-recorded research chapter
whose distilled record is the paper in `paper/`.

This is **not** an "AI OS" (the v0.1 pretense, archived). It is a library
with a microscope on top, sitting in a documented vacuum at the
coordinates the hardware frontier needs.

## Why it's defensible (the evidence)

- **Intel archived Lava on 2026-05-13.** The reference i16 fixed-point SNN
  framework is gone. NeuralOS positions as its spiritual successor.
- **The SNN-on-RISC-V frontier is real and software-starved** —
  SpikeStream, FeNN, SNAP-V, Kraken SoC, IzhiRISC-V: a 2024–2026 burst
  of hardware work with no mature open software library to run on it.
- **The `no_std`/embedded SNN slot is vacant across all languages** —
  27 SNN frameworks listed in the open-neuromorphic guide; every one
  Python/C++/JAX except one low-quality Rust entry.
- **The i16 fixed-point design is externally validated** — IEEE 2025
  *"Full-Integer SNN Inference with RISC-V ISA"*: float-free integer SNN
  is *required* for FPU-less edge silicon.

## The three chapters

### 1. The substrate (now — the active front)

`neuralos-snn` on crates.io (`0.1.0-alpha.2`): `no_std`, i16 fixed-point,
LIF + full pairwise STDP + 4 topologies + CSR sparse matrix + AVX2 kernel
+ the ternary bridge codecs (`i2_s`/`q1_0`/`q2_0`) + shared ternary
matmul + live synaptic transmission (the session-F fix).

**Near-term — the starved original list, first-class again:** NIR
import/export (interop with snnTorch/SpikingJelly — the #1 ecosystem
recommendation), SIMD follow-up/hardening, lock-free + SIMD ports from
the v0.1 archive, more property tests. The `no_std` claim gets its QEMU
`riscv64gc` proof.

### 2. The ternary bridge — CLOSED (research record, honestly labeled)

The gates ran; the record is final:

| Stage | Question | Verdict |
|---|---|---|
| 1 / 1.5b–d | Does a ternary SNN spike and learn? | **YES** (SI 1.000 = i16 parity, stochastic flips, full pairwise STDP) |
| 2 | Can we round-trip a ternary tensor? | **YES** (`i2_s` bit-exact; `q1_0`/`q2_0` import; `docs/TERNARY_FORMAT.md`) |
| 3 | Does one shared kernel compose SNN + dense? | **YES** (4/4 hybrid classification) |
| 4 | Pure-Rust ternary-LLM runtime on Bonsai? | **NO, fork-attributed** (1.7B-Q1_0 3/5 · 4B-Q1_0 4/5 · 4B-Q2_0 5/5 — the model's ceiling, runtime fork-byte-identical) |
| — | Hybrid seam: do pretrained weights adapt under STDP? | **ADAPTS** (pairing-selective, clamp-rectified) |
| — | The loop: can adapted weights re-enter the foreign runtime? | **CLOSED as capability** — final adjudication: **unattributed perturbation** (Branch B) |

Full run logs: `docs/RESEARCH_LOG.md`. Distilled record: `paper/`.
Raw evidence: `evidence/INDEX.md`. Reopening any of this is the
principal's call, on the recorded forks — not default work.

### 3. The lab bench (visible)

The Slint visualizer — a live microscope onto the substrate (spike
raster + synaptic weight heatmap + STDP learning toggle). Starved since
2026-08-08 while the bridge ran; first-class again (Phase-2 items in the
roadmap).

## What it honestly isn't

- **Not the next Urbit, not a product, not a company.** The realistic
  audience today is SNN-on-RISC-V researchers + curious Rust/embedded
  hackers.
- **Sovereignty is tone, not niche.** The *why* wrapping the spine, not
  the value prop.
- **Distro and crypto are seasoning**, not the differentiator.

## Realistic near-term path

1. **Feed the spine:** NIR import/export, SIMD hardening, lock-free
   ports — the original Phase-1 list.
2. **QEMU `riscv64gc` proof** of the `no_std` claim; ESP32-C3 when
   budget allows.
3. **Position publicly as Lava's spiritual successor**; cite the
   Neuromorphic-OS paper (arXiv:2603.26722) and the Full-Integer-SNN
   paper (IEEE 2025).
4. **Outreach to ETH Zürich Benini / PULP** — they have the hardware;
   NeuralOS brings the open software library.
5. **Paper track:** finish and submit the Branch B article from `paper/`
   (its own gated workflow, never again displacing Phase-1/2 work).
