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

**Stage 3 (shared kernel) — PASSED 2026-08-15:**

- `neuralos_snn::kernel` — the shared `no_std` ternary matmul: sequential
  2-bit packed trits × Q15 absmax-normalized i16 activations → i32
  (integer BitNet-analog), zero-alloc, property-tested vs a scalar reference
- `bridge::repack_i2s_to_kernel` — the wire (`i2_s`) → compute (sequential)
  seam; `bridge::wire_gamma_to_substrate` — the imported-γ policy (fog №1);
  `synapse::SCALE` now `pub` with a pinning test
- `examples/ternary_hybrid_gate.rs` — the gate: SNN layer + dense layer
  through ONE kernel, dense weights arriving as `i2_s` wire bytes → **4/4
  classification, YES** (dense weights constructed, not trained —
  composition was the claim)

Gate question for the phase:

> Does a ternary SNN remain mechanically useful enough to justify the rest
> of the bridge? **Answer: YES, three times (Stage 1 gate, Stage 2 format
> gate, Stage 3 kernel gate).**

Stage 4 (full Rust ternary-LLM runtime on Bonsai `Q1_0`) is now the earned
step — explicitly gated, multi-session research territory.

**Stage 4 — CLOSED 2026-08-16 (gate verdict: NO; branch
`stage4-ternary-runtime`, merge call deferred to principal):** s1 —
GGUF container on the real file (310 tensors, all `q1_0` byte-exact).
s2 — `q1_0_matvec` + integer RMSNorm, first-layer compute (`FORWARD:
OK`). s3 — **full 28-block Qwen3 forward**: YaRN RoPE + GQA + integer
softmax + SiLU FFN + tied logits, integer compute path (`FULL: OK`,
4 tok in 14.2 s release). s3.5 — **adversarial review of the whole
bridge arc** (10 agents): YaRN ramp window fixed (one octave high),
softmax exact-sum made true for all n (was false at n=4), f32→milli
small-exponent decade fixed, hostile-scale saturation, 65-token/OOV
panics → loud errors, shared round-half-away helper, dims+config
validated at load, golden lane/byte-order vectors, per-layer health
gates (`FULL: OK` re-run: 28/28 layers alive). s4 — **tokenizer +
incremental decode + generation = the gate**: embedded Qwen2 BPE
(fork-pinned hand-rolled scanner, zero deps), bit-exact KV-cache
refactor (tolerance 0 vs the reviewed forward, synthetic + real),
greedy decode 0.22 tok/s (release, 2-core), chat demonstrator coherent
("Sure! Here's how you can count from 1 to") — but the strict gate
said **NO: 3/5** (digit counting ×2 and "The capital of France is
→ Paris" pass; word-sequence continuations fail). Bridge stops with
shipped artifacts per gate doctrine; Session C's first act is the
fork-logit reference comparison (ISA Decisions 2026-08-16).
**C-pre (2026-08-16) — the NO attributed:** the reference fork
itself (PrismML-Eng/llama.cpp @ 9ca265a, greedy
forced-by-construction) fails the same two prompts and passes the
same three, with the digit prompts byte-identical to ours over 24
greedy steps — the two failures are the 1-bit model's under greedy,
not the runtime's. Merge case presented to the principal (not
executed); two Session-C findings recorded: our tokenizer splits
"France" where the reference reaches the single token, and our
logits drift beyond rounding vs the f32 reference inside the top-10
(deltas up to ~5.4) while agreeing at the verdict-step argmax.

**C-core (2026-08-16) — both findings fixed; fidelity past the bar:**
(1) the tokenizer bug was a stale-rank BPE heap entry firing a grown
pair out of merge order — fixed with the fork's push-time-rank
validation (" France" now reaches ĠFrance/9625, CI trap + real-file
pins); (2) the drift was a **1000x attention-score unit error**
(milli^2 dot /1e3 instead of /1e6 -> softmax saturated into hard
argmax — 15.5% error injected at block 0's attention, found by an
f64 micro-forward that matches the fork to ±0.03), plus exact-gamma
matvec (fp16 mantissa x 2^shift, integer) kept on faithfulness
grounds. After: teacher-forced fork comparison argmax-identical
**36/36 steps**, max |dTop| 0.597 (was 5.407); frozen gate re-run
**3/5 with fork-byte-identical continuations** — the faithful-runtime
state. Merge call with the principal (commits local per protocol).

**Session 4B (2026-08-16) — runtime generalized to Bonsai-4B; gate
NO 4/5, fork-attributed.** Config-driven `rt::model` (geometry from
the file's own KVs — 36 blocks/emb 2560/32-8 heads/FFN 9728/
freq_base 5e6, head_dim 128 and vocab 151 669 unchanged; score
scale derived from head_dim, pinned bit-identical to the 1.7B
constants; alignment-padded tensor slices handled), refactor
regression-proven on 1.7B (gate byte-identical, real-file
incremental≡forward green, e1821be worktree witness). On 4B:
probe/forward/full green (36/36 layers alive), drift 35/36 argmax
(France second witness 12/12, |dTop| 0.064), and the frozen gate
**NO 4/5**: p0/p1/p2/p4 pass with fork-byte-identical continuations
(" five six seven…" — the 1.7B's failure prompt passes on 4B);
"Monday Tuesday Wednesday"→" Thursday" fails on BOTH runtimes.
Per-model record: 1.7B NO 3/5, 4B NO 4/5, both the model's ceiling
under greedy. 8B-vs-stop + merge are the principal's calls.

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
