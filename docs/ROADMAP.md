# NeuralOS v2 — Roadmap

> Execution order for the active repo.
> `docs/VISION.md` is the north star; this document is the shipping
> sequence. Re-baselined 2026-08-20 at the project autopsy — the
> pre-autopsy roadmap (and the session records it had absorbed) is in
> git history (`pre-autopsy-cleanup` tag) and `docs/RESEARCH_LOG.md`.

## Priority order

| # | Component | Status |
|---|---|---|
| **1** | `neuralos-snn` — `no_std` SNN substrate | Active spine. The 2026-08-08 near-term list (NIR, lock-free, SIMD hardening) was starved by the bridge arc and is **first-class again** — NIR DONE through general assembly (slices 1+2 + `build_network`, 2026-08-22 @ alpha.5), QEMU proof landed 2026-08-21; lock-free (re-scoped, below) remains; **SIMD hardening DONE 2026-08-31** (ten-commit branch, ISA § Close-out). |
| **2** | `neuralos-app` — Slint visualizer / lab bench | Untouched since 2026-08-08; Phase-2 items re-opened. |
| **3** | RISC-V deployment proof | **QEMU riscv64gc DONE 2026-08-21** (both legs; `evidence/qemu-riscv-gate/`). **ESP32-C3 silicon DONE 2026-09-09** (`evidence/esp32c3-bringup/`: one LIF neuron on a SuperMini, first spike step 56 (the real-time loop, wall-clock paced; the burst prints 55, as the host does), 14.7 spikes/s, exact host match). Per-step cost, round 27 (alpha.7, published 2026-09-13, ISA round 27): the neuron as a network holds it (in memory, dt at run time) **3,878 ns/step on alpha.6 → 1,579 on alpha.7**, the free loop 2,863 → 563, behavior bit-identical; the host bench about 2 to 2.5 ns/step slower, recorded. The earlier figures, true when written: 1,501 on alpha.5 and 2,842 on alpha.6 (ISA round-23), both the free loop under the firmware's constant dt. HiFive only if a session names it. |
| **4** | Paper track | The Branch B article (in `paper/`) — finish, gate, submit. Must not displace 1–3. |
| **5** | Bridge follow-ups | **Frozen record.** Reopening is the principal's call on the recorded forks. The one active bridge-adjacent task is R4 (harness extraction) below. |

## 0.1.0

**Same graph, same spikes, every target.** A network built from a
`.nir` file runs heapless and panic-free on the ESP32-C3, bit for bit
with the host.

0.1.0 is tagged when the fourteen checks below hold, not on a date. A
stranger can run each one. This list is their one home; a tick names
the PR that made the check hold, as of this tree.

1. [x] Traces from today's network; CI replays them on host, QEMU at release. (Host: CI, PR B, #26. QEMU: PR F, #30, `proofs/qemu-trace-replay/`, the twelve plasticity-off traces bit for bit; PR G, #31, the fourteen, through the library's row writer.)
2. [x] `FixedNetwork` replays them bit for bit on the ESP32-C3. (PR E, #29: twelve of thirteen; `plasticity-on` is outside `FixedNetwork` by design. PR H, #32: the fourteen on the board, PR G's two D8 witnesses among them.)
3. [x] No panic path in the step, proven at link time. (PR E, #29.)
4. [x] No spike ring in the neuron; history is a recorder. (PR D, #28.)
5. [x] MSRV declared and tested on the bumped pin. (PR C, #27: `rust-version` 1.92 checked on 1.92.0 in CI, in the `std`, `no_std` and `simd` configurations; PR L, #37, added the fourth, `unstable-stdp`.)
6. [x] A two-layer snnTorch, norse or rockpool graph builds (LIF→Linear→LIF). (PR G, #31: a two-layer snnTorch graph builds.)
7. [x] `.nir` → fixed arrays → firmware, one documented command. (Arrays: PR G, #31. PR H, #32: `stranger.sh run` on the snnTorch witness, fifteen cases bit for bit.)
8. [x] Worst case from capacity, measured at that case on the C3. (PR I, #33: the two corners of `44·N + 6·S ≤ 262,144` on the board, 0 red; the worst case measured, the neuron corner, 5,850,980 ns/step mean; all-to-all computed at 5,313,354 ns.)
9. [x] STDP behind an unstable feature. (PR L, #37: `unstable-stdp`, off by default, holds `STDPRule`, `Synapse::update_weight`, `set_plasticity_enabled` and `stochastic_ternary_step`, and since round 46 the flip under it, `trit::stochastic_ternary_flip` and `trit::STOCHASTIC_FLIP_RATE`; anything behind it may change or go in a minor release. A network starts with plasticity off, feature or not, so a default build does not learn; the getter and the counter fields stay stable. The 14 plasticity-off traces replay with the feature off and on, the 15 with it on; the default config is a gate, `cargo test -p neuralos-snn`.)
10. [x] The README's first example is a real doctest; `missing_docs` on. (PR J, #34: § Quick start is the board's own neuron, asserting the 147 spikes and the first spike's index 55 the C3 prints; § Usage sketch compiles too; `missing_docs` on with 53 fields documented.)
11. [x] Rustdoc states rounding, saturation, reset, leak, step order, the one-step delay. (PR K, #36: `LIFNeuron::integrate_and_fire` § Semantics and `SpikingNeuralNetwork::step` § Order, each sentence a doctest or a named test, the pulse a refractory `post` drops among them; `cargo doc -D warnings` is a gate. Reworded from "delay is the edge's": no edge carries a delay in this code.)
12. [x] The default build forbids `unsafe` (only `simd` has it). (PR J, #34: the library forbids `unsafe_code` with the feature off and denies it with the feature on, `pub mod simd` carrying the one `allow`; both falsifiers in the commit.)
13. [x] Grep gate on release texts: no "golden", "conformance", "successor", Lava. (Ticked at a stamp, by the grep in that release draft's § Procedure: alpha.8 stamp, 0 hits.)
14. [x] Zero runtime dependencies (`cargo tree -e normal`) at each release. (PR F, #30: `cargo tree -e normal -p neuralos-snn` prints the crate alone, with `std` and without.)

## Current validated state

```bash
cargo check  --workspace --all-targets
cargo test   --workspace                          # offline; the count is the CI log's (one number, one home) + 5 rt model-gated #[ignore]
cargo clippy --workspace --all-targets -- -D warnings
cargo build --no-default-features -p neuralos-snn # the no_std gate
```

Bridge experiments additionally need `models/*.gguf` (gitignored — see
AGENTS.md § Workspace) and, for judged runs, the foreign fork built by
`tools/build_fork.sh` into `fork-build/` (gitignored).

## The 2026-08-20 remediation ladder (post-autopsy)

Strict order — nothing new opens until the rung above is 100%.

| Rung | What | State |
|---|---|---|
| R0 | Recovery tags (`pre-autopsy-cleanup`, `pre-autopsy-cleanup-paper`) | ✅ done |
| R1 | Delete merged `stage4-ternary-runtime`; ignore `.opencode/`, `fork-build/`, `.figvenv/` | ✅ done |
| R2 | Docs truth pass: AGENTS/README/VISION rewrite, RESEARCH_LOG split, this re-baseline | ✅ done |
| R3 | Kill `/tmp` path rot in tools + evidence READMEs; repo-local fork build (`tools/build_fork.sh`) | ✅ done |
| R4 | Extract the shared hybrid harness; rewrite the 6 hybrid examples on it; re-run pins recorded verdicts | ✅ done (2026-08-21; all re-pins exact — `evidence/r4-closeout/`; the stale H1 invivo bar root-caused + the H2 record re-pinned byte-identical) |
| R5 | `evidence/INDEX.md` — session → claim → files | ✅ done |
| R6 | Merge `paper-draft` → main; paper builds from main | ✅ done |
| R7 | Original roadmap work, in order (below) | ⬜ **open — NIR general assembly landed 2026-08-22 (alpha.5); SIMD hardening landed 2026-08-31; lock-free remains** |

## Phase 1 — Substrate hardening (the starved list, now first)

| Work item | Why it matters |
|---|---|
| NIR import/export | Interop with snnTorch/SpikingJelly; the #1 ecosystem recommendation. **DONE through general graph assembly (slices 1+2 2026-08-21, `build_network` 2026-08-22 @ alpha.5; gates: format 4/4, hdf5 5/5, assembly 6/6 + cross-container 3/3 — evidence/INDEX.md).** The spiking edge LIF→Linear→LIF is named and built (D8, PR G). Remaining: the R18 deferral family — the readout to Output (LIF→Linear→Output), direct drive (Input→LIF), encoder-only (lowest pull). |
| Lock-free ports from v0.1 archive | Throughput and future concurrency experiments. Re-scoped (2026-08-22 ruling): A-extension-capable targets only — rv32imc / ESP32-C3 are NOT (no atomics); the target named 2026-09-09 (ISA round-20): QEMU riscv64gc; parked until after 0.1.0, opens when a multi-core board with atomics enters |
| SIMD follow-up / hardening | **IN REVIEW 2026-09-01** (core landed 2026-08-31; two follow-up branches carry review fixes — `fix/lif-scalar-domain` then `fix/simd-fixture-and-record`, in that order) — slice-length contract enforced, overflow domain pinned (`DT_OVER_TAU_MAX`), floor-vs-truncate parking bias fixed (19 mV → ≤ 8 mV, cost +11.7 % of the vector path, scalar-controlled and identical across both benchmark runs; the raw spread is 8–13 %, machine load — corrected 2026-09-01 from "12–15 %", which was one run quoted as if it were both), every rounding-dependent number pinned exactly and reproducible from the tree (`cargo test -p neuralos-snn --features simd -- --ignored`); forks (b) exact ÷1000 and (c) half-cost recorded with triggers in the module doc; record in ISA § Close-out, evidence in `evidence/simd-hardening/` |
| Additional regression/property tests | The transmission-wire lesson: no unit test had ever exercised live transmission until session F. **Traces since 2026-09-13 (PR B, alpha.8): `crates/neuralos-snn/tests/traces/`, 14 regression cases and 1 reference vector, compared by the test gate, regenerated by `cargo run -p neuralos-snn --features unstable-stdp --example trace -- write`; C, D and E prove "behavior unchanged" by that compare.** |
| `no_std` discipline checks | Preserve the embedded/RISC-V posture (CI gate already green) |

## Phase 2 — Lab bench / visualizer

| Work item | Why it matters |
|---|---|
| UI/UX refinement for controls and stats | Better debugging and demonstration value |
| More observability (rates, activity summaries, weight drift) | Makes learning dynamics legible — including the new VoltageResolution grids |
| Smoke validation / render verification workflow | Keeps the visualizer trustworthy as a demo artifact |
| Stability around threading and shutdown | Protects the hardest-won lesson in the app layer |

## Phase 4 — RISC-V deployment proof

1. ✅ QEMU `riscv64gc`, reproducible path (2026-08-21: both legs green —
   bare-metal none-elf gate + full suite under linux-user musl;
   `evidence/qemu-riscv-gate/`)
2. ✅ `neuralos-snn` executing in that environment (Leg A 175/175 cited
   checks; Leg B 195/195 real tests incl. transmission trio + Leg-C pins)
3. ✅ Documentation of what is truly `no_std`-portable vs host-only
   (`network`/`csr`/`stats` are std-gated and ran in Leg B's linux-user
   posture; Leg A's no_std surface is the six core modules — recorded in
   `evidence/qemu-riscv-gate/README.md`)

Silicon landed 2026-09-09 (ESP32-C3, `evidence/esp32c3-bringup/`);
nothing of this phase remains but the lock-free ports (Phase 1's row above).
CI leg parked as a named follow-up (runner cost under TCG unmeasured).

## Phase 5 — Paper + bridge follow-ups (gated, frozen)

- **Paper:** build/figure/language gates green from main (`make`,
  `make figs`, `make gate` in `paper/`); then submission. Paper work
  never displaces an open Phase-1/2 item (AGENTS.md § Session
  discipline).
- **Bridge record:** frozen. Any reopening (criterion forks, 8B/Q2_0
  capacity bets, model-informed coupling) is a recorded fork in the
  ISA — the principal's call, scoped before any session opens.
- **R4 harness extraction** is the one active task: dedupe
  `hybrid_*`/`null_patches` (~1,864 duplicated lines, measured pairwise)
  into a shared rt harness module; acceptance = deterministic re-runs
  reproduce the recorded verdicts; tag `examples-pre-extraction` first.

## What is no longer the center of gravity

Distro work, crypto wrappers, sovereignty seasoning, old summarizer
plans — unchanged from the pre-arc roadmap. They can return, but never
displace substrate + lab bench + gated research.

## Practical next moves

The chapter, not the history. Closed work lives in `ISA.md` and the
records it names in `evidence/INDEX.md`. Three lines, in order:

1. **alpha.9: the pub walk.** What stops being public before the
   crate's surface is promised: paths and re-exports, `LIFNeuron` with
   its noise default, the rest of the field walk. No pinned value
   moves: the traces bit for bit, the firmware `.text` unmoved. Then
   the rename, the front door, the stamp.
2. **0.1.0: the stamp.** The principal's call, once the fourteen checks
   of § 0.1.0 hold again at the final tree, after the rename.
3. **After 0.1.0: the learning chapter** (`docs/VISION.md` § Realistic
   near-term path, item 6).
