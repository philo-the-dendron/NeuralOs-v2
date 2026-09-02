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
| **3** | RISC-V deployment proof | **QEMU riscv64gc DONE 2026-08-21** (both legs; `evidence/qemu-riscv-gate/`). Silicon (ESP32-C3/HiFive) remains, priority-gated (board decided 2026-08-22 — merged-plan step 3). |
| **4** | Paper track | The Branch B article (in `paper/`) — finish, gate, submit. Must not displace 1–3. |
| **5** | Bridge follow-ups | **Frozen record.** Reopening is the principal's call on the recorded forks. The one active bridge-adjacent task is R4 (harness extraction) below. |

## Current validated state

```bash
cargo check  --workspace --all-targets
cargo test   --workspace                          # offline; 307 executed green (3 app, 208 snn, 96 rt) + 5 rt model-gated #[ignore]
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
| NIR import/export | Interop with snnTorch/SpikingJelly; the #1 ecosystem recommendation. **DONE through general graph assembly (slices 1+2 2026-08-21, `build_network` 2026-08-22 @ alpha.5; gates: format 4/4, hdf5 5/5, assembly 6/6 + cross-container 3/3 — evidence/INDEX.md).** Remaining: the R18 deferral family — readout edges (LIF→Linear), direct drive (Input→LIF), encoder-only (lowest pull). |
| Lock-free ports from v0.1 archive | Throughput and future concurrency experiments. Re-scoped (2026-08-22 ruling): A-extension-capable targets only — rv32imc / ESP32-C3 are NOT (no atomics); name the target before porting |
| SIMD follow-up / hardening | **IN REVIEW 2026-09-01** (core landed 2026-08-31; two follow-up branches carry review fixes — `fix/lif-scalar-domain` then `fix/simd-fixture-and-record`, in that order) — slice-length contract enforced, overflow domain pinned (`DT_OVER_TAU_MAX`), floor-vs-truncate parking bias fixed (19 mV → ≤ 8 mV, cost +11.7 % of the vector path, scalar-controlled and identical across both benchmark runs; the raw spread is 8–13 %, machine load — corrected 2026-09-01 from "12–15 %", which was one run quoted as if it were both), every rounding-dependent number pinned exactly and reproducible from the tree (`cargo test -p neuralos-snn --features simd -- --ignored`); forks (b) exact ÷1000 and (c) half-cost recorded with triggers in the module doc; record in ISA § Close-out, evidence in `evidence/simd-hardening/` |
| Additional regression/property tests | The transmission-wire lesson: no unit test had ever exercised live transmission until session F |
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

Remaining in this phase: silicon (ESP32-C3 / HiFive), priority-gated
(board decided 2026-08-22 — merged-plan step 3).
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

1. R4 (harness extraction) — closed 2026-08-21
2. NIR import/export — **DONE through general graph assembly
   (2026-08-22, alpha.5)**; the R18 deferral family remains (readout
   edges / direct drive / encoder-only)
3. QEMU proof — **done 2026-08-21** (both legs; `evidence/qemu-riscv-gate/`)
4. Readout benchmark — **RAN 2026-08-24/27, adjudicated
   PRE-REGISTRATION-UNDEFINED** (1/5 SEPARATED · 2 MIXED; Tier 3 not
   demonstrated, "rests evidenced" not claimed — ISA ruling +
   `evidence/step5-readout/`). Binding constraint is CORPUS LENGTH,
   not replicate count. **Outstanding: the lesion/graft positive
   control**, specified as this benchmark's first arm and never run —
   until it does, a negative here licenses nothing (the council's own
   claim), and the paper's "uncalibrated instrument" limitation stays
   open.
5. **ESP32-C3 bring-up — BOARD NOT ORDERED.** Decided 2026-08-22
   ("purchase within 1–2 weeks"); no purchase recorded since. It gates
   its own step and, via GUARD 3, gates outreach. Longest pole on the
   board, and it is a purchase, not a session.
6. Lock-free ports (A-extension targets only — re-scoped above) — the
   last Phase-1 remainder; **SIMD hardening DONE 2026-08-31** (ISA § Close-out)
7. Visualizer Phase-2 — the lab bench catches up to the substrate the
   bridge arc hardened; also the named instrument for any step-8 arm
