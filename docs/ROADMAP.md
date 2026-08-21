# NeuralOS v2 — Roadmap

> Execution order for the active repo.
> `docs/VISION.md` is the north star; this document is the shipping
> sequence. Re-baselined 2026-08-20 at the project autopsy — the
> pre-autopsy roadmap (and the session records it had absorbed) is in
> git history (`pre-autopsy-cleanup` tag) and `docs/RESEARCH_LOG.md`.

## Priority order

| # | Component | Status |
|---|---|---|
| **1** | `neuralos-snn` — `no_std` SNN substrate | Active spine. The 2026-08-08 near-term list (NIR, lock-free, SIMD hardening) was starved by the bridge arc and is **first-class again**. |
| **2** | `neuralos-app` — Slint visualizer / lab bench | Untouched since 2026-08-08; Phase-2 items re-opened. |
| **3** | RISC-V deployment proof | QEMU `riscv64gc` first, silicon after. Unstarted. |
| **4** | Paper track | The Branch B article (in `paper/`) — finish, gate, submit. Must not displace 1–3. |
| **5** | Bridge follow-ups | **Frozen record.** Reopening is the principal's call on the recorded forks. The one active bridge-adjacent task is R4 (harness extraction) below. |

## Current validated state

```bash
cargo check  --workspace --all-targets
cargo test   --workspace                          # offline; 288 executed green (3 app, 192 snn, 93 rt) + 5 rt model-gated #[ignore]
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
| R7 | Original roadmap work, in order (below) | ⬜ **open — R4 closed 2026-08-21** |

## Phase 1 — Substrate hardening (the starved list, now first)

| Work item | Why it matters |
|---|---|
| NIR import/export | Interop with snnTorch/SpikingJelly; the #1 ecosystem recommendation. **First move.** **Slices 1+2 LANDED 2026-08-21** (slice 1 in `neuralos-snn::nir` — JSON container, Input/Linear/LIF/Output, explicit quant records, reference-emitted fixtures + 4/4 format gate, schema pinned to `neuromorphs/NIR@7883c3c`; slice 2 in `neuralos-rt` behind the `hdf5` feature — `.nir` HDF5 read/write, pre-read filter census, reference-written fixtures, the 5/5 `nir_hdf5_gate` + reference-side interop; populations + the structured-entry seam landed with slice 1's phases). Remaining: general multi-node graph assembly (arbitrary topologies beyond the canonical chain). |
| Lock-free ports from v0.1 archive | Throughput and future concurrency experiments |
| SIMD follow-up / hardening | `simd.rs` untouched since 2026-08-07; keep the performance path honest |
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

1. QEMU `riscv64gc`, reproducible path
2. `neuralos-snn` executing in that environment
3. Documentation of what is truly `no_std`-portable vs host-only

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

1. R4 (harness extraction) — closes the crust mechanism for good
2. NIR import/export — smallest highest-value spine item
3. QEMU proof — makes the `no_std` claim concrete
4. Visualizer Phase-2 — the lab bench catches up to the substrate the
   bridge arc hardened
