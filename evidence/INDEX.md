# Evidence Index

> Raw judge/experiment logs backing the bridge record. Verdicts and
> decisions live in `ISA.md`; the distilled narrative in
> `docs/RESEARCH_LOG.md`; the article in `paper/`. The judge is
> rebuildable: `bash tools/build_fork.sh` (pin `9ca265a`, dump patch
> in `tools/neuralos_dump.patch`).

| Directory / file | Session | Claim it backs | Key files |
|---|---|---|---|
| `corpus_readme_pinned.txt` | E stage 0 | The sha-pinned KLD corpus (12,148 bytes / 4,411 tokens; first 2,000 driven) | — |
| `session-f-judge/` | F (2026-08-19) | Loop closure on the live-wire substrate: 60/60 judged steps moved, max \|Δ\| 0.42, 0/60 argmax flips, continuations byte-identical; run1 == run2 | `p*_run{1,2}.{log,err}` (dumps = `NEURALOS_DUMP` lines), `README.md` |
| `session-h-invivo/` | H (2026-08-19) | In-vivo gate: T1 PASS (G0 tables, clamp caveat 69.8%, class-dissolution finding); export 67,309 cells / 40,126 code bytes; the p3 continuation change, double-run | `invivo.log`, `invivo_export.log`, `p3_run{1,2}.{log,err}` |
| `session-h2/` | H2 (2026-08-19) | Corrected-corpus rerun: P1' PASS, P3' split (1/12 argmax flip), 2nd continuation change | per-prompt run pairs, `README.md` |
| `session-i-primary/` | I (2026-08-20) | **The adjudication.** Dose-matched nulls ×10 + flow-shuffle ×3, all S2-clean. Conjunct: (a) FAIL 0.0798 < 0.2254 · (b) FAIL 8/10 ≫ 1/10 · (c) moot · (d) MIXED band → **BRANCH B: unattributed perturbation** (ISC-84) | `null-d1..d10/`, `null-f1..f3/`, `README.md` (the ruling table) |
| `session-i-stress/` | I (2026-08-20) | Stress arm (report-only): ~2× dose nulls are LOUD — p3 9/10, p4 9/10 (new flip family) — the p3 knife-edge is fragile to perturbation magnitude | `null-r1..r10/`, `README.md` |
| `r4-baselines/` | R4(i) (2026-08-20) | Pre-refactor re-pin baselines for the frozen example family (the R4(iv) contract); judge p0 leg closed by r4-closeout | `README.md` (the protocol + verdict table), `*_run{1,2}.log` |
| `r4-closeout/` | R4(iii/iv) (2026-08-21) | **R4 closed.** Leg-3 re-pins: null_patches 13/13 byte-identical; judge p0 0/12 flips max \|Δ\| +0.4207 exact; stale H1 invivo bar root-caused; H2 re-pin **byte-identical** (21,065.6 s, export tier by-design not run) | `README.md`, `h2_invivo_r4iii.log`, `p0_{base,loop}_run{1,2}.{log,err}`, `null_*`, `h1_invivo_r4iii.log` |
| `nir-hdf5-gate/` | NIR slice 2 (2026-08-21) | **The HDF5 evidence gate.** Reference-written `.nir` read end-to-end in pure Rust (5/5: exact frozen quanta cross-container · fires 9/100@6 · lzf censused out named · export read-back semantically identical · JSON byte-stability untouched); interop leg: the reference's own `read()` loads our export (weights ≤ scale/2) | `README.md` (rebuild), `gate.log`, `verify.log`, `SHA256SUMS` |
| `nir-assembly-gate/` | NIR general assembly (2026-08-22) | **The general four-kind graph gate.** Any reference-emitted Input/Linear/LIF/Output graph assembles and fires (6/6: branch first-spike pins `[3,14,1,3]` · merge summed-fan-in fires where single stalls (step 52 exact) · recurrent D1 pulse assert (−6990 = +10 quanta) · L→L fusion = once-quantized f64 product · 10 named rejections · frozen chain byte-identical through both builders); dynamics are named substrate conventions, no numeric-parity claim | `README.md` (rebuild), `gate.log`, `SHA256SUMS` |
| `qemu-riscv-gate/` | QEMU proof (2026-08-21) | **The riscv64gc no_std proof, both postures.** Leg A bare-metal none-elf: 175/175 cited checks, exit 0 (harness: `proofs/qemu-riscv-leg-a/`). Leg B user-mode musl: the REAL full suite on riscv64gc — 195/195 incl. the transmission trio + both Leg-C pins, exit 0, ~14 s | `README.md` (corrected pre-flight + rebuild), `leg-a.log`, `leg-b.log`, `SHA256SUMS` |

Seeds of record (pinned as constants in the frozen examples): census-
matched control `0x5EED_C0DE_0000_0002` (Fisher–Yates); primary dose
nulls `0xD05E_0000_0000_0001 XOR seed`, seed = 1..10.

Mechanical summary line of record: 1.7B-Q1_0 **NO 3/5** · 4B-Q1_0
**NO 4/5** · 4B-Q2_0 **YES 5/5** — all fork-attributed; hybrid seam
**ADAPTS**; loop **CLOSED** as capability; final downstream ruling
**unattributed perturbation**.
