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

Seeds of record (pinned as constants in the frozen examples): census-
matched control `0x5EED_C0DE_0000_0002` (Fisher–Yates); primary dose
nulls `0xD05E_0000_0000_0001 XOR seed`, seed = 1..10.

Mechanical summary line of record: 1.7B-Q1_0 **NO 3/5** · 4B-Q1_0
**NO 4/5** · 4B-Q2_0 **YES 5/5** — all fork-attributed; hybrid seam
**ADAPTS**; loop **CLOSED** as capability; final downstream ruling
**unattributed perturbation**.
