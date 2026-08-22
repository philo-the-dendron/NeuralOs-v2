# Figures — scripts, not hand-drawn

Every figure is a matplotlib script that reads its data from the
in-tree evidence files (`evidence/…`) or the ISA-recorded tables;
every judge log is sha-verified against its dir's banked
`SHA256SUMS` before parsing. Nothing is hand-plotted; regenerating
a figure re-derives its numbers from the same logs the prose cites.
Where a leaf's underlying dump was never banked (the (a) margin
base-side), the script pins the README ruling number — the banked
record — and cross-checks the verdict direction from the in-tree
dumps. `SOURCE_DATE_EPOCH` is fixed in the Makefile, so `make figs`
regenerates byte-stably.

## Regenerate

```sh
make figs        # from paper/ — uses $FIGPYTHON if set
# or directly:
python3 figs/basins.py          # writes figs/basins.pdf
```

matplotlib lives in a venv on the drafting machine
(`python3 -m venv v && v/bin/pip install matplotlib`); any
Python >= 3.9 with matplotlib >= 3.5 produces byte-similar output.
Generated PDFs are committed so the paper builds LaTeX-only.

## Figure inventory (docs/PAPER_OUTLINE.md figures plan)

| Script | Data source | Shows |
|---|---|---|
| `basins.py` | judge `p*_run1.log` across `session-f-judge`, `session-h2`, `session-i-primary/null-*`, `session-i-stress/null-*` (sha-verified) | Destination basins: 8/10 dose nulls land H2's exact p3 destination; p4's flips land a different basin (the paper's highest-leverage visual); annotations derived from the tallies |
| `ladder.py` | the sealed rule (ISA Decisions 2026-08-19/20); leaf (b) from the sha-verified logs, leaf (a) pinned to the README ruling | Null-ladder flow: families → seals → conjunct → ruling |
| `adjudication_table.py` | leaf (b) from the sha-verified logs; leaf (a) README-pinned + in-tree verdict-invariance cross-check (the unbanked base-side gap, constant +0.0620 offset) | The conjunct (a)–(d) rows and the FAIL marks |
| `flip_cis.py` | judge `p*_run1.log` across `session-f-judge`, `session-h2`, `session-i-primary/null-*`, `session-i-stress/null-*` + each dir's `SHA256SUMS` | Flip-rate table with exact binomial (Clopper–Pearson) 95% CIs; sha-verifies every log before parsing, asserts the banked counts; writes `flip_cis.tex` (included from Sec 6) |
| `mechanism.py` | ISA-recorded mechanism decomposition | The two rectifiers (pairing-selective → clamp-rectified) |

Scripts print the numbers they parsed before writing the PDF —
cross-check the stdout against the README tables when regenerating
(the numbers gate).
