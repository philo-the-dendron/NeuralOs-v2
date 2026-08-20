# Figures — scripts, not hand-drawn

Every figure is a matplotlib script that reads its data from the
in-tree evidence files (`evidence/…`) or the ISA-recorded tables.
Nothing is hand-plotted; regenerating a figure re-derives its numbers
from the same logs the prose cites.

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
| `basins.py` | `evidence/session-i-primary/null-*/p*_run1.log` + `README.md` table | Destination basins: 8/10 dose nulls land H2's exact p3 destination; p4's flips land a different basin (the paper's highest-leverage visual) |
| `ladder.py` | the sealed rule (ISA Decisions 2026-08-19/20) | Null-ladder flow: families → seals → conjunct → ruling |
| `adjudication_table.py` | `evidence/session-i-primary/README.md` | The conjunct (a)–(d) rows and the FAIL marks |
| `mechanism.py` | ISA-recorded mechanism decomposition | The two rectifiers (pairing-selective → clamp-rectified) |

Scripts print the numbers they parsed before writing the PDF —
cross-check the stdout against the README tables when regenerating
(the numbers gate).
