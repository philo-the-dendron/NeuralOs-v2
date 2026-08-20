#!/usr/bin/env python3
"""The null-ladder diagram (Sec 6) — rungs, seals, and the ruling flow.

Structure comes from the sealed decisions (ISA, 2026-08-19/20/21);
the two quantitative leaves (0.0798 vs 0.2254; 8/10) are parsed from
evidence/session-i-primary/README.md so the figure cannot drift from
the record. Prints what it parsed (numbers gate).
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "evidence" / "session-i-primary" / "README.md"

text = README.read_text()
m_a = re.search(r"\(a\) FAIL — ([\d.]+) < ([\d.]+)", text)
m_b = re.search(r"\(b\) FAIL — (\d+)/10", text)
assert m_a and m_b, "ruling leaves not found in README"
DMARGIN_H2, DMARGIN_NULL = m_a.group(1), m_a.group(2)
FLIPS = m_b.group(1)
print(f"parsed from README: (a) {DMARGIN_H2} < {DMARGIN_NULL} ; "
      f"(b) {FLIPS}/10")

fig, ax = plt.subplots(figsize=(9.2, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

BOX = dict(boxstyle="round,pad=0.45", lw=1.1)
def box(x, y, w, h, body, fc="#f5f5f5", ec="#333333", fs=8.3, weight="normal"):
    ax.text(x + w / 2, y + h / 2, body, ha="center", va="center",
            fontsize=fs, linespacing=1.35, fontweight=weight,
            bbox={**BOX, "facecolor": fc, "edgecolor": ec})

def arrow(x1, y1, x2, y2, style="-|>", color="#333333", lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, color=color, lw=lw,
                                 shrinkA=2, shrinkB=2))

# rung column (left/center)
box(4, 84, 40, 12, "RUNG 1 — margin census\n$\\theta=0.05$ on BASELINE margins\n$\\to$ knife-edge set", fc="#eef3fb")
box(4, 66, 40, 12, "RUNG 2 — sibling prompts\nrule-fixed before running\n(weekday rotations, month chain,\noff-circuit negatives) — evidence only", fc="#eef3fb")
box(4, 44, 40, 14, "RUNG 3 — PRIMARY FAMILY\n dose-matched $\\times$10 (exactly\n   87,119 cells, H2 terminal diff)\n value-flip $\\times$3 (reflected values)", fc="#eef3fb", weight="bold")
box(52, 44, 44, 14, "replaced rung — position shuffle:\nSTRUCTURALLY IMPOSSIBLE\n(Thm 1: unique feasible assignment;\ngreedy $\\equiv$ H2, caught by sha)", fc="#fdf0ef", ec="#b94a48")
box(4, 22, 40, 10, "STRESS ARM $\\approx$2$\\times$ dose\n(flips $\\neq$ cells; pre-mortem catch)\nreport-only — never adjudicates", fc="#f2f2f2", ec="#888888")

# seals (right)
box(52, 78, 44, 18, "THE THREE SEALS (before tables read)\n"
    "1 void = mechanical evidence only; re-run once\n"
    "2 p4 bands: 0–1 thr. / 2–4 MIXED / $\\geq$5 content\n"
    "3 language pre-weakened, both branches", fc="#eff7ef", ec="#3d7a3d")

# conjunct + ruling
box(4, 2, 92, 14,
    f"PRE-REGISTERED CONJUNCT — survive ONLY IF ALL HOLD:\n"
    f"(a) $|\\Delta$margin$|$ exceeds max of ALL nulls  —  FAIL: {DMARGIN_H2} < {DMARGIN_NULL}      "
    f"(b) null knife-edge flips $\\leq$ 1/10  —  FAIL: {FLIPS}/10\n"
    f"(c) dose curve outside null band  —  moot      "
    f"(d) weekday-concentrated  —  MIXED (seal 2)",
    fc="#fbf6e9", ec="#8a6d2f", weight="bold")
box(58, 22, 38, 10, "RULING (zero discretionary calls)\nBRANCH B: UNATTRIBUTED\nPERTURBATION", fc="#ffffff", ec="#000000", weight="bold")

arrow(24, 84, 24, 78)
arrow(24, 66, 24, 58)
arrow(24, 58, 24, 32)
arrow(44, 51, 52, 51, style="-|>", color="#b94a48")
arrow(74, 78, 74, 60)
arrow(74, 60, 60, 32, color="#b94a48", lw=1.0)
arrow(50, 9, 58, 22)
arrow(24, 32, 24, 16)

ax.set_title("The null ladder: rungs sealed before the tables were read",
             fontsize=11, pad=10)
fig.tight_layout()
out = Path(__file__).parent / "ladder.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
