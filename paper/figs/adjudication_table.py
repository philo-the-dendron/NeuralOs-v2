#!/usr/bin/env python3
"""The adjudication table as a figure (Sec 6) — the conjunct at a glance.

Values parsed from evidence/session-i-primary/README.md (ruling block)
and cross-checked against the flip table in the same file.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "evidence" / "session-i-primary" / "README.md"
text = README.read_text()

m_a = re.search(r"\(a\) FAIL — ([\d.]+) < ([\d.]+)", text)
m_b = re.search(r"\(b\) FAIL — (\d+)/10", text)
m_p3 = re.search(r"p3: H2 CHANGED · dose (\d+)/10", text)
assert m_a and m_b and m_p3
print(f"parsed: (a) {m_a.group(1)} < {m_a.group(2)} ; (b) {m_b.group(1)}/10 ; "
      f"p3 dose flips {m_p3.group(1)}/10")
assert m_p3.group(1) == m_b.group(1) == "8"

rows = [
    ("(a)", "|$\\Delta$margin| at every knife-edge\n$>$ max over ALL nulls",
     f"H2 {m_a.group(1)} vs null-max {m_a.group(2)}", "FAIL"),
    ("(b)", "null flip-rate on knife-edge set\n$\\leq$ 1/10 (escalate at 0–1/10)",
     f"{m_b.group(1)}/10 nulls flip, all into\nH2's exact destination", "FAIL"),
    ("(c)", "in-vivo dose curve outside the\nnull band at every checkpoint",
     "(a)+(b) decisive — moot", "moot"),
    ("(d)", "effects concentrate on weekday set\nvs off-circuit negatives",
     "p4 at 4/10 = MIXED band (seal 2)", "MIXED"),
]

fig, ax = plt.subplots(figsize=(9.0, 3.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

headers = ["", "criterion (pre-registered)", "observed (primary family)", "verdict"]
xs = [0.4, 1.15, 4.9, 8.6]
for x, h in zip(xs, headers):
    ax.text(x, 9.1, h, fontsize=8.6, fontweight="bold", va="center")

colors = {"FAIL": "#b94a48", "moot": "#999999", "MIXED": "#c9862b"}
y = 7.4
for tag, crit, obs, verdict in rows:
    ax.text(xs[0], y, tag, fontsize=9.5, fontweight="bold", va="center")
    ax.text(xs[1], y, crit, fontsize=8.2, va="center", linespacing=1.4)
    ax.text(xs[2], y, obs, fontsize=8.2, va="center", linespacing=1.4)
    ax.text(9.15, y, verdict, fontsize=9.5, fontweight="bold",
            va="center", ha="center", color=colors[verdict])
    ax.plot([0.3, 9.8], [y - 1.15, y - 1.15], color="#cccccc", lw=0.6)
    y -= 2.3

ax.text(0.4, 0.4, "ANY arm fails $\\to$ capability language demoted. "
        "Ruling: unattributed perturbation (Branch B, by rule).",
        fontsize=8.6, style="italic", va="center")
ax.set_title("The sealed conjunct and its verdicts",
             fontsize=10.5, pad=8)
fig.tight_layout()
out = Path(__file__).parent / "adjudication_table.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
