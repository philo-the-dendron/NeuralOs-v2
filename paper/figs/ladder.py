#!/usr/bin/env python3
"""The null-ladder diagram (Sec 6) — rungs, seals, and the ruling flow.

Structure comes from the sealed decisions (ISA, 2026-08-19/20/21).
De-circulared (P2-W3): the quantitative leaf 8/10 is DERIVED from the
banked judge logs (sha-verified); the margin leaf (0.0798 vs 0.2254)
is pinned to the README ruling block — the banked record — because
the base-side dump that produced it was never banked (the in-tree
re-derivation and its verdict-invariance cross-check live in
adjudication_table.py). Prints what it parsed (numbers gate).
"""
import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"
README = EV / "session-i-primary" / "README.md"


def sha_ok(session_dir: Path, rel: str) -> None:
    sums = {}
    for line in (session_dir / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, _, name = line.partition("  ")
            sums[name.strip().lstrip("./")] = h
    want = sums.get(rel)
    if want is None:
        raise SystemExit(f"no banked sha for {rel}")
    got = hashlib.sha256((session_dir / rel).read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"SHA MISMATCH {rel}: {got} != {want}")


text = README.read_text()
m_a = re.search(r"\(a\) FAIL — ([\d.]+) < ([\d.]+)", text)
assert m_a, "banked (a) ruling not found in README"
DMARGIN_H2, DMARGIN_NULL = m_a.group(1), m_a.group(2)

sha_ok(EV / "session-f-judge", "p3_run1.log")
base_p3 = (EV / "session-f-judge" / "p3_run1.log").read_text()
flips = 0
for i in range(1, 11):
    sha_ok(EV / "session-i-primary", f"null-d{i}/p3_run1.log")
    f = EV / "session-i-primary" / f"null-d{i}" / "p3_run1.log"
    if f.read_text() != base_p3:
        flips += 1
assert flips == 8, f"leaf (b): {flips}/10 != 8/10"
FLIPS = str(flips)
print(f"parsed: (a) banked {DMARGIN_H2} < {DMARGIN_NULL} ; "
      f"(b) derived from logs {FLIPS}/10")

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
