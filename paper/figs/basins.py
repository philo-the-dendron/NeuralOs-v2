#!/usr/bin/env python3
"""The basins figure (Sec 6) — destination basins of the null families.

Reads the actual judge continuation logs from evidence/ (H2, dose x10,
value-flip x3, stress arm) plus the baseline (session-f-judge), bucket
each draw by destination, and render the grid: rows = prompts,
columns = draws grouped by family. Colors: grey = baseline-identical,
one accent = H2's exact destination, other hues = other destinations.

Numbers gate: the script PRINTS the parsed tallies and ASSERTS the
README-recorded counts (evidence/session-i-primary/README.md). If an
assertion fires, the evidence moved — do not publish.
"""
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"

PROMPTS = ["p0", "p1", "p2", "p3", "p4"]
BASELINE_DIR = EV / "session-f-judge"          # baseline bank (ISC-67)
H2_DIR = EV / "session-h2"
DOSE = [f"null-d{i}" for i in range(1, 11)]     # primary family, dose x10
FLIP = [f"null-f{i}" for i in range(1, 4)]      # primary family, value-flip x3
STRESS = [f"null-r{i}" for i in range(1, 11)]   # stress arm, report-only
PRIMARY = EV / "session-i-primary"
STRESS_DIR = EV / "session-i-stress"


def read_continuation(rundir: Path, prompt: str, run: str = "run1") -> str:
    f = rundir / f"{prompt}_{run}.log"
    return f.read_text()


def dest_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def main() -> None:
    # --- parse ---------------------------------------------------------
    data = {}  # data[family][prompt] = list of (label, dest_id, changed)
    baseline = {p: read_continuation(BASELINE_DIR, p) for p in PROMPTS}
    h2 = {p: read_continuation(H2_DIR, p) for p in PROMPTS}

    def fam(name, rundir, members, run="run1"):
        rows = {}
        for p in PROMPTS:
            draws = []
            for m in members:
                try:
                    t = read_continuation(rundir / m, p, run)
                except FileNotFoundError:  # missing = void (seal 1)
                    draws.append((m, None, False))
                    continue
                if not t.strip():  # empty log = mechanical void (seal 1)
                    draws.append((m, None, False))
                    continue
                draws.append((m, dest_id(t), t != baseline[p]))
            rows[p] = draws
        data[name] = rows

    fam("H2\n(in-vivo)", H2_DIR, [""])
    fam("dose-matched x10", PRIMARY, DOSE)
    fam("value-flip x3", PRIMARY, FLIP)
    fam("stress 2x (report-only)", STRESS_DIR, STRESS)

    # --- tallies (printed = the numbers gate) ---------------------------
    def tally(name):
        for p in PROMPTS:
            draws = data[name][p]
            changed = [d for d in draws if d[2]]
            h2d = h2[p]
            n_h2 = sum(1 for d in changed if d[1] == dest_id(h2d))
            voids = [d[0] for d in draws if d[1] is None]
            print(f"{name:24s} {p}: changed {len(changed)}/{len(draws)}"
                  f"  of-which-H2-dest {n_h2}  voids {voids}")

    for name in data:
        tally(name)

    # assertions: the README-recorded primary-family table
    d_p3 = data["dose-matched x10"]["p3"]
    assert sum(1 for d in d_p3 if d[2]) == 8, "p3 dose flips != 8/10"
    assert sum(1 for d in d_p3 if d[1] == dest_id(h2["p3"])) == 8, \
        "p3 dose flips not all in H2's exact destination"
    for p in ["p0", "p1"]:
        assert not any(d[2] for d in data["dose-matched x10"][p]), \
        "p0/p1 dose should be quiet"
    assert sum(1 for d in data["dose-matched x10"]["p2"] if d[2]) == 4
    assert sum(1 for d in data["dose-matched x10"]["p4"] if d[2]) == 4
    for p in ["p2", "p3", "p4"]:
        n = sum(1 for d in data["value-flip x3"][p] if d[2])
        assert n == 3, f"flip {p} != 3/3"
    for p in ["p0", "p1"]:
        assert not any(d[2] for d in data["value-flip x3"][p])
    # stress arm after seal-1 voids: p3 = 8/9 valid, p4 9/10, p2 3/10
    sp3 = data["stress 2x (report-only)"]["p3"]
    valid = [d for d in sp3 if d[1] is not None]
    assert len(valid) == 9 and sum(1 for d in valid if d[2]) == 8, \
        "stress p3 != 8/9 valid (r10 void)"
    print("assertions: primary-family table matches the README")

    # --- figure ----------------------------------------------------------
    fams = [("dose-matched x10", "dose nulls\n(exact 87,119 cells)", DOSE),
            ("value-flip x3", "value-flip nulls\n(reflected values)", FLIP),
            ("stress 2x (report-only)", "stress arm\n(~178k cells)", STRESS)]

    # color per destination, stable per prompt row
    h2c, basec, otherc = "#1f77b4", "#d0d0d0", None
    palette = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
               "#bcbd22", "#17becf"]

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ylabels = []
    for yi, p in enumerate(reversed(PROMPTS)):  # p0 top .. p4 bottom->? keep p0 top
        y = len(PROMPTS) - 1 - yi
        ylabels.append((y, p))
        x = 1.0  # H2 marker at left margin, then families
        h2_changed = h2[p] != baseline[p]
        # H2 single draw
        color = (h2c if h2_changed and p in ("p2", "p3")
                 and dest_id(h2[p]) == dest_id(h2[p]) else basec)
        # H2's own destination drawn in its accent only where it changed
        if h2_changed:
            color = h2c
        ax.add_patch(plt.Rectangle((0.0, y + 0.12), 0.7, 0.76,
                                   facecolor=color, edgecolor="k",
                                   lw=0.4))
        seen = {}
        for name, title, members in fams:
            x0 = {"dose-matched x10": 2.0,
                  "value-flip x3": 13.0,
                  "stress 2x (report-only)": 17.0}[name]
            draws = data[name][p]
            for i, (m, did, changed) in enumerate(draws):
                if did is None:  # void
                    c, hatch = "white", "///"
                elif not changed:
                    c, hatch = basec, None
                elif did == dest_id(h2[p]):
                    c, hatch = h2c, None
                else:
                    k = f"{p}:{did}"
                    if k not in seen:
                        seen[k] = palette[len(seen) % len(palette)]
                    c, hatch = seen[k], None
                ax.add_patch(plt.Rectangle((x0 + i * 1.0, y + 0.12),
                                           0.82, 0.76, facecolor=c,
                                           edgecolor="k", lw=0.4,
                                           hatch=hatch))
    # family separators + titles
    for x in (1.7, 12.7, 16.7, 27.2):
        ax.axvline(x, color="k", lw=0.8)
    titles = [(2.0 + 4.5, "dose nulls $\\times$10\n(exact dose: 87,119 cells)"),
              (13.0 + 1.5, "value-flips\n$\\times$3"),
              (17.0 + 4.5, "stress arm $\\approx$2$\\times$ dose\n(report-only)")]
    for x, t in titles:
        ax.text(x, 5.55, t, ha="center", va="bottom", fontsize=8.5)
    ax.text(0.35, 5.55, "H2\n(in-vivo\nadaptation)", ha="center",
            va="bottom", fontsize=8.5)

    ax.set_xlim(-0.3, 27.5)
    ax.set_ylim(0, 6.6)
    ax.set_yticks([y + 0.5 for y, _ in ylabels])
    ax.set_yticklabels([p.upper() for _, p in ylabels], fontsize=10)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # annotations: the two headline reads
    ax.annotate("8/10 land H2's EXACT destination\n(the basin is generic)",
                xy=(6.5, 0.9), xytext=(6.5, -1.05), ha="center",
                fontsize=8.5, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.annotate("noise reaches doors H2 did not:\n4/10 dose + 3/3 flips here,\nH2 opened 0",
                xy=(19.5, 1.4), xytext=(23.8, -1.0), ha="center",
                fontsize=8.5, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.7))

    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=basec, edgecolor="k", lw=0.4,
                       label="baseline-identical"),
        mpatches.Patch(facecolor=h2c, edgecolor="k", lw=0.4,
                       label="H2's exact destination"),
        mpatches.Patch(facecolor=palette[0], edgecolor="k", lw=0.4,
                       label="other destination (per color)"),
        mpatches.Patch(facecolor="white", edgecolor="k", lw=0.4,
                       hatch="///", label="void (seal 1)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=8, bbox_to_anchor=(0.0, 1.16), ncol=4)

    ax.set_title("Destination basins: equal-dose nulls land where the "
                 "in-vivo adaptation landed — and beyond", pad=46,
                 fontsize=11)
    fig.tight_layout()
    out = Path(__file__).parent / "basins.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
