#!/usr/bin/env python3
"""The adjudication table as a figure (Sec 6) — the conjunct at a glance.

De-circulared (P2-W3): leaf (b) — the 8/10 null flip count — is
derived from the banked judge LOGS (each file sha-verified against
its evidence dir's SHA256SUMS before parsing), not README prose.

Leaf (a) stays pinned to the README ruling block (the banked record
— GUARD 2) and is cross-checked here against the in-tree dumps: the
base-side margin of record (+0.0091 at p3s1) came from the
session-h2-era judge binary's base-side dump, which was NOT banked.
Re-deriving |Δmargin| at p3s1 with the in-tree session-f-judge base
(its step-1 margin is +0.0711) shifts H2's and the null-max values
by exactly +0.0620 each (the recorded cross-build 4th-decimal
variation class, r4-closeout) — the ordering and the (a) FAIL
verdict are base-side-invariant. The figure renders the BANKED
numbers; the cross-check asserts the invariance.
"""
import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"
README = EV / "session-i-primary" / "README.md"
DUMP = re.compile(r"NEURALOS_DUMP step=(\d+) n_out=\d+: (.*)")


def sha_ok(path: Path, session_dir: Path) -> None:
    sums = {}
    for line in (session_dir / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, _, name = line.partition("  ")
            sums[name.strip().lstrip("./")] = h
    rel = str(path.relative_to(session_dir))
    want = sums.get(rel)
    if want is None:
        raise SystemExit(f"no banked sha for {rel}")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"SHA MISMATCH {rel}: {got} != {want}")


def read_log(session_dir: Path, sub: str, prompt: str) -> str:
    d = session_dir / sub if sub else session_dir
    f = d / f"{prompt}_run1.log"
    sha_ok(f, session_dir)
    t = f.read_text()
    assert t.strip(), f"void log {f}"
    return t


def parse_err(path: Path) -> dict:
    steps = {}
    for line in path.read_text(errors="replace").splitlines():
        m = DUMP.match(line)
        if m:
            steps[int(m.group(1))] = [
                (int(a), float(b.replace(",", ".")))
                for a, b in re.findall(r"(\d+):(-?[\d,]+)", m.group(2))]
    return steps


# --- leaf (b): 8/10 derived from the logs, asserted to the bank -------
base_p3 = read_log(EV / "session-f-judge", "", "p3")
h2_p3 = read_log(EV / "session-h2", "", "p3")
flips = h2_dest = 0
for i in range(1, 11):
    t = read_log(EV / "session-i-primary", f"null-d{i}", "p3")
    if t != base_p3:
        flips += 1
        if t == h2_p3:
            h2_dest += 1
print(f"parsed from logs: p3 dose flips {flips}/10, of which H2's "
      f"exact destination {h2_dest}")
assert flips == 8 and h2_dest == 8, "leaf (b) does not reproduce"

# --- leaf (a): banked pin + in-tree verdict-invariance cross-check ----
text = README.read_text()
m_a = re.search(r"\(a\) FAIL — ([\d.]+) < ([\d.]+)", text)
assert m_a, "banked (a) ruling not found in README"
BANKED_H2, BANKED_NULLMAX = float(m_a.group(1)), float(m_a.group(2))
print(f"banked (a): |Δmargin| H2 {BANKED_H2} < null-max {BANKED_NULLMAX}")

base_s1 = parse_err(EV / "session-f-judge" / "p3_run1.err")[1]
t1, t2 = base_s1[0][0], base_s1[1][0]
m_base = base_s1[0][1] - base_s1[1][1]


def dmargin(steps):
    d = dict(steps[1])
    return abs((d[t1] - d[t2]) - m_base)


h2_dm = dmargin(parse_err(EV / "session-h2" / "p3_run1.err"))
null_dms = {f"d{i}": dmargin(parse_err(
    EV / "session-i-primary" / f"null-d{i}" / "p3_run1.err"))
    for i in range(1, 11)}
null_max, null_arg = max(null_dms.values()), max(
    null_dms, key=null_dms.get)
off_h2, off_max = round(h2_dm - BANKED_H2, 4), \
    round(null_max - BANKED_NULLMAX, 4)
print(f"in-tree re-derivation (f-judge base, margin {m_base:+.4f}): "
      f"H2 {h2_dm:.4f}, null-max {null_max:.4f} ({null_arg})")
print(f"constant cross-build offset: H2 {off_h2:+.4f}, "
      f"null-max {off_max:+.4f}")
assert off_h2 == off_max == 0.0620, "offset no longer constant — investigate"
assert h2_dm < null_max and BANKED_H2 < BANKED_NULLMAX, \
    "(a) FAIL verdict not base-side-invariant"

# --- render -----------------------------------------------------------
rows = [
    ("(a)", "|$\\Delta$margin| at every knife-edge\n$>$ max over ALL nulls",
     f"H2 {BANKED_H2} vs null-max {BANKED_NULLMAX}", "FAIL"),
    ("(b)", "null flip-rate on knife-edge set\n$\\leq$ 1/10 (escalate at 0–1/10)",
     f"{flips}/10 nulls flip, all into\nH2's exact destination", "FAIL"),
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
