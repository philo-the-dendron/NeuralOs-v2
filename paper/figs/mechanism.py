#!/usr/bin/env python3
"""The two-rectifiers mechanism figure (Sec 2/4) — why realized drift
potentiates while raw drift is net-negative.

Panel A (parsed): the counter waterfalls — synthetic-era live-wire D-2
(ISC-78, parsed from ISA.md) and the in-vivo H2 run
(evidence/session-h2/run.log). Both show the clamp-rectified shape:
raw net-negative -> clamp absorbs -> applied positive.

Panel B (transcribed, printed): the rectification-at-the-sticking-point
physics pinned by test (ISC-77): 300 uA drive sticks at -59 mV;
+12 uA pulse ratchets +1 mV; -12 uA pulse absorbed.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ISA = ROOT / "ISA.md"
H2LOG = ROOT / "evidence" / "session-h2" / "run.log"

# --- parse the synthetic-era decomposition from the ISA (ISC-78) -----
isa = ISA.read_text()
m = re.search(
    r"raw intra drift ([−\-][\d,]+) \(mean ([−\-][\d.]+)/syn.*?"
    r"clamp-absorbed ([−\-][\d,]+) · APPLIED\s*\+([\d,]+) \(mean \+([\d.]+)/syn",
    isa, re.DOTALL)
assert m, "ISC-78 counter decomposition not found in ISA.md"
def num(s):  # '−739,295' (U+2212) -> -739295
    return int(s.replace("−", "-").replace(",", ""))


RAW, RAW_M, ABSORBED, APPLIED, APPLIED_M = (
    num(m.group(1)), float(m.group(2).replace("−", "-")), num(m.group(3)),
    num(m.group(4)), float(m.group(5)))
print(f"synthetic-era (ISA ISC-78): raw {RAW} (mean {RAW_M}/syn) · "
      f"absorbed {ABSORBED} · applied +{APPLIED} (mean +{APPLIED_M}/syn)")
assert RAW < 0 and APPLIED > 0, "clamp-rectified shape violated"

# --- parse the H2 decomposition from the run log ----------------------
h2 = H2LOG.read_text()
m2 = re.search(r"raw drift intra (-\d+).*?absorbed intra (-\d+).*?"
               r"APPLIED intra \+(\d+)", h2)
assert m2, "H2 counter decomposition not found in run.log"
H2_RAW, H2_ABS, H2_APP = (int(m2.group(1)), int(m2.group(2)),
                          int(m2.group(3)))
print(f"H2 in-vivo (run.log): raw {H2_RAW} · absorbed {H2_ABS} · "
      f"applied +{H2_APP}")
assert H2_RAW < 0 and H2_APP > 0
# the waterfall must reconcile exactly: raw - absorbed = applied
assert RAW - ABSORBED == APPLIED and H2_RAW - H2_ABS == H2_APP, \
    "counter arithmetic does not reconcile"

# --- transcribed physics constants (ISC-77 pinned test), printed ------
STICK_MV, RATCHET_MV, PULSE_UA, DRIVE_UA = -59, +1, 12, 300
print(f"sticking point (transcribed, ISC-77): {DRIVE_UA} uA sticks at "
      f"{STICK_MV} mV; +{PULSE_UA} uA -> {RATCHET_MV:+d} mV; "
      f"-{PULSE_UA} uA absorbed")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                               gridspec_kw={"width_ratios": [1.5, 1]})

# Panel A: counter waterfalls in thousands (raw - absorbed = applied,
# asserted above — the figure itself carries the reconciliation)
runs = [("synthetic-era\n(live-wire D-2)", RAW, ABSORBED, APPLIED),
        ("in-vivo H2\n(the paper's run)", H2_RAW, H2_ABS, H2_APP)]
k = 1000.0
for i, (name, raw, absorbed, applied) in enumerate(runs):
    x = i * 1.0
    axA.bar(x, raw / k, 0.52, color="#b94a48",
            label="raw drift (net, pre-clamp)" if i == 0 else None)
    axA.bar(x, -absorbed / k, 0.52, bottom=raw / k, color="#e8b33c",
            label="clamp un-absorbs (−absorbed)" if i == 0 else None)
    axA.bar(x, applied / k, 0.52, color="#3d7a3d",
            label="APPLIED (realized)" if i == 0 else None)
    axA.text(x, applied / k + 28, f"+{applied:,}", ha="center",
             fontsize=7.6, color="#3d7a3d")
    axA.text(x, raw / k - 34, f"{raw:,}", ha="center", fontsize=7.6,
             color="#b94a48")
axA.axhline(0, color="k", lw=0.8)
axA.set_xticks([0, 1])
axA.set_xticklabels([r[0] for r in runs], fontsize=8.5)
axA.set_ylabel("cumulative plasticity delta (thousands)")
axA.set_ylim(-880, 260)
axA.legend(frameon=False, fontsize=7.6, loc="lower right")
axA.set_title("Rectifier 1 — the weight clamp\n"
              "raw net-negative; realized drift positive",
              fontsize=9)

# Panel B: sticking-point schematic
axB.set_xlim(-3.5, 3.5)
axB.set_ylim(-75, -50)
axB.axhline(-59, color="#3a6ea5", lw=1.4)
axB.text(-3.3, -58.2, "membrane sticks at $-59$ mV under 300 $\\mu$A",
         fontsize=8)
axB.annotate("", xy=(0, -58), xytext=(0, -59),
             arrowprops=dict(arrowstyle="-|>", color="#3d7a3d", lw=1.6))
axB.text(0.25, -58.6, "$+12\\,\\mu$A $\\to$ $+1$ mV ratchet",
         fontsize=8, color="#3d7a3d")
axB.annotate("", xy=(2, -59), xytext=(1.6, -59),
             arrowprops=dict(arrowstyle="-|>", color="#b94a48", lw=1.6))
axB.text(1.05, -61.8, "$-12\\,\\mu$A absorbed\n(no movement)",
         fontsize=8, color="#b94a48")
axB.set_ylabel("membrane (mV)")
axB.set_xticks([])
axB.set_title("Rectifier 2 — the climb barrier\n(asymmetric at the sticking point)",
              fontsize=9)

fig.tight_layout()
out = Path(__file__).parent / "mechanism.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
