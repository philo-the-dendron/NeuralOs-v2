#!/usr/bin/env python3
"""Flip-rate confidence intervals for the session-i tables (Sec 6).

Computed from the FROZEN judge logs — zero re-runs. Every file read
is first verified against its banked SHA256SUMS (evidence dirs of
record); the parsed counts are asserted equal to the README-recorded
flip tables (the banked verdict — GUARD 2: if an assertion fires,
the evidence moved; do not publish). Intervals are exact binomial
(Clopper–Pearson) 95% CIs, pure stdlib, deterministic.

The margin leaf (arm (a)) is deliberately NOT given an interval: the
per-null |Δmargin| values were never banked (only H2's 0.0798 and
the null-max 0.2254 of record), so no distribution exists to
interval-ize; the CIs here quantify the flip-rate families.

Output: figs/flip_cis.tex (committed; included from adjudication.tex).
"""
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"
PROMPTS = ["p0", "p1", "p2", "p3", "p4"]
ALPHA = 0.05

# banked flip tables of record (README.md, the ruling table)
BANKED = {
    "dose": {"p0": 0, "p1": 0, "p2": 4, "p3": 8, "p4": 4},
    "flip": {"p0": 0, "p1": 0, "p2": 3, "p3": 3, "p4": 3},
    "stress": {"p0": 0, "p1": 0, "p2": 3, "p3": 8, "p4": 9},
}
BANKED_STRESS_N = {"p0": 10, "p1": 10, "p2": 10, "p3": 9, "p4": 10}
BANKED_H2_CHANGED = {"p2", "p3"}
DOSE = [f"null-d{i}" for i in range(1, 11)]
FLIP = [f"null-f{i}" for i in range(1, 4)]
STRESS = [f"null-r{i}" for i in range(1, 11)]


def sha_ok(path: Path, session_dir: Path) -> None:
    """Verify path against the session dir's banked SHA256SUMS."""
    sums = {}
    for line in (session_dir / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, _, name = line.partition("  ")
            sums[name.strip().lstrip("./")] = h
    rel = str(path.relative_to(session_dir))
    want = sums.get(rel)
    if want is None:
        raise SystemExit(f"no banked sha for {rel} in {session_dir}/SHA256SUMS")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"SHA MISMATCH {rel}: {got} != {want}")


def continuation(rundir: Path, prompt: str, session_dir: Path):
    """(text | None) — None = void (missing/empty log, seal 1)."""
    f = rundir / f"{prompt}_run1.log"
    sha_ok(f, session_dir)
    t = f.read_text()
    return t if t.strip() else None


def counts(rundir: Path, members, baseline, session_dir):
    """{prompt: (flips, valid)} over one family of run dirs."""
    out = {}
    for p in PROMPTS:
        flips, valid = 0, 0
        for m in members:
            d = rundir / m if m else rundir
            t = continuation(d, p, session_dir)
            if t is None:
                continue  # seal-1 void: denominator adjusted
            valid += 1
            if t != baseline[p]:
                flips += 1
        out[p] = (flips, valid)
    return out


def _pmf_tail_ge(k, n, p):
    """P(X >= k) for X ~ Bin(n, p)."""
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j)
               for j in range(k, n + 1))


def _pmf_tail_le(k, n, p):
    """P(X <= k) for X ~ Bin(n, p)."""
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j)
               for j in range(0, k + 1))


def cp_lower(k, n, alpha=ALPHA):
    """Clopper–Pearson lower bound (bisection, deterministic).

    Solves P(X >= k | p) = alpha/2; the tail is increasing in p, so a
    tail above target means p is too large.
    """
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _pmf_tail_ge(k, n, mid) > alpha / 2:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def cp_upper(k, n, alpha=ALPHA):
    """Clopper–Pearson upper bound (bisection, deterministic)."""
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _pmf_tail_le(k, n, mid) < alpha / 2:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main() -> None:
    base = {p: continuation(EV / "session-f-judge", p,
                            EV / "session-f-judge") for p in PROMPTS}
    h2 = {p: continuation(EV / "session-h2", p, EV / "session-h2")
          for p in PROMPTS}
    h2_changed = {p for p in PROMPTS if h2[p] != base[p]}
    assert h2_changed == BANKED_H2_CHANGED, f"H2 footprint {h2_changed}"

    fam = {
        "dose": counts(EV / "session-i-primary", DOSE, base,
                       EV / "session-i-primary"),
        "flip": counts(EV / "session-i-primary", FLIP, base,
                       EV / "session-i-primary"),
        "stress": counts(EV / "session-i-stress", STRESS, base,
                         EV / "session-i-stress"),
    }
    for f, table in BANKED.items():
        for p in PROMPTS:
            k, n = fam[f][p]
            assert k == table[p], f"{f} {p}: parsed {k}/{n} != banked {table[p]}"
            if f == "stress":
                assert n == BANKED_STRESS_N[p], \
                    f"stress {p}: denominator {n} != banked {BANKED_STRESS_N[p]}"
            else:
                assert n == (10 if f == "dose" else 3), f"{f} {p}: n={n} unexpected"
    print("banked flip tables reproduced from the verified logs:")
    for f in ("dose", "flip", "stress"):
        row = " · ".join(f"{p} {fam[f][p][0]}/{fam[f][p][1]}" for p in PROMPTS)
        print(f"  {f:6s} {row}")

    def cell(k, n):
        lo, hi = cp_lower(k, n), cp_upper(k, n)
        # defining properties of the CP bounds (the numbers gate):
        if k > 0:
            assert abs(_pmf_tail_ge(k, n, lo) - ALPHA / 2) < 1e-6
        if k < n:
            assert abs(_pmf_tail_le(k, n, hi) - ALPHA / 2) < 1e-6
        return f"{k}/{n} [{lo:.2f}, {hi:.2f}]"

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & H2 (single draw) & dose $\times$10 & value-flip $\times$3 "
        r"& stress $\approx$2$\times$ (report-only) \\",
        r"\midrule",
    ]
    h2cell = lambda p: "changed" if p in h2_changed else "identical"
    for p in PROMPTS:
        d, fl, st = fam["dose"][p], fam["flip"][p], fam["stress"][p]
        lines.append(
            f"{p} & {h2cell(p)} & {cell(*d)} & {cell(*fl)} & {cell(*st)} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    # transcribed expected cells (the pinned output of record):
    assert cell(8, 10) == "8/10 [0.44, 0.97]", cell(8, 10)
    assert cell(4, 10) == "4/10 [0.12, 0.74]", cell(4, 10)
    assert cell(0, 10) == "0/10 [0.00, 0.31]", cell(0, 10)
    out = Path(__file__).parent / "flip_cis.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    # spot-assert the rule-of-three anchor: 0/10 upper ~= 0.31 at 95%
    u = cp_upper(0, 10)
    assert abs(u - (1 - (ALPHA / 2) ** 0.1)) < 1e-9, u
    print(f"rule-of-three check: 0/10 95% CI upper = {u:.4f}")


if __name__ == "__main__":
    main()
