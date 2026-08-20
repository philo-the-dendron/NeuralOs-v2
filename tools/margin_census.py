#!/usr/bin/env python3
"""Margin census (null-ladder rung 1): top1-vs-top2 margins for every
prompt×step of a NEURALOS_DUMP set. Knife-edge set = margins < THETA
(pre-registered θ = 0.05). Usage: margin_census.py <p0.err> [...]"""
import sys, re

THETA = 0.05

def parse(path):
    steps = {}
    for line in open(path):
        m = re.match(r'NEURALOS_DUMP step=(\d+)', line)
        if not m:
            continue
        s = int(m.group(1))
        pairs = re.findall(r'(\d+):(-?\d+,\d+)', line)
        steps[s] = {int(i): float(l.replace(',', '.')) for i, l in pairs}
    return steps

print(f"{'file':>28} {'step':>4} {'top1':>6} {'top2':>6} {'margin':>8}  knife?")
knives = []
for path in sys.argv[1:]:
    tag = path.split('/')[-1].replace('_run1.err', '').replace('_run2.err', '')
    steps = parse(path)
    for s in sorted(steps):
        vals = sorted(steps[s].values(), reverse=True)
        if len(vals) < 2:
            continue
        margin = vals[0] - vals[1]
        is_knife = margin < THETA
        if is_knife:
            knives.append((tag, s, margin))
        # print only the small-margin tail (readable): margins < 0.5
        if margin < 0.5:
            top1 = max(steps[s], key=steps[s].get)
            top2_id = max((i for i in steps[s] if i != top1), key=lambda i: steps[s][i])
            print(f"{tag:>28} {s:>4} {top1:>6} {top2_id:>6} {margin:>+8.4f}  {'◄ KNIFE' if is_knife else ''}")
print()
print(f"knife-edge set (margin < {THETA}): {len(knives)} entries across all files")
for tag, s, m in knives:
    print(f"  {tag} step {s}: {m:+.4f}")
