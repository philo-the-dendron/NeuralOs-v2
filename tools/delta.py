#!/usr/bin/env python3
# Mechanical delta table: baseline vs patched NEURALOS_DUMP lines.
# Parses "id:logit" pairs (comma decimal), compares per step:
# argmax id (both), flip?, top-10 id overlap, max |delta| over shared ids.
import sys, re

def parse(path):
    steps = {}
    for line in open(path):
        m = re.match(r'NEURALOS_DUMP step=(\d+)', line)
        if not m: continue
        s = int(m.group(1))
        pairs = re.findall(r'(\d+):(-?\d+,\d+)', line)
        steps[s] = {int(i): float(l.replace(',', '.')) for i, l in pairs}
    return steps

b, p = parse(sys.argv[1]), parse(sys.argv[2])
assert b.keys() == p.keys(), "step sets differ"
flips = 0; overlaps = []; deltas = []; first_changes = []
for s in sorted(b):
    ba, pa = max(b[s], key=b[s].get), max(p[s], key=p[s].get)
    if ba != pa: flips += 1
    ov = len(set(b[s]) & set(p[s])); overlaps.append(ov)
    ds = [abs(p[s][t] - b[s][t]) for t in set(b[s]) & set(p[s])]
    deltas.append(max(ds) if ds else float('nan'))
    first_changes.append(abs(next(iter(sorted(p[s].items(), key=lambda kv: -kv[1]))) [1] - b[s][ba]) if pa in b[s] else float('nan'))
print(f"steps {len(b)} | argmax flips {flips}/{len(b)} | overlap min {min(overlaps)}/10 mean {sum(overlaps)/len(overlaps):.2f} | max|d| shared {max(deltas):+.4f} | mean|d| {sum(deltas)/len(deltas):.4f}")
for s in sorted(b):
    ba, pa = max(b[s], key=b[s].get), max(p[s], key=p[s].get)
    d = abs(p[s][ba] - b[s][ba]) if ba in p[s] else None
    flag = "FLIP" if ba != pa else "same"
    db = f"{d:+.4f}" if d is not None else "  n/a "
    print(f"  step {s:2d}: argmax {ba}->{pa} {flag} | top1logit base {b[s][ba]:+.4f} d {db}")
