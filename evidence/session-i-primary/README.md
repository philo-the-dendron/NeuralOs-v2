# Session I — PRIMARY family adjudication (the sealed rule's ruling)

Zero voids (seal 1 moot); denominators clean: dose ×10, value-flip ×3.
Flip table (continuation vs baseline):
- p0: H2 identical · dose 0/10 · flip 0/3
- p1: H2 identical · dose 0/10 · flip 0/3
- p2: H2 CHANGED (odd chain) · dose 4/10 · flip 3/3
  destinations: d3 "…fifteen seventeen twenty one" (a DIFFERENT coherent
  chain); d4/d6/d8 "sum of" (the dead-wire-era basin)
- p3: H2 CHANGED · dose 8/10 · flip 3/3 — all 8 dose flips land H2's
  EXACT destination (" Thursday" + 10 newlines): the basin is generic
- p4: H2 identical · dose 4/10 · flip 3/3 — all 4 land the stress
  basin ("United States is Washington"): noise reaches MORE doors
Δmargin at p3s1: H2 −0.0798 vs dose nulls max |−0.2254| (dose-6) —
H2 does NOT exceed the null family.

CONJUNCT RULING (mechanical, per the sealed rule):
(a) FAIL — 0.0798 < 0.2254 · (b) FAIL — 8/10 ≫ 1/10 bar (escalation
moot) · (c) moot · (d) p4 4/10 = MIXED band (seal 2: no claim).
→ BRANCH B: unattributed perturbation (not "degradation" — d3's
coherent chain kills the umbrella). The reach claim dies with (a).
Survives: G0/substrate tier, loop machinery, transmission fix,
mechanism work, the L1–L4 seam, and the methodology itself.

## Rebuild

```bash
# 1) Null family generation (HEAD tool, deterministic seeds; needs the
#    base + H2 terminal; 13/13 byte-regeneration already proven once in
#    evidence/r4-closeout/):
cargo run -p neuralos-rt --release --example null_patches
# → models/null-dose-1..10.gguf + models/null-flip-1..3.gguf

# 2) Judge chains (fork pinned at 9ca265a; the frozen five per
#    evidence/session-h2/README.md § Rebuild). One run per null per
#    prompt (single-run protocol — determinism established across the
#    family by protocol), e.g. for dose-3 / p3:
bash tools/build_fork.sh
NEURALOS_DUMP=1 fork-build/llama.cpp/build/bin/llama-completion \
  -m models/null-dose-3.gguf -p 'Monday Tuesday Wednesday' -n 12 \
  --temp 0 --top-k 0 --top-p 1.0 --min-p 0.0 --seed 42 -no-cnv -c 512 -t 4 \
  > null-d3/p3_run1.log 2> null-d3/p3_run1.err
# …same shape for d1..d10 (dose) and f1..f3 (flip), p0..p4 each.
# Baseline side for the flip tables = the H2 judge legs (session-h2).

# 3) Margins via judge_delta, same as session-h2 § Rebuild step 3.
```
