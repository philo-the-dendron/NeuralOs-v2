# Session I — STRESS ARM (report-only, per the v2 amendment)

The over-dosed census-transition nulls (~178k cells ≈ 2× the H2 patch),
judged once each (determinism established across the family by protocol).
Result: LOUD. p3 flips 9/10 (same " Thursday"+newline signature as H2),
p4 flips 9/10 (NEW flip family: "…capital of the United States is
Washington" vs baseline "Japan"), p2 3/10 ("…twelve fifteen seventeen…",
H2's own continuation), p0/p1 0/10.

Reading (report-only, per amendment): at 2× dose, i.i.d. noise of this
composition flips the p3 knife-edge easily — the knife-edge is fragile
to perturbation MAGNITUDE. The steers question therefore rests entirely
on the PRIMARY dose-matched family (87,119 cells exactly), whose judge
chain is queued/running. This table contextualizes; it adjudicates
nothing.

## Rebuild

```bash
# 1) The stress family was emitted by the V1 census-transition tool —
#    HEAD's null_patches builds the dose/flip families only:
git show 4007027:crates/neuralos-rt/examples/null_patches.rs
# (consult the v1 source; single-arg usage on the base Q2_0,
#  H2 census consts baked in — rebuild at that commit to regenerate
#  models/null-random-1..10.gguf, the ~178k-cell family.)

# 2) Judge chains (fork pinned at 9ca265a; the frozen five per
#    evidence/session-h2/README.md § Rebuild). One run per null per
#    prompt, e.g. null-random-1 / p3:
bash tools/build_fork.sh
NEURALOS_DUMP=1 fork-build/llama.cpp/build/bin/llama-completion \
  -m models/null-random-1.gguf -p 'Monday Tuesday Wednesday' -n 12 \
  --temp 0 --top-k 0 --top-p 1.0 --min-p 0.0 --seed 42 -no-cnv -c 512 -t 4 \
  > null-r1/p3_run1.log 2> null-r1/p3_run1.err
# …same shape for r1..r10, p0..p4 each.
```
