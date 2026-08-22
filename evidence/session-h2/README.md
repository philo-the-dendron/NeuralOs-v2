# Session H2 — the corrected-corpus in-vivo run + judge

The H1 corpus infidelity corrected by re-run: true pinned corpus
(evidence/corpus_readme_pinned.txt, sha 18fb5452…), first 2000 of 4411
tokens, single truncated pass (no epochs, no wrap by construction).

run.log: the full H2 experiment — T1 ALL PASS (G0 41,555 > 30,724;
P1′ gap 10,831 = 98.9% of H1's 10,974, floor was 2,744), clamp 69.48%
(vs H1 69.80% — confound bounded side-by-side), dose checkpoints
61,210 → 71,381 → 80,391 → 87,119 cells (monotone), all S2-clean,
final export sha 71f2518a…

Judge (p0–p4 × 2, deterministic byte-identical): p3 STEERS AGAIN —
11/12 flips, " Thursday" + newline-run (knife-edge crossed: margin
+0.0091 → −0.0707) BUT |Δmargin| = 0.0798 < the pre-registered 0.11
bar — P3′ magnitude FAILS, recorded ("the corrected one-flip
language stands"). NEW: p2 flips 4/12 — "…twelve fifteen seventeen
seventeen eighteen" vs baseline "…thirteen fifteen fifteen
seventeen" — the second continuation change in the project's
record. p0/p1/p4 quiet (0 flips).

## Rebuild

```bash
# 1) Substrate leg — the H2 run + the in-vivo export family (~6–8 h;
#    run.log verdict lines re-pinned byte-identical in r4-closeout
#    via PLAIN mode, ~21,065 s, export tier deliberately not run):
cargo run -p neuralos-rt --release --example hybrid_invivo -- \
  models/Ternary-Bonsai-4B-Q2_0.gguf export
# → run.log numbers + models/Ternary-Bonsai-4B-Q2_0-invivo.gguf
#   (sha 71f2518a…) + ck{400,800,1200} checkpoints.

# 2) Judge legs (fork pinned at 9ca265a + the NEURALOS_DUMP patch):
bash tools/build_fork.sh
# The frozen five (p0/p3 shapes verified in-artifact; p1/p2/p4 per the
# ISC-41/55 frozen-set record): p0 '1 2 3 4 5 6 7' · p1 '10 11 12 13'
# · p2 'one two three four' · p3 'Monday Tuesday Wednesday'
# · p4 'The capital of France is'.
# Per prompt pX, ×2 runs (base side = models/Ternary-Bonsai-4B-Q2_0.gguf,
# candidate = the H2-patched file above):
NEURALOS_DUMP=1 fork-build/llama.cpp/build/bin/llama-completion \
  -m models/Ternary-Bonsai-4B-Q2_0-invivo.gguf -p '<pX>' -n 12 \
  --temp 0 --top-k 0 --top-p 1.0 --min-p 0.0 --seed 42 -no-cnv -c 512 -t 4 \
  > pX_run<k>.log 2> pX_run<k>.err

# 3) Margins: judge_delta over the dump pairs
#    (cargo run -p neuralos-rt --release --example judge_delta --
#     <base-side pX.err> <candidate pX_run1.err>).
```
