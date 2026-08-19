# Session F loop-closure judge evidence (live-wire substrate)

The fork judge (llama-completion @ PrismML-Eng/llama.cpp 9ca265a +
NEURALOS_DUMP patch, greedy-forced flags per the session-E protocol)
on the Hebbian-era patched file `models/Ternary-Bonsai-4B-Q2_0-loop.gguf`
(sha256 24ffe5f3…, produced by `hybrid_loop` on the session-F fixed
substrate), double-run per prompt:

- p0–p4 × run1/run2: `.log` = stdout (continuations), `.err` = the
  NEURALOS_DUMP step-0..11 top-10 lines (raw pre-sampler logits).
- Determinism: run1 == run2 byte-identical, all five prompts.
- Baseline for comparison: the session-E banked baseline (identical
  file, pre-patch) — recorded in the ISA (ISC-67); its dumps live in
  /tmp/opencode/se/baseline/ (cache — the verdict numbers are in the
  ISA and commit faa8020's message).

Mechanical summary (delta.py, now at tools/delta.py): 60/60 steps
moved, max |Δ| 0.42 (p0), mean |Δ| 0.057–0.151, 0/60 argmax flips,
continuations byte-identical to baseline on all five prompts.
