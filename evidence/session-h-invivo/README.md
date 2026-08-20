# Session H — in-vivo gate judge evidence

The decisive artifact set: p3 ("Monday Tuesday Wednesday") continuation
divergence under the in-vivo-adapted file (sha adcc7feabc82…), double-run.
- p3_run{1,2}.{log,err}: the continuation change (" Thursday" then
  newline-run vs baseline " Thursday04/05/2018") + the 11/12 argmax-flip
  dump lines. run1 == run2 byte-identical.
- invivo.log: the full Tier-1 run (G0 PASS tables, clamp caveat 69.8%,
  class-dissolution finding, counters).
- invivo_export.log: the export (67,309 cells, 40,126 code bytes, scales 0).
Baseline: evidence/session-f-judge + ISA ISC-67/82. Full p0–p4 × 2 runs
were a /tmp cache, since expired — verdicts live in the ISA;
regenerate the judge with `bash tools/build_fork.sh` + the flags above.
