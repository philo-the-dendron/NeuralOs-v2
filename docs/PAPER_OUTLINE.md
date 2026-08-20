# PAPER_OUTLINE.md — NeuralOS v2, Branch B

> The working outline, assembled from the ISA's record. Every claim row
> cites its evidence. RedTeam attacks the claims table before prose.
> Language rule (seal 3 + adjudication): no "steers"/"editing"; the
> adaptation result is reported as UNATTRIBUTED PERTURBATION; the
> substrate tier keeps its measured claims (arrangement read, G0).

## Title (post-RedTeam)

**A Spiking Substrate That Reads and Round-Trips a Shipped Quantized
LLM's Weights** — the infrastructure claim, most defensible.
("Rewrites" cut — verb form of the banned "editing"; "no_std" is a
toolchain property, not a scientific claim. Method contingency title
held in reserve only if C1 ever falls.)

## Abstract skeleton (post-RedTeam: the negative is bullet 2)

- The substrate: no_std, i16 fixed-point SNN (crates.io, real-artifact
  test), ternary bridge (i2_s bit-exact, Q2_0 two-way).
- **The adjudication, up front:** the adaptation's downstream effects
  FAILED every pre-registered criterion for exceeding equal-dose random
  perturbation (p3 8/10 nulls into the same basin; Δmargin 0.0798 vs
  max-null 0.2254) — term of record: unattributed perturbation. The
  ladder (pre-registered null families) killed our own earlier
  capability reading before it could be published.
- The loop: import a real 4B Q2_0 LLM's tensor → STDP adaptation →
  byte-exact re-export → foreign runtime (llama.cpp) executes it; the
  patched file's behavior changes; a byte-identical control file rules
  out file-surgery artifacts.
- The substrate findings that stand: arrangement-sensitivity under
  synthetic and in-vivo drives (centi grid; sign-dominant drive caveat
  stated), the transmission-bug saga as falsification-driven discovery,
  the clamp-rectified mechanism decomposition.
- The method's novel rungs: the composition-specific null-feasibility
  theorem and the destination-basin analysis that caught what
  flip-counts misread.

## Section plan

1. **Introduction** — the novel intersection is L1∧L2: local plasticity
   writing a SHIPPED production LLM's weights (the field's own surveys
   flag it open, 2309.15942); L3/L4 presented as closure of the chain,
   not novelty. "First" as surveyed, boundaries stated, re-verified
   pre-submission.
2. **The substrate** — i16 hot path, VoltageResolution, the
   transmission fix (a1b) as a case study in falsification-driven
   debugging (two falsified pre-registered predictions → canary → fix
   → mechanism reversal).
3. **The format bridge** — layouts pinned from reference sources;
   real-artifact vectors; encode∘decode = byte identity on real bytes.
4. **The loop** — surgery, S1/S2 gates, control identity (the file
   surgery is a measured transparent transformation).
5. **In-vivo drive** — the model's own attn_norm(embedding) drives the
   substrate; G0 arrangement-vs-census PASS (T1); clamp caveat 69.8%
   (sign-dominant) stated in-text.
6. **The adjudication** — the ladder (primary/stress/flip families),
   the seals, the ruling; p2's FULL denominator (d3's coherent chain —
   the reason "degradation" is not the term of record — plus d4/d6/d8
   in the "sum of" basin, one line each; transcripts in appendix);
   H2's footprint sits inside null experience (an independent noise
   draw matches {p2,p3} at ≈19%); value-flips opening p4 3/3 where H2
   opened 0 = hypothesis-generating only. **The averted-false-positive
   frame: our own ISC-82 recorded "STEERING demonstrated"
   pre-adjudication — flip-counts alone would have published a false
   positive; the ladder caught the authors' first reading.** Figures:
   the basins figure (8/10 nulls into H2's exact destination; p4's
   different basin) — the paper's highest-leverage visual, zero new
   compute.
7. **Discussion** — what would have counted as content (the conjunct);
   why the negative is informative (basins, dose, entanglement
   theorem); the honest boundary of the capability claim.
8. **Related work** — the 16-entry matrix verbatim.
9. **Limitations** — the pre-mortem's sentences, both branches' armor.

## The claims table (RedTeam's target)

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | First demonstrated L1–L4 round-trip, as surveyed (boundaries stated; re-verified pre-submission); downstream effect adjudicated null | PAPER_NOVELTY matrix + loop evidence + ISA adjudication | scoped — scope triple CUT ("verified-attribution" was self-refuting; ternary/no_std excluded zero matrix entries) |
| C2 | The substrate reads model arrangement — H(i,c)>H(i,z) on the CENTI grid under both drives (gap 1.35–1.63×); mV shows rate/Hamming sensitivity only; one slice, one model, one seed; sign-dominant drive | ISC-76/81/83, clamp caveats ISA 1286-1292 | measured, caveated in-text |
| C3 | Deterministic within each experiment (byte-identical double-run exports); observed under two drive protocols differing in corpus/repetition/gain — confounded by design of the correction; de-confound arm named | H1/H2 runs + ISA 1146-1166 | re-scoped |
| C4a | Every adapted file shifts logits on 60/60 steps (max \|Δ\| 0.05–0.53) | 5 judge runs | measured (sensitivity result) |
| C4b | In-vivo-adapted files change greedy continuations on 2/5 frozen prompts, both on the pre-registered low-margin subset; magnitude within the null family (→ C5) | judge runs + margin census | measured — dates binding: prompts frozen → census → nulls |
| C5 | The changes are attributable to adaptation CONTENT | adjudication: failed every pre-registered criterion (a/b/d) | **REFUTED — reported as negative; NOT "statistically indistinguishable" (unpowered equivalence); term: unattributed perturbation** |
| C6 | The ladder's TWO novel rungs — the composition-specific null-feasibility theorem (ISA 1085-1109) and the destination-basin analysis (ISA 1077-1081) — distinguish C5 from its null; pre-registration itself is clinical practice, cited as such | ISA + evidence/session-i-* | demonstrated, rescoped |
| ~~C7~~ | CUT — an independent noise draw matches H2's {p2,p3} footprint at ≈19% (0.4×0.8×0.6); "noise reaches more doors" compared one draw to a union of ten | session-i-primary README arithmetic | cut; one quantified Discussion sentence instead |

## Venue path (post-RedTeam)

arXiv preprint first — timestamps the WORDING, not the seam (the seam
is flagged open in the field's own literature; independent invention
remains possible) — and ONLY after the novelty re-pass (timestamping a
taken seam is worse than useless). Then: TMLR-class first (correctness/
clarity criteria; the ladder speaks directly to the model-editing/TTT/
MeZO rows in our own matrix) — IJCNN main ~30-45% as packaged,
workshops ~60-75%; neuromorphic venues remain the community home.

## Pre-submission gates (risk-reduction per hour, post-RedTeam order)

1. **Novelty re-verification (~4-8 h)** — the only paper-killing gate;
   pass 1 throttled (S2 429), pass 2 covered top-1,000 slices of 2
   queries. NOT a formality.
2. ~~RedTeam claims table~~ — DONE (this fold).
3. **n=1 caveat in-text** (this edit) — replicates ×3 DEFERRED (~25 h
   wall; upgrades C2-C4, cannot resurrect C5; the caveat captures most
   of the reduction).
4. KLD breadth — dead unless something re-runs (nothing does).

## Missing-pieces plan (post-RedTeam additions)

- **Figures plan:** the basins figure first (8/10 nulls → H2's exact
  destination; p4's different basin); then the ladder diagram
  (families/seals/ruling flow), the adjudication table as a figure,
  the two-rectifiers mechanism sketch.
- **Averted-false-positive frame:** ISC-82's "STEERING demonstrated"
  quoted in §6 — the authors' own first reading as the demonstration
  that flip-counts publish false positives.
- **Reproducibility appendix:** file shas, seeds, fork pin (9ca265a +
  dump-patch spec), machine spec, the published crate version, AGPL
  repo — all in the record, now in the plan.
- **Foreign-runtime version statement:** the exact llama.cpp fork/commit
  L4 ran on; a version-bump robustness note named as follow-up.
- **Clamp caveat in the abstract:** the substrate bullet now carries it.
