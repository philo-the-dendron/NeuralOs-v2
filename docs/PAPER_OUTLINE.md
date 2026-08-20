# PAPER_OUTLINE.md — NeuralOS v2, Branch B

> The working outline, assembled from the ISA's record. Every claim row
> cites its evidence. RedTeam attacks the claims table before prose.
> Language rule (seal 3 + adjudication): no "steers"/"editing"; the
> adaptation result is reported as UNATTRIBUTED PERTURBATION; the
> substrate tier keeps its measured claims (arrangement read, G0).

## Title (candidates)

1. **A no_std Spiking Substrate That Reads and Rewrites a Shipped
   Quantized LLM's Weights** — the infrastructure claim, most defensible
2. Backprop-Free Weight Adaptation of a Ternary LLM Through a Spiking
   Network: a Closed, Verified Loop — the seam claim (L1–L4)
3. The Ladder Method: Pre-Registered Null Families for Attributing
   Weight-Perturbation Effects in LLMs — the methodology paper

## Abstract skeleton (honest, every number in the record)

- The substrate: no_std, i16 fixed-point SNN (crates.io, real-artifact
  test), ternary bridge (i2_s bit-exact, Q2_0 two-way).
- The loop: import a real 4B Q2_0 LLM's tensor → STDP adaptation →
  byte-exact re-export → foreign runtime (llama.cpp) executes it; the
  patched file's behavior changes; control-identity proves attribution.
- The mechanism: transmission-bug saga (falsification-driven discovery)
  → Hebbian-reversal → counters → pairing-selective, clamp-rectified
  (two rectifiers: membrane + synaptic bounds).
- The adjudication: continuation changes are statistically
  indistinguishable from equal-dose random perturbation (p3 8/10 into
  the same basin; Δmargin 0.0798 vs max-null 0.2254) — reported as an
  honest negative via the pre-registered ladder.
- The method contribution: seals, unique-flow theorem, basins analysis
  (destination analysis catches what flip-counts misread).

## Section plan

1. **Introduction** — the seam (ternary SNN ↔ ternary LLM), the field's
   own future-work flag (2309.15942), PAPER_NOVELTY.md matrix.
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
   the seals, the ruling; p2 transcripts incl. d3's coherent chain;
   footprint row; "unattributed perturbation" as the term of record.
7. **Discussion** — what would have counted as content (the conjunct);
   why the negative is informative (basins, dose, entanglement
   theorem); the honest boundary of the capability claim.
8. **Related work** — the 16-entry matrix verbatim.
9. **Limitations** — the pre-mortem's sentences, both branches' armor.

## The claims table (RedTeam's target)

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | First demonstrated L1–L4 chain (local plasticity × shipped LLM × quantized re-export × foreign runtime) | PAPER_NOVELTY matrix + loop evidence | defensible as "first in the ternary/no_std/verified-attribution setting" |
| C2 | The substrate reads model arrangement (G0, both drives, both grids) | ISC-76/81/83 | measured |
| C3 | The substrate adapts its weights under the model's own activity (deterministic, twice) | H1/H2 runs | measured |
| C4 | Adapted-weight files change the foreign runtime's behavior (logits + continuations) | 5 judge runs | measured |
| C5 | The changes are attributable to adaptation CONTENT | adjudication | **REFUTED — reported as negative** |
| C6 | The methodology (pre-registered null ladder) distinguishes C5 from its null | this session | demonstrated |
| C7 | Per-prompt footprint differs H2 vs noise ({p2,p3} vs {p2,p3,p4}) | flip table | suggestive only — n=1 vs pooled nulls |

## Venue path

arXiv preprint first (timestamp the seam), then neuromorphic-first
targets (the reachable audience): NCE-journal / IJCNN-class workshops.
Not LLM venues — the result is substrate/method, not LLM capability.

## Pre-submission gates

- PAPER_NOVELTY full re-verification pass (fresh dated section)
- RedTeam on the claims table (C1's "first" scoping especially)
- Replicates ×3 + parity gate (rung 4) — or the n=1 caveat in-text
- KLD breadth (first-drop item if anything re-runs)
