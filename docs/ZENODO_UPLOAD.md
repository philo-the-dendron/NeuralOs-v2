# Zenodo upload guide — Branch B paper + repo artifact (dual DOI)

Every web-form field pre-decided for both deposits. Follow top to
bottom; each deposit is ~5 minutes. Account already live
(principal-operated). Built from the Z0 findings (verified live
2026-08-22, URLs inline) and the three sealed session rulings.

**The three sealed rulings (verbatim, binding):**

1. Byline: "philo-the-dendron" solo, affiliation Independent;
   Caramoussin = code identity, appears ONLY inside the repo URL.
2. Acknowledgments: Variant A, friend named "Soushi888",
   AI-assistance line IN. (In the PDF; nothing to enter on Zenodo.)
3. Replicates: PRE-COMMITTED yes — run only if TMLR reviewers ask
   (~25 h, apparatus exists). Not an upload-time concern; recorded
   in Z3's ISA entry.

## Artifacts (built by `make dist` in `paper/`, 2026-08-22, from commit `42f8d52`)

| File | Size | sha256 |
|---|---|---|
| `paper/dist/neuralos-branchb-paper-42f8d52.zip` | 410,083 B | `330256be016517d1f7eb3245d1d8be62dc4dc1de4e8bd287dab479788884327e` |
| `paper/dist/neuralos-repo-103fa59.tar.gz` | 862,191 B | `33563ae0a63be6209a8c7d296b7186db061655665c4957c3a968f6e501c6204e` |

(Inner PDF sha256, pinned in the zip's MANIFEST.md:
`15baa85ae407ac0c76ef20daadfedcb74a748414cf84685bfb47cad8bfaa3d1a`)

The zip filename's sha (`42f8d52`) is the commit the bundle was built
from — the committed tree contains every file in the zip (provenance:
`paper/MANIFEST.md`). PDF bytes are `make`-stable
(`SOURCE_DATE_EPOCH`); do not rebuild the zip — a re-run of
`make dist` re-stages mtimes and changes the zip's own sha.

## Z0 findings (all verified live 2026-08-22)

| Question | Verdict | Source |
|---|---|---|
| Pseudonymous creator OK? | **Yes.** Free-text person names, no real-name or identity-verification requirement anywhere; ORCID autocomplete is optional; affiliation accepts free text when not in ROR ("add simply the text" → "Independent") | help.zenodo.org/docs/deposit/describe-records/creators/ + about.zenodo.org/policies/ |
| ORCID required? | **Skippable.** Name identifiers are an optional sub-field; only a family name is required for a Person creator | creators page |
| License picker | Required; defaults CC-BY-4.0; full SPDX list searchable; multiple licenses per record supported | help.zenodo.org/docs/deposit/describe-records/licenses/ |
| Size limits | 50 GB per record default (+150 GB allocatable allowance) — our deposits are < 1 MB | help.zenodo.org/docs/deposit/manage-quota/ + policies |
| Resource types | DataCite vocabulary dropdown; mixed types → split records (we do exactly that) | help.zenodo.org/docs/deposit/describe-records/resource-type/ |
| DOI mechanics | Registered at publish; "Get a DOI now!" can RESERVE a DOI pre-publication (embeddable in files); deleting the draft loses the reserved DOI | help.zenodo.org/docs/deposit/describe-records/reserve-doi/ |

## Deposit 1 — the paper

Click-path: zenodo.org → log in → **Uploads** (top bar) →
**New upload** → drag `neuralos-branchb-paper-42f8d52.zip` into
Files → fill metadata below → **Save** (check the reserve-DOI
option first if wanted) → **Publish**.

| Field | Value |
|---|---|
| Resource type | Publication → **Article** |
| Title | A Spiking Substrate That Reads and Round-Trips a Shipped Quantized LLM's Weights |
| Creator | Person. Family name: `philo-the-dendron` (leave Given names EMPTY — single-field pseudonym). Affiliation: `Independent` (free text — it will not autocomplete; that is expected and fine). ORCID: leave empty. |
| Description | (paste below, Description A) |
| License | **Creative Commons Attribution 4.0 International** (the default — keep it) |
| Keywords | spiking neural networks; STDP; local plasticity; ternary quantization; GGUF; llama.cpp; neuromorphic; free software |
| Additional notes | (paste below, Notes A) |
| Version | 1.0 |
| Publication date | 2026-08-22 (default today) |
| Publisher | Zenodo (default) |
| Funding | none — leave empty |
| Contributors | none — leave empty |
| DOI | answer "No" → Publish registers it (or "Get a DOI now!" to reserve first) |

**Description A (paper):**

> We build a spiking substrate that reads, adapts, and round-trips
> the weights of a shipped, quantized large language model: a
> `no_std`, i16 fixed-point SNN library (LIF + STDP, published on
> crates.io, byte-level test vectors against real artifacts) whose
> ternary format bridge round-trips BitNet i2_s bit-exactly and a
> real 4B Q2_0 LLM's tensor two-way. The adjudication is up front:
> the adapted file does change greedy continuations (2 of 5 frozen
> prompts), but those changes failed every pre-registered criterion
> for exceeding equal-dose random perturbation; the term of record
> is "unattributed perturbation." The findings that stand are
> infrastructure and method: the closed loop, the
> arrangement-tracking sensitivity (n=1), a falsification-driven
> structural bug discovery and mechanism reversal, a
> composition-specific null-feasibility theorem, and the
> destination-basin analysis. Full source, evidence index, and
> adjudication record: https://gitea.com/Caramoussin/NeuralOs-v2
> (AGPL-3.0-or-later); library crate `neuralos-snn` on crates.io.
> Companion repository archive:
> https://doi.org/10.5281/zenodo.XXXXXXX [repo DOI — fill after
> Deposit 2, via metadata edit].

**Notes A (paper):**

> Artifact contents: candidate PDF (sha256 pinned in MANIFEST.md),
> full LaTeX source, figure generators (matplotlib,
> SOURCE_DATE_EPOCH-pinned), and the language gate script.
> Reproducibility statement and honest-claim language: see
> MANIFEST.md inside the zip. The bundle is named by the paper's
> source commit (42f8d52, named in the bundle filename); the
> library of record is neuralos-snn
> 0.1.0-alpha.5 (crates.io), tree-identical to repo commit 103fa59.

## Deposit 2 — the repo artifact

Click-path: same → **New upload** → drag
`neuralos-repo-103fa59.tar.gz` into Files → fill → **Publish**.

| Field | Value |
|---|---|
| Resource type | **Software** |
| Title | NeuralOs-v2 source archive at commit 103fa59 — spiking substrate, ternary-bridge evidence record, and the neuralos-snn 0.1.0-alpha.5 library tree |
| Creator | Person. Family name: `philo-the-dendron`, Given names empty. Affiliation: `Independent` (free text). ORCID empty. |
| Description | (paste below, Description B) |
| License | **GNU Affero General Public License v3.0 or later** (SPDX: AGPL-3.0-or-later — search "AGPL" in the picker) |
| Keywords | Rust; spiking neural networks; no_std; fixed-point; STDP; NIR; Neuromorphic Intermediate Representation; GGUF; llama.cpp; AGPL |
| Additional notes | (paste below, Notes B) |
| Version | 1.0 (commit 103fa59) |
| Publication date | 2026-08-22 |
| Funding / Contributors / Publisher | empty / empty / Zenodo default |
| DOI | same mechanics as Deposit 1 |

**Description B (repo):**

> Source archive (git archive, tar.gz) of the NeuralOs-v2 research
> workspace at commit 103fa59 — the tree identical to the published
> crates.io crate neuralos-snn 0.1.0-alpha.5: a no_std, i16
> fixed-point spiking-neural-network library (LIF + STDP, general
> graph assembly, NIR import/export), the Slint visualizer, the
> research runtime holding the adjudicated ternary-bridge evidence
> (GGUF container, Q1_0/Q2_0 compute, tokenizer), the paper built
> from it, and the evidence/ index of raw judge logs. Live
> repository: https://gitea.com/Caramoussin/NeuralOs-v2 (AGPL-3.0).
> Companion paper:
> https://doi.org/10.5281/zenodo.YYYYYYY [paper DOI — fill after
> Deposit 1, via metadata edit].

**Notes B (repo):**

> Gitignored payloads are not redistributed: model weights
> (HuggingFace downloads) and the foreign judge runtime
> (rebuildable via tools/build_fork.sh, pinned llama.cpp fork).
> Quality gates: cargo check/test/clippy green; no_std and simd
> feature gates; the HDF5 NIR evidence gate (see README).

## After both deposits (feeds Z3)

1. Copy both DOIs (record page → "DOI" line, 10.5281/zenodo.XXX).
2. Optional, 2 min: edit each record's metadata (record page →
   "Edit" → Description) to replace XXXXXXX/YYYYYYY with the real
   DOIs, then Save. Zenodo allows metadata edits on published
   records; files stay untouched.
3. Report both DOIs back — Z3 (README citation block + ISA entry)
   runs on that report.
