# MANIFEST — Branch B paper artifact (NeuralOS v2)

## What this is

The candidate PDF, full LaTeX source, figure generators, and the
language gate for the Branch B paper: *"A Spiking Substrate That
Reads and Round-Trips a Shipped Quantized LLM's Weights"* — see
`main.pdf` for the exact title of record.

## Provenance

- **Source repository:** https://gitea.com/Caramoussin/NeuralOs-v2
  (AGPL-3.0-or-later). GitHub mirror:
  https://github.com/philo-the-dendron/NeuralOs-v2
- **Artifact source commit:** the paper tree this bundle was built
  from is committed verbatim in the repository; the bundle filename
  (`neuralos-branchb-paper-<sha>.zip`) carries that commit's sha —
  the zip was built by `make dist` run from exactly that commit.
  Parent commit: `e38bbef` ("paper novelty pass 4 — arXiv gate",
  2026-08-22, both remotes aligned).
- **Overlay included in this version (committed, unlike the base):**
  W0 = byline (`philo-the-dendron`, Independent), date line
  ("Version 1.0, August 2026 — Branch B"), the Acknowledgments
  section, six overfull-hbox line-break fixes, and the authorized
  table surgery (matrix + p2/p3 transcript tables; `\footnotesize`
  + `p{}` wrap columns; no `\resizebox`; zero cell-text change).
- **Library of record:** crate `neuralos-snn 0.1.0-alpha.5` on
  crates.io, tree-identical to repo commit `103fa59` (the repo
  artifact bundle in the companion deposit is the `git archive` of
  exactly that commit).
- **Evidence:** raw judge/experiment logs live in the repository's
  `evidence/` tree, indexed by `evidence/INDEX.md`. Model weights
  are NOT redistributed (gitignored; HuggingFace downloads per
  `AGENTS.md`). The judge runtime is a foreign llama.cpp fork,
  rebuildable from the repository via `tools/build_fork.sh`
  (pinned commit `9ca265a`, branch `prism`).

## Build (verified at bundle time)

```bash
cd paper
make          # tectonic 0.17.0 (via nix), or latexmk/pdflatex
make figs     # python3 + matplotlib, SOURCE_DATE_EPOCH-pinned,
              # byte-stable regeneration (P2-W3)
make gate     # language rules of record — exit 0 "clean"
```

- PDF sha256 (the copy inside this zip):
  `15baa85ae407ac0c76ef20daadfedcb74a748414cf84685bfb47cad8bfaa3d1a`
- Bundle built: 2026-08-22. PDF reproducibility: `make` exports
  `SOURCE_DATE_EPOCH=1700000000`, which tectonic/xdvipdfmx honor —
  `make`-built PDFs are byte-stable (verified: two consecutive
  builds identical). Builds outside `make` embed the wall clock and
  differ. Figure PDFs are byte-stable for the same reason.

## Gates (census at bundle time)

- Language gate (`tools/check_language.sh`): **clean** — the term of
  record "unattributed perturbation" is used; the banned-list words
  appear only in sanctioned, line-marked exceptions.
- Box census (TeX log): six prose overfull hboxes fixed by
  line-break-only edits (substrate canary 110.4pt, adjudication
  theorem 33.4pt, related-work prose 36.2pt, limitations 1.3pt,
  app-repro llama_vocab 45.6pt, app-repro null-families 69.3pt).
  Three table overfulls (L1–L4 matrix 261.8pt; p2/p3 transcript
  tables 220.0/372.3pt — previously clipped at the physical page
  edge) fixed by authorized layout surgery: `\footnotesize` plus
  `p{}` ragged-right wrap columns (matrix: 60/92/78/92/78pt;
  transcripts: 118pt+322pt) and `\usepackage{array}`; `\resizebox`
  NOT used; zero cell text, column order, or row order changed.
  Ink-margin verification at 72 dpi: all 20 pages rightmost ink
  ≤ 565px (text edge 540px, tolerance 570px); text-layer grep
  confirms formerly clipped strings whole ("L3 quant re-export",
  "bitnet.cpp", "neuromorphic HW", "Thursday04/05/2018", all 25
  matrix entries). Remaining known residuals, principal-sanctioned:
  two cosmetic margin protrusions in the reproducibility appendix
  (25.8pt path string, 21.7pt hash string — no breakpoint fix
  exists; ink ≤ 565px) and one loose bibliography line (SpikingBERT
  entry, badness 10000).

## Honest-claim language (from the paper's abstract of record)

> **The adjudication, up front:** the adapted file does change greedy
> continuations (2 of 5 frozen prompts, both on the pre-registered
> low-margin subset) — but those changes *failed every pre-registered
> criterion* for exceeding equal-dose random perturbation (8 of 10
> dose-matched nulls land the in-vivo run's exact destination basin;
> margin shift 0.0798 vs. null-max 0.2254). The term of record is
> *unattributed perturbation*: the pre-registered null ladder —
> committed before its authors could see the null tables — killed
> the authors' own earlier capability reading before it could be
> published. The findings that stand are infrastructure and method.

Bounded claims: the arrangement-tracking result is a sensitivity
result at n=1 (one slice, one model, one seed), not a population
claim; the sign-dominant drive (69.5–69.8% of dim-steps railed at
the clamp) is stated in-text; the de-confound arm and the
version-bump robustness check are named and unrun (Limitations).

## Authorship

`philo-the-dendron` (Independent) — sole author. Code identity
`Caramoussin` appears only inside the repository URL. AI-assisted
tooling statement: see the Acknowledgments section of the PDF.
