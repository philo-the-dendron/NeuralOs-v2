# Paper novelty verification — the plasticity-loop claim

> Status: SEAM HOLDS (as surveyed 2026-08-22, pass 4 — the
> arXiv-preprint gate, run). This document is the paper's novelty
> evidence and carries its own provenance. Fresh re-verification is
> still wise if submission slips months beyond pass 4.

## The claim under test

*No prior work demonstrates backprop-free, local-plasticity
(STDP-class) weight adaptation applied to a shipped, quantized LLM's
weights, exported back through a format bridge, and executed by
foreign inference tooling.*

The chain, decomposed:

| Link | Requirement |
|---|---|
| **L1** | Local/online plasticity rule — no backprop, no global gradients or loss-guided search |
| **L2** | On a **pretrained production** LLM (not scratch-trained, not convert-then-BP) |
| **L3** | Weights re-exported in the model's own quantized format (GGUF / BitNet `i2_s` class) |
| **L4** | Run by external/foreign inference tooling |

## Search provenance

- **Pass 1 (delegated, independent session): 2026-08-18 ·** engines:
  arXiv search UI + DuckDuckGo Lite + direct fetches (arxiv.org/abs,
  github.com). 26 queries logged below. Constraints: Semantic
  Scholar API + arXiv Atom API returned 429 (proxy); OpenReview has
  no fetchable search page from that sandbox.
- **Pass 2 (spot-check, this repo): 2026-08-19 ·** OpenReview **V1
  API** direct (`api.openreview.net/notes/search`, terms "STDP
  language model" + "Hebbian language model"; the V2 endpoint
  returned `searchUnavailable`). 2 × 1,000 relevance-ranked notes
  machine-scanned (delegated reader, keyword matrix + high-risk
  title scan). **Zero L1+L2 candidates.** Boundary: server reports
  10,000 matches per query — the top-1,000 relevance slice was
  scanned, not the totality.

### Query log (pass 1)

arXiv UI: "SpikingLLM" · "SpikeLLM" · "SpikeGPT" · "SpikingBERT" ·
"Deploying Sparse Large Language Models" (0) · SpikeDrive (0) ·
"Hebbian"+"large language model" (9) · "STDP"+"language model" (5) ·
"local plasticity"+"language model" (1, irrelevant) ·
"forward-forward"+"language model" (2, irrelevant) ·
"model editing"+"gradient-free" (2, irrelevant) ·
"zeroth-order"+"fine-tuning"+language models (87) ·
"test-time training"+"language model" (62) · "BitNet"+fine-tuning+
ternary (2) · survey spiking neural networks large language models
(3) · "memristor"+"large language model" (4) ·
"ternary"+"spiking"+"language model" (2) · AlphaEdit knowledge
editing · "on-chip learning"+quantized+neuromorphic (1).
DDG Lite: SpikingLLM "edge devices" · bitnet.cpp i2_s · Intel CAT-Q ·
Tencent AngelSlim · **GGUF STDP OR "local plasticity" OR Hebbian
llama.cpp → zero results (the direct seam probe)** · llama.cpp
finetune training backpropagation GGUF save.
Direct fetches: every arXiv ID in the matrix below + 4 GitHub repos;
one recalled ID (a "SpikingLLM" at 2409.04274) turned out to be a
group-theory paper — discarded; nothing rests on recall.

## The matrix (pass 1, every entry fetched)

| Work | L1 plasticity | L2 shipped LLM | L3 quant re-export | L4 foreign run |
|---|---|---|---|---|
| SpikeGPT [2302.13939](https://arxiv.org/abs/2302.13939) | ✗ (BPTT, self-described) | ✗ scratch | ✗ | ✗ |
| SpikeLLM [2407.04752](https://arxiv.org/abs/2407.04752) | ✗ (conversion in OmniQuant/GPTQ pipelines) | ✓ 7–70B | ✗ W4A4 own pipeline | ✗ |
| SpikingBERT [2308.10873](https://arxiv.org/abs/2308.10873) | ✗ (implicit diff + distillation) | ✗ | ✗ | ✗ |
| Bal et al. [2405.02543](https://arxiv.org/abs/2405.02543) | ✗ distillation | ✗ | ~ own 1/1.58-bit format, GLUE-scale | ✗ |
| Dragon Hatchling [2509.26507](https://arxiv.org/abs/2509.26507) | ✓ Hebbian spiking plasticity in an LLM | ✗ scratch GPT-2-class; plasticity = transient working memory | ✗ | ✗ |
| Chaudhary [2510.21908](https://arxiv.org/abs/2510.21908) | ✓ neuromodulated Hebbian fast weights | ✗ toy scratch models | ✗ | ✗ |
| Bio-Inspired Mamba [2409.11263](https://arxiv.org/abs/2409.11263) | ~ STDP-like hybridized with RTRL | ✗ | ✗ | ✗ |
| S²TDPT [2511.14691](https://arxiv.org/abs/2511.14691) | ✓ STDP self-attention | ✗ CIFAR vision | ✗ | ✗ |
| EMBER [2604.12167](https://arxiv.org/abs/2604.12167) | ✓ STDP+reward SNN | ✗ LLM frozen, untouched | ✗ | ✗ |
| AlphaEdit [2410.02355](https://arxiv.org/abs/2410.02355) | ✗ offline null-space least squares | ✓ | ✗ fp | ✗ |
| In-Place TTT [2604.06169](https://arxiv.org/abs/2604.06169) (+TTT cluster) | ✗ gradient inner loops | ✓ | ✗ | ✗ |
| MeZO/ZO cluster (e.g. [2608.15665](https://arxiv.org/abs/2608.15665)) | ✗ global loss-guided perturbation | ✓ | ✗ | ✗ |
| QES [2602.03120](https://arxiv.org/abs/2602.03120) | ✗ evolution strategies (global search) | ✓ PTQ INT4/8 | ~ in-place GPTQ space | ~ vLLM via own harness |
| CAT-Q [2606.26650](https://arxiv.org/abs/2606.26650) | ✗ calibration PTQ | ✓ 14B–235B→ternary | ~ no GGUF/i2_s bridge | ✗ |
| BitDistill [2510.13998](https://arxiv.org/abs/2510.13998) | ✗ BP distillation | ~ converts FP→1.58-bit | ~ bitnet.cpp-class | ~ bitnet.cpp |
| llama.cpp finetune ([ggml-org](https://github.com/ggml-org/llama.cpp)) | ✗ ggml autodiff = BP | ~ | ✓ in-GGUF | ✗ same toolchain |

**Pass 2 additions (OpenReview, 2026-08-19):** closest L1∧(LM-adjacent)
notes found — "Memory-based Hebbian Parameter Adaptation" (Hebb,
CoRR 2021: Hebbian re-wiring of pretrained nets *for few-shot class
learning*, one LM task — not shipped-LLM weight adaptation);
"Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation"
(CoRR 2023: online local adaptation — vision only); MicroNet (Hebbian
softmax, training-time); SpikeGPT duplicates. **Zero L1+L2.**

## The three nearest neighbors (and their missing links)

1. **QES** — has L2 + half of L3/L4 (quantized-space adaptation of
   PTQ models, vLLM serving). Missing: L1 (global ES loss search, not
   local online plasticity) and the GGUF-class portable export.
2. **Dragon Hatchling** — has L1 (Hebbian spiking plasticity inside
   an LLM). Missing: L2 (scratch GPT-2-class; plasticity is transient
   working memory, not weight adaptation), L3, L4.
3. **BitDistill + bitnet.cpp** — proves our L3/L4 half exists
   natively for ternary (`i2_s`, foreign CPU runtime
   [2502.11880](https://arxiv.org/abs/2502.11880)). Missing: L1 and
   honest L2 — all ternary adaptation is BP distillation/QAT.

## Supporting signal

The field's own literature marks the seam undone: the large-SNN
survey ([2409.02111](https://arxiv.org/abs/2409.02111)) lists the
only training paradigms as ANN→SNN conversion + surrogate-gradient
BP; Rajendran & Simeone
([2309.15942](https://arxiv.org/abs/2309.15942)) name "backprop-free
on-device learning and fine-tuning" as a **future** neuromorphic
principle.

## Boundaries of this survey (stated, not hidden)

- No full OpenReview API coverage (V2 search down; V1 spot-check =
  top-1,000 relevance slices of 2 queries); no Semantic Scholar (429).
  An ICLR/NeurIPS-only paper with **no arXiv mirror** could escape
  this net — priced low (ML convention is arXiv-first) but nonzero.
- **Pre-submission gate:** full pass re-run, fresh section appended
  below with its own date + engines.

---

## Re-verification log

Pass 3 — pre-submission re-verification: 2026-08-20
(the gate run; numbered 3 because the 2026-08-19 OpenReview spot-check
counts as pass 2). **Verdict: SEAM HOLDS.** No L1∧L2 conjunction in
any probe, including the three pass-1 gaps (S2, OpenReview depth,
GitHub) and the citation-graph probe.

Engines + boundaries (stated): Semantic Scholar search throttled
(13/14 HTTP 429; the single success logged); citations endpoints
usable at ~1 req/80 s — all three citation graphs pulled. OpenAlex
(api.openalex.org, title_and_abstract.search — precision mode)
unthrottled substitute. OpenReview V2 search still
searchUnavailable; V2/V1 venue enumeration 403-challenged —
ICLR/NeurIPS 2025-26 accepted-list keyword enumeration NOT covered,
mitigated by arXiv-first convention. GitHub repo search fine; code
search 401 (not covered).

Query log (engine · query · count · disposition): S2 "local
plasticity large language model" 11,874 → top-10 noise · OpenAlex
ta.search "local plasticity"+"large language model" 5 → 1 SNN↔LLM
survey (supporting), rest fringe · OpenAlex Hebbian "fine-tuning"
73 / Hebbian finetuning 0 → neuroscience/vision · OpenAlex STDP
"language model" 38 → Photograph-STDP + SymbolicLight (adjudicated)
· OpenAlex "test-time" "weight update" "local" 3 → irrelevant ·
OpenAlex default neuromorphic fine-tuning LLM 764 → Antahkarana
(false positive, below); 3 other S2-queries 85,005/2,307/522 → too
fuzzy, superseded by precision runs · OpenAlex 2025|2026
"on-device learning" quantized "large language model" 6 → BitRL +
compression duplicates · OpenReview V1 3 terms × offsets
0/1000/2000 = 9,000 notes/term (3× pass-2 depth) → vision
"local/global transformer" false positives; real plasticity titles
all scratch-SNN; zero L1+L2 · GitHub repos: STDP gguf 0 · STDP
GGUF 0 · Hebbian llama 0 · Hebbian gguf 0 · spike llama.cpp
plasticity 0 · STDP llama.cpp 0 · spiking llama.cpp on-device
learning 0 — the seam probe holds at the repo layer.

Citation graphs (the pass-1 gap, closed): Citing Dragon Hatchling
(2509.26507) — 6 citers: Wake-Sleep Compression 2603.25975
(symbolic library, no weights), BDH-CQ 2608.09888 (scratch 150M
in-context, no weight change), S²TDPT (in matrix), 3 title-dead —
no joiner. Citing SpikeLLM (2407.04752) — 40 citers, six
abstract-checked (NEXUS, WTA-Spikingformer, QSLM, SpikingBrain2.0,
Kirin, Spiking Manifesto) + NSLLM — conversion/compression family,
no joiner. Citing BitNet b1.58 (2402.17764) — **447 citers, zero
plasticity-titled** — the ternary community has produced no
plasticity joiner.

New matrix rows (pass 3, abstract-fetched):
| Work | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Antahkarana (Zenodo 10.5281/zenodo.19522347) | ✗ LoRA+BP | ✓ Phi-2 | ✗ | ✗ — v2 self-withdraws headline |
| Photograph-STDP (Zenodo 10.5281/zenodo.20690036) | ✗ STDP on records, weights frozen | ~ | ✗ | ✗ |
| Spike-Aware C++ INT8 (2606.03026) | ✗ inference only | ✗ scratch | ~ INT8 own | ~ own runtime |
| NEXUS (2601.21279) | ✗ STE conversion | ~ converted 70B | ✗ | ~ neuromorphic HW |
| NSLLM (NSR 2025, 10.1093/nsr/nwaf551) | ✗ conversion+PTQ | ~ 1.5B | ~ | ~ own FPGA |
| QSLM (2601.00679) | ✗ automated PTQ | ✗ scratch | ✓ own | ✓ embedded |
| Kirin (2602.08817) | ✗ ANN→SNN | ~ | ~ | ~ |
| WTA-Spikingformer (2604.11321) | ✗ WTA=attention, surrogate BP | ✗ | ✗ | ✗ |
| BitRL (IEEE FET 2026) | ✗ RL global reward (QES-class) | ✓ | ~ 1-bit | ✗ |

Supporting signal: the 2025 SNN↔LLM survey (Synergies and
Divergences) independently lists "biologically inspired local
plasticity rules" as a training-methodology class with no
shipped-LLM adaptation demonstrated — corroborating 2409.02111.

Pass 4 — arXiv preprint gate: 2026-08-22. **Verdict: SEAM
HOLDS.** Engines: arXiv Atom API (export.arxiv.org, working
clean this session), OpenReview V1 API, GitHub repo search,
Semantic Scholar citations (unthrottled today, unlike pass 3) +
OpenAlex cross-check. 41 engine queries total; ~160 abstracts/titles
adjudicated against the matrix. Zero L1∧L2 candidates; zero
all-four. Two near-misses added below.

Query log (engine · query · count · disposition):

- arXiv, pass-1 rerun (15 queries, relevance order): SpikingLLM 1
  · SpikeLLM 1 · SpikeGPT 1 · SpikingBERT 2 · Hebbian+LLM 9 ·
  STDP+LM 5 · "local plasticity"+LM 1 (LLM-Ideoplasticity —
  behavioral politics measurement, weights untouched) ·
  forward-forward+LM 2 · "model editing"+gradient-free 2 ·
  TTT+LM 62 (all gradient inner-loop class) · BitNet+ternary+ft
  2 · SNN+LLM survey 5 · memristor+LLM 4 · ternary+spiking+LM 2 ·
  on-chip+quantized+neuromorphic 1 (EqProp oscillator Ising —
  local rule, non-LLM). **Totals match pass 1 exactly.**
- arXiv, 2026-targeted (10 queries, submittedDate desc):
  "knowledge editing"+"biologically plausible" **0** (the risk
  pair itself) · "local plasticity"+LLM 1 (same Ideoplasticity)
  · Hebbian+fine-tuning 15 (all vision/CNN/neuro: NM-Hebb=CNN,
  NM-Hebb-TTT=vision — pass-2 known) · STDP+transformer 8
  (S²TDPT + vision/hardware/RSA) · neuromorphic+"model editing"
  **0** · "weight editing"+quantized 2 (GROM + GUARD-IT, below)
  · "training-free"+weights+LLM 158 → top-30 date-ordered
  scanned: refusal-neuron/steering/decoding/merging class, zero
  plasticity · Hebbian+LLaMA 1 (Obliviate eval) · STDP+GGUF **0**
  · plasticity+llama.cpp **0** (direct seam probes hold).
- OpenReview V1 (8 terms × top-1,000 relevance slice = 8,000
  notes, keyword matrix + high-risk title scan): 6 unique flags →
  2 abstract-verified false-positives (r-STDP RL sim-to-real,
  ICONS 2022 — surrogate-gradient BP on control tasks, not an
  LLM; Neuromodulation Gated Transformer, ICLR-2023 Tiny —
  scratch SuperGLUE arch) + STDP-Net (vision PAR) + SpikeGPT +
  Spike-driven Transformer + Synaptic-Flow NAS. **Zero L1∧L2.**
  Boundary unchanged: server 10k matches/term, top-1k scanned.
- GitHub repo search (8 queries): STDP+gguf 0 · STDP+llama.cpp 0
  · STDP+bitnet 0 · Hebbian+gguf 0 · Hebbian+llama.cpp 0 ·
  Hebbian+bitnet 0 · plasticity+gguf 0 · spiking+llama.cpp+
  on-device+learning 0. Code search 401 (auth) — not covered,
  same boundary as pass 3.
- Citation forward-sweep (S2, all six graphs pulled full): citing
  Dragon Hatchling 7 (3 new since pass 3: Parity-Bottleneck
  interpretable transformers, AI-subject philosophy, BambooKG
  KG-weighting — no joiner) · citing QES 2 (both ES/ZO-family)
  · citing BitDistill 2 (embeddings + VHDL) · citing SpikeLLM 40
  (20×2026: SpikeVLA, LongSpike, Otters++, BiSpikCLM, Matterhorn,
  TPipe, surveys — all conversion/compression/encoding family;
  BiSpikCLM abstract-fetched: binary MatMul-free LM by
  **distillation**, L1 ✗ L2 ✗) · citing BitNet b1.58 448 (111×
  2026 — zero plasticity-titled; all PTQ/QAT/LUT/accelerator
  class; nearest flavor: EqProp-for-Kuramoto = local rule on
  non-LLM hardware) · citing AlphaEdit 265 (114×2026 — the
  risk-class anchor: the "training-free/gradient-free" 2026
  trend is closed-form optimization (GROM, Soft-RLS), model
  merging (RAIN-Merging), and BP (DOW-KE) — **no plasticity
  joiner**). OpenAlex cites: cross-check consistent (thinner
  graphs, no contradicting signal).

New matrix rows (pass 4, abstract-fetched):

| Work | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| GROM [2608.05783](https://arxiv.org/abs/2608.05783) | ✗ closed-form ridge least-squares (AlphaEdit class) | ✓ | ~ quantization-*robust* edit, not quantized-format-native | ✗ own PyTorch harness |
| HeLa-Mem [2604.16839](https://arxiv.org/abs/2604.16839) | ~ Hebbian dynamics on an **external memory graph** | ✗ LLM weights untouched | ✗ | ✗ |
| GUARD-IT [2605.12765](https://arxiv.org/abs/2605.12765) | ✗ activation steering, no weight change | ✓ | ✗ | ✗ |
| DOW-KE [2608.16932](https://arxiv.org/abs/2608.16932) | ✗ explicit backprop of editing objective | ✓ | ✗ | ✗ |
| BiSpikCLM [2605.13859](https://arxiv.org/abs/2605.13859) | ✗ ANN→SNN distillation | ✗ scratch spiking LM | ~ fully binary, own format | ✗ |
| Fact-writes study [2607.11020](https://arxiv.org/abs/2607.11020) | ✗ BP training writes (analysis paper) | ✓ Qwen3 | ✗ | ✗ |

Pass-4 signal (supports the paper's framing): the 2026 editing
literature is independently converging on "edits must survive
quantization" — GROM's low-bit quantization-recovery attack,
DurableUn (2026, AlphaEdit-citing, quantization-induced recovery
attacks on unlearning) — while the plasticity side stays on
scratch/hybrid models. The L1∧L3 conjunction is now visible as
an acknowledged *problem* in editing, with no local-rule solution
attempted. GO for the arXiv preprint (principal stamp pending).
