# Paper novelty verification — the plasticity-loop claim

> Status: OPEN SEAM (as surveyed). This document is the paper's
> novelty evidence and carries its own provenance. **Pre-submission
> gate: re-run the full pass (incl. OpenReview API + Semantic Scholar
> from an unthrottled network) and append a fresh dated section
> before submitting** — a novelty claim should be as fresh as its
> submission date.

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

*(empty — the pre-submission pass appends here)*
