# NeuralOS v2 — Research Findings (StandardResearch, 2026-08-05)

> 3-agent cross-checked research: Rust SNN ecosystem, RustCrypto, esp-rs/RISC-V, Tauri v2,
> candle/DistilBERT, ternary NN, Huawei-specific ternary work, RISC-V Ottawa event,
> Debian Live ISO. Every URL fetched this session.

## TL;DR

The research **validates 5 of our planned decisions** and **overturns 4**. The biggest
corrections: (1) RustCrypto is the cipherpunk move, not from-scratch crypto — rewriting
audited code is theater; (2) DistilBERT is an encoder, it does not summarize — we need
a generative model (Phi-3-mini or Qwen-2.5-1.5B); (3) no Huawei ternary work exists in
public literature despite genuine search; (4) no Ottawa RISC-V event this August is on
the official calendar (Summit Europe 2026 was Bologna, June 8–11).

## Critical corrections to our plan

| # | What we said | What's actually true | Source (verified) |
|---|---|---|---|
| 1 | **"Build ChaCha20-Poly1305 from scratch for cipherpunk ownership"** | **Theater.** Use `chacha20poly1305` (RustCrypto, v0.11.0, audited by NCC Group 2020, 6.4M downloads/month, 211 LOC readable). Sovereignty = read+pin+understand, not rewrite worse. `x25519-dalek` v3.0.0 has formally-verified `fiat` backend. | lib.rs/crates/chacha20poly1305, lib.rs/crates/x25519-dalek |
| 2 | **"Use DistilBERT via candle for summarization"** | **Wrong model class.** DistilBERT is an encoder (embeddings/classification). Verified candle example exists but only does masked-token prediction. **For summarization use a generative model:** Phi-3-mini (3.8B), Qwen-2.5-1.5B (sweet spot), or Llama-3.2-1B. Budget 1–2 min/summary on CPU for a 5-page paper. | huggingface/candle/examples/distilbert (verified), examples/qwen, examples/llama |
| 3 | **"Huawei ternary work — want to be compatible"** | **No Huawei-specific ternary NN work found** despite genuine search (arXiv "Huawei ternary" = 0 hits; Huawei Noah's Ark Lab GitHub = 23 repos, none ternary). Most likely confusion with **Microsoft BitNet b1.58** (arXiv:2402.17764) or an individual researcher. Default to standard TWN `{-1,0,+1}` + own format spec. | arXiv searches verified; github.com/huawei-noah repo list verified |
| 4 | **"RISC-V conference in Ottawa this August"** | **Not on official RISC-V calendar.** RISC-V Summit Europe 2026 was June 8–11 in Bologna, Italy (already over). Official riscv.org/events/ lists 6 upcoming starting Sept 9, 2026 — none Ottawa, none August. **User must provide event URL before we plan around Ottawa.** | riscv.org/events/ (verified), riscv-europe.org/summit/2026/ (verified) |

## Validated decisions (research confirms we got these right)

| # | Decision | Confirmation | Source |
|---|---|---|---|
| 1 | **Port SNN library from audited v0.1** | **Justified by gap.** Rust SNN ecosystem has 4 spiking crates, ZERO match `no_std + i16 fixed-point + STDP`. arewelearningyet.com has no SNN category at all. | lib.rs/search?q=neural (94 crates scanned), arewelearningyet.com |
| 2 | **esp-rs + ESP32-C3 RISC-V path** | **Proven.** esp-hal v1.0.0 (2,037 stars, updated today), esp-generate is official no_std template, ESP32-C3 is first-class target. | github.com/esp-rs/esp-hal (verified) |
| 3 | **Tauri v2 for desktop app** | **Stable since Oct 2024.** v2.11.5 current, 3.5M downloads/month, MIT/Apache, Windows+Linux+macOS from one codebase. | tauri.app/blog/tauri-20/ (verified), lib.rs/crates/tauri |
| 4 | **candle for local ML** | **Best Rust ML framework for desktop.** 20.9k stars, v0.11.0, MIT/Apache, CPU+CUDA+Metal+WASM. Burn is credible #2 (training + no_std story). Skip rust-bert (libtorch baggage). | huggingface/candle (verified) |
| 5 | **live-build for Debian ISO** | **Still canonical.** Documented first-class `config/packages/chroot/` for custom .deb pre-install. Minimum-effort path. | live-team.pages.debian.net/live-manual (verified) |

## Detailed findings

### SNN ecosystem — **[HIGH] gap confirmed, port is defensible**

Closest matches (all verified live):

- `feagi-npu-neural` v0.0.14 (released **today**, Aug 5 2026, Apache-2.0, `no_std`, zero-alloc, ESP32/WASM targets) — closest match, but **f32 not i16**. **FORK as no_std/zero-alloc pattern reference; do NOT use the math.** 67 downloads/month.
- `hebb` v0.1.0 (May 2026, Apache-2.0) — best algorithm coverage (LIF/Izhikevich/AdEx/Hodgkin-Huxley, STDP+R-STDP, 4 topologies). **USE as algorithm reference, SKIP as dep** (std-only, f32, Uuid-addressed).
- `spiking_neural_networks` v0.24.0 (Nov 2025, Apache-2.0) — most mature, biology-research oriented, deps `ndarray`+`opencl3`+`rayon`. **SKIP for our core; gold-standard biology reference.**
- `ruvector-nervous-system` — vaporware red flags ("Tier 4: t4_agentic_self_model", contributor literally named "claude"). **SKIP, do not cite.**

### Crypto — **[HIGH] USE RustCrypto, building from scratch is theater**

- `chacha20poly1305` v0.11.0 (Jun 2026): no_std ✓, audited by NCC Group, 6.4M downloads/month, 211 LOC readable.
- `x25519-dalek` v3.0.0 (Jul 2026): no_std ✓, BSD-3-Clause, **`fiat` backend = Fiat Crypto formally-verified code from MIT**. 5.1M downloads/month.
- `dryoc` v1.0.0 (Jul 2026): pure-Rust libsodium-compat, MIT, **NOT audited** (README states verbatim). Wraps RustCrypto anyway — adds unaudited layer. SKIP unless you need libsodium wire-format.
- RustCrypto org: led by Artyom Pavlov + Tony Arcieri (curve25519-dalek author). 40 repos, multiple updated today.

**The cipherpunk framing the research uncovered:** "owning" software means *understanding every instruction your CPU executes* — satisfied by vendoring + pinning + reading the 211 LOC + running RFC test vectors yourself. It is NOT satisfied *more* by hand-rolling. Cardano uses formally-verified primitives, not from-scratch rewrites. The algorithm isn't the trust boundary — the implementation is. Hand-rolled ChaCha20 in 2026 has *less* provenance than the audited RustCrypto one (you'll introduce side-channels your eye can't see).

### esp-rs / RISC-V — **[HIGH] proven for embedded, no ML framework fits**

- esp-hal 2k stars, updated today (Aug 5 2026). ESP32-C3 = first-class RISC-V target.
- `esp-generate` = official no_std template generator. `cargo generate esp-rs/esp-generate`.
- **`esp-rs/riscv` repo DOES NOT EXIST** (404). RISC-V support lives inside esp-hal as `esp-riscv-rt`. Generic RISC-V: rust-embedded/riscv v0.16.1 (517k downloads/month).
- **candle does NOT run on RISC-V.** Deps: rayon, tokenizers 0.22, gemm 0.19, memmap2 — std-only, workstation-class. ESP32-C3 has 400KB SRAM; a 66M-param DistilBERT is 650× too big.
- **burn** has `no_std` (Flex backend only, slow). Not realistic for inference.
- **Hand-rolled `no_std` SNN core is the ONLY path on ESP32-C3.** This is our differentiator — no prior art (awesome-esp-rust lists 30+ working C3 projects, none ML/SNN).

### Tauri v2 + candle integration — **[HIGH] both proven, combo is on us**

- Tauri v2.0 stable since Oct 2, 2024. v2.11.5 current.
- **Zero production Tauri+candle references** (1 hobby repo, 2 stars). Build it ourselves — low risk (both Rust-native, compose via `tauri::command`).
- Frontend: **Svelte 5 or SolidJS** is the inference (small bundle matters at 600KB total app size). NOT URL-verified community consensus — confirm before locking in.

### candle + summarization — **[HIGH] use generative model, not DistilBERT**

| Model | candle support | CPU feasibility | Notes |
|---|---|---|---|
| DistilBERT | ✅ example exists | ✅ 66M params | **Encoder only — NOT a summarizer.** Embeddings/classification only. |
| **Phi-3-mini (3.8B)** | ✅ `examples/phi`, `quantized-phi` | ⚠️ ~7GB fp32, 3.5GB q4 | Generative, good summarizer. |
| **Qwen-2.5-1.5B** | ✅ `examples/qwen` | ✅ ~3GB fp32 | **Sweet spot for tower PC CPU.** Strong for size. |
| Llama-3.2-1B | ⚠️ v3 path supported, not named | ✅ ~2GB fp32 | **Verify weights load before basing architecture on it.** |

Budget single-digit tokens/sec on CPU → 1–2 min to summarize a 5-page paper. Acceptable for research tool, not realtime.

### Ternary NN landscape — **[HIGH] standard is `{-1,0,+1}` + FP scale**

- **TWN** (Li et al. 2016, arXiv:1605.04711): threshold Δ ≈ 0.7·E[|W|], scale = mean(|W|>Δ), STE for gradients. 16× compression.
- **BitNet b1.58** (Ma et al. 2024, arXiv:2402.17764, **Microsoft not Huawei**, github.com/microsoft/BitNet 39.8k stars): `Round(W / γ)` where γ = mean(|W|). "1.58 bits" = log₂(3).
- **24 ternary SNN papers exist** on arXiv — real subfield. Canonical: **"Ternary Spike"** (Guo et al. AAAI 2024, arXiv:2312.06372, github.com/yfguo91/Ternary-Spike).

### Ottawa RISC-V event — **[HIGH] does not exist as described**

- RISC-V Summit Europe 2026: **Bologna, Italy, June 8–11** (already over). NOT Ottawa, NOT August.
- Official riscv.org/events/ lists 6 upcoming events starting Sept 9, 2026. **None in Ottawa, none in August.**
- Next RISC-V Summit NA: **Feb 16–18, 2027 in San Francisco**.

User must provide event URL — they may be thinking of (a) a non-RISC-V Ottawa event, (b) different dates, (c) different city. Do not plan travel/demo around the Ottawa-August assumption.

## Updated v2 plan (incorporating corrections)

| Phase | What changes |
|---|---|
| **Crypto** (was Phase 6, "build from scratch") | **Now: USE RustCrypto from day one.** `chacha20poly1305` + `x25519-dalek` with `fiat` backend. Pin, vendor, read the 211 LOC, run RFC vectors. Add `neuralos-crypto` crate as a thin sovereignty wrapper. **Saves 2-3 sessions of wasted from-scratch work.** |
| **App ML** (was "DistilBERT") | **Now: Qwen-2.5-1.5B (preferred) or Phi-3-mini.** Use candle's existing `examples/qwen` as starting point. DistilBERT only for non-summarization tasks (classification, embeddings). |
| **Ternary** (unchanged) | Standard `{-1,0,+1}` + per-tensor scale. Read Ternary-Spike (Guo 2024) for SNN-specific patterns. Spec our own wire format. **No Huawei compat target — there's nothing to be compatible with.** |
| **Ottawa** (was hard August deadline) | **Drop the deadline.** No verified Ottawa event. If user provides real event URL, we replan around that date. |
| **SNN library** (unchanged) | Still #1 priority, port from audit, justified by ecosystem gap. |
| **Distro** (unchanged) | live-build + `config/packages.chroot/` for our `.deb`. |

## Open questions for the principal

1. **Ottawa event** — please paste the actual event URL. Official RISC-V calendar has nothing in Ottawa this August.
2. **"Huawei guy" identity** — could it be (a) someone moving from Huawei to a ternary startup, (b) you misremembered Microsoft BitNet as Huawei, (c) someone you'd meet at a different event? Name or LinkedIn would resolve it.
3. **App ML model preference** — Qwen-2.5-1.5B (~3GB, recommended) vs Phi-3-mini (~3.5GB q4, smarter) vs Llama-3.2-1B (~2GB, verify it loads)?
4. **Frontend framework** — Svelte 5 vs SolidJS for Tauri (both ~equally lean; my lean is Svelte but not URL-verified community consensus)?

## Conflicts / unresolved

- **None material.** All three agents agreed on all material points. The crypto findings in particular are [HIGH] across the board.

---

# Follow-up Research (2026-08-06): Bonsai + Tauri alternatives + summarization model

> Two focused agents dispatched after principal corrections: (1) "Bonsai" model
> identification, (2) pure-Rust GUI alternatives to Tauri (cipherpunk stance).
> All URLs verified this session.

## Principal-driven corrections accepted

1. **RVO Ottawa meetup is real** — Aug 19, 2026 (and July 15). RVO = RISC-V Ottawa
   Community, monthly meetups. Not on global riscv.org calendar; agent missed the
   local-community event layer. **14-day deadline confirmed.**
2. **Huawei ternary accepted as fact** — principal has direct knowledge despite
   public-record silence. Treated as real compat target going forward.
3. **Tauri concern valid** — system webview (WebView2/WebKit/webkit2gtk) is a black
   box. Triggers pure-Rust GUI evaluation.
4. **"Bonsai" is a real model reference** — see below.

## [HIGH] "Bonsai" identified — Prism ML ternary LLM family

Verified: `huggingface.co/prism-ml/Bonsai-1.7B-gguf` and `prism-ml/Ternary-Bonsai-27B-gguf`.
1.17k likes, 761k downloads on the 27B variant alone. Apache-2.0. Whitepaper:
`github.com/PrismML-Eng/Bonsai-demo`.

- **Vendor:** Prism ML (`prismml.com`)
- **What it is:** end-to-end 1-bit (Q1_0) and ternary (2-bit Q2_0_g128) quantized LLMs
  based on Qwen3-1.7B and Qwen3.6-27B
- **Sizes:** 1.7B (0.24 GB deployed, runs on iPhone), 4B, 8B, 27B (~7.2 GB,
  retains 94.6% of FP16 quality)
- **Runtime:** requires Prism ML's **forked llama.cpp** with custom Q1_0/Q2_0_g128
  kernels. **candle does NOT support this format.**

**Strategic implication:** Prism ML's work validates the ternary direction publicly.
The "Huawei ternary" the principal cited as direct knowledge aligns with this
industry direction. Our ternary SNN format spec should track Q1_0/Q2_0_g128 as
one compat reference point.

**For NeuralOS v2's ML stack:** SKIP Bonsai direct (candle can't run it). Use
`Qwen3-1.7B-Instruct` via candle's `quantized_qwen3.rs` if we want the same base
model at proven quality. Bonsai matters for the ternary-format conversation, not
as a runtime dep.

## [HIGH] Tauri dropped — Dioxus too — both webview-based

Critical finding (verified verbatim from Dioxus README): "Render using Webview or -
experimentally - with WGPU or Freya." Desktop default = system webview. The
pure-Rust renderer (Blitz) is **pre-alpha** per Blitz's own README: "we would not
yet recommend building apps with it."

So both Tauri and Dioxus fail the cipherpunk no-black-box test.

### Pure-Rust GUI evaluation (all verified live this session)

| Framework | Stars | License | Renderer | Cipherpunk fit | Verdict |
|---|---|---|---|---|---|
| **Slint** | 23.4k | GPLv3 OR Royalty-free OR Commercial | femtovg / skia / **software-CPU** | **Best** — stable 1.x API, no-GPU escape hatch, fully auditable | **USE** |
| **Iced** | 31.2k | MIT | wgpu + tiny-skia | Strong — Elm arch, async first-class, Halloy production proof | USE (runner-up) |
| **egui** | 30.0k | MIT/Apache | wgpu / glow | Good but immediate mode struggles with long-scroll docs | consider |
| **Floem** | 4.2k | MIT | vger/vello/skia/tiny-skia | Pre-1.0, powers Lapce editor, ships virtual_list | consider |
| **Freya** | 2.9k | MIT | Skia | Dark horse — built-in MarkdownViewer (perfect for summary view) | consider |
| **Dioxus** | 38.5k | MIT/Apache | **WebView (default)** / Blitz (pre-alpha) | Fails — same black box as Tauri | **SKIP** |
| **Tauri 2** | n/a | MIT/CPL | **System WebView only** | Fails cipherpunk test | **SKIP** |
| **Xilem** | 5.5k | Apache | Vello/wgpu | Experimental, MSRV 1.92, breaking churn | SKIP (watch only) |
| **Vizia** | 2.2k | MIT | rust-skia | Mature codebase, smaller community | consider |

### Cipherpunk verdict — Slint is the pick

- **Slint GPLv3 option** is one-way compatible with AGPL-3.0 (FSF confirms). Pick
  GPLv3 option #2 from the LICENSE file, NOT the Royalty-free proprietary option.
- **Software renderer** = no GPU dep, no system Vulkan/Metal/DX12 surprise.
  Ship a binary whose rendering deps are just `libpng` + fonts.
- **Stable 1.x API commitment** — write the UI once, don't touch it for two years.
- **Designer toolchain** (LSP, VS Code extension, live preview, Figma plugin).
- **no_std/embedded story** — same code runs on RP2040/STM32 if we ever want
  the GUI on the ESP32-C3 (probably not, but architecturally clean).

Migration safety: keep data/persistence/ML-inference layer framework-agnostic
behind traits; isolate UI in its own crate. Swapping Slint→Iced later = rewriting
only the view layer (hours-to-days, not weeks).

## [HIGH] Summarization model: Flan-T5-base (was Qwen2.5-1.5B)

Verified `candle-transformers/src/models/` directory contents this session:

- **Causal LMs supported:** based, falcon, gemma, gemma2, gemma3, glm4, granite,
  helium, lfm2, llama, llama2_c, mamba, mamba2, mistral, mixformer, mixtral, mpt,
  olmo, olmo2, persimmon, phi, phi3, qwen3, starcoder2, chatglm, codegeex4_9b,
  bigcode, modernbert
- **Seq2seq (summarization-native):** `t5.rs` only (953 lines). That is the
  ENTIRE seq2seq list.
- **NOT supported:** bart, pegasus, distilbart — the canonical summarization models
- **Quantized variants:** quantized_llama, quantized_mistral, quantized_phi,
  quantized_phi3, quantized_qwen2, quantized_qwen3, quantized_qwen3_moe,
  quantized_t5 (in example), etc.

### Top picks for "summarize a 5-page research paper on tower PC CPU"

| Model | Params | GB | candle | Quality 1-5 | tok/s CPU | Verdict |
|---|---|---|---|---|---|---|
| **Flan-T5-base** | 250M | 1.0 fp16 | ✅ `t5.rs` + `quantized-t5` example | 4 (instruction-tuned zero-shot summarization) | 30-50 | **PRIMARY PICK** |
| Qwen2.5-1.5B-Instruct int4 | 1.5B | 1.0 | ✅ `quantized_qwen2.rs` | 4 (chat-model-summarizes) | 8-15 | FALLBACK |
| Llama-3.2-3B-Instruct int4 | 3.2B | 1.9 | ✅ `quantized_llama.rs` | 4.5 | 5-10 | quality upgrade |
| Phi-3-mini int4 | 3.8B | 2.2 | ✅ `quantized_phi3.rs` | 4.5 | 4-8 | quality upgrade |
| Mistral-7B int4 | 7.2B | 4.2 | ✅ `quantized_mistral.rs` | 5 (best quality) | 2-5 | if RAM ≥16GB |
| BART-large-CNN | 406M | 0.8 | ❌ NO `bart.rs` | 5 (purpose-built) | 20-40 | would need port (~1 week) |
| Bonsai-1.7B (1-bit) | 1.7B | 0.24 Q1_0 | ❌ candle doesn't support Q1_0 | 3 (unverified for summary) | unknown | SKIP |
| DistilBERT | 66M | 0.13 | ✅ but encoder-only | n/a — doesn't summarize | fast | WRONG TOOL |

**Final pick: `google/flan-t5-base` via candle's `quantized-t5` example.** Seq2seq
native, ~1 GB, 30-50 tok/s, Apache-2.0, no chat-template gymnastics — just
`summarize: <text>` → summary. Cipherpunk-friendly (small, deterministic, no
telemetry surface).

If Flan-T5 quality on academic prose is too low, swap to Qwen2.5-1.5B-Instruct int4
(same size, modern chat model, slower but smarter).

Honest gap: BART would be technically best for summarization — port to candle is
~1 week if we want it as a differentiator. Track as Phase 6+ possibility.

## Updated v2 stack (locked)

| Layer | Pick | Source |
|---|---|---|
| SNN library | BUILD from audit | justified by ecosystem gap |
| Crypto | RustCrypto (chacha20poly1305 + x25519-dalek fiat) | audited, cipherpunk-correct |
| Embedded target | esp-rs + ESP32-C3 RISC-V | proven, no ML framework fits |
| **Desktop GUI** | **Slint** (was Tauri) | pure Rust, no webview black box, AGPL-compatible |
| **ML inference** | **candle + Flan-T5-base** (was DistilBERT, then Qwen) | seq2seq native, ~1GB, fast |
| Ternary | Standard TWN `{-1,0,+1}` + own format spec; track Prism ML Q1_0/Q2_0_g128 | direction validated |
| Distro | Debian + live-build + packages.chroot | canonical, minimum effort |

## 14-day RVO reality check (Aug 19, 2026) — HISTORICAL, event not attended

> **Closed 2026-08-27.** The RVO Ottawa meetup (Aug 19) was not attended —
> the principal was unavailable to travel. No technical blocker: the demo's
> falsifier had already passed in QEMU (`evidence/qemu-riscv-gate/`, Leg A
> 175/175, Leg B 195/195, 2026-08-21) and `alpha.5` shipped 2026-08-22. The
> library track was unaffected. Kept below as the planning record of the
> time; the EOY crates.io goal is unchanged and remains G0.

Library is #1 priority. Realistic RVO demo options:

- **Ambitious:** Ternary SNN on owned RISC-V (QEMU) with owned crypto — Phases
  0+1+4+5+6 in 14 days. Hard but achievable with focused work.
- **Safe:** Library + CLI demo that runs ternary SNN on QEMU RISC-V and prints
  spike rates. Less flashy, shippable.
- **Cheapest:** Library + property tests + standard summary doc. Hallway-track
  conversation piece, not a live demo.

Principal picks the demo scope. Library starts regardless.

## Open questions resolved

1. ~~Ottawa event~~ — RESOLVED: RVO meetup Aug 19, 2026.
2. ~~Huawei ternary~~ — ACCEPTED from principal's direct knowledge.
3. ~~App ML model preference~~ — Flan-T5-base picked (over Qwen2.5-1.5B).
4. ~~Tauri frontend~~ — Slint picked (pure Rust, no webview).
5. **Bonsai:** confirmed real (Prism ML), skip as runtime, track format.

