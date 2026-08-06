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
