# NeuralOS v2 — Roadmap

> Source of truth for what's planned, what's locked, what's done.
> Updated 2026-08-05. All stack picks verified by StandardResearch (see `RESEARCH_FINDINGS.md`).

## Priority order (from the principal's head)

| # | Component | Status |
|---|---|---|
| **1** | `neuralos-snn` — `no_std` SNN library | Phase 0 done, Phase 1 in progress |
| 2 | Debian distro (custom Live ISO via `live-build`) | Not started |
| 3/4 | Microkernel RISC-V (QEMU first, then ESP32-C3) | Not started |
| 3/4 | Research-summarization app (Slint + candle + Flan-T5) | Not started |

Ottawa RVO meetup (Aug 19, 2026) is a **parallel demo track**, not the main driver.

## Stack picks (locked, research-verified)

| Layer | Pick | Why |
|---|---|---|
| SNN library | BUILD from v0.1 audit | ecosystem gap verified ([HIGH]) |
| Crypto | RustCrypto (`chacha20poly1305` + `x25519-dalek` `fiat` backend) | audited, cipherpunk-correct, saves 2-3 sessions vs from-scratch |
| Embedded target | `esp-rs` + ESP32-C3 RISC-V; QEMU RISC-V for free dev | proven; no ML framework fits → SNN is differentiator |
| Desktop GUI | **Slint** (GPLv3 option) | pure Rust, no webview black box, AGPL-compatible, stable 1.x |
| ML inference | **candle + Flan-T5-base** via `quantized-t5` example | seq2seq native, ~1GB, 30-50 tok/s on CPU |
| Ternary | Standard TWN `{-1,0,+1}` + own format spec; track Prism ML `Q1_0/Q2_0_g128` | direction validated by Bonsai/BitNet |
| Distro | Debian Testing + `live-build` + `config/packages.chroot/` | canonical, minimum effort |

## Phases

### Phase 0 — Library scaffold + LIF neuron ✅ DONE (commit `71bf09e`)

- Workspace scaffold, single member `neuralos-snn`, AGPL-3.0
- LIF neuron ported from `v0.1/libneuralos/src/core/neural_processing/lif_neuron.rs`
- Both audit-flagged bugs fixed (time ownership, noise seed)
- 17 tests passing (13 unit + 4 property)
- `no_std` verified, clippy clean under `-D warnings`

### Phase 1 — Complete the SNN library (in progress)

| Sub-phase | Source in v0.1 | Est. effort |
|---|---|---|
| 1.1 Synapse + STDP rule | `libneuralos/src/core/spiking_neural_network/synapse.rs` (378 LOC, audit Gem #3) | 1 session |
| 1.2 Network + 4 topologies | `libneuralos/src/core/neural_processing/network.rs` (996 LOC, audit Gem #4) | 1-2 sessions |
| 1.3 STDP plasticity | `libneuralos/src/core/neural_processing/stdp_plasticity.rs` (699 LOC, audit Gem #2) | 1 session |
| 1.4 Sparse synapse matrix (CSR) | inline in network.rs | (in 1.2) |
| 1.5 CLI demo (`neuralos-snn-demo`) | new | 1 session |

**Phase 1 total: ~5 sessions.** Pattern: copy from v0.1 → clean port → fix audit bugs → tests → commit.

### Phase 1 extension (optional, post-RVO) — Lock-free + SIMD from backup

| Piece | Source in v0.1 backup | Notes |
|---|---|---|
| Concurrent LIF | `libneuralos_before_bridge_removal/src/core/lock_free_neural_processing.rs` (596 LOC) | AtomicF32 + CAS spike-gen |
| Thread-pool workers | `libneuralos_before_bridge_removal/src/core/lock_free_neural_network.rs` (710 LOC) | MPMC spike buffer |
| AVX2 SIMD kernel | `libneuralos_before_bridge_removal/src/core/simd_vectorization.rs` (989 LOC) | 101 AVX2 intrinsics, revectorize for RISC-V later |

### Phase 2 — Research-summarization app

Stack: Slint (UI) + candle (ML) + Flan-T5-base + arXiv/PubMed fetchers + SQLite.

| Sub-phase | Est. effort |
|---|---|
| 2.1 Slint hello-world + project scaffold | 1 session |
| 2.2 arXiv/PubMed fetcher + parser | 1 session |
| 2.3 candle Flan-T5 integration + summarize pipeline | 1-2 sessions |
| 2.4 UI: source list, search, summary view | 2-3 sessions |
| 2.5 SQLite persistence for saved summaries | 1 session |

**Phase 2 total: ~6-8 sessions.**

### Phase 3 — Debian "Prime AI" distro

| Sub-phase | Est. effort |
|---|---|
| 3.1 `.deb` package for the app | 1 session |
| 3.2 `live-build` recipe + `config/packages.chroot/` | 1 session |
| 3.3 Theming (NeuralOS Prime session/wallpaper) | 1 session |
| 3.4 Meta-package `neuralos-prime` for stock Debian | 1 session |

**Phase 3 total: ~3-4 sessions.**

### Phase 4 — Ternary quantization

| Sub-phase | Est. effort |
|---|---|
| 4.1 Trit weight type + encoding | 1 session |
| 4.2 Ternary STDP rule | 1 session |
| 4.3 Weight format spec doc + reference impl | 1 session |
| 4.4 Prism ML Q1_0/Q2_0_g128 interop study (read their fork) | 1 session |

**Phase 4 total: ~3-4 sessions.**

### Phase 5 — Crypto layer

| Sub-phase | Est. effort |
|---|---|
| 5.1 `neuralos-crypto` crate: pin RustCrypto, vendor-read, write RFC vector tests | 1 session |
| 5.2 `HardwareBoundKey`-style AEAD wrapper (ChaCha20-Poly1305) | 1 session |
| 5.3 X25519 key exchange wrapper | 1 session |

**Phase 5 total: ~2-3 sessions.** (Saved 2-3 sessions by using RustCrypto vs from-scratch.)

### Phase 6 — RISC-V bare-metal target

| Sub-phase | Est. effort |
|---|---|
| 6.1 QEMU RISC-V `riscv64gc` boot + `no_std` runtime | 1-2 sessions |
| 6.2 Run `neuralos-snn` on QEMU RISC-V | 1 session |
| 6.3 ESP32-C3 (when board arrives): esp-rs + esp-idf scaffold | 1 session |
| 6.4 Sensor → SNN → motor loop on real hardware | 2-3 sessions |

**Phase 6 total: ~5-7 sessions + hardware availability.**

## 14-day RVO short-circuit (Aug 19, 2026)

The RVO meetup is a forcing function for an early demo, not the main sequence. Three scope options:

| Scope | What ships Aug 19 | Risk |
|---|---|---|
| **Safe (recommended)** | Phase 1 complete + CLI demo running ternary SNN on QEMU RISC-V, prints spike rates. Hallway-track demoable. | Low — all proven components |
| **Ambitious** | Safe + owned crypto wrapper (Phase 5.1-5.2) + ternary quantization (Phase 4.1-4.3) + Prism ML format awareness. Live demo on stage. | Medium — 14 days is tight |
| **Cheapest** | Phase 0 + Phase 1.1 only + property tests + summary doc. Conversation piece only. | Very low |

Default if principal doesn't pick: **Safe**.

## Engineering principles (Cardano-grade rigor)

1. **Every line ships with a test or test vector.** No `Ok(())` with "// In a real implementation."
2. **CI gating day 1.** `cargo check` workspace-level must pass before any code lands.
3. **One source of truth.** No parallel copies (leçon v0.1).
4. **Structure earns its existence.** Add a crate only when it has real code — no premature scaffolding.
5. **Honest names.** If a function is called `init_mmu`, it inits the MMU. No theater.
6. **`no_std` by default** for the core library — discipline + enables embedded RISC-V.
7. **No cloud deps.** Local AI only. Never an OpenAI/Anthropic API call.
8. **Every claim cites a verified source.** Per LifeOS constitution.
9. **Don't reinvent wheels.** Use RustCrypto/candle/esp-rs/live-build — open, auditable stacks. "Own my software" = read+pin+understand, not rewrite worse.

## Decision authority (who picks what)

| Decision type | Authority |
|---|---|
| Architecture, stack, phases | Principal (with DA recommendation) |
| Specific function design, tests, code style | DA (Cardano-grade bar) |
| Pushes to remote, new repos, destructive ops | Principal (explicit OK each time, except `~/.claude` trusted repo) |
| Phase ordering changes | Principal |
| Bug-fix decisions during a port | DA (with regression test) |

## Open questions (resolved as of 2026-08-05)

- ~~Ottawa RVO event~~ — RESOLVED: RVO meetup Aug 19, 2026
- ~~Huawei ternary~~ — ACCEPTED from principal's direct knowledge; track Prism ML publicly
- ~~App ML model~~ — Flan-T5-base (candle `quantized-t5` example)
- ~~Tauri frontend~~ — Slint (pure Rust, no webview)
- ~~Crypto approach~~ — RustCrypto (audited, not from-scratch theater)
- ~~Repo mechanics~~ — One repo `NeuralOS` (v0.1 archive), one repo `NeuralOs-v2` (fresh start)
- ~~Backups~~ — Filesystem `NeuralOS-backup-*` ×3 + git tag `v0.1-broken-baseline` (eternal)

## What's done so far

| Date | What | Commit |
|---|---|---|
| 2026-08-05 | v0.1 archived (LESSONS_LEARNED + tag `v0.1-broken-baseline`) | `aeae4bd` on `NeuralOs/architectural-rescue` |
| 2026-08-05 | v2 repo created on Gitea Caramoussin | `NeuralOs-v2` |
| 2026-08-05 | v2 scaffold (workspace + crate skeleton + docs) | `f738036` |
| 2026-08-05 | StandardResearch findings (4 corrections, 5 validations) | `9ef30ec` |
| 2026-08-05 | Follow-up research (Bonsai, Slint, Flan-T5) | `c87fcc1` |
| 2026-08-05 | **Phase 0: LIF neuron ported + tested** | `71bf09e` |

## Immediate next

**Phase 1.1 — Synapse + STDP rule.** Copy `libneuralos/src/core/spiking_neural_network/synapse.rs` from v0.1, clean port, fix audit-flagged issues, add tests, commit. ~1 session.

Confirm and I proceed.
