# NeuralOS v2

> **Sovereignty stack** — open ISA, owned code, real crypto, local AI.
> SNN library at the core, Debian distro, RISC-V microkernel, research-summarization app.

[![crates.io](https://img.shields.io/crates/v/neuralos-snn.svg)](https://crates.io/crates/neuralos-snn)
[![Documentation](https://docs.rs/neuralos-snn/badge.svg)](https://docs.rs/neuralos-snn)

## What it is

NeuralOS v2 is a sovereignty stack in Rust — not an "AI OS," but an open software
stack where every layer belongs to you: readable code, auditable crypto, models that
run locally, open hardware ISA (RISC-V).

The original project (NeuralOS v0.1, archived at tag `v0.1-broken-baseline` in
[`Caramoussin/NeuralOs`](https://gitea.com/Caramoussin/NeuralOs)) accumulated 411K
LOC of which only ~6K were real engineering. The rest was theater — empty functions,
`thread::sleep` masquerading as DMA, hardcoded hardware detection. v2 doesn't repeat
those mistakes.

See [`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md) for the 10 theater patterns
we identified and refuse to repeat.

## Status

| # | Component | State |
|---|---|---|
| **1** | `neuralos-snn` — `no_std` SNN library | **In progress.** LIF neuron (Phase 0) and Synapse + STDP rule (Phase 1.1) ported. 37 tests passing. |
| 2 | Research-summarization app (Slint + candle + Flan-T5-base) | Not started. |
| 3 | Debian "Prime AI" distro (custom Live ISO via `live-build`) | Not started. |
| 4 | RISC-V microkernel target (QEMU first, then ESP32-C3 silicon) | Not started. |

**Priority order:** library → app → distro → microkernel. The library is the spine;
everything else inherits from it.

## Stack (locked, research-verified)

| Layer | Pick | Why |
|---|---|---|
| SNN library | Build from v0.1 audit | Rust ecosystem gap confirmed |
| Crypto | RustCrypto (`chacha20poly1305` + `x25519-dalek` `fiat` backend) | Audited, cipherpunk-correct |
| Embedded target | `esp-rs` + ESP32-C3 RISC-V; QEMU RISC-V for free dev | Proven; no ML framework fits → SNN is differentiator |
| Desktop GUI | Slint (GPLv3 option, AGPL-compatible) | Pure Rust, no webview black box |
| ML inference | candle + Flan-T5-base via `quantized-t5` example | seq2seq native, ~1GB, 30-50 tok/s CPU |
| Ternary quantization | Standard TWN `{-1,0,+1}` + own format spec | Track Prism ML Q1_0/Q2_0_g128 |
| Distro | Debian Testing + `live-build` + `config/packages.chroot/` | Canonical, minimum effort |

Full research findings: [`docs/RESEARCH_FINDINGS.md`](docs/RESEARCH_FINDINGS.md).
Competitive landscape: [`docs/landscape/SUMMARY.md`](docs/landscape/SUMMARY.md).
Architecture vision: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Principles (Cardano-grade rigor)

1. **Every line ships with a test or test vector.** No `Ok(())` with "// In a real implementation."
2. **CI gating from day one.** `cargo check` must pass at the workspace level.
3. **One source of truth.** No parallel copies (v0.1 lesson).
4. **Structure earns its existence.** Add a crate when it has real code — no premature scaffolding.
5. **Hexagonal/DDD serves the code, not the other way around.**
6. **Honest names.** If a function is called `init_mmu`, it inits the MMU.
7. **`no_std` by default** for the core library — discipline + enables RISC-V deployment.
8. **No cloud dependencies.** Local AI only. Never an OpenAI/Anthropic API call.

## Workspace

```
NeuralOs-v2/
├── Cargo.toml                # workspace, single member for now
├── crates/
│   └── neuralos-snn/         # SNN library (Phase 0 + 1.1 done)
│       ├── src/
│       │   ├── lib.rs         # error type, exports
│       │   ├── lif_neuron.rs  # Phase 0 — i16 fixed-point LIF (17 tests)
│       │   └── synapse.rs     # Phase 1.1 — Synapse + STDPRule (24 tests)
│       └── Cargo.toml
└── docs/
    ├── ARCHITECTURE.md       # multi-target vision
    ├── ROADMAP.md            # phases, priorities, RVO short-circuit
    ├── AUDIT_PORT_TABLE.md   # what ports from v0.1, with file:line refs
    ├── RESEARCH_FINDINGS.md  # stack-pick research (2 rounds)
    ├── LESSONS_LEARNED.md    # 10 theater patterns to never repeat
    └── landscape/
        └── SUMMARY.md        # competitive landscape scan
```

## Quickstart

```bash
git clone git@gitea.com:Caramoussin/NeuralOs-v2.git
cd NeuralOs-v2
cargo check
cargo test
```

Expected: 37 tests passing, zero warnings under `cargo clippy --all-targets -- -D warnings`.

For the `no_std` verification (target: ESP32-C3 / QEMU RISC-V):

```bash
cargo build --no-default-features --lib
```

## License

AGPL-3.0-or-later. Full LICENSE text will be added before the first stable tag.

## Relation to NeuralOS v0.1

[`Caramoussin/NeuralOs`](https://gitea.com/Caramoussin/NeuralOs) is the archive.
Tag [`v0.1-broken-baseline`](https://gitea.com/Caramoussin/NeuralOs/releases/tag/v0.1-broken-baseline)
freezes the broken state. The gems to port (LIF neuron, STDP, lock-free primitives,
SIMD kernel, event bus, candle BERT wrapper, AES-GCM key, SQLite repos) are listed
in [`docs/AUDIT_PORT_TABLE.md`](docs/AUDIT_PORT_TABLE.md) with their `file:line`
references and per-bug notes.

## Status as of 2026-08-06

Two phases shipped:

- **Phase 0 — LIF neuron port** (commit `71bf09e`): 470 LOC, 17 tests, both audit-flagged bugs fixed (time ownership, noise seed).
- **Phase 1.1 — Synapse + STDP rule port** (commit `bf166f9`): 637 LOC, 24 tests. Property test caught a sign-convention bug v0.1's tests missed; fixed with `factor.max(0)` clamp.

Next: Phase 1.2 Network + 4 topology builders.
