# NeuralOS v2

> The open `no_std` spiking-neural substrate for RISC-V edge silicon — and the lab bench to watch it think.

[![crates.io](https://img.shields.io/crates/v/neuralos-snn.svg)](https://crates.io/crates/neuralos-snn)
[![Documentation](https://docs.rs/neuralos-snn/badge.svg)](https://docs.rs/neuralos-snn)

## What this repo is

NeuralOS v2 is a pure-Rust workspace centered on one defensible artifact — `crates/neuralos-snn`, a `no_std`-by-default, i16 fixed-point spiking neural network library — with `crates/neuralos-app` as the Slint visualizer that runs it live as a spike-raster + weight-map microscope, and `crates/neuralos-rt` as the (unpublished) research runtime that proved the ternary bridge. The bridge chapter is closed and honestly recorded: the distilled article lives in `paper/`, the full session history in `docs/RESEARCH_LOG.md` and `ISA.md`.

This is **not** the old “AI OS” pretense from v0.1. It is a serious SNN library, a visualizer that makes it legible, and a closed, gated research record documented in `docs/VISION.md`.

The active repo is **`NeuralOs-v2`**. The sibling `NeuralOS` repo is the archived v0.1 codebase and should be treated as read-only source material.

## North star

Project direction lives in:

- `docs/VISION.md` — the current north star
- `docs/ROADMAP.md` — planned phases and sequencing
- `docs/landscape/SUMMARY.md` — competitive/ecosystem landscape
- `docs/RESEARCH_FINDINGS.md` — evidence behind key stack choices

The short version:

- the spine is the `neuralos-snn` library
- the visible artifact is the Slint visualizer
- the ternary bridge is a **closed research chapter** — record in `docs/RESEARCH_LOG.md`, article in `paper/`

## Workspace overview

```text
NeuralOs-v2/
├── crates/
│   ├── neuralos-snn/   # no_std-by-default SNN library (published, the spine)
│   ├── neuralos-app/   # Slint visualizer running the SNN live
│   └── neuralos-rt/    # research runtime (GGUF / Q1_0 / Q2_0 / tokenizer) — publish = false
├── paper/              # the Branch B article (make / make figs / make gate)
├── evidence/           # raw judge/experiment logs (INDEX.md)
├── docs/               # vision, roadmap, research log, format spec, landscape
├── tools/              # fork build + judge tooling
├── models/             # gitignored GGUF weights (HF downloads + derived)
├── Cargo.toml          # workspace manifest
├── rust-toolchain.toml # pinned Rust toolchain
└── README.md
```

## Current status

| Component | State |
|---|---|
| `neuralos-snn` | Active and validated. `cargo test -p neuralos-snn`, `cargo check --workspace --all-targets`, and `cargo clippy --workspace --all-targets -- -D warnings` pass. Published as `0.1.0-alpha.2`. |
| `neuralos-app` | Active Slint visualizer wired directly to `neuralos-snn`. |
| `neuralos-rt` | Research runtime that proved the ternary bridge. `publish = false`; kept as the executable record behind `paper/`. |
| Ternary bridge | **Closed chapter** — gates adjudicated, record in `docs/RESEARCH_LOG.md`, article in `paper/`. Reopening is a recorded fork, not default work. |
| Distro / crypto / bare-metal deployment | Important longer-term directions, but not the current center of gravity of this workspace slice. |

## Design principles

- **Pure Rust.** No webview stack, no cloud runtime, no black-box platform layer.
- **`no_std` by default for the library.** The hot path stays compatible with embedded and RISC-V targets.
- **i16 fixed-point in the hot path.** No floating-point creep in neuron/synapse core logic.
- **Visualizer as microscope, not product theater.** The app exists to reveal the substrate.
- **One source of truth.** No parallel copies, no empty scaffolding, no fake architecture.
- **Tests must prove real behavior.** Regressions should be caught by unit/property tests, not hope.

## Quality gates

These are the commands the repo is expected to satisfy:

```bash
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --no-default-features -p neuralos-snn
```

The first three were validated in this workspace state.

## Running the visualizer

For normal iteration:

```bash
cargo run -p neuralos-app
```

For a headless smoke-style launch/exit:

```bash
NEURALOS_SMOKE_MS=2500 cargo run -p neuralos-app
```

The app is a live microscope over the SNN:

- spike raster
- synaptic weight heatmap
- STDP learning toggle
- sustained-firing mode for visible activity

## Why this exists

The project claim is narrow and concrete:

- open `no_std` SNN software for edge/RISC-V is an under-served space
- integer, fixed-point SNN computation is a real hardware-aligned design axis
- Rust lacks a mature, serious library occupying that slot

NeuralOS v2 aims to fill that gap with a library first, then make it legible through the visualizer, and only then pursue the more ambitious ternary bridge research path.

## Relation to v0.1

`NeuralOS` v0.1 is archived. It contained some real implementation buried under a large amount of theater, dead structure, and misleading abstractions.

NeuralOS v2 is the cleanup:

- one active workspace
- real code only
- honest naming
- tighter invariants
- tests around real behavior

The archive is still useful as a mine for ports and prior audit findings, but it is **not** the active codebase.

## License

AGPL-3.0-or-later.
