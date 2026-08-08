# NeuralOS v2

> The open `no_std` spiking-neural substrate for RISC-V edge silicon — and the lab bench to watch it think.

[![crates.io](https://img.shields.io/crates/v/neuralos-snn.svg)](https://crates.io/crates/neuralos-snn)
[![Documentation](https://docs.rs/neuralos-snn/badge.svg)](https://docs.rs/neuralos-snn)

## What this repo is

NeuralOS v2 is a pure-Rust workspace centered on one defensible artifact — `crates/neuralos-snn`, a `no_std`-by-default, i16 fixed-point spiking neural network library — with `crates/neuralos-app` as the Slint visualizer that runs it live as a spike-raster + weight-map microscope.

This is **not** the old “AI OS” pretense from v0.1. It is a serious SNN library, a visualizer that makes it legible, and a forward research direction documented honestly in `docs/VISION.md`.

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
- the forward chapter is the ternary bridge, explicitly staged and gated

## Workspace overview

```text
NeuralOs-v2/
├── crates/
│   ├── neuralos-snn/   # no_std-by-default SNN library
│   └── neuralos-app/   # Slint visualizer running the SNN live
├── docs/               # vision, roadmap, research, landscape
├── Cargo.toml          # workspace manifest
├── rust-toolchain.toml # pinned Rust toolchain
└── README.md
```

## Current status

| Component | State |
|---|---|
| `neuralos-snn` | Active and validated. `cargo test -p neuralos-snn`, `cargo check --workspace --all-targets`, and `cargo clippy --workspace --all-targets -- -D warnings` pass. |
| `neuralos-app` | Active Slint visualizer wired directly to `neuralos-snn`. |
| Ternary bridge | Planned research direction, documented in `docs/VISION.md`. |
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
