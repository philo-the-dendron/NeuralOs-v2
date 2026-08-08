# AGENTS.md

Read this before touching anything. NeuralOS v2 is the **active** repo — the
fresh start, built by mining the v0.1 archive. Pure Rust, AGPL-3.0, local-only.
Two artifacts share one workspace: a published `no_std` SNN library (the spine)
and a Slint visualizer that runs it live (the microscope).

**Direction lives in `docs/VISION.md` (the north star).** Read it before
scoping work. Evidence base: `docs/ROADMAP.md`, `docs/landscape/SUMMARY.md`,
`docs/RESEARCH_FINDINGS.md`.

## Two repos — don't confuse them

| Repo | Role |
|---|---|
| `NeuralOs-v2` (here) | **Active.** Edit this one. Workspace builds clean. |
| `NeuralOS` (sibling dir) | **Archive** of v0.1 — manifest-broken, READ-ONLY. Mine it for real code (the audit indexed what's worth porting in its `AGENTS.md` + our `docs/AUDIT_PORT_TABLE.md`). Don't build it, don't edit it. |

If `cargo` ever errors with `multiple workspace roots`, you're in the archive,
not here. `cd` to `NeuralOs-v2`.

## The visualizer threading gotcha (READ BEFORE TOUCHING `main.rs`)

The SNN sim runs on a **worker thread**; the UI runs on the Slint event loop.
The bridge between them is the single hardest-won lesson of this codebase:

> **Do NOT check `app_weak.upgrade()` from the worker thread to detect window
> close.** Slint's `Weak::upgrade()` returns `None` from any thread that isn't
> the UI thread — so the worker concludes "window closed" on iteration 1 and
> silently exits. The result: two black panes, no error, no spike ever renders.
> (This took four misdiagnoses to find. Don't repeat it.)

**Correct pattern (already in `main.rs`):** gate the worker's lifetime on a
shared `Arc<AtomicBool>` (`worker_alive`), set false after `app.run()`
returns. Only upgrade the weak **inside** `slint::invoke_from_event_loop` —
that runs on the UI thread where the upgrade is valid.

When verifying the visualizer renders, capture **by window ID**
(`xwd -id <wid>`), not by cropping the root screen — root crops catch the
desktop behind the window and give false "it works" readings.

## Commands

Workspace is two members: `crates/neuralos-snn` (the library) and
`crates/neuralos-app` (the visualizer). `rust-toolchain.toml` pins **1.92.0**
(slint 1.17 MSRV — don't bump without checking).

```bash
# Quality gates (what CI runs, .github/workflows/ci.yml):
cargo check  --workspace --all-targets
cargo test   --workspace                          # offline; live-net tests are #[ignore] where they exist
cargo clippy --workspace --all-targets -- -D warnings
cargo build --no-default-features -p neuralos-snn # the no_std gate (RISC-V/embedded posture)

# Run the visualizer (DISPLAY required, e.g. :0):
cargo run -p neuralos-app --release               # ~2–9 min link on a 2-core CPU; binary cached after
./target/release/neuralos-app                     # rerun without rebuild
NEURALOS_SMOKE_MS=2500 cargo run -p neuralos-app  # headless launch+clean-exit smoke (CI/display-less)

# SIMD benchmark (AVX2 vs scalar, x86_64 only, behind the `simd` feature):
cargo run --example bench_simd --features simd --release
```

**Iterate in debug** (`cargo run -p neuralos-app`, ~7 s build) — release is
only worth the multi-minute link when you want real smoothness.

## Features

- `neuralos-snn`: `default = ["std"]`, `std`, `simd` (implies std, x86_64-only,
  AVX2 batch LIF kernel). The published crate's default config is what CI tests.
- `neuralos-app`: no features (candle/summarizer code was deleted at
  `v0.1-summarizer-demo` tag — the app is the SNN visualizer now, not a
  research summarizer).

## Remotes (push carefully)

`origin` has **two push URLs**: Gitea (`Caramoussin/NeuralOs-v2`, canonical)
and the GitHub mirror (`philo-the-dendron/NeuralOs-v2`). A normal
`git push origin main` hits both. If Gitea has commits you don't (e.g. a web
edit), **rebase, never force-push Gitea**. The GitHub mirror is the only
force-push target, and only when it has desynced pre-rebase history — get
explicit OK first.

## Published crate

`neuralos-snn` is on crates.io at `0.1.0-alpha.1` (12 downloads at time of
writing). The app consumes it via path dep (`../neuralos-snn`), so lib edits
take effect locally without republishing — but a real bugfix or API addition
may warrant an `alpha.2` republish. Bump the workspace `version` in the root
`Cargo.toml` and `cargo publish -p neuralos-snn` when that's the call.

## Conventions that matter

- **Pure Rust, `no_std`-by-default for the library.** The hot path
  (`lif_neuron`, `synapse`) is `no_std`; `network` is `std`-gated; `simd` is
  std+x86_64-gated. Keep new library code `no_std` unless it genuinely needs
  `alloc`/`std`.
- **i16 fixed-point, no float in the hot path.** This is the design axis the
  IEEE 2025 "Full-Integer SNN Inference with RISC-V ISA" paper validates, and
  it's what makes the ternary bridge tractable. Don't quietly introduce f64.
- **GUI = Slint, never Tauri/Dioxus.** System webview is a black box — the
  cipherpunk stance rejects it. Slint GPLv3 option is AGPL-compatible.
- **Crypto = RustCrypto, never from-scratch.** "Own my software" means
  read+pin+understand audited code, not rewrite it worse.
- **Every line ships with a test or test vector.** No `Ok(())` with "// real
  impl later." Property tests (`proptest`) where tractable — the v0.1 lesson.
- **Honest names.** If a function is `init_mmu`, it inits the MMU. (v0.1 was
  full of empty impressive-named stubs. v2 isn't.)

## A library behavior to know

`SpikingNeuralNetwork::step()` calls `decay_synaptic_current` each step so the
`adaptation_current` (which accumulates +2/spike) doesn't grow unboundedly and
silence the network. **If you add new per-step logic, preserve this decay** —
without it every sustained simulation self-quenches to ~0 Hz after ~3 s. There
is also a `set_plasticity_enabled(bool)` toggle (default ON in the lib, OFF in
the visualizer's sustained-firing mode).

## Do not

- Don't create another `libneuralos_*` copy or a v3. Refactor in place;
  rollback via git. (v0.1 died of parallel copies.)
- Don't edit anything in the `NeuralOS` archive repo.
- Don't add empty scaffolding crates — "structure earns its existence."
- Don't write scaffolding, stub functions, or `Ok(())` placeholders.
- Don't reintroduce the research-summarizer app (commodity; re-scoped away).
- Don't check `Weak::upgrade()` off the UI thread (see the gotcha above).
