---
task: "NeuralOS research-summarization desktop app (Slint + candle + Flan-T5)"
slug: 20260806-154310_neuralos-app
project: NeuralOs-v2
phase: climbing
progress: 15/19
started: 2026-08-06T15:43:10Z
updated: 2026-08-06T16:40:00Z
principal_stated_goal: "Start the app. Create neuralos-app crate, scaffold Slint UI, integrate candle + Flan-T5 for summarization, add arXiv/PubMed fetcher."
principal_stated_goal_source: conversation
principal_stated_goal_signal: 4
principal_stated_goal_locked: 2026-08-06T15:43:10Z
context_sufficient: true
interview_invoked: false
---

## Problem

The SNN library (`neuralos-snn`) is at alpha and published, but NeuralOS has no
user-facing artifact — nothing a person can run and point at the real world. The
sovereignty stack stays abstract until a researcher can type a query, fetch real
papers, and read a locally-generated summary on hardware they own, with no packet
leaving the machine after the fetch. Phase 2 (ROADMAP) is "Not started." This run
opens it.

## Vision

A Slint desktop app — pure-Rust render stack, no webview black box. A researcher
types a query, sees real arXiv hits, clicks one, and a Flan-T5 summary appears
generated on the tower CPU. Euphoric surprise: a paper summary in ~30s with zero
AI egress, in a binary whose entire render+ML+IO stack is auditable Rust, on owned
hardware. **This run delivers the first vertical slice**: a window that opens, a
search that returns live arXiv titles, wired through a framework-agnostic core —
proving the architecture end-to-end before the 1GB model lands (2.3) or SQLite
(2.5) ship.

## Out of Scope

- **No Flan-T5 / candle inference this slice.** Needs a ~1GB model download and a
  HuggingFace fetch path — its own slice (ROADMAP 2.3). The `Summarize` seam is
  defined but not implemented.
- **No PubMed this slice.** arXiv first (ROADMAP 2.2 partial); PubMed E-utilities
  is a later clone of the `Fetch` seam.
- **No SQLite persistence this slice** (ROADMAP 2.5). Fetched Papers live in UI
  memory only.
- **No cloud AI, ever.** Local-only inference is the cipherpunk posture, not a
  toggle. No OpenAI/Anthropic calls in any slice.
- **No accounts, telemetry, auto-updater, web/mobile target.** Desktop, single
  user, one machine.
- **No neuralos-snn runtime dependency this slice.** Nothing uses it yet; an
  unused dep violates "structure earns its existence." Revisit when an SNN pane
  is wired (see Not yet specified).

## Principles

- Local AI only. A feature that needs a cloud model is a different product.
- No black-box webview. Slint (pure Rust), not Tauri/Dioxus. Renderer must be
  auditable; software-renderer escape hatch preserved.
- Structure earns its existence. No empty crates, no unused deps, no `Ok(())`
  stubs marked "// real impl later." Every module closes a real claim.
- Framework-agnostic core behind traits; UI isolated. Swapping Slint→Iced later
  rewrites only the view layer.
- Honest names and honest verification. "Should work" is forbidden; a window
  either opened (verified) or it didn't.

## Constraints

- Workspace member at `crates/neuralos-app` — one source of truth (ROADMAP §3).
  Single crate with a `lib` (framework-agnostic core) + `bin` (Slint UI), so the
  view layer is replaceable without touching fetch/parse/models.
- Rust stable, edition 2021, AGPL-3.0-or-later (workspace-inherited).
- Slint GPLv3 option (AGPL-compatible per FSF), version 1.17.x; femtovg backend
  (freetype+fontconfig present in-dev), software renderer kept as fallback.
- Blocking HTTP via `ureq` from a `std::thread` worker — smaller audit surface
  than async tokio+reqwest; UI event loop never blocks. Results hand back via
  `slint::invoke_from_event_loop`.
- Atom XML parsed with `quick-xml` (streaming, small, no unsafe) — never a DOM lib.
- `Fetch` is a trait; `ArxivFetcher` is one impl, `MockFetcher` enables offline
  UI tests. Default `cargo test` is fully offline; live-network tests carry
  `#[ignore]`.

## Goal

"Start the app. Create neuralos-app crate, scaffold Slint UI, integrate candle +
Flan-T5 for summarization, add arXiv/PubMed fetcher." — this run delivers the
first vertical slice: a `neuralos-app` workspace member whose Slint window
searches live arXiv and renders real titles through a trait-backed core, with the
`Summarize`/`Store` seams defined for later slices. Flan-T5 integration,
PubMed, and SQLite are explicitly deferred to named later slices.

## Features

### F0 · Cross-cutting

Why: the app has to live in the workspace cleanly, share the SNN crate's rigor,
and be verifiably free of the things it must never contain (cloud AI, panics,
hidden network calls in tests).

- [x] ISC-1: `neuralos-app` is a workspace member; `cargo check -p neuralos-app`
  succeeds and workspace-level `cargo check` still succeeds with the SNN crate
  unaffected.
- [x] ISC-2: The crate exposes a `lib` (framework-agnostic core: `Paper` model +
  `Fetch` trait + `FetchError`) and a `bin` (Slint UI); `cargo build -p
  neuralos-app` builds both.
- [x] ISC-3: `cargo clippy -p neuralos-app --all-targets -- -D warnings` is clean.
- [x] ISC-4: Anti (cloud-AI) — `rg -i "openai|anthropic|api\.openai|api\.anthropic"`
  over `crates/neuralos-app/src` returns empty.
- [x] ISC-5: Anti (panic-free prod) — `rg "\.unwrap\(\)|\.expect\(|panic!"` over
  non-test `src` returns empty (mirrors the SNN crate's leçon v0.1).
- [x] ISC-6: Anti (offline-default tests) — default `cargo test -p neuralos-app`
  makes zero network calls; every live-network test carries `#[ignore]`.

### F1 · Fetch layer

Why: before any model can summarize, the app must turn a free-text query into real
`Paper` records through a swappable seam — so arXiv, PubMed, and a mock all share
one shape.

- [x] ISC-7: `Paper` captures id, title, authors (`Vec<String>`), summary, abs_url,
  pdf_url, published (ISO-8601 string), primary_category — the fields arXiv Atom
  actually exposes.
- [x] ISC-8: `trait Fetch { fn search(&self, query: &str, max_results: usize) ->
  Result<Vec<Paper>, FetchError>; }` is the one seam; `ArxivFetcher` and
  `MockFetcher` both implement it.
- [x] ISC-9: A real captured arXiv Atom fixture parses to exact known field values
  (unit test asserts title, author count, abs_url, pdf_url, primary_category).
- [x] ISC-10: `ArxivFetcher::query_url` builds the correct Atom endpoint URL for a
  query + max_results (unit test asserts the encoded URL byte-for-byte).
- [x] ISC-11: A `#[ignore]`d live test calls real arXiv for "spiking neural" and
  asserts ≥1 Paper with a non-empty title and an `https://arxiv.org/abs/...`
  abs_url.

### F2 · Slint UI shell

Why: the architecture is only proven when a human action (type + click) flows
through the worker thread, the `Fetch` seam, and back into rendered UI text —
deterministically, without the model or a network call in the default path.

- [x] ISC-12: The Slint UI is declared in `ui/app.slint` and compiles with zero
  diagnostics via `slint-build`; it exposes `search-query` (string property),
  `search-clicked` (callback), and `results` (model of `{title, abs_url}`) to Rust.
- [x] ISC-13: `cargo run -p neuralos-app` launches the Slint event loop and exits
  cleanly on window close (verified headless via Xvfb + software renderer; pixel
  appearance is `[DEFERRED-VERIFY]` pending a display, named in Verification).
- [x] ISC-14: A search with the offline `MockFetcher` populates the `results` model
  deterministically — entering a query and clicking search makes canned titles
  appear, no network, no model (deterministic UI-behavior test).

### F3 · Summarize layer (ROADMAP 2.3 — in progress)

Why: local summarization is the differentiator that makes this a sovereignty tool
rather than a reader. A model-agnostic `Summarize` trait (mirrors `Fetch`) keeps
the model choice non-binding — the engine is shipped Qwen2.5-1.5B-Instruct int4
(the principal's pick over Flan-T5-base), and Flan-T5 / Llama-3.2-3B are one
trait-impl swap away.

- [x] ISC-15: A `Summarize` trait + `SummarizeError` live in the framework-agnostic
  core (lib), mirroring `Fetch`; `MockSummarizer` implements it for offline tests.
- [ ] ISC-16: A `QwenSummarizer` implements `Summarize`, loading
  `bartowski/Qwen2.5-1.5B-Instruct-GGUF` (`Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`,
  ~1 GB int4) via `candle-transformers`' quantized qwen2 + `hf-hub`, downloading
  on first summarize into a local cache.
- [ ] ISC-17: A `summarize-smoke` example binary downloads the model, summarizes a
  real arXiv abstract, and prints tok/s (the on-this-CPU benchmark).
- [ ] ISC-18: The summarize prompt applies the Qwen2.5 chat template + a
  summarize instruction; output is the generated summary string.
- [ ] ISC-19: Anti (still local-AI) — `Summarize` runs fully on-device via candle;
  no `openai`/`anthropic`/cloud identifiers introduced with the model path.

### F4 · Persistence (deferred — ROADMAP 2.5)

Why: saved summaries make the tool accumulate value, but a DB schema is premature
until the summarize pipeline shapes the row.

_ISCs deferred to Not yet specified._

### F5 · PubMed source (deferred — ROADMAP 2.2 remainder)

Why: a second `Fetch` impl proves the seam generalizes beyond arXiv. Cloned after
arXiv + summarize prove the spine.

_ISCs deferred to Not yet specified._

## Not yet specified

- fog: does the app ever invoke `neuralos-snn` at runtime? The principal's context
  names it "a dependency," but this slice has no call site. Resolves when we
  decide whether the app ships an SNN-demo/visualization pane or the dependency
  is monorepo cohesion only. — must resolve before F3 lands (so the trait surface
  is right).
- fog: does headless verification of the Slint window need a standing Xvfb in CI,
  or do we accept "event loop launches + exits clean" as the floor and gate pixel
  checks behind a display? — resolves at F2 verification. (Partly resolved: dev
  box has `DISPLAY=:0`, so local verify needs no Xvfb; CI strategy still open.)
- fog: SQLite schema (one-table JSON blob vs normalized paper+summary) — defers to
  F4 and depends on what F3's summarize output actually is.
- fog: full-PDF summarization (needs a PDF→text step) — abstract-first is the F3
  scope; full text is a later sub-slice once a PDF parser is chosen.

## Test Strategy

```yaml
- isc: ISC-1
  type: build
  check: workspace + new crate compile
  threshold: exit 0
  tool: cargo check --workspace
  anchors_to: principal_stated_goal

- isc: ISC-2
  type: build
  check: lib + bin both produce artifacts
  threshold: cargo build -p neuralos-app exits 0; lib + bin targets present
  tool: cargo build -p neuralos-app
  anchors_to: principal_stated_goal

- isc: ISC-3
  type: lint
  check: clippy pedantic clean
  threshold: zero warnings under -D warnings
  tool: cargo clippy -p neuralos-app --all-targets -- -D warnings
  anchors_to: ISC-5

- isc: ISC-4
  type: anti-grep
  check: no cloud-AI identifiers in source
  threshold: empty match
  tool: rg -i "openai|anthropic|api\.openai|api\.anthropic" crates/neuralos-app/src
  anchors_to: Out of Scope

- isc: ISC-5
  type: anti-grep
  check: no unwrap/expect/panic outside tests
  threshold: empty match in non-test src
  tool: rg "\.unwrap\(\)|\.expect\(|panic!" crates/neuralos-app/src
  anchors_to: Principles

- isc: ISC-6
  type: anti-network
  check: default test run is offline
  threshold: zero #[ignore] tests run; no network deps in default path
  tool: cargo test -p neuralos-app (inspect live tests are all #[ignore])
  anchors_to: Constraints

- isc: ISC-7
  type: unit
  check: Paper fields parse from fixture
  threshold: exact asserted values
  tool: cargo test -p neuralos-app --lib parse::paper_from_fixture
  anchors_to: ISC-9

- isc: ISC-8
  type: compile
  check: Fetch trait + two impls typecheck
  threshold: compiles with ArxivFetcher + MockFetcher
  tool: cargo build -p neuralos-app --lib
  anchors_to: ISC-2

- isc: ISC-9
  type: unit
  check: real Atom fixture -> known Paper
  threshold: title/author-count/abs-url/pdf-url/primary-category match fixture
  tool: cargo test -p neuralos-app --lib arxiv::tests::parse_real_fixture
  anchors_to: ISC-7

- isc: ISC-10
  type: unit
  check: query URL is byte-exact
  threshold: encoded URL equals expected string
  tool: cargo test -p neuralos-app --lib arxiv::tests::query_url_encoding
  anchors_to: ISC-8

- isc: ISC-11
  type: integration-live
  check: live arXiv returns results
  threshold: >=1 Paper, non-empty title, abs_url matches ^https://arxiv.org/abs/
  tool: cargo test -p neuralos-app --lib -- --ignored live_arxiv
  anchors_to: Vision

- isc: ISC-12
  type: build
  check: slint file compiles via slint-build with no diagnostics
  threshold: zero diagnostics; generated bindings link
  tool: cargo build -p neuralos-app (slint-build stderr clean)
  anchors_to: ISC-2

- isc: ISC-13
  type: smoke-run
  check: binary launches event loop, exits clean on close
  threshold: exit 0 under Xvfb + SLINT_BACKEND=software
  tool: xvfb-run -a SLINT_BACKEND=software cargo run -p neuralos-app (or flagged DEFERRED)
  anchors_to: Vision

- isc: ISC-14
  type: behavior
  check: MockFetcher search populates UI results deterministically
  threshold: canned titles appear for any query, no network/model
  tool: cargo test -p neuralos-app --test ui_mock_search
  anchors_to: ISC-8
```

## Decisions

- 2026-08-06 15:43: Crate location = workspace member at `crates/neuralos-app`,
  not a separate repo. Reasoned default over the "or separate project" fork in the
  principal's context — ROADMAP §3 "one source of truth" + the existing
  `crates/neuralos-snn` member make this near-unambiguous. Redirect if a split was
  intended.
- 2026-08-06 15:43:: `neuralos-snn` NOT added as a runtime dep this slice. The
  context calls it "a dependency," but no call site exists yet and an unused dep
  violates "structure earns its existence" + trips clippy. Fog recorded; revisit
  at F3.
- 2026-08-06 15:43: Single crate with `lib` + `bin` (not separate `neuralos-app`
  + `neuralos-app-ui` crates). The lib holds the framework-agnostic core so the
  view layer is swappable; splitting crates now is premature structure.
- 2026-08-06 15:43: Blocking `ureq` over async `reqwest`+`tokio`. Smaller audit
  surface, no runtime to reason about, cipherpunk-aligned; UI responsiveness is
  preserved by running fetch on a `std::thread` and returning via
  `slint::invoke_from_event_loop`.
- 2026-08-06 15:43: Pixel-verification of the Slint window is `[DEFERRED-VERIFY]`
  on a headless host — the honest floor is "event loop launches + exits clean
  under Xvfb." Named follow-up, blocks nothing in this slice.
- 2026-08-06 16:18: Resolved the headless-verify fog: the dev box has a real X
  display at `DISPLAY=:0` (xdpyinfo reachable), so no Xvfb was needed locally.
  Added an env-gated `NEURALOS_SMOKE_MS` hook in `main.rs` so CI/headless runs
  can still verify launch+clean-exit-0 without a display or interaction.
- 2026-08-06 16:18: No independent second look elected for this slice (claim 11).
  Not publish-bound/auth/core-surface this run (no `cargo publish`), so the
  visibility floor doesn't mandate a Log row; correctness rests on the test
  suite + clippy + live integration + pixel stats. Principal can request a
  fresh-context review (`Agent(subagent_type:"Max")`) before the app's first
  release — flagged for the F3/Phase-2.3 close.
- 2026-08-06 16:18: Fog in `## Not yet specified` is intentionally retained as
  the persistent app-ISA's cross-slice backlog (F3 model UX, F4 schema, the SNN
  dependency question, CI display strategy). Each graduates when its slice runs;
  none were pursued this run, so none block `phase: complete` for these 14 claims.
- 2026-08-06 16:40: Model pick — principal chose **Qwen2.5-1.5B-Instruct (int4)**
  over Flan-T5-base, accepting the CPU cost for quality. Hardware reality
  (probed this session): i5-6200U, 2C/4T, ~8.6 GB free RAM — *not* the "tower PC"
  the session context assumed. Q4_K_M gguf (`bartowski/Qwen2.5-1.5B-Instruct-GGUF`,
  ~1 GB) is the sweet spot: 6× Flan-T5-base's params at the same footprint.
  Llama-3.2-3B-int4 stays one trait-impl swap away for max quality. Acquisition =
  download-on-first-run (principal's pick) via `hf-hub` into a local cache.
- 2026-08-06 16:40: Engine-first sub-slice — shipping the framework-agnostic
  `Summarize` core (trait + error + `MockSummarizer`, ISC-15) before the candle
  `QwenSummarizer`, so the heavy integration (candle first-compile + 1 GB
  download + quantized-loader) drops into a verified seam. The UI wiring (summary
  pane + download progress) follows after the engine benchmark proves the model
  runs acceptably on this i5.

## Learning

- 2026-08-06 | conjectured: quick-xml errors on truncated XML (`<feed><entry><title>oops</title>`), so the `FetchError::Parse` path would close on a truncated fixture.
  refuted by: the malformed-xml test panicked — `parse_arxiv_atom` returned `Ok([])` with no entry closed.
  learned: truncation is not an error in quick-xml; only structurally illegal tokens surface as Err (e.g. an undefined entity rejected by `Text::unescape`).
  criterion now: the Parse-path test drives `&nosuchent;` through unescape; ISC-9's suite is parse_real_fixture + malformed + empty.

- 2026-08-06 | conjectured: build the `VecModel` on the worker thread and move it into `slint::invoke_from_event_loop`.
  refuted by: slint models are `Rc`-backed (not `Send`); the closure failed to cross threads.
  learned: cross the thread boundary with plain `Send` data (`Vec<(String,String)>`), build the `Rc`-backed Slint model on the UI thread inside `invoke_from_event_loop`.
  criterion now: main.rs ships `Send` data and builds `TitleRow`/`VecModel` on the UI side — recorded as the worker→UI contract.

- 2026-08-06 | conjectured: the project's `channel = "stable"` pin would compile slint 1.17.
  refuted by: local `stable` resolved to rustc 1.89.0; slint 1.17.1 MSRV is 1.92 → resolver error.
  learned: pin a project-local toolchain (`rust-toolchain.toml` `channel = "1.92.0"`) so the global default is untouched for other projects.
  criterion now: `rust-toolchain.toml` pinned to 1.92.0 with the reason in-comment.

- 2026-08-06 | conjectured: the dev box is the "tower PC" the session context named, so a 3B+ model would summarize comfortably and Flan-T5-base was just the "safe" pick.
  refuted by: `free -h` / `nproc` showed a 2016 mobile i5-6200U (2C/4T) with 8.6 GB free — CPU-bound, not RAM-bound.
  learned: probe the actual hardware before sizing a local model; the binding constraint (CPU vs RAM) flips the model tradeoff entirely.
  criterion now: model choice is non-binding via the `Summarize` trait; Qwen2.5-1.5B-int4 ships first and the on-device benchmark (ISC-17) gates any move to a 3B model.

## Verification

- ISC-1: `cargo check --workspace` exit 0 (neuralos-snn + neuralos-app both compile).
- ISC-2: `cargo build -p neuralos-app` → lib + bin artifacts both link.
- ISC-3: `cargo clippy -p neuralos-app --all-targets -- -D warnings` → 0 warnings, exit 0.
- ISC-4: Grep `(?i)openai|anthropic|api\.openai|api\.anthropic` over `src` → No files found.
- ISC-5: Grep `\.unwrap\(\)|\.expect\(|panic!` over `src` → 8 hits, all inside `#[cfg(test)] mod tests`; production code panic-free.
- ISC-6: default `cargo test -p neuralos-app --lib` → 8 passed, 1 ignored (the only live-network test).
- ISC-7: `parse::tests::parse_real_fixture` asserts every Paper field on the captured fixture.
- ISC-8: `ArxivFetcher` + `MockFetcher` both implement `Fetch`; lib compiles (exit 0).
- ISC-9: captured fixture (NeuroCoreX, primary `cs.NE`, 5 authors) parses to exact asserted values.
- ISC-10: `query_url_encoding` + `query_url_drops_extra_whitespace` assert the byte-exact URL.
- ISC-11: `cargo test -- --ignored live_arxiv_search_returns_papers` → 1 passed in 0.53s (real arXiv).
- ISC-12: `slint-build` compiles `ui/app.slint` with zero diagnostics; bindings link (`cargo build` exit 0).
- ISC-13: `NEURALOS_SMOKE_MS=800 cargo run` on `DISPLAY=:0` → exit 0; `xwininfo -name NeuralOS` → window 0x4600035; captured PNG (1280×720) has 5542 distinct colors + full RGB spread → non-degenerate render. Pixel itself human-unviewed (model is text-only); PNG kept at `/tmp/neuralos.png`.
- ISC-14: `tests/ui_mock_search.rs` passes — MockFetcher → present_titles is deterministic and query-independent.
- Regression: `cargo test -p neuralos-snn` → 59 passed, 0 failed (toolchain pin + new member broke nothing).
- ISC-15: `Summarize` trait + `SummarizeError` in lib; `MockSummarizer` impl; `mock_summarizer_is_deterministic_and_offline` passes; `cargo clippy -p neuralos-app --all-targets -- -D warnings` clean.
