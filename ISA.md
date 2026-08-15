---
task: "NeuralOS v2 — substrate, lab bench, gated ternary bridge"
slug: 20260815-125500_neuralos-v2
project: NeuralOS v2
phase: complete
progress: 10/10
started: 2026-08-15T12:55:00Z
updated: 2026-08-15T14:10:00Z
principal_stated_goal: "ok lets do this — start Stage 2 (format bridge), the step the Stage 1.5 gates earned"
---

## Problem

Stages 1 through 1.5d proved the ternary SNN substrate learns (SI 1.000 =
i16 parity, full pairwise STDP, both bridge canaries green). But the learned
weights live only inside NeuralOS — they speak no format any other ternary
system reads, and no external ternary model's weights can enter ours. The two
ternary ecosystems that actually ship models (Microsoft BitNet b1.58, Prism ML
Bonsai Q1_0/Q2_0) each define their own byte layout; today we interoperate
with neither. The bridge is mechanically open but format-silent.

## Vision

NeuralOS speaks the lingua franca of both ternary fields. An SNN trained in
`neuralos-snn` exports its weights as a BitNet-compatible `i2_s` byte stream
that BitNet-side tooling can consume; a Bonsai `Q1_0`/`Q2_0` tensor decodes
into our `Trit` substrate and runs on the SNN — real weights flow both ways,
bit-exactly, integer-only, `no_std`. Euphoric surprise: the first
`cargo run --example ternary_format_gate` prints round-trip evidence with
byte-level test vectors derived from the actual C/Python reference sources —
not from our own re-implementation of ourselves.

## Out of Scope

- **No GGUF container parsing.** Stage 2 is tensor-level: block layouts and
  scales. Container/metadata/file IO belongs to Stage 4 (running real models).
- **No MLX formats** (Prism's Apple-side `mlx` fork) — import paths noted in
  the spec, not implemented.
- **No `Q1_0` export.** Q1_0 is binary `{−γ, +γ}`; ternary zeros are
  unrepresentable. Export would be silently lossy, so it does not exist.
  Import only (binary ⊂ ternary is lossless).
- **No `NativeTernary` (2.000 bpw, 2026 paper).** No shipping models use it
  yet; it enters as an import path only when something real emits it.
- **No new dependencies, no float types, no `alloc`.** Buffer-based codecs,
  integer-only, like the hot path.
- **No candle / llama.cpp integration.** Stage 3+ territory.

## Principles

- **Authoritative layouts come from the reference sources, verbatim** —
  PrismML-Eng/llama.cpp C code and microsoft/BitNet's conversion script —
  not from blog posts, model-card labels, or our own inference. Where the
  labels disagree with the code (the "g128" case), the code wins.
- **Honest lossiness boundaries.** Every conversion that loses information
  either does not exist (Q1_0 export) or is a loud error (Q2_0 code 3), never
  a silent clamp.
- **Every byte layout claim ships with a test vector** a human can verify
  against the reference source by hand.
- **`no_std` discipline is the point, not the constraint** — a RISC-V edge
  device should be able to decode Bonsai weights with this code as-is.

## Constraints

- Library code in `crates/neuralos-snn` stays `no_std` by default; the bridge
  module uses caller-provided buffers (`&[u8]` in, `&mut [u8]` / `&mut [Trit]`
  out), zero heap.
- Integer-only: no `f32`/`f64` types in the bridge module. fp16/fp32 scales
  are carried as raw bits (`u16`/`u32`); numeric views are fixed-point
  (`milli = round(value × 1000)` in `i32`, saturating).
- i16 fixed-point hot path untouched — the bridge is adjacent to, not inside,
  `lif_neuron`/`synapse`.
- CI gates green: `cargo check --workspace --all-targets`,
  `cargo test --workspace`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo build --no-default-features -p neuralos-snn`.

## Goal

Stage 2 of the ternary bridge ships and the gate answers YES: a `no_std`
format module (`bridge`) in `neuralos-snn` that encodes/decodes
BitNet-compatible `i2_s` tensors and decodes Prism `Q1_0`/`Q2_0` tensors,
proven by `examples/ternary_format_gate.rs` round-tripping a ternary tensor
bit-exactly and byte-level test vectors pinned to the reference sources.

## Claims

- [x] ISC-1: `docs/TERNARY_FORMAT.md` specifies, bit-level, the four
  byte layouts (Trit-native, `i2_s`, `q1_0`, `q2_0`), the code→trit tables,
  the scale conventions (γ=mean|w| vs max|w|), length/alignment rules, and
  the lossiness boundaries — each format with one worked byte example
  identical to a test vector in the crate. Falsifier: file absent, or any
  worked example disagrees with its corresponding unit-test vector.
- [x] ISC-2: `bridge::encode_i2_s(trits, scale_bits, out)` reproduces the
  microsoft/BitNet `i2_s` byte layout — transposed 4-lane packing (element
  `i` → byte `(i>>7)*32 + (i&31)`, shift `6 − 2·((i>>5)&3)`), codes
  {0,1,2}={−1,0,+1}, 32-byte tail with LE f32 scale bits in the first 4
  bytes; requires `n % 128 == 0`. Falsifier: known-vector test against a
  hand-computed byte string derived from the reference script.
- [x] ISC-3: `bridge::decode_i2_s(bytes, out)` is the exact inverse:
  trits and scale bits round-trip losslessly; code 3 is a loud error; short
  buffers are a loud error. Falsifier: proptest round-trip over random
  ternary slices (n multiple of 128) + explicit error-path unit tests.
- [x] ISC-4: `bridge::decode_q1_0(bytes, out)` decodes the Prism fork's
  `block_q1_0` — 18 bytes per 128 weights: LE fp16 scale (γ=mean|w|), then
  16 sign bytes (bit `j%8` of byte `j/8`, set → `+γ`); requires
  `n % 128 == 0`. Falsifier: hand-crafted 128-element vector with known
  sign pattern + scale bits, exact-equality assertions.
- [x] ISC-5: `bridge::decode_q2_0(bytes, out)` decodes the fork's
  `block_q2_0` — 18 bytes per 64 weights: LE fp16 scale (max|w|), then 16
  bytes of 2-bit codes (LSB-first lanes, `00`=−1 `01`=0 `10`=+1); code `11`
  is a loud `UnsupportedCode` error (reference quantizer cannot emit it);
  requires `n % 64 == 0`. Falsifier: hand-crafted 64-element vector covering
  all three codes + an explicit code-3 error test.
- [x] ISC-6: scale plumbing is integer-only and bit-faithful:
  `half_to_f32_bits` widens fp16→fp32 bit-exactly (pure integer), and
  `half_to_milli` gives the fixed-point numeric view with documented
  saturation. Falsifier: known-half vectors (1.0, 0.5, −2.0, max finite,
  max subnormal, zero) asserting exact f32 bit patterns and milli values.
- [x] ISC-7: `examples/ternary_format_gate.rs` runs the Stage-2 gate: builds
  a deterministic ternary tensor (LFSR), encodes `i2_s`, decodes back,
  asserts bit-exact trits + scale; decodes the `q1_0` and `q2_0` test
  vectors; prints per-check evidence and `STAGE 2 GATE: YES|NO`. Falsifier:
  running the example under `cargo run -p neuralos-snn --example
  ternary_format_gate` exits 0 printing YES; any failed check flips NO and
  exits nonzero.
- [x] ISC-8: discipline gates green with the new module: workspace tests,
  clippy `-D warnings`, `no_std` build all pass. Falsifier: any of the three
  CI commands failing.
- [x] ISC-9: docs tell the truth after the run: `docs/VISION.md` carries a
  Stage 2 result section (layouts, gate verdict, "g128" correction);
  `docs/ROADMAP.md` Phase 3 reflects Stage 2 done. Falsifier: grep for the
  Stage 2 sections/updates finding them absent or contradicting the gate
  output.
- [x] ISC-10: the bridge surface is exercised at its consumer seam: at least
  one test imports a decoded tensor's trits into `Trit::to_weight` i16
  space and back without leaving the ternary grid. Falsifier: no test
  connects decode output to the existing `Trit` substrate API.

## Anti-claims

- Anti: no `f32`/`f64` type, literal, field, or cast in
  `crates/neuralos-snn/src/bridge.rs` outside `#[cfg(test)]` (the widening
  helper's *name* contains `f32` because it produces f32 *bits* — integers).
- Anti: no silent clamping in any decode path — wrong lengths, unsupported
  codes, and short buffers return `Err`, never best-effort output.
- Anti: no `Q1_0`/`Q2_0` encode functions exist (we do not author Prism
  files; only read them).
- Anti: no new entries in `[dependencies]` of either crate.
- Anti: no GGUF file parsing in this stage (grep `gguf` in src = doc
  comments only).

## Test Strategy

| isc | type | check | threshold | tool | anchors_to |
|---|---|---|---|---|---|
| ISC-1 | doc | spec file exists; worked examples byte-equal their unit-test vectors | 4/4 formats | Read + grep | docs/TERNARY_FORMAT.md |
| ISC-2 | unit | known-vector encode matches hand-computed bytes | exact | cargo test | bridge::tests::i2_s_known_vector |
| ISC-3 | property | encode→decode round-trip trit+scale equality | 100% cases | cargo test (proptest) | bridge::tests::prop_i2_s_round_trip |
| ISC-4 | unit | q1_0 test vector decodes to exact trits + scale bits | exact | cargo test | bridge::tests::q1_0_known_vector |
| ISC-5 | unit | q2_0 test vector decodes exact; code 3 errors | exact | cargo test | bridge::tests::q2_0_known_vector + q2_0_code3_rejected |
| ISC-6 | unit | fp16 known vectors bit-exact | exact | cargo test | bridge::tests::half_known_vectors |
| ISC-7 | gate | example prints YES, exit 0 | yes | cargo run --example | examples/ternary_format_gate.rs |
| ISC-8 | build | tests+clippy+no_std all green | 0 failures | cargo | CI commands |
| ISC-9 | doc | VISION/ROADMAP Stage-2 sections present and match gate | 2/2 docs | grep | docs/VISION.md, docs/ROADMAP.md |
| ISC-10 | unit | decoded tensor ↔ Trit::to_weight round-trip stays on-grid | exact | cargo test | bridge::tests::decoded_trits_feed_trit_substrate |

## Decisions

- 2026-08-15 · Layouts pinned verbatim from reference sources:
  `block_q1_0` = fp16 `d` (γ=mean\|w\|) + 16 sign bytes per 128 weights;
  `block_q2_0` = fp16 `d` (max\|w\|) + 16×2-bit codes per 64 weights;
  BitNet `i2_s` = transposed 4-lane 2-bit packing per 128 values + 32B tail
  with LE f32 scale. Sources: PrismML-Eng/llama.cpp `ggml-common.h` +
  `ggml-quants.c` (fetched this session), microsoft/BitNet
  `utils/convert-hf-to-gguf-bitnet.py::quantize_to_i2_s`.
- 2026-08-15 · The "Q2_0_g128" label in `docs/RESEARCH_FINDINGS.md` is
  loose: the fork's C code defines `QK2_0 = 64` (group size 64, 2.25 bpw).
  The C code is authoritative; our docs get the correction.
- 2026-08-15 · `i2_s` requires `n % 128 == 0`, not merely `% 4`: the
  reference's transposed packing drops elements when `n % 128 != 0` (bytes
  beyond `n//4` are truncated), so a permissive codec would be silently
  lossy. We reject such lengths.
- 2026-08-15 · Native wire format = BitNet-compatible `i2_s` (export) +
  Trit-native packed (internal convenience); NativeTernary deferred until
  shipping models exist. Grounds: VISION.md's format decision ("BitNet
  Round() native + Prism Q1_0 import") — ecosystems that ship models beat
  paper formats.
- 2026-08-15 · Q1_0/Q2_0 import-only. Binary {−γ,+γ} embeds losslessly into
  ternary; the reverse direction would drop zeros (Q1_0) or invent scale
  semantics (Q2_0 code 3) — dishonest to ship.
- 2026-08-15 · Buffer-based no-`alloc` API (caller-provided slices): the
  crate is `no_std` without alloc today; keeping it that way preserves the
  RISC-V posture (decode Bonsai weights on the edge device).

- 2026-08-15 · Both fog entries (SNN-facing fp16-γ→i16-SCALE=1000
  conversion policy; Trit-native packing's wire role) are Stage 3 scope,
  not this run's — killed here, to be re-articulated as claims in the
  Stage 3 ISA when that run opens.

## Learning

- conjectured: the hand-computed `q1_0` (0xB5) and `q2_0` (0x94) expected
  trit patterns in the known-vector tests were correct.
  refuted by: `cargo test` failures showing decoder output `[+1,−1,+1,−1,…]`
  vs my expected `[+1,−1,+1,+1,…]` (bit extraction order) and
  `[−1,0,0,+1]` vs `[−1,0,+1,+1]` (2-bit lane packing order).
  learned: the *decoder* was right both times — I nearly "fixed" correct
  code to match wrong vectors. Hand-derived byte vectors are the highest
  bug-density artifact in format work.
  criterion now: every hand-computed vector is re-derived from the flat
  index formula (written in the spec) before running, and the property
  tests (which don't depend on hand math) must agree with the known
  vectors on overlap.
- conjectured: `i2_s` needs only `n % 4 == 0` (four 2-bit codes per byte).
  refuted by: reading the reference's output path — it truncates at `n/4`
  bytes after a 128-element transposed reshape, so non-multiple-of-128
  lengths leave live elements in dropped bytes.
  learned: length rules come from the reference's *output truncation*,
  not from the element-packing math alone.
  criterion now: `n % 128 == 0` required + rejection tests
  (`i2_s_rejects_bad_lengths`), rule documented in the spec.

## Verification

- ISC-1: docs/TERNARY_FORMAT.md; worked examples byte-equal
  `bridge::tests::{i2_s,q1_0,q2_0}_known_vector`
- ISC-2: `bridge::tests::i2_s_known_vector` (32 packed bytes + tail vs
  hand-computed `AA 00 55 AA…`)
- ISC-3: `bridge::tests::prop_i2_s_round_trip` +
  `i2_s_rejects_{bad_lengths,short_buffers,code_three}`
- ISC-4: `bridge::tests::q1_0_known_vector` (0xB5 pattern, scale 0x3C00)
- ISC-5: `bridge::tests::q2_0_known_vector` + `q2_0_code3_rejected`
- ISC-6: `bridge::tests::half_known_vectors` +
  `prop_half_widening_matches_f32_reference` (all 65 536 halves)
- ISC-7: `cargo run -p neuralos-snn --example ternary_format_gate` this
  session → `STAGE 2 GATE: YES`, exit 0
- ISC-8: clippy `-D warnings` clean · 115 tests pass · `--no-default-features`
  builds (all run this session)
- ISC-9: VISION.md "Stage 2 — RUN 2026-08-15, result: YES" section +
  ROADMAP.md "Stage 2 (format bridge) — PASSED 2026-08-15"
- ISC-10: `bridge::tests::decoded_trits_feed_trit_substrate`
