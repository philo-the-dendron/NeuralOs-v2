---
task: "NeuralOS v2 — substrate, lab bench, gated ternary bridge"
slug: 20260815-125500_neuralos-v2
project: NeuralOS v2
phase: climbing
progress: 29/29
started: 2026-08-15T12:55:00Z
updated: 2026-08-15T21:15:00Z
principal_stated_goal: "ok so we are ready for step 3 ?" → "go" — Stage 3 (shared kernel + hybrid gate), locked choices: A·classification demo, A·absmax i16 activations
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

(ISC-1..10: Stage 2, ISC-11..17: Stage 3, ISC-18..23: Stage 4 session 1 — all closed, see Verification.)

- [x] ISC-24: `neuralos-rt` ships `q1_0_matvec`: row-major Q1_0 blocks ×
  i16 activations → i32, per-block fp16 γ applied in the milli domain
  (`partial × γ_milli / 1000`, i64 intermediate, saturating i32 out —
  bounds documented), `n % 128 == 0` enforced. Falsifier: property test
  vs a decode_q1_0 + scalar-product reference disagrees on any case.
- [x] ISC-25: `q1_0_row_to_milli` materializes one embedding row (2048
  values, ±γ per element, per-block scales) from the real file; sane =
  nonzero variance, |values| ≤ fp16-max-milli. Falsifier: real-row test
  fails bounds or is all-zero.
- [x] ISC-26: integer RMSNorm (`rms_norm_milli`): pure-integer isqrt +
  `y = x·w/rms` in milli with a documented integer eps floor; property:
  isqrt exactness (r² ≤ n < (r+1)²); unit: known vectors + scale-up
  behavior. Falsifier: any test failure.
- [x] ISC-27: `examples/bonsai_forward.rs` executes the real first-layer
  slice — token ids → embedding rows → blk.0.attn_norm RMSNorm →
  q/k/v projections through q1_0_matvec on real q1_0 tensors — printing
  per-stage stats (mean/absmax/nonzero) and exiting 0 only if every
  stage's output is nonzero and finite-bounded. Falsifier: example
  exits nonzero or prints an all-zero/degenerate stage.
- [x] ISC-28: CI gates green workspace-wide with the additions.
  Falsifier: any of the four commands failing.
- [x] ISC-29: docs truth: VISION session-2 note (compute path decision +
  first-layer numbers), ROADMAP line, TERNARY_FORMAT untouched unless
  constants changed. Falsifier: docs absent/contradicting output.

(ISC-1..10: Stage 2 — closed 2026-08-15. ISC-11..17: Stage 3 — closed
2026-08-15. Both see Verification.)

- [x] ISC-18: work happens on `stage4-ternary-runtime`, pushed to both
  remotes (Gitea canonical + GitHub mirror); `main` stays always-green —
  branch merges only at honest milestones with all four CI gates green.
  Falsifier: branch absent/unpushed, or a red-gates merge to main.
- [x] ISC-19: container + type research pinned from the fork's own source
  this session: GGUF layout (magic/version/n_tensors/n_kv, 13 KV value
  types with string=8/array=9, tensor-info order, pow2 alignment default
  32) from `PrismML-Eng/llama.cpp` `gguf.h`+`gguf.cpp`; `GGML_TYPE_Q1_0
  = 41`, `GGML_TYPE_Q2_0 = 42` from its `ggml/include/ggml.h`; recorded
  in ISA Decisions + `docs/TERNARY_FORMAT.md`.
  Falsifier: constants contradict the fetched source (re-fetch diff).
- [x] ISC-20: new workspace member `crates/neuralos-rt` with a
  buffer-based `gguf` module: `GgufFile::parse(&[u8])` returning version,
  KV pairs (all 13 types, arrays flat-only like the reference), tensor
  infos (name, dims, type, offset), alignment, and validated
  `tensor_data(name)` slicing. Reference-faithful validation: magic,
  version ∈ {2,3}, duplicate names, n_dims ≤ 4, pow2 alignment, offsets
  within buffer.
  Falsifier: synthetic-vector tests fail, or any error path untested.
- [x] ISC-21: the parser reads the real `Bonsai-1.7B-Q1_0.gguf` (248 MB,
  HF `prism-ml/Bonsai-1.7B-gguf`): 310 tensors parse, `general.architecture`
  is qwen3, all 310 data slices fall inside the file, and every Q1_0
  tensor's byte size equals `rows × (cols/128) × 18` computed from its
  dims.
  Falsifier: `bonsai_probe` example fails any of these on the real file.
- [x] ISC-22: the Stage-2 codec meets real model bytes: `decode_q1_0`
  decodes the first block of `token_embd.weight` from the real file and
  the fp16 scale lands in a sane milli range (order 1–100 milli).
  Falsifier: decode errors, or scale milli outside [1, 100].
- [x] ISC-23: CI gates green with the new crate: check/test/clippy/
  no_std (snn) — the rt crate is std by design (file IO at the edges,
  parse core on byte slices).
  Falsifier: any CI command failing.

(ISC-1..10: Stage 2 format bridge — all closed 2026-08-15, see Verification.)

- [x] ISC-11: `kernel` module ships the shared `no_std` ternary primitive:
  `pack_trits` (sequential 2-bit, 4 trits/byte), `ternary_matvec`
  (packed row-major weights × i16 activations → i32, |acc| ≤ n·32767
  documented), and `absmax_normalize_q15` (per-vector integer absmax into
  Q15, the BitNet activation-quant analog) — all buffer-based, zero-alloc.
  Falsifier: module absent, or property test vs an unpacked scalar
  reference finds a disagreement.
- [x] ISC-12: `bridge::repack_i2s_to_kernel` converts wire (`i2_s`
  transposed) → compute (sequential) packing bit-exactly without an
  intermediate trit buffer; rejects n%128≠0, code 3, short buffers.
  Falsifier: encode→repack→unpack-sequential round-trip property test
  fails, or an error-path test fails.
- [x] ISC-13: fog №1 resolved: `bridge::wire_gamma_to_substrate(milli)`
  maps an imported fp16 γ (milli view) into the i16 substrate domain via
  `synapse::SCALE` (now `pub`), one saturating formula, coupling pinned by
  a test asserting `SCALE == 1000`. Falsifier: function absent, saturation
  untested, or SCALE still private.
- [x] ISC-14: `examples/ternary_hybrid_gate.rs` runs the Stage-3 gate: one
  balanced 128-neuron ternary SNN (STDP off, fixed γ), 4-group gapped
  round-robin drive (1.5c constants), per-trial spike counts → Q15 absmax
  activations → a 4×128 dense layer whose weights ENTER as `i2_s` wire
  bytes and compute via the same `ternary_matvec` kernel → 4/4 groups
  classified (chance 25%), margins printed. Falsifier: any group
  misclassified, or `cargo run --example ternary_hybrid_gate` exits
  nonzero printing NO.
- [x] ISC-15: the dense layer's weights reach the kernel ONLY through the
  wire path (encode_i2_s → repack_i2s_to_kernel) — the composition is the
  claim. Falsifier: the example hands raw trits to pack_trits directly.
- [x] ISC-16: discipline gates green with the new module: workspace tests,
  clippy -D warnings, no_std build. Falsifier: any CI command failing.
- [x] ISC-17: docs tell the truth: VISION.md gains a Stage 3 result section
  (kernel, gate verdict, honest "constructed not trained" scope) and the
  stages-table row; ROADMAP Phase 3 reflects Stage 3. Falsifier: grep
  finds the sections absent or contradicting the gate output.

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

(Stage-2 anti-claims stand. Stage-3 additions:)
- Anti: no `f32`/`f64` type, literal, or cast in `kernel.rs` outside
  `#[cfg(test)]`.
- Anti: no heap allocation in any `kernel`/`bridge` public function —
  buffer-based signatures only (grep `Vec<\|vec!` in non-test kernel/bridge
  code = zero).
- Anti: no training claim — the dense layer is constructed (+1 on group,
  −1 on other E, 0 on inhibitory), stated in the example header and
  VISION section; composition is the gate, learning was Stage 1.5's.
- Anti: no change to `lif_neuron`/`synapse` hot-path math (SCALE visibility
  change is additive only).

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

## Not yet specified

(Stage-4 fog — sessions 3+, in scope, not yet claims:)

- fog: activation math surface — f32 from scratch (sovereignty-max) vs
  pulling candle-core for ops while keeping Q1_0 ours; revisit when the
  first attention layer is written (bitnet.cpp keeps float activations —
  precedent supports f32).
- fog: tokenizer — Qwen BPE from the GGUF's embedded tokenizer data;
  scope when generation needs it.
- fog: Stage-4 gate definition — "coherent completion from the real 1.7B
  model" needs a measurable bar (fixed prompts + expected-token checks)
  before the generation session.

## Test Strategy

| isc | type | check | threshold | tool | anchors_to |
|---|---|---|---|---|---|
| ISC-24 | property | matvec vs decode-reference on synthetic blocks | 100% | cargo test | rt::q1_0::tests::prop_matvec_matches_reference |
| ISC-25 | integration | real embedding row: nonzero variance, bounded | pass | cargo test (ignored-offline) + example run | rt tests + bonsai_forward |
| ISC-26 | unit+property | isqrt exactness; norm known vectors | 100% | cargo test | rt::norm::tests |
| ISC-27 | gate | example prints stats all-nonzero, exit 0 | yes | cargo run --example | examples/bonsai_forward.rs |
| ISC-28 | build | 4 CI gates | 0 failures | cargo | CI commands |
| ISC-29 | doc | VISION/ROADMAP updated to match | 2/2 docs | grep | docs/ |

| isc | type | check | threshold | tool | anchors_to |
|---|---|---|---|---|---|
| ISC-18 | git | branch exists on both remotes, main untouched | present | git ls-remote | remotes |
| ISC-19 | source-pin | constants match fetched fork headers | exact | diff vs /tmp fetches | ISA Decisions |
| ISC-20 | unit | synthetic GGUF round-trip + error paths | 100% | cargo test | rt::gguf::tests |
| ISC-21 | integration | real-file probe: 310 tensors, sizes, bounds | 100% | cargo run --example bonsai_probe | examples/bonsai_probe.rs |
| ISC-22 | integration | real-file q1_0 block decode, scale sane | milli ∈ [1,100] | cargo run --example bonsai_probe | same |
| ISC-23 | build | 4 CI gates green | 0 failures | cargo | CI commands |

| isc | type | check | threshold | tool | anchors_to |
|---|---|---|---|---|---|
| ISC-11 | unit+property | matvec vs scalar reference; absmax known+props | 100% exact | cargo test | kernel::tests (prop_matvec_matches_scalar et al.) |
| ISC-12 | unit+property | encode→repack→sequential-unpack round trip | 100% exact | cargo test | bridge::tests::prop_repack_round_trip |
| ISC-13 | unit | known milli→substrate vectors + saturation + SCALE==1000 | exact | cargo test | bridge::tests::wire_gamma_known_vectors |
| ISC-14 | gate | example prints 4/4 + YES, exit 0 | yes | cargo run --example | examples/ternary_hybrid_gate.rs |
| ISC-15 | code-inspect | example's dense path goes through encode_i2_s + repack | present | grep | example source |
| ISC-16 | build | tests+clippy+no_std green | 0 failures | cargo | CI commands |
| ISC-17 | doc | VISION/ROADMAP Stage-3 sections present, match gate | 2/2 docs | grep | docs/VISION.md, docs/ROADMAP.md |

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

- 2026-08-15 (session 2) · **Fog №1 resolved: Q1_0 compute path = per-block
  decode-matvec now** (`q1_0_matvec`: sign-bit partial sums × per-block
  γ_milli). The fused/LUT kernel (bitnet.cpp-style) is deferred until
  profiling shows need — correctness first, measured speedups after.
- 2026-08-15 (session 2) · f32 *weights* (norm tensors) convert to the
  milli domain at load (`f32_bits_to_milli`, pure integer from bits) — the
  f32-vs-candle *activation surface* fog stays open but narrows: this
  session's entire compute path is integer (milli + i16 activations).
- 2026-08-15 (session 2) · Model facts pinned from the real file: qwen3,
  28 blocks, emb 2048, 16 Q heads / 8 KV heads (GQA), head_dim 128,
  rms eps 1e-6, tokenizer gpt2 (BPE). Recorded for sessions 3+ (attention,
  MoE-less FFN 6144, generation).

- 2026-08-15 · **From-scratch, no candle (session-1 posture).** The Q1_0
  path is already ours (Stage 2 codecs + Stage 3 kernel); candle supports
  neither Q1_0 nor the fork's type numbers, and its loaders are coupled to
  its own model structs. Revisit trigger (fog): if the f32 op surface
  (attention/RoPE/norm) proves too costly from scratch, pull candle-core
  for ops only — decision recorded here, not silently.
- 2026-08-15 · Pinned from fork source: GGUF container per fork `gguf.h` +
  `gguf.cpp` reader (v3 file, v1 rejected, alignment pow2 default 32,
  arrays flat-only — nested arrays are a reader error in the reference
  too); `GGML_TYPE_Q1_0 = 41`, `GGML_TYPE_Q2_0 = 42` (fork's
  `ggml/include/ggml.h`, past TQ1_0=34). Real-file cross-check: header
  hexdump matches (GGUF v3, 310 tensors, 32 KV).
- 2026-08-15 · `neuralos-rt` is std (file IO at the edges); parse core
  operates on caller-provided byte slices (no_std-friendly later if the
  edge story demands). Model weights live in gitignored `models/` — never
  committed.
- 2026-08-15 · Branch protocol (principal-ratified previous turn):
  `stage4-ternary-runtime` pushed to both remotes; merges to main only at
  honest milestones with green gates; no force-push to Gitea.

- 2026-08-15 · Stage-3 shape ratified by principal: A·classification gate
  (which group fired — proven 1.5c input paradigm, unambiguous 25%-chance
  metric) + A·absmax Q15 activations (BitNet-style per-vector integer
  normalization; bounded by construction). The toy-sequence and raw-count
  alternatives were explained and declined.
- 2026-08-15 · Both Stage-2 fog items graduate here: γ-conversion policy →
  ISC-13; Trit-native packing role → decided as *compute format* under the
  kernel (sequential 2-bit), with `i2_s` remaining the *wire* format
  (ISC-11/12) — repack is the seam between them.
- 2026-08-15 · Dense-layer weights are constructed, not trained (risk named
  pre-run, principal accepted): Stage 3's claim is composition (wire →
  kernel → coherent output), not dense-layer learning.
- 2026-08-15 · `synapse::SCALE` becomes `pub` — the γ policy needs it, and
  the coupling gets one home + a pinning test instead of a magic 1000.

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

- 2026-08-15 · Stage 3 closed: all seven claims on evidence. Dense-layer
  logits came out perfectly symmetric (±486824 etc.) because non-driven
  counts are exactly 0 — the 1.5c containment is total with fixed ternary
  weights, making the margin story cleaner than estimated.

## Learning

- conjectured: writing the example in one pass after the module would
  compile near-clean (the module had just compiled clean).
  refuted by: 6 compile errors in the example — a bogus import name, an
  i16→u32 From that doesn't exist, a u16−u32 mix, a leftover garbage
  arithmetic line, and proptest's macro not supporting inline format
  captures.
  learned: examples are new crates — type-conversion friction that the
  lib's internal conventions absorb does not carry over; and proptest
  macros expand format strings through concat, breaking `{var}` captures.
  criterion now: example-first mental compile check on types across crate
  boundaries; positional format args (`"row {}", r`) in all proptest
  assertions.

## Verification (Stage 3)

- ISC-11: kernel::tests (pack/matvec/absmax known vectors +
  prop_matvec_matches_scalar_reference, prop_absmax_bounds_and_attainment,
  prop_pack_unpack_round_trip) — 129 total green
- ISC-12: bridge::tests::{repack_known_vector, repack_rejects_bad_input,
  prop_repack_round_trip}
- ISC-13: bridge::tests::{wire_gamma_known_vectors, scale_constant_is_pinned}
- ISC-14: `cargo run -p neuralos-snn --example ternary_hybrid_gate` this
  session → 4/4 OK, `STAGE 3 GATE: YES`, exit 0
- ISC-15: example source — dense path is encode_i2_s →
  repack_i2s_to_kernel (grep: 4 refs, no pack_trits shortcut)
- ISC-16: clippy -D warnings clean · 129 tests pass · --no-default-features
  builds (all run this session)
- ISC-17: VISION.md "Stage 3 — RUN 2026-08-15, result: YES" + stages-table
  row + near-term path; ROADMAP.md "Stage 3 (shared kernel) — PASSED"

- 2026-08-15 · Session 1 closed: ISC-18..23 all on evidence. This ISA
  remains OPEN as the Stage-4 carrier (fog = sessions 2+); `phase` stays
  `climbing` between sessions by design — resume reads this artifact.

## Verification (Stage 4, session 1)

- ISC-18: branch `stage4-ternary-runtime` created + pushed (see push
  output below)
- ISC-19: constants pinned from fork `ggml/include/ggml.h` (Q1_0=41,
  Q2_0=42) + `gguf.h`/`gguf.cpp` reader; recorded in Decisions +
  docs/TERNARY_FORMAT.md §GGUF container
- ISC-20: rt::gguf::tests — 11 passing (synthetic round-trip, alignment,
  magic/version/dup/dims/pow2/count/nested-array error paths,
  truncated-tail contract)
- ISC-21: bonsai_probe on the real 248 MB file — 310 tensors in-bounds,
  197 q1_0 byte-exact vs dims, architecture qwen3
- ISC-22: token_embd first block — scale 0x26f0 = 27 milli ∈ [1,100],
  signs +65/−63
- ISC-23: check/test/clippy green workspace-wide this session (see run
  output); no_std gate re-run at commit time

- 2026-08-15 · Session 2 closed: ISC-24..29 on evidence. Fog №1 (Q1_0
  compute path) graduated to a Decision + shipped code. Remaining fog is
  sessions 3+: f32 surface, tokenizer (gpt2 confirmed), gate bar.

## Learning

- conjectured: hand-computed RMSNorm expectations ([848, 1131]) matched
  the implementation.
  refuted by: the code produced [849, 1132] — I'd computed rms from the
  unrounded mean; the code rounds mean(x²) first.
  learned: third session in a row where hand-derived expected values were
  wrong while the implementation was right — the pattern is now
  structural: derive expected values FROM the documented algorithm steps,
  or don't hand-derive at all (property-test against an independent
  reference instead).
  criterion now: hand-vectors only for bit layouts (Stage 2's strength);
  numeric tests prefer reference-implementation comparisons (matvec test
  already does this — the pattern to copy).

## Verification (Stage 4, session 2)

- ISC-24: rt::q1_0::tests::{matvec_matches_reference_single_block,
  matvec_matches_reference_multi_block_multi_row, rejects_bad_sizes}
- ISC-25: rt::q1_0::tests::row_to_milli_matches_decode + bonsai_forward
  emb stage (2048/2048 nonzero, absmax 8–35 milli)
- ISC-26: rt::norm::tests (f32 known vectors, isqrt exactness 0..10k +
  perfect squares at scale, norm unit/uniform/zero/scale-invariance)
- ISC-27: bonsai_forward on the real file → all stages sane,
  `FORWARD: OK`, exit 0
- ISC-28: 4 CI gates green workspace-wide (153 tests total)
- ISC-29: VISION session-2 section + ROADMAP sessions 1–2 line
