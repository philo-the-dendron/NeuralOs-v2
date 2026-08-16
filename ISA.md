---
task: "NeuralOS v2 — substrate, lab bench, gated ternary bridge"
slug: 20260815-125500_neuralos-v2
project: NeuralOS v2
phase: climbing
progress: 43/43
started: 2026-08-15T12:55:00Z
updated: 2026-08-16T21:30:00Z
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

(Closed: ISC-1..10 s2, 11..17 s3, 18..23 s4-s1, 24..29 s4-s2, 30..36 s4-s3, 43 C-pre — see Verification.)

- [ ] ISC-44: tokenizer stale-rank bug fixed — BPE heap entries validated
  against the push-time rank (fork's text-equality check); regression
  trap in CI (synthetic wrong-rank shape) + real-file pin
  (" France" = 9625, full prompt ids). Falsifier: bpe_stale_rank_entry_does_not_fire_early + real_france_prompt_tokenization_pinned.
- [ ] ISC-45: attention score unit chain corrected (dot is milli²; milli
  score = dot × 88.3883 / 1e6, was /1e3 → every score 1000× too large →
  softmax saturated to hard argmax) — all three sites + the circular
  reference rewritten in real units. Falsifier: attention_pipeline test
  (honest reference) + drift harness numbers.
- [ ] ISC-46: q1_0 matvec applies γ at fp16-EXACT precision (integer
  mantissa×2^shift), milli quantization removed from the compute path;
  hostile-scale fallback preserves Session-A saturation; references
  exact-decoded, regression pin added. Falsifier: matvec tests incl.
  matvec_gamma_is_fp16_exact_not_milli.
- [ ] ISC-47: drift converged past C-pre's bar: teacher-forced fork
  comparison argmax 36/36 steps across p2/p3/p4, overlap ≥9/10 mean,
  max |Δtop| 0.597 (was 5.407); per-block int-vs-f64 error ≤1.1%
  decaying (was 15.5% injection at block 0). Falsifier: harness tables
  (evidence below) + f64 reference validated against the fork (±0.03).
- [ ] ISC-48: frozen gate re-run verbatim: 3/5 strict (unchanged), but
  continuations now byte-identical to the fork's greedy (" Paris, and
  the capital of Spain…", " four four the first part…", ": 10:00 AM -
  12") — the faithful-runtime state; chat demonstrator coherent; NO
  verdict stands, now demonstrated at generation level. Falsifier: gate
  log + fork logs side-by-side.
- [ ] ISC-49: 4 CI gates green (195 tests incl. real-file suite:
  France pin, vocab pins, incremental≡forward exact); probe/forward/
  full green on the real file. Falsifier: any failing.
- [ ] ISC-50: docs truth (VISION C-core section, ROADMAP); ISA ledger
  updated; merge call presented to the principal (push held per
  protocol). Falsifier: docs absent/contradicting.

(ISC-1..10: Stage 2, 11..17: Stage 3, 18..23: Stage 4 s1, 24..29: Stage 4 s2, 30..36: Stage 4 s3, 37..42: Stage 4 s4, 43: C-pre — closed, see Verification.)

- [x] ISC-37: `rt::token` ships the gpt2 byte-level BPE tokenizer from
  the GGUF's embedded data (`tokenizer.ggml.{model,pre,tokens,
  token_type,merges}`): GPT-2 byte↔unicode table, Qwen2
  pre-tokenizer as a hand-rolled scanner pinned verbatim from the
  fork's `unicode_regex_split_custom_qwen2` (contractions /
  optional-char+letters / SINGLE digit / space+punct+newlines /
  `\s*[\r\n]+` / run-minus-one / run — zero new dependencies), BPE
  by the fork's (rank, left-index) priority-queue semantics, special
  partition (control+user_defined, longest first), literal special
  decode. Falsifier: byte-table pins, scanner vectors + lossless
  property, BPE rank/position vectors, synthetic round-trips fail;
  real-file: counts, pinned special ids, table-derived expected ids,
  round-trips (all green).
- [x] ISC-38: incremental decode — `Qwen3::{new_session, prefill,
  step}` with a persistent append-only KV cache; `forward_inner`
  untouched; the incremental path reproduces the full forward
  BIT-EXACTLY (tolerance 0) on a nonzero synthetic model in CI and on
  the real Bonsai file (4-token prompt + 1 appended token, all
  positions). Session carries the fog-(g) residual witness.
  Falsifier: either exact-equality test finding one differing i32.
- [x] ISC-39: greedy generation — `Qwen3::argmax_logit` (O(vocab)
  scan, ties to lowest id, same units/degenerate contract as
  `topk_logits`) + the deterministic decode loop in the gate example;
  stop on eos 151645 or a cap. Falsifier: tie-break/peak tests, or
  two gate runs disagreeing (greedy must be deterministic — verified:
  identical output across two full runs).
- [x] ISC-40: `examples/bonsai_generate.rs` — THE GATE: 5 strict
  prompts (chosen before any run, rationale in the header) judged by
  decoded-TEXT prefix against pinned expected strings (ids never the
  contract — Qwen splits digits, " 8" is two tokens) + 1 chat
  demonstrator via the embedded template (fragments asserted against
  the template text; structural pass only, NEVER verdict-bearing);
  YES = 5/5 strict AND all residuals under the soundness rail; tok/s
  per phase printed. Falsifier: 4/5 strict (or fewer) printing NO
  with evidence and exit 1.
- [x] ISC-41: the gate RAN (release, 2026-08-16) — verdict **NO:
  3/5 strict**. PASS: "1 2 3 4 5 6 7"→" 8 9 10 11 1",
  "10 11 12 13"→" 14 15 16 17", "The capital of France is"→" Paris,
  which is the capital of France…". FAIL: "one two three four"→
  "-digit numbers. The problem is that the numbers are not unique",
  "Monday Tuesday Wednesday"→": 10:00 AM\nWednesday: ". Chat
  demonstrator (structural, non-verdict): PASS — "Sure! Here's how
  you can count from 1 to". Residuals 18.1–29.1 M under the 66.6 M
  rail (13→33 positions). Decode 0.22 tok/s, prefill 0.26–0.29 tok/s
  (2-core, release). No prompt-fishing: the set and expected strings
  were fixed pre-run; the failures are recorded as the result.
  Falsifier: the recorded log contradicting any of this.
- [x] ISC-42: discipline + protocol: 4 CI gates green (193 tests),
  ignored real-file tests 3/3 green on the final code, probe/forward/
  full still green, VISION/ROADMAP carry the honest NO, branch
  pushed; **no merge to main — the gate said NO, the merge call is
  the principal's** (Session C's first act: the fork-logit
  comparison, see Decisions).   Falsifier: any gate red, docs absent,
  or an unprincipled merge.

- [x] ISC-43 (C-pre): the 3/5 NO is ATTRIBUTED by reference
  comparison: PrismML-Eng/llama.cpp @ 9ca265a (branch prism,
  CPU-only scratch build, /tmp/opencode/cpre) run
  greedy-by-construction (`--temp 0 --top-k 0 --top-p 1.0 --min-p
  0.0 --seed 42 -no-cnv`) on all 5 frozen prompts FAILS THE SAME
  TWO and passes the same three (p0/p1 continuations byte-identical
  to ours over 24 greedy steps); at the verdict step the reference's
  own top-10 puts " five" at rank 4 (10.517 vs its argmax " four"
  11.486) and " Thursday" outside its top-10 — the two failures are
  the model's under greedy, not our runtime's. Counter-evidence
  recorded honestly (both named Session-C findings, neither
  gate-bearing): logit-level fidelity is NOT within tolerance
  (top-logit deltas up to 5.4, top-10 overlap 5-8/10, later-step
  argmax flips on near-ties) and our tokenizer splits "France"
  ([434, 34106] vs fork [9625]). The gate example, prompts, expected
  strings, and verdict logic are untouched (re-run reproduces the
  recorded NO, content-identical); work committed LOCALLY only — no
  push, no merge (the principal's gate). Falsifier: any captured
  log line contradicting the tables in Decisions/Verification, a
  pushed branch, or a merge commit on main.

- [x] ISC-30: `rt::math` ships the integer softmax machinery — a 1024-entry
  Q12 `2^frac` table, `exp_q12(x_milli)` via max-subtract + exp2
  decomposition, `softmax_q12(logits_milli) -> probs Q12` summing to 4096
  exactly (remainder-corrected). Falsifier: f64-reference property test
  (max prob deviation > 3/4096) or sum ≠ 4096.
- [x] ISC-31: `rt::math::silu_milli` (integer, via the same exp table):
  zero at zero, sign-correct, matches an f64 reference within a
  documented tolerance. Falsifier: any test failure.
- [x] ISC-32: `rt::math::RopeTables` — YaRN per the fork's `ops.cpp`
  (`ramp = 1−clamp((i0/2−low)/(high−low))`, `theta = interp·(1−r) +
  extrap·r`, `mscale·(1+0.1·ln(1/freq_scale))`, corr_dims via the
  fork's `corr_dim` formula, beta_fast=32/beta_slow=1, base 1e6, factor
  4, orig_ctx 8192); cos/sin in milli computed at the load edge (std
  f64, integer after). `apply_rope` rotates half-split pairs. Falsifier:
  angle-0 identity, norm preservation ±tolerance, f64-reference rotation
  on random vectors.
- [x] ISC-33: `rt::q1_0::matvec_scaled` — the unit-chaining wrapper
  (absmax-normalize to i16 → `q1_0_matvec` → rescale by amax/32767)
  returning true milli-domain outputs. Falsifier: reference test vs an
  f64 dequant path disagrees beyond rounding tolerance.
- [x] ISC-34: `rt::model::Qwen3` loads config + tensors from the real
  file (incl. tied-output detection: no `output.weight` → logits =
  h·emb^T) and runs `forward(tokens) -> hidden` through ALL 28 blocks:
  per block attn_norm → QKV → per-head q/k RMSNorm → YaRN RoPE → GQA
  scores/√128 → integer softmax → context → out proj → residual →
  ffn_norm → gate/up → SiLU → down → residual. Falsifier: unit tests
  on attention internals (single-head vs f64 reference) fail.
- [x] ISC-35: `examples/bonsai_full.rs` (release) runs a 4-token prompt
  through the real model: per-block health gates (hidden nonzero,
  bounded, no i32 rails), final `output_norm` RMSNorm, last-position
  logits over the tied embedding, top-5 token ids printed, wall time
  reported, exit 0 only if all gates pass. Falsifier: any block
  degenerate, exit nonzero, or absurd runtime (>5 min release).
- [x] ISC-36: CI gates green workspace-wide; docs truth (VISION
  session-3 note with numbers, ROADMAP line).
  Falsifier: any CI command failing or docs absent/contradicting.

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
- fog (from s3.5 review, deferred tasks, named):
  (a) reference-logit equivalence — **DISCHARGED 2026-08-16
  (C-pre)**: ran against the fork itself (Decisions + Verification,
  s4-C-pre entries); outcome: gate-verdict equivalence PROVEN (the
  reference fails the same two prompts), logit-level equivalence
  REFUTED within tolerance (deltas up to 5.4) — the residual
  question moved to (g2);
  (b) alpha.2 republish checklist adds: `#[non_exhaustive]` on all
  error enums, const-hoisting policy for bridge format consts,
  `decode_q1_0` scale_bits_out API friction decision;
  (c) gguf lazy arrays — parse memory can reach ~32× file size on
  hostile `array of u8` blobs (documented in gguf.rs; typed/lazy
  storage is the real fix);
  (d) tensor-layout contiguity validation (offsets in order, aligned;
  today's exact-size consumers contain it);
  (e) probe scale-window [1,100] is empirical (real value 27), not a
  derived bound — widen when a second real model arrives;
  (f) incremental KV decode (forward currently re-runs the full prompt;
  generation session must own this);
  (g) residual watch: real-model residual absmax reached 17.9M of the
  66.6M norm-soundness rail at 4 tokens — session 4 should gate it
  per-layer (forward_with_health already returns it) and watch growth
  with prompt length. (Graduated s4 — Session::max_abs_residual.)
  (g2) logit-drift tolerance policy (NEW, C-pre): fork-vs-ours
  top-logit deltas measured at −0.24…−5.41 with top-10 rank
  shuffles and later-step argmax flips on near-ties — decide the
  acceptance bar (and whether/how to narrow it) plus the
  tokenizer "France" split fix; owned by Session C's delta
  redteam.

## Test Strategy

| isc | type | check | threshold | tool | anchors_to |
|---|---|---|---|---|---|
| ISC-37 | unit+property+real | byte table, scanner vectors + lossless, BPE rank/pos, specials, round-trips; real: counts, special ids, expected ids | 100% | cargo test (real: --ignored) | rt::token::tests |
| ISC-38 | unit+real | incremental vs full forward, ALL positions + appended token | exact (tol 0) | cargo test (synthetic CI + real --ignored) | rt::model::tests::{incremental_matches_forward_synthetic_exact, real_incremental_matches_forward_exact} |
| ISC-39 | unit | argmax tie→lowest id, peak row wins, agrees with topk[0] | exact | cargo test | rt::model::tests::argmax_tie_breaks_lowest_and_finds_peak |
| ISC-40 | gate | example prints per-prompt evidence + verdict, exit code honest | yes | cargo run --release --example | examples/bonsai_generate.rs |
| ISC-41 | evidence | recorded run: 3/5 strict, deterministic across 2 runs, NO + exit 1 | as recorded | run log (this session) | ISA Verification |
| ISC-42 | build+doc+git | 4 CI gates, ignored 3/3, probe/forward/full green, docs truth, branch pushed, NO merge | 0 fail | cargo + git | CI + docs/ + remotes |
| ISC-43 | evidence | reference E2E 5/5 prompts + fork/ours logit captures; verdict-step argmax table; frozen-gate re-run content-identical; local commit only | as recorded | llama-completion + refcmp + run logs (ISA Verification) | Decisions 2026-08-16 (C-pre) |

| isc | type | check | threshold | tool | anchors_to |
|---|---|---|---|---|---|
| ISC-30 | property | softmax vs f64 ref; exact Q12 sum | ≤3/4096 dev; sum=4096 | cargo test | rt::math::tests |
| ISC-31 | unit+property | silu vs f64 ref, zero, sign | tol documented | cargo test | rt::math::tests |
| ISC-32 | unit+property | rope identity/norm/f64-rotation | tol documented | cargo test | rt::math::tests |
| ISC-33 | unit | matvec_scaled vs f64 dequant ref | rounding tol | cargo test | rt::q1_0::tests |
| ISC-34 | unit | attention internals vs f64 ref | tol | cargo test | rt::model::tests |
| ISC-35 | gate | bonsai_full on real file, release | all gates, exit 0 | cargo run --release --example | examples/bonsai_full.rs |
| ISC-36 | build+doc | 4 CI gates + docs | 0 fail | cargo + grep | CI + docs/ |

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

- 2026-08-16 (C-core) · **Root cause of the C-pre logit drift: a 1000x
  unit error in the attention score chain.** dot is milli² (real×1e6);
  milli scores need dot × 88.3883/1e6; the code divided by 1e3. Every
  attention distribution collapsed to near-one-hot (secondary context
  mass lost per head per block ≈ the measured 15.5% block-0 injection).
  Found by the f64 microscope (per-block Frobenius + stage-by-stage
  diff), validated by the f64 reference matching the FORK's logits to
  ±0.03 — the reference was the independent witness both times.
- 2026-08-16 (C-core) · Tokenizer fix: heap entries carry the push-time
  rank and are validated at pop (fork semantics: text-equality). The
  stale-rank shape (entry pushed for (l,m); m grows via lower-rank
  merges; grown pair fires at the stale position) produced [" F",
  "rance"] on " France" vs the fork's single token 9625.
- 2026-08-16 (C-core) · Exact-γ (fp16 mantissa×2^shift, integer) in
  q1_0_matvec: measured NOT the drift driver (kept anyway — strictly
  more faithful, milli-γ carried 0.4–1.9% per-block error). Ladder items
  ②③ (norm-weight/rope precision) NOT needed for the bar — residual
  drift (max 0.6/18 logit) sits under it; recorded as future polish.
- 2026-08-16 (C-core) · forward_block_states diagnostic API added
  (substep snapshots: emb, attn-l, ffn-l) — permanent instrument, used
  by the harness; capture flag keeps normal forward allocation-free.

- 2026-08-16 (s4-C-pre) · **THE 3/5 NO IS ATTRIBUTED — the reference
  fails the same two prompts.** Session C's first act (the fork-logit
  comparison, ISC-42's named dependency) RAN and is **DISCHARGED** —
  this entry is the evidence pointer. Reference:
  PrismML-Eng/llama.cpp @ 9ca265a57f85f2117942490f421f64a226dd9847
  (branch prism), llama-completion built CPU-only in /tmp (cmake
  4.4.2 from a pip wheel unpacked to /tmp — no system mutation; see
  Verification for the exact commands), greedy forced by construction
  (`--temp 0 --top-k 0 --top-p 1.0 --min-p 0.0 --seed 42 -no-cnv` —
  the tool auto-enables chat mode on template-bearing models, so
  `-no-cnv` is mandatory). E2E: fork fails "one two three four"
  → " four four the first part…" and "Monday Tuesday Wednesday"
  → ": 10:00 AM - 12"; passes the same three, with p0 " 8 9 10 11 1"
  and p1 " 14 15 16 17" BYTE-IDENTICAL to ours. Verdict-step logits:
  p3 argmax ":" both (8.554 / 7.917); p4 on the fork's ids argmax
  " Paris" both (18.385 margin 4.45 / ours 12.978 margin 0.77); p2
  argmax differs (fork " four" @11.486 with " five" rank 4 @10.517,
  margin 0.97; ours "-digit" @11.250 near-tie over " four" @11.003,
  " five" rank 13 @9.197) — both runtimes deny " five" the top.
  **The merge case is presented to the principal; NOT executed this
  session** (protocol: local commit, stop, wait — the push gate and
  the merge call are the principal's).
- 2026-08-16 (s4-C-pre) · **Two honest findings the comparison
  surfaced (recorded, not fixed — Session C delta-redteam scope):**
  (1) **Tokenizer divergence, OUR side**: "France" → ours
  [434 "Fr", 34106 "ance"] vs fork [9625 "France"]; the reference
  BPE reaches the single token, ours does not — p4's E2E comparison
  is VOID per the session pin and its logit comparison ran on the
  fork's ids instead; p0–p3 tokenizations match ours exactly. A
  real rt::token bug candidate (merge-chain application), not
  papered over. (2) **Logit-level fidelity to the f32 reference is
  NOT within tolerance**: top-logit deltas −0.24…−5.41
  (content-dependent), top-10 set overlap 5–8/10, argmax flips at
  p2 s0 (0.25-margin near-tie blob), p3 s8 (same pair {198,481},
  opposite order, our margin 0.29 / fork 0.69), p4 s6/s8/s10/s11
  after the first flip — compounding integer-vs-f32 numerics. A
  systematic YaRN config mismatch is RULED OUT (the GGUF carries
  rope.scaling type=yarn factor=4.0 orig 8192 = our pinned values,
  and p0/p1 match byte-exact for 24 steps, which a systematic
  scaling error would poison; NB: session-3.5's "orig_ctx ABSENT"
  note was checking a shorter key name than the file's
  `qwen3.rope.scaling.original_context_length`). The integer path
  never promised f32 exactness — but the drift is now MEASURED, and
  a tolerance decision belongs to Session C.

- 2026-08-16 (s4) · **THE GATE SAID NO — Stage 4 closes unmerged by
  design.** Honest verdict recorded: 3/5 strict (digit counting ×2 and
  factual recall pass; word-sequence continuations fail). Per gate
  doctrine the bridge stops here with shipped artifacts: the format
  bridge, shared kernel, GGUF container, full integer forward, and now
  tokenizer + incremental decode + deterministic generation are all
  real, tested, and green — what failed is the 1-bit 1.7B's
  *capability* on two prompts, not the runtime (the chat demonstrator
  replies coherently; failures are continuation-shaped, not
  garbage-shaped). Branch pushed, main untouched; **the merge call is
  the principal's.** **Session C's first act: the fork-logit
  reference comparison** (fog (a)) — pin PrismML-Eng/llama.cpp logits
  for a fixed prompt against ours BEFORE any quality judgment beyond
  this gate; the deferred dependency now has a name and a reason
  (two failed prompts are within the range a subtly-off logit could
  cause, and equivalence would sharpen the NO into "model capability"
  vs "runtime drift").
- 2026-08-16 (s4) · Pre-tokenizer = hand-rolled scanner, no `regex`
  crate. Rationale: the pattern needs `(?!\S)` lookahead which `regex`
  does not support (only fancy-regex would, a heavy add); the fork
  ITSELF hand-rolls (`unicode_regex_split_custom_qwen2`); the scanner
  is ~100 deterministic lines pinned rule-by-rule from that source.
  **The pinned pattern has `\p{N}` — ONE digit per piece** — not the
  `\p{N}{1,3}` of the GPT-4/llama3 patterns the plan drafted from
  recall; source-pinning caught it before a single test ran.
  Documented deviations: Rust-std unicode classes (`is_alphabetic` ≈
  \p{L}+wider), no unassigned-cp distinction in the punct rule, ASCII
  case-folding for contractions — none affect ASCII gate text.
- 2026-08-16 (s4) · Tokenizer facts that shaped the gate: Qwen splits
  digits (`" 8"` = `Ġ`+`8`, no `Ġ8` token, no multi-digit tokens) —
  strict prompts are judged by decoded-TEXT prefix, ids are never the
  contract (principal pin №2). The Bonsai chat template (read from the
  file) inserts NO default system prompt on the non-tools path
  (unlike official Qwen3) and its `add_generation_prompt` block
  pre-closes thinking: `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  — rendering asserts each fragment verbatim against the embedded
  template string instead of shipping a Jinja engine.
- 2026-08-16 (s4) · Incremental decode is a NEW path; `forward_inner`
  is byte-untouched (mission constraint). Equivalence is by
  construction (attention is position-local given the caches; the FFN
  is position-local; same arithmetic order — integer exactness makes
  order-identical computation bit-identical) AND pinned by tests:
  synthetic nonzero model in CI + the real file (tolerance 0), both
  green on the final code. `Session` grows per position (~229 KB), not
  by `max_pos` up front. fog (f) graduated.
- 2026-08-16 (s4) · Residual watch (fog (g)) graduated into evidence:
  `Session::max_abs_residual` per layer boundary; the gate prints it
  per prompt and vetoes on the 66.6 M rail. Growth observed this run:
  18.1 M @ 13 positions → 29.1 M @ 33 positions — well under the rail
  at gate-scale prompts; longer generations keep the witness.
- 2026-08-16 (s4) · Hardening carried into reviewed code (same class
  as the s3.5 OOV/65-token fixes, no math change on real files):
  `topk_logits` and `argmax_logit` bound their embedding scan by the
  tensor's actual row count, not the VOCAB const — real files are
  identical (151 669 rows), synthetic/short tensors no longer panic.
- 2026-08-16 (s4) · Sampling NOT implemented (temperature/top-k were
  the stretch pressure valve): greedy-only, per "honesty over reach".
  The file's `general.sampling.*` KVs (temp 0.5, top_k 20) are
  recorded as provenance for a future session.
- 2026-08-16 (s4, principal pins) · Gate design locked pre-run:
  6 prompts = 5 strict + 1 structural chat demonstrator; **YES = 5/5
  strict in the gate logic itself** (no partial credit, no near-miss
  language — 4/5 prints NO + exit 1); strict PASS = decoded-text
  prefix vs pinned expected string, ids resolved at runtime only; on
  NO, record evidence and name the fork-logit comparison as Session
  C's first act (done, above).
- 2026-08-16 (s4) · Deferred fog ledger, owners named: (a) reference-
  logit equivalence → **Session C, first act**; (b) alpha.2 hygiene
  bundle (`#[non_exhaustive]`, const hoisting, decode_q1_0 friction) →
  Session C publish checklist; (c) gguf lazy arrays → Session C; (d)
  tensor contiguity validation → Session C; (e) probe scale-window
  derivation → stays open until a second real model arrives.

- 2026-08-15 (s3.5) · **Adversarial review dispositions** (10 agents;
  every finding adopted / rebutted / deferred — full ledger, severity
  order):
  - **Adopted (fixed + tests this session):** YaRN ramp window (element
    index i0 = 2·pair fed to the pinned formula — was one octave high;
    window pairs 17..34 pinned in tests); softmax exact-sum (floor +
    largest-remainder — the remainder-carry version summed 4097 at
    n=4; counterexamples + tie-family tests added); f32_bits_to_milli
    small-exponent decade (shift ≥ 64; early-return e < −34, signed
    clamp, e ≥ 0 saturates by sign — f64 sweep over all exponents ×
    mantissas pins it); fp16 −inf γ saturation in q1_0_row_to_milli
    (−(i32::MIN) panic → i64-clamp saturation); 65-token score panic
    (Vec-sized) + OOV token panic (embed guard + `TokenOutOfRange`);
    matvec_scaled contract (validates data/out sizes before any work —
    zero path included; checked rows·row_bytes); round-half-away drift
    (FFN + rope rounded half-up on negatives; one shared
    `div_round_half_away`, f64-tested, replaces every inline site);
    rms_norm Σx² checked (release silent wrap → always-on panic with
    message); topk logits rescaled to true milli (doc was false);
    loader dims validation (transposed tensors reject) + config-KV
    cross-checks (`ConfigMismatch`); `absmax_normalize_q15` → u16
    (32768 wrap); golden i2_s/q1_0/q2_0 vectors (lane AND byte order —
    the period-8 known vector was permutation-blind); exhaustive
    65 536-pattern fp16→milli pin; gguf HashSet dup-scan + sorted-pass
    slice validation (two O(n²) DoS paths); u32 gate compares in
    examples (i32::MIN wrap hole); per-layer forward health
    (`ForwardHealth` — dead-attention layers can no longer pass
    bonsai_full); residual soundness rail 6.66e7 (not the i32 rail,
    which sat 32× too high); `rt` publish = false; rt reuses snn's
    Q1_0_BLOCK/BYTES (single-sourced); doc-truth fixes (tensor_data
    inference rule, alignment fallback policy, v2 acceptance, exp2
    octave "2^−11", silu cliff, mscale²-into-cos/sin lineage note,
    score-scale split rationale, # Panics sections); bonsai_forward on
    matvec_scaled (true milli labels); probe fixes (print order,
    197/197 totals, u128 size math, rope-KV provenance printed).
  - **Rebutted (reason):** `matvec_scaled` → `matvec_milli` rename
    (documented in ISA/VISION; churn > gain); `forward(&mut self)` →
    `&self` (session-4 KV cache needs mut — comment added); topk
    error-on-degenerate-hidden (example gates catch it upstream; doc
    states the sentinel); examples' `.expect` panics (exit-code contract
    still holds, scope is examples); `Q10Error` rename (doc line added
    instead); negative-finite f32 milli −i32::MAX vs MIN one-LSB
    asymmetry (fixed anyway by the signed-clamp — moot).
  - **Deferred (named tasks in Not-yet-specified fog list):**
    reference-logit equivalence test; alpha.2 hygiene bundle
    (non_exhaustive, const hoisting, decode_q1_0 friction); gguf lazy
    arrays; contiguity validation; probe scale-window derivation;
    incremental KV decode; per-layer residual gating in session 4.
- 2026-08-15 (s3.5) · Honest correction of session-3 evidence: the
  "per-block health gates" claim for bonsai_full (ISC-35) was in fact
  final-hidden-only — a dead attention layer would have passed. The
  claim is now true (per-layer deltas via `forward_with_health`), and
  the top-5 changed after the YaRN fix (expected; the recorded run was
  made with subtly wrong RoPE tables — ranking plausibility, not
  correctness evidence). ISC-30's "sums to 4096 exactly" was true only
  for n ≤ 3 + luck; now true for all n by construction.
- 2026-08-15 (s3.5) · Rope provenance from the real file (probe now
  prints it): `qwen3.rope.freq_base = F32(1000000.0)` ✓ matches pin;
  `qwen3.rope.original_context` ABSENT — the 8192 default is our pin
  from the fork's runtime config, documented as such (load() checks it
  when present).

- 2026-08-15 (s3) · YaRN pinned verbatim from the fork's `ggml-cpu/ops.cpp`
  (`rope_yarn` + `rope_yarn_ramp` + `ggml_rope_yarn_corr_dims` in
  `ggml.c`): factor 4 → freq_scale 0.25, mscale_total = 1·(1+0.1·ln 4),
  beta_fast 32 / beta_slow 1, base 1e6, orig_ctx 8192. cos/sin tables at
  the load edge (std f64 constants — same doctrine as f32 norm weights:
  compute-path stays integer).
- 2026-08-15 (s3) · Integer softmax = Q12 exp2 table (1024 entries, load
  edge) + max-subtract + `2^(int+frac)` decomposition, clamp below 2^-12
  to 0. SiLU shares the table. probs sum to 4096 exactly (last element
  carries the remainder).
- 2026-08-15 (s3) · Scale chain: activations live in milli (i32);
  matvec input = absmax-normalized i16, output rescaled by amax/32767
  (`matvec_scaled`); attention score scale 1/√128 as 88.3883‰ in i64;
  softmax logits in milli; GQA 16Q/8KV head_map h→h/2.
- 2026-08-15 (s3) · Logits = tied embedding (no output.weight in file —
  verified): logits[t] = h·emb_t over 151669 rows; top-5 ids only
  (tokenizer = session 4).

- 2026-08-15 (principal directive) · **crates.io republish deferred to
  stage completion.** `neuralos-snn` stays `0.1.0-alpha.1` on crates.io
  until Stage 4 closes (gate YES/NO recorded); then one deliberate
  `alpha.2` (or `0.1.0`) bump ships `bridge` + `kernel` + `pub SCALE`
  (+ any later API) together — one republish at a stable point, not one
  per session. Pre-publish checklist at that time: workspace version
  bump, CHANGELOG-worthy commit summary, `cargo publish -p neuralos-snn
  --dry-run`, then publish.

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

- 2026-08-15 · Session 3 closed: ISC-30..36 on evidence. Merged to main
  as milestone 2 after green gates.

## Learning

- conjectured: the integer sigmoid's rounding term was
  `(e·Q12 + Q12 + e/2)/(Q12+e)`.
  refuted by: the f64-reference silu test (off by ~20% at x=−5.9) — the
  correct round-half-away numerator is `(e·Q12 + (Q12+e)/2)`.
  learned: rounding-helper formulas need the same reference-testing as
  the math itself; "round to nearest" written three ways in one file is
  two too many.
  criterion now: one shared round-half-away helper per module, tested
  once against f64, used everywhere (model.rs still has inline sites —
  session-4 cleanup candidate).
- conjectured (session-2 learning inverted): when integer and reference
  disagree, the integer side is wrong.
  refuted by: the attention-pipeline ×1000 mismatch — the REFERENCE
  double-divided by 1000; the integer code was right.
  learned: the f64 reference is also code and also wrong sometimes;
  disagreements get settled by deriving units from first principles on
  paper, not by assuming which side is guilty.
  criterion now: on int-vs-ref mismatch, write the unit chain (milli² →
  milli → Q12) explicitly before touching either side.

## Verification (Stage 4, session 3.5 — adversarial review)

- 4 CI gates green after all fixes: check/test/clippy −D warnings/no_std
  (183 tests: 46 rt + 134 snn + 3 doc) — all run this session
- math: `div_round_half_away_matches_f64` (2000-case sweep + ties);
  `softmax_matches_f64_and_sums_exact` now includes both constructed
  breakers + 20 tie-counts + 64-way tie + spread; `silu_matches_f64`
  swept ±9000 with the documented cliff; `rope_matches_f64_reference`
  re-derived with the corrected window (pairs 17/22/32/34 pinned)
- norm: `f32_milli_sweep_matches_f64_reference` (all 256 exponents × 4
  mantissas × 2 signs vs f64); targeted decade vectors
  (0x0080_0000, 0x2A80_0000, 0x2B00_0000 → 0)
- q1_0: `matvec_scaled_validates_sizes_before_any_work` (zero+short,
  short-out, absurd rows); `hostile_scale_blocks_saturate_not_panic`
  (±inf fp16 γ); `matvec_scaled_matches_f64_2048_wide` (real row width);
  negative-γ case in the multi-block reference test
- model: `forward_past_64_tokens_is_ok_not_panic` (65/128/129 + empty),
  `forward_out_of_vocabulary_token_is_err_not_panic`,
  `forward_with_health_reports_layer_evidence`,
  `loader_rejects_transposed_dims`, `loader_config_kv_mismatches_are_loud`
- gguf: `version_2_parses`, `non_u32_alignment_falls_back_to_32`,
  `unsorted_offsets_follow_min_greater_rule`,
  `many_distinct_tensors_parse_fast` (50k tensors, both O(n²) paths gone)
- snn: `i2_s_lane_order_golden_vector` (period 3, independent
  column-major expected bytes), `q1_0_byte_order_golden_vector`,
  `q2_0_byte_and_lane_order_golden_vector`, `absmax_i16_min_returns_32768_not_wrapped`,
  `half_to_milli_exhaustive_vs_f64` (all 65 536 patterns)
- Real model: bonsai_probe `PROBE: YES` (197/197, rope KV printed),
  bonsai_forward `FORWARD: OK` (true milli units), bonsai_full
  `FULL: OK` (release): 28/28 layer deltas alive (40 302 → 4 193 896),
  residual absmax 17 962 389 < 66 600 000 rail, forward 14.6 s, logits
  0.8 s, top-5 (16, 11555) (17, 10800) (18, 10505) (15, 9564) (20, 9360)
- Anti-claims re-grepped post-fix: no float type/literal/cast in
  bridge.rs/kernel.rs outside tests; no heap in public functions

## Learning (Stage 4, session 3.5)

- conjectured: the three YaRN test derivations (impl, math test, model
  test) verified the ramp formula.
  refuted by: the red-team derivation check — all three shared the same
  transcription slip (pair index fed where the pinned formula takes
  element index i0), so the suite was circular and stayed green while
  the interpolation window sat one octave high.
  learned: N mutually-consistent derivations are ONE derivation
  copy-pasted N times; "independent reference" must differ in
  *construction* (the fixed test derives from the formula text with
  i0 = 2i and pins the window boundaries explicitly), not just in
  authorship.
  criterion now: reference-fidelity tests must include at least one
  structural invariant (e.g. r = 0 for all pairs ≥ high) that a shared
  transcription error cannot satisfy.
- conjectured (5th and 6th instances, same session): my review-session
  test vectors were hand-derived correctly (0x3900_0000 → 1; q1_0
  byte-order positives = elements 0..7; absmax i16::MAX → 32767;
  0x2E66 → "fp16 0.05" per the copied comment — actually ≈0.1).
  refuted by: cargo test, four times — each expectation wrong while the
  implementation was right (verified by first-principles derivation
  after the fact).
  learned: the Stage-2 lesson is now structural and unconditional:
  hand-derived numeric expectations are forbidden; derive from the
  formula in the test (set comprehensions, not magic arrays), or
  reference-test against f64 — and NEVER copy a numeric claim from a
  comment (0x2E66 "0.05" was itself wrong from session 2).
  criterion now: every new expected value in a test is either computed
  in-test from the documented algorithm or compared against an f64
  reference; literals only for bit patterns already pinned by
  exhaustive sweeps.

## Verification (Stage 4, session 3)

- ISC-30: rt::math::tests::{exp2_table_basics, softmax_matches_f64_and_sums_exact,
  softmax_underflow_guard_puts_mass_somewhere}
- ISC-31: rt::math::tests::silu_matches_f64 (−6..6 milli-range sweep)
- ISC-32: rt::math::tests::{rope_position_zero_is_identity,
  rope_preserves_norm_approximately, rope_matches_f64_reference}
- ISC-33: rt::q1_0::tests::{matvec_scaled_matches_f64_dequant_reference,
  matvec_scaled_zero_input_is_zero}
- ISC-34: rt::model::tests::attention_pipeline_matches_f64_reference
- ISC-35: bonsai_full release run — FULL: OK, 4 tok/28 blocks 14.2 s,
  logits 0.8 s, top-5 printed, exit 0
- ISC-36: 4 CI gates green (163 tests total); VISION session-3 section +
  ROADMAP s1–s3 line

## Learning (Stage 4, session 4)

- conjectured: the Qwen2 split pattern groups digits `\p{N}{1,3}`
  (drafted into the session plan from tokenizer recall).
  refuted by: the fork's own source — `llama-vocab.cpp` PRE_TYPE_QWEN2
  carries `\p{N}` (single digit), and its custom scanner emits one
  digit per piece; the commented "original regex from tokenizer.json"
  agrees.
  learned: plan-stage recall is test-stage bug source; the
  verify-from-source mandate extends to PATTERNS, not just layouts —
  and this one was caught before costing anything because the source
  was fetched before the scanner was written.
  criterion now: any pinned pattern/constant enters the plan only
  with its fetched-source citation attached.
- conjectured (5 more instances this session, all test-side): my
  scanner/BPE test vectors were right (" 8" one token; "14" one
  piece; " x" split; "a\tb" split; merge chains self-assemble from
  partial operands).
  refuted by: cargo test + a vocab probe — Qwen has no `Ġ8`/`14`
  tokens (digit splitting BY DESIGN), rule 2 fires before the
  whitespace rules (so " x"/"\tb" glue to words), and a merge whose
  operands aren't themselves merge-reachable never applies ("he"
  needs ("h","e") before ("he","llo")).
  learned: same structural lesson, new surface — for VOCAB-shaped
  expectations, probe the artifact first (the 10-line python vocab
  scan settled in minutes what three test-fail rounds muddied).
  criterion now: before pinning tokenization expectations, dump the
  real vocab's relevant entries; never assert a token exists because
  "it obviously would".

## Verification (Stage 4, session 4)

- ISC-37: rt::token::tests — byte_table_pins_and_bijection,
  scanner_rule_vectors, scanner_is_lossless, bpe_merges_by_rank_then_position,
  encode_decode_roundtrip_synthetic, specials_partition_and_decode,
  decode_rejects_bad_input (CI); real_vocab_and_specials_pinned +
  real_roundtrips_and_expected_ids (ignored, run this session:
  151 669 tokens / 151 387 merges, eos 151645, pad 151643, digit-
  splitting pinned, chat prompt tokenizes with specials split out)
- ISC-38: incremental_matches_forward_synthetic_exact (nonzero
  synthetic, CI) + real_incremental_matches_forward_exact (real file,
  ignored, 788 s debug) — every hidden state equal at tolerance 0,
  including the appended 5th token
- ISC-39: argmax_tie_breaks_lowest_and_finds_peak (tie→id 0, peak row
  wins, agrees with topk[0])
- ISC-40/41: `cargo run -p neuralos-rt --release --example
  bonsai_generate` — run TWICE this session, identical output
  (greedy deterministic): strict 3/5, chat demonstrator structural
  PASS, STAGE 4 GATE: NO, exit 1. Full log evidence in the run
  (prompts, ids, generated text, expected, residuals, tok/s)
- ISC-42: 4 CI gates green this session (check/test/clippy -D
  warnings/no_std — 193 tests: 56 rt + 134 snn + 3 doc); ignored
  real-file tests 3/3; bonsai_probe PROBE: YES, bonsai_forward
  FORWARD: OK, bonsai_full FULL: OK (release, top-5 identical to
  session 3 — determinism across sessions); VISION session-4 section
  + stages-table row + ROADMAP line carry the NO; branch pushed to
  both remotes, main untouched

- 2026-08-16 · Session 4 closed: ISC-37..42 all on evidence. **Stage 4
  gate verdict: NO (3/5 strict)** — recorded, not appealed. The ISA
  stays OPEN as the Stage-4/Session-C carrier (fog ledger above).

## Verification (Stage 4, session C-pre)

- Reference build: fork cloned `--depth 1 --branch prism` @
  9ca265a57f85f2117942490f421f64a226dd9847; cmake 4.4.2 from a pip
  wheel unpacked to /tmp/opencode/cpre/cmake (no sudo, no system
  mutation); configure `-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
  -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF`, then `--build build --target
  llama-completion -j3` — 100%, clean link.
- E2E (5 frozen prompts, `-n 12 --temp 0 --top-k 0 --top-p 1.0
  --min-p 0.0 --seed 42 -no-cnv -c 512 --verbose-prompt`): fork 3/5
  with the SAME pass/fail pattern as ours. p0 " 8 9 10 11 1" and
  p1 " 14 15 16 17" byte-identical to ours (24 greedy steps). p2
  fork " four four the first part of the question is to find the"
  vs ours "-digit numbers. The problem is that the numbers are not
  unique" (both FAIL " five"). p3 fork ": 10:00 AM - 12" vs ours
  ": 10:00 AM\nWednesday: " (both FAIL " Thursday"). p4 fork
  " Paris, and the capital of Spain is Madrid. Which of" — E2E VOID
  per session pin (tokenizer divergence below); its logit
  comparison ran on the fork's ids instead. Double-run
  determinism: p2/p4 regenerations byte-identical.
- Tokenization: fork ids == our gate ids exactly on p0–p3
  (13/11/4/3 tokens); p4 DIVERGES — fork [785, 6722, 315, 9625
  "France", 374] vs ours [785, 6722, 315, 434 "Fr", 34106 "ance",
  374]. Model KVs (dumped this session): add_bos_token=0 (no BOS
  either side), eos 151645, gpt2 BPE.
- Fork logit capture: env-gated `NEURALOS_DUMP` patch at the
  completion.cpp sample site (raw top-10 before the sampler chain;
  scratch patch in /tmp only) — full 12-step traces for p2/p3/p4.
- Our logit capture: /tmp/opencode/cpre/refcmp scratch bin (path-dep
  on crates/neuralos-rt; repo untouched) — top-10, argmax, and
  expected-token rank per step at the same positions. Scratch-tool
  quirk (documented, /tmp only): its "rank" line searches the wide
  (post-top-10) list first, so a printed rank ≥1990 means a top-10
  member (true rank = printed−1990) and a printed rank <1990 means
  true rank = printed+10; the raw top-10 lines are exact and are
  what the tables below cite.
- Verdict-step (step 0) comparison — p3: argmax ":" BOTH (fork
  8.5537 / ours 7.917); " Thursday" outside fork top-10 (≤5.381),
  ours true rank 1201 @1.704. p4 on the fork's ids [785, 6722, 315,
  9625, 374]: argmax " Paris" BOTH (fork 18.3848, margin 4.45 over
  " which" 13.9318; ours 12.978, margin 0.77 over " the" 12.213);
  pre-divergence argmax agreement s0–s5 (6/6), first flip at s6
  (" France" 15.257 vs " Spain" 14.033 ours; fork " Spain" 16.352
  vs " France" 14.528). p2: fork argmax " four" @11.4858 with
  " five" rank 4 @10.5174; ours argmax "-digit" @11.250 (near-tie
  over " four" @11.003), " five" true rank 13 @9.197 — no runtime
  puts " five" on top.
- Fidelity deltas (honest counterweight): top-logit deltas p2 s0
  −0.24, p3 s0 −0.64, p3 s1 +0.33, p4(fork-ids) s0 −5.41; top-10
  set overlap 7/10 (p2 s0), 8/10 (p3 s0), 7/10 (p4 s0); later argmax
  flips p3 s8 (ours "\n" 12.487 over " AM" 12.198; fork " AM" 13.391
  over "\n" 12.704 — same pair, opposite order) and p4 s6/s8/s10/s11
  after the first flip. YaRN mismatch ruled out: GGUF carries
  rope.scaling type=yarn factor=4.0 orig 8192 (= our pin), and p0/p1
  byte-exact agreement over 24 steps would be impossible under a
  systematic scaling error.
- Frozen-gate proof: `target/release/examples/bonsai_generate`
  re-run on the untouched example — verdict-bearing content identical
  to /tmp/opencode/s4/gate_run.log (only wall-clock lines differ):
  STAGE 4 GATE: NO — 3/5, exit 1.
- Discipline: 4 CI gates green this session (check/test 193/clippy
  -D warnings/no_std); bonsai_probe PROBE: YES, bonsai_forward
  FORWARD: OK, bonsai_full FULL: OK (release, top-5 identical to
  s3/s3.5/s4 — cross-session determinism); zero code edits (docs
  only); commit LOCAL, not pushed, not merged.

- 2026-08-16 · Session C-pre closed: ISC-43 on evidence — the 3/5
  attributed (model capability under greedy, reproduced by the
  reference), fidelity + tokenizer findings handed to Session C.
  The ISA stays OPEN as the Session C carrier.

## Learning (Stage 4, session C-pre)

- conjectured (twice, caught twice in the same session): the fork's
  captured top-10s read cleanly at first pass — " five" absent from
  the fork's p2 top-10; " Paris" rank 1990 (not argmax) in my own
  tool's output.
  refuted by: a slow second pass over the grepped lines before
  writing the ISA — " five" IS the fork's p2 rank-4 entry (10.5174,
  the 5th field, one near-miss from changing the failure-margin
  story), and the scratch tool's rank line had a list-order quirk
  (wide-list-before-top-10) that mislabeled an argmax as rank 1990.
  learned: attribution numbers survive a fast read and die on a
  re-read; a near-miss rank claim is exactly the kind of error that
  quietly overstates a capability margin (1.5 logits of story).
  criterion now: every rank/margin figure in an attribution table is
  re-derived from the captured raw line in a second pass before it
  enters any doc, and scratch-tool output quirks get documented
  beside the tool at capture time, never trusted silently.

## Learning

- conjectured (session-3 → C-pre): the integer attention pipeline's
  unit chain was sound; its test proved it.
  refuted by: the f64 microscope — the reference INSIDE the test shared
  the same wrong /1000 chain; both sides divided twice and agreed.
  learned: a reference that re-encodes the implementation's unit
  assumptions is not a reference. The f64-from-scratch mini-forward
  (real units end-to-end, validated against the FORK before use) broke
  the circularity — two independent witnesses or it didn't happen.
  criterion now: unit-chain tests derive units from first principles
  (real units in the reference; conversions only at the boundary under
  test) and the strongest available external anchor (fork logits)
  validates the reference itself at least once per session.
- conjectured: γ milli-quantization (0.4–1.9%/block) was the drift
  driver (my hypothesis ladder's #1).
  refuted by: exact-γ changed nothing (5.4→6.0); the score-scale bug
  was 1000x bigger and hid behind it.
  learned: measurement before mutation only works if the measurement
  can see the actual bug — per-block Frobenius localized the injection
  in one run after two hypothesis-fixes failed. Hypothesis ladders are
  for ORDERING experiments, not replacing them.
  criterion now: when a precision fix doesn't move the needle, stop
  fixing and build the next-finer microscope before the next mutation.

## Verification (C-core)

- ISC-44: token::tests::{bpe_stale_rank_entry_does_not_fire_early (CI),
  real_france_prompt_tokenization_pinned (ignored, real file)}
- ISC-45: model::tests::attention_pipeline_matches_f64_reference
  (rewritten honest reference, tighter tolerance) + drift harness
- ISC-46: q1_0::tests::{matvec_matches_reference_*,
  matvec_gamma_is_fp16_exact_not_milli, hostile-scale tests unchanged}
- ISC-47: harness (/tmp/opencode/cpre/refcmp bins drift+micro):
  argmax 36/36, overlap 9-10/10, max Δtop 0.597; per-block err 1.1%→0.4%
- ISC-48: gate log (release): 3/5 strict, fork-byte-identical
  continuations, chat PASS, residuals 18-29M < 66.6M rail
- ISC-49: 195 workspace tests green; clippy -D warnings clean; no_std
  build clean; bonsai_probe YES / bonsai_full OK on the real file
- ISC-50: VISION + ROADMAP C-core sections; this ledger; commits local
  only — merge/push presented to the principal
