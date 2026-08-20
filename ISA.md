---
task: "NeuralOS v2 — substrate, lab bench, gated ternary bridge"
slug: 20260815-125500_neuralos-v2
project: NeuralOS v2
phase: climbing
progress: 83/83
started: 2026-08-15T12:55:00Z
updated: 2026-08-19T02:45:00Z
principal_stated_goal: "Session H2: corrected-corpus re-run — T1 PASS, P1\u2032 PASS (98.9% gap), P3\u2032 magnitude FAIL (crossed but 0.0798 < 0.11), p2 SECOND continuation change; null ladder next"
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

(Closed: ISC-1..10 s2, 11..17 s3, 18..23 s4-s1, 24..29 s4-s2, 30..36 s4-s3, 43 C-pre, 44..50 C-core, 51..56 s4-4B, 57..62 s4-D, 63..64 s4-D2, 65..67 sE, 68..70 sE-0, 71 sE-1, 72 sE-1c, 73..76 sF, 77 sF-c, 78..80 sG, 81..83 sH+H2 — see Verification.)

- [x] ISC-57 (s4-D) · **The Stage-2 q2_0 pin was wrong; re-pinned from
  source + file before any compute was built on it.** The first probe
  run on the real Q2_0 file failed ALL 253 q2_0 tensors against the
  old 64 w/18 B arithmetic (file measures 680 B per 2560-wide row);
  the fork's `ggml/src/ggml-common.h:187` says `QK2_0 = 128`,
  `qs[QK2_0/4]` (34 B/block, 2.125 bpw; fp16 max|w| + 32 LSB-first
  code bytes; quantizer `clamp(round(w/amax)+1, 0, 3)` → code 3
  unreachable; dequant `11 → +2·d`). `bridge::decode_q2_0` + tests +
  the Stage-2 gate example re-derived from the C formulas; Stage-2
  gate re-run YES; HF ships Q2_0_g64.gguf + PQ2_0.gguf variants NOT
  on disk, unreadable by any type in the fork source we hold
  (existence recorded). Falsifier: q2_0_block_geometry_is_pinned +
  the re-run gate log.
- [x] ISC-58: `rt::q2_0` ships the compute seam — `q2_0_matvec`
  (fp16-EXACT max|w| scale via the shared `half_scale_mant_shift`,
  branch-free `(code−1)·a` inner loop, stride-aware code-3 pre-scan
  that fires before ANY output write, hostile-scale milli fallback,
  saturating i32 out), `q2_0_row_to_milli`, `matvec_scaled`; bounds
  commentary re-derived for the seam (worst finite fp16 65504 = mant
  2047 « 5 → |block result| ≤ 2.75e14; 20-block 2560-wide row sum
  ≈ 5.5e15 « i64::MAX — the i64 bound holds at the seam by
  construction). Falsifier: 10 tests incl. reference through the
  published decode_q2_0 (independent lane path), f64 real-units sweep
  at 2560, fp16-exact-vs-milli pin, code-3 head+tail out-untouched.
- [x] ISC-59: the model is type-routed: `QuantData{Q10,Q20}` carries
  per-tensor bytes; `quant_tensor` dispatches on the tensor's own GGML
  type (41 → `q1_0_tensor` VERBATIM; 42 → same validation at 128w/34B;
  else loud); LayerSlices/emb/embed/topk/argmax and all 8 per-layer
  matvec sites route through it; mixed files (f32 norms + either
  weight format) Just Work. Q1_0 structural identity is pinned by
  EVIDENCE, not assertion: 1.7B gate byte-diff 43/43, both Q1_0
  ignored suites green, probe green on both files. Falsifier:
  q2_0_model_incremental_matches_forward_exact (tolerance 0),
  quant_tensor_routes_by_type_and_rejects_unknown, q2_0 padding,
  real_files_load_with_expected_configs (now 3 files).
- [x] ISC-60: the Q2_0 evidence chain on the real file: probe PROBE:
  YES (253/253 byte-exact, config == 4B Q1_0's, first block 0x24c8 =
  18.68 milli max|w|, trits +37/0×43/−48); forward FORWARD: OK (real
  ternary sparsity ~70% dense rows, per-format density gate); full
  FULL: OK (36/36 layers, residual 4 237 901 « 60 023 992 rail,
  top-5 digit tokens); decode_q2_0's real-bytes gap CLOSED (Stage-2
  pin now eats a real file, first-block census continuous across the
  re-pin). Falsifier: /tmp/opencode/s4d/{probe,forward,full}_q2_0.log.
- [x] ISC-61: **THE GATE on Ternary-Bonsai-4B-Q2_0 (frozen example,
  path arg only, zero edits): YES — 5/5. THE FAMILY'S FIRST YES.**
  p0 " 8 9 10 11 1", p1 " 14 15 16 17", p2 " five six seven eight
  nine ten\n\nWhat is the sum of", p3 **" Thursday04/05/2018 "** —
  byte-identical to the fork's 4C Q2_0 greedy INCLUDING the trailing
  space (dump-11 token 220), the weekday chain the 1-bit destroyed —
  p4 " Paris. The capital of Japan is Tokyo. The capital of". Chat
  demonstrator structural PASS ("1 \n2 \n3 \n4 \n5", clean eos).
  Residuals 4 143 575–4 313 332 strict + 6 274 426 chat: **6.2–9.5%
  of the derived 60.0 M rail** (calmer than Q1_0's 11–15 M — ternary
  zeros shrink the residual stream; recorded finding, not a
  footnote). Wall 11:21, RSS 2.16 GB. Per-model record: **1.7B Q1_0
  NO 3/5 · 4B Q1_0 NO 4/5 · 4B Q2_0 YES 5/5.** Falsifier:
  /tmp/opencode/s4d/gate_q2_0.log contradicting any line.
- [x] ISC-62: drift vs the 4C fork anchors (pins 1+2 — teacher-forced
  on the fork's OWN ids, ALL 12 steps each, values parsed
  mechanically from the raw logs): **p3 argmax 12/12, 0 flips, mean
  overlap 9.83/10, max |Δtop| 0.289** (step 0: ours " Thursday" top-1
  14.629 vs fork 14.6527); **p2 argmax 11/12, one step-6 flip** (ours
  271 @13.326 vs fork 44214 " eleven" @13.662 — a 0.34-logit near-tie,
  exactly where our gate greedy legitimately diverged after " ten";
  same class as s4b's recorded 0.1-margin flip), mean overlap 9.5/10,
  max |Δtop| 0.336. Both INSIDE the C-core bar (argmax agreement,
  ≥9/10 overlap, |Δtop| ≤ 0.597) and better than 4B Q1_0's 0.427.
  **Pin-1 disposition: agreement — no unexplained divergence — the
  verdict is recorded on clean logit evidence.** Discipline: 4 CI
  gates green (214 tests: 76 rt + 135 snn + 3 app), ignored real-file
  suite 5/5 (release, 155 s, incl. the Q2_0 loader member), 1.7B
  gate byte-diff 43/43 vs the s4b baseline; commits LOCAL only.
  Falsifier: drift_q2_0.log + any red gate.

- [x] ISC-63 (s4-D2) · **`finalize_synapses` — the public external-wiring
  CSR path.** Before it, the reverse CSR (post-firing LTP pass) was
  only buildable inside `build_topology`: callers wiring synapses via
  public `add_synapse` had no path to a correct CSR — unsorted adds
  corrupt the forward slices and the reverse CSR stays empty,
  silently regressing plasticity to the pre-1.5d LTD-only substrate.
  Shipped as one additive method mirroring build_topology's own
  post-build finalize (`SparseSynapseMatrix::finalize` + stats
  refresh), contract documented (call once after all adds; NOT
  idempotent — rebuilding means clear-and-re-add). Falsifier:
  finalize_synapses_sorts_external_adds_and_builds_both_csrs +
  finalize_synapses_makes_ltp_reachable_on_external_wiring (finalized
  net potentiates pre-before-post; unfinalized frozen at 0).
- [x] ISC-64 (s4-D2) · **THE HYBRID EXPERIMENT: HYBRID GATE: ADAPTS —
  pretrained Bonsai structure survives import AND discriminates under
  local STDP.** `examples/hybrid_gate.rs` (frozen): 262,144-trit
  decode of the real `blk.0.attn_q.weight` [2560,4096] first
  512×512 slice (dims pinned from the file — the mission note's
  [2560,2560] was wrong, caught by the example's own assertion),
  substrate γ=125 (recorded decision: LLM fp16 block scales are
  meaningless to SNN dynamics; the proven 1.5x constant carries),
  full-minus-diagonal density = 261,632 synapses, 1.5c drive
  verbatim. G1 import integrity: 0/261,632 round-trip mismatches,
  first real-data zero fraction 0.3655, zero sign crossings by
  construction (Excitatory-clamped/Inhibitory-clamped bounds census
  printed in-gate). G2 spiking fidelity: imported vs census-matched
  control vs zero-weight comparator ALL identical (35,157 spikes,
  34.33 Hz/neuron vs 0.10 floor) — the regime is drive-dominated
  (mechanism printed in-gate: recurrent ±12 μA never gates a spike
  at I_ACTIVE=600), so G2 verifies non-degeneracy + sustain +
  containment, and the structure claim lives in G3. G3 selective
  adaptation: 16,183,885 plasticity events, 321,571 bucket flips,
  Hamming 21.79% < 50% (majority intact), retention by source class
  −1 85.5% / 0 85.4% / +1 62.7%, sign-crossing transitions (+1↔−1)
  EXACTLY 0 (the SynapseType bounds asymmetry held), intra mean Δ
  −0.3133 vs inter +0.0000 → **Δ-SI 1.0000** (level-SI −0.9990
  supporting, confounded by pretrained levels). Deterministic: two
  runs byte-identical modulo the wall line. Wall 20.5 s, peak RSS
  1050 MB < 1536   budget. Falsifier: /tmp/opencode/s4d/hybrid_gate.log
  (+ .run2 for the determinism diff) contradicting any line.

- [x] ISC-65 (sE) · **`encode_q2_0` — the export codec; Q2_0 is now a
  two-way format.** The exact byte-level inverse of `decode_q2_0`:
  per 128-trit block, 2-byte LE fp16 scale (caller-supplied bits —
  re-export passes the imported bits through untouched) + 32 bytes of
  LSB-first 2-bit lanes, codes = trit+1. Code 3 is unconstructible by
  construction (every lane written from a Trit) — the encoder can
  never emit the code the reference quantizer cannot. `no_std`,
  buffer-based, loud errors, re-exported at the crate root. Falsifier:
  encode_q2_0_reproduces_known_vector_bytes + _golden_vector_bytes
  (decode→encode = byte identity on both pinned vectors) +
  encode_q2_0_rejects_bad_input + prop_q2_0_round_trip (distinct
  per-block scales, dirty buffer).
- [x] ISC-66 (sE) · **THE SURGERY: `examples/hybrid_loop.rs` (frozen)
  — LOOP EXPORT: CLEAN.** Phase 1 re-runs D-2 verbatim with its
  recorded numbers ASSERTED as preconditions before any write (spikes
  35,157 ×3 · events 16,183,885 · flips 321,571 · Hamming 57,005 ·
  Δ-SI 1.0000 — all reproduced; the surgery operates only on the
  exact recorded adapted state). Phase 2 exports the adapted 512×512
  slice through encode_q2_0 with the file's ORIGINAL fp16 scale bits
  (recorded decision: substrate adapted STRUCTURE at γ=125;
  magnitudes stay the model's own) and splices a patched GGUF copy as
  512 disjoint 136-byte chunks (row r, bytes [r·680, r·680+136) of
  blk.0.attn_q.weight — the non-contiguity the pre-session audit
  caught; tensor window computed from dims, never inferred slice
  ends). S1 containment: 29,734 code bytes differ, ALL inside
  declared chunks, outside = 0, scale bytes = 0 (asserted in-loop per
  row). S2 disk round-trip: the written file re-parses as GGUF and
  its 262,144 first-slice trits decode to EXACTLY the adapted slice,
  0 mismatches. Deterministic: two runs' patched files sha256-identical
  (87078612…). Falsifier: /tmp/opencode/se/hybrid_loop.log + the
  sha256 pair.
- [x] ISC-67 (sE) · **THE LOOP GATE: CLOSED — substrate-adapted
  weights measurably change the foreign runtime's behavior.** Judge:
  the reference fork rebuilt at the pinned commit 9ca265a (branch
  prism, HEAD == pin verified; cmake 4.4.2 pip-wheel, CPU-only flags
  per the C-pre record) + the NEURALOS_DUMP patch rewritten to spec
  (env-gated, completion.cpp sample site, raw top-10 pre-sampler —
  one API rename, llama_vocab_n_tokens). Instrument sanity: baseline
  reproduces the ISA-recorded 4C anchors EXACTLY (p3 step-0
  7794:14.6527; p2 step-0 4236:13.2278). Protocol: 5 frozen prompts,
  greedy-forced flags verbatim + `-t 4` pinned, `-n 12`, double-run
  each variant (both byte-identical). Result (mechanical table,
  parsed from raw dumps): **60/60 steps across all 5 prompts show
  logit deltas** from the patched file — max |Δ| per prompt p0 0.452
  · p1 0.088 · p2 0.534 · p3 0.124 · p4 0.093; mean |Δ| 0.05–0.22;
  top-10 overlap dips to 8/10 (p2) and 9/10 (p4) — even membership
  moved; **0/60 argmax flips** (greedy continuations byte-identical
  — honest effect size: 57,005 of 10.5 M weights = 0.5%, one
  attention slice, layer 0 of 36). Attribution chain, every link
  mechanically verified: same binary + same flags + same machine +
  both variants double-run deterministic ⊕ patched file differs from
  original ONLY inside the declared chunks (S1) ⊕ chunks carry ONLY
  the STDP-adapted trits at original scales (S2 + ISC-65 + asserted
  D-2 preconditions). The widening ladder did NOT fire (G3 asked for
  deltas on ≥1 prompt; all 60 steps delivered). Falsifier:
  /tmp/opencode/se/{baseline,patched}/p*_run{1,2}.{log,err} + the
  delta.py table contradicting any line.

- [x] ISC-68 (sE-0) · **CONTROL IDENTITY — the codec+surgery path is a
  MEASURED transparent transformation.** `hybrid_loop` control mode
  (3rd arg `control`): the full surgery pipeline (decode → assert →
  encode_q2_0 → splice → S1 → write → S2) run with the UNADAPTED
  source trits — code bytes changed 0/65,536, scale 0/4,096, S1
  outside 0, S2 0 mismatches, and the written file is BYTE-IDENTICAL
  to the original (sha256 4e0bf8b737b0431528b… == source both sides;
  full-file identity assert in-example). This converts the
  encode-exactness construction argument into a measured fact and
  closes the attribution chain: any behavioral delta on the adapted
  file is the STDP trits alone. Falsifier:
  /tmp/opencode/se/stage0/hybrid_loop_control.log + the sha pair.
- [x] ISC-69 (sE-0) · **The KLD instrument stands, deterministic.**
  `llama-perplexity` built from the same fork build dir (target +
  flags verified: `-f --chunks --ppl-output-type 1 --save-all-logits
  --kl-divergence --kl-divergence-base`); corpora mechanically
  pinned — corpus A = fork README lines 1–180 @ 9ca265a (sha
  18fb5452…, `-c 128 --chunks 2`), corpus B = the 5 frozen prompts +
  their banked baseline continuations concatenated (sha 781d1e21…,
  `-c 32`, teacher-forced). Determinism: baseline run1 == run2 Final
  lines EXACT on both corpora (PPL 15.6723 ± 4.63200; 5.1400 ±
  2.51660). Falsifier: /tmp/opencode/se/stage0/ppl_*_{run2,}.log.
- [x] ISC-70 (sE-0) · **The Stage-0 answer: the drive-informed
  adaptation is noise on generic text and a small consistently-signed
  degrading shift on the model's own continuations.** Corpus A
  (generic): mean PPL(Q)/PPL(base) = 1.002780 ± 0.010918 —
  indistinguishable from zero; mean KLD   0.004271, median 0.000547,
  max 0.289815, correlation of log-ratios 99.93%. Corpus B (the
  model's own high-confidence text): ln(PPL(Q)/PPL(base)) = +0.14895,
  +0.07446, +0.04904 — **3/3 chunks positive** (conservative claim:
  direction-consistent; no σ figure is quoted without a mechanical
  per-token computation); Δp RMS 0.36–0.50%; Same-top-p 100.000% on
  every chunk (no argmax movement — agrees with the completion
  judge). Reading, coherent with the deep-dive finding (STDP read
  the drive schedule; the ±12 μA weight channel was
  quantization-absorbed at 600 μA): the adaptation acts as a mild,
  largely directionless perturbation that measurably raises the NLL
  of the model's own preferred continuations. Stage 1 (amplitude
  sweep to open the weight→firing channel) proceeds on this
  baseline. Falsifier: /tmp/opencode/se/stage0/ppl_patched_*.log +
  STAGE0_SUMMARY.md.

- [x] ISC-72 (sE-1c) · **THE FINER-RULER SWEEP: honest NO — and the
  probe that found the REAL blocker: recurrent transmission in
  `step()` is structurally dead.** Frozen
  `examples/hybrid_sweep_cmv.rs`: the stage-1 protocol verbatim on
  the new centi-mV grid (VoltageResolution::CentiMillivolt —
  `fad081f`, dead zone ≈ 2 μA, default mV grid bit-identical, pinned
  mV trace + the 12 μA pair + the 160 μA pair tests). Result: totals
  shift to the centi pinned state (600 μA: 35,975 ×3 — expected,
  drive currents now integrate without mV truncation; E fires at
  every amplitude down to 170: 12.50/10.00/7.50/5.00/2.50/2.50 Hz,
  cliff to 0 at 150) but the three nets' TRAINS remain IDENTICAL at
  every amplitude (Hamming 0, L1 0 across the grid) — my
  pre-registered prediction (divergence at 600 μA) is FALSIFIED,
  honestly. The falsification forced the code re-read that found
  it: `step()` Phase 2 injects `weight/10` into the postsynaptic
  current AFTER Phase 1's integration, and the next step's opening
  loop calls `clear_synaptic_current()` BEFORE Phase 1 — the pulse
  is born and destroyed without ever reaching
  `integrate_and_fire`. Empirically pinned by
  `recurrent_current_is_never_integrated_in_step_bug_pinned`
  (network.rs tests: 2-neuron net, presynaptic fires, postsynaptic
  membrane NEVER leaves −70). Consequences, now with true mechanism:
  D-2/session-E's "imported/control/zero fire identically" is
  explained STRUCTURALLY (weights are silent in step(), full stop) —
  the mV-grid dead zone (real for input currents, unit-proven) was
  the co-blocker, but upstream of it the wire is cut. Also refuted:
  the session-E audit's "pulse integrates with one-step delay"
  reading — it re-encoded the code COMMENT's intent, not the code's
  behavior.   Falsifier: /tmp/opencode/se/stage1c_sweep.log + the
  canary test.

- [x] ISC-73 (sF) · **THE TRANSMISSION FIX (a1b) — synapses transmit
  again.** `step()` reordered per the reviewed split: adaptation
  decay (pre-integrate, phase-identical to history — the a1b
  invariant that keeps transmission-live the ONLY variable) →
  integrate (reads the previous step's pulses — the one-step delay
  the orchestrator always claimed) → clear-after-read → propagate →
  plasticity. `decay_synaptic_current` split into
  `decay_adaptation_current` (+ the existing pub clear) — the old
  combined fn REMOVED (public-API change, alpha.2 manifest); its
  synaptic half had been structurally dead since the port.
  `tau_synapse_us` and `delay_us` marked decorative honestly.
  Falsifier: transmission_is_live_one_step_delayed_centimv (post
  −7,000 → −6,988 exactly one step after pre fires — the delay
  itself pinned), transmission_is_live_one_step_delayed_mv_strong_weight
  (−70 → −68), transmission_pulses_sum_across_presynaptic_spikes
  (two pres ⇒ +24 quanta) — all exact-value, both grids.
- [x] ISC-74 (sF) · **The suite is green with ZERO unit-pin
  failures — the bisection predicate had nothing to bisect.** 152
  snn tests green unchanged: no library unit pin ever exercised live
  transmission through `step()` (the second reviewer's shared-miss,
  confirmed empirically). The 1.5-lineage examples re-run IDENTICAL
  (ternary_gate 3308 spikes / 3587 flips; selectivity YES) —
  physics-consistent: balanced-topology weights are 8–20 ⇒ 0–2 μA
  pulses, quantization-absorbed on the mV grid (dead zone per-type
  E ~200 / I ~100 μA, docs corrected this session).
- [x] ISC-75 (sF) · **D-2 RE-RUN POST-FIX — the new pinned state,
  and THE HEBBIAN REVERSAL.** G2 (fixed weights): **35,115 /
  35,136 / 35,157** — the first weight-borne rate divergence in the
  lineage (imported −42, control −21 vs the zero-net drive baseline;
  ±12 μA pulses now tip spike timing). G3 (STDP on): firing 35.13
  Hz, events 18,817,891, flips 708,029, Hamming 64,877 (24.80% <
  50%), sign crossings 0, mean Δ intra **+0.1075** vs inter +0.0000
  — Δ-SI = **−1.0000**: the discrimination SIGN REVERSED. Pre-fix:
  intra −0.3133 (same-step co-fire tie-break LTD, anti-Hebbian,
  dead-wire artifact). Post-fix: intra-group pre spikes causally
  drive post spikes one step later ⇒ pre-before-post ⇒ LTP ⇒
  Hebbian potentiation — the regime the 1.5c scope note named as
  the honest LTP exercise. |Δ-SI| = 1.0000 = perfect
  discrimination both ways; the SIGN is the mechanism label. The
  FROZEN gate's directional condition (meanΔ_inter > meanΔ_intra,
  written for the LTD-carried regime) reads this as selective FAIL
  → HYBRID GATE (phase 1): COLLAPSES, exit 1, **surgery NOT run** —
  recorded as the honest output of the frozen criterion
  (/tmp/opencode/sf/hybrid_{gate,loop}_fixed.log). The criterion
  fork goes to the principal (Decision below).
- [x] ISC-76 (sF) · **BOTH SWEEPS POST-FIX — A\* = 600 μA on both
  grids; predictions 2/4 exact, 2 under-called in our favor.**
  mV: Hamming 58,779 at 600 (timing-tipping among driven neurons,
  predicted) but rate-L1 = 689 ≠ ≈0 (under-called): below the old
  cliff, coherent same-group bursts STACK past the shrunken margin
  — at 300 μA imported fires 1.14 Hz, control 0.99, zero 0.00
  (RECRUITMENT — a WHO channel on mV; the reviewer's random-σ
  analysis missed that bursts are coherent, not Gaussian). centi:
  H(i,c) = 39,011 > H(i,z) = 23,907 ≈ H(c,z) = 23,958 —
  **ARRANGEMENT carries more divergence than census content**
  (imported vs its own shuffle is the largest gap — the strongest
  model-informed signal in the project; the reviewer predicted
  imp-vs-zero largest — under-called, recorded). I-population rates
  diverge too (123.67/123.92/125.00 at 600 — live inhibition reads
  weights). At centi 300 μA zero OUT-FIRES wired nets (7.50 vs 6.97
  Hz — the live I-wall drags); at centi 150 recruitment again
  (0.70/0.62/0.00); centi 100 all-silent (no bootstrap). Falsifier:
  /tmp/opencode/sf/sweep_{mv,cmv}_fixed.log — every row. Visualizer
  smoke green (exit 0).

- [x] ISC-77 (sF-c) · **THE CRITERION AMENDMENT ((i)-amended,
  third-review fork call) + THE LOOP UN-PARKED ON THE LIVE WIRE.**
  The second review's degeneracy proof, adopted whole: Δ-SI ≡ ±1
  by schedule geometry (40 ms group gaps vs the 20 ms STDP window ⇒
  inter Δ ≡ 0 in every era ⇒ the 0.05 floor gated nothing), so the
  D-2 gate now asserts the RAW, non-degenerate fields — intra
  |mean Δ| ≥ 0.05 (degree of discrimination) · flips > 0 · Hamming
  < 0.50 · sign crossings = 0 · sustained firing — with the class
  direction PRINTED as the era's mechanism label (Hebbian-carried /
  LTD-carried) and Δ-SI demoted to a supporting label. The stale
  LTD parenthetical replaced by the computed label; the dt-pairing
  histogram instrumented (three NetworkStats counters, additive —
  alpha.2 manifest): **same-step 951,578 · post-leads 4,110,289 ·
  pre-leads 4,110,289 — EXACTLY equal LTD/LTP counts**, a real
  structural fact (each in-window adjacency on an edge yields one
  post-leads pairing at the pre's step and one pre-leads pairing at
  the post's step; the net Hebbian drift lives in the dt-dependent
  magnitudes, not the counts). [Session-G correction: the per-class
  COUNTERS (ISC-78) refuted the simple causal-LTP reading — the
  realized potentiation is clamp-rectified, not LTP-dominant.] hybrid_gate re-run: **HYBRID
  GATE: ADAPTS** on the live wire, mechanism [Hebbian-carried],
  intra |Δ| 0.1075 PASS. hybrid_loop re-run: D-2 preconditions
  reshaped to THREE totals (35,115/35,136/35,157 · events
  18,817,891 · flips 708,029 · Hamming 64,877 · intra +0.1075 /
  inter 0.0000 — asserted from the re-run, never transcribed) —
  **surgery UN-PARKED: LOOP EXPORT: CLEAN**, 64,877 cells exported
  (31,607 code bytes vs the dead-wire era's 29,734), S1 outside 0,
  scale 0, S2 0 mismatches, patched files sha256-identical
  (24ffe5f3…) across double-runs. Doc repairs: delay_us decorative
  marker (synapse.rs), climb-barrier + rectification documented
  (lif_neuron) and PINNED (rectification_at_the_sticking_point:
  300 μA sticks at −59; +12 μA pulse ratchets +1 mV; −12 μA
  absorbed). Falsifier: /tmp/opencode/sf/hybrid_{gate,loop}_
  {criterion,fixed}.log + the sha pair.

- [x] ISC-78 (sG) · **THE MECHANISM LABEL, EARNED — and AMENDED:
  pairing-selective, clamp-rectified (the inferred "Hebbian-carried"
  refuted by its own counters).** Fourth-review Finding A, tested
  before push: `Synapse` gained two cumulative counters
  (`raw_stdp_delta` = Σ deltas before clamping; `absorbed_delta` =
  Σ clamped-away remainder), and run_hybrid decomposes them per
  class. The measured decomposition (live-wire D-2 state, re-run):
  **raw intra drift −739,295 (mean −17.85/syn — LTD events
  dominate: same-step 951,578 + post-leads 4,110,289 at −5 vs
  pre-leads 4,110,289 at +4) · clamp-absorbed −839,029 · APPLIED
  +99,734 (mean +2.41/syn) · inter exactly 0/0 (no pairings —
  schedule geometry).** Mechanism, counted: intra co-firing drives
  a net-NEGATIVE raw drift; the E-class 0-floor absorbs the LTD;
  the applied residue potentiates → buckets move +0.1075. The
  class-differential is timing-driven (only co-firing pairs pair
  at all); the DIRECTION is bounds-driven. The label logic in both
  examples now COMPUTES the mechanism from the counters (three
  cases: raw-LTP-dominant = Hebbian-carried; raw-negative +
  applied-positive = clamp-rectified; applied-negative =
  LTD-carried); this run prints [PAIRING-SELECTIVE,
  CLAMP-RECTIFIED]. The ADAPTS verdict is unchanged (the gate is
  degree-based: intra |mean Δ| 0.1075 ≥ 0.05 PASS); the loop
  re-exports CLEAN with identical numbers (counters additive,
  determinism intact). Falsifier: /tmp/opencode/sf/hybrid_gate_
  mech.log + hybrid_loop_g.log contradicting any line.

- [x] ISC-79 (sG) · **SESSION G COMPLETE — Bank & Verify, all five
  legs.** (1) label earned/amended (ISC-78). (2) Evidence banked as
  evidence: `tools/delta.py` committed beside its claim;
  `evidence/session-f-judge/` = the fork-judge logs (p0–p4 × 2
  runs + README), not transcriptions. (3) Pushed BOTH remotes
  (99e341a..3b512df then the release commit) — the record is
  durable. (4) **alpha.2 PUBLISHED**: crates.io max_version
  0.1.0-alpha.2 (verified via API with UA, 01:12:31Z), the bridge
  + kernel + session-F fixes + the real-artifact vector test
  (token_embd's true first block 0x24C8, census +37/0×43/−48,
  encode∘decode = BYTE IDENTITY on real bytes) — and the alpha.1
  correction recorded: the tag ships no bridge module, the broken
  pin never left the machine, the "published wrong codec" story
  was an overclaim. (5) The novelty pass (PAPER_NOVELTY.md) is
  DELEGATED to the fresh session as independent searches; synthesis
  here. Falsifier: crates.io API + git remotes + the evidence dir.

- [x] ISC-80 (sG-5) · **THE NOVELTY PASS: OPEN SEAM — measured, not
  asserted.** `docs/PAPER_NOVELTY.md`: the full chain (L1 local
  plasticity · L2 shipped LLM · L3 quantized re-export · L4 foreign
  run) verified against a 26-query logged survey (delegated
  independent session, 2026-08-18, arXiv UI + DDG Lite + direct
  fetches; every cited ID fetched, one recalled ID caught and
  discarded) PLUS a this-repo OpenReview spot-check (2026-08-19, V1
  API, 2 × 1,000 relevance-ranked notes machine-scanned: zero L1+L2
  candidates; nearest = "Memory-based Hebbian Parameter Adaptation"
  2021 — few-shot class learning, not shipped-LLM weight
  adaptation). 16-entry matrix + 3 near-misses with missing links
  (QES: no L1; Dragon Hatchling: no L2/L3/L4; BitDistill+bitnet.cpp:
  proves our L3/L4 half, missing L1/L2). Direct seam probe — GGUF ×
  (STDP | local plasticity | Hebbian) — logged EMPTY. Supporting
  signal: the field's own survey (2409.02111) lists only conversion
  + surrogate-BP as paradigms; Rajendran & Simeone (2309.15942) name
  backprop-free on-device fine-tuning as FUTURE work. Boundaries
  documented in-doc (no S2 — 429; OpenReview top-slice only) and
  the PRE-SUBMISSION GATE named: full pass re-run + fresh dated
  section before submitting. **The fork resolves on this data:
  Stage 2 (in-vivo drive) runs as the paper's final pre-registered
  experiment; then the draft.** Falsifier: any paper demonstrating
  the full chain (→ reframe "first in field's record" to "first in
  the ternary/no_std/verified-attribution setting").

- [x] ISC-71 (sE-1) · **THE AMPLITUDE SWEEP: honest NO — the
  weight→firing channel does not open by amplitude alone.** Frozen
  `examples/hybrid_sweep.rs`: I_ACTIVE ∈ {600, 450, 300, 240, 200,
  170, 150, 125, 100} μA, I_INH=600 fixed (single variable), 1.5c
  schedule verbatim, STDP off, imported vs census-matched control vs
  zero — metrics spike-TRAIN Hamming per pair (not totals) +
  per-neuron rate L1 + per-population Hz. Result: **zero divergence
  at every amplitude, every pair** (H = 0, L1 = 0 across the grid).
  600 μA reproduces D-2 exactly (35,157 ×3); 450 μA fires E at 8.00
  Hz ×3 (32,294 ×3, still identical); at 300 μA and below the E
  population is EXACTLY silent (totals 25,750 ×3 = the I population
  alone, 103 × 125 Hz × 2 s; I Hz 125.00 at every amplitude — I
  firing never depends on weights at fixed 600 μA drive). The
  pre-registered prediction (onset ≤ 300 μA) is FALSIFIED — honestly.
  Mechanism (recorded reading): the continuously-driven I population
  (125 Hz through ±12 μA synapses) parks a ~−97 μA inhibitory
  background on every E neuron, raising the effective E threshold
  from ~150 to ~250 μA; above it (450: margin ~300 μA) the recurrent
  σ ≈ 40–50 μA + integer-mV quantization still absorbs everything
  (6–7σ), below it E silence is self-consistent (E→E excitation
  exists only if E fires — no bootstrap). The channel needs a
  different coupling, not a different amplitude. Falsifier:
  /tmp/opencode/se/stage1/sweep.log (full table) contradicting any
  line; wall 13.2 s, RSS 1050 MB < 1536.

- [x] ISC-51: the 4B's config pinned from the file BEFORE any code —
  probe + full `qwen3.*`/`general.*` KV dump on both tiers, diff in
  Decisions: 36 blocks, emb 2560, 32/8 heads, head_dim 128
  (key_length == value_length == 128 VERIFIED), FFN 9728, vocab
  151 669 (tensor-derived — no vocab_size KV), and TWO silent
  breakers found: `qwen3.rope.freq_base` = **5e6** (1e6 on 1.7B —
  the old pin would have run wrong rope tables) and the embedding
  slice carrying 24 B of alignment padding (54 600 840 → 54 600 864;
  the old exact-size rule rejected the file). Falsifier: probe logs
  (/tmp/opencode/s4b/probe_{4b,17b}*.log) + the KV dump lines.
- [x] ISC-52: `rt::model` is config-driven: `ModelConfig::from_gguf`
  reads the fork's own KVs (required keys loud on absent/non-numeric;
  value_length/freq_base/eps/scaling-type cross-checked; emb/ffn %128,
  head_dim %2, heads %kv_heads invariants), runtime dims replace every
  1.7B constant (attention context `heads·head_dim` — genuinely
  non-square on 4B's q/attn_output), the score scale DERIVES from
  head_dim at the load edge and is pinned: `score_scale(128) ==
  (88, 3883)` bit-identical to the replaced constants, tensor loads
  tolerate exact-or-alignment-padded sizes (padding never copied).
  Falsifier: from_gguf_reads_the_4b_config_block,
  from_gguf_is_loud_on_missing_and_broken,
  score_scale_pins_the_1_7b_split, loader_accepts_alignment_padding…
  + ignored real_files_load_with_expected_configs (BOTH files).
- [x] ISC-53: the refactor is regression-proven behavior-preserving
  on 1.7B (principal pin №1, hard gate): fresh pre-refactor gate log
  at e1821be captured, post-refactor re-run byte-diffed — 43/43
  verdict-bearing lines identical (ids, texts, pass/fail, residuals,
  verdict; only wall-clock lines excluded); real-file suite green
  (incremental≡forward exact, release 69 s + debug 888 s); and the
  full-forward path pinned by an independent witness (e1821be
  worktree run reproduces the new bonsai_full numbers EXACTLY —
  the s3.5-recorded top-5 was a pre-C-core artifact, see Learning).
  Falsifier: any diff line in gate_17b_{baseline,post}.norm diff.
- [x] ISC-54: the 4B evidence chain, release, RSS on every run:
  probe PROBE: YES (398 tensors, 253/253 q1_0 incl. the padded
  embedding, first-block scale 19 milli ∈ [1,100]); forward FORWARD:
  OK; full FULL: OK (36/36 layers alive, residual 10.96 M « 60.0 M
  derived rail, 51.5 s); drift (teacher-forced vs fork 4B dumps):
  p2 12/12, p4 12/12 max |Δtop| 0.064 (the France/tokenizer
  second witness — fork ids [785, 6722, 315, 9625, 374] ==
  ours), p3 11/12 with ONE flip at step 1 (" June" 5534 @13.04 vs
  ours " " 220 @13.14 — a 0.1-margin near-tie); fork E2E 4/5.
  Peak RSS: 1.17 GB (gate/drift), 584 MB (probe/forward), 1.14 GB
  (full). Falsifier: /tmp/opencode/s4b/{probe,forward,full,drift,
  gate,fork4b}* logs.
- [x] ISC-55: THE GATE on Bonsai-4B (frozen prompts/expected/verdict,
  zero edits beyond the approved type-only `.cloned()`): **NO — 4/5**.
  PASS p0 " 8 9 10 11 1", p1 " 14 15 16 17", p2 " five six seven
  eight nine nine one zero one one two three" (the 1.7B failure
  prompt PASSES on 4B), p4 " Paris. Paris is the capital of France.
  Paris is the" — all four BYTE-IDENTICAL to the fork's continuations.
  FAIL p3: ours ", 2024, 10:0" vs fork ", June 12, 2018," — the fork
  fails the same prompt (verdict step 0: both argmax ","; " Thursday"
  outside the fork's top-10); the continuation divergence is the
  measured step-1 near-tie. Chat demonstrator PASS ("3\n\n4\n\n5",
  clean eos). Residuals 11.1–15.1 M under the 66.6 M rail. Wall
  37:22, decode 0.037–0.083 tok/s. **Attribution (same session): the
  4/5 is the 4B model's ceiling under greedy on the frozen set —
  fork-reproduced, four continuations fork-byte-identical.** The
  8B-vs-stop call is the principal's. Falsifier: gate_4b.log +
  fork4b/*.log contradicting any line here.
- [x] ISC-56: discipline + docs truth: 4 CI gates green on the
  refactored code (200 tests: 63 rt + 134 snn + 3 app; clippy
  -D warnings clean; no_std clean), ignored real-file suite 5/5
  (release) + the incremental test re-run in debug (888 s) for
  profile parity with the recorded evidence; VISION/ROADMAP/ISA
  carry the per-model record; commits LOCAL only — no push, no
  merge (the principal's gates). The pin-2 deep-dive ladder did
  NOT trigger (condition "4B lands 3/5" — it landed 4/5); rung
  availability recorded in Decisions. Falsifier: any gate red,
  docs absent/contradicting, or a pushed branch/merge commit.


- [x] ISC-44: tokenizer stale-rank bug fixed — BPE heap entries validated
  against the push-time rank (fork's text-equality check); regression
  trap in CI (synthetic wrong-rank shape) + real-file pin
  (" France" = 9625, full prompt ids). Falsifier: bpe_stale_rank_entry_does_not_fire_early + real_france_prompt_tokenization_pinned.
- [x] ISC-45: attention score unit chain corrected (dot is milli²; milli
  score = dot × 88.3883 / 1e6, was /1e3 → every score 1000× too large →
  softmax saturated to hard argmax) — all three sites + the circular
  reference rewritten in real units. Falsifier: attention_pipeline test
  (honest reference) + drift harness numbers.
- [x] ISC-46: q1_0 matvec applies γ at fp16-EXACT precision (integer
  mantissa×2^shift), milli quantization removed from the compute path;
  hostile-scale fallback preserves Session-A saturation; references
  exact-decoded, regression pin added. Falsifier: matvec tests incl.
  matvec_gamma_is_fp16_exact_not_milli.
- [x] ISC-47: drift converged past C-pre's bar: teacher-forced fork
  comparison argmax 36/36 steps across p2/p3/p4, overlap ≥9/10 mean,
  max |Δtop| 0.597 (was 5.407); per-block int-vs-f64 error ≤1.1%
  decaying (was 15.5% injection at block 0). Falsifier: harness tables
  (evidence below) + f64 reference validated against the fork (±0.03).
- [x] ISC-48: frozen gate re-run verbatim: 3/5 strict (unchanged), but
  continuations now byte-identical to the fork's greedy (" Paris, and
  the capital of Spain…", " four four the first part…", ": 10:00 AM -
  12") — the faithful-runtime state; chat demonstrator coherent; NO
  verdict stands, now demonstrated at generation level. Falsifier: gate
  log + fork logs side-by-side.
- [x] ISC-49: 4 CI gates green (195 tests incl. real-file suite:
  France pin, vocab pins, incremental≡forward exact); probe/forward/
  full green on the real file. Falsifier: any failing.
- [x] ISC-50: docs truth (VISION C-core section, ROADMAP); ISA ledger
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

- 2026-08-20 (sI, AMENDMENT COMMITTED BEFORE THE STRESS CHAIN'S
    RESULTS ARE READ — the pre-registration discipline itself) ·
    **NULL FAMILY v2: dose-matched primary + stress arm + shuffle
    rung.** The pre-mortem found the category error: the running
    null_patches census-matched TRANSITION totals (1,112,771 flips)
    as CELL placements — flips ≠ cells (H2 averaged ~12.8
    flips/cell; terminal patch 87,119 cells; the running family
    landed ~178k). That family is RELABELED THE STRESS ARM,
    report-only: stress flips contextualize, NEVER adjudicate.
    **PRIMARY FAMILY v2:** dose-matched ×10 — exactly 87,119
    changed cells, composition from the H2 TERMINAL DIFF (decode
    the H2 patched GGUF's 512×512 slice vs the original; positions
    AND values from the artifact, never re-run). **Adjudication:**
    the sH2 conjunct (a)–(d) binds on the PRIMARY family;
    escalation 0–1/10 → n=20, seeds 11–20 pre-generated now.
    **Census keying (clarification):** the knife-edge set is θ=0.05
    on BASELINE margins (model properties before perturbation);
    the patched side is an outcome, never a key. **Shuffle ×3 bar
    (pre-stated):** "reproduces the signature" = ≥10/12 p3 flips OR
    the p3 step-1 knife-edge crossing; anything less = does NOT
    reproduce. **Shuffle gotcha:** the value permutation is
    conditioned on new ≠ source (re-draw); a shrunk dose ABORTS
    loudly, never silently. **Tables:** knife-edge flips AND total
    flips per prompt, side by side — a primary-null p2 flip kills
    "nulls quiet" even off the baseline knife-edge set. **Replicates
    parity gate:** the embeddings-only capture path must first
    re-run H2 and match export sha 71f2518a… (or the recorded drive
    stats: RMS 0.0447, k=10060.46, clamp 69.477%, dim-199 1786×)
    before any rung-4 replicate counts as a replicate. **Scope
    trim:** KLD breadth is the first drop if morning tables demand
    any re-run.

- 2026-08-19 (sH2, PRE-REGISTERED AND COMMITTED BEFORE ANY CODE —
    the new precedence rule's first live use) · **THE H2
    REGISTRATION v2 + THE NULL-LADDER DECISION RULE v2.** H1's
    corpus infidelity (the run used the 1,024-byte session-F
    README, sha 2d64e907…, not the pinned 18fb5452… slice; the
    code comment claiming identity was FALSE — root cause: the
    registration's own constraints were mutually unsatisfiable at
    ~3.5k tokens and were resolved silently in code) is corrected
    by RE-RUN, not annotation. **Registration v2:** corpus = the
    TRUE pinned slice (18fb5452…), FIRST 2,000 tokens in order,
    single truncated pass — no epochs, never wraps by construction;
    init 400 = the first 400 steps of the same stream (no token
    driven twice — strictly cleaner than H1's 5–6× re-drive);
    token-2000 text window printed (truncation context); head-bias
    (first-2000-of-README ≠ corpus statistics) named, not fixed.
    All else H1-amended-verbatim (E-dims 0..408, I_INH wall, centi,
    counter battery). **P1′ (with teeth):** G0 gap H(i,c)−H(i,z) ≥
    ~25% of H1's 10,974 rate-normalized — else "single-pass
    weakens G0" AMENDS the coupling story, never silently passes.
    **P3′ (with teeth):** |Δmargin at p3 step-1| ≥ 0.11 (0.5 ×
    H1's 0.213), sign recorded. **Confounds named:** H2 vs H1
    changes text (1KB→12KB), repetition (6×→1×), and k (clamp
    fraction printed side-by-side H1-vs-H2); the de-confound arm
    (true corpus head × 6 epochs) is a named follow-on. The
    "selective" verdict line is relabeled DESCRIPTIVE
    (drift-liveness — the D-2 contrast is dissolved). **H2b
    checkpoint flags:** implemented BEFORE the H2 run, quarters of
    the LEARN phase (400-step increments), dose axis = measured
    changed-cell count (non-monotonicity handled by recording), the
    100%-checkpoint sha ASSERTED equal to the plain export sha
    (invariance proves the machinery did not perturb the artifact);
    every derivative sha-pinned at write; S2 post-write re-read on
    EVERY export, nulls included. **The null-ladder decision rule
    v2 (pre-registered):** (1) margin census → knife-edge set =
    ALL prompt×step top1−top2 margins < θ = 0.05, published table;
    (2) siblings rule-fixed BEFORE running (7 weekday-triple
    rotations + 1 month chain + length-matched off-circuit
    negatives "red green blue" + digit run), baseline margins
    first, EVIDENCE-ONLY; (3) random nulls ×10 — census-matched,
    region-matched i.i.d. — PLUS a position-shuffle rung (H2's
    changed-cell set, permuted assigned new-values: isolates
    content-vs-placement while preserving per-row clustering), all
    with full S2 asserts and the 5-prompt judge; (4) JUDGED
    in-vivo replicates ×3 (flip-seed variants exported + judged —
    the in-vivo side becomes n=3); (5) dose-response both arms
    (H2b checkpoints; random trimmed to measured per-dose counts).
    **Adjudication (the conjunct):** "steers" survives ONLY IF
    (a) primary continuous statistic — |Δmargin| at every
    knife-edge — in-vivo exceeds the max of ALL nulls'; (b) null
    flip-rate on the knife-edge set ≤ 1/10 (escalation: 0–1/10 →
    n=20, ≤ 2/20; the rule-of-three CI at n=10 is too wide to
    carry the claim alone); (c) in-vivo dose curve outside the
    null band (mean ± 2SD) at every checkpoint; (d) in-vivo
    effects concentrated on the weekday set vs off-circuit
    negatives. ANY arm fails → degradation-by-perturbation
    language, recorded. **Banking upgrades standing:** model-sha +
    invocation line in every judge log (run_prompts.sh → tools/),
    both export legs + full p0–p4 ×2 + delta tables in git.

- [x] ISC-82 (sH) · **TIER 2/3: THE IN-VIVO EXPORT STEERS — the
  first continuation change in the project's record.** Export
  (deterministic: sha adcc7feabc82… ×2 independent runs; 67,309
  cells, 40,126 code bytes, scales 0) judged by the standing fork
  protocol (5 frozen prompts, greedy-forced, double-run —
  byte-identical ×5). **p3 "Monday Tuesday Wednesday": baseline
  " Thursday04/05/2018" → in-vivo-adapted " Thursday" then
  "\n\n\n…" — the continuation DIVERGES COMPLETELY after the
  shared first token: 11/12 argmax FLIPS, top-10 overlap collapses
  to mean 3.58/10, max |Δtop| 10.36.** p0/p1/p4: 0 flips, mean |Δ|
  0.054–0.063 (the familiar perturbation scale); p2: 0 flips,
  overlap 9.83, max |Δ| 0.272. After four prior runs of 0/60 flips
  at this footprint (synthetic-era exports), this is qualitatively
  new: the in-vivo adaptation (1,125,221 flips, both classes
  potentiated, clamp-rectified) moved the weekday prompt's
  dynamics decisively. Attribution chain: same binary + flags +
  machine, both runs deterministic ⊕ patched file differs only
  inside the declared chunks (export asserts) ⊕ chunks carry only
  the in-vivo-adapted trits at original scales ⊕ baseline anchors
  reproduced (step-0 7794:14.7523 vs the recorded 14.6527 family —
  first token IDENTICAL " Thursday", divergence begins at step 1).
  Honest readings, both recorded: (i) STEERING demonstrated —
  Tier-3's named stretch, achieved without widening; (ii) the
  direction is DEGRADATION-shaped (the date chain, a
  Q2_0-recovered capability, is disrupted — consistent with the
  stage-0 finding that adaptation degrades the model's own
  continuations); the claim language stays "steers" not
  "improves", per the standing constraint. p3's sensitivity is
  itself the finding: the weekday-knowledge circuit was the
  4C-identified quantization-fragile one — the adaptation found
  it. Falsifier: /tmp/opencode/se/invivo/p3_run{1,2}.{log,err} +
  the delta table; KLD corpus scoring is named follow-up (the
  continuation divergence already exceeds what KLD adds).

- [x] ISC-83 (sH2) · **THE CORRECTED-CORPUS RUN: T1 ALL PASS,
  P1′ PASS with teeth, P3′ magnitude FAILS (split verdict), and a
  SECOND continuation change.** True pinned corpus (18fb5452…),
  first 2000 of 4411 tokens, single truncated pass (cut context
  printed; head-bias named). **P1′ PASS: G0 gap 41,555 − 30,724 =
  10,831 = 98.9% of H1's 10,974** (floor 2,744) — single-pass does
  NOT weaken the arrangement signal. Clamp 69.48% vs H1's 69.80%
  (confound bounded). Adaptation: 43.65 Hz, flips 1,112,771,
  Hamming 33.30% (higher than H1's 25.7% — more text diversity),
  mechanism [CLAMP-RECTIFIED] (pairing perfectly symmetric again:
  7,030,561 = 7,030,561). Dose checkpoints monotone: 61,210 →
  71,381 → 80,391 → 87,119 cells; every export S2-clean; final sha
  71f2518a… **Judge (×2 deterministic): p3 steers AGAIN** — 11/12
  flips, " Thursday"+newline-run, the knife-edge CROSSED (margin
  +0.0091 → −0.0707) — **but |Δmargin| = 0.0798 < the
  pre-registered 0.11 bar: P3′ magnitude FAILS, both outcomes
  pre-accepted, the corrected one-flip language stands.** **NEW: p2
  flips 4/12** — "…twelve fifteen seventeen seventeen eighteen" vs
  baseline "…thirteen fifteen fifteen seventeen" — the project's
  second continuation change, on the counting prompt (divergence at
  " thirteen"→" fifteen", step 5). p0/p1/p4 quiet (0 flips, mean
  |Δ| 0.04–0.11). Wall 8.24 h — the embeddings-only capture ran the
  FULL 36-layer forward (only states[0] is used): an
  embeddings-only public path is named follow-up (would cut ~90% of
  the wall). RSS 6593 MB (recorded). Falsifier:
  evidence/session-h2/ (run.log + judge ×2) + the census table.

- [x] ISC-81 (sH) · **THE IN-VIVO GATE, TIER 1: ALL PASS — the
  model's own activations drive weight-reading firing, and the
  substrate adapts under them.** Frozen `examples/hybrid_invivo.rs`
  (the amended registration, built verbatim): drive =
  attn_norm(embedding) of the sha-pinned corpus (332 tokens × 6
  whole epochs = 1992 steps, wrap excluded), k = 9721.41 μA/unit →
  corpus RMS 450 μA target, driven dims 0..408, I_INH=600 wall,
  CENTI grid, STDP on, counters per the registration. **G2′
  tripwire PASS** (weights borne in trains). **G0 PASS: H(i,c)
  41,190 > H(i,z) 30,216** (rate-L1 agrees: 1440 > 1040) — under
  the model's OWN activity, firing reads arrangement over census
  (1.36×), independent of the synthetic schedule. Adaptation: 43.93
  Hz sustained, events 25,322,176, flips 1,125,221, Hamming 67,309
  = 25.73% (contained), sign crossings 0, applied intra +105,182 /
  inter +1,687,293 — mechanism [PAIRING-SELECTIVE,
  CLAMP-RECTIFIED] exactly as pre-registered-expected. **TWO
  HONESTY FLAGS, recorded per the registration itself:** (a) the
  CLAMP CAVEAT fired hard — 69.80% of dim-steps railed at ±1000 μA
  (post-RMSNorm activations are far heavier-tailed than the corpus
  RMS; the effective drive is sign-dominant, not
  amplitude-graded; hottest dim railed 90% of steps) — the gates
  passed on a shared drive so G0's contrast stands, but
  "amplitude-informed" is NOT what ran; per-dim standardization or
  a higher clamp is named follow-up; (b) the 4-group intra/inter
  class structure DISSOLVED under data-driven co-firing — most
  pairs pair (inter applied-drift positive too, +0.0777 meanΔ vs
  intra +0.0606, inverted vs the synthetic era): the model's
  activity treats the slice as one co-active assembly, not four
  groups; class-selectivity claims do not transfer from the
  synthetic regime. Falsifier: /tmp/opencode/sh/invivo.log (full
  tables); wall 2298.6 s, RSS 2436 MB (the 1536 budget was the mV
  single-buffer box — the run holds model + corpus + 3 trains;
  recorded).

- 2026-08-19 (sH, PRE-REGISTERED before any code — the session-H
    registration artifact) · **THE IN-VIVO GATE, FROZEN.** Purpose:
    does the model's OWN activity carry more coupling structure
    than the synthetic 1.5c schedule? Three tiers, stated up front:
    **Tier 1 (substrate, hard-gated)** — GRID: CentiMillivolt,
    PINNED (the G0 inequality's only lineage demonstration is
    ISC-76's centi result; a D-2-verbatim import is the mV build
    where single weight pulses are dead-zone-absorbed — spurious
    fail; attack-pass amendment #1). G0: real activations carry
    structure (arrangement-vs-census: the in-vivo-driven imported
    net's train must diverge from its census-shuffle control MORE
    than from zero-net); G2′ (relabelled per amendment #3): the
    WIRE-LIVENESS TRIWIRE — imported vs control vs zero not all
    identical (vacuous as a divergence gate post-F; the arrangement
    claim rests on G0 alone). METRIC BATTERY (amendment #2): all
    three pairwise spike-TRAIN Hammings + per-neuron rate-L1 +
    per-population Hz — Hamming alone conflates rate and timing
    under variable drive; near-ties route to seed replicates as a
    follow-on. P1 (pre-registered prediction): divergence appears —
    the live wire + rectifier physics are amplitude-driven and the
    corpus RMS is set WITHIN the validated amplitude range
    (amendment #8 wording; auditable via the printed per-step
    |current| histogram). **Tier 2 (judge, recorded whatever
    it says)** — KLD + continuation diff of the in-vivo-exported
    patched file vs baseline AND vs the synthetic-era export
    (evidence, not gate); P2: in-vivo export deltas ≥ synthetic
    export's under identical footprint. **Tier 3 (steers — STRETCH,
    not gate)** — argmax flip or continuation change on a targeted
    prompt; footprint physics stated up front (0.5% of one layer ×
    36 layers measured 0/60 flips twice); a Tier-3 pass implies a
    widened export as a FOLLOW-ON session, not today. Named
    falsifiers: T1 NO = in-vivo drive shows no weight-borne
    divergence (recorded; the coupling story pivots to
    regime-dependence); T2 NO = in-vivo deltas < synthetic's
    (recorded; the synthetic schedule was already sufficient);
    T3 NO = byte-identical continuations (expected; recorded).
    **Drive design, frozen (attack-pass amended):** source =
    `attn_norm(embedding)` — THE model's own input to the adapted
    tensor (computed via `forward_block_states` capture +
    `rms_norm_milli` on the captured embeddings, no new deps);
    mapping = 1 token → 1 substrate step (consecutive tokens give
    causal pre→post pairings at dt≈1 ms, factor 0.95 — food for
    the PAIRING-SELECTIVE, CLAMP-RECTIFIED channel (amendment #7:
    "LTP channel" contradicted by ISC-78; the per-class
    raw/absorbed/applied counters are REQUIRED in the example;
    expect the clamp-rectified regime under dense co-firing);
    N-steps-per-token REJECTED: it would MANUFACTURE sustained
    same-step LTD — the mechanism the dead-wire era ran on only as
    an artifact); scaling = ONE frozen global constant k set so the
    corpus-wide RMS lands mid-band (~450 μA; printed once) —
    per-step per-dim current = clamp(h_dim × k, ±1000), sign
    preserved, with the CLAMP AUDIT printed (hit fraction + per-dim
    rail concentration; >10% clamped = recorded caveat —
    amendment #5); DRIVEN DIMS 0..408 ONLY (amendment #6, fork (a)):
    token features drive the E population; the I population keeps
    the VALIDATED fixed I_INH=600 wall regime — mechanism
    attribution stays on validated ground; the 512-dim purity
    variant is a named follow-on. Corpus = the sha-pinned README
    slice (18fb5452…), tokens repeated in WHOLE EPOCHS — run
    floor(2000/N) epochs and STOP BEFORE THE WRAP pairing
    (amendment #4: the epoch-boundary last→first pairing lands at
    dt≈1 ms full factor on one arbitrary dim pair every epoch —
    concentrated artifact, excluded by construction); population:
    the 512-neuron slice net (409 E / 103 I), live substrate
    (post-F), STDP on, γ=125, census-shuffle + zero controls,
    seeds D-2-verbatim. TIER-2 ADDITION (from the pass): a
    1.5c-driven comparison column (same net, synthetic drive)
    recorded beside the in-vivo run — the purpose statement
    ("more structure than synthetic") is otherwise undecidable. New frozen example
    `hybrid_invivo.rs`; hybrid_gate/hybrid_loop untouched.
    Language constraint carried: "backprop-free local adaptation
    of a shipped quantized LLM through a spiking substrate" —
    "editing/steers" only on a Tier-3 pass.

- 2026-08-19 (sF) · **The fix landed; the lineage re-pinned; the D-2
    gate criterion fork goes to the principal.** The a1b reorder is
  in (ISC-73): synapses transmit, one-step-delayed, exactly-value-
  tested on both grids. The whole library suite survived green — no
  unit pin ever exercised live transmission (ISC-74); the 1.5
  examples reproduce IDENTICALLY (small weights, mV absorption —
  physics, not luck). The D-2 experiment re-ran into a new pinned
  state (ISC-75) whose headline is the **Hebbian reversal**: with
  the wire live, correlated intra-group pairs POTENTIATE (pre
  causally drives post one step later → LTP) instead of the
  dead-wire era's co-fire LTD — Δ-SI flipped −0.31-signed to
  +0.11-signed intra, |Δ-SI| = 1.0000 both eras. The frozen gate's
  DIRECTIONAL condition encoded the dead-wire mechanism and now
  reads COLLAPSES; hybrid_loop's surgery is parked (exit 1, by
  design). Options for the principal: **(i) redefine selectivity to
  |Δ-SI| ≥ floor with the direction PRINTED as the mechanism label**
  (LTD-carried vs Hebbian-carried) — my recommendation: the
  magnitude is the selectivity claim, the sign is the mechanism,
  and pretending the old direction is still the physics would be
  the circular-green failure mode in reverse; **(ii) keep the
  frozen criterion, record the post-fix D-2 as COLLAPSES** — honest
  but mislabels perfect Hebbian discrimination as a failure;
  **(iii) re-freeze the example entirely with both directions
  reported and no pass/fail on sign.** Whichever is called: the
  pre-fix ADAPTS verdict stands as the history of the dead-wire
  substrate; the new pinned state (35,115/35,136/35,157 · flips
  708,029 · Hamming 24.80% · Δ-SI −1.0000 Hebbian) is the D-2
  record going forward, and hybrid_loop's asserted preconditions
  update FROM THIS RE-RUN (never by transcription — the second
  reviewer's doctrine) once the criterion is called. Sweep
  predictions: 2/4 exact, 2 under-called in our favor — coherent
  group bursts (not random σ) cross sub-threshold margins on mV,
  and weight ARRANGEMENT out-carries census content on centi; both
  recorded as findings (ISC-76). alpha.2 manifest grows: the decay
  split + fad081f renames are public-API changes.
- 2026-08-18 (sE-1c) · **THE TRANSMISSION BUG: found, pinned, fix fork to
    the principal.** `step()` Phase 2 injects `weight/10` after
  integration; the next step clears it before integrating — the
  recurrent current NEVER reaches `integrate_and_fire` (canary-pinned,
  ISC-72). This is the true mechanism behind every
  "weights-don't-matter" result in the lineage (D-2 G2, stage 1, stage
  1c) and it invalidates the co-blocker framing: the mV grid was real
  but irrelevant — the wire was cut. The fix is a loop reorder
  (integrate BEFORE clear, or defer injection to next-step integration
  — either makes the Phase-2 pulse live with a one-step delay, which
  the code comment always claimed it had). But the fix RE-PINS
  EVERYTHING downstream of step(): 1.5b/1.5c/1.5d selectivity numbers,
  D-2's 35,157/flip/Hamming/Δ-SI set, stage-1/1c curves, the
  visualizer's sustained-firing behavior — every recorded network
  dynamic in the repo's history. That is a lineage-wide re-pin and a
  de-facto substrate-semantics change: THE PRINCIPAL'S CALL, not mine.
  Options on the table: (a) fix now, re-pin everything in one
  dedicated session (the canary flips loudly, every pinned suite
  re-runs, new lineage begins — the honest path to a live channel);
  (b) fix behind a `transmission_live` opt-in flag (default dead =
  historical numbers keep, experiments opt in — but then the SUBSTRATE
  ITSELF stays broken-by-default and the library ships a bug as a
  feature); (c) accept dead transmission as documented behavior (the
  "instantaneous-synapse model" reading) — rejected as dishonest: the
  comment says the pulses CONTRIBUTE; they don't. My recommendation:
  (a) — a spiking network whose synapses don't transmit is broken, and
  every future result inherits the break; the re-pin cost is one
  session, once.
- 2026-08-18 (sE-1) · **The amplitude sweep said NO — the coupling
    redesign conversation reopens, options to the principal.**
    The sweep (ISC-71) closed the amplitude road: identical trains at
    every amplitude, an exact E-silence cliff below ~450 μA
    (self-consistent — no recurrent bootstrap), and no weight-borne
    divergence anywhere. Three directions now on the table, none
    chosen: (a) **coupling constant** — the substrate's weight/10
    transmission truncation (±125 → ±12 μA) is the actual knob; a
    stronger recurrent gain (or conductance-based transmission, the
    bio path the step loop bypasses) would rescale σ vs margin
    without touching amplitudes — but changes what imported weights
    MEAN dynamically and must be gated as a substrate change; (b)
    **in-vivo drive (stage 2 brought forward)** — the model's own
    layer-0 activations as input currents; coupling then rides on
    real input statistics rather than a synthetic schedule, but the
    same transmission gain question decides whether it can matter;
    (c) **balanced background** — sweep I_INH down with I_ACTIVE (two
    variables, new experiment class) so the inhibitory wall drops
    with the margin. All three need the same prerequisite honesty:
    whatever changes, the 1.5c/D-2 lineage ends and a new pinned
    state begins. The principal picks the road; the sweep curve rides
    into the paper either way as the honest measurement that
    amplitude alone cannot couple a 512-slice at these dynamics.
- 2026-08-18 (sE-0) · **The judge upgrades to the KLD instrument; the
    redesign is staged 0→1→2 (scale last) — the principal's calls.**
    Deep-dive (read-only audit, pre-Stage-0) found the design flaw
    that matters: STDP's learning signal is drive-informed, not
    model-informed — at I_ACTIVE=600 the E-neuron threshold margin is
    ~450 μA while a recurrent ±12 μA (weight/10, i16-truncated) pulse
    is integer-mV-quantization-absorbed on the excitatory climb
    (an 8.6–11.2σ event needed to gate; empirically imported/control/
    zero all fired 35,157), so Δ-SI 1.0000 measured our DRIVE's
    correlation structure with the pretrained weights as canvas.
    Corroborated by mechanical sign analysis of the session-E dumps
    (5–8/12 positive steps per prompt, binomial p ≈ 0.12–0.23 —
    directionless at the top-1). Decisions: (a) the fork's
    llama-perplexity (--save-all-logits + --kl-divergence) becomes
    the standing sensitive judge, corpora sha-pinned; (b) the staged
    redesign — stage 0 instrument/control (DONE, ISC-68..70), stage 1
    amplitude sweep to open the weight→firing channel (grid extended
    to 100 μA, below the ~150 μA E-threshold, on the audit's
    arithmetic; I_INH fixed at 600 — single variable; spike-TRAIN
    comparison, not total counts, with pre-registered A* criterion),
    stage 2 in-vivo drive (its own design conversation), scale LAST;
    (c) the drive-domination finding + the sweep curve are themselves
    publishable figures. An earlier /tmp summary note's loose "~6σ"
    phrasing on corpus B is SUPERSEDED by ISC-70's conservative
    claim. The session-E widened-slice plan (rung B) is WITHDRAWN —
    scaling a signal that is structurally not about the model
    multiplies noise.
- 2026-08-18 (sE) · **THE LOOP GATE: CLOSED — and the claim it earns
    is capability, not quality.** For the first time in the field's
  record: a shipped quantized LLM's weights, imported bit-exactly into
  a spiking substrate, adapted under local backprop-free STDP,
  re-encoded through the format bridge, and RUN by foreign tooling
  with MEASURABLE behavior change (60/60 judged steps moved; 0/60
  argmax flips — one attention slice of 36 layers, 0.5% of weights).
  Recorded decisions: (a) ORIGINAL fp16 scale bits pass through —
  the substrate adapted structure at γ=125; magnitudes stay the
  model's own; (b) 512×512 slice for D-2 state continuity (widening
  ladder pre-declared, did not fire); (c) effect size honestly small
  — no improvement claim is made or implied anywhere; the loop's
  value is that it CLOSES, with every attribution link mechanically
  verified; (d) judge = the pinned fork commit + spec-rewritten dump
  patch (scratch tools stay in /tmp by policy — numbers banked
  same-session here and in the commit message when committed);
  (e) alpha.2 republish (q2_0 layout fix + encode_q2_0) consolidated
    AFTER this session per the principal's call. All evidence in
  /tmp/opencode/se/; verdict numbers banked in this entry. Commits
  held for the principal's review — no commit/push/merge until the
  working-tree diff is reviewed together.
- 2026-08-17/18 (s4-D2 recovery) · **Machine died mid-session; session D
    slice 2 recovered from git + this ISA, evidence regenerated by
    re-run.** The shutdown hit between the last D-2 code commit
    (12:22) and the docs leg — the working tree was clean, both D-2
    commits (`819e495` finalize_synapses, `7d4df1e` hybrid_gate) were
    intact with full verdicts in their commit messages, and all four
    model files survived. /tmp died with the machine: every
    /tmp/opencode/* evidence log (s4d falsifiers + the 4C fork-anchor
    dumps + the C-pre refcmp scratch harness) was lost. Recovery per
    the principal's calls: (a) 4 CI gates re-run green on the
    committed code (216 tests: 76 rt + 137 snn + 3 app — the +2 snn
    are finalize_synapses' own); (b) hybrid_gate re-run twice — every
    G1/G2/G3 number matches the 7d4df1e commit message exactly,
    determinism re-proven byte-identical; (c) slice-1 logs
    regenerated: probe/forward/full/gate all reproduce ISC-60/61
    number-for-number (gate YES 5/5 again, p0–p4 texts byte-identical,
    residuals 4 143 575–4 313 332 + chat 6 274 426, RSS 2.16 GB; wall
    13:08 vs 11:21 recorded — post-reboot load, wall line only);
    ignored real-file suite 5/5 (release, 88 s). NOT regenerated (out
    of approved scope, anchors lost with /tmp): drift_q2_0.log (needs
    the 4C fork dumps + refcmp harness) and the 1.7B 43/43 gate
    byte-diff (needs the s4b baseline) — their ISA-recorded numbers,
    parsed mechanically per this ISA's own discipline, stand as the
    record. Commits stay LOCAL per the principal's verification gate.
- 2026-08-17 (s4-D) · **GATE ON TERNARY-BONSAI-4B-Q2_0: YES — 5/5.
    The family's first YES (2-bit tier); the per-model record now
  reads 1.7B-Q1_0 NO 3/5 · 4B-Q1_0 NO 4/5 · 4B-Q2_0 YES 5/5 — three
  honest rows, all fork-attributed, all in the paper.** The 4C coda's
  physics landed exactly: 2-bit restored what 1-bit destroyed (p3 "
  Thursday04/05/2018 " — fork-byte-identical to the 4C greedy,
  trailing space included), p2's " five" passes with room to spare,
  and the drift puts our logits at the C-core bar against the fork's
  own dumps (p3 12/12 argmax, |Δtop| ≤ 0.289). The single honest
  blemish INSIDE the passing continuations: our p2 greedy diverges
  from the fork's after " ten" (step-6 near-tie, 0.34 logits —
  measured, not hand-waved; both runtimes consistent with their own
  choice). The gate example's YES line still prints "on real Q1_0
  weights" — a FROZEN-artifact string inaccuracy on this run,
  recorded here rather than edited (the loaded file and all evidence
  lines name Q2_0 explicitly). The Q2_0 slice is the ternary seam
  Session D slice 2 (Bonsai weights → Trit → SNN → STDP) builds on.
- 2026-08-17 (s4-D) · **4B Q1_0 gate NOT re-run — rationale (principal
  pin, recorded as decided):** the routing is STRUCTURAL IDENTITY for
  ty 41 — `quant_tensor` calls `q1_0_tensor` verbatim, so a Q1_0 file
  constructs `QuantData::Q10` and executes the same functions on the
  same bytes as pre-session-D; the only mutable surface is the call
  indirection, and the active guards are (a) the 1.7B gate
  byte-diff 43/43 verdict-bearing lines, (b) both Q1_0 ignored
  real-file suites green (incl. incremental≡forward exact), (c)
  probes green on both files. A 37-minute re-run of the 4B Q1_0 gate
  would add no guard the first three do not already encode; its
  recorded verdict (NO 4/5) stands untouched.
- 2026-08-17 (s4-D) · **Probe scale-window policy update (fog (e)
  grows its second data point):** the q2_0 first-block scale reads
  18.68 milli (max|w|) — inside even the q1_0 window [1,100]; the
  q2_0 provisional window is [1,1000] (max ≥ mean convention, first
  real file — narrows for the next one). Curiosity recorded, not
  claimed: the 4B Q1_0 file's first block carries the SAME fp16
  scale bits (0x24c8) despite the mean-vs-max convention difference.
- 2026-08-17 (s4-D) · **Q2_0 residuals are structurally calmer** —
  4.1–6.3 M vs Q1_0's 11–15 M on the same prompts (6–9% of the 60.0 M
  derived rail): real ternary zeros genuinely shrink the residual
  stream. Also: the branch-free q2_0 inner loop (`(code−1)·a`, four
  LSB-first lanes per byte) took the 4-token full forward 193 s →
  21.6 s with BIT-IDENTICAL output (the pre-optimization run is the
  witness — residual + top-5 equal); the gate inherits it at 11:21
  wall.
- 2026-08-17 (s4-D) · **THE STAGE-2 Q2_0 PIN WAS WRONG — re-pinned from
  source + file before any compute was built on it.** The probe-first
  discipline caught it on the first run: all 253 q2_0 tensors FAILED the
  Stage-2 arithmetic (18 B per 64 weights, 720 B per 2560-wide row) — the
  real file measures **680 B/row = 34 B per 128 weights**. The fork's
  actual `ggml/src/ggml-common.h:187` defines `QK2_0 = 128`,
  `qs[QK2_0/4]` (fp16 `d = max|w|` + 32 code bytes; quantizer
  `q = clamp(round(w/amax)+1, 0, 3)` → code 3 still unreachable; dequant
  `11 → +2·d`; lanes LSB-first unchanged). The 2026-08-15 "correction"
  ("C code defines QK2_0 = 64, the g128 label was loose") had it
  BACKWARDS — Stage 2 fetched/misread the source and then validated the
  wrong layout against self-derived hand vectors (the ISA's own
  recurring lesson, now at format-spec scale). Fixed: `bridge.rs`
  (`Q2_0_BLOCK = 128`, 34 B decode, tests re-derived from the C
  formulas + a geometry pin citing the file measurement),
  `ternary_format_gate` (re-run: YES), `bonsai_probe` per-type
  arithmetic. Blast radius contained: no q2_0 bytes were ever consumed
  by anything before this session (the 4C evidence is fork-side,
  unaffected). The `alpha.2` republish checklist gains this as a
  must-ship fix.
- 2026-08-17 (s4-D) · **Ternary-Bonsai-4B-Q2_0.gguf facts, pinned
  BEFORE the compute path existed** (probe log
  /tmp/opencode/s4d/probe_q2_0.log): 1 074 969 344 B (= the 4C fetch,
  md5 0bffe9323f3e27e64574f8884fbfecef); GGUF v3, **398 tensors =
  f32×145 + q2_0×253** (all 253 byte-exact vs `rows × ⌈cols/128⌉ × 34`;
  the 145 f32 are the norm tensors incl. output_norm — mixed file, as
  expected); `token_embd.weight` carries **24 B alignment padding**
  (103 134 920 → 103 134 944, same shape as the 4B Q1_0 — the only
  padded tensor); config KVs IDENTICAL to the 4B Q1_0 (36 blocks,
  emb 2560, FFN 9728, 32Q/8KV, key_length=value_length=128, rope base
  **5e6**, YaRN yarn/4.0/8192, eps 1e-6, context 32 768); first
  embedding block: **scale fp16 0x24c8 = 18.68 milli (max|w|)**, trits
  +37/0×43/−48 of 128 — and the 4B Q1_0 file's first block carries the
  SAME fp16 scale bits (0x24c8 = 19 milli) despite the mean-vs-max
  convention difference (recorded observation, not a claim about the
  quantizers). HF repo (fetched this session) ships FOUR variants:
  Q2_0.gguf (on disk, this file), **Q2_0_g64.gguf and PQ2_0.gguf
  (NOT downloaded — no type in the fork source we hold reads a "g64"
  layout; existence recorded per mission, contents unverified)**, and
  F16.gguf (used by 4C, deleted-after? no — on disk 8 GB). Probe scale
  window for q2_0 set provisionally to [1, 1000] (max|w| ≥ mean|w|;
  first real value 19 milli — fog (e) gains its second data point,
  same 19 as the Q1_0 4B). Peak RSS 1.03 GB (probe).

- 2026-08-16 (s4-4B) · **GATE ON BONSAI-4B: NO — 4/5, attributed
  same-session.** Same frozen prompts, same verdict logic, bigger
  tier: p0/p1/p2/p4 PASS with continuations BYTE-IDENTICAL to the
  reference fork's greedy (incl. p2 " five…" — the 1.7B's failure
  prompt — and p4 " Paris. Paris is the capital of France. Paris is
  the"); p3 FAILS on both runtimes (ours ", 2024, 10:0", fork
  ", June 12, 2018,"; both argmax "," at the verdict step, " Thursday"
  outside the fork's top-10; the continuation divergence is the
  drift-measured step-1 near-tie, 0.1 logit). Teacher-forced fidelity
  on 4B: argmax 35/36 across p2/p3/p4, mean overlap 9/10, max |Δtop|
  0.427 — same class as C-core's 1.7B bar (36/36, 0.597). The 4/5 is
  the model's ceiling under greedy on this set; **the 8B-vs-stop call
  is the principal's** (per-model record: 1.7B NO 3/5, 4B NO 4/5,
  both fork-attributed — a YES would have flipped Stage 4 to "the
  stack" per gate doctrine; it did not arrive).
- 2026-08-16 (s4-4B) · **The 4B config diff — and the two silent
  breakers the mission's "flag anything" instruction was for.**
  Diff vs 1.7B (pinned from the files' own KVs): blocks 28→36, emb
  2048→2560, heads 16→32, FFN 6144→9728, freq_base **1e6→5e6**;
  kv_heads 8, head_dim 128 (key==value), YaRN yarn/4.0/8192, eps
  1e-6, vocab 151 669 all UNCHANGED (the vocab guess from Qwen3-4B
  upstream was wrong — Bonsai keeps the shared 151 669). Breaker 1:
  `rope.freq_base` differs per tier — the old `expect_kv` pin would
  have been a hard ConfigMismatch (good) but a default would have
  been silently wrong rope tables; it is now a REQUIRED KV.
  Breaker 2: the 4B `token_embd.weight` slice carries 24 B of
  alignment padding (54 600 840 → 54 600 864, the next tensor's
  aligned offset — the 1.7B total was 32-divisible by luck);
  tensor-size checks now accept exact-or-alignment-padded and copy
  only the formula bytes. Also: the attention context is
  `heads·head_dim` (4096 ≠ emb 2560 on 4B) — q/attn_output are
  genuinely non-square on 4B; dims validation covers it.
- 2026-08-16 (s4-4B) · **Pin dispositions (principal's three):**
  (1) frozen-file edit hardened into the pin-1 gate — pre-refactor
  baseline captured at e1821be, post-refactor byte-diff 43/43
  verdict-bearing lines identical; the one mid-run scare (bonsai_full
  top-5 "changed") was localized to a STALE RECORD, not the code
  (e1821be worktree witness reproduced the new numbers exactly; see
  Learning). (2) The deep-dive ladder did NOT trigger — its
  condition is "4B lands 3/5 AND the fork agrees"; 4B landed 4/5
  with fork agreement on the single failure. Rung (a)
  (Ternary-Bonsai-4B Q2_0, quantization severity) and rung (b)
  (Bonsai-4B.gguf unquantized, ~8 GB mmap vs 2.3 GB free) remain
  AVAILABLE to the principal if he wants the p3 failure decomposed
  (quant vs base capability) — at 4/5 with both runtimes denying
  " Thursday" at step 0, the question is narrow. (3) RAM: peak RSS
  1.17 GB on every 4B heavy run vs 2.3 GB available — headroom held;
  the buffer-drop-after-load option stayed in reserve, unused.
- 2026-08-17 (s4-4C) · **P3 DISAMBIGUATED: quantization severity —
  the 1-bit quant destroyed the weekday chain; the base and 2-bit
  ternary have it decisively.** Fork-side ladder only (greedy forced,
  identical flags, NEURALOS_DUMP step-0 top-10s): F16 base puts
  " Thursday" (7794) TOP-1 at 14.6529 (+3.054 margin); Q2_0 TOP-1 at
  14.6527 (+3.083) with the step-0 top-10 IDENTICAL to F16 in order,
  max |Δlogit| 0.0286 — 2-bit ternary is near-lossless against its own
  base on this prompt; Q1_0 (s4b) had 7794 outside the top-10 (top-1
  "," at 12.81). The prompt-shape hypothesis is FALSIFIED — the
  unquantized base completes the bare prompt at top-1. Generations:
  Q2_0 " Thursday04/05/2018"; F16 " Thursday04/10/2018" (same weekday
  head; date digits diverge — the drift-measured near-tie class).
  p2 corroborates: " five" margin +0.930 (Q1_0) → +2.972 (Q2_0),
  3.2x. Chat shape (void-check PASSED — specials as control
  ids in the verbose token log) goes to clarification-land at EVERY
  width: Q1_0 "It seems like your message might be a typo or
  incomplete." ("It" top-1 13.17, <|im_end|> rank-4); **F16 chat
  step-0 is the same animal — "It" top-1 at 13.66, no 7794 in the
  top-10, top-3 order identical to Q1_0's** — the template's framing
  suppresses completion-mode on the base itself, so chat-shape can
  never surface this knowledge class at any bit-width (and the
  failure at the frozen gate, which runs BARE, is not a chat
  artifact at all).
  Matrix row: base PASS + Q2_0 PASS → quantization severity. **4/5
  stands as the frozen Q1_0 gate's honest cap; 8B-under-1-bit is a
  capacity bet now grounded (2-bit restored the class at 4B), and the
  cheap sibling option is on the table — the frozen prompts on Q2_0-4B
  would very likely read 5/5 (outside the frozen gate's quant; both
  calls the principal's).** No runtime changes; docs-only diff.
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

- conjectured (session F, folded into the criterion commit): the
  live-wire discrimination was "Hebbian — pre causally drives post
  one step later → LTP"; the label was printed as computed from the
  class-mean signs even after the histogram shipped.
  refuted by: the session-G per-class counters, run BEFORE push —
  raw intra drift is net NEGATIVE (−739,295; LTD events outnumber
  LTP ~1.3:1 per event and −5 vs +4 per event), the 0-floor absorbs
  −839,029, and only the APPLIED residue (+99,734) is positive. The
  sign of the realized movement was carried by the BOUNDS, not by
  LTP dominance; "counted, not inferred" had been claimed for the
  pairing histogram while the LABEL itself was still an inference
  from two signs.
  learned: an instrument earns its name only when the claim it
  guards is computed FROM the instrument's output — a histogram
  beside an inferred label is decoration. Decompose realized
  movement into raw / absorbed / applied before naming any
  plasticity mechanism.
  criterion now: mechanism labels are computed from per-class
  decompositions (raw, clamp-absorbed, applied), with the
  computation visible in the printing code; sign-only label logic
  is a review flag.
- conjectured (session F, from the second review's Job-3
  derivation): with transmission live, the mV grid would show
  timing-only divergence (Hamming > 0, rate-L1 ≈ 0) and centi would
  carry the recruitment; random-σ statistics were the basis.
  refuted by: the post-fix sweeps — mV showed BOTH (Hamming 58,779
  AND rate-L1 689, with sub-cliff recruitment at 300 μA:
  1.14/0.99/0.00 Hz), because same-group bursts are COHERENT, not
  Gaussian: a synchronized ~5-spike volley stacks ~+60 μA into one
  step and crosses margins random fluctuation cannot. The reviewer's
  own σ math was right about noise and wrong about structure.
  learned: in driven group-structured regimes, coherent transient
  sums dominate over variance statistics — dead-zone margins
  computed against σ under-call what correlated populations do.
  criterion now: any per-neuron margin analysis in this substrate
  quotes BOTH the random-σ crossing time AND the coherent-volley
  threshold; sweep predictions pre-register which regime they
  assume.
- observed (session F): the D-2 discrimination sign REVERSED with
  the transmission fix — intra Δ went −0.3133 (LTD-carried) to
  +0.1075 (LTP-carried), |Δ-SI| = 1.0000 on both sides of the fix.
  why it matters: the pre-fix "selectivity" was an artifact of the
  same-step co-fire tie-break (dead wire ⇒ spikes were drive-only ⇒
  correlated pairs met only in the LTD tie-break); the wire live,
  causality flows pre→post and the pair meets in LTP instead. The
  gate's directional condition was a mechanism label mistaken for a
  correctness invariant.
  learned: a metric's direction encodes the mechanism of the era it
  was written in — when the mechanism changes under a fix, the
  metric's SIGN must be re-derived, not inherited; the magnitude
  claim (discrimination) and the mechanism claim (which rule
  carries it) are separate assertions and should be reported
  separately.
  criterion now: selectivity metrics report |effect| and named
  direction as two fields; gates assert magnitude, print mechanism.
- conjectured (stage 1, then stage 1c, twice in one day): first, that
  lowering the drive amplitude would open the weight→firing channel;
  then, that the centi-mV grid would.
  refuted by: two frozen sweeps — zero train divergence at every
  amplitude on BOTH grids — and finally by a 5-line probe: the
  recurrent current is never integrated in step() at all (Phase 2
  injects after integration; the next step clears before integrating).
  The channel both experiments swept levers on was structurally dead
  upstream of both levers.
  learned: when a second lever ALSO produces zero signal, stop
  modeling the physics and trace the plumbing — two falsified
  mechanism-models in a row means the mechanism isn't at the layer
  being swept. Worse: the session-E audit had already READ this code
  and reported "a pure one-step pulse with delay" — it re-encoded the
  code comment's INTENT, not the code's BEHAVIOR. Both failures are
  the same class: an unverified mechanism claim laundering itself as
  an audited fact.
  criterion now: before any experiment sweeps a lever, a minimal
  end-to-end probe must prove the lever is CONNECTED through the
  actual pipeline (the 2-neuron transmission probe now exists as a
  permanent canary); and an analysis claim about signal flow cites
  the executed instruction path or a probe output — never the
  comment above it.
- observed (session E): the hybrid_loop surgery assert
  `changed_cells == D-2 Hamming` fired on first run — 57,300 vs
  57,005, 295 phantom changes.
  refuted by: the assert itself, before any byte reached disk — the
  adapted-slice reconstruction initialized the DIAGONAL to Zero
  instead of copying the source trit (the diagonal carries no
  synapse in the full-minus-diagonal build). The doc comment said
  "keeps its source trit"; the code didn't do it. 295 = exactly the
  nonzero diagonal count of the imported slice.
  learned: a cross-assert between two independently-derived
  representations of the same quantity (cell-delta count vs synapse
  Hamming) catches stated-intent-vs-code drift that unit tests on
  either half alone cannot — the encoder's tests were green, the
  surgery's logic was green, only their JOIN was wrong.
  criterion now: whenever two computations claim the same number,
  assert their equality at the join even when both "pass" their own
  checks — especially before a write path runs.
- observed (session D-2 recovery, 2026-08-17/18): the machine died
  mid-session and every evidence log under /tmp/opencode/ died with
  it — yet the session recovered fully in under an hour.
  why it worked: the durable record carried the load — verdicts and
  headline numbers lived in commit MESSAGES (not just the diff), the
  ISA carried the parsed evidence + falsifier paths, and the examples
  were frozen + deterministic, so every lost log was REGENERABLE by
  re-run and cross-checked number-for-number against the recorded
  values (all matched; only wall lines moved).
  learned: /tmp evidence is a CACHE, not a record — treat it as
  disposable by policy, and the policy costs nothing if the three
  durables (verdict-in-commit-message, ISA-parsed numbers,
  deterministic frozen examples) are already standing discipline.
  The one gap: scratch harnesses that live only in /tmp (refcmp) or
  baselines diffed against /tmp logs (the 43/43 byte-diff) are NOT
  regenerable — their claims are only as durable as this file's
  recorded numbers.
  criterion now: verdict-bearing numbers get INTO the commit message
  or the ISA before the session ends (already the norm — now the
  stated reason), and a scratch tool whose output backs an ISA claim
  gets its source committed beside the claim the first time it
  carries a falsifier, or the claim is marked anchors-lost-on-reboot.
- conjectured (Stage 2, 2026-08-15, latent until session D): the q2_0
  layout was correctly pinned from the fork's C source as 64 weights /
  18 bytes, validated by hand-derived test vectors.
  refuted by: the first real q2_0 file — every one of its 253 q2_0
  tensors failed the arithmetic (680 B/row measured, 720 predicted);
  the fork's actual ggml-common.h defines QK2_0 = 128, qs[32] (34 B).
  The Stage-2 fetch misread the source, and the self-consistent hand
  vectors then "confirmed" the wrong layout for two sessions — the
  known hand-vector failure mode, now at format-spec scale, and it
  survived precisely because NOTHING ever consumed a real q2_0 byte.
  learned: a format pin is not verified by tests derived from the
  same reading that produced the pin — circularity again, third
  distinct surface (math units, shared derivations, now layout
  pins). The probe-first discipline (measure the real artifact's
  byte arithmetic BEFORE building compute on the pin) is what caught
  it in one run.
  criterion now: any layout pin gets either an independent re-read of
  the reference source against the derived constants or a real-file
  byte-arithmetic check before anything downstream trusts it; a codec
  with no real file to eat stays marked UNVERIFIED-BY-ARTIFACT in the
  spec.
- conjectured (session D, four instances in one sitting, all
  test-side): my q2_0 test expectations — the zero-input TooShort
  case at 128-wide; the hostile-scale census (assumed ±1 alternating
  codes where the builder emitted −1/0); the golden-vector zero count
  (hand formula 154 vs constructed 153); the 2560-wide sweep (40
  blocks — the PRE-re-pin 64-group mental model leaking into the
  post-re-pin test).
  refuted by: cargo test, four times; every fix was "derive from the
  construction, not from a mental model of it."
  learned: the ISA's standing lesson now fires on NEW-geometry
  derivatives too — when a layout constant changes, tests written in
  its shadow inherit stale arithmetic; sweep for the old constant's
  multiples (40 = 2560/64) in every test that touches the new one.
  criterion now: after any geometry re-pin, grep the new module's
  tests for block counts derived from the OLD geometry before
  running.

## Verification (Session F — the transmission fix + lineage re-pin)

- ISC-73: the a1b reorder (network.rs step(): decay_adaptation_
  current → integrate → clear-after-read → propagate → plasticity);
  three exact-value transmission tests green; lib suite 152; the
  removed `decay_synaptic_current` + renamed fields = alpha.2
  manifest items; tau_synapse_us/delay_us marked decorative; AGENTS.md
  step-order warning updated.
- ISC-74: workspace suite green (152 snn + 76 rt + 3 app); zero
  unit-pin failures post-fix; ternary_gate (3308/3587, GATE: YES) +
  ternary_selectivity (GATE: YES) byte-stable vs the recorded
  1.5d-era numbers — small weights are mV-absorbed (per-type dead
  zones documented).
- ISC-75: /tmp/opencode/sf/hybrid_gate_fixed.log (+ hybrid_loop_
  fixed.log, exit 1 by design): G2 35,115/35,136/35,157; G3 firing
  35.13 Hz, events 18,817,891, flips 708,029, Hamming 64,877 =
  24.80%, sign crossings 0, intra +0.1075 / inter +0.0000, Δ-SI
  −1.0000 (Hebbian); frozen directional criterion ⇒ COLLAPSES/
  selective-FAIL recorded; surgery parked. Criterion fork = the
  principal's (Decision above).
- ISC-76: /tmp/opencode/sf/sweep_mv_fixed.log + sweep_cmv_fixed.log
  — full 9-row tables both grids, A\* = 600 both; mV 600: H(i,c)
  58,779, L1 689, totals 35,115/35,136/35,157; mV 300 recruitment
  row (1.14/0.99/0.00 E-Hz); centi 600: H(i,c) 39,011 > H(i,z)
  23,907 ≈ H(c,z) 23,958; centi 300 zero-outfires-wired (7.50 vs
  6.97); centi 150 recruitment (0.70/0.62/0.00); centi 100 all-
  silent; I-rates diverge at 600 on both grids. Prediction scorecard
  in the claims. Visualizer smoke (NEURALOS_SMOKE_MS=3000) exit 0.
  4 CI gates green on the final tree.

## Verification (Session E stage 1c — the finer-ruler sweep + the transmission bug)

- ISC-72: /tmp/opencode/se/stage1c_sweep.log + .time — the centi-grid
  table (totals 35,975/33,930/31,885/29,840/27,795/27,795/25,750 ×3
  down the grid; train Hamming 0 and L1 0 at every amplitude, every
  pair); VoltageResolution shipped in fad081f (149 snn tests at the
  seam; 150 with the 1c canary — pinned mV trace, dead-zone pairs,
  network pairs);
  `recurrent_current_is_never_integrated_in_step_bug_pinned` green
  (presynaptic fires; postsynaptic membrane never leaves −70);
  clippy/check/no_std green.

## Verification (Session E stage 1 — the amplitude sweep, honest NO)

- ISC-71: /tmp/opencode/se/stage1/sweep.log + sweep.time — the full
  9-amplitude table (E/I Hz per net, totals, 3 pairwise train
  Hammings, 3 pairwise rate L1s per row — all zeros across the
  grid); 600 μA row reproduces D-2's 35,157 ×3; cliff row pair
  450/300 (32,294 ×3 with E at 8.00 Hz → 25,750 ×3 with E at
  exactly 0); clippy -D warnings clean on the example; determinism
  inherited (same seeds/noise as the D-2 lineage; single run
  suffices for a zero-divergence claim at identical inputs — every
  pair differs only in weights and produced identical trains).

## Verification (Session E stage 0 — instrument + attribution package)

- ISC-68: /tmp/opencode/se/stage0/hybrid_loop_control.log — control
  mode full pipeline on unadapted trits; code 0/65,536 · scale 0/4,096
  · S1 outside 0 · S2 0 mismatches · CONTROL IDENTITY assert (written
  file == original, byte for byte) · sha256 4e0bf8b737b0431528b… both
  files.
- ISC-69: llama-perplexity @ the pinned build dir; corpora sha
  18fb5452… (readme lines 1–180) + 781d1e21… (frozen prompts +
  banked continuations); baseline double-runs Final-exact on both
  (15.6723 / 5.1400); kld files base_readme.kld + base_prompts.kld.
- ISC-70: ppl_patched_readme.log — ratio 1.002780 ± 0.010918, mean
  KLD 0.004271, max 0.289815, cor 99.93%; ppl_patched_prompts.log —
  chunk ln-ratios +0.14895/+0.07446/+0.04904 (3/3 positive), Δp RMS
  0.36–0.50%, Same-top-p 100.000% ×3 chunks; conservative reading in
  the claims section; STAGE0_SUMMARY.md carries the full tables.

## Verification (Session E — the loop-closer, LOOP GATE: CLOSED)

- ISC-65: bridge tests green (27 bridge incl. the 4 new):
  encode_q2_0_reproduces_known_vector_bytes,
  encode_q2_0_reproduces_golden_vector_bytes (decode→encode byte
  identity on both pinned vectors), encode_q2_0_rejects_bad_input
  (odd length, short out, short SCALES — per-block indexing is
  load-bearing), prop_q2_0_round_trip (distinct per-block scales,
  dirty out-buffer).
- ISC-66: /tmp/opencode/se/hybrid_loop.log (+ .time, run2 +
  loop-determinism.gguf) — phase-1 D-2 numbers all asserted+reproduced
  (spikes 35,157 ×3 · events 16,183,885 · flips 321,571 · Hamming
  57,005 · Δ-SI 1.0000); tensor window abs 112,588,032 +
  2,785,280 B dims-derived; 512×136 B chunks; code bytes 29,734
  changed · scale 0 · outside 0; patched file 1,074,969,344 B; S2
  262,144 trits from disk == adapted, 0 mismatches; wall 12.0 s, RSS
  2134 MB (loop budget 2560 — two file buffers, header documents it);
  sha256 87078612… ×2 identical.
- ISC-67: fork @ 9ca265a (clone HEAD == pin), llama-completion built
  clean after one API rename (llama_vocab_n_tokens — the C-pre patch
  spec predates the rename); run_prompts.sh protocol (5 frozen
  prompts, greedy-forced + `-t 4` `-c 512` `-n 12`, NEURALOS_DUMP=1,
  double-run per variant): baseline run1==run2 byte-identical ×5;
  patched run1==run2 ×5; baseline reproduces 4C anchors (p3 step-0
  7794:14.6527, p2 step-0 4236:13.2278); delta.py mechanical table:
  60/60 steps moved (max |Δ| 0.534 p2 / 0.452 p0 / 0.124 p3 / 0.093
  p4 / 0.088 p1), overlap 8–10/10, argmax flips 0/60, continuations
  byte-identical to baseline on all 5. Working tree left UNCOMMITTED
  for the principal's review per standing instruction.

## Verification (Stage 4, session D slice 2 — the hybrid seam, recovered)

- ISC-63: network.rs tests green in the 4-gate re-run —
  finalize_synapses_sorts_external_adds_and_builds_both_csrs,
  finalize_synapses_makes_ltp_reachable_on_external_wiring (A/B:
  finalized potentiates pre-before-post, unfinalized frozen at 0);
  137 snn tests total (135 + the 2 new).
- ISC-64: /tmp/opencode/s4d/hybrid_gate.log (+ hybrid_gate.run2 +
  .time) — regenerated post-reboot, every number cross-checked
  against the 7d4df1e commit message: G1 0/261,632 mismatches, zero
  fraction 0.3655; G2 35,157 spikes all three comparators, floor
  0.10 Hz; G3 321,571 flips, Hamming 0.2179, sign crossings 0,
  intra −0.3133 / inter +0.0000, Δ-SI 1.0000; VERDICT "HYBRID GATE:
  ADAPTS"; wall 20.5 s, RSS 1050 MB < 1536. Determinism: diff of the
  two runs (wall line stripped) EMPTY. Also in the commit: probe's
  stale pre-re-pin comment corrected output-neutrally, attn_q dims
  pinned [2560,4096].
- Recovery evidence (this session): 4 CI gates green (216 tests);
  slice-1 falsifiers regenerated and matching — probe_q2_0.log
  (253/253, 0x24c8, +37/0×43/−48), forward_q2_0.log (FORWARD: OK),
  full_q2_0.log (36/36, residual 4 237 901, digit top-5, RSS
  2.12 GB), gate_q2_0.log (YES 5/5, p0–p4 byte-identical to ISC-61,
  chat structural PASS 6 274 426, RSS 2.16 GB, wall 13:08), ignored
  real-file suite 5/5 (release, 88 s). Drift + 1.7B byte-diff NOT
  re-run — anchors lost with /tmp, ISA numbers stand (Decision above).

## Verification (Stage 4, session D — Q2_0 native + the family's first YES)

- ISC-57: probe_q2_0.log (0/253 against the old pin → 253/253 after
  re-pin, PROBE: YES); fork source ggml-common.h:187-192 +
  ggml-quants.c quantize/dequantize_row_q2_0 read this session;
  bridge tests re-derived (q2_0_block_geometry_is_pinned pins
  128/34/680); Stage-2 gate re-run: YES; snn suite 135 green
- ISC-58: rt::q2_0::tests 10/10 — reference via published decode_q2_0,
  f64 real-units 2560-wide sweep, fp16-exact pin, code-3 head+tail
  with out untouched, hostile ±inf, golden lane/byte order,
  size-validation-first; 1.7B Q1_0 forward/gate still green
- ISC-59: model tests 76 green incl. q2_0 incremental≡forward
  (tolerance 0) + routing + padding; real_files_load 3/3 (release);
  **1.7B frozen-gate byte-diff: gate_17b_session_d.norm vs
  gate_17b_post.norm — ZERO diff lines, 43/43 verdict-bearing** (3/5
  NO reproduced, exit 1, wall 8:37 vs 9:09 baseline)
- ISC-60: forward_q2_0.log (FORWARD: OK after per-format dispatch;
  the first run's NO was the example's own q1_0 assumptions — 360 B
  stride + 100%-density gate — both fixed with rationale); full
  full_q2_0.log (36/36 alive, residual 4 237 901 < 60 023 992, top-5
  digits, 21.6 s forward post-optimization, RSS 2.12 GB)
- ISC-61: gate_q2_0.log — strict 5/5, chat structural PASS, STAGE 4
  GATE: YES, exit 0; wall 11:21, RSS 2.16 GB; p3 continuation
  fork-byte-identical to 4C (mechanical comparison vs the archived
  anchor); residuals 4.14–6.27 M
- ISC-62: drift_q2_0.log — teacher-forced on fork ids, 12 steps each,
  anchors parsed mechanically: p3 argmax 12/12, overlap 9.83/10 mean,
  max |Δtop| 0.289; p2 argmax 11/12 (step-6 near-tie flip, 0.34),
  overlap 9.5/10, max |Δtop| 0.336 — pin-1 agreement, verdict
  recorded on clean evidence; 4 CI gates green (214 tests), real-file
  suite 5/5 release (155 s), commits local, no push/merge


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

## Verification (Stage 4, session 4B — Bonsai-4B)

- Baseline discipline: 4 CI gates green at e1821be BEFORE any edit
  (195 tests; ignored real-file 4/4 in 712 s debug); fresh 1.7B gate
  baseline log captured at e1821be (3/5 NO, 9:09 wall, RSS 540 MB)
  — /tmp/opencode/s4b/gate_17b_baseline.log.
- ISC-51: probe on both tiers with the full config-KV dump
  (probe_{17b,4b}.log); 4B pre-padding-fix showed the ONE failing
  tensor (token_embd, +24 B) — post-fix 253/253 q1_0 byte-exact,
  PROBE: YES; scale window [1,100] holds on the second real model
  (19 milli; fog (e) evidence gained, window unchanged).
- ISC-52: model.rs tests — from_gguf_reads_the_4b_config_block,
  from_gguf_is_loud_on_missing_and_broken (5 loud paths),
  score_scale_pins_the_1_7b_split ((88, 3883) + (125,0)/(62,5000)
  derived splits), loader_accepts_alignment_padding_and_copies_exact_bytes,
  residual_rail_derives_per_model (emb 2048 → 67 108 864 = 2^26
  exactly; 2560 tighter); ignored real_files_load_with_expected_configs
  green on BOTH files (config equality vs the pinned tables, 2-token
  forwards, all layers alive, residuals under the per-model rail).
- ISC-53: pin-1 hard gate — gate_17b_baseline.norm vs
  gate_17b_post.norm: ZERO diff lines (43/43 verdict-bearing lines);
  real-file suite post-refactor 5/5 release (69 s) +
  real_incremental_matches_forward_exact in debug (888 s, profile
  parity with the 712 s baseline); full-forward witness: e1821be
  git-worktree bonsai_full run == new-code run EXACTLY (residual
  17 965 859; top-5 (16, 11618) (17, 10688) (18, 10646) (15, 10181)
  (19, 9605)) — the s3.5-recorded numbers were pre-C-core.
- ISC-54: probe 4B PROBE: YES (RSS 584 MB); bonsai_forward 4B
  FORWARD: OK (q 4096-wide, k/v 1024); bonsai_full 4B FULL: OK —
  36/36 layers alive, residual 10 958 495 < 60 023 992 derived rail,
  51.5 s, RSS 1.14 GB; drift 4B (refcmp, RSS 1.17 GB): p2 argmax
  12/12 overlap 9/10 Δtop 0.427; p3 11/12 (one step-1 flip, 0.1
  margin); p4 12/12 Δtop 0.064; tokenizer witness: fork 4B p4 ids
  [785, 6722, 315, 9625, 374] == ours; fork E2E 4/5 (greedy forced,
  same flags as C-pre).
- ISC-55: THE GATE 4B — /tmp/opencode/s4b/gate_4b.log: strict 4/5,
  STAGE 4 GATE: NO, exit 1; residuals 11 158 256–15 077 728 all
  under the 66.6 M rail; wall 37:22, RSS 1 172 820 KB; fork
  continuations second-pass-verified from raw dump lines (p3 fork
  text is ", June 12, 2018," — an earlier from-memory read said
  "2027, 12, 2028" and was WRONG; the raw line wins, again).
- ISC-56: cargo check/test (200: 63 rt + 134 snn + 3 app)/clippy
  -D warnings/no_std all green on the final code; VISION + ROADMAP
  carry the 4B sections; commits local only.

## Verification (Stage 4, session 4C — p3 disambiguation coda)

- Apparatus: the SAME fork binary as s4b (build pinned at 9ca265a +
  the env-gated NEURALOS_DUMP patch — `getenv` in completion.cpp,
  dump = raw top-10 logits pre-sampler); identical run flags to the
  gate (greedy forced: --temp 0 --top-k 0 --top-p 1.0 --min-p 0.0
  --seed 42 -no-cnv -c 512 -n 12). All logs+time -v captures in
  /tmp/opencode/s4c/. Models fetched from HF prism-ml/
  Ternary-Bonsai-4B-gguf: Q2_0 1 074 969 344 B, F16 8 049 911 840 B
  (GGUF v3 magic checked on both).
- Rung (a): p3_q2_0.log step=0 — `7794:14,6527 220:11,5701 6602:
  11,4052 7920:11,2683 …`; greedy emitted " Thursday" as token 1 →
  7794 = " Thursday" TOP-1, margin +3.0826. Generation (mechanical
  sed strip of dump suffixes): " Thursday04/05/2018". p2_q2_0.log
  step=0: 4236 " five" TOP-1 at 13.2278, margin +2.9722 vs s4b
  fork4b/p2.log (4236 at 11.3576, 11 at 10.4280, margin +0.9296).
- Rung (b): p3_f16.log step=0 — `7794:14,6529 220:11,5987 6602:
  11,4116 7920:11,2777 …` — TOP-1, margin +3.0542; Q2_0↔F16 step-0
  top-10 identical token sets in identical order, max |Δ| 0.0286.
  The 30-min guard fired after the step-10 dump (TERM delayed by a
  D-state disk wait, one token from natural finish — 11/12 tokens);
  salvage per plan: sed-strip text " Thursday04/10/2018". Load 3:20,
  RSS peak 3.6 GB, decode disk-bound ~2.5 min/token (mmap streaming
  8 GB/token-pass through 3.5 Gi available — memory gate held, no
  swap-in, no OOM).
- Chat void-check (principal's added gate): p3_chat_q1_0_v.time
  verbose token log — `'<|im_start|>':151644 … '<|im_end|>':151645 …
  '<think>':151667 … '</think>':151668`, embd_inp.size() 15 —
  specials tokenized as control ids, run VALID (not void). Same
  line banked for F16 (p3_chat_f16.time). Q1_0 chat generation
  (sed-strip): "It seems like your message might be a typo or
  incomplete."; step-0 top-10 headed by "It" (2132:13,1701) with
  <|im_end|> (151645) at rank 4 — weekday knowledge absent via chat
  at 1-bit too. F16 chat (p3_chat_f16.log): step-0 top-10 headed by
  "It" (2132:13,6572), NO 7794 present, top-3 order (2132, 785, 40)
  identical to Q1_0's chat run — clarification-land on the base
  itself; the chat framing, not quantization, buries the weekday
  chain there. Guard-salvaged F16 chat text: 'It looks like you've
  written "Monday Tuesday' (9 of 12 tokens). Determinism witness:
  non-verbose vs verbose chat runs' step-0 dumps byte-identical.
- 4C makes NO runtime changes: 4 CI gates re-run green on the
  docs-only diff; commits local per protocol.


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

## Learning (Stage 4, session 4B)

- conjectured: the refactor had changed bonsai_full's arithmetic —
  the 1.7B top-5 came out different from the s3.5-recorded run
  (residual 17 965 859 vs recorded 17 962 389; 5th id 19 vs 20)
  while the gate (incremental path) was byte-identical.
  refuted by: an e1821be worktree running the OLD bonsai_full
  reproduced the NEW numbers exactly — the s3.5 top-5 citation was
  recorded BEFORE C-core's score-scale fix (which changed every
  forward output), and C-core's verification re-ran bonsai_full as
  "OK" without re-pinning the numbers. The record was stale, not the
  code; the incremental-path byte-diff had said so all along (the
  exact-equality test chains forward ≡ incremental ≡ e1821be).
  learned: when a core fix legitimately changes outputs, every
  number downstream of it in the records goes stale at that moment —
  re-pin them in the same session as the fix, or the next session
  pays a false-alarm localization to find out.
  criterion now: a behavior-changing fix re-runs and re-pins every
  evidence number it touches (the fix's Verification block lists
  them), and an apparent regression is first checked against the
  TESTS that pin path-equivalence (they encode the ground truth the
  prose may lag).
- conjectured (second-pass catch, C-pre's lesson repeating): the
  fork's p3 continuation from my first read of the interleaved
  dump log — ", June 12, 2027, 12, 2028,".
  refuted by: stripping the dump suffixes from the raw lines and
  re-deriving — the text is ", June 12, 2018,". I had merged
  top-10 entries into the continuation while skimming.
  learned: interleaved dump logs are a NEW surface for the old
  failure mode — attribution numbers survive a fast read and die on
  a re-read, now twice in this ISA's history.
  criterion now: every quoted continuation is produced by a
  mechanical strip-and-concatenation of the raw log (the sed one-
  liner), never by reading interleaved output by eye.

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
