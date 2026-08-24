# Step-5 Readout Benchmark — PRE-REGISTRATION (RATIFIED)

Status: RATIFIED 2026-08-23 (principal stamp, this session's review).
Supersedes PREREG-DRAFT.md verbatim-plus-amendments; every amendment
is a recorded ruling in ISA.md (Decisions 2026-08-23). No arm may
deviate from this document; deviations void the affected arm.
Ledger authority: ISA "Decision (same session — sequencing + step-5
success criteria)" + "Finding + Decision (step-5 prep — the
drive-domain measurement)". GUARD 2 intact: session-I/H2
adjudications stay frozen; this is a NEW pre-registration.

## 1. Question and success criterion (Tier 3, ratified)

Does plasticity-ON adaptation separate from BOTH controls — paired
plasticity-OFF and dose-matched shuffled-drift null — on a readout the
calibration proves can tell loud from quiet?

- **Tier 3 demonstrated** ⟺ ≥2/3 ON replicates SEPARATED (bands in §5).
- **Rests evidenced** ⟺ ≤1/3 SEPARATED, no MIXED-only escalation rescue.
- **Clean null** ⟺ 0/3, all replicates null-consistent.
Delta-zero is publishable; success is validity, not sign.

## 2. Instrument ruling (ratified 2026-08-23)

Arms run UNOPTIMIZED on the H2-comparable drive (full-forward capture,
~6–8 h/run; `hybrid_invivo` plain mode). The embeddings-only capture is
a named follow-up PARKED for step 8 (an instrument change — different
drive statistics; cannot reproduce H2 pins bit-exact). The ON arms run
the drive **as H2 ran it** — milli-domain scaling, sign-dominant at the
rail — because comparability with the adjudicated record is the
benchmark's point; the measured distribution and its cause are recorded
in ISA (2026-08-23 finding) and carried as the clamp covariate (§7).

## 3. Arms and N

| Arm | N | Definition |
|---|---|---|
| ON | 3 | in-vivo STDP-on export per replicate window (§4) |
| OFF | 1 driven (r0) + 2 identity (r1/r2) | r0: FULL driven run with `set_plasticity_enabled(false)`; its export MUST assert byte-≡ base (sha) — the end-to-end plasticity-toggle proof (the session-F lesson: integration bugs hide in unexercised seams). r1/r2: control-mode identity surgery (ISC-68 class) + judge double-run — contamination tripwires. |
| NULL | 10× per ON replicate (30 files) | dose-matched shuffled-drift of THAT replicate's terminal diff (exact cell count + composition asserted); seeds from `null_seeds.txt` (201–230), never minted at run time |
| DOMAIN-CORRECTED | 1 | ON window-0 mechanics with the gain applied in norm units (`raw = v_milli/1000 × k`) — the sH registration's stated intent, measured 2.74% clamped / RMS 450.0 μA. REPORT-ONLY covariate arm; records the clamp-starvation answer behaviorally and seeds step-8's design. Never adjudicates Tier 3. |
| FREE (dose-response) | 3 | H2b checkpoints ck400/800/1200 judged — report-only, zero new training. Files sha-pinned at prep (ISA 2026-08-23); counts re-verified via surgery decode before judging. |

Judge protocol: frozen five prompts (p0–p4, session-h2 pins);
single-run per NULL file (family determinism protocol), DOUBLE-run for
every ON/OFF/DOMAIN file (run1==run2 byte-identical asserted).

## 4. Replicate axis (the named fork — default pre-registered)

In-vivo runs are deterministic given (model, corpus window); replicates
vary the corpus window. Replicate r ∈ {0,1,2} drives on pinned-corpus
tokens [1000·r, 1000·r+2000) (4,411 tokens total, sha-pinned corpus).
**LIMITATION (named):** windows overlap 50% pairwise on a 4,411-token
corpus — replicates share adaptation statistics; independence is
limited by corpus length. Kept for H2-length dose comparability;
recorded, not hidden. ON_r pairs with OFF_r on the same window.

**k is procedure-pinned, not constant-pinned** (Rider A): each window's
k is derived by H2's frozen method (corpus RMS → 450 μA target) from
its own tokens, logged as a pin from its run.log. Probe-derived
expectations (re-derived per run, never transcribed): r0 10060.46 ·
r1 10101.90 · r2 10007.65. Init-400 = each window's own first 400
steps. Seeds for NULL shuffles: `null_seeds.txt` (201–230;
escalation 231–240), pre-generated, committed.

## 5. Metrics, calibration gate, bands

Metrics (all mechanical, from judge logs vs base continuations;
destination = the log line minus its frozen prompt prefix, byte-exact):

- **M1 flip-profile** — per-prompt flip indicator (context; UNINFORMATIVE
  alone by session-I record: null p3 envelope [0.44, 0.97]).
- **M2 destination partition** — each flipped file's exact continuation
  string classified against the FROZEN basin list (below). Metric = ON
  destination set vs its NULL family's destination set.
- **M3 margin exceedance** — max |Δmargin| at knife-edge steps; metric
  = ON value vs its NULL family's max (session-I conjunct (a) shape,
  dose-matched to the ON arm's OWN diff).

**THE FROZEN BASIN LIST (G2 — enumerated from banked logs, byte-exact
whole-file continuations = file text minus the frozen prompt prefix;
`\n` NOTATION: `\\n` = literal backslash-n (a Qwen text artifact),
`\n` = U+000A. The list is a frozen set of observed strings — no
semantic/prefix matching, ever; every string sha-covered via its
source log's SHA256SUMS):**

| id | prompt | destination (continuation, byte-exact) | first-class members |
|---|---|---|---|
| B1 | p3 | ` Thursday ` + `\\n`×10 + `\n\n` | H2; dose d1,d2,d3,d5,d6,d8,d9,d10 p3 |
| B2a | p2 | ` five six seven eight nine ten eleven twelve fifteen seventeen seventeen eighteen\n\n` | H2 |
| B2b | p2 | ` five six seven eight nine ten eleven twelve fifteen seventeen twenty one\n\n` | d3 |
| B3 | p2 | ` five six seven eight nine ten\n\nWhat is the sum of\n\n` | d4, d6, d8 |
| B4 | p4 | ` Paris. The capital of the United States is Washington, D\n\n` | d1, d4, d5, d6 p4 |
| B5a | p3 | ` Thursday04:00 PM\n\nThe following is a\n\n` | f1, f3 |
| B5b | p3 | ` Thursday04:00 PM\n10:0\n\n` | f2 |

Base (unflipped) destinations — session-f-judge, byte-exact: p0
` 8 9 10 11 1\n\n` · p1 ` 14 15 16 17\n\n` · p2 ` five six seven
eight nine ten eleven twelve thirteen fifteen fifteen seventeen\n\n`
· p3 ` Thursday04/05/2018 \n\n` (trailing space) · p4 ` Paris. The
capital of Japan is Tokyo. The capital of\n\n`. All other flipped
strings observed in the banked family (e.g. f1's ` five six\n1. 2.
3. \n\n`) are NOVEL by definition.

**Calibration gate (AMENDED 2026-08-23, mechanical-parity design —
the ratified loud/quiet sets were data-checked against the banked
family first: d4 is NOT quiet (flips p2→B3 AND p4→B4, same shape as
d1 — no destination metric separates them), and H2-as-loud is
incoherent with session-I's adjudication, which ruled H2
null-family-indistinguishable; a gate demanding H2 separate would
self-kill the benchmark on the banked record before any arm runs).
The gate validates the INSTRUMENT, not the experiment's outcome:**

- Loud probes: d1, d3 — M2 must classify their destinations into
  B1/B2b/B4 byte-exactly (d1: p3→B1, p4→B4; d3: p2→B2b, p3→B1);
  M3 must reproduce the banked Δmargin values from the sha-verified
  logs.
- Quiet probes: d7, the loop export — zero flips detected.
- H2 is NOT a calibration probe; it is the adjudicated case this
  benchmark re-tests under the §5 bands.
- Gate: all mechanical-parity checks exact. Any mismatch →
  instrument bug → stop, fix, re-gate.

**Discrimination bands (per ON replicate, vs paired OFF + own NULLs):**

- **SEPARATED** — M2 AND M3 both fire: (M2: some ON flip-destination
  byte-exact string appears nowhere in its NULL family's) AND (M3: ON
  max|Δmargin| > its NULL family's max). [Amendment 2026-08-23: the
  former "OFF inside the NULL envelope" precondition is DEAD-LETTERED
  — OFF contamination is caught by the byte-≡ asserts of §3, earlier
  and cheaper than an envelope test OFF's identity exports could
  never honestly leave. The bands require M2∧M3 for SEPARATED — the
  three-band structure demands it (a disjunct SEPARATED would make
  MIXED a subset of SEPARATED, a contradiction); the escalation
  ladder reads MIXED → n=5.]
- **MIXED** — exactly one of M2/M3 fires.
- **NULL-CONSISTENT** — neither fires.

**Escalation (pre-authorized):** any replicate MIXED → extend ON to
n=5 (windows 3,4 — tokens [3000,4400) wrap + [0,2000) overlap noted;
seeds 231–240). One escalation, then verdict on the full set.

## 6. Void protocol (inherited verbatim from session I)

Void = mechanical evidence only (exit ≠ 0 / empty dump; cause recorded
as SUSPECTED, hypothesis not fact). Re-run once. Voids twice →
excluded, denominator adjusted and noted. No discretionary calls with
the verdict visible.

## 7. Clamp covariate (the H2 caveat, now measured)

The 2026-08-23 probe (ISA finding) measured H2's drive: pre-clamp p50
≈ 3.1×10⁵ μA (≈1000× the 450 μA target — milli/norm-unit domain
mismatch in the scaling application); clamped fraction 69.477%
UNCHANGED at every ceiling the i16 input path can carry. Therefore:

- Clamp-fraction is logged per arm as a first-class covariate; any arm
  >50% clamped is FLAGGED in every table (the ON arms WILL carry this
  flag — it is the recorded regime, not a defect of this benchmark).
- The former CLAMP-RELAXED (ceiling ×2) arm is REPLACED by the
  DOMAIN-CORRECTED arm (§3): its <50% condition is pre-met by
  measurement (2.74%); its readout is behavioral (does
  amplitude-graded drive change the adaptation's readout signature?),
  recorded as the named step-8 design if it separates.

## 8. Evidence and rebuild

`evidence/step5-readout/` — SHA256SUMS over every judge log; README
carries full rebuild commands (null generation → judge chains →
aggregator). Aggregator (`step5_aggregate`, extends judge.rs) emits
the verdict table mechanically from the bands in §5; no hand-counted
verdicts. Prep-phase artifacts already banked: `clamp_probe.log`
(sha 61e04bbb…), `null_seeds.txt`, this file, `PREP.md`.

## 9. Falsifiers

- This file superseded only by a new stamped pre-registration.
- Calibration gate re-runnable from banked d1/d3/d4/d7/loop/H2 files.
- run1==run2 byte-identity asserted per ON/OFF/DOMAIN file; OFF-r0
  export byte-≡ base (sha) asserted in-pipeline.
- Probe pins: every H2 banked run stat reproduced by
  `step5_clamp_probe` before its distribution is believed.
- Aggregator output asserted against this document's bands; any
  mismatch = instrument bug, runs void.
