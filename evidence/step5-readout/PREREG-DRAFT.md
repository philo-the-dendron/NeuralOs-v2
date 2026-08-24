# Step-5 Readout Benchmark — PRE-REGISTRATION (DRAFT, awaiting principal stamp)

Status: SUPERSEDED — ratified as PREREG.md (2026-08-23) with all
review rulings folded (OFF-arm redesign, Rider A/B, G2 basin freeze,
G4 limitation, domain-corrected arm). Kept for provenance; PREREG.md
is the authority.
Ledger authority: ISA.md "Decision (same session — sequencing + step-5
success criteria, 2026-08-23)" + the M1 step-5 entry. GUARD 2 intact:
this is a NEW pre-registration; the session-I/H2 adjudications stay frozen.

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
~6–8 h/run). The embeddings-only capture is a named follow-up PARKED
for step 8 (it is an instrument change — different drive statistics —
not a refactor-pin; it cannot reproduce H2 pins bit-exact).

## 3. Arms and N

| Arm | N | Definition |
|---|---|---|
| ON | 3 | in-vivo STDP-on export per replicate window (§4) |
| OFF | 3 | paired: same window, `set_plasticity_enabled(false)` |
| NULL | 10× per ON replicate (30 files) | dose-matched shuffled-drift of THAT replicate's terminal diff (`shuffled_copy`, exact cell count + composition asserted) |
| CLAMP-RELAXED | 1 | ON window-0 with drive ceiling ×2 — REPORT-ONLY covariate arm |
| FREE (dose-response) | 3 | H2b checkpoints ck400/800/1200 judged — report-only, zero new training |

Judge protocol: frozen five prompts (p0–p4, session-h2 pins);
single-run per NULL file (family determinism protocol), DOUBLE-run for
every ON/OFF/CLAMP file (run1==run2 byte-identical asserted).

## 4. Replicate axis (the named fork — default pre-registered)

In-vivo runs are deterministic given (model, corpus); replicates must
vary something real. Default: **corpus window offset** — replicate
r ∈ {0,1,2} drives on pinned-corpus tokens [1000·r, 1000·r+2000)
(4,411 tokens total, sha-pinned corpus). ON_r pairs with OFF_r on the
same window. Seeds for NULL shuffles: 201–210 (r0), 211–220 (r1),
221–230 (r2) — pre-generated, committed.

## 5. Metrics, calibration gate, bands

Metrics (all mechanical, from judge dumps vs base continuations):

- **M1 flip-profile** — per-prompt flip indicator (context; UNINFORMATIVE
  alone by session-I record: null p3 envelope [0.44, 0.97]).
- **M2 destination partition** — each flip's exact continuation string
  classified: {recorded basin families} ∪ {novel}; metric = ON
  destination set vs its NULL family's destination set.
- **M3 margin exceedance** — max |Δmargin| at knife-edge steps; metric
  = ON value vs its NULL family's max (session-I conjunct (a) shape,
  now dose-matched to the ON arm's OWN diff).

**Calibration gate (runs before any ON/OFF arm is trusted):**

- Loud set: d1 (basin-identical), d3 (coherent-chain), H2 terminal.
- Quiet set: d4, d7, loop export (0/60 record).
- Gate: M2+M3 separate loud from quiet with zero overlap. FAIL →
  instrument uninformative by construction → stop, fix, re-gate.
  (Discharges the ISA's own warning; the session-I readout's known
  weakness is flip-rate-only.)

**Discrimination bands (per ON replicate, vs paired OFF + own NULLs):**

- **SEPARATED** — OFF inside the NULL envelope AND (M2: ON destination
  set ⊄ its NULL family's) OR (M3: ON max|Δmargin| > its NULL max).
- **MIXED** — exactly one of M2/M3 fires with OFF inside envelope.
- **NULL-CONSISTENT** — neither fires, or OFF leaves the envelope.

**Escalation (pre-authorized):** any replicate MIXED → extend ON to
n=5 (windows 3,4 — tokens [3000,4400) wrap + [0,2000) overlap noted;
seeds 231–240). One escalation, then verdict on the full set.

## 6. Void protocol (inherited verbatim from session I)

Void = mechanical evidence only (exit ≠ 0 / empty dump; cause recorded
as SUSPECTED, hypothesis not fact). Re-run once. Voids twice → excluded,
denominator adjusted and noted. No discretionary calls with the verdict
visible.

## 7. Clamp covariate (the H2 caveat made testable)

Clamp-fraction logged per run as a first-class covariate. Any arm >50%
clamped is FLAGGED in every table (H2 ran 69.477%). The CLAMP-RELAXED
arm tests "the first test was clamp-starved" — report-only unless its
clamp-fraction <50% AND it separates, in which case it is recorded as
the named follow-up design for step 8, still not adjudicating Tier 3.

## 8. Evidence and rebuild

`evidence/step5-readout/` — SHA256SUMS over every judge log; README
carries full rebuild commands (null generation → judge chains →
aggregator). Aggregator (Rust, extends judge.rs) emits the verdict
table mechanically from the bands in §5; no hand-counted verdicts.

## 9. Falsifiers

- This file superseded only by verbatim-stamped PREREG.md.
- Calibration gate re-runnable from banked d1/d3/d4/d7/loop/H2 files.
- run1==run2 byte-identity asserted per ON/OFF/CLAMP file.
- Aggregator output asserted against this document's bands; any
  mismatch = instrument bug, runs void.
