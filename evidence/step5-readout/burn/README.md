# Step-5 readout benchmark — the burn family (RAW EVIDENCE)

Authority: `../PREREG.md` (RATIFIED 2026-08-23) + its escalation
amendment (ruled 2026-08-26, before any escalation arm ran).
Orchestration: `../BURN.md`. Prep pins: `../PREP.md`.

**STATUS: raw evidence, banked. The adjudication is OPEN.**
This README records what RAN and what the mechanical aggregator
PRINTED. It rules nothing. The verdict entry — the reading of the
n=5 result against PREREG §1's bands — is a separate ISA decision
and does not exist yet. Nothing below should be cited as a finding.

## What ran

| Arm | N | Dirs |
|---|---|---|
| ON (in-vivo, STDP on) | 5 | `on-r0` … `on-r4` |
| OFF driven (toggle proof) | 1 | `off-r0` |
| OFF identity (contamination tripwires) | 2 | `identity-r1`, `identity-r2` |
| NULL (dose-matched shuffled-drift ×10 per ON) | 50 | `null-r{0..4}-s2{01..50}` |
| DOMAIN-CORRECTED (report-only covariate) | 1 | `domain` |
| FREE (H2b dose-response, report-only) | 3 | `free-ck400`, `free-ck800`, `free-ck1200` |

63 arm directories · 783 files · 4.8 MB. Judge protocol per PREREG §3:
frozen five prompts (p0–p4, session-h2 pins); single-run per NULL,
double-run (run1 == run2 byte-identical asserted) for every
ON/OFF/DOMAIN file.

Runner console output: `../burn-chain.console.log`
(sha256 `e15bb70814d3b48faedd1785c870580bd19a018a802efd4f2baaf6f3e99fb4cf`),
`../burn-escalation.console.log`
(sha256 `4658af0bbe8e2c12949f4fd605b333d1dec63932f70f30faa3945ecab817c3f0`).
Per-leg logs in `logs/`.

## Calibration gate

`../calibrate.log` — **GATE PASS**, 8/8 (loud cases flagged loud,
quiet cases flagged quiet) against banked session-I/loop legs.
PREREG §1 conditions the benchmark on a readout the calibration
proves can tell loud from quiet; that condition is met.

## Timeline

- Main chain (n=3): `2026-08-24T19:55:42Z` → `2026-08-26T06:11:48Z`
  (~34.3 h) — legs pre · off · rep0 · rep1 · rep2 · domain · free · verdict.
- Escalation (n=5, PREREG §5, one pre-authorized): rep3, then rep4
  `2026-08-26T18:49:14Z` → verdict `2026-08-27T05:51:54Z` (~11.0 h).

Two halts interrupted the window; both are recorded below.

## Mechanical aggregator output (verbatim, `logs/leg-verdict.log`)

n=3 pass, 2026-08-26:

```
  on-r0: 2 flips · M3 Some(4.194099999999999) · nulls 10 · M3max 4.2565 → Mixed
  on-r1: 1 flips · M3 Some(4.2509) · nulls 10 · M3max 4.2290 → Mixed
  on-r2: 2 flips · M3 Some(4.197399999999998) · nulls 10 · M3max 4.2523 → NullConsistent

verdict: 0/3 SEPARATED · 2 MIXED → MIXED present — escalation ladder §5 (one, pre-authorized)
```

n=5 pass, 2026-08-27:

```
  on-r0: 2 flips · M3 Some(4.194099999999999) · nulls 10 · M3max 4.2565 → Mixed
  on-r1: 1 flips · M3 Some(4.2509) · nulls 10 · M3max 4.2290 → Mixed
  on-r2: 2 flips · M3 Some(4.197399999999998) · nulls 10 · M3max 4.2523 → NullConsistent
  on-r3: 1 flips · M3 Some(4.1372) · nulls 10 · M3max 4.2202 → NullConsistent
  on-r4: 2 flips · M3 Some(4.224199999999998) · nulls 10 · M3max 4.2209 → Separated

verdict: 1/5 SEPARATED · 2 MIXED → MIXED present — escalation ladder §5 (one, pre-authorized)
```

**Read the trailing clause with care.** The aggregator's fallback
branch (`step5_aggregate.rs`, the `mixed > 0` arm) prints the
escalation prompt whenever any MIXED remains. PREREG §5 authorises
**one** escalation — spent on the n=5 pass. The printed line is a
stale prompt from a threshold rule written for n=3
(`step5_aggregate.rs:18`: "≥2/3 SEPARATED = demonstrated"), not a
ruling that the ladder fires again. The 2026-08-26 amendment held
the threshold at the ratified literal `s >= 2`, decided after the
n=3 verdict and before any escalation arm ran.

## Named limitations carried from PREREG (not discovered here)

- **Window overlap (§4).** ON replicates drive on pinned-corpus
  tokens `[1000·r, 1000·r+2000)` of a 4,411-token corpus. r0–r2
  overlap 50% pairwise; r4's `[0,1589)` wrap overlaps r0's `[0,2000)`
  by **79%** — the heaviest pairing in the design, disclosed at its
  strongest before the burn. The replicates are not independent
  draws; independence is bounded by corpus length.
- **The {r0,r4} caveat (§5 amendment).** Escalation-majority verdicts
  resting solely on the {r0,r4} pair must carry that caveat visibly
  in the close-out.
- **Sign-dominant drive (§2).** Arms ran the H2-comparable drive by
  ruling, for comparability with the adjudicated record; the measured
  clamp distribution is the covariate that `domain/` exists to record.

## The two halts (both amendment residue; no run was at risk)

1. **rep3** — fixed at `7b64c6f`. The original build's
   `231..=240 must not enter a main family` refusal survived the
   escalation edit and contradicted the 2026-08-26 ruling that
   231–240 *are* r3's decade. Deterministic panic at
   `step5_nulls.rs:110`, **after** ON-r3 completed and banked
   (export sha `ce42f09d…`). Decade selection + the full seed-map
   contract extracted to `harness::decade_for`, test-pinned for all
   five replicates plus violation cases.
2. **rep4** — fixed at `bb2e0e5` ("fence-catch #3"). The banked-evidence
   guard's `*r4-*` glob was written for the R4 *remediation* family
   (`evidence/r4-baselines`, `r4-closeout`); the escalation then minted
   replicate-4, whose judge dirs (`null-r4-s24x`, `on-r4`) matched.
   Deterministic refusal at the first r4 judge, **after** ON-r4
   (sha `837d0fe2…`) and all 10 nulls were banked. Globs re-anchored at
   the `evidence/` root; both directions self-checked post-fix.

Both fired loudly at cheap steps, downstream of the expensive training
runs. Neither altered an arm; both were recovered per `../BURN.md` §3
(tail resume — never re-dispatch a banked ON run).

## Integrity

Per-directory `SHA256SUMS` (63 files, 710 hashed entries + this
family's `logs/`). Full verification:

```bash
cd evidence/step5-readout/burn
LC_ALL=C; for d in */; do (cd "$d" && sha256sum -c SHA256SUMS); done
```

Last verified 2026-08-27: **62/62 families clean, 0 mismatches**
(`logs/SHA256SUMS` added at bank time, 11/11 OK).

## Rebuild

Full chain and per-leg commands: `../BURN.md`. Model weights are
gitignored by design (`models/`, 98 GB) and reconstructible from the
provenance table in `../../INDEX.md` § Models manifest. Judge fork
pinned at `9ca265a` (`bash tools/build_fork.sh`).
