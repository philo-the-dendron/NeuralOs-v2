# Step-5 Calibration — the lesion/graft positive control — PRE-REGISTRATION (DRAFT v9)

Status: DRAFT v9, 2026-09-05, after reviewer passes 1–8 (51 findings
in total, every cited fact verified at source by the adjudicator, all
adopted; §9 lists them). Awaiting reviewer pass 9 (delta only) and
principal stamp. No arm file is generated and no judge run starts before the
stamp line (§10) carries philo's word and a date. Ledger authority:
ISA.md constraint 1 ("the lesion/graft positive control is a
PREREQUISITE, not an option"); `paper/sections/limitations.tex`
("Calibration — a lesion/graft positive control driven through the
same splice machinery — precedes any future interpretation"). GUARD 1
(append-only ISA) and GUARD 2 (frozen adjudications) intact: nothing
here re-opens session I, H2, or the step-5 verdict. Outputs live only
under `evidence/step5-calibration/` and `models/cal-*.gguf`.

## 1. Question and what each verdict licenses

Can the step-5 readout (frozen five prompts, M2 destination partition,
M3 margin exceedance) separate a STRUCTURED, function-bearing change in
the surgery window from an EQUAL-DOSE random perturbation, when both
go through the same splice machinery?

It does not ask whether adaptation happened (that is the frozen step-5
record). It asks whether the judge could have seen structured content
had it been there. Five verdicts, pre-written (rule in §6):

- **CALIBRATED** — a supermajority of judged files in BOTH arm classes
  separate from their primary nulls (§6: 3/4 of lesions, 2/3 of
  grafts, unanimity when a denominator sits at its floor). Licenses:
  the "uncalibrated instrument" limitation narrows to its scale clause
  (|Δmargin| still has no behavioral magnitude); the step-5 negative
  gains "a readout shown to separate structured change from
  equal-count noise: for grafts, from noise at the same concentration;
  for lesions, from scattered noise only"; step 8's gate question
  becomes answerable.
- **DAMAGE-ONLY** — lesions clear their bar; grafts do not. Header:
  "{Gs} of {Gj} graft files separated vs LOCAL, below the 2/3 bar".
  Let Gscat = graft files SEPARATED vs SCAT whose SCAT family is
  intact (5 files). A graft whose SCAT family was depleted by the void
  protocol (§5) does NOT count toward Gscat, the conservative direction
  and the same convention as an undefined M3 max; the licence names it:
  "{n} graft file(s) carried a degraded SCAT family and are excluded
  from Gscat". The licence is chosen by (Gs, Gscat), every case with
  exactly one home:
  - Gs = 0, Gscat = 0 → "the readout does not see transplanted content
    at all: 0 of {Gj} grafts separated vs LOCAL, and 0 of the {Gj − n}
    with intact SCAT families separated vs SCAT".
  - Gs = 0, Gscat > 0 → "the readout sees concentration, not content:
    {Gscat} of {Gj} grafts separated against scattered noise, 0 of {Gj}
    against noise at the same concentration".
  - Gs > 0 → "the readout saw transplanted content in {Gs} of {Gj}
    grafts, below the 2/3 bar; the limitation stands for content
    because the bar was not cleared; the {Gs} separating file(s) are
    printed verbatim (§6 global print rule) as evidence for the next
    design, not as licence".
  In every case the limitation narrows for damage and stands for
  content; step 8 needs a content-sensitive readout before it opens. If
  the graft leg is INERT (§6 rule 3), the sentence "{Gq} of {Gj} graft
  files produced no flips; content-sensitivity is untested on those and
  rests on the remaining {Gj−Gq}" is APPENDED to the chosen case; it
  replaces nothing, so a separating graft stays on record.
- **UNINFORMATIVE** — lesions below their bar. Header: "{Ls} of {Lj}
  lesion files separated vs SCAT, below the 3/4 bar". Licenses: "the
  readout separated {Ls} of {Lj} cut heads from the same damage
  scattered, below the 3/4 bar; at this dose in this window the
  evidence does not clear the pre-registered bar"; the limitation
  stands as written; the step-5 negative keeps exactly the one claim
  the paper gives it; step 8 is closed until a new instrument exists.
  Any graft that separated is printed verbatim (§6 global print rule)
  and named as "{Gs} of {Gj} grafts separated; not promoted"; the
  graft-INERT sentence above is appended when §6 rule 3 fired.
- **INERT** — three or more judged lesion files are quiet (zero flips
  on all five prompts). Licenses: nothing about the readout. Cutting a
  fifth of one head's q-projection changes nothing at 5 prompts × 12
  greedy tokens; the control is void as to the instrument and a
  stronger perturbation is the named next step.
- **VOID** — a denominator floor (§6) is not met, a kill criterion
  (§7) fired, or the session cap stopped the run. Licenses nothing.
  What ran is reported file by file, unfinished arms named unfinished.

Wording for `limitations.tex` is proposed from the verdict and ruled by
philo; this document does not edit the paper.

## 2. Instrument (unchanged from step 5)

Splice: `harness::splice_and_verify` with `expect_src = Some(base_slice)`
on every write (the codec transparency check; S2 disk round-trip;
scale bytes asserted untouched). Nulls: `harness::dose_matched_null`
(exact changed-cell count and full per-class composition, seeded FY
over src-class cells of the slice it is given). Judge:
`tools/run_prompts.sh` verbatim — frozen five prompts, `-n 12 --temp 0
--seed 42`, SHA256SUMS per outdir; double-run files must satisfy the
script's run1 == run2 byte-identity assert (exit 3 otherwise).

Metrics, as ratified in `evidence/step5-readout/PREREG.md` §5:
**M2 fires on byte-exact continuation strings** — some flip destination
of the arm file appears in none of its null family's continuations.
The frozen basin list B1–B5b and NOVEL are reporting vocabulary only;
they never collapse two distinct strings into one bucket, and nothing
is added to the list. **M3** — the arm file's max|Δmargin| at
knife-edge steps exceeds its null family's max.

Base: `models/Ternary-Bonsai-4B-Q2_0.gguf`, sha
`4e0bf8b737b0431552f8c2c97695ab7c0cb214c94bcdeb4f5f267e67ddf28b8b`
(PREP.md pin), re-verified before `--measure` and before `--generate`.

Window (ExperimentParams, unchanged): `blk.0.attn_q.weight`, rows
0..512 of 4096 (heads 0–3 of 32, 128 rows each), cols 0..512 of 2560.
Banked census of this window (evidence/r4-baselines/loop_run2.log):
+1 × 83,089 · 0 × 95,802 · −1 × 83,253 of 262,144; nonzero 0.6345.

## 3. Arms, null families, N

| Arm | Files | Definition | Judge |
|---|---|---|---|
| LESION-h, h∈{0,1,2,3} | 4 | every trit in window rows [128h, 128h+128) × cols [0,512) set to 0. Diff classes (+1→0), (−1→0) only. | double |
| GRAFT-k, k∈{1,18,35} | 3 | window rows [0,128) × cols [0,512) replaced by the same rows/cols of `blk.k.attn_q.weight` (trained weights, wrong layer). Any of the six diff classes. | double |
| IDENTITY | 1 | base slice spliced onto base through the same pipeline; MUST be byte-≡ base (sha) and MUST show zero flips. Contamination tripwire (step 5 §3). | double |
| SCAT null, 5 per arm file | 35 | `dose_matched_null(base_window, arm_window, seed)` over the full 512×512 window: same count and composition, positions random over the whole window. This is step 5's own null construction. | single |
| LOCAL null, 5 per GRAFT file | 15 | `dose_matched_null(base_block, graft_block, seed)` over the arm's own 128×512 footprint, embedded at that footprint; same count and composition, positions random INSIDE the block. Separates structure from concentration. | single |

**Why LESION has no LOCAL family (adjudicator finding on reviewer B6):**
a lesion zeroes every nonzero cell of its block, so a dose-matched null
drawn inside that block must zero the same count from the same pools,
which is every one of them: the local null IS the lesion, byte for
byte. For a full-block zeroing, structure and concentration are the
same fact, and §1's DAMAGE-ONLY/UNINFORMATIVE wording says
"coherent, localized" for that reason. For a graft the block is
partially changed (~2/3 of cells), so a local shuffle is a real null.

Bands for GRAFT are computed against BOTH families. §6 decides on the
LOCAL band (the deconfounded one), and the SCAT band rides in the
verdict string, e.g. `DAMAGE-ONLY (graft: NULL-CONSISTENT vs LOCAL,
SEPARATED vs SCAT)`, because a local null is by construction more
similar to its arm than a scattered one, so the graft leg faces the
harder test and a DAMAGE-ONLY whose grafts beat the lesions' own null
construction is a different finding from one whose grafts beat
neither (reviewer B8). Bands for LESION are computed against the SCAT
family; a MID family (256×512 half-window, 50% density) was considered
for lesions and rejected on wall-time (20 more runs breach the cap),
not on principle (reviewer G2).

**Escalation: none.** Reviewer B4, verified: both metrics are one-sided
against the arm (M2 needs a destination absent from the family, M3
needs the arm above the family max), so adding nulls can only demote a
band, never promote it. Step 5's escalation added ON replicates; here
the arms are deterministic. The seeds beyond the first five of each
decade are reserved and unused.

Judge runs: 8 double-run files (16) + 50 nulls = 66. At 2–4 min per
file (BURN.md) that is 2.2–4.4 h.

## 4. Dose accounting

Predicted from the banked census before measurement (the prediction
stays for the record), then filled from `evidence/step5-calibration/measure.log` (sha
`bba8d7e0bb1582d35a2ec72c97a28eabbd611e3ae9ba6ed02cb4b8099975ea76`,
pinned in `SHA256SUMS`; produced by `step5_calibration --measure` at
commit e11a53b; every reader check PASS, no kill):

| Arm | predicted | measured | composition (from→to : n) |
|---|---|---|---|
| LESION-0 | ≈ 41,580 | 42,213 | (−1→0 : 21,191) (+1→0 : 21,022) |
| LESION-1 | ≈ 41,580 | 40,577 | (−1→0 : 20,439) (+1→0 : 20,138) |
| LESION-2 | ≈ 41,580 | 41,549 | (−1→0 : 20,616) (+1→0 : 20,933) |
| LESION-3 | ≈ 41,580 | 42,003 | (−1→0 : 21,007) (+1→0 : 20,996) |
| GRAFT-1 | 41,580–44,760 | 43,586 | (−1→0 : 7,479) (−1→+1 : 6,936) (0→−1 : 7,551) (0→+1 : 7,290) (+1→−1 : 6,761) (+1→0 : 7,569) |
| GRAFT-18 | 41,580–44,760 | 43,388 | (−1→0 : 7,887) (−1→+1 : 6,627) (0→−1 : 7,288) (0→+1 : 7,358) (+1→−1 : 6,539) (+1→0 : 7,689) |
| GRAFT-35 | 41,580–44,760 | 43,424 | (−1→0 : 7,559) (−1→+1 : 6,806) (0→−1 : 7,395) (0→+1 : 7,493) (+1→−1 : 6,798) (+1→0 : 7,373) |
| IDENTITY | 0 | 0 | (none) |

Per-head block zero fraction (lesion footprints): h0 0.3559 · h1 0.3808
· h2 0.3660 · h3 0.3591.

Graft provenance, from the log: GRAFT-1 source `blk.1.attn_q.weight`,
data offset 133,467,360, block sha `402c4a1c…`; GRAFT-18 `blk.18`,
offset 589,574,368, sha `bfbf655c…`; GRAFT-35 `blk.35`, offset
1,045,681,376, sha `c84818e8…`. The two offset gaps are EQUAL,
456,107,008 B each = 17 × 26,829,824 B, seventeen layer strides, which
confirms the sources are evenly spaced at 1 / 18 / 35 (18 − 1 = 35 −
18); a wrong-layer typo in the const table would break the equality
even if the block shas did not catch it (reviewer J1). Shas pairwise
distinct.

Heads sum to [−1 83,253 · 0 95,802 · +1 83,089], the banked window
census, exactly. Lesion doses sum to 166,342, the banked nonzero count.
Graft doses against independence: the pre-registered interval
41,580–44,760 was derived from the WINDOW census, rounded (exact
endpoints 41,585.5 and 44,763.75); it GOVERNS the §8 correlation
report and stands as written, pre-data. Head 0's own block census
([21,191, 23,323, 21,022], the graft base) gives the integer-exact
interval 42,213–44,514 (65,536 minus the block's most and least common
class counts), stated alongside; a value falling between the two
intervals is reported under both readings (reviewer J4). Internal
check: the block interval's lower endpoint, 42,213, is exactly
LESION-0's measured dose, because an all-zero source changes precisely
the block's nonzero cells; one number proves both (reviewer J2). All
three measurements sit inside both intervals. With the source
censuses now in the log, the exact independence point per graft is
65,536 − (21,191·n₋₁ + 23,323·n₀ + 21,022·n₊₁)/65,536 = 43,634 /
43,610 / 43,640 against measured 43,586 / 43,388 / 43,424: each graft
sits 0.1–0.5% below its prediction, a small same-sign deficit, well
inside the interval, and no correlation large enough to affect dose
comparability (reviewer H2). LOCAL pools have ≥ 1.46× headroom on every
class (tightest: GRAFT-18, −1 class, 14,514 of 21,191). The graft block
sha is stated by the generator's doc comment to be over the DECODED
block (one byte per trit, content not q2_0 packing); the log prints
the digest only, so that statement is checked against the code at
commit e11a53b, not against the log.

**Dose gap, disclosed (reviewer S1):** step 5's record changed
61,210–80,391 cells; these arms change ≈ 0.6× that. A single head's
block is bounded above by 65,536 cells, so no single-head arm can reach
the step-5 band. Fork taken: keep the head as the unit (a clean
functional structure) and disclose the gap, rather than widen to 192
rows (a head and a half is no longer a unit). Widening to 192 rows was never a
live fork: a 512-row window holds two disjoint 192-row blocks, so Lj
caps at 2 and §6 rule 1 voids the design before any file is judged
(reviewer N4). Consequence for reading (reviewer S8): dose is matched
on changed-cell COUNT only. The lesion's damage is 100% dense in a
quarter of the window against step 5's ~25% spread across all of it,
so 0.6× is a count ratio, not a claim that the perturbation is
weaker, and separation at this count is not automatically the harder
test. A non-separation carries the caveat "at 0.6× count", and INERT
(§1) is the outcome that catches a perturbation too small to register.

**Reader checks that can only fire on a broken reader (reviewer B10;
the earlier ±5% band is withdrawn: under independence any class mix
gives 41,580–44,760 changed graft cells, inside ±5%, so the band could
fire only on real correlation between blk.0 and blk.k, a true fact,
not a reader fault):**
- LESION: the four heads' decoded censuses sum EXACTLY to the banked
  window census (+1 × 83,089 · 0 × 95,802 · −1 × 83,253); each block
  decodes to 65,536 trits; each patched block is all-zero. Per-head
  zero fraction is a measurement, reported, never a gate.
- GRAFT: `blk.k.attn_q.weight` has blk.0's dims and Q2_0 type; its
  block census sums to 65,536; changed > 0 (the read-the-same-tensor
  failure). The measured changed count is reported against the
  independence interval 41,580–44,760 as a RESULT: a value below it is
  measured correlation between the layers and goes to §8, not to the
  kill list.
- IDENTITY: 0 changed cells.
- Graft provenance (reviewer D5; three distinct layers give doses
  within 0.45% of each other, so dose cannot tell blk.17 from blk.18):
  `--measure` records, per graft, the tensor name resolved from the
  GGUF's own tensor list (not the const-table string), its data offset,
  and the sha256 of the decoded 128×512 source block, and asserts the
  three block shas are pairwise distinct.
- Block placement (reviewer D6): the census-sum identity is invariant
  under a permutation of the four blocks, so `--measure` also asserts
  block h covers window rows [128h, 128h+128) exactly. `decode_slice`'s
  own dims/type asserts already enforce the graft-tensor kill.
Every one of these is a kill (§7). All are read back from
`measure.log` (§7), never from a transcription. For the record
(reviewer D8), with its provenance named because the sentence above
forbids transcriptions: on the v2-code run at commit f07b63f, reported
by the builder and reproduced line for line by the adjudicator in a
detached worktree, superseded by `measure.log` before the stamp, the
withdrawn ±5% band would have passed every arm (lesions −2.43% to +1.51%, grafts within 0.6%),
and the four lesion doses sum to 166,342, the banked nonzero count
exactly. B10 stands on the correlation risk and on principle, not on a
near miss.

## 5. Bands (per arm file vs its primary null family)

- **SEPARATED** — M2 AND M3 fire.
- **MIXED** — exactly one fires.
- **NULL-CONSISTENT** — neither fires.
- **QUIET** — the arm file shows zero flips on all five prompts; it is
  NULL-CONSISTENT for §6's L/G counts and counted separately for INERT.

Families are exactly 5 files. A PRIMARY family (SCAT for lesions,
LOCAL for grafts) reduced below 5 by the void protocol voids its ARM
(that arm leaves the judged set), because a smaller family lowers the
M3 max and biases toward the positive verdict (reviewer S2). A
depleted graft SCAT family degrades only the SCAT band in the verdict
string (reviewer S7). M3 convention when a family has no knife-edge
step (reviewer N5), and likewise when the ARM file has none (reviewer
D3): the max is undefined and M3 does NOT fire for that arm. This is
conservative for both arm classes because the §6 tree is monotone in
Ls and Gs (thresholds are ≥, rules 2–3 do not read them), so a
non-firing M3 can only move a verdict down the ladder. The aggregator
prints `m3: family-empty` or `m3: arm-empty` and never panics.
Consequence, disclosed: if a primary family is entirely quiet and the
arm flips, M2 fires trivially and M3 cannot, so that arm bands at most
MIXED (also in §8).

## 6. Verdict rule (mechanical; the aggregator's unit test pins this table)

Let Lj = lesion files judged (not voided), Ls = of those SEPARATED,
Lq = of those QUIET; Gj, Gs, Gq likewise for grafts (bands vs LOCAL).

1. If Lj < 3 or Gj < 2 → **VOID** (denominator floor).
2. Else if Lq ≥ 3 → **INERT**.
3. Else if Gq ≥ 2 → graft leg INERT (reviewer B7): verdict is
   **DAMAGE-ONLY** if Ls/Lj ≥ 3/4 else **UNINFORMATIVE**, with the
   quantified §1 graft-INERT sentence appended in either case. Ordering
   note (reviewer
   D1): since Gs ≤ Gj − Gq, Gq ≥ 2 forces Gs/Gj ≤ 1/3, so this rule
   never changes a label relative to rules 4–6; its whole effect is the
   licence text.
4. Else if Ls/Lj ≥ 3/4 and Gs/Gj ≥ 2/3 → **CALIBRATED**.
5. Else if Ls/Lj ≥ 3/4 → **DAMAGE-ONLY**.
6. Else → **UNINFORMATIVE**.

Global print rule (reviewer D2): any graft file SEPARATED while the
verdict is not CALIBRATED is printed verbatim under every rule, never
promoted. The verdict string always appends the graft SCAT band (§3).

Fractions, not counts, so a voided arm changes the denominator and not
the bar. With Lj = 3 the lesion bar is 3/3; with Gj = 2 the graft bar
is 2/2. The rule is applied once, to the single run; there is no
second reading.

**Amendment of constraint 1's wording, named (reviewer S5):** ISA
constraint 1 says "if the graft lands inside the null family, the
readout is uninformative by construction" — graft-decides. This rule
makes lesions decisive and treats graft failure as DAMAGE-ONLY, a
partial result, because a readout that sees coherent damage but not
content is a different instrument from one that sees nothing, and the
paper's next step differs between the two. The bar is not lowered: a
graft failure still forbids step 8 from opening.

## 7. Protocol, kill criteria, cap

Order:
1. Reviewer delta pass on the current version. Fixes land above the
   stamp; the stamp is dated after them.
2. Code committed on `work/step5-calibration` BEFORE the stamp: the
   generator (`step5_calibration.rs`, modes `--measure` and
   `--generate`), the aggregator mode `step5_aggregate --poscontrol
   <root>` (named to be unmistakable next to the existing `--calibrate`,
   which is the banked-log parity gate and NOT this control), and a
   unit test that pins §6 by EXHAUSTIVE enumeration of every tuple
   (Lj ≤ 4, Ls + Lq ≤ Lj, Gj ≤ 3, Gs + Gq ≤ Gj), asserting exactly one
   verdict per tuple and hand-written per-verdict counts (reviewer
   S11; totality and exclusivity tested, not the author's examples).
   Implementation note (reviewer N6): `ExperimentParams.tensor` is
   `&'static str`; the graft tensor names come from a const table,
   not a runtime format (constraint 4: the literal is authored before
   ratification).
3. `--measure` (writes nothing to `models/`; tees its stdout to
   `evidence/step5-calibration/measure.log`, sha-pinned in
   `SHA256SUMS`): §4 filled FROM that log, reader checks applied
   (reviewer B9). `--generate` likewise tees to `generate.log`, so a
   splice panic is on record.
4. This file, `seeds.txt`, and the code committed; then philo's stamp.
5. `--generate`: files named exactly
   `models/cal-lesion-h{h}.gguf`, `models/cal-graft-k{k}.gguf`,
   `models/cal-identity.gguf`,
   `models/cal-{arm}-scat-s{seed}.gguf`,
   `models/cal-graft-k{k}-local-s{seed}.gguf`;
   every write through `splice_and_verify(expect_src = Some(base))`
   and `assert_unbanked`; `arms.txt` written beside this file listing
   every arm and its families. No name ends in `-control.gguf`,
   `-invivo.gguf` or `-loop.gguf` (banked suffixes refused by the
   guard).
6. Judge: double-run files with `--double`, nulls single; outdirs
   `evidence/step5-calibration/burn/<file-stem>/`.
7. `step5_aggregate --poscontrol evidence/step5-calibration/burn`:
   bands, §6 verdict, `scat:` lines for grafts, IDENTITY result. Arm
   list from `arms.txt`, never hardcoded.
8. Record: ISA.md verification entry (append), RESEARCH_LOG, ROADMAP
   item 5; PR on Gitea; merge on philo's word after CI green.

Seeds: `evidence/step5-calibration/seeds.txt`, committed before any
generation. One decade per family, first five used, last five reserved
and unused: SCAT LESION-0 301–310 · LESION-1 311–320 · LESION-2
321–330 · LESION-3 331–340 · GRAFT-1 341–350 · GRAFT-18 351–360 ·
GRAFT-35 361–370 · LOCAL GRAFT-1 371–380 · GRAFT-18 381–390 · GRAFT-35
391–400. Disjoint from every seed step 5 or session I used (1–10,
201–250). `harness::decade_for` cannot serve these (it asserts r ∈
0..=4 and the 201–250 range); the generator carries its own selector
with its own range assert (reviewer N1).

**Kill criteria — any one stops the run; fix, restart from
`--measure`, re-date the stamp:**
- base sha ≠ the §2 pin, at either check;
- any panic in `splice_and_verify` (codec, scale byte, expect_src);
- any §4 reader check failing — §4 is the one list of them; this
  bullet does not restate it — read from `measure.log`;
- IDENTITY output not byte-≡ base, or any flip on IDENTITY;
- run1 ≠ run2 on any double-run file (script exit 3);
- judge binary missing (exit 2);
- any PRIMARY null family below 5 after the void protocol (voids the
  arm; if that drops Lj < 3 or Gj < 2, VOID); a depleted graft SCAT
  family degrades only its SCAT band (§5).

Void protocol (step 5 §6 verbatim): exit≠0 or empty dump = void,
re-run once; twice → the file is excluded and the family/arm rule
above applies.

Session cap: one burn window, ≤ 5 h judge wall-time (worst case 4.4 h
+ voids). Past the cap: stop, verdict VOID, report what ran.

## 8. Limitations named before the data

- The window covers 512 of a head's 2560 input dims; LESION-h cuts a
  fifth of one head's q-projection, not the head.
- Arm dose is ≈ 0.6× the step-5 record (§4), by construction of a
  single-head unit.
- For lesions, structure and concentration are one fact (§3); only the
  graft arm separates them.
- A lesion is coherent loss of function; a graft is functional content
  in the wrong place. Neither is adaptation. This control bounds what
  the readout can see; it does not manufacture adaptation.
- Five nulls per family, half of step 5's ten, no escalation (it could
  not change a verdict).
- The three GRAFT files share the footprint rows [0,128), so their
  failures are correlated; Gj = 3 is less independent than Lj = 4
  (reviewer N7).
- Cap margin is thin: 4.4 h worst case against a 5 h cap leaves room
  for roughly 9–18 voided files before VOID by cap.
- A MID null family for lesions (half-window, 50% density) would have
  interpolated the concentration axis; rejected on wall-time (§3).
- If a primary family is entirely quiet, its arm can reach at most
  MIXED (§5 M3 convention); the most favourable data pattern this
  control can produce is therefore capped by design, not read up.
- One window, one layer, one model, five prompts, greedy decoding, as
  in step 5. Nothing here generalizes past that scope.

## 9. Changes by version (all reviewer passes, 2026-09-05)

B1 M2 wording fixed to string-level · B2 INERT added, UNINFORMATIVE
scoped to files that flipped · B3 §6 rewritten as fractions with
floors, VOID and unfinished named · B4 escalation deleted with the
proof · B5 kill list, double-run assert, `--poscontrol` name, outcome
literal before stamp · B6 LOCAL family for grafts, lesion case argued ·
S1 doses predicted, dead range check replaced by a ±5% sanity check,
0.6× gap disclosed · S2 family floor · S3 §1 wording matched to §6 ·
S4 expect_src, filenames, commit-before-generate pinned · S5
amendment named · S6 IDENTITY arm · N1 selector · N2 filename template
avoids banked suffixes · N3 (clean) kept.

**v3 (reviewer pass 2, 2026-09-05):** B7 Gq branch · B8 graft bands
vs both families, SCAT band in the verdict string · B9 measure.log /
generate.log sha-pinned · B10 ±5% withdrawn, structural reader checks
· G1 CALIBRATED wording · G2 MID named as rejected · S7 floor scoped
to primary family · S8 count-vs-density honesty · S10 supermajority
wording · S11 exhaustive enumeration test · N4 192-row fork shown dead
by the floor · N5 M3 empty-family convention · N6 const tensor table ·
N7 graft footprint correlation, cap margin. N8 (clean) kept.

**v4 (reviewer pass 3, 2026-09-05):** D4 §7 floor wording matched to
§5 (an amendment applied in one of two records; the sweep rule) · D2
quantified graft-INERT substitution, global anomaly print · D3 arm-side
empty M3, monotonicity argument, quiet-family cap disclosed · D5 graft
provenance (resolved tensor name, offset, block sha, pairwise distinct)
· D6 block-row assert · D7 stale "pass 2 on v2" line · D8 the withdrawn
band would have passed, said plainly. D1 (rule-3 ordering
verdict-neutral) recorded in §6.

**v5 (reviewer pass 4, 2026-09-05):** E1 §1 DAMAGE-ONLY and
UNINFORMATIVE quantified and given a Gs > 0 clause with a pointer to
the global print rule (two absolutes over threshold rules) · E2 §7 kill
bullet points at §4 instead of restating a partial list · E3 quoted
numbers carry their provenance and supersession · §9 heading · graft
dims/type rule reduced from four records to two.

**v6 (reviewer pass 5, 2026-09-05):** F1 UNINFORMATIVE licence
quantified (the header was, the paper-bound sentence was not) · F2
DAMAGE-ONLY licence keyed on (Gs, Gscat) with one home per case, the
"weaker of the two" import removed · §7 graft dims/type bullet removed,
one record left in §4.

**v7 (reviewer pass 6, 2026-09-05):** G1 Gscat defined over intact
SCAT families only, depleted ones excluded and named (a quantity used
in §1 whose qualifier lived in §5) · G2 graft-INERT sentence appends,
never replaces.

**v8 (reviewer pass 7, 2026-09-05):** H1 tightest LOCAL pool named
correctly (21,191 over 14,514, 1.46×) · H2 "no correlation measured"
replaced by the exact per-graft independence points and the measured
0.1–0.5% same-sign deficit; the block-census interval named, the
pre-registered one kept · H3 the (0, 0) licence quantified over intact
SCAT families · table column split into dose table, zero fractions,
provenance · the decoded-block-sha statement attributed to the code,
not the log.

**v9 (reviewer pass 8, 2026-09-05):** J1 offset gaps are 17 strides,
equal, which is itself a provenance check (I had copied "one layer
stride" from the builder's message unchecked) · J2 block interval
integer-exact 42,213–44,514, lower endpoint = LESION-0's dose · J3
window interval marked rounded with exact endpoints · J4 the
pre-registered interval governs the correlation report, the block
interval is stated alongside.

## 10. Stamp

_unstamped_
