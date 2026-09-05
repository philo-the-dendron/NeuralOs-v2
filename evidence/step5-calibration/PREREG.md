# Step-5 Calibration — the lesion/graft positive control — PRE-REGISTRATION (DRAFT v29)

Status: DRAFT v29, 2026-09-05 (all dates in this document are local, America/Toronto). Review history, pass by pass and
finding by finding, lives in §9 and nowhere else (reviewer T1: a count
in this header went stale by eight passes). In short: in-family
reviewer passes plus one independent recount, two cross-family lanes
(Meta and MiniMax-family; a third lane on the §6 rule was refused or
silent on three lineages and is covered in-family by the exhaustive
tuple-space test and the read of `judge::poscontrol_verdict` against
the §6 text, rule for rule — the recount covers the knife-edge
measurement, not the rule), every cited
fact verified at source by the adjudicator, every finding adopted.
Principal rulings: 2026-09-05, base model — "yes we use the bonzai 4b
ternary because the other Q1 have not good answer"; 2026-09-05, null
count — the adjudicator put two options, "1. Ten nulls per primary
family, recommended … 2. Keep five, and write 'CALIBRATED at n=5 is a
weaker claim' into the licence", with the MID lesion family offered as
an add-on to option 1; philo: "yes lets do 1", selecting ten per
primary family; MID was included with it as recommended and flagged,
not separately ruled (reviewer T2: the answer is quoted with the
question it answered). Stamp steps 4a–4e were done at frozen commit
046df7a and are being REDONE: the reviewer's stamp-time read of the
pinned burn script found its per-write space check testing the 126 GiB
start-of-run total instead of the 20 GiB floor (a disk at exactly the
documented minimum would die at the second null — the N1 defect in
code, caught because the hygiene edit sent the reviewer to the pinned
blob). The fix changes the script, so the frozen commit, PINNED.sha256
and §7's constants change with it, under §7's own consequence clause.
Steps 4c–4e redone at the new frozen commit 6b3eab0; second stamp-time
read CLEARED (six checks plus a read of the script's numeric logic).
Awaiting 4f and the principal stamp (4g). No arm file is generated and no judge run starts
before the stamp line (§10) carries philo's word and a date. Ledger authority:
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
had it been there. Every licence below is read under one measured fact
about the frozen readout (§4): its M3 leg is decided at three base
knife-edge steps, all on prompts p3 and p4 (p3 s4, p3 s9, p4 s5), so
for p0, p1 and p2 the two metrics read disjoint evidence: a change
whose visible effect is on those prompts fires M2 and is then
SEPARATED only if its logit perturbation at the three p3/p4 steps
happens to exceed its whole null family's; otherwise it caps at MIXED
however large it is. This is a discrimination test of the readout as
used, not a magnitude calibration. Five verdicts, pre-written (rule in
§6):

- **CALIBRATED** — a supermajority of judged files in BOTH arm classes
  separate from their primary nulls (§6: 3/4 of lesions, 2/3 of
  grafts, unanimity when a denominator sits at its floor). Licenses:
  the "uncalibrated instrument" limitation narrows to its scale clause
  (|Δmargin| still has no behavioral magnitude); the step-5 negative
  gains "a readout shown to separate structured change from
  equal-count noise at ≈42k changed cells concentrated in a 128×512
  block: for grafts, from noise at the same concentration; for lesions,
  from scattered noise (and, reported, from half-window noise)". The
  licence is scoped to that count and density: the step-5 negative
  keeps its caveat that a 61k–80k diffuse change was judged by an
  instrument calibrated on a 42k dense one, and step 8 is not promoted
  on this dose alone; its gate question becomes answerable, not
  answered. This is a discrimination result: the readout can tell
  these two structured changes from equal-count random change; it does
  not map |Δmargin| or any flip to a behavioral or functional unit
  (cross-family finding X5).
- **DAMAGE-ONLY** — lesions clear their bar; grafts do not. Header:
  "{Gs} of {Gj} graft files separated vs LOCAL, below the 2/3 bar".
  Let Gscat = graft files SEPARATED vs SCAT whose SCAT family is
  intact (5 files). A graft whose SCAT family was depleted by the void
  protocol (§5) does NOT count toward Gscat, the conservative direction
  and the same convention as an undefined M3 max; the licence names it:
  "{n} graft file(s) carried a degraded SCAT family and are excluded
  from Gscat". The licence is chosen by (Gs, Gscat), every case with
  exactly one home:
  - Gs = 0, Gscat = 0 → "no graft cleared M2 ∧ M3: 0 of {Gj} vs LOCAL
    and 0 of the {Gj − n} with intact SCAT families vs SCAT. Where a
    graft banded MIXED on M2, the readout DID register a destination
    change it could not corroborate at the three base knife-edge steps
    (p3 s4, p3 s9, p4 s5): a failure of corroboration, not blindness to
    content. A graft is layer k's trit pattern re-encoded under layer
    0's scales (§2); whether it carried function at this site is not
    demonstrated, so this outcome does not distinguish an
    uncorroborating readout from an inert transplant".
  - Gs = 0, Gscat > 0 → "the readout separated {Gscat} of {Gj} grafts
    from scattered noise and 0 of {Gj} from noise at the same
    concentration: sensitivity to concentration is shown, sensitivity to
    content is not established (same transplant caveat)".
  - Gs > 0 → "the readout saw transplanted content in {Gs} of {Gj}
    grafts, below the 2/3 bar; the limitation stands for content
    because the bar was not cleared; the {Gs} separating file(s) are
    printed verbatim (§6 global print rule) as evidence for the next
    design, not as licence".
  In every case the limitation narrows for damage and stands for
  content; step 8 needs a content-sensitive readout before it opens.
  The damage clause itself is scoped: lesions separated "{Ls} of {Lj}
  concentrated zeroings in a quarter window from the same count
  scattered"; whether that is coherence or concentration is read from
  the MID band, reported as "{Lmid} of {Lj} separated vs half-window
  noise" and never entering §6. If
  the graft leg is INERT (§6 rule 3), the sentence "{Gq} of {Gj} graft
  files produced no flips; content-sensitivity is untested on those and
  rests on the remaining {Gj−Gq}" is APPENDED to the chosen case; it
  replaces nothing, so a separating graft stays on record. When Gj − Gq
  = 0 the tail reads instead "and is untested: no graft file produced
  flips and none entered the banding comparison" (cross-family finding
  X1), so the sentence never rests content-sensitivity on zero files.
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
  Pre-commitment (cross-family finding X7): an UNINFORMATIVE verdict is
  NOT attributed to dose, window fraction, or prompt scope in
  `limitations.tex` or anywhere else unless a follow-up varies that one
  factor and shows the verdict move. §8 names those factors as limits
  of scope, not as explanations available after the data.
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
scale bytes asserted untouched). Consequence for grafts, named
(cross-family finding M1): the source trits are re-encoded against
blk.0's own fp16 block scales, so a graft carries layer k's trit
PATTERN under layer 0's magnitudes. It is trained structure in the
wrong place; it is not shown to be function in place, and no arm in
this design tests that. The licences in §1 say so. Nulls: `harness::dose_matched_null`
(exact changed-cell count and full per-class composition, seeded FY
over src-class cells of the slice it is given). Judge:
`tools/run_prompts.sh` verbatim — frozen five prompts, `-n 12 --temp 0
--seed 42`, SHA256SUMS per outdir; double-run files must satisfy the
script's run1 == run2 byte-identity assert (exit 3 otherwise). The
judge BINARY is pinned by content, not only by existence (cross-family
finding X3: a rebuilt fork that still exits 0 and is still
deterministic would pass every other kill): `--measure` prints the
sha256 of every judge binary in `fork-build/llama.cpp/build/bin` and
its libraries, SEVEN files (the driver, five shared objects, and the
dump-patched impl; the log's own summary line says seven), plus the
fork commit it was built from, and asserts each sha equals the
constant pinned in code (reviewer Q3 and R1: the front-end binary is a
thin shell over the shared objects that do the inference, and
`libggml-cpu.so` is the one that computes the logits, so pinning one
of seven, or five of seven, enforces nothing against a rebuilt
kernel). Values from
`measure.log` (sha `10418244…`, commit d1cba18):
`llama-completion` `45c3213681383dc7f5f54bd0b16585c769e3c00450fbc987c250b96214de6246` ·
`libllama-completion-impl.so` `64a1e1848ca747d2ac369a8d93f6a0f46ac7317796683d8c54df3f97bbd32886` ·
`libllama-common.so` `911ad4a6548fbae453652df30692154eb3100d068eed35207c5e264c25f2dff3` ·
`libllama.so` `9f2d10aca3e3e3080650f80624db503459e77e0a889f30f262536b65f79e6ab6` ·
`libggml.so` `ea7f9fd250bc72879a765a0a6156816776a4f570253064fedc446b3cdb7ad63a` ·
`libggml-base.so` `123cbdcd73c0ed5b91f67f593d0a8e2a533113bdb8c2b4e33645ceb88362071d` ·
`libggml-cpu.so` `3e1e166ce399d9e20032deed5e4f3fae3ad7b513a1ad689e906c93f8e9d1bc0a` ·
fork commit `9ca265a57f85f2117942490f421f64a226dd9847`, the one pinned
foreign runtime of the step-5 record. The fork commit is a CONSTRAINT,
not a recording (reviewer Q4): `tools/build_fork.sh` carries it as
`PIN`, the generator carries it as a code constant, and `--measure`
asserts that PIN, constant and the checked-out fork all agree (the
log banner says so in those words). Correction of the record: an
earlier draft here said the log had shown "PIN=" with an empty value;
it had not — "PIN=" was a label naming the source, misread by the
reviewer and adopted by the adjudicator without reading the banner.
The defect did not exist. The banner's trailing `PIN=` was ambiguous
enough to be read as an empty assignment and was reworded in the
d1cba18 log to name the three-way agreement; the three-way assert
exists anyway (reviewer W1: both halves on the record — the misread
and the wording that invited it). The digests above were extracted from the log by script, never
typed (a hand transcription in an earlier draft of this paragraph
carried two slips, caught before the stamp).

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
| GRAFT-k, k∈{1,18,35} | 3 | head block h(k) = 0 / 1 / 2 for k = 1 / 18 / 35: window rows [128h, 128h+128) × cols [0,512) replaced by the SAME rows/cols of `blk.k.attn_q.weight` (that head's projection from the wrong layer, not a shifted copy). Footprints rotated across heads so the three grafts share no base block (cross-family finding M6). Any of the six diff classes. | double |
| IDENTITY | 1 | base slice spliced onto base through the same pipeline; MUST be byte-≡ base (sha) and MUST show zero flips. Contamination tripwire (step 5 §3). | double |
| SCAT null | 10 per LESION file (40, PRIMARY) · 5 per GRAFT file (15, reported) | `dose_matched_null(base_window, arm_window, seed)` over the full 512×512 window: same count and composition, positions random over the whole window. Step 5's own null construction. | single |
| LOCAL null | 10 per GRAFT file (30, PRIMARY) | `dose_matched_null(base_block, graft_block, seed)` over the graft's own 128×512 block, embedded there; same count and composition, positions random INSIDE the block. Separates structure from concentration. | single |
| MID null | 5 per LESION file (20, reported) | `dose_matched_null` over the 256×512 half-window containing block h (rows [256·⌊h/2⌋, +256)), embedded there: the same zeroing count at 50% density against the lesion's 100% and SCAT's ~16%. Interpolates the concentration axis; never enters §6. | single |

Primary families carry TEN files, matching step 5's instrument
(cross-family finding M2: "absent from 5" and "max over 5" make
SEPARATED easier than the ratified n=10, the wrong direction for a
positive control). Reported families carry five.

**Why LESION has no LOCAL family (adjudicator finding on reviewer B6):**
a lesion zeroes every nonzero cell of its block, so a dose-matched null
drawn inside that block must zero the same count from the same pools,
which is every one of them: the local null IS the lesion, byte for
byte. For a full-block zeroing, structure and concentration are the
same fact at the block scale, which is why §1's damage clause reads
"concentrated zeroings in a quarter window from the same count
scattered" and leaves coherence-versus-concentration to the MID band.
For a graft the block is partially changed (~2/3 of cells), so a local
shuffle is a real null.

Bands for GRAFT are computed against BOTH families. §6 decides on the
LOCAL band (the deconfounded one), and the SCAT band rides in the
verdict string, e.g. `DAMAGE-ONLY (graft: NULL-CONSISTENT vs LOCAL,
SEPARATED vs SCAT)`, because a local null is by construction more
similar to its arm than a scattered one, so the graft leg faces the
harder test and a DAMAGE-ONLY whose grafts beat the lesions' own null
construction is a different finding from one whose grafts beat
neither (reviewer B8). Bands for LESION are computed against the SCAT
family; the MID band rides in the verdict string the same way the
graft SCAT band does, e.g. `… (lesion: 3/4 vs SCAT, 1/4 vs MID)`, so
DAMAGE-ONLY can say whether the readout saw coherence or only
concentration (cross-family finding M4). MID was first rejected on
wall-time (reviewer G2) and re-admitted when the run went overnight,
as a flagged add-on to the null-count recommendation philo selected;
it was not separately ruled (header, §9 v20).

**Escalation: none.** Reviewer B4, verified: both metrics are one-sided
against the arm (M2 needs a destination absent from the family, M3
needs the arm above the family max), so adding nulls can only demote a
band, never promote it. Step 5's escalation added ON replicates; here
the arms are deterministic. Primary decades use all ten seeds; reported
families use the first five of theirs and the rest are unused.

Judge runs: 8 double-run files (16) + 105 nulls (40 SCAT-lesion, 15
SCAT-graft, 30 LOCAL, 20 MID) = 121. At 2–4 min per file (BURN.md)
that is 4.0–8.1 h: an overnight run.

## 4. Dose accounting

Predicted from the banked census before measurement (the prediction
stays for the record), then filled from `evidence/step5-calibration/measure.log` (sha
`10418244cef98ee8b69c943a7dd9b5f3f006feefdf0a7f3ee188204f7589d862`,
pinned in `SHA256SUMS`; produced by `step5_calibration --measure` at
commit d1cba18 with the rotated graft footprints, the knife-edge block,
the seven-file judge pin and the three-way fork assert; every reader
check PASS, no kill; supersedes the e11a53b, 0ba1320 and a626996 logs,
whose doses and compositions it reproduces exactly — verified by the
adjudicator against every row of the table below and by the builder
from the diff. Stamp step 4a re-verifies this citation against the log
at the frozen commit):

| Arm | predicted | measured | composition (from→to : n) |
|---|---|---|---|
| LESION-0 | ≈ 41,580 | 42,213 | (−1→0 : 21,191) (+1→0 : 21,022) |
| LESION-1 | ≈ 41,580 | 40,577 | (−1→0 : 20,439) (+1→0 : 20,138) |
| LESION-2 | ≈ 41,580 | 41,549 | (−1→0 : 20,616) (+1→0 : 20,933) |
| LESION-3 | ≈ 41,580 | 42,003 | (−1→0 : 21,007) (+1→0 : 20,996) |
| GRAFT-1 | 41,580–44,760 | 43,586 | (−1→0 : 7,479) (−1→+1 : 6,936) (0→−1 : 7,551) (0→+1 : 7,290) (+1→−1 : 6,761) (+1→0 : 7,569) |
| GRAFT-18 | 41,580–44,760 | 43,526 | (−1→0 : 7,500) (−1→+1 : 6,477) (0→−1 : 7,946) (0→+1 : 7,903) (+1→−1 : 6,407) (+1→0 : 7,293) |
| GRAFT-35 | 41,580–44,760 | 43,568 | (−1→0 : 7,594) (−1→+1 : 6,561) (0→−1 : 7,609) (0→+1 : 7,410) (+1→−1 : 6,689) (+1→0 : 7,705) |
| IDENTITY | 0 | 0 | (none) |

Per-head block zero fraction (lesion footprints): h0 0.3559 · h1 0.3808
· h2 0.3660 · h3 0.3591.

**Base knife-edge steps (the M3 set, margin < θ = 0.05, from the banked
`evidence/session-f-judge/p{0..4}_run1.err`, shas matched to that
directory's SHA256SUMS in the log):** p0 0 of 12 · p1 0 of 12 · p2 0 of
12 · p3 2 of 12 · p4 1 of 12 · total 3 of 60. Reader check: total > 0
(PASS; the log's knife block carries the per-prompt counts with the sha
of each base dump). Independently recounted by the in-family reviewer
with its own dump parser: same three steps, margins 0.0285 (p3 s4), 0.0195 (p3 s9),
0.0260 (p4 s5); all 60 base steps carry 10 logit values, so the
thin-step path never triggers; run1 and run2 margins identical on all
five prompts; the next-smallest margin is p2 s6 at 0.0578, so p2 joins
the knife set at θ = 0.06 and the set is three points, not a shelf.
Consequence, disclosed in §1 and §8: M3 for every arm is defined (a
weight change perturbs p3/p4 logits whether or not it flips them) and
decided at those three steps alone, so for p0–p2 M2 and M3 read
disjoint evidence; an arm whose visible effect is confined to p0–p2 is
SEPARATED only if its p3/p4 perturbation beats its family's, else it
caps at MIXED. Under exchangeability with ten nulls an arm ranks first
on M3 by chance one time in eleven, so M3 alone is weak evidence in
either direction. Two hard blocks also follow from the code: an arm
dump missing any step the base carries returns None for that whole
prompt (`cand.get(&s)?`), and a base step with fewer than two values
would do the same for its prompt; the first is a void condition (§7),
the second cannot occur on the banked base. That is a property of the
frozen readout, not of any arm, and it is exactly the kind of fact
this control exists to put on record. The metric is not changed
(GUARD 2); the aggregator additionally reports an M2-only tally per
arm class (files where M2 fired regardless of M3), as reporting, never
licence.

Graft provenance, from the log: GRAFT-1 source `blk.1.attn_q.weight`
head block 0, data offset 133,467,360, block sha `402c4a1c…`; GRAFT-18
`blk.18` head block 1, offset 589,574,368, sha `4384a9b4…`; GRAFT-35
`blk.35` head block 2, offset 1,045,681,376, sha `5bd5547b…`. The two
offset gaps are EQUAL,
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
report and stands as written, pre-data. Each graft's own head-block
census gives an integer-exact interval (65,536 minus the block's most
and least common class counts), stated alongside: h0 42,213–44,514 ·
h1 40,577–45,398 · h2 41,549–44,920; a value falling between a block
interval and the pre-registered one is reported under both readings
(reviewer J4). Internal check: each block interval's lower endpoint is
exactly that head's LESION dose (42,213 / 40,577 / 41,549), because an
all-zero source changes precisely the block's nonzero cells; one
number proves both (reviewer J2). All three measurements sit inside
both their intervals. With the source
censuses in the log, the exact independence point per graft is 65,536
− Σ(base-block class count × source-block class count)/65,536: GRAFT-1
[21,191, 23,323, 21,022]·[21,088, 23,530, 20,918] → 43,633; GRAFT-18
[20,439, 24,959, 20,138]·[20,815, 23,903, 20,818] → 43,544; GRAFT-35
[20,616, 23,987, 20,933]·[20,759, 24,267, 20,510] → 43,572; against
measured 43,586 / 43,526 / 43,568: each graft sits 0.01–0.11% below its
prediction, a same-sign deficit too small to affect dose comparability
(reviewer H2). LOCAL pools have ≥ 1.45× headroom on every class
(tightest: GRAFT-35, +1 class, 14,394 of 20,933). MID pools are two
heads wide (~41k per class against ~21k demand). The graft block
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
forbids transcriptions: on the v2-code run at commit f07b63f (grafts
then all on head 0), reported by the builder and reproduced line for
line by the adjudicator in a detached worktree, superseded by the
committed `measure.log`, the withdrawn ±5% band would have passed
every arm (lesions −2.43% to +1.51%, grafts within 0.6%),
and the four lesion doses sum to 166,342, the banked nonzero count
exactly. B10 stands on the correlation risk and on principle, not on a
near miss.

## 5. Bands (per arm file vs its primary null family)

- **SEPARATED** — M2 AND M3 fire.
- **MIXED** — exactly one fires.
- **NULL-CONSISTENT** — neither fires.
- **QUIET** — the arm file shows zero flips on all five prompts; it is
  NULL-CONSISTENT for §6's L/G counts and counted separately for INERT.
- **M2-only tally** (reporting, never licence): per arm class, the
  number of files where M2 fired against the primary family, whatever
  M3 did. It exists because M3 is decided entirely at three base
  steps on p3/p4 (§4), so an effect visible to M2 on p0–p2 can be
  large and still fail corroboration there.

Primary families are exactly 10 files, reported families exactly 5. A
PRIMARY family (SCAT for lesions, LOCAL for grafts) reduced below 10
by the void protocol voids its ARM
(that arm leaves the judged set), because a smaller family lowers the
M3 max and biases toward the positive verdict (reviewer S2). A
depleted reported family (graft SCAT, lesion MID) degrades only its
band in the verdict string (reviewer S7). M3 convention when a family has no knife-edge
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
promoted. The verdict string always appends the graft SCAT band and the lesion
MID band (§3), and the M2-only tally (§5).

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
4. Stamp sequence, in this order, each step complete before the next
   (reviewer P1):
   a. From `measure.log` (step 3), fill §2's two values: the judge
      binary sha256 and the fork commit it was built from.
   b. Name the frozen commit: it is step 2's CODE commit on
      `work/step5-calibration`, the one carrying `step5_calibration.rs`
      and `step5_calibration_burn.sh` as they will run. It is NOT the
      stamp commit, which comes later and points back at it (a file
      cannot carry the sha of the commit that contains it).
   c. Compute the two digests from that commit, `git show
      <commit>:<path> | sha256sum`, never from the working tree.
   d. Write `evidence/step5-calibration/generator/PINNED.sha256` with
      the frozen commit and the two digests. That file is AUTHORITATIVE;
      the burn asserts against it.
   e. Transcribe the same three values into §7's placeholders and
      verify the transcription against `PINNED.sha256`; §7's copy is
      human-readable, never a second source (reviewer P2).
   f. Commit this file, `seeds.txt` and `PINNED.sha256`.
   g. philo's stamp line and date in §10; commit that.
   No placeholder survives into the stamp commit. Enforced, not
   narrated (reviewer P3, sourced per reviewer Q1): `--generate` does
   not scan this file for prose tokens (a rule that quotes the token it
   forbids can never clear); it parses the FIELDS — §2's seven judge
   digests and fork commit, §7's frozen commit and two generator
   digests, eleven values in all (7 + 1 + 1 + 2) — requires each to be a well-formed 40-
   or 64-hex value, and
   refuses if §7's three do not match `PINNED.sha256` byte for byte. Consequence, stated: any
   post-stamp edit to the generator or the burn script, including a
   red-commit fix landed as a new commit on top as rule (i) permits,
   changes a pinned digest, fires the burn's assert, and therefore
   means re-stamping from step 3; a new commit on top is allowed, it is
   not free.
5. `--generate`: files named exactly
   `models/cal-lesion-h{h}.gguf`, `models/cal-graft-k{k}.gguf`,
   `models/cal-identity.gguf`,
   `models/cal-{arm}-scat-s{seed}.gguf`,
   `models/cal-graft-k{k}-local-s{seed}.gguf`,
   `models/cal-lesion-h{h}-mid-s{seed}.gguf`;
   every write through `splice_and_verify(expect_src = Some(base))`
   and `assert_unbanked`; `arms.txt` written beside this file listing
   every arm and its families. No name ends in `-control.gguf`,
   `-invivo.gguf` or `-loop.gguf` (banked suffixes refused by the
   guard).
6. Judge: double-run files with `--double`, nulls single; outdirs
   `evidence/step5-calibration/burn/<file-stem>/`. Disk (measured
   2026-09-05: 91.4 GiB free; the full set is 113 files × 1.0011 GiB =
   113.1 GiB) forces an arm-by-arm burn: `--generate --plan` writes
   arms.txt; then, per arm, `--generate --arm <stem>` writes that arm
   and its families (15 nulls for each of the seven perturbation arms;
   IDENTITY has none), so at most 23 files are on disk at once (the 8
   kept arm files plus one arm's 15 transient nulls, ≈ 23 GiB). The
   judge runs them, every null gguf is sha256-pinned into
   `evidence/step5-calibration/nulls.SHA256SUMS`, and only then are
   the null ggufs deleted (arm files and IDENTITY stay). A deleted null
   is regenerable byte-for-byte from four pinned inputs: the committed
   seed, the sha-pinned base (§2), the kept arm file, and the
   generator, which `burn/BURN.log` pins BY CONTENT, not by branch:
   the generator COMMIT sha plus the sha256 of
   `crates/neuralos-rt/examples/step5_calibration.rs` and of
   `tools/step5_calibration_burn.sh` (reviewer K5: a branch head is a
   mutable ref that this repo's own `work/*` rewrite and delete rules
   move; a content pin survives both). A digest proves identity, not
   availability (reviewer L1), so two more rules close the window
   between burn start and the merge: (i) the generator commit is FROZEN
   from BURN START until `work/step5-calibration` merges, in both disk
   paths (reviewer N3) — no rewrite of that branch, force-with-lease
   suspended, any red-commit fix lands as a new commit on top;
   `burn/BURN.log` records the commit as frozen-at with a timestamp;
   (ii) the digests of `step5_calibration.rs` and
   `step5_calibration_burn.sh` at the frozen commit are computed AT
   STAMP TIME and written both here and into the committed
   `evidence/step5-calibration/generator/PINNED.sha256`, so the check
   has a source that predates the burn (reviewer O1: a digest pinned by
   the burn itself and checked by the same burn is a tautology, the
   dead-check class of v1's ±5% band): frozen commit `6b3eab0529c955e7e9d6739fb8c959760c3aa67e`,
   `step5_calibration.rs` sha256 `3fcc1a2fa918d6d628fadf42cd42621b821cbc2c3a7e87a32c449a4f19535040`,
   `step5_calibration_burn.sh` sha256 `b0c40eddc6851212eb4b9b01e9573792c48e80aac4cf588f3e204610b3ab40b7`. At burn start
   the script extracts both files FROM THE FROZEN COMMIT (`git show
   <commit>:<path>`, never the working tree — the CipherPulse PR 3
   lesson, a tree measured for a commit), asserts each extraction
   against `PINNED.sha256` and stops on mismatch (which fires if the
   branch moved between stamp and burn despite the freeze), and writes
   the copies into `evidence/step5-calibration/generator/` with their
   own `SHA256SUMS` entry; the copies are committed with the burn
   evidence (reviewer N2), so availability depends on neither git
   reachability nor the working tree. Regeneration provenance is then settled once that
   commit is an ancestor of merged `main`. The loop is the committed
   `tools/step5_calibration_burn.sh`, which deletes nothing that is not
   a null of the current arm with its outdir and its sha already
   written. Free-space floor: 20 GiB, evaluated with `stat -f` on the
   filesystem that will RECEIVE the file, before each arm's generate
   and before each write batch; a failed check stops the burn and
   never falls back to another device (reviewer L2). Step 5's own null
   files are not touched. Option, philo's offer, TAKEN 2026-09-05: an
   external disk with ≥ 126 GiB free (105 nulls × 1.0011 GiB = 105.1
   GiB, plus the 20 GiB floor, which on a keep-path disk is never
   reclaimed and would otherwise trip at the 86th null — reviewer N1)
   mounted before the burn, its path given to the script; nulls are
   written there and KEPT instead of deleted. The disk philo plugged is
   exFAT with free space far above the requirement; its mount path and
   target directory are
   given at run time in `CAL_NULL_DIR`, never written here (this repo
   has a public mirror and a local path carries a machine account name;
   the operator's environment is where the path lives anyway). exFAT is
   accepted although
   ext4 was asked for: the record of record is the sha pin (§ L4
   parity), not the kept files, so a filesystem without ownership or
   journaling costs nothing evidentiary; the path contains a space and
   the target is a subdirectory, not the mount point itself, so the
   script quotes every path, runs under `set -u`, holds the target in one
   quoted variable, passes shellcheck as a condition of being committed
   (reviewer R4: the mount name contains a space, so one unquoted
   expansion writes to a wrong directory on the root filesystem), and asserts the MOUNT that contains the target
   (`findmnt -T`) rather than `mountpoint` on the directory. Two exFAT
   edges named (reviewer R3): the filesystem is case-insensitive,
   harmless while every generated name is lowercase and no two differ
   only by case, so any later naming scheme must keep that; and it has
   no journal, so an unclean unmount can truncate a null, which the sha
   pin detects only because every null's sha is computed by READING THE
   WRITTEN FILE BACK from the disk, never from the in-memory buffer. The
   script asserts that mount is present,
   not merely a directory, before every arm and every write batch, so
   an unmounted or vanished disk stops the burn instead of silently
   filling the root filesystem; and after each batch the script checks
   that one written null's device id (`stat -c %d`) equals the mount's,
   a mismatch voiding that ARM, not the run (reviewer O2: a disk that
   vanishes mid-batch lands ≤ 15 files on root, which has room, and the
   next arm's assert catches it). Base and arm files never leave
   `models/`. Both paths yield the SAME evidentiary record (reviewer
   L4): kept nulls live on an uncommitted external disk and are not
   `evidence/` artifacts, so in either path the record of record is the
   sha pin plus regenerability from the four pinned inputs, and a
   kept-nulls run is not better evidenced than a deleted-nulls run.
7. `step5_aggregate --poscontrol evidence/step5-calibration/burn`:
   bands, §6 verdict, `scat:` lines for grafts, IDENTITY result. Arm
   list from `arms.txt`, never hardcoded.
8. Record: ISA.md verification entry (append), RESEARCH_LOG, ROADMAP
   item 5; PR on Gitea; merge on philo's word after CI green.

Seeds: `evidence/step5-calibration/seeds.txt`, committed before any
generation. One decade per family; primary families use all ten,
reported families the first five: SCAT LESION-0 301–310 · LESION-1
311–320 · LESION-2 321–330 · LESION-3 331–340 (primary, ten each) ·
SCAT GRAFT-1 341–350 · GRAFT-18 351–360 · GRAFT-35 361–370 (reported,
five each) · LOCAL GRAFT-1 371–380 · GRAFT-18 381–390 · GRAFT-35
391–400 (primary, ten each) · MID LESION-0 401–410 · LESION-1 411–420 ·
LESION-2 421–430 · LESION-3 431–440 (reported, five each). Disjoint
from every seed step 5 or session I used (1–10, 201–250). `harness::decade_for` cannot serve these (it asserts r ∈
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
- judge binary missing (exit 2), or any of the SEVEN judge file sha256s
  ≠ its pin in §2, or the built fork ≠ §2's fork commit (all read from
  `measure.log`);
- any PRIMARY null family below 10 after the void protocol (voids the
  arm; if that drops Lj < 3 or Gj < 2, VOID); a depleted reported
  family degrades only its band (§5);
- base knife-edge count 0 (a §4 reader check; measured 3, so this
  cannot fire on the banked base, and it stays because a re-pinned base
  could differ).

Void protocol (step 5 §6 verbatim, plus one case): exit≠0, empty dump,
or an arm dump missing any step the base carries (which would make M3
None for that prompt) = void, re-run once; twice → the file is excluded
and the family/arm rule above applies; the aggregator prints the case.

Session cap: one burn window, ≤ 9 h judge wall-time (worst case 8.1 h
+ voids). Past the cap: stop, verdict VOID labelled INCOMPLETE, report
what ran file by file; an incomplete run is evidence for nothing
(cross-family finding M7).

## 8. Limitations named before the data

- The window covers 512 of a head's 2560 input dims; LESION-h cuts a
  fifth of one head's q-projection, not the head.
- Arm dose is ≈ 0.6× the step-5 record (§4), by construction of a
  single-head unit.
- For lesions the LOCAL null is the lesion itself (§3), so
  concentration is separated not by a local family but by the reported
  MID band at 50% density; the graft arm separates it by construction.
- A lesion is coherent loss of function; a graft is layer k's trit
  pattern under layer 0's scales, trained structure in the wrong place
  whose function in place is not demonstrated (§2). Neither is
  adaptation. This control bounds what the readout can see; it does not
  manufacture adaptation, and it cannot distinguish a blind readout
  from an inert transplant on the graft leg.
- The frozen readout's M3 leg is decided at 3 of 60 base steps, on p3
  and p4 only (§4); for p0–p2 the two metrics read disjoint evidence,
  and an effect visible only there is SEPARATED only by an
  uncorroborated p3/p4 perturbation, else MIXED. The M2-only tally
  reports it; nothing licenses on it.
- Power, informational (cross-family finding X2), under binomial
  independence of files, PER LEG: P(Ls ≥ 3 of 4) = 4p³ − 3p⁴ reaches
  0.8 at a per-file detection probability p ≈ 0.79; P(Gs ≥ 2 of 3) =
  3p² − 2p³ reaches 0.8 at p ≈ 0.71. CALIBRATED needs both legs, so at
  those rates the verdict itself is reached 0.64 of the time; 0.80
  joint needs p ≈ 0.85 for lesions and ≈ 0.80 for grafts (reviewer
  K4). Below those, UNINFORMATIVE and DAMAGE-ONLY are as consistent
  with low power as with insensitivity; the §1 pre-commitment forbids
  reading them as the latter without a follow-up.
- M2's power depends on how much of the model's output-basin space the
  ten primary nulls happen to cover (cross-family finding X4); coverage
  is not estimated here.
- "Calibration" here means discrimination between two structured
  changes and equal-count noise, not a magnitude scale (X5).
- Primary null families are ten, as in step 5; reported families
  (graft SCAT, lesion MID) are five. No escalation (it could not change
  a verdict).
- The three GRAFT files sit on three different heads, so they share no
  base block; they still share one source-layer choice each and one
  prompt set. Gj = 3 remains a small denominator.
- Cap margin: 8.1 h worst case against a 9 h cap leaves room for
  roughly 13–27 voided files before VOID by cap.
- If a primary family is entirely quiet, its arm can reach at most
  MIXED (§5 M3 convention); the most favourable data pattern this
  control can produce is therefore capped by design, not read up.
- Calibration, if reached, is at ≈42k changed cells at 100% density in
  a quarter window (lesions) or ~66% density in a quarter window
  (grafts). The step-5 record changed 61k–80k cells at ~25% density
  over the whole window. The licence transfers a readout property
  across that gap only as far as §1 says, which is not to step 8.
- Deferred, defaults correct (reviewer Z1): the burn script's keep-disk
  start-gate constant bakes in the 20 GiB floor while the floor itself
  is overridable through `CAL_MIN_FREE_GIB`; with a raised floor a disk
  could clear the gate and trip mid-burn, and the gate message would
  name the wrong floor. This run uses defaults, where the two agree. The
  fix (derive the gate constant from the floor) is a script change and
  therefore a re-stamp, deferred to the next revision of the tool.
- One window, one layer, one model, five prompts, greedy decoding, as
  in step 5. Nothing here generalizes past that scope.

## 9. Changes by version (all reviewer passes, 2026-09-05)

**v2 (in-family pass 1, 2026-09-05):** B1 M2 wording fixed to string-level · B2 INERT added, UNINFORMATIVE
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

**v10 (cross-family pass, Meta lineage, 2026-09-05; philo's ruling on
n):** M1 graft "functional content" was an assumption — the splice
keeps layer 0's scales, so a graft is a trit pattern, not function in
place; licences rewritten, §2 says it · M2 primary families to ten
(philo, option 1) · M3 base knife-edge count measured: 3 of 60, p3/p4
only; disclosed in §1, §4, §8; M2-only tally added; metric unchanged
(GUARD 2) · M4 MID family re-admitted, band in the verdict string,
DAMAGE-ONLY reads coherence vs concentration from it · M5 CALIBRATED
scoped to this count and density, step 8 not promoted on it · M6 graft
footprints rotated to heads 0/1/2, re-measured (0ba1320) · M7 VOID by
cap labelled INCOMPLETE. Run: 121 judge runs over 113 files, 4.0–8.1 h,
cap 9 h.
Seeds extended to 440.

**v11 (MiniMax-family cross-family lane, in-family recount,
2026-09-05):** X1 Gj − Gq = 0 tail · X2 power line · X3 judge binary
pinned by sha · X4 M2 basin-coverage note · X5 discrimination, not
magnitude · X7 UNINFORMATIVE pre-commitment against dose/window/prompt
attribution (X6, shared footprint, already fixed by rotation) · recount
confirmed 3 of 60 with margins and thin-step check; my "M3 undefined
for p0–p2" framing corrected to disjoint-evidence; (0,0) licence made
true when M2 fired; missing-step dumps void; chance rate 1/11 stated.
Scoped §6-rule lane: qwen and grok refused by the gateway (HTTP 401),
lightning silent at 560 s; covered by the in-family recount and the
exhaustive test.

**v11 addendum (disk, 2026-09-05):** arm-by-arm burn, nulls sha-pinned
then deleted, arm files kept; committed burn script; step-5 artifacts
untouched.

**v12 (in-family pass 10, 2026-09-05):** K1 §8 lesion bullet
contradicted §1/§3 after MID was re-admitted · K2 §5 tally rationale
had reverted to "invisible to M3" · K3 §3 quoted §1 wording that no
longer existed · K4 lesion power point corrected to ≈ 0.79 and both
figures marked per-leg with the 0.64 joint · K5 regeneration pinned by
generator commit sha plus file shas, not a branch head · K6 IDENTITY
has no family · K7 sha file path · philo's external-disk offer written
as an option (nulls kept, not deleted, when mounted).

**v13 (in-family pass 11, 2026-09-05):** L1 generator commit frozen
from first deletion to merge, source files copied into evidence at burn
start (identity vs availability) · L2 floor measured on the receiving
filesystem, mount-point assert, no silent fallback · L3 disk figures in
GiB, ≥ 106 GiB for the nulls · L4 both paths yield the same record.

**v14 (in-family pass 12, 2026-09-05):** N1 keep-path disk requirement
≥ 126 GiB (nulls + floor; 106 would trip the floor five arms in) · N2
generator copies extracted from the frozen commit with digest assert,
committed with their own SHA256SUMS entry · N3 freeze from burn start in
both paths.

**v15 (in-family pass 13, 2026-09-05):** O1 generator digests pinned
at stamp time in §7 and in a committed PINNED.sha256, the burn asserts
against them (the BURN.log-sourced assert was a tautology) · O2
per-batch device-id check, mismatch voids the arm.

**v16 (in-family pass 14, 2026-09-05):** P1 stamp sequence written out
a–g, frozen commit = the code commit, never the stamp commit · P2
PINNED.sha256 authoritative, §7 a transcription · P3 --generate refuses
on any surviving placeholder or a §7/PINNED mismatch · post-stamp code
fix means re-stamp, said plainly.

**v17 (in-family pass 15 + the disk, 2026-09-05):** Q1 placeholder
gate parses fields instead of scanning for the tokens the rule itself
quotes · Q2 §4 cites the regenerated measure.log 29b3e50c at a626996 ·
Q3 five judge binaries pinned, filled from the log · Q4 fork commit
9ca265a as a constraint carried by build_fork.sh · disk option taken:
exFAT accepted with the reason, path moved to `CAL_NULL_DIR` at v25, mount asserted via
findmnt on the containing mount.

**v18 (in-family pass 16, 2026-09-05):** R1 seven judge files pinned,
not five — `libggml-base.so` and `libggml-cpu.so` were missing, and the
latter computes the logits; I had read the log through a line window
that ended two lines early (the count-before-concluding rule, again) ·
R2 the two hand-transcription slips recorded as caught before the
stamp, evidence the field gate is needed · R3 exFAT edges: case folding,
sha by read-back · R4 set -u, one quoted variable, shellcheck.

**v19 (in-family pass 17, 2026-09-05):** S1 the §7 kill bullet still
enforced one sha while §2 pinned seven — the amendment reached one of
two records, the D4 class, and the reviewer's own paraphrase of that
bullet in pass 16 had masked it · S2 the gate counts eleven fields, not
ten.

**v20 (in-family pass 18, 2026-09-05):** T1 header counts dropped,
§9 is the one record of review history · T2 the null-count ruling is
quoted with the question it answered; MID marked as included with the
recommendation, not separately ruled.

**v21 (in-family pass 19, 2026-09-05):** U1 header names what covers
the failed §6-rule lane (exhaustive test + verdict-function read), not
the recount · U2 §3 carries MID's recommended-not-ruled flag.

**v22 (builder status + in-family pass 20 cosmetic list, 2026-09-05):**
§2 "PIN= empty" corrected — the defect did not exist, the three-way
assert does · §4 cites the final measure.log 10418244 at d1cba18 · §9
heading dated for both days, pass-1 entry labelled, "121 judge files"
is 121 runs over 113 files.

**v23 (in-family pass 21, 2026-09-05):** W1 the Q4 correction records
both halves: the token was misread, and the banner was reworded because
the token was ambiguous.

**v24 (stamp steps a–e, 2026-09-05):** §7's three constants filled from
PINNED.sha256, itself written from `git show 046df7a:<path> | sha256sum`;
§2 already carried its eight from the final log; zero placeholders.

**v25 (public-repo hygiene, 2026-09-05, builder's sweep before push):**
the local mount path and account name removed from §7; the path lives
in `CAL_NULL_DIR` at run time. No pinned value touched; frozen commit
unchanged.

**v26 (stamp-time read of the pinned script, 2026-09-05):** Y1 the
burn script's per-write check enforced the 126 GiB total instead of the
20 GiB floor; fixed in code → new frozen commit, steps 4c–4e redone ·
the disk's exact free space removed from §7 (a drive fingerprint, not
an identifier; philo's call, taken conservatively).

**v27 (new frozen commit 6b3eab0, 2026-09-05):** Y1 fixed by removing the
space check from `require_mount` (it asserts only the mount; `require_free`
owns per-write space, now also called on the null path per arm) · Y2,
found by the red probe leg: every threshold was compared as a
locale-formatted string (fr_CA decimal comma), so `125,4 < 20` was true
and `10,00 > 9` false — the floor stopped a run with 125 GiB free and
would have passed one with 5, and the 9 h cap could never fire; all
thresholds now compared in integer bytes or seconds with `LC_ALL=C`,
formatting for display only · Y3 a bare `[ … ] &&` statement returned 1
under `set -e` on every delete-path run, written as `if` · probe legs
recorded: old script red at 125.4 GiB; new script green at 126.5→125.4
and at 20.5, red at 19.5 with exit 3 VOID (INCOMPLETE), cap red at −1 h,
delete path green · §7 constants and PINNED.sha256 rewritten from `git
show` at the new commit.

**v28 (second stamp-time read, 2026-09-05):** CLEARED; Z1 (override-path
gate constant) recorded as deferred with defaults verified correct.

**v29 (date correction, 2026-09-05 14:20 EDT):** from v14 onward the
adjudicator wrote "2026-09-06" in this document while the local date was
still 2026-09-05 (the whole review ran on one day, 2026-09-05, roughly
10:00 to 14:20 EDT; the commit timestamps, which git takes from the
clock, were right all along). Eighteen occurrences corrected. The
adjudicator had not read the clock; philo noticed the time.

## 10. Stamp

_unstamped_
