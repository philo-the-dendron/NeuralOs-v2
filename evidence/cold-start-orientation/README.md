# Cold-start orientation measurement — the C1 bank

> **FROZEN BY MARKER, 2026-08-29.** Species 2 record (frozen human
> doc): immutable because this marker says so. Correct by superseding
> file, never by editing.

Banked on the principal's ruling (2026-08-29) closing finding C1 of
the 2026-08-28 adjudication: the orientation layer's before/after
measurement was cited as measured with no entry in `evidence/`. This
record is that entry.

## Method

Two cold starts, identical conditions. A fresh plain agent session is
opened from the repo root with no context pasted, and asked one
question: *"What is this project, what's the current state, and what
should I work on next?"* Recorded: tool calls to a defensible answer,
blind seeks into the 5,472-line `ISA.md` ledger, and project rules
named unprompted. The re-measurement is typed by the principal in a
fresh session — a relayed cross-session prompt is a different context
and voids the comparison. The baseline session, once lost, wrote the
fix itself (`CLAUDE.md`, the orientation file): the agent that just
hunted knows what would have saved it the calls.

## Result

| | Baseline (2026-08-28, pre-CLAUDE.md) | After (2026-08-29, orientation branch) |
|---|---|---|
| Tool calls to orient | 11 | 4 |
| Blind seeks into the ISA ledger | 3 | 0 |
| Project rules named unprompted | 0 | 6 |

Root cause found by the baseline run: **`AGENTS.md` is not auto-loaded
by the agent harness** — a repo-root `CLAUDE.md` is the only file a
fresh session receives from the repo. The doctrine existed; none of it
was reaching cold sessions.

## After-run verdict (tester, verbatim)

"wrote-from:fix/orientation-followups@5e66e6e UNMERGED (CI+review)
answered 'where was this written, is it landed, what gates landing' in
one read, and open(session) assigned every open item to a role. I
didn't need git to learn the branch was unmerged or who owes what.
Under the old line I'd have had to reconstruct all of that from git
log plus the tail." The E2 legality clause and the head: partition
both fired unprompted in the same run.

## Provenance and honest limits

The runs are recorded in `ISA.md`: baseline and fix in § Session open
+ close-out (the orientation doc, 2026-08-28); the after-run verdict
in § Amendment (E1/E2 ratified + head: partition grammar, cold-start
verified, 2026-08-28); the banking ruling closes § C1 of the
adjudicated-findings session. The raw session transcripts are
session-local to the operator's private agent infrastructure and are
deliberately not committed here; this is therefore a frozen
contemporaneous human record, not a machine log, and it claims
species-2 standing, not species-1. The measured deltas are as
recorded at adjudication and ratified by the principal at banking
time. The after-run was measured against the orientation branch's
dirty tree at base 5e66e6e; the content it exercised merged unchanged
in a1e7e19.
