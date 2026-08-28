# NeuralOS v2

`no_std`, i16 fixed-point spiking-neural-network library for RISC-V edge
silicon (`crates/neuralos-snn`, published), plus a Slint visualizer
(`neuralos-app`) and an unpublished research runtime (`neuralos-rt`).
Not an "AI OS"; that was v0.1. The sibling `NeuralOS` repo is that
archived v0.1 and is read-only source material.

This file points. It does not restate. Anything copied here becomes a
fourth record that drifts, and drift is this project's failure mode.

## Read in this order before acting

1. `ISA.md` frontmatter `head:` line: branch, last session, open items,
   active guards. One read, current state — and only that line: the
   `phase: complete` / `progress:` fields two lines above it describe a
   closed task, not the repo (§ Trap below).
2. `docs/ROADMAP.md` § Priority order and § Practical next moves: the
   present-tense claimant on open work. README carries one
   external-facing status line and is not a state source.
3. `ISA.md` tail, the last `## Close-out` / `## Amendment`: what the
   previous session actually did and what it left open.
4. `AGENTS.md` § Session discipline and § Session protocol: read before
   scoping any session. Discipline decides whether a session is legal
   to open at all (spine-first budget, consolidation cadence, freeze
   evidence never source). Protocol carries the roles, verify-first
   ("a second model agreeing is not a source"), the interrupt rule,
   and the findings taxonomy (blocking / cosmetic-list / record-only).
5. `docs/VISION.md`: the charter. Read when the question is direction,
   not next action.

## Do not violate

- **GUARD 1: `ISA.md` is APPEND-ONLY until paper submission.**
  `paper/figs/mechanism.py` regex-parses its ISC-78 entry. Never edit a
  past entry; correct by appending, and tombstone the superseded one as
  true-when-written. Live-state lines (frontmatter, claim counts) are
  the only exception.
- **GUARD 2:** touching a frozen example's recorded verdict, or
  re-litigating Branch B, IS reopening the closed bridge chapter. Stop.
  Only the principal opens it.
- **GUARD 3:** no outreach before the `.nir` exporter exists AND is
  stranger-usable (binaries on the release); the exporter and the
  ESP32-C3 board landing together gate outreach.
- **Never force-push Gitea.** `origin` has two push URLs, so one
  `git push origin main` hits Gitea (canonical) and the GitHub mirror
  together. If Gitea is ahead, rebase. The mirror is the only
  force-push target, and only with the principal's explicit OK.
- **`evidence/` holds three species.** sha-pinned machine outputs are
  immutable and never edited. Frozen-by-marker human docs (PREREG.md)
  are immutable because the marker says so. Living orchestration docs
  (BURN.md, PREP.md) are kept true, with git as their changelog.
  Anything that must survive a deposit (a Zenodo tarball carries no
  `.git`) is pointed at the frozen files, never copied into the
  living ones.
- **Every amendment ends with a grep sweep.** Move a figure (n, seeds,
  arms, windows, a published version) and you sweep the whole tree for
  every record of it in the same session. Forward citations name
  symbols or sections, never line numbers.

## Merging

Work lands on `work/<name>` or `fix/<name>`; CI runs on the branch.
**The Gitea run is the gate** (`.gitea/workflows/ci.yml`). The GitHub
mirror's run is a second opinion that never blocks a merge: do not wait
on it, do not treat its red as a gate. Merge to main requires branch-CI
green plus review passed. Those are mechanical conditions with no
discretion, and the builder may merge on them. Branches are deleted
after merge.

## Trap: ISA.md is not a charter

Its `## Problem`, `## Vision`, `## Goal`, `## Out of Scope` and its
`principal_stated_goal` are scoped to the ternary bridge Stage 2, a
CLOSED chapter. They read like a project charter and are not one.
`phase: complete` and `progress: 88/88` describe that task, not the
repo. For direction read `docs/VISION.md`; for state read `head:`.

## Gates

Commands live in `AGENTS.md` § Commands. Local green is one machine's
opinion; CI on the pushed ref is the gate. And green gates only prove
the code runs. They say nothing about whether the record is true, which
is where this project's defects actually land.
