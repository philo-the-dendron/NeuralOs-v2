@AGENTS.md

# NeuralOS v2

`no_std`, i16 fixed-point spiking-neural-network library for RISC-V edge
silicon (`crates/neuralos-snn`, published), plus a Slint visualizer
(`neuralos-app`) and an unpublished research runtime (`neuralos-rt`).
Not an "AI OS"; that was v0.1. The sibling `NeuralOS` repo is that
archived v0.1 and is read-only source material.

This file points. It does not restate — except for the rules in
§ Do not violate, whose violation is unrecoverable and which it
restates deliberately: an agent that reaches such a rule by pointer
has usually already broken it. Each restated guard cites its source
so the amendment grep sweep catches the copy. Everything else copied
here becomes a fourth record that drifts, and drift is this project's
failure mode.

## Read in this order before acting

1. `ISA.md` frontmatter `head:` line: `main@<sha>`, `wrote-from:` (the
   branch it was written from and its merge state), `open(session):`
   (session-scoped open items, each assigned a role), `next-work:`
   (the pointer to ROADMAP § Practical next moves). One read, current
   state — and only that line: the `phase:` / `progress:` fields in
   the same frontmatter block describe a closed task, not the repo
   (§ Trap below).
2. `docs/ROADMAP.md` § Priority order and § Practical next moves: the
   present-tense claimant on open work. README carries one
   external-facing status line and is not a state source.
3. `ISA.md` tail, the last `## Close-out` / `## Amendment`: what the
   previous session actually did and what it left open.
4. `AGENTS.md` § Session discipline and § Session protocol: read before
   scoping any session — and if this session will change anything,
   after reading the list, before executing ROADMAP work: ROADMAP
   names the next work, discipline decides whether opening it is
   legal, and legality is decided first.
   A reporting-only turn may take the list as written. Discipline
   carries the spine-first budget, the consolidation cadence and
   freeze-evidence-never-source; Protocol carries the roles,
   verify-first ("a second model agreeing is not a source"), the
   interrupt rule, and the findings taxonomy (blocking /
   cosmetic-list / record-only).
5. `docs/VISION.md`: the charter. Read when the question is direction,
   not next action.

## Do not violate

- **GUARD 1: `ISA.md` is APPEND-ONLY until paper submission.**
  `paper/figs/mechanism.py` regex-parses its ISC-78 entry. Never edit a
  past entry; correct by appending, and tombstone the superseded one as
  true-when-written. Live-state lines (frontmatter, claim counts) are
  the only exception. Source: `ISA.md` § Decision (M1 — the merged
  ship-then-measure plan).
- **GUARD 2:** touching a frozen example's recorded verdict, or
  re-litigating Branch B, IS reopening the closed bridge chapter. Stop.
  Only the principal opens it. Source: `ISA.md` § Decision (M1 — the
  merged ship-then-measure plan).
- **GUARD 3:** no outreach before the `.nir` exporter exists AND is
  stranger-usable (binaries on the release); the exporter and the
  ESP32-C3 board landing together gate outreach. Source: `ISA.md`
  § Decision (M1 step-4 amendment — the pure-Rust exporter ruling),
  the widened rule, artifact-named per B1b.
- **Never force-push Gitea, and no branch on GitHub.** Gitea is canonical
  and the only gate. GitHub is a mirror, and philo's
  meaning of the word is the rule: "no branch on github, github is a
  mirror." It carries `main` and Gitea's tags, nothing else; no branch,
  PR, or CI run exists there except on `main`. `origin` pushes to Gitea
  alone. The mirror is written only after a Gitea merge, by the
  procedure in § Merge procedure, and the release binaries live on the
  Gitea release.
  `origin` has one push URL since 2026-09-06. If Gitea is ahead,
  rebase. GitHub is the only force-push target, and only with the
  principal's explicit OK. Source: `AGENTS.md` § Remotes (push
  carefully) and § Merge procedure.
- **`evidence/` holds three species.** sha-pinned machine outputs are
  immutable and never edited. Frozen-by-marker human docs (PREREG.md)
  are immutable because the marker says so. Living orchestration docs
  (BURN.md, PREP.md) are kept true, with git as their changelog.
  Anything that must survive a deposit (a Zenodo tarball carries no
  `.git`) is pointed at the frozen files, never copied into the
  living ones. Source: `AGENTS.md` § Session discipline (the autopsy
  doctrine).
- **Every amendment ends with a grep sweep.** Move a figure (n, seeds,
  arms, windows, a published version) and you sweep the whole tree for
  every record of it in the same session. Forward citations name
  symbols or sections, never line numbers. Source: `AGENTS.md`
  § Session discipline (the autopsy doctrine).

## Merging

Work lands on `work/<name>` or `fix/<name>`; CI runs on the branch.
**The Gitea run is the gate** (`.gitea/workflows/ci.yml`). GitHub runs
the same workflow on `main` only, after the sync in `AGENTS.md` § Merge
procedure: the one off-box build, never a gate. Merge to main requires branch-CI
green plus review passed, and every commit on the branch green on its
own, proven by the `per-commit` CI job on the pull request, whose log the
merger reads (source: `AGENTS.md` § Session protocol, Git discipline). Those are mechanical conditions
with no discretion, and the builder may merge on them. Branches are
deleted after merge.

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
