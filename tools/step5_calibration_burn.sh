#!/usr/bin/env bash
# tools/step5_calibration_burn.sh — the step-5 CALIBRATION burn window,
# arm by arm (PREREG §7 step 6).
#
# The whole set does not fit on the working disk: 8 arm files + 105 nulls
# = 113 gguf = 113.1 GiB against 91.4 GiB free (measured 2026-09-05). Two
# paths, chosen by CAL_NULL_DIR:
#
#   unset  — each arm is generated, judged, its nulls sha-pinned and then
#            DELETED before the next arm starts. Peak 8 arm files + one
#            arm's 15 nulls = 23.0 GiB.
#   set    — an external mount with ≥ 130 GB free: nulls are moved there
#            after generation, judged in place and KEPT. Nothing is
#            deleted.
#
# Base and arm files always stay in models/, and cal-identity is never
# touched. The sha pin into nulls.SHA256SUMS is written on both paths.
#
# A deleted null stays provable: same seed + same base + same generator is
# byte-identical by construction (the r4-closeout regeneration proof).
# BURN.log pins what "same generator" means BY CONTENT — the commit sha,
# and the sha256 of the generator source and of this script as they are on
# disk at start — because work/* branches in this repo get rewritten and
# deleted, so a branch name would orphan the recipe. The run refuses to
# start if either file is dirty in the working tree.
#
# Usage:
#   bash tools/step5_calibration_burn.sh            # all arms, resuming
#   bash tools/step5_calibration_burn.sh lesion-h0  # one arm
#
# Overridable for a dry run (defaults are the real paths):
#   CAL_MODELS_DIR · CAL_EVIDENCE_DIR · CAL_JUDGE · CAL_GENERATE
#   CAL_NULL_DIR · CAL_CAP_HOURS (9) · CAL_MIN_FREE_GIB (20)
#
# Exit codes: 0 ok · 2 setup/refusal · 3 cap or disk stop = VOID (INCOMPLETE)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODELS=${CAL_MODELS_DIR:-models}
EV=${CAL_EVIDENCE_DIR:-evidence/step5-calibration}
BURN="$EV/burn"
ARMS="$EV/arms.txt"
NULLSUMS="$EV/nulls.SHA256SUMS"
JUDGE=${CAL_JUDGE:-tools/run_prompts.sh}
CAP_HOURS=${CAL_CAP_HOURS:-9}
MIN_FREE_GIB=${CAL_MIN_FREE_GIB:-20}
NULL_DIR=${CAL_NULL_DIR:-}
ONLY_ARM=${1:-}

GEN_SRC="crates/neuralos-rt/examples/step5_calibration.rs"
SELF="tools/step5_calibration_burn.sh"

[ -f "$ARMS" ] || { echo "no $ARMS — run: step5_calibration --generate --plan" >&2; exit 2; }
[ -x "$JUDGE" ] || { echo "judge chain not executable: $JUDGE" >&2; exit 2; }

# The regeneration recipe must be committed content, not a working copy.
for f in "$GEN_SRC" "$SELF"; do
  if [ -n "$(git status --porcelain -- "$f")" ]; then
    echo "REFUSING: $f is dirty in the working tree — the regeneration recipe would name" >&2
    echo "content that is not in any commit. Commit it, then start the burn." >&2
    exit 2
  fi
done

mkdir -p "$BURN"
LOG="$BURN/BURN.log"
say() { echo "$*" | tee -a "$LOG"; }

# The generator. Built once here so BURN.log pins the exact binary that
# produced every file; a dry run points CAL_GENERATE at a stub.
if [ -z "${CAL_GENERATE:-}" ]; then
  cargo build -p neuralos-rt --release --example step5_calibration >&2
  CAL_GENERATE="$ROOT/target/release/examples/step5_calibration"
fi
[ -x "$CAL_GENERATE" ] || { echo "generator not executable: $CAL_GENERATE" >&2; exit 2; }

avail_bytes() { df -B1 --output=avail "$1" | tail -1 | tr -d ' '; }
# The floor is checked on the filesystem that will RECEIVE the file, and a
# failure stops the burn. There is no fallback device: writing somewhere
# else would put half the run on a disk BURN.log does not name.
require_free() { # $1 = receiving dir, $2 = what is about to be written
  local dir="$1" what="$2" b f
  b=$(avail_bytes "$dir"); f=$(gib "$b")
  if awk -v f="$f" -v m="$MIN_FREE_GIB" 'BEGIN{exit !(f<m)}'; then
    say "STOP: $dir has ${f} GiB free, under the ${MIN_FREE_GIB} GiB floor, before $what"
    say "      no fallback device — VOID (INCOMPLETE)"
    exit 3
  fi
}
# A vanished or unmounted external disk must stop the run, not silently
# become a directory on the root filesystem.
is_mountpoint() { # $1 = path
  if command -v mountpoint >/dev/null 2>&1; then
    mountpoint -q "$1"
  else
    [ "$(stat -c %d "$1" 2>/dev/null)" != "$(stat -c %d "$1/.." 2>/dev/null)" ]
  fi
}
require_mount() { # $1 = what is about to happen
  [ "$KEEP_NULLS" = 1 ] || return 0
  if ! is_mountpoint "$NULL_DIR"; then
    say "STOP: $NULL_DIR is not a mount point before $1 — unmounted or vanished"
    say "      refusing to write nulls onto the root filesystem — VOID (INCOMPLETE)"
    exit 3
  fi
  local b f
  b=$(avail_bytes "$NULL_DIR"); f=$(gib "$b")
  # 105.1 GiB of nulls plus the 20 GiB floor, which is never reclaimed on a
  # keep disk: at 106 the floor would trip around the 86th null.
  if awk -v f="$f" 'BEGIN{exit !(f<126)}'; then
    say "STOP: $NULL_DIR has ${f} GiB free, under the 126 GiB the kept null set needs"
    say "      (105.1 GiB of nulls + the ${MIN_FREE_GIB} GiB floor), before $1"
    exit 3
  fi
}
gib() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }
gb() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1000000000}'; }
elapsed_h() { awk -v s="$START" 'BEGIN{printf "%.2f", (systime()-s)/3600}'; }

KEEP_NULLS=0
if [ -n "$NULL_DIR" ]; then
  [ -d "$NULL_DIR" ] || { echo "CAL_NULL_DIR does not exist: $NULL_DIR" >&2; exit 2; }
  if ! is_mountpoint "$NULL_DIR"; then
    echo "REFUSING: CAL_NULL_DIR $NULL_DIR is a directory, not a mount point. An external" >&2
    echo "disk that is not mounted would silently fill the root filesystem instead." >&2
    exit 2
  fi
  NB=$(avail_bytes "$NULL_DIR")
  # 126 GiB: 105 × 1.0011 GiB of nulls = 105.1, plus the 20 GiB floor that
  # a keep disk never reclaims.
  if awk -v b="$NB" 'BEGIN{exit !(b >= 135291469824)}'; then
    KEEP_NULLS=1
  else
    echo "REFUSING: CAL_NULL_DIR $NULL_DIR has $(gib "$NB") GiB free, under the 126 GiB a kept" >&2
    echo "null set needs (105.1 GiB of nulls + the ${MIN_FREE_GIB} GiB floor, never reclaimed)." >&2
    echo "Unset it to run the delete-after-pin path instead." >&2
    exit 2
  fi
fi

# The regeneration recipe is frozen BY CONTENT beside the evidence, so it
# survives a branch rewrite or a deleted work/* branch: git reachability is
# not what makes a deleted null provable, these bytes are. The copies come
# from the COMMIT, never from the working tree, so what is frozen is what
# BURN.log names — and each copy's digest is asserted against the recorded
# one before the first arm runs.
FROZEN_AT=$(date -Is)
FROZEN_COMMIT=$(git rev-parse HEAD)
mkdir -p "$EV/generator"
PINNED="$EV/generator/PINNED.sha256"
# The digests are checked against a file written at STAMP TIME and
# committed with the stamp — not against anything this run produces. A
# digest pinned by the burn and checked by the burn cannot fail.
if [ ! -f "$PINNED" ]; then
  echo "REFUSING: $PINNED is missing. It is written at stamp time from the frozen commit" >&2
  echo "and committed with the stamp; without it there is nothing independent to check" >&2
  echo "the frozen generator against." >&2
  exit 2
fi
: > "$EV/generator/SHA256SUMS"
for f in "$GEN_SRC" "$SELF"; do
  bn=$(basename "$f")
  want=$(awk -v n="$bn" '{ sub(/^\*/, "", $2); if ($2 == n) { print $1; exit } }' "$PINNED")
  if [ -z "$want" ]; then
    echo "REFUSING: $PINNED has no line for $bn" >&2
    exit 2
  fi
  git show "$FROZEN_COMMIT:$f" > "$EV/generator/$bn"
  got=$(sha256sum "$EV/generator/$bn" | cut -d' ' -f1)
  if [ "$want" != "$got" ]; then
    echo "REFUSING: $bn at $FROZEN_COMMIT is $got, $PINNED says $want" >&2
    echo "The stamped generator and this commit are not the same code." >&2
    exit 2
  fi
  echo "$got  $bn" >> "$EV/generator/SHA256SUMS"
done
GEN_SRC_SHA=$(awk -v n="$(basename "$GEN_SRC")" '{ sub(/^\*/, "", $2); if ($2 == n) { print $1; exit } }' "$PINNED")
SELF_SHA=$(awk -v n="$(basename "$SELF")" '{ sub(/^\*/, "", $2); if ($2 == n) { print $1; exit } }' "$PINNED")

START=$(date +%s)
{
  echo "=== step-5 calibration burn — $(date -Is) ==="
  echo "commit    : $FROZEN_COMMIT   (a commit, never a ref name)"
  echo "gen src   : $GEN_SRC"
  echo "gen srcsha: $GEN_SRC_SHA"
  echo "script    : $SELF"
  echo "scriptsha : $SELF_SHA"
  echo "frozen-at : $FROZEN_AT — copies extracted from the commit into $EV/generator/,"
  echo "            digests asserted against $PINNED (written at stamp time), listed in"
  echo "            $EV/generator/SHA256SUMS (commit them with the burn evidence)"
  echo "generator : $CAL_GENERATE"
  echo "gen binsha: $(sha256sum "$CAL_GENERATE" | cut -d' ' -f1)"
  echo "judge     : $JUDGE"
  echo "models    : $MODELS"
  echo "evidence  : $EV"
  if [ "$KEEP_NULLS" = 1 ]; then
    echo "nulls     : KEPT in $NULL_DIR ($(gb "$(avail_bytes "$NULL_DIR")") GB free)"
  else
    echo "nulls     : pinned then DELETED per arm (no CAL_NULL_DIR)"
  fi
  echo "cap       : ${CAP_HOURS} h — a cap stop is VOID (INCOMPLETE)"
  echo "min free  : ${MIN_FREE_GIB} GiB on the receiving filesystem, before every write"
  echo ""
  echo "DOCTRINE, from THIS MOMENT (burn start, both paths) until this branch merges:"
  echo "  the generator commit above is FROZEN. No rewrite of work/step5-calibration,"
  echo "  --force-with-lease suspended, a red commit is fixed by a NEW commit on top."
  echo "  A rewritten history would orphan the recipe that makes a deleted null provable."
} | tee -a "$LOG"

mapfile -t LINES < <(grep -v '^[[:space:]]*#' "$ARMS" | grep -v '^[[:space:]]*$')

for line in "${LINES[@]}"; do
  arm=$(echo "$line" | awk '{print $1}')
  fams=$(echo "$line" | cut -s -d' ' -f2- | tr -d '*')
  [ -z "$ONLY_ARM" ] || [ "$ONLY_ARM" = "$arm" ] || continue
  if [ -f "$BURN/$arm.done" ]; then
    say "arm $arm: already done ($BURN/$arm.done) — skipping"
    continue
  fi

  FB=$(avail_bytes "$MODELS"); FREE=$(gib "$FB"); EL=$(elapsed_h)
  say ""
  say "--- arm $arm · free ${FREE} GiB · elapsed ${EL} h · families [${fams:-none}] ---"
  require_free "$MODELS" "arm $arm generate"
  require_mount "arm $arm"
  if awk -v e="$EL" -v c="$CAP_HOURS" 'BEGIN{exit !(e>c)}'; then
    say "STOP: ${EL} h exceeds the ${CAP_HOURS} h cap — VOID (INCOMPLETE)"
    exit 3
  fi

  arm_void=0
  say "generate: $arm"
  "$CAL_GENERATE" --generate --arm "$arm" >&2

  # 1. the arm file itself, double-run. It stays in models/ always.
  armgguf="$MODELS/cal-$arm.gguf"
  [ -f "$armgguf" ] || { say "REFUSING: $armgguf missing after generate"; exit 2; }
  say "judge   : $armgguf → $BURN/cal-$arm (double)"
  "$JUDGE" "$armgguf" "$BURN/cal-$arm" --double

  # 2. its nulls — IDENTITY has none, and this loop then does nothing.
  if [ -z "$fams" ]; then
    say "nulls   : none for $arm (the tripwire has no family) — nothing to pin, nothing to delete"
  fi
  for fam in $fams; do
    for f in "$MODELS"/cal-"$arm"-"$fam"-s*.gguf; do
      [ -e "$f" ] || continue
      stem=$(basename "$f" .gguf)

      # Name shape and family membership are checked BEFORE anything is
      # moved or removed, on both paths.
      case "$stem" in
        cal-"$arm"-scat-s*|cal-"$arm"-local-s*|cal-"$arm"-mid-s*) ;;
        *) say "REFUSING: $f is not a null of $arm"; exit 2 ;;
      esac
      if ! echo " $fams " | grep -q " $fam "; then
        say "REFUSING: family $fam is not listed for $arm in $ARMS"; exit 2
      fi

      if [ "$KEEP_NULLS" = 1 ]; then
        require_mount "moving $stem.gguf"
        require_free "$NULL_DIR" "moving $stem.gguf"
        mv "$f" "$NULL_DIR/$stem.gguf"
        f="$NULL_DIR/$stem.gguf"
        # The file must be ON the mount, not in a directory that shadows it:
        # a mount that vanished between the check and the write would leave
        # the null on the root filesystem with the right path.
        fdev=$(stat -c %d "$f"); mdev=$(stat -c %d "$NULL_DIR")
        if [ "$fdev" != "$mdev" ]; then
          say "VOID arm $arm: $stem.gguf landed on device $fdev, $NULL_DIR is $mdev —"
          say "     it is not on the mount. Arm VOID, run continues (§7)."
          arm_void=1
          break 2
        fi
        say "moved   : $stem.gguf → $NULL_DIR (device $fdev)"
      fi

      say "judge   : $f → $BURN/$stem (single)"
      "$JUDGE" "$f" "$BURN/$stem"

      if [ ! -f "$BURN/$stem/SHA256SUMS" ]; then
        say "REFUSING: $BURN/$stem/SHA256SUMS is missing — the judge leg is not pinned"; exit 2
      fi
      sha=$(sha256sum "$f" | cut -d' ' -f1)
      touch "$NULLSUMS"
      if grep -q "  $stem.gguf\$" "$NULLSUMS"; then
        prev=$(grep "  $stem.gguf\$" "$NULLSUMS" | cut -d' ' -f1)
        if [ "$prev" != "$sha" ]; then
          say "REFUSING: $NULLSUMS already pins $prev for $stem.gguf, this file is $sha"; exit 2
        fi
      else
        echo "$sha  $stem.gguf" >> "$NULLSUMS"
      fi
      if ! grep -q "^$sha  $stem.gguf\$" "$NULLSUMS"; then
        say "REFUSING: the sha line for $stem.gguf did not land in $NULLSUMS"; exit 2
      fi

      if [ "$KEEP_NULLS" = 1 ]; then
        say "pinned  : $sha  $stem.gguf — judged, pinned, KEPT in $NULL_DIR"
      else
        rm -f "$f"
        say "pinned  : $sha  $stem.gguf — judged, pinned, deleted"
      fi
    done
  done

  if [ "$arm_void" = 1 ]; then
    say "arm $arm: VOID — no done marker written, it will be retried on the next run"
    continue
  fi
  date -Is > "$BURN/$arm.done"
  say "arm $arm: done · free $(gib "$(avail_bytes "$MODELS")") GiB · elapsed $(elapsed_h) h"
done

say ""
say "burn done — $(elapsed_h) h elapsed. Aggregate with:"
say "  cargo run -p neuralos-rt --release --example step5_aggregate -- --poscontrol $BURN"
