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
# Every threshold below is compared in INTEGER BYTES or SECONDS, never on a
# formatted number: this box runs fr_CA, where printf "%.1f" yields "125,4",
# and mawk then treats that as a STRING, so `125,4 < 20` was TRUE by
# lexicographic order. C numeric formatting on top, so the log reads the
# same everywhere and nothing re-parses a localized figure.
export LC_ALL=C

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

# Thresholds as integers, derived once. The awk here runs under LC_ALL=C
# and its output is consumed as an integer, never re-parsed as a decimal.
MIN_FREE_BYTES=$(awk -v g="$MIN_FREE_GIB" 'BEGIN{printf "%d", g*1073741824}')
CAP_SECONDS=$(awk -v h="$CAP_HOURS" 'BEGIN{printf "%d", h*3600}')
# 105 nulls × 1.0011 GiB + the 20 GiB floor = 126 GiB.
KEEP_DISK_BYTES=135291469824

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
  local dir="$1" what="$2" b
  b=$(avail_bytes "$dir")
  if [ "$b" -lt "$MIN_FREE_BYTES" ]; then
    say "STOP: $dir has $(gib "$b") GiB free, under the ${MIN_FREE_GIB} GiB floor, before $what"
    say "      no fallback device — VOID (INCOMPLETE)"
    exit 3
  fi
}
# The null directory is a SUBDIRECTORY of the external mount, not the mount
# itself, and its path contains a space — so `mountpoint -q` on it is wrong
# twice. Ask which filesystem contains it instead, and quote everything.
mount_of() { # $1 = path -> the mount target that contains it
  findmnt -n -o TARGET -T "$1" 2>/dev/null | tail -1
}
mount_dev() { # $1 = path -> device id of the containing mount
  stat -c %d "$(mount_of "$1")" 2>/dev/null
}
require_mount() { # $1 = what is about to happen
  [ "$KEEP_NULLS" = 1 ] || return 0
  local m
  m=$(mount_of "$NULL_DIR")
  if [ -z "$m" ] || [ "$m" = "/" ]; then
    say "STOP: '$NULL_DIR' is contained by '${m:-nothing}' before $1 — the external disk is"
    say "      unmounted or vanished; refusing to write nulls onto the root filesystem"
    say "      — VOID (INCOMPLETE)"
    exit 3
  fi
  if [ "$m" != "$NULL_MOUNT" ]; then
    say "STOP: '$NULL_DIR' is now on '$m', not the '$NULL_MOUNT' it started on, before $1"
    say "      — VOID (INCOMPLETE)"
    exit 3
  fi
  # No space check here. The 126 GiB figure is the WHOLE RUN's requirement
  # and belongs only at the start-of-run gate below: free space on a keep
  # disk falls monotonically, so re-testing the total before every write
  # would kill a disk that started at exactly the documented minimum on its
  # second file. Per-write space is `require_free`, which tests the floor
  # and is called on this same directory right before each move.
}
gib() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }
gb() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1000000000}'; }
elapsed_h() { awk -v s="$START" 'BEGIN{printf "%.2f", (systime()-s)/3600}'; }
elapsed_s() { echo $(( $(date +%s) - START )); }

KEEP_NULLS=0
NULL_MOUNT=""
if [ -n "$NULL_DIR" ]; then
  [ -d "$NULL_DIR" ] || { echo "CAL_NULL_DIR does not exist: '$NULL_DIR'" >&2; exit 2; }
  NULL_MOUNT=$(mount_of "$NULL_DIR")
  if [ -z "$NULL_MOUNT" ] || [ "$NULL_MOUNT" = "/" ]; then
    echo "REFUSING: '$NULL_DIR' is contained by '${NULL_MOUNT:-nothing}', not by an external" >&2
    echo "mount. A disk that is not mounted would silently fill the root filesystem." >&2
    exit 2
  fi
  NULL_MOUNT_DEV=$(stat -c %d "$NULL_MOUNT")
  NB=$(avail_bytes "$NULL_DIR")
  # 126 GiB: 105 × 1.0011 GiB of nulls = 105.1, plus the 20 GiB floor that
  # a keep disk never reclaims. This is the WHOLE RUN's requirement and it
  # is tested HERE ONLY — free space falls as the run writes, so re-testing
  # a total before every write would kill a disk that started at exactly
  # the minimum. Per-write space is require_free's floor.
  if [ "$NB" -ge "$KEEP_DISK_BYTES" ]; then
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
  # PINNED.sha256 is `commit <sha>` plus two sha256sum lines carrying the
  # REPO-RELATIVE paths, so match on the path, not the basename.
  want=$(awk -v n="$f" '$1 != "commit" { sub(/^\*/, "", $2); if ($2 == n) { print $1; exit } }' "$PINNED")
  if [ -z "$want" ]; then
    echo "REFUSING: $PINNED has no line for $f" >&2
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
GEN_SRC_SHA=$(awk -v n="$GEN_SRC" '$1 != "commit" { sub(/^\*/, "", $2); if ($2 == n) { print $1; exit } }' "$PINNED")
SELF_SHA=$(awk -v n="$SELF" '$1 != "commit" { sub(/^\*/, "", $2); if ($2 == n) { print $1; exit } }' "$PINNED")
PINNED_COMMIT=$(awk '$1 == "commit" { print $2; exit }' "$PINNED")
if [ -z "$PINNED_COMMIT" ]; then
  echo "REFUSING: $PINNED has no \`commit <sha>\` line" >&2
  exit 2
fi
if [ "$PINNED_COMMIT" != "$FROZEN_COMMIT" ]; then
  echo "REFUSING: $PINNED pins commit $PINNED_COMMIT, HEAD is $FROZEN_COMMIT — the branch" >&2
  echo "moved between stamp and burn despite the freeze." >&2
  exit 2
fi

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
    echo "nulls     : KEPT in '$NULL_DIR'"
    echo "            mount '$NULL_MOUNT' (device $NULL_MOUNT_DEV), $(gib "$(avail_bytes "$NULL_DIR")") GiB free"
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
  if [ "$KEEP_NULLS" = 1 ]; then
    require_free "$NULL_DIR" "arm $arm nulls"
  fi
  if [ "$(elapsed_s)" -gt "$CAP_SECONDS" ]; then
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
        # Across filesystems mv is a copy, and exFAT has no journal: a
        # truncated copy would look like a file. Digest before, digest the
        # read-back after, and require them equal.
        before=$(sha256sum "$f" | cut -d' ' -f1)
        mv "$f" "$NULL_DIR/$stem.gguf"
        sync -f "$NULL_DIR/$stem.gguf" 2>/dev/null || sync
        f="$NULL_DIR/$stem.gguf"
        after=$(sha256sum "$f" | cut -d' ' -f1)
        if [ "$before" != "$after" ]; then
          say "VOID arm $arm: $stem.gguf read back as $after after the move, was $before —"
          say "     a short or corrupted copy. Arm VOID, run continues (§7)."
          arm_void=1
          break 2
        fi
        # The file must be ON the mount, not in a directory that shadows it:
        # a mount that vanished between the check and the write would leave
        # the null on the root filesystem with the right path.
        fdev=$(stat -c %d "$f")
        if [ "$fdev" != "$NULL_MOUNT_DEV" ]; then
          say "VOID arm $arm: $stem.gguf landed on device $fdev, the '$NULL_MOUNT' mount is"
          say "     $NULL_MOUNT_DEV — it is not on the disk. Arm VOID, run continues (§7)."
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
      # Read back from disk, never from anything held in memory.
      sync -f "$f" 2>/dev/null || sync
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
