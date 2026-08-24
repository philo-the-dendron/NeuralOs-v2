#!/usr/bin/env bash
# tools/burn.sh — the step-5 burn-window runner (BURN.md, mechanical).
#
# Each leg is a fenced unit: set -e halts the chain on ANY failure, the
# trap writes a HALT sentinel (leg name + timestamp) that refuses all
# further legs until an operator clears it, and every leg tees to its
# own log under evidence/step5-readout/burn/logs/.
#
# A void (§6) is a nonzero exit — same loud path. Nothing sits silent.
#
# Usage (from anywhere; the script cd's to the repo root):
#   tools/burn.sh pre      # verify-free + identity tripwires + their judges (minutes)
#   tools/burn.sh off      # driven OFF-r0 (~6-8 h) + judge --double
#   tools/burn.sh rep0|rep1|rep2   # ON-r → nulls-r → judge nulls → judge ON (--double)
#   tools/burn.sh domain   # DOMAIN arm + judge --double
#   tools/burn.sh free     # judge the banked ck files (single-run)
#   tools/burn.sh verdict  # step5_aggregate over the burn root
#   tools/burn.sh all      # pre off rep0 rep1 rep2 domain free verdict, in order
#
# Clearing a halt: inspect the log, fix, then
#   rm evidence/step5-readout/burn/HALT
# and RE-START FROM THE FAILED LEG ONLY — never re-run `all` blindly:
# completed 6–8 h arms are already banked; a blind re-run duplicates
# them (deterministic re-pins cost wall-time, never correctness, but
# the waste is real).
#
# set -E (errtrace): the ERR trap MUST fire inside shell functions —
# every expensive command here runs inside one; without -E the HALT
# sentinel is dead code on exactly the failure paths that matter.
#
# Power-loss / reboot recovery: BURN.md §Power-loss / reboot recovery
# (forensics via the logs' start/done lines; paranoia pin; resume from
# the interrupted leg only — never re-run `all` blindly).
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BURN="evidence/step5-readout/burn"
LOGS="$BURN/logs"
HALT="$BURN/HALT"
mkdir -p "$LOGS"

LEG=${1:?usage: tools/burn.sh <pre|off|rep0|rep1|rep2|domain|free|verdict|all>}

if [ -f "$HALT" ]; then
  echo "HALT sentinel present (from a previous leg):" >&2
  cat "$HALT" >&2
  echo "Inspect $LOGS, fix, then: rm $HALT" >&2
  exit 1
fi

on_err() {
  {
    echo "HALT: leg [$LEG] failed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "  failed command line: ${BASH_COMMAND}"
    echo "  log: $LOGS/leg-$LEG.log"
  } > "$HALT"
  echo "=== LEG [$LEG] FAILED — HALT written, chain stopped ===" >&2
}
trap on_err ERR

# Per-leg log, console output preserved (tee).
exec > >(tee -a "$LOGS/leg-$LEG.log") 2>&1
echo "=== leg [$LEG] start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

run_example() { # relay to cargo, release, quiet build noise kept in log
  cargo run -p neuralos-rt --release --example "$@"
}

judge() { # $1 model, $2 dir, $3 maybe --double
  tools/run_prompts.sh "$1" "$BURN/$2" ${3:-}
}

leg_pre() {
  run_example step5_aggregate -- --verify-free
  run_example hybrid_invivo -- --identity 1
  run_example hybrid_invivo -- --identity 2
  judge models/Ternary-Bonsai-4B-Q2_0-invivo-identity-r1.gguf identity-r1 --double
  judge models/Ternary-Bonsai-4B-Q2_0-invivo-identity-r2.gguf identity-r2 --double
}

leg_off() {
  run_example hybrid_invivo -- --off
  judge models/Ternary-Bonsai-4B-Q2_0-invivo-off-r0.gguf off-r0 --double
}

leg_rep() { # $1 = r
  local r=$1
  run_example hybrid_invivo -- --window "$r"
  run_example step5_nulls -- --replicate "$r"
  local f d
  for f in models/null-r${r}-s*.gguf; do
    d=$(basename "$f" .gguf)
    judge "$f" "$d"
  done
  judge "models/Ternary-Bonsai-4B-Q2_0-invivo-r${r}.gguf" "on-r${r}" --double
}

leg_domain() {
  run_example hybrid_invivo -- --domain-corrected
  judge models/Ternary-Bonsai-4B-Q2_0-invivo-domain.gguf domain --double
}

leg_free() {
  judge models/Ternary-Bonsai-4B-Q2_0-invivo-ck400.gguf  free-ck400
  judge models/Ternary-Bonsai-4B-Q2_0-invivo-ck800.gguf  free-ck800
  judge models/Ternary-Bonsai-4B-Q2_0-invivo-ck1200.gguf free-ck1200
}

leg_verdict() {
  run_example step5_aggregate -- "$BURN"
}

case "$LEG" in
  pre)     leg_pre ;;
  off)     leg_off ;;
  rep0)    leg_rep 0 ;;
  rep1)    leg_rep 1 ;;
  rep2)    leg_rep 2 ;;
  domain)  leg_domain ;;
  free)    leg_free ;;
  verdict) leg_verdict ;;
  all)
    for l in pre off rep0 rep1 rep2 domain free verdict; do
      echo "--- chain: dispatching leg [$l] ---"
      # A child leg's failure writes its own SPECIFIC halt sentinel
      # (the || arm runs outside the ERR trap, so this parent never
      # overwrites it with a generic one).
      "$0" "$l" || { echo "chain halted by leg [$l] — see $HALT" >&2; exit 1; }
    done
    ;;
  *) echo "unknown leg: $LEG" >&2; exit 2 ;;
esac

echo "=== leg [$LEG] done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
