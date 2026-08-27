#!/usr/bin/env bash
# tools/run_prompts.sh — the frozen-five judge chain (step-5 burn).
#
# The banked invocation of record, verbatim
# (evidence/session-h2/README.md § Rebuild; build_fork.sh):
#   NEURALOS_DUMP=1 llama-completion -m <model> -p '<prompt>' -n 12
#     --temp 0 --top-k 0 --top-p 1.0 --min-p 0.0 --seed 42 -no-cnv -c 512 -t 4
#
# Protocol (PREREG §3): single-run per NULL/FREE file (family
# determinism); DOUBLE-run for every ON/OFF/IDENTITY/DOMAIN file, with
# the run1==run2 byte-identity assert (the contamination tripwire).
# Emits SHA256SUMS over every log+err in the outdir (PREREG §8).
#
# Usage:
#   tools/run_prompts.sh <model.gguf> <outdir> [--double]
# Exit codes: 0 ok · 2 judge missing · 3 double-run determinism fail.
set -euo pipefail

JUDGE="fork-build/llama.cpp/build/bin/llama-completion"
if [ ! -x "$JUDGE" ]; then
  echo "judge missing: $JUDGE — build it first: bash tools/build_fork.sh" >&2
  exit 2
fi

MODEL=${1:?usage: run_prompts.sh <model.gguf> <outdir> [--double]}
OUT=${2:?usage: run_prompts.sh <model.gguf> <outdir> [--double]}
DOUBLE=${3:-}
if [ "$DOUBLE" != "" ] && [ "$DOUBLE" != "--double" ]; then
  echo "unknown flag: $DOUBLE (expected --double)" >&2; exit 2
fi
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 2; }

# Banked-evidence guard (operator-error class): never write into an
# existing adjudicated evidence dir — burn outputs live under
# evidence/step5-readout/burn/ only.
# NAMESPACE NOTE (fence-catch #3, 2026-08-26): "r4" below means the R4
# REMEDIATION-LADDER family (evidence/r4-baselines, r4-closeout) — NOT
# the escalation's replicate-4 (on-r4, null-r4-s24x). The globs are
# anchored at the evidence/ root for exactly that reason: step5-readout/
# sits between, so burn-root r4 dirs pass while banked R4 dirs refuse.
case "$OUT" in
  evidence/session-*|evidence/r4-*|evidence/qemu-riscv-gate|evidence/nir-hdf5-gate|evidence/nir-assembly-gate)
    echo "REFUSING: $OUT matches a banked evidence family — burn outputs live under evidence/step5-readout/burn/" >&2
    exit 2 ;;
esac

mkdir -p "$OUT"

# The frozen five (session-h2 README, byte-exact prompts).
PROMPTS=(
  '1 2 3 4 5 6 7'
  '10 11 12 13'
  'one two three four'
  'Monday Tuesday Wednesday'
  'The capital of France is'
)

run_one() { # $1 = outfile prefix, $2 = prompt
  NEURALOS_DUMP=1 "$JUDGE" -m "$MODEL" -p "$2" -n 12 \
    --temp 0 --top-k 0 --top-p 1.0 --min-p 0.0 --seed 42 \
    -no-cnv -c 512 -t 4 > "$OUT/${1}_run1.log" 2> "$OUT/${1}_run1.err"
}

for i in 0 1 2 3 4; do
  p="p${i}"
  echo "judge $MODEL → $OUT/$p"
  run_one "$p" "${PROMPTS[$i]}"
  if [ "$DOUBLE" = "--double" ]; then
    NEURALOS_DUMP=1 "$JUDGE" -m "$MODEL" -p "${PROMPTS[$i]}" -n 12 \
      --temp 0 --top-k 0 --top-p 1.0 --min-p 0.0 --seed 42 \
      -no-cnv -c 512 -t 4 > "$OUT/${p}_run2.log" 2> "$OUT/${p}_run2.err"
    if ! cmp -s "$OUT/${p}_run1.log" "$OUT/${p}_run2.log"; then
      echo "DOUBLE-RUN FAIL: $p run1 != run2 — void protocol §6" >&2
      exit 3
    fi
  fi
done

( cd "$OUT" && sha256sum p*_run*.log p*_run*.err > SHA256SUMS )
echo "judge chain done: $OUT ($( [ "$DOUBLE" = "--double" ] && echo double-run || echo single-run ))"
