#!/bin/bash
# Ladder rung 2: rule-fixed sibling prompts (7 weekday rotations + month
# chain + off-circuit negatives), single run each — margins from the dump.
#
# Rebuild the judge first:  bash tools/build_fork.sh
# Usage: tools/sibling_prompts.sh <model.gguf> <tag>
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; TAG="$2"
FORK_DIR="${NEURALOS_FORK_DIR:-$REPO_ROOT/fork-build}"
DIR="${NEURALOS_OUT_DIR:-$FORK_DIR/se/$TAG}"
BIN="${NEURALOS_LLAMA_BIN:-$FORK_DIR/llama.cpp/build/bin/llama-completion}"
mkdir -p "$DIR"
PROMPTS=("Tuesday Wednesday Thursday" "Wednesday Thursday Friday" "Thursday Friday Saturday" "Friday Saturday Sunday" "Saturday Sunday Monday" "Sunday Monday Tuesday" "January February March" "red green blue" "2 4 6 8")
for i in "${!PROMPTS[@]}"; do
  NEURALOS_DUMP=1 "$BIN" -m "$MODEL" -p "${PROMPTS[$i]}" -n 12 --temp 0 --top-k 0 --top-p 1.0 --min-p 0.0 --seed 42 -no-cnv -c 512 -t 4 > "$DIR/s${i}.log" 2> "$DIR/s${i}.err"
  echo "s$i exit=$? [${PROMPTS[$i]}]"
done
