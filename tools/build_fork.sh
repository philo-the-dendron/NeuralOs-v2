#!/bin/bash
# Build the foreign judge runtime (llama-completion) for bridge experiments.
#
# Pin:   PrismML-Eng/llama.cpp @ 9ca265a (branch prism, CPU-only build).
# Patch: tools/neuralos_dump.patch — env-gated (NEURALOS_DUMP=1) top-10
#        pre-sampler logit dump at the completion sample site, comma
#        decimals (the exact format tools/delta.py parses).
# Out:   fork-build/llama.cpp/build/bin/llama-completion (gitignored).
#
# Judge invocation of record (greedy forced, double-run per variant):
#   NEURALOS_DUMP=1 fork-build/llama.cpp/build/bin/llama-completion \
#     -m <model.gguf> -p "<prompt>" -n 12 --temp 0 --top-k 0 --top-p 1.0 \
#     --min-p 0.0 --seed 42 -no-cnv -c 512 -t 4
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FORK_DIR="${NEURALOS_FORK_DIR:-$REPO_ROOT/fork-build}"
SRC="$FORK_DIR/llama.cpp"
PIN="9ca265a57f85f2117942490f421f64a226dd9847"

if [ ! -d "$SRC/.git" ]; then
    git clone https://github.com/PrismML-Eng/llama.cpp "$SRC"
fi
if [ "$(git -C "$SRC" rev-parse HEAD)" != "$PIN" ]; then
    echo "fork HEAD != pin, checking out $PIN" >&2
    git -C "$SRC" fetch origin "$PIN" 2>/dev/null || git -C "$SRC" fetch origin
    git -C "$SRC" checkout "$PIN"
fi

# Re-apply the dump patch from a clean tree (idempotent rebuilds).
git -C "$SRC" checkout -- tools/completion/completion.cpp
git -C "$SRC" apply "$REPO_ROOT/tools/neuralos_dump.patch"

# cmake via nix when absent (same fallback pattern as paper/Makefile's tectonic).
if command -v cmake >/dev/null 2>&1; then
    CMAKE=(cmake)
else
    CMAKE=(nix shell nixpkgs#cmake nixpkgs#ninja -c cmake)
fi
"${CMAKE[@]}" -S "$SRC" -B "$SRC/build" -DLLAMA_CURL=OFF
"${CMAKE[@]}" --build "$SRC/build" --target llama-completion -j "$(nproc)"

BIN="$SRC/build/bin/llama-completion"
[ -x "$BIN" ] || { echo "build failed: $BIN missing" >&2; exit 1; }
echo "judge ready: $BIN"
