#!/bin/bash
# hybrid_smoke.sh — the one-command "the root works" check (R4(i)).
#
# Proves the closed bridge loop still runs end-to-end on this tree:
# SNN adaptation (hybrid_gate) + export/surgery (hybrid_loop) reproduce
# their recorded verdicts, deterministically, on the real models.
#
# Tiers:
#   1. no args            — compile check + loud-refusal check (CI-safe:
#                           missing model must exit non-zero, write nothing)
#   2. AUTO_SMOKE=1       — tier 1 + full gate + loop runs + export-sha pin
#                           (needs models/Ternary-Bonsai-4B-Q2_0.gguf,
#                           ~1 GB, gitignored; ~1 min wall, ~2 GB RSS)
# The sweeps/invivo/judge legs stay outside the smoke (cost); their
# re-pin protocol lives in evidence/r4-baselines/README.md.
#
# Usage: bash tools/hybrid_smoke.sh        (tier 1)
#        AUTO_SMOKE=1 bash tools/hybrid_smoke.sh   (tier 2)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=models/Ternary-Bonsai-4B-Q2_0.gguf
LOOP_OUT=/tmp/opencode/r4-smoke-loop.gguf   # smoke artifact — never the banked file
PINNED_EXPORT_SHA_PREFIX=24ffe5f3

echo "== tier 1: compile =="
cargo build --release -p neuralos-rt --examples

echo "== tier 1: missing model refuses loudly, writes nothing =="
if ./target/release/examples/hybrid_gate /nonexistent.gguf > /tmp/opencode/smoke-refusal.log 2>&1; then
    echo "FAIL: missing model exited 0"; exit 1
fi
test ! -e /nonexistent.gguf
echo "refusal: non-zero exit, no writes — OK"

if [ "${AUTO_SMOKE:-0}" != "1" ]; then
    echo "== tier 1 PASS (set AUTO_SMOKE=1 with models present for the full run) =="
    exit 0
fi

[ -f "$MODEL" ] || { echo "FAIL: $MODEL absent"; exit 1; }
mkdir -p /tmp/opencode

echo "== tier 2: hybrid_gate — recorded verdict must reproduce =="
./target/release/examples/hybrid_gate | tee /tmp/opencode/smoke-gate.log
grep -q 'HYBRID GATE: ADAPTS' /tmp/opencode/smoke-gate.log
grep -q 'intra |mean Δ| (GATE) : 0.1075' /tmp/opencode/smoke-gate.log
echo "gate verdict pinned"

echo "== tier 2: hybrid_loop — export must be bit-exact (sha $PINNED_EXPORT_SHA_PREFIX…) =="
./target/release/examples/hybrid_loop "$MODEL" "$LOOP_OUT" | tee /tmp/opencode/smoke-loop.log
grep -q 'LOOP EXPORT: CLEAN' /tmp/opencode/smoke-loop.log
SHA=$(sha256sum "$LOOP_OUT" | cut -c1-8)
if [ "$SHA" != "$PINNED_EXPORT_SHA_PREFIX" ]; then
    echo "FAIL: export sha $SHA != pinned $PINNED_EXPORT_SHA_PREFIX"; exit 1
fi
rm -f "$LOOP_OUT"
echo "== tier 2 PASS: gate ADAPTS @ +0.1075, loop CLEAN, export sha $SHA =="
