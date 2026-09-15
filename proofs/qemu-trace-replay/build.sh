#!/usr/bin/env bash
# proofs/qemu-trace-replay/build.sh [LOG]
#
# The frozen traces under QEMU riscv64gc: build the replay bare metal,
# run it on QEMU's virt machine, and diff what its UART printed against
# the tree's trace files with the board's diff tool
# (tools/esp32c3_trace_diff.py), so a wrong writer is red, not trusted.
# Green only if QEMU exits 0 (the test device's pass code, written after
# the end line) and the diff's last line is "N cases, 0 red", N the
# plasticity-off traces of crates/neuralos-snn/tests/traces/ counted here
# from their header lines, as the diff reads them: every one replayed row
# for row (12 at round 33, 14 since PR G). A trace file deleted from the
# tree lowers both counts, but the replay still prints that case, and the
# diff calls a header outside the tree's set red.
#
# A release-round gate, run by hand and pinned, like Leg A: not a CI job
# (QEMU is not on the runner image). The capture goes to LOG, default
# target/replay.log under the crate (CARGO_TARGET_DIR honored); a LOG
# that exists is refused, so a pinned log is never written over (a
# re-run writes a new name and compares, evidence/qemu-riscv-gate/
# README.md). The crate's own format gate runs first: `cargo fmt --all`
# at the root never reaches this workspace. Needs the
# riscv64gc-unknown-none-elf target on the pin and qemu-system-riscv64
# (8.2.2 on the laptop that banked the log).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
cd "$here"
target_dir=${CARGO_TARGET_DIR:-$here/target}
elf=$target_dir/riscv64gc-unknown-none-elf/release/qemu-trace-replay
log=$target_dir/replay.log
if [ $# -gt 0 ]; then
  log=$1
  if [ -e "$log" ]; then
    echo "build.sh: $log exists; a pinned log is never written over, name a new one" >&2
    exit 1
  fi
fi

cargo fmt -- --check
cargo build --release --locked

# The count the diff must read: the tree's plasticity-off traces.
want=0
for trace in "$repo"/crates/neuralos-snn/tests/traces/*.trace; do
  if head -1 "$trace" | grep -q ' plasticity=off '; then
    want=$((want + 1))
  fi
done

mkdir -p "$(dirname "$log")"
qemu=0
timeout 120 qemu-system-riscv64 -machine virt -nographic -bios none \
  -kernel "$elf" </dev/null >"$log" || qemu=$?
diff=0
verdict=$(python3 "$repo/tools/esp32c3_trace_diff.py" "$log") || diff=$?
printf '%s\n' "$verdict"
echo "log sha256 $(sha256sum "$log" | cut -d' ' -f1)  ${log#"$repo"/}"

if [ "$qemu" -ne 0 ]; then
  echo "qemu-trace-replay: RED, QEMU exited $qemu (0 is the pass code after the end line, 1 a panic, 124 the timeout)" >&2
  exit 1
fi
if [ "$diff" -ne 0 ] || [ "$(tail -1 <<<"$verdict")" != "$want cases, 0 red" ]; then
  echo "qemu-trace-replay: RED, the diff does not read $want cases, 0 red" >&2
  exit 1
fi
echo "qemu-trace-replay: green, $want cases, 0 red: every plasticity-off trace replays bit for bit under QEMU riscv64gc"
