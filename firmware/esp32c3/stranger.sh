#!/usr/bin/env bash
# firmware/esp32c3/stranger.sh build <graph.nir>
# firmware/esp32c3/stranger.sh run <graph.nir> [--port /dev/ttyACM0] [--seconds 20]
#
# The one command (docs/ROADMAP.md § 0.1.0, check 7): a stranger's
# `.nir` on the ESP32-C3, the board's rows against the host's trace.
# Runs from any directory; the repo root comes from git, as build.sh
# finds it. README.md beside this file is the stranger's page.
#
# build: neuralos-nir2json --sim-units --freeze writes
#   target/stranger/<stem>.{json,rs,trace} under this crate (gitignored
#   by **/target/; the sidecar beside the JSON) with --steps 150, which
#   keeps a capture inside 20 s; then the capacity check below; then
#   build.sh and build.sh clippy with NEURALOS_GRAPH=<the module> and
#   CARGO_TARGET_DIR=<that directory>, so the remap, the personal-string
#   gate, the shas, the canary and the lint apply to this configuration
#   and the default artifact under target/ never moves; the ELF is
#   copied to <stem>.elf. build.rs's slot takes the module (its header
#   carries the rules). A stranger without a board runs this phase.
# run: build, then `espflash flash` (never `cargo run`: its runner
#   monitors the port), then tools/esp32c3_capture.py alone on the port
#   into <stem>.capture.log, then tools/esp32c3_trace_diff.py with
#   --trace <stem>.trace. The verdict is the diff's last line and exit
#   code. The bench rules are in esp32c3_capture.py's header. A capture
#   is green only when the fragment before the reset carries no header:
#   a reboot mid-replay is red by design, not a shorter compare.
#
# Paths. The root is resolved first, and NEURALOS_GRAPH and
# CARGO_TARGET_DIR are exported absolute, since a relative one resolves
# against whatever directory cargo is read from. The converter runs from
# the root with the caller's target dir: under this crate,
# .cargo/config.toml would build it for the chip. Two graphs with one
# stem share one scratch name; the last build wins, and the trace beside
# it is always that build's. build.sh's `cargo clean -p
# esp-bootloader-esp-idf` hits the scratch target dir only; its
# trim-paths canary still probes at the default target/canary, a cheap
# pre-compile that exits 101 by design (tools/remap.sh).
#
# The capacity check: a floor, not a measured stack, provisional until
# check 8. It reads N and S from the module's two `pub const` lines,
# matched whole (the freezer's tested format; the sidecar carries no
# counts), refuses unless exactly one of each is found, and refuses by
# name, exit 2, when 44·N + 6·S exceeds 65,536 bytes: a neuron is 44
# bytes and a synapse 6, both pinned by tests, so about 1,489 neurons or
# 10,922 synapses. The formula counts the arrays alone; the caller's
# input and fired arrays, the frames, the printer and the statics are
# outside it. 65,536 is about a fifth of the C3's 313 KB of DRAM, whose
# stack is whatever the statics leave (esp-hal's stack.x sizes nothing);
# it coincides with the QEMU harness's 64K stack line and is not derived
# from the chip. PR I replaces it: capacity defined (u16 ids, N and S,
# RAM, the link), the largest case built in the main clone through
# build.sh, and ns per step, the spike fold, a full replay at 0 red and
# the stack's high-water mark banked.
set -euo pipefail

usage() {
  echo "usage: $0 build <graph.nir>" >&2
  echo "       $0 run <graph.nir> [--port /dev/ttyACM0] [--seconds 20]" >&2
  exit 2
}

here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
[ $# -ge 2 ] || usage
phase=$1
case $phase in
  build | run) ;;
  *) usage ;;
esac
if [ ! -f "$2" ]; then
  echo "stranger.sh: $2: no such file" >&2
  exit 2
fi
nir=$(realpath -e -- "$2")
shift 2
port=/dev/ttyACM0
seconds=20
while [ $# -gt 0 ] && [ "$phase" = run ]; do
  case $1 in
    --port)
      [ $# -ge 2 ] || usage
      port=$2
      shift 2
      ;;
    --seconds)
      [ $# -ge 2 ] || usage
      seconds=$2
      shift 2
      ;;
    *) usage ;;
  esac
done
[ $# -eq 0 ] || usage

# What this script prints is repo-relative, the record's convention.
rel() { printf '%s' "${1#"$repo"/}"; }

stem=$(basename "$nir" .nir)
out=$here/target/stranger
module=$out/$stem.rs
trace=$out/$stem.trace
elf=$out/$stem.elf
mkdir -p "$out"

# 1. The converter, from the root (the header says why), its paths
#    repo-relative so its own lines print them that way.
(cd "$repo" && cargo run --locked -p neuralos-nir2json -- --sim-units \
  --freeze "$(rel "$module")" --steps 150 "$(rel "$nir")" "$(rel "$out/$stem.json")")

# 2. The capacity check (the header). 10# reads the counts as decimal.
n_lines=$(grep -cE '^    pub const N: usize = [0-9]+;$' "$module" || true)
s_lines=$(grep -cE '^    pub const S: usize = [0-9]+;$' "$module" || true)
if [ "$n_lines" != 1 ] || [ "$s_lines" != 1 ]; then
  echo "stranger.sh: REFUSED: $(rel "$module") holds $n_lines 'pub const N' and $s_lines 'pub const S' lines, not one of each" >&2
  exit 2
fi
n=$(sed -nE 's/^    pub const N: usize = ([0-9]+);$/\1/p' "$module")
s=$(sed -nE 's/^    pub const S: usize = ([0-9]+);$/\1/p' "$module")
bytes=$((44 * 10#$n + 6 * 10#$s))
if [ "$bytes" -gt 65536 ]; then
  echo "stranger.sh: REFUSED, capacity: 44·$n + 6·$s = $bytes bytes of arrays, over the floor of 65,536 (a floor, not a measured stack, provisional until check 8; the header)" >&2
  exit 2
fi
echo "capacity: 44·$n + 6·$s = $bytes bytes of arrays, within the floor of 65,536 (a floor, not a measured stack, provisional until check 8)"

# 3. The firmware with the graph in its slot, through build.sh.
#    Assigned first: `echo "$(f)"` hides f's exit status (tools/remap.sh).
export NEURALOS_GRAPH=$module
export CARGO_TARGET_DIR=$out
built=$("$here/build.sh")
printf '%s\n' "$built"
"$here/build.sh" clippy
cp "$out/riscv32imc-unknown-none-elf/release/neuralos-esp32c3" "$elf"
elf_sha=$(sed -n 's/^elf  sha256 \([0-9a-f]\{64\}\) .*/\1/p' <<<"$built")
text_sha=$(sed -n 's/^text sha256 \([0-9a-f]\{64\}\) .*/\1/p' <<<"$built")
copy_sha=$(sha256sum "$elf" | cut -d' ' -f1)
if [ -z "$elf_sha" ] || [ -z "$text_sha" ] || [ "$copy_sha" != "$elf_sha" ]; then
  echo "stranger.sh: the shas did not read back from build.sh, or the copy differs from the ELF it built" >&2
  exit 1
fi
echo "stranger: $(rel "$nir"), $n neurons, $s synapses"
echo "  module      $(rel "$module")"
echo "  trace       $(rel "$trace")"
echo "  elf         $(rel "$elf")"
echo "  elf  sha256 $elf_sha"
echo "  text sha256 $text_sha"
[ "$phase" = run ] || exit 0

# 4. The board: flash, capture from the reset, diff.
log=$out/$stem.capture.log
espflash flash --port "$port" --non-interactive "$elf"
python3 "$repo/tools/esp32c3_capture.py" "$port" "$seconds" "$log"
echo "capture: $(rel "$log")"
exec python3 "$repo/tools/esp32c3_trace_diff.py" --trace "$trace" "$log"
