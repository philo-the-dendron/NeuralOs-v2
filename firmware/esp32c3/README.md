# neuralos-esp32c3

`neuralos-snn` on an ESP32-C3. What it runs, in order, is the module
note at the top of [`src/main.rs`](src/main.rs): timed burst arms, the
library's frozen traces replayed over serial, then a neuron that blinks
the LED. The board's record, every capture and pin, is
[`evidence/esp32c3-bringup/README.md`](../../evidence/esp32c3-bringup/README.md).

## Your graph on the chip: one command

```bash
firmware/esp32c3/stranger.sh build <graph.nir> [--steps 150]     # no board needed
firmware/esp32c3/stranger.sh run <graph.nir> [--port /dev/ttyACM0] [--seconds 20] [--steps 150]
```

`build` converts the graph (`neuralos-nir2json --sim-units --freeze`:
`--steps` steps of the graph's own drive, 150 unless given, 1 on every
input feature; `run` takes `--steps` too and replays that many), checks
the capacity bar below, and builds this firmware with the graph in
its slot (`build.rs`, `NEURALOS_GRAPH`) through `build.sh`: the path
remap, the personal-string gate, the ELF and `.text` shas, clippy. It
writes into `firmware/esp32c3/target/stranger/`: `<stem>.json` and its
sidecar, `<stem>.rs` (the arrays), `<stem>.trace` (the host's run of
them) and `<stem>.elf`. `run` does the same, flashes the ELF, captures
20 s from the reset into `<stem>.capture.log`, and diffs the capture
with `tools/esp32c3_trace_diff.py --trace <stem>.trace`. The verdict is
the diff's last line and its exit code:

```text
<case>: identical, <rows> rows     one line per case: the library's, then yours
<stem> (given): identical, <rows> rows
<N> cases, 0 red
```

N counts the library's plasticity-off cases and yours. Each must equal
its file row for row, the spikes and every membrane; a case missing,
seen twice or cut off is red. A capture is green only when the fragment
before the reset carries no header: a reboot mid-replay is red by
design, not a shorter compare, so run it again. The evidence README's
eighth entry holds a real capture.

The file stem names the module (a letter or `_` first, not a Rust
keyword). Two graphs with one stem share one scratch name: the last
build wins, and the trace beside it is always that build's.

## Capacity

`stranger.sh` refuses, exit 2, a graph whose arrays exceed the bar,
`44·N + 6·S ≤ 262,144` bytes (a neuron is 44 bytes, a synapse 6). The
arrays live on the stack, in `main`'s frame, one copy, and the bar is
a measurement: at both of its corners, the most neurons (one layer of
5,957) and the most synapses (two layers of 201, all to all between
them), the stack's high-water mark on the C3 leaves at least 16 KiB
free and the replay is bit for bit. The evidence README's ninth entry
holds each corner's frame, mark and ns per step; the worst case is the
neuron corner, several times slower than a 1 ms step, so a graph at
the bar replays exactly but not in real time. Every build prints its
own mark after the replays, `stack: <mark> of <total> bytes
high-water after the replays`. The script's header has the rest.

The corners regenerate from the bar through snnTorch's own exporter in
the repo's `.nirenv` (`tools/gen_snnTorch_corner.py`; its docstring
derives the sizes), and run with fewer rows in a longer window:

```bash
.nirenv/bin/python3 tools/gen_snnTorch_corner.py neurons 262144 firmware/esp32c3/target/stranger/neurons.nir
.nirenv/bin/python3 tools/gen_snnTorch_corner.py dense 262144 firmware/esp32c3/target/stranger/dense.nir
firmware/esp32c3/stranger.sh run firmware/esp32c3/target/stranger/neurons.nir --steps 20 --seconds 30
```

`--steps` and `--seconds` size together for a large graph. A row of
the capture costs a few bytes a neuron, more on a step where many fire
(their ids are printed), and the link streams rows at the rate the
ninth entry gives, so the window must hold the library's cases, the
arms and `--steps` of your rows; a capture cut off is red, never a
shorter compare.

## The default firmware

```bash
firmware/esp32c3/build.sh          # release, remapped, gated; prints the shas
firmware/esp32c3/build.sh clippy
espflash flash --port /dev/ttyACM0 firmware/esp32c3/target/riscv32imc-unknown-none-elf/release/neuralos-esp32c3
python3 tools/esp32c3_capture.py /dev/ttyACM0 20 board.log
python3 tools/esp32c3_trace_diff.py board.log
```

Needs, on the pinned toolchain, the `riscv32imc-unknown-none-elf`
target and the `llvm-tools` component; espflash 4 for the board.
