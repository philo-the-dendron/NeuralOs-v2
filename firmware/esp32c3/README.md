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
the capacity floor below, and builds this firmware with the graph in
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
<stem>: identical, <rows> rows
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

## The capacity floor

`stranger.sh` refuses, exit 2, a graph whose arrays exceed 65,536
bytes, `44·N + 6·S` (a neuron is 44 bytes, a synapse 6). It is a floor,
not a measured stack, provisional until check 8 of `docs/ROADMAP.md`
§ 0.1.0: the formula counts the arrays alone, and the number is the
QEMU harness's stack line, not the chip's. The script's header has the
rest.

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
