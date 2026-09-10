# ESP32-C3 bring-up — first spike on silicon (2026-09-09)

The evidence gate for ROADMAP step 5 and the round-18 open item: the
`firmware/esp32c3` LIF neuron ran on a real ESP32-C3, and the two
numbers round 18 refused to claim from the host (first spike, ns/step)
are read here from the board's own serial line.

Board: ESP32-C3 SuperMini (Amazon B0GQM4SRS3, 4-pack), identified from
the chip by `espflash board-info`, not from the listing: esp32c3
revision v0.4, 40 MHz crystal, 4 MB flash, secure boot and flash
encryption disabled, native USB-Serial-JTAG (`303a:1001`, `/dev/ttyACM0`,
no bridge chip), MAC `70:af:09:07:f6:3c`. Blue LED on GPIO8 (active
low), red LED is power. Host: the laptop, espflash 4.5.0, Rust 1.92.0
(the pin), esp-hal 1.1.2, esp-bootloader-esp-idf 0.5.0.

## Files

| File | What |
|---|---|
| `factory-empty-flash.log` | 8 s of the board as shipped: the ROM's `invalid header: 0xffffffff` loop. The flash is blank; the ROM finds no image and the watchdog resets it, which re-enumerated USB every ~2.6 s: 66 enumerations from plug-in (14:09:16) to the `board-info` that parked it in the bootloader (14:12:09), `kernel-usb.log`. Not a defect. |
| `boot-no-descriptor.log` | 15 s after flashing the `main@3525492` ELF with `--ignore-app-descriptor`: the ESP-IDF v5.5.1 second-stage bootloader loads the image, reads garbage where the app descriptor should be, and refuses the partition, 132 times in the window. Not one firmware line. The blocking finding, fixed by 427b6cb. |
| `bringup.log` | **The banked run.** Fixed ELF (sha below) flashed without the flag; one reset, then 20 s: ROM `rst:` line, bootloader `Loaded app`, the firmware banner, the burst line, 295 spike lines. |
| `kernel-usb.log` | the host kernel's USB lines for the board's port over the plug-in window, hostname stripped: 66 `New USB device found` / 65 `USB disconnect` for `303a:1001`, then none once flashed |
| `host-bench-alpha5.log`, `host-bench-alpha6.log` | the host spike-path bench, one run per tree (§ Host bench); rebuilt by `proofs/spike-path-bench/README.md` § Run |
| `SHA256SUMS` | pins the logs and this README |
| `../../tools/esp32c3_capture.py` | the capture tool (reset + read from one process) |
| `../../firmware/esp32c3/` | the firmware crate; the ELF is rebuildable, not committed |
| `../../proofs/spike-path-bench/` | the host bench harness (standalone crate) |

## The numbers (all from `bringup.log`, nothing rounded before division)

```
rst:0x15 (USB_UART_CHIP_RESET),boot:0xd (SPI_FAST_FLASH_BOOT)
I (125) boot: Loaded app from partition at offset 0x10000
neuralos-esp32c3: LIF neuron, centi-mV grid, dt=1000 us, input=160 uA
burst: 10000 steps in 15012 us -> 1501 ns/step, 147 spikes
spike at 56331 us (step 56)
```

| Measurement | Board | Host, round 18 |
|---|---|---|
| Burst: 10,000 steps (decay + integrate), no delay | 15,012 µs → **1,501 ns/step** | none; the first board number |
| Burst spikes | 147 in 10,000 simulated ms = **14.7/s** | 14.7/s (exact) |
| First spike, real-time loop | **step 56**, 56,331 µs | step 55 |
| Sustained: spikes in the log after the reset | **295** (raw `grep -c` says 296: one stale pre-reset line, see below) | |
| Sustained rate, spikes / last stamp | 295 / 20.106432 s = **14.672/s** | 14.7/s |
| Sustained rate, intervals / span | 294 / 20.050101 s = **14.663/s** | |
| Inter-spike interval | mean 68,198 µs, sd 676, min 67,334, max 69,460; 67–69 loop steps | |
| Wall time per loop step | 20,106,432 µs / 19,993 steps = **1.0057 ms** | 1 ms nominal |

Step 56 vs the host's 55: the loop stamps each step with wall time and a
loop step costs ~5.7 µs over its nominal millisecond, so simulated time
runs ~0.6 % slow against the step counter; the burst, which advances
simulated time exactly `DT_US` per step, matches the host to the spike.

The pre-reset line: `bringup.log` opens with one `spike at 44753625 us
(step 44534)` from the previous boot, emitted between the tool's input
flush and the reset edge (~0.3 s). The banked window starts at the
`rst:` line; every count above is taken from that line onward.

USB timing: the firmware banner prints within ~1 ms of boot and arrived
over USB. esp-println's `auto` backend routes to USB-Serial-JTAG only
after seeing a USB start-of-frame, and after this reset (the link
survives it, no re-enumeration) the flag is already set. So the
pre-board concern that the banner might go to UART0 did not occur on a
software reset. A cold plug-in with no host attached is untested and
moot: nothing listens then.

Capture discipline (two runs lost the boot lines before the third got
them): the reader and the resetter must be one process, and the input
flush goes before the reset, never after. Both rules are in the tool.

Not measured here, and not claimed: power, WiFi/BLE (unused), anything
about the network layer (`network` is std-gated; one neuron only).

## Host bench: the spike path, alpha.5 vs alpha.6 (2026-09-10)

The question: the alpha.6 ring replaced `heapless::Vec` in the spike
history (`remove(0)`, a 64-entry shift on every spike past the 64th,
became an O(1) overwrite). Did it show up in the per-step cost? One
harness, `proofs/spike-path-bench/`, public API only, run against two
trees by the procedure in its README: the alpha.6 tree (`main@820d81a`)
and a worktree at `371e8c6` (the alpha.5 tree). Two arms: `forced`
(threshold at the membrane floor, refractory 0, constant input: every
step fires and records a spike) and `control` (no input: no step
fires, the same loop minus the spike path). 10 repetitions of
10,000,000 steps each, release profile, fat LTO, one codegen unit.

Box: Intel i5-6200U (2 cores, 4 threads), Linux 6.8, cpufreq governor
`powersave`, nothing pinned, other processes present. The logs carry
every repetition; the table carries min and median per arm. The
alpha.5 log opens with the `cargo tree` line naming the worktree's
spine; its absolute path was made repo-relative before banking (the
burn-script convention, no home path in the record).

| Arm | Tree | min ns/step | median ns/step | spikes / 10^7 steps | Log |
|---|---|---|---|---|---|
| forced | alpha.5 (`371e8c6`) | 17.125 | 18.092 | 10,000,000 | `host-bench-alpha5.log` |
| forced | alpha.6 (`820d81a`) | 12.508 | 13.809 | 10,000,000 | `host-bench-alpha6.log` |
| control | alpha.5 (`371e8c6`) | 10.661 | 11.412 | 0 | `host-bench-alpha5.log` |
| control | alpha.6 (`820d81a`) | 10.442 | 12.194 | 0 | `host-bench-alpha6.log` |

Comparison, medians: forced 18.092 → 13.809 (−4.283 ns/step);
control 11.412 → 12.194 (+0.782, inside the repetition spread of both
runs: control repetitions range 10.661–12.615 on alpha.5 and
10.442–12.594 on alpha.6). Forced minus control, same tree: 6.680
ns/step on alpha.5, 1.615 on alpha.6.

## Rebuild + run (from the repo root; board on /dev/ttyACM0)

```bash
(cd firmware/esp32c3 && cargo build --release --locked)
sha256sum firmware/esp32c3/target/riscv32imc-unknown-none-elf/release/neuralos-esp32c3
#   102af8c6374766f290777b4e760a60318a0496f387fbb6a3b534200f3ebd5aed  at 427b6cb (release profile, lto fat)
espflash board-info --port /dev/ttyACM0
espflash flash --port /dev/ttyACM0 firmware/esp32c3/target/riscv32imc-unknown-none-elf/release/neuralos-esp32c3
python3 tools/esp32c3_capture.py /dev/ttyACM0 20 bringup.log
grep -a -c 'spike at' bringup.log            # 296 here: 295 + the pre-reset line
```

The ELF sha is reproducible only under the same toolchain and lock; a
different sha with the same log lines is not a finding. Reproducing
`boot-no-descriptor.log` means flashing an ELF built from 3525492 with
`--ignore-app-descriptor`; `factory-empty-flash.log` needs a blank
board (`espflash erase-flash` recreates the state).
