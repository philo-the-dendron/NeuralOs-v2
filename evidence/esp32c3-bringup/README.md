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
| `alpha6-remeasure.log` | the second board run, alpha.6 firmware, 20 s after reset (§ Second entry) |
| `burst-loop-alpha5.dis`, `burst-loop-alpha6.dis` | the timed burst loop of each flashed ELF, disassembled by address (§ The mechanism): the machine output the finding rests on; extraction and the one scrub documented there |
| `host-bench-alpha5.log`, `host-bench-alpha6.log` | the host spike-path bench, one run per tree (§ Host bench); rebuilt by `proofs/spike-path-bench/README.md` § Run |
| `board-r27-alpha6.log` | the third board run (ISA round 27, item 1, 2026-09-12): the alpha.6 spine and the firmware with two timed burst arms, free and pinned, each line named; 20 s after reset. Built by `build.sh` at 5c3e3ad, ELF `3df8a19b…`, `.text` pinned in § Rebuild + run. The figures and the reading: ISA round 27, item 1 |
| `burst-loops-r27-alpha6.dis` | both burst loops of that ELF, cut by address (free `0x42011e1e`–`0x420120ac`, pinned `0x420122d4`–`0x4201260e`) with the extraction and the scrub of § The mechanism: the listing ISA round 27's item 3 is decided by |
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

## Second entry: the alpha.6 firmware on the same board (2026-09-10)

Same SuperMini (MAC `70:af:09:07:f6:3c`, esp32c3 revision v0.4), same
port, same tool, 20 s after reset. The firmware crate is unchanged
since the first entry except its lock (the spine's `heapless` edge
gone, the spine at alpha.6) and one word in its description; same
profile (release, lto fat, one codegen unit), same clock config
(`CpuClock::max()`), same toolchain 1.92.0. ELF rebuilt on
`main@820d81a` with `--locked`: sha
`e26e174999804ce43163f5c2f1cf3113f480a9cdb84148bfc7fedc00da425ebb`,
190,372 B (first entry: `102af8c6…`, 190,564 B; the ISA round-22 text
names an intermediate `6e4c6366…` from the heapless-out commit, never
flashed). Log: `alpha6-remeasure.log`. It opens with two stale
pre-reset lines (the freshly flashed firmware was already running when
the capture tool reset the board; the first is spliced into a
bootloader line); every figure below is from after the `rst:` line.

| Measurement | First entry (alpha.5 spine) | This entry (alpha.6 spine) |
|---|---|---|
| Burst: 10,000 steps (decay + integrate), no delay | 15,012 µs → 1,501 ns/step | 28,425 µs → **2,842 ns/step** |
| Burst spikes | 147 | 147 |
| First spike, real-time loop | step 56, 56,331 µs | step 56, 56,330 µs |
| Sustained: spikes after the reset | 295 | 295 |
| Sustained rate, spikes / last stamp | 295 / 20.106432 s = 14.672/s | 295 / 20.095308 s = 14.680/s |
| Sustained rate, intervals / span | 294 / 20.050101 s = 14.663/s | 294 / 20.038978 s = 14.671/s |
| Inter-spike interval | mean 68,198 µs, sd 676, min 67,334, max 69,460; 67–69 loop steps | mean 68,160 µs, sd 731, min 67,333, max 69,517; 67–69 loop steps |
| Wall time per loop step | 1.0057 ms | 1.0054 ms |

Comparison: every real-time-loop figure reproduces the first entry
within one loop step or one microsecond; the burst spike count is
identical; the burst time is 28,425 µs against 15,012 µs, +13,413 µs
over 10,000 steps = +1,341 ns on every step, spiking or not (147 of
the 10,000 steps spike). The expectation stated before the run was a
sub-noise shift. It is not. The mechanism is read from the two flashed
binaries in the next section (ISA round-23), which also names what it
leaves open.

## The mechanism, read from the two flashed ELFs (2026-09-10)

Both ELFs that ran were still in the firmware's cargo target dir
(`firmware/esp32c3/target/riscv32imc-unknown-none-elf/release/deps/`,
as `neuralos_esp32c3-398b097a7ec0fca9` and
`neuralos_esp32c3-7d7a21b38e2c4ae8`) and hash to the two pins above,
so what follows is read from the binaries that produced the logs, not
from rebuilds. The hot-path source is byte-identical between the two
trees: the alpha.5 → alpha.6 diff of `lif_neuron.rs` touches the
history type and `spike()`, and nothing in `decay_adaptation_current`
or `integrate_and_fire`.

`main` was disassembled from each ELF and the timed burst loop cut out
by address, from the loop's first instruction to its exit target:
`burst-loop-alpha5.dis` (`0x42011d1c`–`0x42011e74`) and
`burst-loop-alpha6.dis` (`0x42011d0c`–`0x42011f66`). The counts:

| Burst loop body | alpha.5 ELF | alpha.6 ELF |
|---|---|---|
| Instructions | 115 | 195 |
| Calls to the ROM's `__divdi3` (64-bit software division) | 1 | 2 |
| Hardware `divu` | 0 | 1 |
| Loads and stores against the stack frame | 5 | 34 |

What the listings show. In the alpha.5 burst loop LLVM kept the whole
neuron in registers and folded `resistance_mohm` (100), the centi-mV
scale (100) and `tau_membrane_us` (20,000) to constants: the current
term's `× 100 × 100 / 1000` became `× 10` (`slli`/`add`, no
division), `dt_over_tau` became the constant 50 (`li a2, 0x32`), and
one 64-bit division by 1000 survived, the one in `delta_v`. In the
alpha.6 burst loop the neuron lives on the stack: resistance, scale
and tau are loaded every step (`lhu 0x17a(sp)`, `lbu 0x180(sp)`,
`lw 0x158(sp)`), both divisions by 1000 are calls to the ROM's
software 64-bit divide, `dt_over_tau` costs a hardware `divu` on top,
and every field is written back each step.

The fact that frames both numbers: the alpha.5 ELF's real-time loop
already has the alpha.6 shape, two `__divdi3` calls (`main` at
`0x420122f0` and `0x42012334`) and every field loaded from the frame.
So 1,501 ns was one loop the compiler happened to scalarize, and
2,842 ns is what the step costs on this core whenever the struct is
in memory, which is what a network holding a `Vec<LIFNeuron>` always
pays. The library did not get slower in the general case; one lucky
measurement became an unlucky one.

Verified here: the two loop bodies, the identical source, the pins.
Consistent but not measured: +1,341 ns is 215 cycles at 160 MHz, which
one extra ROM 64-bit division plus the `divu` plus the frame traffic
would account for; the split needs an instrumented loop on the board,
and the finding does not need the split. Not established: why LLVM
stopped scalarizing the struct. The visible suspects are the ring's
two variable-index stores and the bounds check on `buf[head]`, the
only panic edge inside the loop (the source path string
`crates/neuralos-snn/src/lif_neuron.rs` is in the alpha.6 ELF's
rodata and absent from alpha.5's). That is the first experiment of
the fix brief, and it needs no board. The host bench going the other
way (§ Host bench) is consistent: x86-64 divides 64-bit in hardware,
and the forced arm measured the `remove(0)` shift, which is gone.

Extraction, so the two files can be rebuilt from the ELFs (the
`llvm-tools` rustup component under the pinned toolchain):

```bash
LLVM=$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin
$LLVM/llvm-objdump -d --no-show-raw-insn --print-imm-hex \
    --start-address=0x42011d1c --stop-address=0x42011e74 <alpha.5 ELF> > burst-loop-alpha5.dis
$LLVM/llvm-objdump -d --no-show-raw-insn --print-imm-hex \
    --start-address=0x42011d0c --stop-address=0x42011f66 <alpha.6 ELF> > burst-loop-alpha6.dis
```

One scrub, on line 2 of each file: llvm-objdump prints the ELF's
absolute path there, and the local path was replaced by
`sha256:<the ELF's sha256>`. Nothing else was edited; no path or
personal string occurs in the listings themselves. The ELFs are not
committed (no binaries in the tree); until they are attached to the
Gitea release they exist only in the target dir named above, and no
rebuild reproduces their sha (§ Rebuild + run).

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
firmware/esp32c3/build.sh                    # since round 26: cargo build --release --locked under rustc's path remap, then the personal-string gate, the ELF and .text shas, the trim-paths canary (tools/remap.sh); the two pins below were built by the bare cargo line the script wraps
sha256sum firmware/esp32c3/target/riscv32imc-unknown-none-elf/release/neuralos-esp32c3
#   102af8c6374766f290777b4e760a60318a0496f387fbb6a3b534200f3ebd5aed  at 427b6cb (release profile, lto fat)
#   e26e174999804ce43163f5c2f1cf3113f480a9cdb84148bfc7fedc00da425ebb  at 820d81a (same profile; the alpha.6 spine, second entry)
espflash board-info --port /dev/ttyACM0
espflash flash --port /dev/ttyACM0 firmware/esp32c3/target/riscv32imc-unknown-none-elf/release/neuralos-esp32c3
python3 tools/esp32c3_capture.py /dev/ttyACM0 20 bringup.log
grep -a -c 'spike at' bringup.log            # 296 here: 295 + the pre-reset line
```

The ELF sha pins a build, not a source state. Two things move it
between builds of the same source under the same toolchain and lock,
both found on 2026-09-10 when a scratch rebuild of 427b6cb gave
`26be7a1e…` with a byte-identical `.text`: the ESP-IDF app descriptor
carries a build stamp (`esp-bootloader-esp-idf` 0.5.0's build script,
`SOURCE_DATE_EPOCH` or the wall clock; the alpha.5 ELF says
`2026-09-09 18:33:44`, the alpha.6 ELF linked on the 10th says
`2026-09-09 18:34:07` from the cached build script, the scratch
rebuild got `2026-09-10 22:51:51`), and the symbol-name hashes in
`.strtab` follow the clone path. The code section is what a rebuild
is compared against:

```bash
llvm-objcopy -O binary --only-section=.text <ELF> text.bin && sha256sum text.bin
#   d14cce25ec36906b0150ec5deaf9675dc199da4fbe39feb334bfddf2c9960af4  alpha.5 (102af8c6…)
#   d672ca37eb1ef67ba40b212904fa0cf0d4a331a68bcf7e76f8bc288617027f6b  alpha.6 (e26e1749…)
#   6d407501400cb9beb554b2f5df7ce031f9bae08c43a274bd41ef8df07bc0ddaa  alpha.6 under the round-26 remap (build.sh, PR #23's item-1 tree; the spine and firmware sources of 820d81a, unchanged)
#   f7193ca7415e2d8d8c006a8cff67c97a3f5e51e2965639d442a65293a8354821  round-27 baseline (build.sh at 5c3e3ad: the alpha.6 spine, the firmware with the free and pinned burst arms; ISA round 27, item 1)
```

The third pin is the second one rebuilt by `build.sh` (round 26,
2026-09-11): rustc's `--remap-path-prefix` shortens each of the 17
embedded path strings by 12 bytes (`/home/<user>/` → `~/`), `.rodata`
is 204 bytes shorter, and 182 of the 9,196 instructions in `.text`
differ, every one a `lui`/`addi` address immediate (`llvm-objdump -d`
of both, diffed; the one `mv` in the list is `addi` with a zero
immediate). No instruction was added, removed or reordered, so the
behavior figures of the second entry stand unmeasured. The two
earlier pins stand as built by the bare cargo line; they are not
reproduced by the script and are not expected to be. The fourth pin is
not a rebuild: 5c3e3ad changes the firmware's source (the pinned burst
arm, ISA round 27, item 1) on the same alpha.6 spine.

### Release asset (the procedure since round 26)

A firmware asset on a Gitea release is built by `build.sh` at the
tagged commit (the descriptor stamp is that commit's time), copied
from the target dir into `firmware/esp32c3/dist/` under the alpha.6
name pattern
`neuralos-esp32c3-<fw>-snn-<lib>-riscv32imc-unknown-none-elf.elf`,
listed in `dist/SHA256SUMS`, uploaded with that file, downloaded back,
checked with `sha256sum -c`, and gated once more on the download:

```bash
(source tools/remap.sh && remap_env >/dev/null && remap_gate <downloaded.elf>)
```

No scrub: the remap is what the alpha.6 release's § Scrub did by hand,
at compile time. The stamped release drafts under `docs/releases/`
describe the procedure of their day; this paragraph is the living one.

The alpha.5 pin reproduced byte-identical from a clone at another
path. The alpha.6 pin reproduces only from a clone at the same path:
the ring's bounds check embeds the source path in rodata, and a
rebuild of 820d81a from a 72-byte-longer path moved 45 `addi` address
immediates in `.text` by exactly 72 and nothing else (verified by
diffing the two `main` listings). That sentence describes the bare
line; under the script's remap the four source roots become fixed
aliases, so the third pin is expected from any clone path, home or
cargo home (the ELF sha still moves: the descriptor stamp is the
commit date, and the symbol-name hashes follow the package path). A
different ELF sha with the same log lines is not a finding; a
different `.text` with the same log lines is one to read. Reproducing
`boot-no-descriptor.log` means flashing an ELF built from 3525492 with
`--ignore-app-descriptor`; `factory-empty-flash.log` needs a blank
board (`espflash erase-flash` recreates the state).
