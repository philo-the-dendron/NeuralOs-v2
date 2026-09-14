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
(the pin then), esp-hal 1.1.2, esp-bootloader-esp-idf 0.5.0.

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
| `board-r27-alpha6.log`, `board-r27-alpha7.log` | the third entry (ISA round 27, 2026-09-12): the baseline on the alpha.6 spine (built at 5c3e3ad, ELF `3df8a19b…`) and the fix on the alpha.7 spine (built at 1affcd6, ELF `d4d0239a…`), the firmware with two timed burst arms, free and pinned, each line named; 20 s after reset; `.text` pins in § Rebuild + run (§ Third entry) |
| `burst-loops-r27-alpha6.dis`, `burst-loops-r27-alpha7.dis` | both burst loops of each ELF, cut by address with the extraction and the scrub of § The mechanism (baseline: free `0x42011e1e`–`0x420120ac`, pinned `0x420122d4`–`0x4201260e`; fix: free `0x42011dca`–`0x42011f56`, pinned `0x420121b4`–`0x42012590`); the baseline's is the listing ISA round 27's item 3 was decided by (§ Third entry) |
| `host-bench-r27-alpha6.log`, `host-bench-r27-alpha7.log` | the host spike-path bench, round 27's regression check, one run per tree, each opening with the line that names its commit and spine version (§ Host bench, round 27) |
| `host-bisect-r27.log` | the follow-up to that check: the bench per commit of the round, medians per round (§ Host bench, round 27) |
| `board-r30-alpha7-before.log`, `board-r30-pin.log` | the fourth entry (ISA round 30, 2026-09-13): the board as round 27 left it (the 1.92.0 build on flash), then the same source built by `build.sh` on 1.98.1 at 885ff53 (ELF `971b2bcb…`), 20 s after reset each (§ Fourth entry) |
| `host-bench-r30-2x2.log` | round 30's rider: the host spike-path bench as a 2×2, the alpha.6 tree (7fe2388) and the alpha.7 tree (26d68b0), each built on 1.92.0 and 1.98.1, three rounds in rotated order, the per-round medians appended by the run itself (§ Host bench, round 30) |
| `board-r31-recorder.log` | the fifth entry (ISA round 31, 2026-09-13): the neuron without its spike ring, the firmware source unchanged, built by `build.sh` on 1.98.1 at d3adc5b (ELF `dc12d19a…`), 20 s after reset (§ Fifth entry) |
| `burst-loops-r31-recorder.dis` | both burst loops of that ELF, cut by address as in § Third entry (free `0x42011c52`–`0x42011ddc`, pinned `0x42011fd6`–`0x42012358`); no ring store in either (§ Fifth entry) |
| `board-r32-network.log` | the sixth entry (ISA round 32, 2026-09-14): the firmware with the network arm and the twelve replays of the frozen traces, built by `build.sh` on 1.98.1 from that entry's sources (ELF `ab7612df…`), 20 s after reset (§ Sixth entry) |
| `board-r33-alpha8.log` | the seventh entry (ISA round 33, 2026-09-14): the sixth entry's firmware source under the version `0.1.0-alpha.8`, built by `build.sh` on 1.98.1 at 6ddf774 (ELF `a7be3286…`), 20 s after reset (§ Seventh entry) |
| `host-bench-r33-alpha8.log` | round 33's rider: the host spike-path bench, the alpha.7 tree (a974b9e) against the alpha.8 tree (6ddf774), both built on 1.98.1, three rounds in rotated order, the per-round medians appended by the run itself (§ Host bench, round 33) |
| `SHA256SUMS` | pins the logs and this README |
| `../../tools/esp32c3_capture.py` | the capture tool (reset + read from one process) |
| `../../tools/esp32c3_trace_diff.py` | the replay diff: every plasticity-off trace of `crates/neuralos-snn/tests/traces/` against its case in a capture, one line per case (§ Sixth entry) |
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
measurement became an unlucky one. [Round 27, 2026-09-12: this holds
under the firmware's constant dt; with dt at run time, as a network
passes `time_step_us`, the alpha.6 step costs 3,878 ns (§ Third
entry).]

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

## Third entry: round 27, the divisions, two burst arms (2026-09-12)

Same SuperMini (MAC `70:af:09:07:f6:3c`, esp32c3 revision v0.4), same
port, same tool, 20 s after reset; both ELFs built by `build.sh`
(§ Rebuild + run holds the pins). The firmware gained a second timed
burst beside the first (ISA round 27, item 1). The free arm is the
burst of the first two entries: the neuron and the constant `DT_US` in
plain sight of the optimizer. The pinned arm steps a fresh neuron with
the same id through `core::hint::black_box(&mut n)` once per step,
with dt read once through `black_box(DT_US)`, as a network holds a
neuron: in memory, dt at run time. Both arms print ns/step, spikes,
the first spike step and a checksum of the spike steps (a wrapping
`× 31 + i` fold). Baseline: the alpha.6 spine, firmware at `5c3e3ad`,
ELF `3df8a19b…`, `board-r27-alpha6.log`. Fix: the alpha.7 spine (items
2 to 4, the inline hint, the bump), the same firmware source, built at
`1affcd6`, ELF `d4d0239a…`, `board-r27-alpha7.log`. Each log opens
with stale pre-reset spike lines (two in the baseline, one in the
fix); every figure below is from after the `rst:` line.

| Measurement | Baseline (alpha.6 spine) | Fix (alpha.7 spine) |
|---|---|---|
| Burst, pinned arm (the cost a network pays) | 38,789 µs → **3,878 ns/step** | 15,797 µs → **1,579 ns/step** |
| Burst, free arm | 28,633 µs → 2,863 ns/step | 5,636 µs → **563 ns/step** |
| Burst spikes, both arms | 147 | 147 |
| First spike step, both arms | 55 | 55 |
| Checksum of the spike steps, both arms | `0b78b456` | `0b78b456` |
| First spike, real-time loop | step 56, 56,332 µs | step 56, 56,241 µs |
| Sustained: spikes after the reset | 294 | 295 |
| Sustained rate, spikes / last stamp | 294 / 20.026913 s = 14.680/s | 295 / 20.066479 s = 14.701/s |
| Sustained rate, intervals / span | 293 / 19.970581 s = 14.672/s | 294 / 20.010238 s = 14.692/s |
| Inter-spike interval (sample sd) | mean 68,159 µs, sd 678, min 67,333, max 70,363; 67–70 loop steps | mean 68,062 µs, sd 635, min 67,201, max 69,362; 67–69 loop steps |
| Wall time per loop step | 20,026,913 µs / 19,917 = 1.0055 ms | 20,066,479 µs / 19,993 = 1.0037 ms |

Comparison: the pinned burst −2,299 ns/step (−59.3 %), the free burst
−2,300 ns/step (−80.3 %); spike count, first spike step and checksum
identical in both arms on both spines, and on the alpha.6 spine
identical to the x86-64 host replay (the item-1 review, ISA round 27).
The real-time loop is reported, not gated: it stamps each step with
the wall clock and the noise is seeded by the stamp, so a spike can
move by a step; its first spike stays at step 56. Across rounds the
comparison is approximate: the first two entries had no tally code
(ISA round 27, record-only (d)).

The listings, `burst-loops-r27-alpha6.dis` and
`burst-loops-r27-alpha7.dis`: each loop cut from its flashed ELF by
address, from its first instruction to its exit target, with § The
mechanism's extraction; counted by the script that reproduces that
section's 195 and 34.

| Burst loop body | Free, baseline | Free, fix | Pinned, baseline | Pinned, fix |
|---|---|---|---|---|
| Range | `0x42011e1e`–`0x420120ac` | `0x42011dca`–`0x42011f56` | `0x420122d4`–`0x4201260e` | `0x420121b4`–`0x42012590` |
| Instructions | 214 | 132 | 268 | 320 |
| ROM `__divdi3` calls | 2 | 0 | 2 | 0 |
| ROM `__udivdi3` calls | 0 | 0 | 1 | 0 |
| Hardware `divu` | 1 | 0 | 0 | 1 |
| Calls to the cold fallbacks | — | 1 | — | 3 |
| Loads and stores against the frame | 42 | 8 | 15 | 19 |
| Loads and stores through the neuron pointer | — | — | 33 | 31 |
| Ring stores (computed index) | 2 | 0 | 2 | 2 |
| Branches out, besides the exit | 0 | 0 | 1 (`panic_bounds_check`) | 0 |

The reading. The pinned loop has no ROM division left, and its only
calls are to the cold fallbacks, each behind its guard
(`div_1000_wide` twice, `dt_over_tau_wide` once): both `/ 1000` are a
multiply-high by the constant, `dt_over_tau` is one hardware `divu`,
and the ring mask removed the bounds check. The loop is longer (320
instructions against 268) because the arithmetic that sat behind three
calls is now inline. −2,299 ns is 368 cycles per step at 160 MHz,
which three ROM divisions per step would account for; the split per
call is not measured, and the finding does not need it. The free loop
is back in registers: 132 instructions, dt/τ folded to 50 by the
constant, its frame traffic two range-check constants reloaded on each
integrating step plus the save and restore around the cold call. Its
563 ns/step is below the 1,501 of alpha.5's register loop, which still
called the ROM once per step; the free fix loop also has no ring store
(the firmware never reads the spike history, and with the neuron in
registers the stores are gone), where alpha.5's loop stored the ring
once per spike and called `memmove`, the heapless shift, so the gap is
not the ROM call alone. The tally's three counts, spilled at the
baseline (ISA round 27, item 1), are in registers in both fix loops;
the only stores to them are around the cold calls. The inline hint on
`integrate_and_fire` (ISA round 27, a deviation from the brief's list,
the principal's call) keeps the step inlined in the pinned loop;
without it that loop called the step out of line every step.

§ The mechanism's reading, that 2,842 ns is what a network holding a
`Vec<LIFNeuron>` always pays, holds for the struct in memory under the
firmware's constant dt. With dt at run time, as a network passes
`time_step_us`, the alpha.6 step costs 3,878 ns (the pinned baseline
above); that section stands as written on 2026-09-10.

## Host bench, round 27: the alpha.6 tree against alpha.7 (2026-09-12)

The same harness as § Host bench, now as round 27's regression check
(ISA round 27, item 6): the alpha.7 tree must not be slower on either
arm. Same box as that section (Intel i5-6200U, 2 cores, 4 threads,
Linux 6.8, governor `powersave`, nothing pinned), same session, back to
back, both trees built first: a worktree at `7fe2388` under
`bench-rebuild/` (the PR #23 merge commit, the alpha.6 spine; it
carries the bench and its lock, so nothing is copied) and this tree at
`2c3ce09` (after the bump, item 8a). Each log opens with a line naming
its commit and the spine version read from that tree's bench lock,
written by the command through the same `tee`, never onto a pinned
name:

```bash
git worktree add --detach bench-rebuild/r27-alpha6 7fe2388
bench() {   # $1 = tree, $2 = log name; both trees built first with cargo build --release --locked
  ( printf 'tree %s, neuralos-snn %s\n' \
      "$(git -C "$1" rev-parse --short HEAD)" \
      "$(awk '$0=="name = \"neuralos-snn\""{getline; gsub(/version = |"/,""); print}' "$1/proofs/spike-path-bench/Cargo.lock")"
    cd "$1/proofs/spike-path-bench" && cargo run --release --locked
  ) | tee "evidence/esp32c3-bringup/$2"
}
bench bench-rebuild/r27-alpha6 host-bench-r27-alpha6.log
bench . host-bench-r27-alpha7.log
git worktree remove --force bench-rebuild/r27-alpha6
```

| Arm | Tree | min ns/step | median ns/step | Log |
|---|---|---|---|---|
| forced | alpha.6 (`7fe2388`) | 12.103 | 13.084 | `host-bench-r27-alpha6.log` |
| forced | alpha.7 (`2c3ce09`) | 14.511 | 15.131 | `host-bench-r27-alpha7.log` |
| control | alpha.6 (`7fe2388`) | 10.559 | 11.364 | `host-bench-r27-alpha6.log` |
| control | alpha.7 (`2c3ce09`) | 12.900 | 13.848 | `host-bench-r27-alpha7.log` |

The check fails. alpha.7 is slower on both arms: medians +2.047
ns/step forced and +2.484 control. The repetition ranges overlap by
0.015 ns on the forced arm (alpha.6 up to 14.526, alpha.7 from 14.511)
and not at all on the control arm (up to 12.359, from 12.900).

The follow-up, from scratch builds and not part of the check: the
round's commits bisected. Each tree is a `git archive` of
`Cargo.toml`, `rust-toolchain.toml`, `crates/` and
`proofs/spike-path-bench/`, built with `cargo build --release --locked
--offline`; the seven binaries ran in three rounds in rotated order,
two of them the two ends rebuilt with `RUSTFLAGS="-C
llvm-args=-x86-branches-within-32B-boundaries"` as a control for
Skylake's jump-alignment erratum. Summary: `host-bisect-r27.log`. Each
figure below is the median of the three rounds' medians, the share in
brackets; the last column is the loop that holds the step's one
division, read from each binary's `objdump -d` of the bench's `run`:

| Tree (commit) | forced | control | loop: instructions / memory operands |
|---|---|---|---|
| alpha.6 (`7fe2388`) | 11.156 | 9.778 | 94 / 19 |
| + item 2, `div_1000` (`822beb3`) | 11.697 (+0.541) | 10.611 (+0.833) | 145 / 39 |
| + item 3, the `dt_over_tau` guard (`f2007a7`) | 11.724 (+0.027) | 10.692 (+0.081) | 145 / 39 |
| + item 4, the ring mask (`91d92a9`) | 13.051 (+1.327) | 11.547 (+0.855) | 107 / 23 |
| + the inline hint (`3f252d2`, the alpha.7 code) | 13.671 (+0.620) | 12.282 (+0.735) | 111 / 24 |
| alpha.6, jump-aligned | 10.858 | 9.831 | |
| alpha.7 code, jump-aligned | 13.553 | 12.108 | |

The aligned builds keep the gap (+2.695 forced, +2.277 control), so
the cost is code, not branch layout. In every one of these binaries
the whole step is inlined into the bench loop; only `div_1000_wide`
stays out of line from item 2 on, and the `dt_over_tau` guard folds
away (the bench's dt is a constant). Item 2 is added work: its range
checks sit where x86-64 had no division to save (the `i64 / 1000` was
already a multiply-high), and the loop grows from 94 instructions to
145. Item 4 shrinks the loop to 107, the scalarization it also
produces on the chip, and the smaller loop runs slower; why is not
established (a listing gives size, not the critical path). The hint
changes the layout of a loop that was already inlined. Absolute
figures move between runs on this laptop (alpha.6 medians from 11.1 to
13.1 ns); only comparisons within one run are read.

Ruling (the principal, 2026-09-12): recorded, not fixed. The chip is
the target and item 7 decides the round; the alpha.7 notes state this
host cost. The two options not taken: limiting item 2's fast path to
32-bit targets (it would recover item 2's share only), and a deeper
look at the x86 code first.

## Fourth entry: round 30, the pin moves to 1.98.1 (2026-09-13)

Same SuperMini (MAC `70:af:09:07:f6:3c`, esp32c3 revision v0.4), same
port, same tool, 20 s after reset. The toolchain pin moved from 1.92.0
to 1.98.1 (PR C); no spine or firmware source changed since round 27's
fix. Before: the board as round 27 left it, its flash holding the fix
built on 1.92.0, captured the same afternoon,
`board-r30-alpha7-before.log`. After: `build.sh` at `885ff53` on
1.98.1, ELF `971b2bcb…`, `.text` `33656de9…` (§ Rebuild + run),
flashed, `board-r30-pin.log`. The gate: 0 hits. The trim-paths canary
on 1.98.1: still unstable (exit 101); the remap stays. Each log opens
with stale pre-reset spike lines (two before, one after); every figure
below is from after the `rst:` line.

| Measurement | Before (1.92.0 build) | After (1.98.1 build) |
|---|---|---|
| Burst, pinned arm | 15,798 µs → 1,579 ns/step | 15,856 µs → **1,585 ns/step** |
| Burst, free arm | 5,636 µs → 563 ns/step | 5,694 µs → **569 ns/step** |
| Burst spikes, first spike step, checksum, both arms | 147, 55, `0b78b456` | 147, 55, `0b78b456` |
| First spike, real-time loop | step 56, 56,241 µs | step 59, 59,276 µs |
| Spikes after the reset | 295 | 295 |

The gate holds: both burst arms identical in spike count, first spike
step and checksum, the figures the traces compare also pins on the
host (`one-neuron-board.trace`). The new compiler costs 58 µs per
10,000 steps on both arms, 5.8 ns per step (+0.4 % pinned, +1.0 %
free), about one cycle at 160 MHz; not read further. The real-time
loop is reported, not gated: its noise is seeded by the wall-clock
stamp, and the two bursts before it now take 116 µs longer, so every
stamp moves; its first spike lands at step 59, three steps after round
27's step 56, with the same 295 spikes in 20 s.

The `.text` moved with the compiler alone: at `ff15c8e` (the same
sources) `build.sh` in the main clone gives `0de3bd91…` on 1.92.0 and
`33656de9…` on 1.98.1. One correction to § Rebuild + run, which
expects the remapped pin "from any clone path": that does not hold.
The per-commit loop's scratch worktree built `361dc540…` from the same
commit on 1.92.0, every section the same size, the jump-table labels
renumbered (the functions in another order), and a second worktree at
another path (`bench-rebuild/c0`, PR #27's reproduction of its CI red)
built the same `361dc540…`. In the main clone, moving the target dir or
setting CI's `RUSTFLAGS` left `0de3bd91…`. So worktree builds differ
from the main clone's, not one path from another; the codegen-unit name
differs, so the crate hash moved, by a cause not isolated here. A
`.text` pin reproduces from the checkout it was built in. Record-only,
not fixed in PR C.

## Host bench, round 30: tree × compiler (2026-09-13)

Round 27's regression check once more, as PR C's rider: does the
alpha.7 tree's x86-64 cost (§ Host bench, round 27) survive the new
pin? Same box (Intel i5-6200U, governor `powersave`, nothing pinned),
round 27's bisect method: `git archive` trees of `Cargo.toml`,
`rust-toolchain.toml`, `crates/` and `proofs/spike-path-bench/` for the
alpha.6 tree (`7fe2388`) and the alpha.7 tree (`26d68b0`), each built
with `cargo +<toolchain> build --release --locked --offline` on 1.92.0
and on 1.98.1; the four binaries built first, then three rounds in
rotated order in one run, `host-bench-r30-2x2.log` (the per-round
medians appended by the run itself). Absolute figures move between
runs on this laptop; only this run's comparisons are read. Each figure
is the median of the three rounds' medians, ns/step:

| Arm | Tree | 1.92.0 | 1.98.1 | Compiler |
|---|---|---|---|---|
| forced | alpha.6 | 13.020 | 12.784 | −0.236 |
| forced | alpha.7 | 15.000 | 15.274 | +0.274 |
| control | alpha.6 | 11.041 | 10.567 | −0.474 |
| control | alpha.7 | 13.546 | 13.209 | −0.337 |
| **alpha.7 − alpha.6, forced** | | +1.980 | **+2.490** | |
| **alpha.7 − alpha.6, control** | | +2.505 | **+2.642** | |

The cost stays on the new pin: the alpha.7 tree is 2.0 to 2.6 ns/step
slower on both compilers, and no round of one tree reaches a round of
the other in any cell. The compiler moves each cell by less than
0.5 ns, in both directions: its rounds overlap in two cells, touch in
one (alpha.6 forced) and separate only on the alpha.6 control arm
(−0.47). The cost is in the code, as round 27's bisect found, not in
the compiler; round 27's ruling stands (recorded, not fixed).

## Fifth entry: round 31, the ring leaves the neuron (2026-09-13)

Same SuperMini (MAC `70:af:09:07:f6:3c`, esp32c3 revision v0.4), same
port, same tool, 20 s after reset. The spine's neuron keeps no spike
history since PR D (ISA round 31): the ring became `SpikeRecorder`,
which the firmware does not use, and the firmware source did not
change. `build.sh` at `d3adc5b` on 1.98.1 in the main clone: ELF
`dc12d19a…`, `.text` `61639145…` (§ Rebuild + run), the gate 0 hits,
the trim-paths canary still unstable on 1.98.1 (exit 101). Flashed
from a copy of that ELF, `board-r31-recorder.log`; it opens with two
stale pre-reset spike lines, the first spliced into a bootloader line,
and every figure below is from after the `rst:` line. The comparison
is § Fourth entry's after column: the same board, compiler and
firmware source, the same day.

| Measurement | Round 30 (the ring in the neuron) | Round 31 (no ring) |
|---|---|---|
| Burst, pinned arm | 15,856 µs → 1,585 ns/step | 15,595 µs → **1,559 ns/step** |
| Burst, free arm | 5,694 µs → 569 ns/step | 5,704 µs → **570 ns/step** |
| Burst spikes, first spike step, checksum, both arms | 147, 55, `0b78b456` | 147, 55, `0b78b456` |
| First spike, real-time loop | step 59, 59,276 µs | step 55, 55,161 µs |
| Spikes after the reset | 295 | 295 |

The gate holds: both burst arms identical in spike count, first spike
step and checksum, the figures `one-neuron-board.trace` pins on the
host. The pinned arm is 261 µs faster per 10,000 steps, 26 ns per step
(−1.6 %), about four cycles at 160 MHz; the free arm is 10 µs slower,
1 ns per step (+0.2 %). The same ELF measured twice moved by at most
1 µs (round 27's fix, § Third entry, against § Fourth entry's before
column), so both moves are the build's; neither is read further. The
real-time loop is reported, not gated: its noise is seeded by the
wall-clock stamp, and the bursts before it now end at other times; its
first spike lands at step 55, with the same 295 spikes in 20 s.

The listing, `burst-loops-r31-recorder.dis`: both loops of the flashed
ELF, each cut by address from its first instruction to its exit
target with § The mechanism's `llvm-objdump` line (LLVM 22.1.8, from
1.98.1's `llvm-tools`). Line 2 of each half names the ELF by its sha
because the copy it was cut from was named `sha256:<sha>`; nothing was
scrubbed. The counts follow rules that reproduce every figure of
§ The mechanism's table and of § Third entry's listing table from
their pinned files: a load or store based on `sp` is a frame access;
in the pinned arm the base register of most other loads and stores is
the neuron pointer (`s9` and `s11` there, `s3` here); a store based on
neither is a computed-index store; a branch out targets neither the
range nor the exit.

| Burst loop body | Free, round 31 | Pinned, round 31 |
|---|---|---|
| Range | `0x42011c52`–`0x42011ddc` | `0x42011fd6`–`0x42012358` |
| Instructions | 132 | 288 |
| ROM `__divdi3` calls | 0 | 0 |
| ROM `__udivdi3` calls | 0 | 0 |
| Hardware `divu` | 0 | 1 |
| Calls to the cold fallbacks | 1 | 3 |
| Loads and stores against the frame | 9 | 12 |
| Loads and stores through the neuron pointer | — | 27 |
| Ring stores (computed index) | 0 | 0 |
| Branches out, besides the exit | 0 | 0 |

The reading. Neither loop has a computed-index store: every store
through the pinned arm's pointer is at a fixed offset of the struct,
where § Third entry's fix loop stored the ring twice on a spike step.
Against that fix (built on 1.92.0) the pinned loop is 288 instructions
against 320, with 12 frame and 27 pointer accesses against 19 and 31;
the compiler moved in between (§ Fourth entry) and round 30's loops on
1.98.1 were never cut, so those differences are not the ring's alone,
and the 26 ns/step, measured on one compiler, has no listing to split
it.

## Sixth entry: round 32, the network on the chip (2026-09-14)

Same SuperMini (MAC `70:af:09:07:f6:3c`, esp32c3 revision v0.4), same
port, same tool, 20 s after reset. The spine gained `FixedNetwork`
(PR E, ISA round 32), and the firmware two parts after its neuron
arms: a third timed arm, the network arm, which steps the frozen
`feedforward-8` (8 neurons, 6 synapses, from
`crates/neuralos-snn/tests/traces/frozen.rs`) 10,000 times on its
constant drive, behind `black_box` once per step as the pinned neuron
is; then the replays, every frozen case printing its trace's header
line and its rows through the library's `row.rs`, and one end line,
`# neuralos-trace end`. `build.sh` in the main clone on 1.98.1, with
the sources of this entry's commit before it was made: ELF
`ab7612df…`, its descriptor stamped with the time of `550fb48`, the
head then (a build at this entry's commit gives another ELF sha, and
the same `.text`), `.text` `61639145…` → `b5351531…` (§ Rebuild +
run), the gate 0 hits, the trim-paths canary still unstable on 1.98.1
(exit 101). Flashed from a
copy of that ELF, `board-r32-network.log`, 47,728 bytes. The capture
holds two boots: it opens with the tail of the boot that followed the
flash (a bootloader line spliced into another, the banner and both
neuron arms), cut by the capture's reset during that boot's network
arm, then the full run. Every figure below is from after the `rst:`
line.

| Measurement | Round 31 | Round 32 |
|---|---|---|
| Burst, pinned arm | 15,595 µs → 1,559 ns/step | 15,774 µs → **1,577 ns/step** |
| Burst, free arm | 5,704 µs → 570 ns/step | 3,676 µs → 367 ns/step |
| Burst spikes, first spike step, checksum, both neuron arms | 147, 55, `0b78b456` | 147, 55, `0b78b456` |
| Burst, network arm (8 neurons, 6 synapses a step) | none | 140,586 µs → **14,058 ns/step** |
| Network arm: spikes, first spike step, checksum | none | **3,377, 5, `7e700ee1`** |
| First spike, real-time loop | step 55, 55,161 µs | step 55, 55,149 µs |
| Spikes after the reset | 295 | 290 |

The gates hold, at this entry's commit. Both neuron arms keep the
fold. The network arm's three numbers are the pin `tests/traces.rs`
holds on the host (`the_network_arm_folds_to_its_pin`). And
`python3 tools/esp32c3_trace_diff.py evidence/esp32c3-bringup/board-r32-network.log`,
which reads the tree's trace files, finds each of the twelve headers
once and the end line after them, and prints:

```text
centi-mv-grid: identical, 30 rows
chain-3: identical, 100 rows
feedforward-8: identical, 150 rows
inhibitory: identical, 100 rows
neuron-reference: identical, 150 rows
nir-chain-fixture: identical, 100 rows
one-neuron-board: identical, 147 rows
plasticity-off: identical, 150 rows
recurrent-transmission: identical, 30 rows
refractory: identical, 40 rows
saturation-floor-ceiling: identical, 100 rows
ternary-weights: identical, 100 rows
12 cases, 0 red
```

So every plasticity-off trace of `tests/traces/` replays bit for bit
on the chip, the spikes and every membrane, row for row, from the
arrays the host steps; `plasticity-on` is outside `FixedNetwork` by
design. The network arm costs 14,058 ns per step of the whole network,
its spike tally included. The pinned neuron arm, the comparable
figure, moved 1,559 → 1,577 ns/step (+18, +1.2 %). The free arm moved
570 → 367 with its firmware source unchanged; the one spine change in
the neuron arms' path is PR E's `dt_over_tau`, which divides by a
`NonZero` now, behavior identical. The move is recorded and not
explained, since there is no listing this round. The real-time loop is
reported, not gated: its first spike lands at step 55, and it counts
290 spikes in the window against 295, since the network arm and the
replays now run before it starts and the larger image boots 22 ms
later (the bootloader's `Loaded app` at 147 ms against 125).

## Seventh entry: round 33, the version string (2026-09-14)

Same SuperMini (MAC `70:af:09:07:f6:3c`, esp32c3 revision v0.4), same
port, same tool, 20 s after reset. Why this entry exists: the bump to
`0.1.0-alpha.8` (PR F, `6ddf774`) changes no source and still moves the
`.text`, because the symbol hashes follow the package version and the
link order follows them. The version string moves the `.text`; the
behavior does not, and this entry reads that on the board. `build.sh`
at `6ddf774` in the main clone on 1.98.1: ELF `a7be3286…`, `.text`
`b5351531…` → `6f8ec538…` (§ Rebuild + run), the value measured before
the PR on `df5e55d`'s sources with the version edited, the gate 0 hits,
the trim-paths canary still unstable on 1.98.1 (exit 101). Flashed from
a copy of that ELF, `board-r33-alpha8.log`, 47,730 bytes. As in the
sixth entry, the capture holds two boots: it opens with the tail of the
boot that followed the flash (a bootloader line spliced into another,
the banner and both neuron arms), cut by the capture's reset during
that boot's network arm, then the full run. Every figure below is from
after the `rst:` line.

| Measurement | Round 32 | Round 33 (alpha.8) |
|---|---|---|
| Burst, pinned arm | 15,774 µs → 1,577 ns/step | 15,767 µs → **1,576 ns/step** |
| Burst, free arm | 3,676 µs → 367 ns/step | 3,672 µs → 367 ns/step |
| Burst spikes, first spike step, checksum, both neuron arms | 147, 55, `0b78b456` | 147, 55, `0b78b456` |
| Burst, network arm (8 neurons, 6 synapses a step) | 140,586 µs → 14,058 ns/step | 140,586 µs → **14,058 ns/step** |
| Network arm: spikes, first spike step, checksum | 3,377, 5, `7e700ee1` | 3,377, 5, `7e700ee1` |
| First spike, real-time loop | step 55, 55,149 µs | step 57, 57,139 µs |
| Spikes after the reset | 290 | 290 |

The gates hold. Both neuron arms and the network arm keep their fold,
and `python3 tools/esp32c3_trace_diff.py evidence/esp32c3-bringup/board-r33-alpha8.log`
prints the twelve plasticity-off cases identical, with the sixth
entry's row counts, the end line present, `12 cases, 0 red`. The three
burst times move by at most 7 µs per 10,000 steps (pinned −7, free −4,
network 0), under one ns per step; not read further. The real-time
loop is reported, not gated: its first spike lands at step 57, against
55 in round 32, with 290 spikes in the window, as in round 32.

## Host bench, round 33: alpha.7 against alpha.8 (2026-09-14)

Round 30's check once more, as PR F's rider: the alpha.8 tree against
the alpha.7 tree, on one compiler, 1.98.1. Same box (Intel i5-6200U,
governor `powersave`, nothing pinned), round 30's method: `git archive`
trees of `Cargo.toml`, `rust-toolchain.toml`, `crates/` and
`proofs/spike-path-bench/` for the alpha.7 tree (`a974b9e`, the PR #24
merge commit) and the bumped tree (`6ddf774`), each built with
`cargo +1.98.1 build --release --locked --offline`; the two binaries
built first, then three rounds in rotated order in one run,
`host-bench-r33-alpha8.log`. The log opens with a line naming both
trees and closes with the per-round medians, appended by the run
itself. The harness is the same code in both trees (its one change is
a doc comment); on alpha.8 the forced arm's spike path stores nothing,
since the neuron keeps no history (PR D).

```bash
for t in a7:a974b9e a8:6ddf774; do
  d=bench-rebuild/r33-${t%%:*}; mkdir -p "$d"
  git archive "${t#*:}" Cargo.toml rust-toolchain.toml crates proofs/spike-path-bench | tar -x -C "$d"
  (cd "$d/proofs/spike-path-bench" && cargo +1.98.1 build --release --locked --offline)
done
# then one run into the log: rounds a7 a8 / a8 a7 / a7 a8, each binary
# as built, a "=== round <r> <tree>" line before each
```

Absolute figures move between runs on this laptop; only this run's
comparisons are read. Each figure is the median of the three rounds'
medians, ns/step:

| Arm | alpha.7 (`a974b9e`) | alpha.8 (`6ddf774`) | alpha.8 − alpha.7 |
|---|---|---|---|
| forced | 15.428 | 14.815 | −0.613 |
| control | 13.930 | 13.979 | +0.049 |

The reading. On the control arm, the loop where no step fires, alpha.8
is where alpha.7 is: its three rounds (13.703 to 14.076) sit inside
alpha.7's (13.456 to 14.562). On the forced arm alpha.8 is 0.6 ns/step
faster, and its slowest round (15.274) is below alpha.7's fastest
(15.338) by 0.064 ns, a margin this laptop's drift between runs can
cover; not read further. Alpha.6 is not in this run, so round 27's cost
against it (§ Host bench, round 30) is not re-measured here.

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
#   0de3bd91f3d64c64060c70b1bbc868a3717fbd6f8236d423e9dd3bbdaf035322  round-27 fix (build.sh at 1affcd6: the alpha.7 spine, the baseline's firmware source; ISA round 27, item 7)
#   33656de99e9e008b5e8681a86de041227bc1f470cc424e3f3108087e3f0d80c5  round 30, the pin (build.sh at 885ff53 on 1.98.1, main clone: the alpha.7 spine and firmware source; ISA round 30)
#   61639145c30824f334785f1bd11b88ba60aabe9fee4763dd2514b151b2bb46a1  round 31, the recorder (build.sh at d3adc5b on 1.98.1, main clone: the neuron without its ring, the firmware source unchanged; ISA round 31)
#   b5351531a8765a5a2aadd41132d64abe366d6115d2137ca9293ae3dab2ec1643  round 32, the network (build.sh on 1.98.1, main clone, from the sixth entry's sources: the network arm and the twelve replays; ISA round 32)
#   6f8ec5385617a1345d251591d237a77718eed1952686d7ab50a5ca0962a06a77  round 33, the version (build.sh at 6ddf774 on 1.98.1, main clone: the sixth entry's sources under 0.1.0-alpha.8; ISA round 33)
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
arm, ISA round 27, item 1) on the same alpha.6 spine. The fifth is
round 27's fix: the same firmware source, the alpha.7 spine, built at
1affcd6. The sixth is that source under the new pin, 1.98.1, built at
885ff53 in the main clone (§ Fourth entry). The seventh is the same
firmware source on the spine without the neuron's ring, built at
d3adc5b in the main clone (§ Fifth entry). The eighth is the firmware
with the network arm and the replays, on the spine with
`FixedNetwork`, built from the sixth entry's sources in the main clone
(§ Sixth entry). The ninth is those sources under the version
`0.1.0-alpha.8`, built at 6ddf774 in the main clone (§ Seventh entry):
no source changed, and the `.text` did.

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
