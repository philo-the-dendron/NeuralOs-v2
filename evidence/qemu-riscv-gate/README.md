# QEMU riscv64gc gate — the no_std proof (2026-08-21)

The evidence gate for the VISION claim "runs on RISC-V edge silicon":
`neuralos-snn` proved on `riscv64gc` under QEMU, both postures —
bare-metal (Leg A) and linux-user running the REAL test suite (Leg B,
the wire crossing the border: `network.rs` transmission + Leg-C pins
executed on riscv64gc).

## Corrected pre-flight (arrival checks supersede the brief)

The commission said NOTHING qemu was installed (from ISA R17's
2026-08-22 note). Reality at session open, verified before any
install:

- `qemu-user-static` + `qemu-system-misc` 1:8.2.2-0ubuntu1.18 already
  installed (`dpkg -l`: `ii` both) — the principal-side install ran
  before the session opened
- `qemu-system-riscv64` and `qemu-riscv64-static` on PATH, version 8.2.2
- binfmt_misc handlers `qemu-riscv64` registered
  (`/proc/sys/fs/binfmt_misc/`) — riscv64 binaries execute transparently
- The only genuinely missing pieces were the rustup targets (userland)

So the sudo `apt install` line was skipped (ruling at scope time); no
sudo ran this session.

## Commissioning correction (recorded, not hidden)

The commission's Leg B triple — `riscv64gc-unknown-linux-gnu` +
`rust-lld` + static + no system cross-toolchain — was contradictory:
rustup ships no riscv64 glibc sysroot, so the gnu link fails on
missing `crt1.o`/`libc` no matter the linker. Ruling at scope time:
**musl swap** — `riscv64gc-unknown-linux-musl` with rustup's
self-contained static musl libc (`-C link-self-contained=yes -C
target-feature=+crt-static`), linked by `rust-lld`, zero system deps.
Claim language stays "riscv64gc under QEMU"; Leg A none-elf covers
bare-metal. No libc overclaim in either direction.

## Files

| File | What |
|---|---|
| `leg-a.log` | Leg A run transcript — 175/175 cited checks, raster, `LEG A PASS`, `qemu_exit=0` |
| `leg-b.log` | Leg B full-suite run — 187 unit + 8 integration = 195/195 (matches host count), `cargo_test_exit=0`, wall clock |
| `SHA256SUMS` | pins both logs |
| `../../proofs/qemu-riscv-leg-a/` | the committed, rebuildable Leg A harness crate (standalone workspace; repo workspace untouched) |

## Leg A — bare-metal none-elf (the edge posture)

Harness: `proofs/qemu-riscv-leg-a` — `no_std`/`no_main`, RAM entry
`0x8000_0000`, no allocator, own `[workspace]` table (excluded from
the repo workspace by construction; `cargo check --workspace` from the
root stays green and never sees it). Exit via the sifive_test
poweroff device (`0x5555` → 0, `0x13333` → 1). GOTCHA found and fixed
live: the test device is at `0x0010_0000` (1 MiB), NOT `0x0100_0000`
— one hex digit, writes silently swallowed into unmapped space.

175 exact-value checks, each citing its source unit test by name
(same numbers, same names; drift would be a finding, never adjusted
to match): lif (mV trace, dead-zone pair, leak convergence,
adaptation decay +2/spike, centi bounds, rectification ratchet) ·
synapse (AMPA/GABA defaults, STDP zero-dt/outside-window, clamping,
self-connection) · bridge (i2_s/q1_0/q2_0 known vectors byte-exact,
the REAL Bonsai Q2_0 first-block census +37/0×43/−48 with decode→encode
byte identity, 12 fp16 widening vectors, γ policy, repack order) ·
trit (boundaries γ=125/γ=5, half-up scale, projection snap,
draw-0/draw-max flips) · kernel (pack 0x52, absmax incl. i16::MIN→
32768 no-wrap, matvec [800, −6207]) · nir quantizers (threshold_q
−5550 on centi grid, dyadic arena [16384, −32767, 8192], scale
1/32767, denormal-loud, full error taxonomy).

Honest shape note: `network.rs`/`csr.rs`/`stats.rs` are std-gated —
the network-level transmission wire is Leg B's job; the raster
(4 neurons × 40 steps, same 300 µA drive: mV grid silent, centi grid
fires) is LIFNeuron-level on target.

```bash
# rebuild + run (all copy-pasteable from repo root)
rustup target add riscv64gc-unknown-none-elf
cd proofs/qemu-riscv-leg-a && cargo build
timeout 120 qemu-system-riscv64 -machine virt -nographic -bios none \
  -kernel target/riscv64gc-unknown-none-elf/debug/qemu-riscv-leg-a
# gate: echo $? == 0 → PASS (1 → FAIL); UART carries the check log
```

The FAIL direction is verified too: a deliberately-broken check was
probed (exit 1, `FAIL [ 26] lif::centi_mode membrane got=-7000
want=-7001`), then restored — the gate bites.

## Leg B — user-mode, the REAL suite (the wire crosses)

Full workspace test suite of the crate — NOT a subset — built static
musl with rust-lld (no system cross-toolchain) and executed via
binfmt (`qemu-riscv64-static` 8.2.2 underneath):

```bash
rustup target add riscv64gc-unknown-linux-musl
CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_MUSL_LINKER=rust-lld \
CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-C link-self-contained=yes -C target-feature=+crt-static" \
  cargo test -p neuralos-snn --target riscv64gc-unknown-linux-musl
# gate: exit 0; 187 + 8 = 195 passed, 0 failed; ~14 s wall under TCG
```

Named tests that crossed the border (in `leg-b.log`): the
transmission trio (`transmission_is_live_one_step_delayed_centimv`,
`..._mv_strong_weight`, `transmission_pulses_sum_across_presynaptic_spikes`),
the two Leg-C pins (`plasticity_off_freezes_weights_under_adapting_drive`,
`adaptation_decay_runs_before_integration_exact`), the F1/R4(ii) divisor
pins, and the CSR pins (`sparse_matrix_iter_returns_added_synapses`,
`csr_finalize_recovers_correct_members_for_unsorted_insertion` — CSR is
std-gated, so it crossed here, not in Leg A). Doctests: the crate has
0 (reported `0 passed`, trivially clean). Threads under qemu-user+musl
behaved; `--test-threads=1` was not needed.

## Verdicts of record

- Leg A: **PASS** — exit 0, 175/175 cited checks, bare-metal riscv64gc,
  no allocator
- Leg B: **PASS** — full suite 195/195 on riscv64gc under QEMU
  user-mode, exit 0
- Host battery after both legs: unchanged (294 workspace tests green,
  clippy `-D warnings` clean, no_std build green, zero `crates/` edits)

## Parked follow-up (named, not started)

QEMU riscv64 CI leg — runner cost under TCG unmeasured (Leg B was 14 s
warm here, but CI cost needs its own measurement); not added to
ci.yml this session by commission.
