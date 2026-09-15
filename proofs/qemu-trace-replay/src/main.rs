//! The frozen traces under QEMU `riscv64gc`, bare metal: every
//! plasticity-off case of `crates/neuralos-snn/tests/traces/` stepped by
//! `FixedNetwork`, its trace's header line and its rows printed over the
//! UART by the library's writer, then one end line,
//! `# neuralos-trace end`, then exit 0.
//!
//! The replay is the firmware's (`firmware/esp32c3/src/main.rs`, step 3):
//! `frozen.rs` by `#[path]` from the library's tests directory,
//! `for_each_frozen!` over the cases, the rows through the library's row
//! writer, `neuralos_snn::fixed::row`. `build.sh` runs it and
//! diffs the capture with `tools/esp32c3_trace_diff.py`, the board's
//! diff, so a wrong writer is red, not trusted.
//!
//! The boot is Leg A's (`proofs/qemu-riscv-leg-a/src/main.rs`), copied,
//! not shared: `_start` in assembly, RAM at `0x8000_0000` (`link.x`), the
//! 16550 UART of QEMU's `virt` machine at `0x1000_0000`, the SiFive test
//! device at `0x0010_0000` for the exit (`0x5555` exits 0, `0x13333`
//! exits 1). A panic prints over the UART and exits 1.

#![no_std]
#![no_main]

use core::arch::global_asm;
use core::fmt::Write as _;
use core::panic::PanicInfo;
use core::ptr::{read_volatile, write_volatile};

use neuralos_snn::fixed::row;
use neuralos_snn::FixedNetwork;

// The frozen cases, by path into the library's tests directory, as the
// firmware includes them. rustfmt leaves the generated file as written
// (its head says why).
#[rustfmt::skip]
#[path = "../../../crates/neuralos-snn/tests/traces/frozen.rs"]
mod frozen;

const UART_BASE: usize = 0x1000_0000;
const SIFIVE_TEST: usize = 0x0010_0000;

global_asm!(
    ".section .text._start, \"ax\"",
    ".global _start",
    "_start:",
    "csrw mie, zero",
    "csrw mip, zero",
    "fence iorw, iorw",
    "li t0, 0x2000",
    "csrs mstatus, t0",
    ".option push",
    ".option norelax",
    "la gp, __global_pointer$",
    ".option pop",
    "la sp, _stack_top",
    "la t0, _bss_start",
    "la t1, _bss_end",
    "1:",
    "bgeu t0, t1, 2f",
    "sd zero, 0(t0)",
    "addi t0, t0, 8",
    "j 1b",
    "2:",
    "call rust_main",
    "3:",
    "wfi",
    "j 3b",
);

fn uart_putc(c: u8) {
    unsafe {
        while read_volatile((UART_BASE + 5) as *const u8) & 0x20 == 0 {}
        write_volatile(UART_BASE as *mut u8, c);
    }
}

fn poweroff(code: u32) -> ! {
    unsafe { write_volatile(SIFIVE_TEST as *mut u32, code) };
    loop {
        core::hint::spin_loop();
    }
}

struct Uart;

impl core::fmt::Write for Uart {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            uart_putc(b);
        }
        Ok(())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_main() -> ! {
    // Every frozen case: its header line, then its rows through the
    // library's writer, stepped as the firmware steps them.
    macro_rules! replay {
        ($case:ident) => {{
            use frozen::$case as c;
            let _ = writeln!(Uart, "{}", c::HEADER);
            let mut net = FixedNetwork::new(c::NEURONS, c::SYNAPSES, c::DT_US);
            let mut fired = [false; c::N];
            let mut step = 0u32;
            for &(count, input) in c::DRIVE {
                for _ in 0..count {
                    let time_us = net.time_us();
                    net.step(&input, &mut fired);
                    if !c::SPIKES_ONLY || fired.contains(&true) {
                        // Uart's write_str never fails.
                        let _ = row(&mut Uart, step, time_us, &fired, net.neurons());
                    }
                    step = step.wrapping_add(1);
                }
            }
        }};
    }
    frozen::for_each_frozen!(replay);
    let _ = writeln!(Uart, "# neuralos-trace end");
    poweroff(0x5555)
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let _ = writeln!(Uart, "panic: {info}");
    poweroff(0x3333 | (1 << 16))
}
