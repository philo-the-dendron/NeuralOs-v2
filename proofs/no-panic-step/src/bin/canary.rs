//! The proof's canary: a panic path the link must refuse, under both
//! profiles, the linker naming the handler's undefined symbol. If this ever
//! links, the handler no longer makes a panic a red link, and the main
//! binary's green proves nothing (build.sh fails on it).

#![no_std]
#![no_main]

#[path = "../bare.rs"]
mod bare;

use core::hint::black_box;

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let table: &[u32] = &[1, 2, 3, 4];
    loop {
        // An index the optimizer cannot see, so the bounds check stays.
        black_box(table[black_box(7_usize)]);
    }
}
