//! The proof's canary: a panic path the link must refuse, under both
//! profiles, the linker naming the handler's undefined symbol. Its index
//! is in range, so the bounds check survives only while `black_box` hides
//! the index: the canary is refused only if the handler makes a panic a
//! red link and the optimizer cannot see through `black_box`, the two
//! things the main binary's green rests on. If it ever links, one of them
//! failed, and build.sh fails on it.

#![no_std]
#![no_main]

#[path = "../bare.rs"]
mod bare;

use core::hint::black_box;

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let table: &[u32] = &[1, 2, 3, 4];
    loop {
        // Index 1 is in range: the bounds check stays only because
        // `black_box` hides the index, a panic path the link must refuse.
        black_box(table[black_box(1_usize)]);
    }
}
