//! What both binaries share: the panic handler that makes a reachable
//! panic a red link.
//!
//! The handler calls `neuralos_fixed_step_has_a_panic_path`, which nothing
//! defines. When no panic is reachable from `_start`, the linker's section
//! garbage collection drops the handler and its reference, and the link is
//! green. When one is, the reference is live and lld refuses it:
//! `undefined symbol: neuralos_fixed_step_has_a_panic_path`, referenced
//! by the handler (`__rustc::rust_begin_unwind`, in its codegen unit). The
//! linker names the handler, never the site; build.sh prints the build that
//! finds the site.

#[cfg(not(feature = "handler-loops"))]
extern "C" {
    fn neuralos_fixed_step_has_a_panic_path() -> !;
}

#[cfg(not(feature = "handler-loops"))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // SAFETY: never runs. A green link holds no reference to the symbol,
    // and a red one produces no binary.
    unsafe { neuralos_fixed_step_has_a_panic_path() }
}

/// The site-finding handler (build.sh's recipe): the link goes through,
/// and every call into the panic machinery stays in the disassembly.
#[cfg(feature = "handler-loops")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
