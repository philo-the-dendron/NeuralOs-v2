//! The import's memory grows with the weights, not with the square of
//! a root's width. `build_stages` composed each stage from an `f64`
//! identity as wide as its root on both sides, 8 bytes a pair of
//! features, and stores one now only where a root meets a Linear twice
//! (round 56). A global allocator counts the bytes `build_network`
//! holds at its peak on Input(4,000) → Linear 1×4,000 → LIF, the graph
//! NIR 2b measured.
//!
//! The counter is its test binary's one global allocator, so this file
//! holds one test. It lives in the converter's tests, a crate with
//! `publish = false`: its `unsafe impl` ships in no package, and
//! `tools/mutants.sh` runs these tests for every mutant of the library.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use neuralos_snn::lif_neuron::VoltageResolution;
use neuralos_snn::nir::{NirBuilder, NirImportOptions, NirLifParams};

/// `System`, counting the bytes live and their peak.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: each call goes to `System` with the caller's own arguments;
// the counters read the layout alone.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on unchanged.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(live, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, passed on unchanged.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// The features of the graph's one Input.
const N: u32 = 4_000;

/// The most `build_network` may hold at its peak beyond what was live
/// before it, 1 MiB: about 25 times the 42,020 bytes it held after the
/// fix, 122 times under the 128,138,020 it held before (round 56).
const BOUND: usize = 1 << 20;

#[test]
fn the_import_stores_no_identity_as_wide_as_its_root() {
    let mut b = NirBuilder::new(NirImportOptions::new(
        100,
        VoltageResolution::CentiMillivolt,
    ));
    let inp = b.add_input("input", &[N]).expect("input");
    let weights = vec![1.0; N as usize];
    let l = b.add_linear("l", &weights, 1, N as usize).expect("linear");
    let lif = b
        .add_lif_population(
            "a",
            &NirLifParams {
                tau_s: &[0.005],
                r_ohm: &[5e8],
                v_leak_v: &[0.0],
                v_threshold_v: &[0.01],
                v_reset_v: Some(&[0.0]),
            },
        )
        .expect("lif");
    let out = b.add_output("output", &[1]).expect("output");
    for (from, to) in [(inp, l), (l, lif), (lif, out)] {
        b.add_edge(from, to).expect("edge");
    }
    let g = b.build().expect("builds");
    let before = LIVE.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    let built = g.build_network().expect("assembles");
    let held = PEAK.load(Ordering::SeqCst) - before;
    drop(built);
    assert!(
        held <= BOUND,
        "build_network held {held} bytes at its peak, past {BOUND}"
    );
}
