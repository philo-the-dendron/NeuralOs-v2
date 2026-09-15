//! A LIF neuron and a `FixedNetwork` from `neuralos-snn` on an ESP32-C3.
//!
//! What it does, in order, once esp-hal's init has run and the stack below
//! `main`'s frame is painted (the mark, step 3):
//! 1. burst, two arms of `BURST_STEPS` calls of the neuron step back to
//!    back, timed with the chip clock; simulated time advances by `DT_US`
//!    per step. Each arm prints ns per step, the spike count, the first
//!    spike's step and a checksum of the spike steps.
//!    - free: the neuron and `DT_US` in plain sight of the optimizer, which
//!      may fold the constant and keep the neuron in registers;
//!    - pinned: the neuron behind `core::hint::black_box` once per step and
//!      dt read once through it, as a network holds them (neurons in
//!      memory, `time_step_us` a run-time field). Same id and inputs as the
//!      free arm, so the same spikes; its figure is the cost a network pays
//!      (ISA round 27).
//! 2. network: the frozen `feedforward-8` of the library's traces (8
//!    neurons, 6 synapses, `frozen.rs`), `BURST_STEPS` steps of
//!    `FixedNetwork::step` on its constant drive, timed, the network behind
//!    `black_box` once per step as the pinned arm's neuron is. It prints
//!    the same line; its checksum folds every spike, step then id
//!    (`× 31 + step`, `× 31 + id`), and `tests/traces.rs` pins its three
//!    numbers on the host (the network arm). In a build that names a
//!    stranger's graph (the slot, below), a second network arm, `stranger`:
//!    that graph, `STRANGER_BURST_STEPS` steps on its first drive run,
//!    timed around the step alone; the same line with that step count, then
//!    the largest single step in µs. Reported, not pinned: the replay is the
//!    stranger's gate.
//! 3. replay: every frozen case (`for_each_frozen!`), its trace's header
//!    line, then its rows through the library's writer
//!    (`neuralos_snn::fixed::row`) into `esp_println::Printer`; then, in
//!    a build that names a stranger's graph, that graph the same way; then
//!    one end line, `# neuralos-trace end`.
//!    `tools/esp32c3_trace_diff.py` diffs a capture against
//!    `tests/traces/`, and against the stranger's trace with `--trace`.
//!    Then the mark, one line outside every case, `stack: <mark> of
//!    <total> bytes high-water after the replays`: the stack's depth since
//!    the paint, `main`'s frame included.
//! 4. loop: one step every `DT_US`, paced by a busy-wait delay (no hardware
//!    timer peripheral yet); the LED toggles and a line is printed on each
//!    spike
//!
//! The neuron step is two calls, in the order `Network::step` uses them:
//! `decay_adaptation_current` then `integrate_and_fire`. Each spike adds
//! +2 µA of adaptation and only the decay call removes it; without the decay
//! the neuron fires a few times and goes silent for good (verified on the
//! host: 6 spikes then dark on this grid and input).
//!
//! Grid and input: centi-millivolt grid, 160 µA. On the default millivolt
//! grid 160 µA truncates to ΔV = 0 every step and the neuron never fires
//! (`above_threshold_current_blind_on_mv_spikes_on_centi` in
//! `lif_neuron.rs`); on the centi grid the same current fires, first spike
//! at step 55, then about 15 spikes per second sustained, a blink the eye
//! can see. Noise stays at the crate default (±5 µA at most); the tests use
//! noise 0, so the trace here is not the tests' trace, only the same
//! arithmetic.
//!
//! Board note: on the Espressif DevKitM-1 / DevKitC-02, GPIO8 drives a WS2812
//! RGB LED, not a plain LED, so a bare toggle shows nothing there; the serial
//! line is the ground truth on every board. Boards with a plain LED on GPIO8
//! (SuperMini and friends) blink.
//!
//! Build: `./build.sh` in this directory, the release build under rustc's
//! path remap, then the personal-string gate, the ELF and `.text` shas and
//! the trim-paths canary (`tools/remap.sh` carries the why; target from
//! `.cargo/config.toml`, linker script from `build.rs`). Flash + monitor:
//! `cargo run --release`, a plain dev build with no remap; a release asset
//! comes from `build.sh` only (evidence README § Release asset). The
//! stranger slot: `build.rs` reads `NEURALOS_GRAPH`, the path of a
//! `neuralos-nir2json --freeze` module, and adds it to the replay
//! (`cfg(stranger)`; the rules are in its header); unset, the build is the
//! default one.

#![no_std]
#![no_main]

use esp_hal::clock::CpuClock;
use esp_hal::delay::Delay;
use esp_hal::gpio::{Level, Output, OutputConfig};
use esp_hal::main;
use esp_hal::time::Instant;
use esp_println::{println, Printer};
use neuralos_snn::fixed::row;
use neuralos_snn::lif_neuron::{LIFNeuron, NeuronType, VoltageResolution};
use neuralos_snn::FixedNetwork;

// The frozen cases, by path into the library's tests directory: the
// arrays are the cases' derivative and live with them, so the library's
// package stays self-contained. rustfmt leaves the generated file as
// written (its head says why). Their rows go through the library's one
// row writer, `neuralos_snn::fixed::row`.
#[rustfmt::skip]
#[path = "../../../crates/neuralos-snn/tests/traces/frozen.rs"]
mod frozen;

// The stranger slot: a `neuralos-nir2json --freeze` module, in a build
// that names one in NEURALOS_GRAPH. build.rs copies it into OUT_DIR as
// graph.rs with one line more, `pub use self::<name> as graph;`, and sets
// `cfg(stranger)` (its header carries the rules). Two allows. dead_code is
// the one frozen.rs carries and the freezer's module does not: the replay
// reads only the constants it needs. large_const_arrays: clippy refuses a
// const array above 16,384 bytes (from 373 neurons), far inside
// stranger.sh's capacity bar, and `FixedNetwork::new` takes the arrays by
// value either way, so the capacity bar stays the one limit (measured at
// 400 neurons, and at the bar's two corners).
#[cfg(stranger)]
#[allow(dead_code, clippy::large_const_arrays)]
mod stranger {
    include!(concat!(env!("OUT_DIR"), "/graph.rs"));
}

/// The network arm's drive: `feedforward-8`'s, one constant run, read
/// from the frozen arrays at compile time.
const NETWORK_INPUT: [i16; frozen::feedforward_8::N] = frozen::feedforward_8::DRIVE[0].1;
const _: () = assert!(
    frozen::feedforward_8::DRIVE.len() == 1,
    "the network arm steps feedforward-8's drive as one constant run"
);

/// The stranger's arm's drive: its graph's first drive run as one
/// constant input, a reference into the frozen arrays (`.rodata`, no copy
/// on the stack); a graph with no drive run fails the build here.
#[cfg(stranger)]
const STRANGER_INPUT: &[i16; stranger::graph::N] = &stranger::graph::DRIVE[0].1;

/// Simulation step, µs. Same value the crate's own tests use.
const DT_US: u32 = 1_000;
/// Constant input current, µA. Above the excitatory threshold current
/// (~150 µA, V_ss = −54 mV); fires on the centi-mV grid, silent on the mV
/// grid. See the module note.
const INPUT_UA: i16 = 160;
/// Steps in each timed burst arm.
const BURST_STEPS: u32 = 10_000;
/// Steps in the stranger's timed arm: a step costs about 10 ms at the
/// capacity's worst case, so `BURST_STEPS` steps would fill the capture's
/// whole window.
#[cfg(stranger)]
const STRANGER_BURST_STEPS: u32 = 100;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic: {}", info);
    loop {}
}

// The ESP-IDF app descriptor the second-stage bootloader reads before it
// boots the factory partition. Without it the bootloader reads garbage as the
// minimum efuse revision and refuses the image (observed 2026-09-09).
esp_bootloader_esp_idf::esp_app_desc!();

fn neuron(id: u16) -> LIFNeuron {
    LIFNeuron::new_with_type_resolution(
        id,
        NeuronType::Excitatory,
        VoltageResolution::CentiMillivolt,
    )
}

/// One neuron step: decay, then integrate. Returns true on a spike. `dt_us`
/// is a parameter so the pinned arm can pass a value the optimizer cannot
/// see; every other call passes `DT_US`.
fn step(n: &mut LIFNeuron, dt_us: u32, now_us: u32) -> bool {
    n.decay_adaptation_current();
    n.integrate_and_fire(INPUT_UA, dt_us, now_us)
}

/// One line per burst arm, named, so a log reads without the brief. The
/// counts come in by value: each arm keeps them in locals whose address never
/// escapes, so the timed loop is free to hold them in registers (at the
/// alpha.6 baseline the allocator spilled them on spike steps, ISA round 27
/// item 1); a struct passed by reference would be stored to on every spike.
fn report(
    arm: &str,
    steps: u32,
    elapsed_us: u64,
    spikes: u32,
    first_spike_step: u32,
    checksum: u32,
) {
    println!(
        "burst {}: {} steps in {} us -> {} ns/step, {} spikes, first spike step {}, checksum {:08x}",
        arm,
        steps,
        elapsed_us,
        elapsed_us * 1_000 / u64::from(steps),
        spikes,
        first_spike_step,
        checksum
    );
}

// The stack's two ends, from esp-hal's linker script (`ld/sections/stack.x`):
// `.stack` runs from `_stack_end`, its lowest address, up to `_stack_start`,
// its highest, and the stack grows down from `_stack_start`. Only their
// addresses are taken, by `&raw const`, which is safe code since Rust 1.82:
// the symbols need no `unsafe`.
extern "C" {
    static _stack_end: u32;
    static _stack_start: u32;
}

/// The paint's word: a stack word that still holds it when read back was
/// not written since the paint.
const STACK_PAINT: u32 = 0xA5A5_A5A5;
/// Bytes the paint leaves alone at both of its ends: above `_stack_end`,
/// where esp-hal keeps `__stack_chk_guard` (60 bytes up, `stack.x`, written
/// at boot), and below the painter's own local, so its frame stays
/// unpainted.
const STACK_MARGIN: usize = 256;

/// Paint the stack below `main`'s frame with `STACK_PAINT`, from
/// `_stack_end + STACK_MARGIN` up to this function's `sp` less
/// `STACK_MARGIN`, `sp` read as the address of a local. Never inlined, so
/// the local sits in a frame below `main`'s: a local of `main` can sit
/// anywhere in a frame of a quarter megabyte at the capacity's corners,
/// and the paint would then run into `main`'s live frame.
#[inline(never)]
fn stack_paint() {
    let here = 0u32;
    let top = core::hint::black_box(&here) as *const u32 as usize - STACK_MARGIN;
    let mut word = (&raw const _stack_end) as usize + STACK_MARGIN;
    while word < top {
        // SAFETY: `word` is 4-aligned (`.stack` is ALIGN(4); the margin and
        // the step are multiples of 4) and inside `.stack`, below this
        // function's frame and so below every live frame: nothing reads
        // these words until a callee's or a trap's frame writes them.
        unsafe { (word as *mut u32).write_volatile(STACK_PAINT) };
        word += 4;
    }
}

/// The stack's high-water mark since the paint, bytes: `_stack_start` less
/// the lowest word above `_stack_end + STACK_MARGIN` that no longer holds
/// `STACK_PAINT`. It counts `main`'s frame, every callee's frame and any
/// trap frame; a used word that happens to hold `STACK_PAINT` reads as
/// unused.
#[inline(never)]
fn stack_high_water() -> usize {
    let top = (&raw const _stack_start) as usize;
    let mut word = (&raw const _stack_end) as usize + STACK_MARGIN;
    // SAFETY: `word` is 4-aligned and stays inside `.stack`, below
    // `_stack_start`; the scan only reads.
    while word < top && unsafe { (word as *const u32).read_volatile() } == STACK_PAINT {
        word += 4;
    }
    top - word
}

#[main]
fn main() -> ! {
    let p = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    // The paint, right after esp-hal's init: from here on, every frame
    // under `main`'s is measured by the mark's line after the replays.
    stack_paint();
    let mut led = Output::new(p.GPIO8, Level::Low, OutputConfig::default());
    let delay = Delay::new();

    println!(
        "neuralos-esp32c3: LIF neuron, centi-mV grid, dt={} us, input={} uA",
        DT_US, INPUT_UA
    );

    // 1. Timed burst, two arms (module note): no delay between steps,
    //    simulated time advances by DT_US per step. Measures the two calls
    //    together, the real per-step cost. Each arm counts its spikes, the
    //    first spike's step and a checksum, a wrapping fold of the spike
    //    steps (`× 31 + i`, shifts and adds, no constant to hold), in locals
    //    updated on spike steps only, with no panic check.
    //
    //    Free: the neuron and DT_US in plain sight of the optimizer.
    let mut n = neuron(0);
    let (mut spikes, mut first_spike_step, mut checksum) = (0u32, u32::MAX, 0u32);
    let t0 = Instant::now();
    for i in 0..BURST_STEPS {
        if step(&mut n, DT_US, i.wrapping_mul(DT_US)) {
            if spikes == 0 {
                first_spike_step = i;
            }
            spikes = spikes.wrapping_add(1);
            checksum = checksum.wrapping_mul(31).wrapping_add(i);
        }
    }
    let elapsed_us = t0.elapsed().as_micros();
    report(
        "free",
        BURST_STEPS,
        elapsed_us,
        spikes,
        first_spike_step,
        checksum,
    );

    //    Pinned: a fresh neuron with the free arm's id, because the noise is
    //    seeded by `id ^ current_time_us`, so it computes the same values.
    //    `black_box(&mut n)` once per step: the optimizer must assume the
    //    neuron is read and written behind its back, so it stays in memory,
    //    as in a network. dt is read once through `black_box` and passed in,
    //    a run-time value like a network's `time_step_us`; the timestamp
    //    stays `i × DT_US`.
    let mut n = neuron(0);
    let dt_us = core::hint::black_box(DT_US);
    let (mut spikes, mut first_spike_step, mut checksum) = (0u32, u32::MAX, 0u32);
    let t0 = Instant::now();
    for i in 0..BURST_STEPS {
        if step(core::hint::black_box(&mut n), dt_us, i.wrapping_mul(DT_US)) {
            if spikes == 0 {
                first_spike_step = i;
            }
            spikes = spikes.wrapping_add(1);
            checksum = checksum.wrapping_mul(31).wrapping_add(i);
        }
    }
    let elapsed_us = t0.elapsed().as_micros();
    report(
        "pinned",
        BURST_STEPS,
        elapsed_us,
        spikes,
        first_spike_step,
        checksum,
    );

    // 2. The network arm: the frozen feedforward-8, BURST_STEPS steps on
    //    its constant drive, behind `black_box` once per step as the
    //    pinned neuron is: in memory, read and written behind the
    //    optimizer's back. The checksum folds every spike, step then id
    //    (`× 31 + step`, `× 31 + id`); tests/traces.rs pins the three
    //    numbers on the host.
    let mut net = FixedNetwork::new(
        frozen::feedforward_8::NEURONS,
        frozen::feedforward_8::SYNAPSES,
        frozen::feedforward_8::DT_US,
    );
    let mut fired = [false; frozen::feedforward_8::N];
    let (mut spikes, mut first_spike_step, mut checksum) = (0u32, u32::MAX, 0u32);
    let t0 = Instant::now();
    for i in 0..BURST_STEPS {
        core::hint::black_box(&mut net).step(&NETWORK_INPUT, &mut fired);
        for (id, spiked) in (0u32..).zip(fired) {
            if spiked {
                if spikes == 0 {
                    first_spike_step = i;
                }
                spikes = spikes.wrapping_add(1);
                checksum = checksum
                    .wrapping_mul(31)
                    .wrapping_add(i)
                    .wrapping_mul(31)
                    .wrapping_add(id);
            }
        }
    }
    let elapsed_us = t0.elapsed().as_micros();
    report(
        "network",
        BURST_STEPS,
        elapsed_us,
        spikes,
        first_spike_step,
        checksum,
    );

    //    The stranger's arm, in a build that names a graph (the slot): its
    //    network behind `black_box` once per step, as the network arm's,
    //    STRANGER_BURST_STEPS steps on its first drive run. Timed around
    //    the step alone, one sample a step with the timer read inside it:
    //    the fold over `fired` runs between the samples, since at thousands
    //    of neurons its scan is about a tenth of a step. It prints the
    //    network arm's line, then its largest sample, the cost of a firing
    //    step, which a real-time loop must fit. One binding holds the
    //    stranger's network for this arm and for its replay below, reset
    //    there by assignment, so the frame holds one copy whatever the
    //    stack coloring does: two would not fit at the capacity's corners.
    #[cfg(stranger)]
    let mut stranger_net = FixedNetwork::new(
        stranger::graph::NEURONS,
        stranger::graph::SYNAPSES,
        stranger::graph::DT_US,
    );
    #[cfg(stranger)]
    let mut stranger_fired = [false; stranger::graph::N];
    #[cfg(stranger)]
    {
        let (mut spikes, mut first_spike_step, mut checksum) = (0u32, u32::MAX, 0u32);
        let (mut elapsed_us, mut largest_us) = (0u64, 0u64);
        for i in 0..STRANGER_BURST_STEPS {
            let t0 = Instant::now();
            core::hint::black_box(&mut stranger_net).step(STRANGER_INPUT, &mut stranger_fired);
            let sample_us = t0.elapsed().as_micros();
            elapsed_us += sample_us;
            largest_us = largest_us.max(sample_us);
            for (id, &spiked) in (0u32..).zip(stranger_fired.iter()) {
                if spiked {
                    if spikes == 0 {
                        first_spike_step = i;
                    }
                    spikes = spikes.wrapping_add(1);
                    checksum = checksum
                        .wrapping_mul(31)
                        .wrapping_add(i)
                        .wrapping_mul(31)
                        .wrapping_add(id);
                }
            }
        }
        report(
            "stranger",
            STRANGER_BURST_STEPS,
            elapsed_us,
            spikes,
            first_spike_step,
            checksum,
        );
        println!("burst stranger: largest step {} us", largest_us);
    }

    // 3. The replays: every frozen case, its header line, then its rows
    //    through the library's writer; then the stranger's graph, in a
    //    build that names one, on the arm's binding; then the end line,
    //    all before the loop below, whose spike lines would otherwise land
    //    in the last case. tools/esp32c3_trace_diff.py compares them with
    //    the host's files. replay_rows! steps a network it is given;
    //    replay! builds one per case, whose path for_each_frozen! gives as
    //    a name, which replay_frozen! makes a path into frozen.
    macro_rules! replay_rows {
        ($c:ident, $net:ident, $fired:ident) => {{
            let mut step = 0u32;
            for &(count, input) in $c::DRIVE {
                for _ in 0..count {
                    let time_us = $net.time_us();
                    $net.step(&input, &mut $fired);
                    if !$c::SPIKES_ONLY || $fired.contains(&true) {
                        // Printer's write_str never fails.
                        let _ = row(&mut Printer, step, time_us, &$fired, $net.neurons());
                    }
                    step = step.wrapping_add(1);
                }
            }
        }};
    }
    macro_rules! replay {
        ($case:path) => {{
            use $case as c;
            println!("{}", c::HEADER);
            let mut net = FixedNetwork::new(c::NEURONS, c::SYNAPSES, c::DT_US);
            let mut fired = [false; c::N];
            replay_rows!(c, net, fired);
        }};
    }
    macro_rules! replay_frozen {
        ($case:ident) => {
            replay!(frozen::$case)
        };
    }
    frozen::for_each_frozen!(replay_frozen);
    #[cfg(stranger)]
    {
        use stranger::graph as c;
        println!("{}", c::HEADER);
        // `fired` needs no reset: each step writes every entry.
        stranger_net = FixedNetwork::new(c::NEURONS, c::SYNAPSES, c::DT_US);
        replay_rows!(c, stranger_net, stranger_fired);
    }
    println!("# neuralos-trace end");

    //    The mark: the stack's high-water since the paint, one line after
    //    the end line, outside every case, so the diff never reads it.
    println!(
        "stack: {} of {} bytes high-water after the replays",
        stack_high_water(),
        (&raw const _stack_start) as usize - (&raw const _stack_end) as usize
    );

    // 4. Real-time loop: one step per DT_US of wall time. now_us wraps at
    //    about 71 minutes; fine for a skeleton.
    let mut n = neuron(1);
    let mut i: u32 = 0;
    let t0 = Instant::now();
    loop {
        let now_us = t0.elapsed().as_micros() as u32;
        if step(&mut n, DT_US, now_us) {
            led.toggle();
            println!("spike at {} us (step {})", now_us, i);
        }
        i = i.wrapping_add(1);
        delay.delay_micros(DT_US);
    }
}
