//! One LIF neuron from `neuralos-snn` on an ESP32-C3.
//!
//! What it does, in order:
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
//! 2. loop: one step every `DT_US`, paced by a busy-wait delay (no hardware
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
//! comes from `build.sh` only (evidence README § Release asset).

#![no_std]
#![no_main]

use esp_hal::clock::CpuClock;
use esp_hal::delay::Delay;
use esp_hal::gpio::{Level, Output, OutputConfig};
use esp_hal::main;
use esp_hal::time::Instant;
use esp_println::println;
use neuralos_snn::lif_neuron::{LIFNeuron, NeuronType, VoltageResolution};

/// Simulation step, µs. Same value the crate's own tests use.
const DT_US: u32 = 1_000;
/// Constant input current, µA. Above the excitatory threshold current
/// (~150 µA, V_ss = −54 mV); fires on the centi-mV grid, silent on the mV
/// grid. See the module note.
const INPUT_UA: i16 = 160;
/// Steps in each timed burst arm.
const BURST_STEPS: u32 = 10_000;

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
fn report(arm: &str, elapsed_us: u64, spikes: u32, first_spike_step: u32, checksum: u32) {
    println!(
        "burst {}: {} steps in {} us -> {} ns/step, {} spikes, first spike step {}, checksum {:08x}",
        arm,
        BURST_STEPS,
        elapsed_us,
        elapsed_us * 1_000 / BURST_STEPS as u64,
        spikes,
        first_spike_step,
        checksum
    );
}

#[main]
fn main() -> ! {
    let p = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
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
    report("free", elapsed_us, spikes, first_spike_step, checksum);

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
    report("pinned", elapsed_us, spikes, first_spike_step, checksum);

    // 2. Real-time loop: one step per DT_US of wall time. now_us wraps at
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
