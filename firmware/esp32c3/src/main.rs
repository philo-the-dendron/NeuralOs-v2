//! One LIF neuron from `neuralos-snn` on an ESP32-C3.
//!
//! What it does, in order:
//! 1. burst: `BURST_STEPS` calls of the neuron step back to back, timed with
//!    the chip clock → prints ns per step, the first board measurement
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
//! Build: `cargo build --release` in this directory (target from
//! `.cargo/config.toml`, linker script from `build.rs`). Flash + monitor:
//! `cargo run --release`.

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
/// Steps in the timed burst.
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

/// One neuron step: decay, then integrate. Returns true on a spike.
fn step(n: &mut LIFNeuron, now_us: u32) -> bool {
    n.decay_adaptation_current();
    n.integrate_and_fire(INPUT_UA, DT_US, now_us)
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

    // 1. Timed burst: no delay between steps, simulated time advances by DT_US
    //    per step. Measures the two calls together, the real per-step cost.
    let mut n = neuron(0);
    let mut spikes: u32 = 0;
    let t0 = Instant::now();
    for i in 0..BURST_STEPS {
        if step(&mut n, i.wrapping_mul(DT_US)) {
            spikes += 1;
        }
    }
    let elapsed_us = t0.elapsed().as_micros();
    println!(
        "burst: {} steps in {} us -> {} ns/step, {} spikes",
        BURST_STEPS,
        elapsed_us,
        elapsed_us * 1_000 / BURST_STEPS as u64,
        spikes
    );

    // 2. Real-time loop: one step per DT_US of wall time. now_us wraps at
    //    about 71 minutes; fine for a skeleton.
    let mut n = neuron(1);
    let mut i: u32 = 0;
    let t0 = Instant::now();
    loop {
        let now_us = t0.elapsed().as_micros() as u32;
        if step(&mut n, now_us) {
            led.toggle();
            println!("spike at {} us (step {})", now_us, i);
        }
        i = i.wrapping_add(1);
        delay.delay_micros(DT_US);
    }
}
