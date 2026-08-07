//! NeuralOS SNN visualizer — Slint entry point.
//!
//! A worker thread owns the [`SimRunner`] (which owns the `SpikingNeuralNetwork`)
//! and ticks it at ~60 fps, posting each raster frame + stats line back to the UI
//! thread via `slint::invoke_from_event_loop`. The UI never blocks; shared state
//! (running, input-drive, step) crosses the boundary via atomics.
//!
//! This is the keystone wiring: `neuralos-app` now runs `neuralos-snn` at runtime.

use neuralos_app::SimRunner;
use slint::{Image, SharedPixelBuffer};
use std::sync::atomic::{AtomicBool, AtomicI16, Ordering};
use std::sync::Arc;
use std::time::Duration;

slint::include_modules!();

/// Shared control state between the UI thread (writer) and the worker (reader).
#[derive(Default)]
struct Controls {
    running: AtomicBool,
    learning: AtomicBool,
    input_drive_ua: AtomicI16,
    step_once: AtomicBool,
}

fn main() -> Result<(), slint::PlatformError> {
    const NEURON_COUNT: u16 = 128;
    const TIME_STEP_US: u32 = 1000; // 1 ms sim per tick
    const FRAME_INTERVAL: Duration = Duration::from_millis(16); // ~60 fps cap

    let app = App::new()?;

    // --- Initial UI state ---
    let runner_init = SimRunner::new(NEURON_COUNT, TIME_STEP_US)
        .expect("balanced 128-neuron network must construct");
    app.set_header_text(format!("Balanced E/I · {} neurons", NEURON_COUNT).into());
    app.set_stats_text("Running…".into());
    // 600 μA: above the balanced network's self-quenching threshold so firing
    // is visibly sustained on open. (300 μA quiets to ~1.6 Hz as inhibition
    // wins — interesting to explore by sliding down, but a bad default.)
    app.set_input_drive(600.0);
    app.set_running(true); // alive on open — the point of a microscope is to see the thing
    app.set_learning(false); // plasticity off → fixed weights → sustained firing

    // --- Shared controls ---
    let controls = Arc::new(Controls {
        running: AtomicBool::new(true),
        learning: AtomicBool::new(false),
        input_drive_ua: AtomicI16::new(600),
        step_once: AtomicBool::new(false),
    });
    {
        let weak = app.as_weak();
        let controls = controls.clone();
        app.on_run_clicked(move || {
            let now_running = !controls.running.load(Ordering::Relaxed);
            controls.running.store(now_running, Ordering::Relaxed);
            if let Some(app) = weak.upgrade() {
                app.set_running(now_running);
                app.set_stats_text(if now_running { "Running…".into() } else { "Paused.".into() });
            }
        });
    }

    // --- Step (single tick while paused) ---
    {
        let weak = app.as_weak();
        let controls = controls.clone();
        app.on_step_clicked(move || {
            // Pause if currently running, then do one tick.
            controls.running.store(false, Ordering::Relaxed);
            controls.step_once.store(true, Ordering::Relaxed);
            if let Some(app) = weak.upgrade() {
                app.set_running(false);
            }
        });
    }

    // --- Learn toggle (STDP on/off) ---
    {
        let weak = app.as_weak();
        let controls = controls.clone();
        app.on_learn_clicked(move || {
            let now = !controls.learning.load(Ordering::Relaxed);
            controls.learning.store(now, Ordering::Relaxed);
            if let Some(app) = weak.upgrade() {
                app.set_learning(now);
            }
        });
    }

    // --- Input-drive slider (live write to shared state) ---
    {
        let controls = controls.clone();
        app.on_drive_edited(move |v| {
            controls.input_drive_ua.store(v as i16, Ordering::Relaxed);
        });
    }

    // --- Worker thread: owns the sim, posts frames to the UI ---
    // Lifecycle: `worker_alive` is the ONLY thing the worker checks to keep
    // running. Do NOT call `app_weak.upgrade()` from this thread — Slint Weak
    // upgrades are only valid on the UI thread and return None off-thread,
    // which previously made the worker exit on iteration 1 (the black-panes bug).
    // The weak is upgraded INSIDE the invoke_from_event_loop closure (UI thread).
    let app_weak = app.as_weak();
    let worker_alive = Arc::new(AtomicBool::new(true));
    let worker_alive_w = worker_alive.clone();
    std::thread::spawn(move || {
        let mut runner = runner_init;
        let mut last_learning = runner.learning();
        while worker_alive_w.load(Ordering::Relaxed) {
            // Sync the learning toggle (only call into the net on a real change).
            let want_learning = controls.learning.load(Ordering::Relaxed);
            if want_learning != last_learning {
                runner.set_learning(want_learning);
                last_learning = want_learning;
            }

            let should_step = controls.running.load(Ordering::Relaxed)
                || controls.step_once.swap(false, Ordering::Relaxed);

            if should_step {
                let drive = controls.input_drive_ua.load(Ordering::Relaxed);
                runner.tick(drive);
                let stats = runner.stats_text();
                let (rw, rh, rbytes) = runner.raster_display();
                let raster_frame = rbytes.to_vec();
                let (ww, wh, wbytes) = runner.weight_matrix_display();
                let weight_frame = wbytes.to_vec();

                let app_weak = app_weak.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    // Upgraded ON the UI thread — valid here. No-op if the window closed.
                    let Some(app) = app_weak.upgrade() else { return };
                    let mut rbuf = SharedPixelBuffer::<slint::Rgba8Pixel>::new(rw as u32, rh as u32);
                    rbuf.make_mut_bytes().copy_from_slice(&raster_frame);
                    app.set_raster(Image::from_rgba8(rbuf));
                    let mut wbuf = SharedPixelBuffer::<slint::Rgba8Pixel>::new(ww as u32, wh as u32);
                    wbuf.make_mut_bytes().copy_from_slice(&weight_frame);
                    app.set_weight_map(Image::from_rgba8(wbuf));
                    app.set_stats_text(stats.into());
                });
            }

            std::thread::sleep(FRAME_INTERVAL);
        }
    });

    // --- Headless smoke hook: quit after N ms so CI can verify launch + clean exit ---
    if let Ok(ms) = std::env::var("NEURALOS_SMOKE_MS") {
        if let Ok(ms) = ms.parse::<u64>() {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(ms));
                let _ = slint::quit_event_loop();
            });
        }
    }

    let result = app.run();
    // Signal the worker to exit once the window has closed.
    worker_alive.store(false, Ordering::Relaxed);
    result
}
