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
    app.set_input_drive(300.0); // a gentle baseline drive so activity is visible immediately
    app.set_running(true); // alive on open — the point of a microscope is to see the thing

    // --- Shared controls ---
    let controls = Arc::new(Controls {
        running: AtomicBool::new(true),
        input_drive_ua: AtomicI16::new(300),
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

    // --- Input-drive slider (live write to shared state) ---
    {
        let controls = controls.clone();
        app.on_drive_edited(move |v| {
            controls.input_drive_ua.store(v as i16, Ordering::Relaxed);
        });
    }

    // --- Worker thread: owns the sim, posts frames to the UI ---
    let app_weak = app.as_weak();
    std::thread::spawn(move || {
        let mut runner = runner_init;
        loop {
            let app_weak = app_weak.clone();
            // Exit when the window closes (weak becomes un-upgradable).
            if app_weak.upgrade().is_none() {
                break;
            }

            let should_step = controls.running.load(Ordering::Relaxed)
                || controls.step_once.swap(false, Ordering::Relaxed);

            if should_step {
                let drive = controls.input_drive_ua.load(Ordering::Relaxed);
                runner.tick(drive);
                let stats = runner.stats_text();
                let (w, h, bytes) = runner.raster_display();
                let frame = bytes.to_vec();

                let _ = slint::invoke_from_event_loop(move || {
                    let Some(app) = app_weak.upgrade() else { return };
                    // Build a SharedPixelBuffer from the raw RGBA bytes and push
                    // it into the Slint Image property.
                    let mut buf = SharedPixelBuffer::<slint::Rgba8Pixel>::new(w as u32, h as u32);
                    buf.make_mut_bytes().copy_from_slice(&frame);
                    app.set_raster(Image::from_rgba8(buf));
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

    app.run()
}
