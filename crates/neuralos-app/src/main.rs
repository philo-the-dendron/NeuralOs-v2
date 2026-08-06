//! NeuralOS desktop app — Slint UI entry point.
//!
//! Thin view layer over the framework-agnostic core in `neuralos_app`. A search
//! runs the fetch on a worker thread; the raw `Vec<(title, url)>` crosses to the
//! UI thread via `slint::invoke_from_event_loop`, and the Slint `VecModel` is
//! built there (slint models are `Rc`-backed, not `Send`). The event loop never
//! blocks on the network.

use neuralos_app::{arxiv::ArxivFetcher, present_titles, Fetch};
use slint::{ModelRc, VecModel};

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    let app = App::new()?;

    let weak = app.as_weak();
    app.on_search_clicked(move || {
        let Some(app) = weak.upgrade() else { return };
        let query: String = app.get_search_query().to_string();
        let weak2 = app.as_weak();

        std::thread::spawn(move || {
            // Worker thread: real arXiv fetch, blocking, off the UI loop.
            // Errors degrade to an empty result list rather than crashing the app.
            let papers = ArxivFetcher::new().search(&query, 20).unwrap_or_default();
            // Send plain (Send) data across; build the Rc-backed model on the UI thread.
            let rows: Vec<(String, String)> = present_titles(&papers);

            let _ = slint::invoke_from_event_loop(move || {
                let Some(app) = weak2.upgrade() else { return };
                let slint_rows: Vec<TitleRow> = rows
                    .into_iter()
                    .map(|(title, abs_url)| TitleRow {
                        title: title.into(),
                        abs_url: abs_url.into(),
                    })
                    .collect();
                let model = std::rc::Rc::new(VecModel::from(slint_rows));
                app.set_results(ModelRc::new(model));
            });
        });
    });

    // Smoke hook (test-only, env-gated): if NEURALOS_SMOKE_MS is set, quit the
    // event loop after N ms so headless CI can verify the UI launches and exits
    // cleanly without a display or interaction. No effect on normal runs.
    if let Ok(ms) = std::env::var("NEURALOS_SMOKE_MS") {
        if let Ok(ms) = ms.parse::<u64>() {
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(ms));
                let _ = slint::quit_event_loop();
            });
        }
    }

    app.run()
}
