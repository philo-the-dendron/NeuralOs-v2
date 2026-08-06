//! NeuralOS desktop app — Slint UI entry point.
//!
//! Thin view layer over the framework-agnostic core in `neuralos_app`. Both
//! search (arXiv) and summarize (Qwen2.5-1.5B-Instruct int4, behind the `qwen`
//! feature) run their heavy work on a worker thread; results return to the UI
//! thread via `slint::invoke_from_event_loop` so the event loop never blocks.

use neuralos_app::{arxiv::ArxivFetcher, pubmed::PubmedFetcher, Fetch};
use slint::{ModelRc, VecModel};

#[cfg(feature = "qwen")]
use neuralos_app::qwen::{QwenConfig, QwenSummarizer};
#[cfg(feature = "qwen")]
use neuralos_app::Summarize;
#[cfg(feature = "qwen")]
use std::sync::{Arc, Mutex};

slint::include_modules!();

/// Lazily-initialized shared summarizer (None until first summarize click).
/// Persists across clicks so the ~5 s model load happens once per session.
#[cfg(feature = "qwen")]
type SharedSummarizer = Arc<Mutex<Option<QwenSummarizer>>>;

fn main() -> Result<(), slint::PlatformError> {
    let app = App::new()?;

    #[cfg(feature = "qwen")]
    let summarizer: SharedSummarizer = Arc::new(Mutex::new(None));

    // --- Search -------------------------------------------------------------
    let weak = app.as_weak();
    app.on_search_clicked(move || {
        let Some(app) = weak.upgrade() else { return };
        let query: String = app.get_search_query().to_string();
        let source_index = app.get_source_index();
        let weak2 = app.as_weak();
        std::thread::spawn(move || {
            // Worker thread: pick the source by index, fetch off the UI loop.
            let papers = match source_index {
                1 => PubmedFetcher::new().search(&query, 20).unwrap_or_default(),
                _ => ArxivFetcher::new().search(&query, 20).unwrap_or_default(),
            };
            // Carry the abstract so a row click can feed the summarizer.
            let rows: Vec<(String, String, String)> = papers
                .iter()
                .map(|p| (p.title.clone(), p.abs_url.clone(), p.summary.clone()))
                .collect();

            let _ = slint::invoke_from_event_loop(move || {
                let Some(app) = weak2.upgrade() else { return };
                let slint_rows: Vec<TitleRow> = rows
                    .into_iter()
                    .map(|(title, abs_url, abstract_text)| TitleRow {
                        title: title.into(),
                        abs_url: abs_url.into(),
                        abstract_text: abstract_text.into(),
                    })
                    .collect();
                let model = std::rc::Rc::new(VecModel::from(slint_rows));
                app.set_results(ModelRc::new(model));
            });
        });
    });

    // --- Summarize ----------------------------------------------------------
    #[cfg(feature = "qwen")]
    {
        let weak = app.as_weak();
        let sm = summarizer.clone();
        app.on_summarize_clicked(move || {
            let Some(app) = weak.upgrade() else { return };
            let abstract_text: String = app.get_selected_abstract().to_string();
            if abstract_text.trim().is_empty() {
                app.set_status_text("Select a paper first — click a title in the list.".into());
                return;
            }
            // Honest pre-status: covers model load (first run) + the ~80 s generation.
            app.set_status_text(
                "Loading model (first run downloads ~1 GB) + summarizing (~80 s)…".into(),
            );

            let weak2 = app.as_weak();
            let sm = sm.clone();
            std::thread::spawn(move || {
                // Lazy init: download + load on first click, reuse after.
                let init_err: Option<String> = match sm.lock() {
                    Ok(mut guard) => {
                        if guard.is_none() {
                            match QwenSummarizer::new(QwenConfig::default()) {
                                Ok(s) => {
                                    *guard = Some(s);
                                    None
                                }
                                Err(e) => Some(format!("model load failed: {e}")),
                            }
                        } else {
                            None
                        }
                    }
                    Err(poison) => {
                        // Recover the guard rather than crashing on a poisoned mutex.
                        let mut guard = poison.into_inner();
                        if guard.is_none() {
                            match QwenSummarizer::new(QwenConfig::default()) {
                                Ok(s) => {
                                    *guard = Some(s);
                                    None
                                }
                                Err(e) => Some(format!("model load failed: {e}")),
                            }
                        } else {
                            None
                        }
                    }
                };

                if let Some(err) = init_err {
                    let weak3 = weak2.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(app) = weak3.upgrade() {
                            app.set_status_text(err.into());
                        }
                    });
                    return;
                }

                // Summarize (greedy/deterministic).
                let result: Result<String, String> = match sm.lock() {
                    Ok(guard) => match &*guard {
                        Some(s) => s.summarize(&abstract_text).map_err(|e| e.to_string()),
                        None => Err("model not loaded".to_string()),
                    },
                    Err(poison) => {
                        let guard = poison.into_inner();
                        match &*guard {
                            Some(s) => s.summarize(&abstract_text).map_err(|e| e.to_string()),
                            None => Err("model not loaded".to_string()),
                        }
                    }
                };

                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(app) = weak2.upgrade() {
                        match result {
                            Ok(summary) => {
                                app.set_summary_text(summary.into());
                                app.set_status_text("Summary ready.".into());
                            }
                            Err(e) => {
                                app.set_status_text(("summarize failed: ".to_string() + &e).into());
                            }
                        }
                    }
                });
            });
        });
    }

    #[cfg(not(feature = "qwen"))]
    {
        let weak = app.as_weak();
        app.on_summarize_clicked(move || {
            if let Some(app) = weak.upgrade() {
                app.set_status_text(
                    "Summarize disabled — rebuild with --features qwen to enable local AI.".into(),
                );
            }
        });
    }

    // Smoke hook (test-only, env-gated): quit the event loop after N ms so
    // headless CI can verify the UI launches and exits cleanly.
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
