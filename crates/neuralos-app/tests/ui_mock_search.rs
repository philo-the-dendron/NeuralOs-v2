//! ISC-14: a MockFetcher search flows through `present_titles` deterministically.
//!
//! This is the offline presentation-seam test — it proves the data wiring the UI
//! bin calls, without a display, a model, or a network call. The actual Slint
//! event loop launch is covered separately by the `cargo run` smoke (ISC-13).

use neuralos_app::{mock::MockFetcher, present_titles, Fetch};

#[test]
fn mock_search_populates_titles_deterministically() {
    let fetcher = MockFetcher::demo();

    let papers = fetcher
        .search("anything", 10)
        .expect("mock fetch must not fail");
    assert!(!papers.is_empty(), "demo mock returns canned papers");

    let rows = present_titles(&papers);
    assert_eq!(rows.len(), papers.len());
    assert!(
        rows.iter().all(|(t, _)| !t.is_empty()),
        "every canned title is non-empty"
    );
    assert!(
        rows.iter()
            .all(|(_, url)| url.starts_with("https://arxiv.org/abs/")),
        "every canned abs_url is well-formed"
    );

    // Deterministic + query-independent: different query -> same canned rows.
    let again = present_titles(&fetcher.search("totally different", 10).unwrap());
    assert_eq!(rows, again, "mock is deterministic and query-independent");
}
