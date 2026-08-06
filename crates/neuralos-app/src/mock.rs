//! Offline mock fetcher + summarizer — deterministic, for UI/dev tests.

use crate::{Fetch, FetchError, Paper, Summarize, SummarizeError};

/// Returns canned papers regardless of query. No network.
#[derive(Debug, Clone, Default)]
pub struct MockFetcher {
    /// The papers `search` will return (taken up to `max_results`).
    pub papers: Vec<Paper>,
}

impl MockFetcher {
    /// Build a mock holding the given canned papers.
    #[must_use]
    pub fn new(papers: Vec<Paper>) -> Self {
        Self { papers }
    }

    /// A demo mock with two canned SNN-flavored papers — for the UI behavior test.
    #[must_use]
    pub fn demo() -> Self {
        Self::new(vec![
            Paper {
                id: "mock://1".into(),
                title: "Spiking Neural Networks for Edge Computing".into(),
                authors: vec!["A. Researcher".into()],
                summary: "Canned abstract one.".into(),
                abs_url: "https://arxiv.org/abs/0000.00001".into(),
                pdf_url: Some("https://arxiv.org/pdf/0000.00001".into()),
                published: "2026-01-01T00:00:00Z".into(),
                primary_category: Some("cs.AI".into()),
            },
            Paper {
                id: "mock://2".into(),
                title: "Ternary Quantization of SNN Weights".into(),
                authors: vec!["B. Author".into()],
                summary: "Canned abstract two.".into(),
                abs_url: "https://arxiv.org/abs/0000.00002".into(),
                pdf_url: None,
                published: "2026-02-01T00:00:00Z".into(),
                primary_category: Some("cs.NE".into()),
            },
        ])
    }
}

impl Fetch for MockFetcher {
    fn search(&self, _query: &str, max_results: usize) -> Result<Vec<Paper>, FetchError> {
        Ok(self.papers.iter().take(max_results).cloned().collect())
    }
}

/// Offline summarizer — deterministic, no model, no network. For UI/dev tests.
///
/// Returns `<prefix>: <first N words of input>` so the seam is exercised
/// end-to-end (input read, non-empty string out) without any inference.
#[derive(Debug, Clone)]
pub struct MockSummarizer {
    /// Prefix prepended to the canned summary.
    pub prefix: String,
    /// How many input words to echo.
    pub words: usize,
}

impl Default for MockSummarizer {
    fn default() -> Self {
        Self {
            prefix: "Mock summary".into(),
            words: 30,
        }
    }
}

impl MockSummarizer {
    /// Default mock summarizer (`"Mock summary"` prefix, 30 words).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Summarize for MockSummarizer {
    fn summarize(&self, text: &str) -> Result<String, SummarizeError> {
        let kept: Vec<&str> = text.split_whitespace().take(self.words).collect();
        Ok(format!("{}: {}", self.prefix, kept.join(" ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_returns_capped_papers_offline() {
        let m = MockFetcher::demo();
        let two = m.search("anything", 10).expect("mock ok");
        assert_eq!(two.len(), 2);
        let one = m.search("anything", 1).expect("mock ok");
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].title, "Spiking Neural Networks for Edge Computing");
    }

    #[test]
    fn demo_is_query_independent_and_deterministic() {
        let m = MockFetcher::demo();
        let a = m.search("spiking", 10).unwrap();
        let b = m.search("completely different", 10).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn mock_summarizer_is_deterministic_and_offline() {
        let s = MockSummarizer::new();
        let abstract_text =
            "Spiking neural networks encode information in temporal patterns of discrete events \
             and are well suited to low-power edge inference, but training them remains hard.";
        let out = s.summarize(abstract_text).expect("mock summarize ok");
        assert!(out.starts_with("Mock summary:"), "prefix applied: {out}");
        assert!(!out.is_empty());
        // deterministic: same input -> identical output, no model/network.
        assert_eq!(out, s.summarize(abstract_text).unwrap());
        // respects the words cap.
        let short = MockSummarizer { prefix: "p".into(), words: 3 }.summarize(abstract_text).unwrap();
        assert_eq!(short, "p: Spiking neural networks");
    }
}
