//! Offline mock fetcher — deterministic, for UI/dev tests.

use crate::{Fetch, FetchError, Paper};

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
}
