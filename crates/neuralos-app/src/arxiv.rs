//! arXiv Atom API source.

use crate::{parse, Fetch, FetchError, Paper};
use std::time::Duration;

/// arXiv Atom API base. Free, no key, rate-limited (~1 req/3s); we honor that
/// by being a single-user desktop client.
const ENDPOINT: &str = "https://export.arxiv.org/api/query";

/// arXiv source: blocking HTTP via [`ureq`], Atom parse via [`parse`].
#[derive(Debug, Clone)]
pub struct ArxivFetcher {
    agent: ureq::Agent,
}

impl Default for ArxivFetcher {
    fn default() -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(30))
                .build(),
        }
    }
}

impl ArxivFetcher {
    /// Create a fetcher with a default 30s-timeout agent.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the arXiv Atom query URL for a free-text search.
    ///
    /// Each whitespace-separated term is scoped `all:<term>` and joined with
    /// `+AND+` (arXiv uses `+` for the encoded space). Deterministic for a given
    /// `(query, max_results)` pair — asserted byte-for-byte in tests.
    #[must_use]
    pub fn query_url(query: &str, max_results: usize) -> String {
        let terms = query
            .split_whitespace()
            .map(|t| format!("all:{t}"))
            .collect::<Vec<_>>()
            .join("+AND+");
        format!(
            "{ENDPOINT}?search_query={terms}&start=0&max_results={max_results}&sortBy=relevance"
        )
    }
}

impl Fetch for ArxivFetcher {
    fn search(&self, query: &str, max_results: usize) -> Result<Vec<Paper>, FetchError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let url = Self::query_url(query, max_results);
        let body = match self.agent.get(&url).call() {
            Ok(resp) => resp
                .into_string()
                .map_err(|e| FetchError::Network(e.to_string()))?,
            Err(e) => return Err(FetchError::Network(format!("{e:?}"))),
        };
        parse::parse_arxiv_atom(&body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_url_encoding() {
        let url = ArxivFetcher::query_url("spiking neural", 5);
        assert_eq!(
            url,
            "https://export.arxiv.org/api/query?search_query=all:spiking+AND+all:neural&start=0&max_results=5&sortBy=relevance"
        );
    }

    #[test]
    fn query_url_drops_extra_whitespace() {
        let url = ArxivFetcher::query_url("  spiking   neural  ", 1);
        assert_eq!(
            url,
            "https://export.arxiv.org/api/query?search_query=all:spiking+AND+all:neural&start=0&max_results=1&sortBy=relevance"
        );
    }

    #[test]
    fn empty_query_returns_empty_without_network() {
        let papers = ArxivFetcher::new().search("   ", 10).expect("empty query ok");
        assert!(papers.is_empty());
    }

    #[test]
    #[ignore = "live network: run with `cargo test -- --ignored`"]
    fn live_arxiv_search_returns_papers() {
        let papers = ArxivFetcher::new()
            .search("spiking neural", 5)
            .expect("live arXiv search should succeed in a networked env");
        assert!(!papers.is_empty(), "arXiv must return >=1 paper");
        for p in &papers {
            assert!(!p.title.is_empty());
            assert!(
                p.abs_url.starts_with("https://arxiv.org/abs/"),
                "abs_url shape: {}",
                p.abs_url
            );
        }
    }
}
