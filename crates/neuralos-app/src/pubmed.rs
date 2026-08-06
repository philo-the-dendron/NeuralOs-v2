//! `PubMed` (NCBI E-utilities) source — a second `Fetch` impl.
//!
//! Two-call flow (arXiv is one): `esearch.fcgi` returns the PMIDs for a query,
//! then `efetch.fcgi` batch-fetches those articles as XML. Both parsed with
//! `quick-xml` (no new deps). NCBI rate-limits to ~3 req/s without an API key —
//! fine for a single-user desktop client doing two sequential calls per search.

use crate::{Fetch, FetchError, Paper};
use std::time::Duration;

const ESEARCH: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";
const EFETCH: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi";

/// `PubMed` source: blocking HTTP via [`ureq`], XML parse via [`parse_esearch_ids`]
/// / [`parse_pubmed_set`].
#[derive(Debug, Clone)]
pub struct PubmedFetcher {
    agent: ureq::Agent,
}

impl Default for PubmedFetcher {
    fn default() -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(30))
                .build(),
        }
    }
}

impl PubmedFetcher {
    /// Create a fetcher with a default 30s-timeout agent.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `esearch.fcgi?db=pubmed&term=<q>&retmax=<n>&retmode=xml`. Deterministic.
    #[must_use]
    pub fn esearch_url(query: &str, ret_max: usize) -> String {
        let term = query.split_whitespace().collect::<Vec<_>>().join("+");
        format!("{ESEARCH}?db=pubmed&term={term}&retmax={ret_max}&retmode=xml")
    }

    /// `efetch.fcgi?db=pubmed&id=<ids>&rettype=abstract&retmode=xml`. Deterministic.
    #[must_use]
    pub fn efetch_url(ids: &[String]) -> String {
        let joined = ids.join(",");
        format!("{EFETCH}?db=pubmed&id={joined}&rettype=abstract&retmode=xml")
    }
}

impl Fetch for PubmedFetcher {
    fn search(&self, query: &str, max_results: usize) -> Result<Vec<Paper>, FetchError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        // 1. esearch → PMIDs.
        let body = match self
            .agent
            .get(&Self::esearch_url(query, max_results))
            .call()
        {
            Ok(resp) => resp
                .into_string()
                .map_err(|e| FetchError::Network(e.to_string()))?,
            Err(e) => return Err(FetchError::Network(format!("esearch: {e:?}"))),
        };
        let ids = parse_esearch_ids(&body)?;
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        // 2. efetch → articles.
        let body = match self.agent.get(&Self::efetch_url(&ids)).call() {
            Ok(resp) => resp
                .into_string()
                .map_err(|e| FetchError::Network(e.to_string()))?,
            Err(e) => return Err(FetchError::Network(format!("efetch: {e:?}"))),
        };
        parse_pubmed_set(&body)
    }
}

/// Parse `<eSearchResult><IdList><Id>…</Id>` → PMIDs.
fn parse_esearch_ids(body: &str) -> Result<Vec<String>, FetchError> {
    let mut reader = quick_xml::Reader::from_str(body);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut ids = Vec::new();
    let mut want = false;
    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => return Err(FetchError::Parse(format!("esearch xml: {e}"))),
            Ok(Event::Eof) => break,
            Ok(Event::Start(start)) if local_name(start.name()) == "Id" => want = true,
            Ok(Event::End(start)) if local_name(start.name()) == "Id" => want = false,
            Ok(Event::Text(txt)) if want => {
                let t = txt
                    .unescape()
                    .map_err(|e| FetchError::Parse(format!("esearch unescape: {e}")))?;
                let t = t.trim();
                if !t.is_empty() {
                    ids.push(t.to_string());
                }
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(ids)
}

/// Parse `<PubmedArticleSet><PubmedArticle>…` → [`Paper`]s.
///
/// Captures: PMID → `id`/`abs_url`, `ArticleTitle` → title, `AbstractText`
/// (concatenated across sections) → summary, `LastName` → authors, the first
/// `Year` → published, the journal `Title` → `primary_category`.
fn parse_pubmed_set(body: &str) -> Result<Vec<Paper>, FetchError> {
    let mut reader = quick_xml::Reader::from_str(body);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut papers = Vec::new();
    let mut cur: Option<Paper> = None;
    let mut text_for: Option<String> = None;
    let mut in_author = false;
    let mut year_set = false;
    let mut journal_set = false;
    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => return Err(FetchError::Parse(format!("pubmed xml: {e}"))),
            Ok(Event::Eof) => break,
            Ok(Event::Start(start)) => {
                let local = local_name(start.name());
                match local.as_str() {
                    "PubmedArticle" => {
                        cur = Some(Paper::default());
                        year_set = false;
                        journal_set = false;
                    }
                    "Author" => in_author = true,
                    "PMID" | "ArticleTitle" | "AbstractText" | "Year" | "Title" => {
                        text_for = Some(local);
                    }
                    "LastName" if in_author => text_for = Some(local),
                    _ => {}
                }
            }
            Ok(Event::End(start)) => {
                let local = local_name(start.name());
                match local.as_str() {
                    "PubmedArticle" => {
                        if let Some(mut p) = cur.take() {
                            if let Some(pmid) = p.id.strip_prefix("pubmed:") {
                                if p.abs_url.is_empty() {
                                    p.abs_url = format!("https://pubmed.ncbi.nlm.nih.gov/{pmid}/");
                                }
                            }
                            papers.push(p);
                        }
                        text_for = None;
                    }
                    "Author" => {
                        in_author = false;
                        text_for = None;
                    }
                    "PMID" | "ArticleTitle" | "AbstractText" | "Year" | "Title" | "LastName" => {
                        text_for = None;
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(txt)) => {
                if let (Some(name), Some(p)) = (text_for.as_deref(), cur.as_mut()) {
                    let unescaped = txt
                        .unescape()
                        .map_err(|e| FetchError::Parse(format!("pubmed unescape: {e}")))?;
                    let t = unescaped.trim();
                    if t.is_empty() {
                        continue;
                    }
                    match name {
                        "PMID" => p.id = format!("pubmed:{t}"),
                        "ArticleTitle" => p.title.push_str(&collapse_ws(t)),
                        "AbstractText" => {
                            if !p.summary.is_empty() {
                                p.summary.push(' ');
                            }
                            p.summary.push_str(&collapse_ws(t));
                        }
                        "LastName" => p.authors.push(t.to_string()),
                        "Year" if !year_set => {
                            p.published = t.to_string();
                            year_set = true;
                        }
                        "Title" if !journal_set => {
                            p.primary_category = Some(collapse_ws(t));
                            journal_set = true;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(papers)
}

fn local_name(qname: quick_xml::name::QName) -> String {
    String::from_utf8_lossy(qname.local_name().into_inner()).into_owned()
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

use quick_xml::events::Event;

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/pubmed_sample.xml");

    #[test]
    fn parse_real_fixture() {
        let papers = parse_pubmed_set(FIXTURE).expect("fixture must parse");
        assert_eq!(papers.len(), 2, "fixture has 2 PubmedArticles");
        let first = &papers[0];
        assert_eq!(first.id, "pubmed:42559667");
        assert!(
            first.title.starts_with("NeuroSuite"),
            "title captured: {}",
            first.title
        );
        assert!(!first.authors.is_empty(), "authors captured");
        assert!(!first.summary.is_empty(), "abstract captured");
        assert_eq!(
            first.abs_url,
            "https://pubmed.ncbi.nlm.nih.gov/42559667/",
            "abs_url built from PMID"
        );
        assert!(
            first.primary_category.as_deref().is_some_and(|c| !c.is_empty()),
            "journal captured as primary_category"
        );
        assert!(!first.published.is_empty(), "year captured");
    }

    #[test]
    fn esearch_url_encoding() {
        let url = PubmedFetcher::esearch_url("spiking neural", 5);
        assert_eq!(
            url,
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=spiking+neural&retmax=5&retmode=xml"
        );
    }

    #[test]
    fn efetch_url_encoding() {
        let url = PubmedFetcher::efetch_url(&["1".to_string(), "2".to_string()]);
        assert_eq!(
            url,
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=1,2&rettype=abstract&retmode=xml"
        );
    }

    #[test]
    fn parse_esearch_ids_extracts_ids() {
        let body = "<?xml version='1.0'?><eSearchResult><IdList><Id>111</Id><Id>222</Id></IdList></eSearchResult>";
        let ids = parse_esearch_ids(body).unwrap();
        assert_eq!(ids, vec!["111".to_string(), "222".to_string()]);
    }

    #[test]
    fn empty_query_returns_empty_without_network() {
        let papers = PubmedFetcher::new()
            .search("   ", 10)
            .expect("empty query ok");
        assert!(papers.is_empty());
    }

    #[test]
    #[ignore = "live network: run with `cargo test -- --ignored`"]
    fn live_pubmed_search_returns_papers() {
        let papers = PubmedFetcher::new()
            .search("spiking neural", 5)
            .expect("live PubMed search should succeed");
        assert!(!papers.is_empty(), "PubMed must return >=1 paper");
        for p in &papers {
            assert!(!p.title.is_empty());
            assert!(
                p.abs_url.starts_with("https://pubmed.ncbi.nlm.nih.gov/"),
                "abs_url shape: {}",
                p.abs_url
            );
        }
    }
}
