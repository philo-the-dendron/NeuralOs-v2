//! Framework-agnostic core for the `NeuralOS` research-summarization app.
//!
//! Holds the data model ([`Paper`]), the fetch seam ([`Fetch`] trait), and the
//! concrete source implementations (`arxiv`, `mock`). The Slint UI in `main.rs`
//! depends on this lib; nothing here depends on the UI, so the view layer is
//! replaceable without touching fetch, parse, or models.

#![warn(missing_debug_implementations)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]

pub mod arxiv;
pub mod mock;
pub mod parse;

/// A single fetched paper hit, source-agnostic.
///
/// Fields mirror what the `arXiv` Atom feed actually exposes; other sources
/// (`PubMed`) fill the same shape.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Paper {
    /// Stable source id, e.g. `oai:arXiv.org:2306.14753` or the abs URL.
    pub id: String,
    /// Whitespace-collapsed title.
    pub title: String,
    /// Authors in feed order.
    pub authors: Vec<String>,
    /// Abstract / source summary text (raw, untruncated).
    pub summary: String,
    /// Human-readable landing page (`arXiv` abs).
    pub abs_url: String,
    /// PDF link when the source exposes one.
    pub pdf_url: Option<String>,
    /// Publication timestamp, ISO-8601 as the source emits it.
    pub published: String,
    /// Primary subject category, e.g. `cs.AI`.
    pub primary_category: Option<String>,
}

/// Why a fetch failed. Kept small and non-leaky — no internal HTTP types escape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchError {
    /// Network or HTTP failure (transport, TLS, non-success status).
    Network(String),
    /// Response body could not be parsed as the expected feed format.
    Parse(String),
}

impl core::fmt::Display for FetchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Network(m) => write!(f, "fetch network error: {m}"),
            Self::Parse(m) => write!(f, "fetch parse error: {m}"),
        }
    }
}

impl std::error::Error for FetchError {}

/// The swappable fetch seam. `arXiv`, `PubMed`, and the offline mock all implement it.
///
/// Implementations are blocking by contract — callers run them off the UI thread.
pub trait Fetch: Send + Sync {
    /// Search the source for `query`, returning at most `max_results` papers.
    fn search(&self, query: &str, max_results: usize) -> Result<Vec<Paper>, FetchError>;
}

/// Project papers to the `(title, abs_url)` pairs the UI result list renders.
///
/// Named seam: the UI bin calls this, so the presentation mapping is testable
/// without a display, against any `Fetch` impl (incl. the offline mock).
#[must_use]
pub fn present_titles(papers: &[Paper]) -> Vec<(String, String)> {
    papers
        .iter()
        .map(|p| (p.title.clone(), p.abs_url.clone()))
        .collect()
}

/// Why a summarize call failed. Non-leaky — no candle/tokenizer types escape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SummarizeError {
    /// Model load/init failure (missing weights, unsupported format, OOM).
    Model(String),
    /// Tokenizer failure (encode/decode, chat-template).
    Tokenizer(String),
    /// Inference/generation failure (forward pass, sampling).
    Infer(String),
}

impl core::fmt::Display for SummarizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Model(m) => write!(f, "summarize model error: {m}"),
            Self::Tokenizer(m) => write!(f, "summarize tokenizer error: {m}"),
            Self::Infer(m) => write!(f, "summarize inference error: {m}"),
        }
    }
}

impl std::error::Error for SummarizeError {}

/// The swappable summarize seam — mirrors [`Fetch`]. The input is the raw text
/// to condense (caller picks abstract vs. full text); the output is the summary.
///
/// Implementations are blocking by contract — callers run them off the UI
/// thread, exactly as `Fetch`. First impl: `MockSummarizer` (offline); the
/// candle-backed `QwenSummarizer` (Qwen2.5-1.5B-Instruct int4) drops in here.
pub trait Summarize: Send + Sync {
    /// Condense `text` into a summary.
    fn summarize(&self, text: &str) -> Result<String, SummarizeError>;
}
