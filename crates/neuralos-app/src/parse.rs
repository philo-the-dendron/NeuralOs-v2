//! arXiv Atom feed parser (streaming, via `quick-xml`).
//!
//! Turns an Atom `<feed>` body into [`Vec<Paper>`]. Handles the namespaced
//! `arxiv:primary_category` and `link` `rel`/`type` attributes the feed uses.
//! No DOM, no unsafe, no allocations beyond the returned `Vec`.

use crate::{FetchError, Paper};

/// Parse an arXiv Atom feed body into papers, in document order.
///
/// Entries missing required fields are still returned with empty strings; callers
/// that need stronger guarantees filter at the seam. Feed-level errors (malformed
/// XML) return [`FetchError::Parse`].
pub fn parse_arxiv_atom(body: &str) -> Result<Vec<Paper>, FetchError> {
    let mut reader = quick_xml::Reader::from_str(body);
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let mut papers = Vec::new();
    let mut cur: Option<Paper> = None;
    // local name of the element whose text we are currently accumulating
    let mut text_for: Option<String> = None;
    let mut in_author = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => return Err(FetchError::Parse(format!("xml read: {e}"))),
            Ok(Event::Eof) => break,
            Ok(Event::Start(start)) => {
                let local = local_name(start.name());
                match local.as_str() {
                    "entry" => cur = Some(Paper::default()),
                    "author" => in_author = true,
                    "id" | "title" | "summary" | "published" => text_for = Some(local),
                    "name" if in_author => text_for = Some(local),
                    "link" => {
                        if let Some(p) = cur.as_mut() {
                            absorb_link(&start, p);
                        }
                    }
                    "primary_category" => {
                        if let Some(p) = cur.as_mut() {
                            set_primary_category(&start, p);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(start)) => {
                let local = local_name(start.name());
                match local.as_str() {
                    "link" => {
                        if let Some(p) = cur.as_mut() {
                            absorb_link(&start, p);
                        }
                    }
                    "primary_category" => {
                        if let Some(p) = cur.as_mut() {
                            set_primary_category(&start, p);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(txt)) => {
                if let (Some(name), Some(p)) = (text_for.as_deref(), cur.as_mut()) {
                    let unescaped = txt
                        .unescape()
                        .map_err(|e| FetchError::Parse(format!("unescape: {e}")))?;
                    let trimmed = unescaped.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match name {
                        "id" => p.id.push_str(trimmed),
                        "title" => p.title.push_str(&collapse_ws(trimmed)),
                        "summary" => p.summary.push_str(&collapse_ws(trimmed)),
                        "published" => p.published.push_str(trimmed),
                        "name" => p.authors.push(trimmed.to_string()),
                        _ => {}
                    }
                }
            }
            Ok(Event::End(start)) => {
                let local = local_name(start.name());
                match local.as_str() {
                    "entry" => {
                        if let Some(mut p) = cur.take() {
                            if p.abs_url.is_empty() {
                                // arXiv `<id>` is itself the abs URL; use it as fallback.
                                p.abs_url.clone_from(&p.id);
                            }
                            papers.push(p);
                        }
                        text_for = None;
                    }
                    "author" => {
                        in_author = false;
                        text_for = None;
                    }
                    "id" | "title" | "summary" | "published" | "name" => text_for = None,
                    _ => {}
                }
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(papers)
}

/// Collapse any run of whitespace into a single space.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Strip the XML namespace prefix, returning the local name (lowercased bytes).
fn local_name(qname: quick_xml::name::QName<'_>) -> String {
    String::from_utf8_lossy(qname.local_name().into_inner()).into_owned()
}

/// Read a `link` start element into the paper's `abs_url` / `pdf_url`.
fn absorb_link(start: &quick_xml::events::BytesStart<'_>, p: &mut Paper) {
    let Some(href) = attr(start, "href") else { return };
    let rel = attr(start, "rel");
    let typ = attr(start, "type");
    match rel.as_deref() {
        Some("alternate") if p.abs_url.is_empty() => p.abs_url = href,
        Some("related") if matches!(typ.as_deref(), Some("application/pdf")) => p.pdf_url = Some(href),
        _ => {}
    }
}

/// Read `primary_category`'s `term` attribute into the paper.
fn set_primary_category(start: &quick_xml::events::BytesStart<'_>, p: &mut Paper) {
    if p.primary_category.is_none() {
        p.primary_category = attr(start, "term");
    }
}

/// Read a (non-prefixed) attribute value by name.
fn attr(start: &quick_xml::events::BytesStart<'_>, name: &str) -> Option<String> {
    for a in start.attributes().with_checks(false).flatten() {
        if String::from_utf8_lossy(a.key.as_ref()) == name {
            return a.unescape_value().ok().map(String::from);
        }
    }
    None
}

use quick_xml::events::Event;

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/arxiv_spiking.xml");

    #[test]
    fn parse_real_fixture() {
        let papers = parse_arxiv_atom(FIXTURE).expect("fixture must parse");
        assert_eq!(papers.len(), 2, "fixture has exactly 2 entries");

        let first = &papers[0];
        assert!(!first.title.is_empty());
        assert!(!first.title.contains('\n'), "title whitespace collapsed");
        assert!(
            first.authors.len() >= 2,
            "first fixture paper has multiple authors"
        );
        assert_eq!(
            first.primary_category.as_deref(),
            Some("cs.NE"),
            "primary category read from arxiv:primary_category"
        );
        assert!(
            first.abs_url.starts_with("http"),
            "abs_url populated (from link or id fallback)"
        );
        assert!(
            first.pdf_url.as_ref().is_some_and(|u| u.contains("/pdf/")),
            "pdf_url captured from rel=related application/pdf link"
        );
        assert!(
            first.published.ends_with('Z'),
            "published timestamp captured"
        );

        let second = &papers[1];
        assert!(!second.title.is_empty(), "second paper title parsed");
        assert!(!second.authors.is_empty(), "second paper authors parsed");
    }

    #[test]
    fn malformed_xml_is_parse_error_not_panic() {
        // quick-xml is lenient about truncation, but `Text::unescape()` rejects
        // an undefined entity reference — the path our parser maps to Parse.
        let body =
            "<feed xmlns='http://www.w3.org/2005/Atom'><entry><title>a&nosuchent;b</title></entry></feed>";
        let res = parse_arxiv_atom(body);
        assert!(matches!(res, Err(FetchError::Parse(_))), "undefined entity -> Parse, got {res:?}");
    }

    #[test]
    fn empty_feed_yields_no_papers() {
        let body = "<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'></feed>";
        let papers = parse_arxiv_atom(body).expect("empty feed parses");
        assert!(papers.is_empty());
    }
}
