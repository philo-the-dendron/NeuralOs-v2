//! Margin census (null-ladder rung 1): top1-vs-top2 margins for
//! every prompt×step of a `NEURALOS_DUMP` set. Knife-edge set =
//! margins < θ (pre-registered θ = 0.05).
//!
//! Rust port of `tools/margin_census.py` (deleted after
//! byte-compatibility was proven against the banked logs). Only the
//! small-margin tail (margin < 0.5) is printed per file — the
//! census of record stays readable.
//!
//! CLI contract mirrors the Python script: a zero-file invocation
//! prints the header + empty census and exits 0; unreadable files
//! exit 1.
//!
//! Usage: `cargo run -p neuralos-rt --release --example
//! margin_census -- [<p0.err> ...]`

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parsed: Vec<(String, neuralos_rt::judge::Dump)> = args[1..]
        .iter()
        .map(|path| {
            let tag = path
                .rsplit('/')
                .next()
                .unwrap_or(path)
                .replace("_run1.err", "")
                .replace("_run2.err", "");
            let dump = neuralos_rt::judge::parse_dump_file(path);
            (tag, dump)
        })
        .collect();
    let refs: Vec<(&str, &neuralos_rt::judge::Dump)> =
        parsed.iter().map(|(t, d)| (t.as_str(), d)).collect();
    print!("{}", neuralos_rt::judge::margin_census(&refs));
}
