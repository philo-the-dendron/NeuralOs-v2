//! Writes the regression traces and the reference vector of
//! `tests/traces/`. The cases and the format are `tests/traces/cases.rs`;
//! the compare is `cargo test -p neuralos-snn --test traces`, part of the
//! test gate.
//!
//! Run: `cargo run -p neuralos-snn --example trace -- write`
//!
//! Regenerates every case in place and prints one line per file with its
//! size and row count. A trace that moved is a behavior change: commit
//! the regenerated files with a message that says which cases moved and
//! why.

#[path = "../tests/traces/cases.rs"]
mod cases;

fn main() {
    let mut args = std::env::args().skip(1);
    match (args.next().as_deref(), args.next()) {
        (Some("write"), None) => write(),
        _ => {
            eprintln!("usage: cargo run -p neuralos-snn --example trace -- write");
            std::process::exit(2);
        }
    }
}

fn write() {
    let mut total = 0usize;
    for case in cases::CASES {
        let text = cases::render(case);
        let path = cases::path(case);
        if let Err(e) = std::fs::write(&path, &text) {
            eprintln!("{}: {e}", path.display());
            std::process::exit(1);
        }
        println!(
            "{:<26} {:>7} B  {:>5} rows",
            case.name,
            text.len(),
            text.lines().count() - 1
        );
        total += text.len();
    }
    println!("{} files, {total} B, in tests/traces/", cases::CASES.len());
}
