//! Mechanical delta table: baseline vs patched judge logs.
//!
//! Rust port of `tools/delta.py` (deleted after byte-compatibility
//! was proven against the banked logs; the module's tests pin the
//! exact outputs, including adversarial fixtures banked from the
//! recovered Python). Parses `NEURALOS_DUMP` lines (comma decimals),
//! compares per step: argmax id (both), flip?, top-10 id overlap,
//! max |delta| over shared ids.
//!
//! CLI contract mirrors the Python script: missing arguments exit 1,
//! extra arguments are ignored, step-set mismatch / empty dumps /
//! unreadable files land on stderr and exit 1.
//!
//! Usage: `cargo run -p neuralos-rt --release --example judge_delta
//! -- <base.err> <patched.err>`

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: judge_delta <base.err> <patched.err>");
        std::process::exit(1);
    }
    let base = neuralos_rt::judge::parse_dump_file(&args[1]);
    let patched = neuralos_rt::judge::parse_dump_file(&args[2]);
    // The Python script died on AssertionError/ValueError (exit 1);
    // the library's panics carry the same messages — mapped here to
    // the same exit status.
    match std::panic::catch_unwind(|| neuralos_rt::judge::delta_table(&base, &patched)) {
        Ok(table) => print!("{table}"),
        Err(_) => std::process::exit(1),
    }
}
