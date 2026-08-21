//! Mechanical delta table: baseline vs patched judge logs.
//!
//! Rust port of `tools/delta.py` (deleted after byte-compatibility
//! was proven against the banked logs; the module's tests pin the
//! exact outputs). Parses `NEURALOS_DUMP` lines (comma decimals),
//! compares per step: argmax id (both), flip?, top-10 id overlap,
//! max |delta| over shared ids.
//!
//! Usage: `cargo run -p neuralos-rt --release --example judge_delta
//! -- <base.err> <patched.err>`

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: judge_delta <base.err> <patched.err>");
        std::process::exit(2);
    }
    let base = neuralos_rt::judge::parse_dump_file(&args[1]);
    let patched = neuralos_rt::judge::parse_dump_file(&args[2]);
    print!("{}", neuralos_rt::judge::delta_table(&base, &patched));
}
