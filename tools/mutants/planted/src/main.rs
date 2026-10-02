//! The planted CLI: an `ExitCode` match whose arms only the arm rule
//! deletes (cargo-mutants deletes arms only beside a `_` arm).

use std::process::ExitCode;

/// Says why, then exit 1: the helper an arm returns through.
fn usage(why: &str) -> ExitCode {
    eprintln!("planted: {why}");
    ExitCode::from(1)
}

fn main() -> ExitCode {
    let arg = std::env::args().nth(1).unwrap_or_default();
    match planted::run(&arg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e @ planted::E::Drive(_)) => return usage(&format!("{e:?}")),
        Err(e) => {
            eprintln!("planted: refused, {e:?}");
            ExitCode::from(2)
        }
    }
}
