//! The planted CLI's exit codes, read from its process.

use std::process::Command;

fn exit(arg: &str) -> i32 {
    Command::new(env!("CARGO_BIN_EXE_planted"))
        .arg(arg)
        .status()
        .expect("the planted binary runs")
        .code()
        .expect("an exit code")
}

#[test]
fn exits() {
    assert_eq!(exit("ok"), 0);
    assert_eq!(exit("drive"), 1);
    assert_eq!(exit("other"), 2);
}
