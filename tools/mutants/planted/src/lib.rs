//! Planted cases for `tools/mutants.sh selftest`: each item is a trap a
//! mutation pass met (ISA rounds 53 and 54), and `EXPECTED.tsv`, beside
//! `Cargo.toml`, holds every mutant both tools make here with its verdict
//! and why.

use std::collections::HashSet;

#[cfg(feature = "x")]
pub mod gated;

/// Where each growth stops: past the selftest's 1 GB cap, so the cap
/// kills it first, and within the box, so a cap that fails gives a wrong
/// count, never a full box.
const BOUND: usize = 200_000_000;

/// A refusal, which `main` maps to an exit code.
#[derive(Debug)]
pub enum E {
    /// Exit 1, through `usage`.
    Drive(&'static str),
    /// Exit 2.
    Other,
}

/// What `main` runs: `drive` and `other` refuse, any other word passes.
pub fn run(arg: &str) -> Result<(), E> {
    match arg {
        "drive" => Err(E::Drive("drive")),
        "other" => Err(E::Other),
        _ => Ok(()),
    }
}

/// A match guard, then an `if` whose condition holds a string.
pub fn kind(key: &str, v: Option<&[u8]>, n: usize) -> usize {
    match v {
        Some(v) if v.len() != n => n,
        Some(_) => 0,
        None => {
            if key.is_empty() {
                1
            } else if key == "nodes" {
                2
            } else {
                3
            }
        }
    }
}

/// A dropped call a test sees: `trim()`, read by `"  # x"`.
pub fn skip(line: &str) -> bool {
    let line = line.trim();
    line.is_empty() || line.starts_with('#')
}

/// A dropped call no test sees: the tests give `sorted` sorted input.
pub fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v
}

/// An `if` whose condition alone reads a local: `if true` leaves it
/// unused, a warning `-D warnings` would make an error.
pub fn level(xs: &[u32]) -> u32 {
    let total: u32 = xs.iter().sum();
    if total > 10 {
        return 2;
    }
    1
}

/// Growth cargo-mutants makes unbounded: `+=` made `-=` keeps `total`
/// under `limit`, and only `BOUND` stops the fill.
pub fn fill(limit: i64) -> usize {
    let mut v: Vec<i64> = Vec::new();
    let mut total: i64 = 0;
    while total < limit && v.len() < BOUND {
        total += 1;
        v.push(total);
    }
    v.len()
}

/// The nodes a walk back from `to` reaches, the shape of the converter's
/// `depth` walk (round 54): its `if` made `true`, every pop pushes again,
/// and only `BOUND` stops the growth.
pub fn reach(edges: &[(usize, usize)], to: usize) -> usize {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut work = vec![to];
    while work.len() < BOUND {
        let Some(v) = work.pop() else { break };
        for &(a, b) in edges {
            if b == v && seen.insert(a) {
                work.push(a);
            }
        }
    }
    seen.len()
}

/// A loop that allocates nothing: `+=` made `*=` keeps `n` at 0, and only
/// the timeout stops it.
pub fn count(limit: u32) -> u32 {
    let mut n = 0;
    while n < limit {
        n += 1;
    }
    n
}

/// A recursion: its base case made `false`, the stack overflows and the
/// test binary dies by SIGABRT, a catch.
pub fn depth(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    1 + depth(n - 1)
}

/// An item gated inside a file every build reads: with `x` off it is
/// compiled out and its mutants read MISSED, the limit the tool names.
#[cfg(feature = "x")]
pub fn thrice(n: u32) -> u32 {
    n * 3
}

/// A `format!` argument: no rule reads a macro's arguments.
pub fn label(names: &[&str]) -> String {
    format!("{} nodes", names.join(", "))
}

/// An arm over several lines, deleted by the arm rule (its fn returns
/// `ExitCode`): a catch reaches the arm's first line only, so the
/// `eprintln!` inside, which no mutant reaches, reads blind.
pub fn code(empty: bool) -> std::process::ExitCode {
    match empty {
        true => {
            eprintln!("planted: empty");
            std::process::ExitCode::from(4)
        }
        _other => std::process::ExitCode::from(5),
    }
}

/// The same with an arm cargo-mutants deletes, beside a `_` arm: the
/// `let` inside reads blind.
pub fn tag(word: &str, other: String) -> String {
    match word {
        "x" => {
            let s = String::from("ex");
            s
        }
        _ => other,
    }
}

/// A call statement over three lines, deleted by the call rule: unlike an
/// arm's, its span reaches every line, so the argument's line is not
/// blind.
pub fn push_name(v: &mut Vec<String>) {
    v.push(String::from(
        "a name long enough that rustfmt keeps this call statement over three lines",
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds() {
        assert_eq!(kind("x", Some(&[1, 2]), 3), 3);
        assert_eq!(kind("x", Some(&[1, 2, 3]), 3), 0);
        assert_eq!(kind("", None, 0), 1);
        assert_eq!(kind("nodes", None, 0), 2);
        assert_eq!(kind("x", None, 0), 3);
    }

    #[test]
    fn skips() {
        assert!(skip("  # x"));
        assert!(skip("   "));
        assert!(!skip("1 2"));
    }

    #[test]
    fn sorts() {
        assert_eq!(sorted(vec![1, 2]), vec![1, 2]);
    }

    #[test]
    fn levels() {
        assert_eq!(level(&[5, 5]), 1);
        assert_eq!(level(&[5, 6]), 2);
    }

    #[test]
    fn fills() {
        assert_eq!(fill(3), 3);
    }

    #[test]
    fn reaches() {
        assert_eq!(reach(&[(0, 1), (1, 2), (3, 4)], 2), 2);
    }

    #[test]
    fn counts() {
        assert_eq!(count(3), 3);
        assert_eq!(count(0), 0);
    }

    #[test]
    fn depths() {
        assert_eq!(depth(3), 3);
    }

    #[cfg(feature = "x")]
    #[test]
    fn thrice() {
        assert_eq!(super::thrice(2), 6);
    }

    #[test]
    fn labels() {
        assert_eq!(label(&["a", "b"]), "a, b nodes");
    }

    #[test]
    fn codes() {
        assert_eq!(code(true), std::process::ExitCode::from(4));
        assert_eq!(code(false), std::process::ExitCode::from(5));
    }

    #[test]
    fn tags() {
        assert_eq!(tag("x", String::new()), "ex");
        assert_eq!(tag("y", "z".into()), "z");
    }

    #[test]
    fn pushes() {
        let mut v = Vec::new();
        push_name(&mut v);
        assert_eq!(v.len(), 1);
    }
}
