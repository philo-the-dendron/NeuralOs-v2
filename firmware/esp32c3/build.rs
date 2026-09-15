// Link against esp-hal's linker scripts (linkall.x pulls in the chip memory
// map, riscv-rt, and the ROM symbol tables).
//
// Done here, not as `rustflags` in .cargo/config.toml: a RUSTFLAGS variable
// in the environment replaces config rustflags wholesale, and the CI
// workflow sets RUSTFLAGS="-D warnings" for every job. PR #13 run 180
// linked without linkall.x for that reason ("undefined symbol:
// _stack_start" and friends). A build-script link-arg survives any
// environment.
//
// The stranger slot (PR H): NEURALOS_GRAPH names a module written by
// `neuralos-nir2json --freeze`. This script copies it into OUT_DIR as
// graph.rs with one line more, `pub use self::<name> as graph;`, and sets
// `cfg(stranger)`, under which main.rs includes it and replays it after
// the frozen set. A file that is missing, unreadable, or holds not exactly
// one line `pub mod <name> {` fails the build by name, never falls back to
// the default build: a silent default would flash the fourteen-case
// firmware to an operator who believes it is theirs. Two artifacts
// concatenated would smuggle a second module and a second
// `for_each_frozen!`, and no such line means the wrong file. Unset,
// nothing is written and no path is printed; main.rs's include is
// cfg-stripped before any file is read, so a stale graph.rs is inert. The
// variable reruns this script when it changes, and cargo then rebuilds
// the crate, which is why the stranger build owns a target dir.
use std::env;
use std::fs;
use std::path::Path;
use std::process::exit;

fn main() {
    println!("cargo:rustc-link-arg-bins=-Tlinkall.x");
    println!("cargo:rerun-if-changed=build.rs");
    stranger_slot();
}

/// The slot (the header). Its first two lines print on every build: the
/// rerun guard, and the cfg declared, since an undeclared cfg fires
/// `unexpected_cfgs`, which CI's `-D warnings` makes an error.
fn stranger_slot() {
    println!("cargo:rerun-if-env-changed=NEURALOS_GRAPH");
    println!("cargo:rustc-check-cfg=cfg(stranger)");
    let Some(given) = env::var_os("NEURALOS_GRAPH") else {
        return;
    };
    let path = fs::canonicalize(&given).unwrap_or_else(|e| fail(Path::new(&given), &e.to_string()));
    println!("cargo:rerun-if-changed={}", path.display());
    let text = fs::read_to_string(&path).unwrap_or_else(|e| fail(&path, &e.to_string()));
    let names: Vec<&str> = text
        .lines()
        .filter_map(|line| line.strip_prefix("pub mod ")?.strip_suffix(" {"))
        .filter(|name| !name.is_empty() && !name.contains(char::is_whitespace))
        .collect();
    let [name] = names.as_slice() else {
        fail(
            &path,
            &format!(
                "{} lines `pub mod <name> {{`, not one: not a `neuralos-nir2json --freeze` module",
                names.len()
            ),
        );
    };
    let mut graph = text.clone();
    if !graph.ends_with('\n') {
        graph.push('\n');
    }
    graph.push_str(&format!("pub use self::{name} as graph;\n"));
    let out = Path::new(&env::var_os("OUT_DIR").expect("cargo sets OUT_DIR")).join("graph.rs");
    fs::write(&out, graph).unwrap_or_else(|e| fail(&out, &e.to_string()));
    println!("cargo:rustc-cfg=stranger");
}

/// A hard build error that names the path.
fn fail(path: &Path, why: &str) -> ! {
    eprintln!("NEURALOS_GRAPH: {}: {why}", path.display());
    exit(1);
}
