// Link with link.x, in this directory (Leg A's, copied).
//
// Done here, not as `rustflags` in .cargo/config.toml as Leg A does: a
// RUSTFLAGS variable in the environment replaces config rustflags
// wholesale (the firmware's build.rs has the story, PR #13 run 180). A
// build-script link-arg survives any environment. Leg A's other flag,
// `-C relocation-model=static`, is the target's default already
// (`rustc --print target-spec-json` on 1.98.1: static, code model
// medium), so nothing else is needed.
fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    println!("cargo:rustc-link-search={dir}");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=link.x");
}
