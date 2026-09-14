// Link with link.x, in this directory.
//
// Done here, not as `rustflags` in .cargo/config.toml: a RUSTFLAGS variable
// in the environment replaces config rustflags wholesale, and the CI
// workflow sets RUSTFLAGS="-D warnings" for every job (the firmware's
// build.rs has the story, PR #13 run 180). A build-script link-arg survives
// any environment.
fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    println!("cargo:rustc-link-search={dir}");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=link.x");
}
