// Link with link.x, in this directory.
//
// Done here, not as `rustflags` in .cargo/config.toml: a RUSTFLAGS
// variable in the environment replaces config rustflags wholesale, and
// the link then fails on `_stack_top` (proofs/qemu-trace-replay/build.rs
// has the story). A build-script link-arg survives any environment. The
// config's other flag, `-C relocation-model=static`, went with it: it is
// the target's default, and the ELF is the same without it.
fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    println!("cargo:rustc-link-search={dir}");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=link.x");
}
