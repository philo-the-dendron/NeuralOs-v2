// Link against esp-hal's linker scripts (linkall.x pulls in the chip memory
// map, riscv-rt, and the ROM symbol tables).
//
// Done here, not as `rustflags` in .cargo/config.toml: a RUSTFLAGS variable
// in the environment replaces config rustflags wholesale, and the CI
// workflow sets RUSTFLAGS="-D warnings" for every job. PR #13 run 180
// linked without linkall.x for that reason ("undefined symbol:
// _stack_start" and friends). A build-script link-arg survives any
// environment.
fn main() {
    println!("cargo:rustc-link-arg-bins=-Tlinkall.x");
    println!("cargo:rerun-if-changed=build.rs");
}
