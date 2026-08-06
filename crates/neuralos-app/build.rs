// Compile the Slint UI at build time. Non-fatal-free: errors stop the build
// with a readable message, never an unchecked panic.
fn main() {
    if let Err(e) = slint_build::compile("ui/app.slint") {
        eprintln!("slint-build: failed to compile ui/app.slint:\n{e}");
        std::process::exit(1);
    }
}
