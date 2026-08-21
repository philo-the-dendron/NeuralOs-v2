// Structural encoding of the HDF5 plugin path (never README-only).
//
// The vendored HDF5 build's compiled-in plugin search directory
// (`/usr/local/hdf5/lib/plugin`) never exists on dev boxes or CI, and
// the HDF5 library wants `HDF5_PLUGIN_PATH` to name an existing
// directory at file-open time (ISA R7 pre-flight finding). This script
// records the repo's plugin directory as a compile-time env var; the
// runtime helper `nir_hdf5::ensure_plugin_dir` points
// `HDF5_PLUGIN_PATH` there (creating it if absent) before any HDF5
// call. The dir stays empty until a plugin actually ships — gzip and
// lzf-census files need no plugins (deflate is compiled in by `zlib`;
// the census rejects uncompiled filters before any read).

use std::{env, fs, path::PathBuf};

fn main() {
    let plugin_dir: PathBuf = [&env::var("CARGO_MANIFEST_DIR").unwrap(), "..", "..", "tools", "hdf5-plugins"]
        .iter()
        .collect();
    println!(
        "cargo:rustc-env=NEURALOS_HDF5_PLUGIN_DIR={}",
        plugin_dir.display()
    );
    // The dir must exist for HDF5_PLUGIN_PATH to be valid at runtime;
    // the runtime helper also creates it, but make the build-time state
    // true as well so fresh checkouts never see a dangling path.
    let _ = fs::create_dir_all(&plugin_dir);
    println!("cargo:rerun-if-changed=build.rs");
}
