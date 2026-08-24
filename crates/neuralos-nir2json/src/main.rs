//! The CLI: `neuralos-nir2json <input.nir> <output.json>`.
//!
//! Exit codes: 0 converted · 1 usage/IO · 2 named refusal (filter
//! census, out-of-subset node, layout/schema violation). The sidecar
//! `<output>.meta.json` carries the file-level audit stamp (f32
//! widening, node census, options).

use std::process::ExitCode;

use neuralos_nir2json::{convert_file_opts, stamp_json};
use neuralos_snn::nir::NirImportOptions;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sim_units = args.iter().any(|a| a == "--sim-units");
    let positional: Vec<&String> = args.iter().filter(|a| *a != "--sim-units").collect();
    if positional.len() != 2 {
        eprintln!("usage: neuralos-nir2json [--sim-units] <input.nir> <output.json>");
        eprintln!("  --sim-units : interpret LIF parameters in the ecosystem's simulation-unit");
        eprintln!("                 convention (r×1000 → MΩ, voltages as mV, centi grid) — stamped");
        eprintln!("  exit 0: converted (sidecar <output>.meta.json written)");
        eprintln!("  exit 1: usage / IO error");
        eprintln!("  exit 2: named refusal — filter census, out-of-subset node, layout,");
        eprintln!("          sim-unit parameters without --sim-units");
        return ExitCode::from(1);
    }
    let (input, output) = (
        std::path::PathBuf::from(positional[0]),
        std::path::PathBuf::from(positional[1]),
    );

    let converted = match convert_file_opts(&input, NirImportOptions::default(), sim_units) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("neuralos-nir2json: REFUSED — {e}");
            return ExitCode::from(2);
        }
    };

    if let Err(e) = std::fs::write(&output, &converted.json) {
        eprintln!("neuralos-nir2json: cannot write {}: {e}", output.display());
        return ExitCode::from(1);
    }
    let mut sidecar = output.clone().into_os_string();
    sidecar.push(".meta.json");
    let sidecar = std::path::PathBuf::from(sidecar);
    if let Err(e) = std::fs::write(&sidecar, stamp_json(&converted.stamp, &input)) {
        eprintln!("neuralos-nir2json: cannot write {}: {e}", sidecar.display());
        return ExitCode::from(1);
    }

    println!(
        "converted {} → {} ({} B; {} nodes, {} edges; f32-widened: {} dataset{})",
        input.display(),
        output.display(),
        converted.json.len(),
        converted.stamp.node_census.len(),
        converted.stamp.node_census.len().saturating_sub(1),
        converted.stamp.f32_datasets.len(),
        if converted.stamp.f32_datasets.is_empty() { "s" } else { "" },
    );
    println!("  nir version: {}", converted.stamp.nir_version);
    if converted.stamp.sim_units {
        println!("  sim-units  : transform APPLIED (r×1000 → MΩ, V as mV, centi grid) — see sidecar");
    }
    println!("  sidecar    : {}", sidecar.display());
    ExitCode::SUCCESS
}
