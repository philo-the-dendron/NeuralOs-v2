//! The CLI: `neuralos-nir2json [--sim-units] [--freeze <out.rs> [--steps N]
//! [--input v1,v2,…]] <input.nir> <output.json>`.
//!
//! Exit codes: 0 converted, and frozen with `--freeze` · 1 usage/IO · 2
//! named refusal (filter census, out-of-subset node, layout/schema
//! violation; with `--freeze`, a graph that does not assemble, plasticity
//! on, more than 65,535 neurons), nothing written. The sidecar
//! `<output>.meta.json` carries the file-level audit stamp (f32 widening,
//! node census, options); `--freeze` writes `<out.rs>` and `<out>.trace`
//! (README § Freeze).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use neuralos_nir2json::{
    FreezeError, convert_file_opts, effective_options, freeze, module_name, stamp_json,
};
use neuralos_snn::nir::NirImportOptions;

/// The run `--freeze` writes when `--steps` is not given.
const DEFAULT_STEPS: u32 = 150;

/// The command line, parsed.
struct Args {
    sim_units: bool,
    freeze: Option<PathBuf>,
    steps: Option<u32>,
    input: Option<Vec<i16>>,
    paths: Vec<String>,
}

/// The flags and the two paths, or why not.
fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut out = Args {
        sim_units: false,
        freeze: None,
        steps: None,
        input: None,
        paths: Vec::new(),
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--sim-units" => out.sim_units = true,
            "--freeze" => out.freeze = Some(args.next().ok_or("--freeze needs <out.rs>")?.into()),
            "--steps" => {
                let v = args.next().ok_or("--steps needs N")?;
                match v.parse::<u32>() {
                    Ok(n) if n > 0 => out.steps = Some(n),
                    _ => return Err(format!("--steps {v}: a step count, at least 1")),
                }
            }
            "--input" => {
                let v = args.next().ok_or("--input needs v1,v2,…")?;
                let values: Result<Vec<i16>, _> =
                    v.split(',').map(|x| x.trim().parse::<i16>()).collect();
                out.input =
                    Some(values.map_err(|_| format!("--input {v}: comma-separated i16 values"))?);
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            _ => out.paths.push(arg),
        }
    }
    if out.paths.len() != 2 {
        return Err("two paths: <input.nir> <output.json>".into());
    }
    if out.freeze.is_none() && (out.steps.is_some() || out.input.is_some()) {
        return Err("--steps and --input go with --freeze".into());
    }
    Ok(out)
}

fn usage(why: &str) -> ExitCode {
    eprintln!("neuralos-nir2json: {why}");
    eprintln!(
        "usage: neuralos-nir2json [--sim-units] [--freeze <out.rs> [--steps N] [--input v1,v2,…]] <input.nir> <output.json>"
    );
    eprintln!("  --sim-units : interpret LIF parameters in the ecosystem's simulation-unit");
    eprintln!("                 convention (r×1000 → MΩ, voltages as mV, centi grid) — stamped");
    eprintln!("  --freeze    : also build the network and write <out.rs>, the arrays a");
    eprintln!("                 FixedNetwork steps, and <out>.trace, their run on the host");
    eprintln!("  --steps N   : the run --freeze writes, default {DEFAULT_STEPS} steps");
    eprintln!("  --input …   : one i16 per input feature for that run, default 1 each");
    eprintln!("  exit 0: converted (sidecar <output>.meta.json written), frozen with --freeze");
    eprintln!("  exit 1: usage / IO error");
    eprintln!("  exit 2: named refusal — filter census, out-of-subset node, layout,");
    eprintln!("          sim-unit parameters without --sim-units; with --freeze, a graph");
    eprintln!("          that does not assemble, plasticity on, more than 65,535 neurons");
    ExitCode::from(1)
}

/// Write a file, or say why not: exit 1.
fn write(path: &Path, bytes: &[u8]) -> Result<(), ExitCode> {
    std::fs::write(path, bytes).map_err(|e| {
        eprintln!("neuralos-nir2json: cannot write {}: {e}", path.display());
        ExitCode::from(1)
    })
}

fn main() -> ExitCode {
    let args = match parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(why) => return usage(&why),
    };
    let (input, output) = (PathBuf::from(&args.paths[0]), PathBuf::from(&args.paths[1]));

    let converted = match convert_file_opts(&input, NirImportOptions::default(), args.sim_units) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("neuralos-nir2json: REFUSED — {e}");
            return ExitCode::from(2);
        }
    };

    // --freeze runs whole, in memory, before any file is written: a
    // refusal leaves nothing behind.
    let frozen = match &args.freeze {
        None => None,
        Some(out_rs) => {
            let Some(name) = out_rs
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(module_name)
            else {
                return usage(&format!(
                    "--freeze {}: its file stem gives no module name (a letter or _ first, not a Rust keyword)",
                    out_rs.display()
                ));
            };
            let opts = effective_options(NirImportOptions::default(), args.sim_units);
            let steps = args.steps.unwrap_or(DEFAULT_STEPS);
            match freeze(&converted.json, opts, &name, steps, args.input.as_deref()) {
                Ok(f) => Some((out_rs.clone(), name, steps, f)),
                Err(e @ FreezeError::Input { .. }) => return usage(&e.to_string()),
                Err(e) => {
                    eprintln!("neuralos-nir2json: REFUSED — {e}");
                    return ExitCode::from(2);
                }
            }
        }
    };

    let mut sidecar = output.clone().into_os_string();
    sidecar.push(".meta.json");
    let sidecar = PathBuf::from(sidecar);
    if let Err(code) = write(&output, &converted.json)
        .and_then(|()| write(&sidecar, stamp_json(&converted.stamp, &input).as_bytes()))
    {
        return code;
    }

    println!(
        "converted {} → {} ({} B; {} nodes, {} edges; f32-widened: {} dataset{})",
        input.display(),
        output.display(),
        converted.json.len(),
        converted.stamp.node_census.len(),
        converted.stamp.node_census.len().saturating_sub(1),
        converted.stamp.f32_datasets.len(),
        if converted.stamp.f32_datasets.is_empty() {
            "s"
        } else {
            ""
        },
    );
    println!("  nir version: {}", converted.stamp.nir_version);
    if converted.stamp.sim_units {
        println!(
            "  sim-units  : transform APPLIED (r×1000 → MΩ, V as mV, centi grid) — see sidecar"
        );
    }
    println!("  sidecar    : {}", sidecar.display());

    if let Some((out_rs, name, steps, f)) = frozen {
        let trace = out_rs.with_extension("trace");
        if let Err(code) =
            write(&out_rs, f.module.as_bytes()).and_then(|()| write(&trace, f.trace.as_bytes()))
        {
            return code;
        }
        println!(
            "  frozen     : {} (module {name}: {} neurons, {} synapses, {steps} steps)",
            out_rs.display(),
            f.neurons,
            f.synapses,
        );
        println!("  trace      : {}", trace.display());
    }
    ExitCode::SUCCESS
}
