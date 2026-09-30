//! The CLI: `neuralos-nir2json [--sim-units] [--dt µs] [--freeze <out.rs>
//! [--steps N] [--input v1,v2,…] [--drive <file>]] <input.nir>
//! <output.json>`.
//!
//! Exit codes: 0 converted, and frozen with `--freeze` · 1 usage/IO · 2
//! named refusal (filter census, out-of-subset node, layout/schema
//! violation; with `--freeze`, a graph that does not assemble, plasticity
//! on, more than 65,535 neurons), nothing written. The sidecar
//! `<output>.meta.json` carries the file-level audit stamp (f32 widening,
//! node census, options); `--freeze` writes `<out.rs>` and `<out>.trace`,
//! prints the library's assembly notes and adds its report to the
//! sidecar (README § Freeze).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use neuralos_nir2json::{
    Assembly, FreezeError, SIM_DT_US, convert_file_opts, effective_options, freeze, module_name,
    stamp_json,
};
use neuralos_snn::nir::NirImportOptions;

/// The run `--freeze` writes when `--steps` is not given.
const DEFAULT_STEPS: u32 = 150;

/// The time step otherwise, the library's default.
const NATIVE_DT_US: u32 = 1_000;

/// The command line, parsed.
struct Args {
    sim_units: bool,
    dt_us: Option<u32>,
    freeze: Option<PathBuf>,
    steps: Option<u32>,
    input: Option<Vec<i16>>,
    drive: Option<PathBuf>,
    paths: Vec<String>,
}

/// The flags and the two paths, or why not.
fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut out = Args {
        sim_units: false,
        dt_us: None,
        freeze: None,
        steps: None,
        input: None,
        drive: None,
        paths: Vec::new(),
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--sim-units" => out.sim_units = true,
            "--dt" => {
                let v = args.next().ok_or("--dt needs µs")?;
                match v.parse::<u32>() {
                    Ok(n) if n > 0 => out.dt_us = Some(n),
                    _ => return Err(format!("--dt {v}: a time step in µs, at least 1")),
                }
            }
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
            "--drive" => out.drive = Some(args.next().ok_or("--drive needs <file>")?.into()),
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            _ => out.paths.push(arg),
        }
    }
    if out.paths.len() != 2 {
        return Err("two paths: <input.nir> <output.json>".into());
    }
    if out.freeze.is_none() && (out.steps.is_some() || out.input.is_some() || out.drive.is_some()) {
        return Err("--steps, --input and --drive go with --freeze".into());
    }
    if out.drive.is_some() && (out.steps.is_some() || out.input.is_some()) {
        return Err("--drive gives the runs and their steps: not with --steps or --input".into());
    }
    Ok(out)
}

/// A `--drive` file's runs: one a line, a step count, at least 1, then one
/// `i16` per input feature, comma-separated (`150 1,0`). Blank lines and
/// lines that start with `#` are skipped; any other line that does not
/// read is refused with its number.
fn parse_drive(text: &str) -> Result<Vec<(u32, Vec<i16>)>, String> {
    let mut runs = Vec::new();
    for (n, line) in (1..).zip(text.lines()) {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let run = match line.split_whitespace().collect::<Vec<_>>()[..] {
            [steps, values] => steps.parse::<u32>().ok().filter(|&s| s > 0).zip(
                values
                    .split(',')
                    .map(|v| v.parse::<i16>())
                    .collect::<Result<Vec<_>, _>>()
                    .ok(),
            ),
            _ => None,
        };
        runs.push(run.ok_or(format!(
            "line {n}: a step count, at least 1, then comma-separated i16 values"
        ))?);
    }
    if runs.is_empty() {
        return Err("no run".into());
    }
    Ok(runs)
}

/// The library's assembly notes, one line each, as the conversion's
/// summary prints: each fused stage, the undriven populations and
/// encoders, the gain note. None when the library noted nothing.
fn assembly_notes(a: &Assembly) -> Vec<String> {
    let mut lines: Vec<String> = a
        .fused
        .iter()
        .map(|f| format!("  fused      : {} (one encoder stage)", f.chain.join(" → ")))
        .collect();
    if !a.undriven.is_empty() {
        lines.push(format!(
            "  undriven   : {} (no input reaches them)",
            a.undriven.join(", ")
        ));
    }
    if a.multi_linear_gain {
        lines.push(
            "  gain note  : more than one drive Linear, each scaled by its own absmax".into(),
        );
    }
    lines
}

/// A count and its noun, the noun plural but for one: "1 dataset",
/// "2 datasets".
fn counted(n: usize, noun: &str) -> String {
    let s = if n == 1 { "" } else { "s" };
    format!("{n} {noun}{s}")
}

fn usage(why: &str) -> ExitCode {
    eprintln!("neuralos-nir2json: {why}");
    eprintln!(
        "usage: neuralos-nir2json [--sim-units] [--dt µs] [--freeze <out.rs> [--steps N] [--input v1,v2,…] [--drive <file>]] <input.nir> <output.json>"
    );
    eprintln!("  --sim-units : interpret LIF parameters in the ecosystem's simulation-unit");
    eprintln!("                 convention (true-scale weights, a voltage scale per node,");
    eprintln!("                 NIR's firing rule, centi grid) — stamped");
    eprintln!("  --dt µs     : the time step, default {SIM_DT_US} with --sim-units (snnTorch's");
    eprintln!("                 exporter assumes 0.1 ms), else {NATIVE_DT_US}");
    eprintln!("  --freeze    : also build the network and write <out.rs>, the arrays a");
    eprintln!("                 FixedNetwork steps, and <out>.trace, their run on the host");
    eprintln!("  --steps N   : the run --freeze writes, default {DEFAULT_STEPS} steps");
    eprintln!("  --input …   : one i16 per input feature for that run, default 1 each");
    eprintln!("  --drive f   : runs instead, one a line: steps, then v1,v2,… (not with");
    eprintln!("                 --steps or --input)");
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

    let dt_us = args.dt_us.unwrap_or(if args.sim_units {
        SIM_DT_US
    } else {
        NATIVE_DT_US
    });
    let opts = NirImportOptions {
        dt_us,
        ..NirImportOptions::default()
    };
    let converted = match convert_file_opts(&input, opts, args.sim_units) {
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
            let opts = effective_options(opts, args.sim_units);
            let runs: Vec<(u32, Option<Vec<i16>>)> = match &args.drive {
                None => vec![(args.steps.unwrap_or(DEFAULT_STEPS), args.input.clone())],
                Some(path) => {
                    let text = match std::fs::read_to_string(path) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("neuralos-nir2json: cannot read {}: {e}", path.display());
                            return ExitCode::from(1);
                        }
                    };
                    match parse_drive(&text) {
                        Ok(runs) => runs.into_iter().map(|(n, v)| (n, Some(v))).collect(),
                        Err(why) => return usage(&format!("--drive {}: {why}", path.display())),
                    }
                }
            };
            let drive: Vec<(u32, Option<&[i16]>)> =
                runs.iter().map(|(n, v)| (*n, v.as_deref())).collect();
            match freeze(&converted.json, opts, &name, &drive) {
                Ok(f) => {
                    let steps: u64 = drive.iter().map(|&(n, _)| u64::from(n)).sum();
                    Some((out_rs.clone(), name, steps, f))
                }
                Err(FreezeError::Input {
                    run,
                    given,
                    features,
                }) => {
                    let what = match &args.drive {
                        None => "--input".to_string(),
                        Some(path) => format!("--drive {}: run {run}", path.display()),
                    };
                    return usage(&format!(
                        "{what} gives {}; the graph has {}, one value each",
                        counted(given, "value"),
                        counted(features, "input feature")
                    ));
                }
                Err(e @ FreezeError::Drive(_)) => return usage(&e.to_string()),
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
    if let Err(code) = write(&output, &converted.json).and_then(|()| {
        let assembly = frozen.as_ref().map(|(_, _, _, f)| &f.assembly);
        write(
            &sidecar,
            stamp_json(&converted.stamp, &input, assembly).as_bytes(),
        )
    }) {
        return code;
    }

    println!(
        "converted {} → {} ({} B; {} nodes, {} edges; f32-widened: {})",
        input.display(),
        output.display(),
        converted.json.len(),
        converted.stamp.node_census.len(),
        converted.stamp.node_census.len().saturating_sub(1),
        counted(converted.stamp.f32_datasets.len(), "dataset"),
    );
    println!("  nir version: {}", converted.stamp.nir_version);
    if converted.stamp.sim_units {
        println!(
            "  sim-units  : transform APPLIED (a voltage scale per node, true scale, centi grid) — see sidecar"
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
        for line in assembly_notes(&f.assembly) {
            println!("{line}");
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::{assembly_notes, counted, parse, parse_drive};
    use neuralos_nir2json::{Assembly, Fused};

    fn args(line: &str) -> Result<super::Args, String> {
        parse(line.split_whitespace().map(String::from))
    }

    #[test]
    fn a_drive_file_reads_its_runs_and_skips_blank_and_comment_lines() {
        let text = "# two bursts\n150 1,0\n\n   \n  # indented\n  3   0,-2 \n";
        assert_eq!(
            parse_drive(text),
            Ok(vec![(150, vec![1, 0]), (3, vec![0, -2])])
        );
    }

    #[test]
    fn a_drive_file_refuses_a_line_by_its_number() {
        for (text, n) in [
            ("1 1\n0 1\n", 2),
            ("1 1\n1 1,x\n", 2),
            ("5\n", 1),
            ("1 1 1\n", 1),
            ("1 70000\n", 1),
        ] {
            let why = parse_drive(text).expect_err("refused");
            assert!(why.starts_with(&format!("line {n}:")), "{text:?}: {why}");
        }
        assert_eq!(parse_drive("# nothing\n"), Err("no run".to_string()));
    }

    #[test]
    fn drive_goes_with_freeze_and_not_with_steps_or_input() {
        assert!(args("--freeze g.rs --drive d.txt a.nir a.json").is_ok());
        for line in [
            "--drive d.txt a.nir a.json",
            "--freeze g.rs --drive d.txt --steps 5 a.nir a.json",
            "--freeze g.rs --drive d.txt --input 1 a.nir a.json",
        ] {
            assert!(args(line).is_err(), "{line}");
        }
    }

    #[test]
    fn steps_and_input_each_go_with_freeze() {
        for flag in ["--steps 5", "--input 1"] {
            assert!(args(&format!("--freeze g.rs {flag} a.nir a.json")).is_ok());
            assert_eq!(
                args(&format!("{flag} a.nir a.json")).err().as_deref(),
                Some("--steps, --input and --drive go with --freeze"),
                "{flag}"
            );
        }
    }

    #[test]
    fn a_time_step_of_one_microsecond_is_a_time_step() {
        assert_eq!(args("--dt 1 a.nir a.json").map(|a| a.dt_us), Ok(Some(1)));
        assert!(args("--dt 0 a.nir a.json").is_err());
    }

    #[test]
    fn the_summary_counts_datasets_in_the_plural_but_one() {
        assert_eq!(
            [
                counted(0, "dataset"),
                counted(1, "dataset"),
                counted(2, "dataset")
            ],
            ["0 datasets", "1 dataset", "2 datasets"]
        );
    }

    #[test]
    fn the_assembly_notes_name_what_the_library_noted() {
        let mut a = Assembly {
            neurons: 3,
            synapses: 1,
            inputs: 1,
            drive_linears: 1,
            stages: 1,
            fused: vec![],
            undriven: vec![],
            multi_linear_gain: false,
            plasticity_frozen: true,
        };
        assert!(assembly_notes(&a).is_empty(), "a graph with nothing noted");
        a.fused.push(Fused {
            chain: vec!["l1".into(), "l2".into()],
            scales: vec![1.0, 1.0],
        });
        a.undriven = vec!["c".into(), "d".into()];
        a.multi_linear_gain = true;
        assert_eq!(
            assembly_notes(&a),
            [
                "  fused      : l1 → l2 (one encoder stage)",
                "  undriven   : c, d (no input reaches them)",
                "  gain note  : more than one drive Linear, each scaled by its own absmax",
            ]
        );
    }
}
