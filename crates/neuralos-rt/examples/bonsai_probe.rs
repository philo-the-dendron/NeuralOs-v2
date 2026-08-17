//! Stage 4, session 1 probe: point the Stage-2 codecs at a REAL Bonsai
//! model file. The evidence this prints is ISC-21/22's falsifier run:
//!
//! - container parses (GGUF v3, N tensors, M KV pairs)
//! - `general.architecture` says what the model thinks it is
//! - every tensor's data slice falls inside the file
//! - every Q1_0 tensor's byte size matches `rows × (cols/128) × 18`
//! - `token_embd.weight`'s first block decodes through
//!   `bridge::decode_q1_0` with a sane fp16 scale (milli view)
//!
//! Usage: `cargo run -p neuralos-rt --example bonsai_probe -- [path]`
//! (default `models/Bonsai-1.7B-Q1_0.gguf`).

use neuralos_rt::{GgufFile, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0};
use neuralos_snn::{decode_q1_0, decode_q2_0, half_to_milli, Trit};

fn type_name(ty: u32) -> String {
    match ty {
        0 => "f32".into(),
        1 => "f16".into(),
        8 => "q8_0".into(),
        30 => "bf16".into(),
        34 => "tq1_0".into(),
        t if t == GGML_TYPE_Q1_0 => "q1_0".into(),
        t if t == GGML_TYPE_Q2_0 => "q2_0".into(),
        t => format!("type({t})"),
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Bonsai-1.7B-Q1_0.gguf".into());
    let buf = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    println!("file: {path} ({} bytes)", buf.len());

    let f = GgufFile::parse(&buf).expect("GGUF container must parse");
    println!(
        "container: v{}, {} tensors, {} KV pairs, alignment {}, data @ {}",
        f.version,
        f.tensors.len(),
        f.kv.len(),
        f.alignment,
        f.data_start
    );

    let arch = match f.value("general.architecture") {
        Some(neuralos_rt::MetadataValue::String(s)) => s.clone(),
        _ => "<missing>".into(),
    };
    println!("architecture: {arch}");

    // Type census — the whole model at a glance.
    let mut census = std::collections::BTreeMap::new();
    for t in &f.tensors {
        *census.entry(type_name(t.ty)).or_insert(0_u32) += 1;
    }
    let census_str: Vec<String> = census
        .iter()
        .map(|(k, v)| format!("{k}×{v}"))
        .collect();
    println!("tensor types: {}", census_str.join(", "));

    // First + a few interesting tensor infos.
    for t in f.tensors.iter().take(3) {
        println!(
            "  {} dims={:?} {} @ +{}",
            t.name,
            t.dims,
            type_name(t.ty),
            t.offset
        );
    }

    // --- ISC-21: every tensor in bounds; per-type byte sizes exact.
    // Session D: the check is per-format — q1_0 blocks are 128 w / 18 B,
    // q2_0 blocks are 64 w / 18 B (docs/TERNARY_FORMAT.md §q2_0). The
    // q2_0 arithmetic IS the group-64 pin: a g128-laid-out file would
    // fail every tensor here.
    let mut checked_q1_0 = 0_usize;
    let mut total_q1_0 = 0_usize;
    let mut checked_q2_0 = 0_usize;
    let mut total_q2_0 = 0_usize;
    let mut failures = 0_usize;
    let mut padded_accepts: Vec<String> = Vec::new();
    for t in &f.tensors {
        let data = match f.tensor_data(t) {
            Ok(d) => d,
            Err(e) => {
                println!("FAIL bounds: {} -> {e}", t.name);
                failures += 1;
                continue;
            }
        };
        if t.ty == GGML_TYPE_Q1_0 || t.ty == GGML_TYPE_Q2_0 {
            // (weights per block, bytes per block): q1_0 = 128/18
            // (fp16 γ + 16 sign bytes), q2_0 = 128/34 (fp16 max|w| d +
            // 32 code bytes — session-D re-pin from the fork's
            // ggml-common.h QK2_0 = 128; the first real q2_0 file
            // measures 680 B per 2560-wide row, refuting 64/18).
            let (qk, bpb): (u128, u128) = if t.ty == GGML_TYPE_Q1_0 {
                (128, 18)
            } else {
                (128, 34)
            };
            let (checked, total) = if t.ty == GGML_TYPE_Q1_0 {
                (&mut checked_q1_0, &mut total_q1_0)
            } else {
                (&mut checked_q2_0, &mut total_q2_0)
            };
            *total += 1;
            // Row-major: dim[0] = contiguous width. rows = product of the
            // rest. u128 so hostile dims cannot overflow the expected-size
            // product into a wrapped pass (review finding). The slice may
            // carry alignment padding (the 4B's token_embd sits 24 B short
            // of a 32-byte boundary) — accept the formula size or its
            // alignment-rounded-up form, nothing else.
            let cols = t.dims.first().copied().unwrap_or(0);
            let rows: u128 = t.dims.iter().skip(1).map(|&d| d as u128).product::<u128>().max(1);
            let expected: u128 = rows * ((cols as u128).div_ceil(qk)) * bpb;
            let align = f.alignment.max(1) as u128;
            let padded = expected.div_ceil(align) * align;
            if data.len() as u128 != expected && data.len() as u128 != padded {
                println!(
                    "FAIL size: {} got {} bytes, expected {expected} (or {padded} padded)",
                    t.name,
                    data.len()
                );
                failures += 1;
            } else {
                *checked += 1;
                if data.len() as u128 != expected {
                    padded_accepts.push(t.name.clone());
                }
            }
        }
    }
    // Full config-block provenance (4B session): dump every qwen3.* /
    // general.* KV verbatim — the config diff vs 1.7B is evidence, and
    // load() cross-checks the ones it consumes. Strings truncate to
    // 60 chars (chat templates are multi-KB).
    for (k, v) in &f.kv {
        if k.starts_with("qwen3.") || k.starts_with("general.") {
            let vs = match v {
                neuralos_rt::MetadataValue::String(s) => {
                    format!("{:?}", &s.chars().take(60).collect::<String>())
                }
                other => format!("{other:?}"),
            };
            println!("config kv: {k} = {vs}");
        }
    }
    if failures == 0 {
        println!(
            "bounds+size: all {} tensors sliced in-bounds; {checked_q1_0}/{total_q1_0} q1_0 + {checked_q2_0}/{total_q2_0} q2_0 tensors byte-exact vs dims",
            f.tensors.len()
        );
    } else {
        println!(
            "bounds+size: {} failure(s) across {} tensors ({checked_q1_0}/{total_q1_0} q1_0 + {checked_q2_0}/{total_q2_0} q2_0 byte-exact)",
            failures,
            f.tensors.len()
        );
    }
    if padded_accepts.is_empty() {
        println!("padding: no tensor needed alignment-padded acceptance");
    } else {
        println!(
            "padding: {} tensor(s) accepted alignment-padded (formula bytes only): {:?}",
            padded_accepts.len(),
            padded_accepts
        );
    }

    // --- ISC-22: the Stage-2 codec meets real model bytes — on BOTH
    // tiers it ships for. The embedding tensor's own type picks the
    // codec; q2_0 is where decode_q2_0 first eats real file bytes
    // (session D closes the Stage-2 gap).
    let emb = f.tensor("token_embd.weight").expect("token_embd.weight exists");
    let data = f.tensor_data(emb).expect("embedding data slice");
    let milli;
    if emb.ty == GGML_TYPE_Q2_0 {
        let mut trits = [Trit::Zero; 128];
        let mut scales = [0_u16; 1];
        decode_q2_0(&data[..34], &mut trits, &mut scales).expect("first q2_0 block decodes");
        milli = half_to_milli(scales[0]);
        let plus = trits.iter().filter(|t| **t == Trit::One).count();
        let zero = trits.iter().filter(|t| **t == Trit::Zero).count();
        let minus = trits.iter().filter(|t| **t == Trit::MinusOne).count();
        println!(
            "token_embd first block (q2_0): scale fp16 {:#06x} = {} milli (max|w|); trits +{}/0×{}/−{} of 128",
            scales[0], milli, plus, zero, minus
        );
    } else {
        let mut trits = [Trit::Zero; 128];
        let mut scales = [0_u16; 1];
        decode_q1_0(&data[..18], &mut trits, &mut scales).expect("first block decodes");
        milli = half_to_milli(scales[0]);
        let plus = trits.iter().filter(|t| **t == Trit::One).count();
        let minus = 128 - plus;
        println!(
            "token_embd first block: scale fp16 {:#06x} = {} milli; signs +{}/−{}",
            scales[0], milli, plus, minus
        );
    }

    if failures > 0 {
        println!("PROBE: NO ({failures} failures)");
        std::process::exit(1);
    }
    // Empirical order-of-magnitude smoke window. Q1_0 (γ = mean|w|):
    // the two real files read 27/19 milli — window [1,100], recorded in
    // ISA ISC-22. Q2_0 (d = max|w|): first real file this session; the
    // max convention runs larger than the mean, so the provisional
    // window is one decade wider — the observed value is recorded in
    // the ISA and narrows this for the next file (fog (e)).
    let (lo, hi) = if emb.ty == GGML_TYPE_Q2_0 { (1, 1000) } else { (1, 100) };
    if !(lo..=hi).contains(&milli) {
        println!("PROBE: NO (embedding scale milli {milli} outside [{lo},{hi}])");
        std::process::exit(1);
    }
    println!("PROBE: YES — real Bonsai file reads clean through our container + codec");
}
