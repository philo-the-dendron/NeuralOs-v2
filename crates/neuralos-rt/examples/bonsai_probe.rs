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
use neuralos_snn::{decode_q1_0, half_to_milli, Trit};

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

    // --- ISC-21: every tensor in bounds; Q1_0 byte sizes exact.
    let mut checked_q1_0 = 0_usize;
    let mut failures = 0_usize;
    for t in &f.tensors {
        let data = match f.tensor_data(t) {
            Ok(d) => d,
            Err(e) => {
                println!("FAIL bounds: {} -> {e}", t.name);
                failures += 1;
                continue;
            }
        };
        if t.ty == GGML_TYPE_Q1_0 {
            // Row-major: dim[0] = contiguous width. rows = product of the rest.
            let cols = t.dims.first().copied().unwrap_or(0);
            let rows: u64 = t.dims.iter().skip(1).product();
            let expected = rows * cols.div_ceil(128) * 18;
            if data.len() as u64 != expected {
                println!(
                    "FAIL size: {} got {} bytes, expected {expected}",
                    t.name,
                    data.len()
                );
                failures += 1;
            } else {
                checked_q1_0 += 1;
            }
        }
    }
    println!(
        "bounds+size: all {} tensors sliced in-bounds; {checked_q1_0} q1_0 tensors byte-exact vs dims",
        f.tensors.len()
    );

    // --- ISC-22: the Stage-2 codec meets real model bytes.
    let emb = f.tensor("token_embd.weight").expect("token_embd.weight exists");
    let data = f.tensor_data(emb).expect("embedding data slice");
    let mut trits = [Trit::Zero; 128];
    let mut scales = [0_u16; 1];
    decode_q1_0(&data[..18], &mut trits, &mut scales).expect("first block decodes");
    let milli = half_to_milli(scales[0]);
    let plus = trits.iter().filter(|t| **t == Trit::One).count();
    let minus = 128 - plus;
    println!(
        "token_embd first block: scale fp16 {:#06x} = {} milli; signs +{}/−{}",
        scales[0], milli, plus, minus
    );

    if failures > 0 {
        println!("PROBE: NO ({failures} failures)");
        std::process::exit(1);
    }
    if !(1..=100).contains(&milli) {
        println!("PROBE: NO (embedding scale milli {milli} outside [1,100])");
        std::process::exit(1);
    }
    println!("PROBE: YES — real Bonsai file reads clean through our container + codec");
}
