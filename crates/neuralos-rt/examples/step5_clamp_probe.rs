//! Step-5 prep instrument (Rider B): the TRUE pre-clamp drive distribution,
//! measured — not projected.
//!
//! Why this exists: the CLAMP-RELAXED arm (PREREG §7) must pick a ceiling
//! BEFORE burning a 6–8 h run, and the banked H2 histogram cannot project
//! it — run.log's buckets 100–600 µA are all zero and everything above is
//! merged with "railed". But the drive source is attn_norm(embedding)
//! (ISA sH registration) — layer-0 arithmetic only. The H2 run computed it
//! through `forward_block_states` (full 36-layer forward, ~8 h) because
//! that path existed; mathematically it is an embedding-row lookup
//! (`q2_0_row_to_milli`, the same pub decode the model path delegates to)
//! + `rms_norm_milli` with `blk.0.attn_norm.weight`.
//!
//! This probe computes exactly that, in seconds.
//!
//! Self-verification: window r0 (tokens [0,2000)) MUST reproduce every
//! banked H2 pin (run.log): k = 10060.46 µA/unit, 4411 corpus tokens,
//! clamp 568321/818000 = 69.477% at ±1000, hist
//! [249679, 0, 0, 0, 568321], hottest dim 199 railed 1786×. If any pin
//! misses, the probe is wrong (loud assert) — the probe is never trusted
//! over the banked record.
//!
//! Findings this reports (per window r0/r1/r2):
//! - the derived k per window (Rider A procedure pins),
//! - the |I| pre-clamp distribution (percentiles + max),
//! - clamp fractions at ceilings 1000/2000/3000/10000/32767 (the i16
//!   input rail — `c as i16` saturates there regardless of clamp config),
//! - the domain-corrected counterfactual (k applied to norm units, not
//!   milli) — line 343 of hybrid_invivo.rs multiplies MILLI values by a k
//!   derived from NORM-UNIT RMS, so the pre-clamp drive is ~1000× the
//!   450 µA target; the corrected-domain row shows what the registration
//!   actually intended.
//!
//! Usage (recorded in evidence/step5-readout/):
//!   cargo run -p neuralos-rt --release --example step5_clamp_probe \
//!     [model.gguf] [corpus.txt]

use neuralos_rt::{f32_bits_to_milli, q2_0_row_to_milli, rms_norm_milli, GgufFile, Tokenizer};

const DRIVEN_DIMS: usize = 409;
const TARGET_RMS_UA: f64 = 450.0;
const H2_STEPS: usize = 2000;
/// Replicate windows (PREREG §4 + the escalation amendment): r0–r2
/// contiguous; r3/r4 WRAP the 4,411-token corpus (r3 = [3000,4411) +
/// [0,589) · r4 = [4000,4411) + [0,1589) — exactly 2,000 steps each,
/// dose-comparable). The wrap is materialized once below so every
/// window slice is contiguous in the working buffer.
const WINDOWS: [(usize, &str); 5] = [
    (0, "r0"),
    (1000, "r1"),
    (2000, "r2"),
    (3000, "r3 (wraps)"),
    (4000, "r4 (wraps)"),
];

// Banked H2 pins (evidence/session-h2/run.log) — window r0 only.
const H2_K: f64 = 10060.46;
const H2_TOKENS: usize = 4411;
const H2_CLAMPED: u64 = 568_321;
const H2_TOTAL: u64 = 818_000;
const H2_HIST: [u64; 5] = [249_679, 0, 0, 0, 568_321];
const H2_TOP_DIM: usize = 199;
const H2_TOP_DIM_RAILS: u64 = 1786;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Ternary-Bonsai-4B-Q2_0.gguf".into());
    let corpus_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "evidence/corpus_readme_pinned.txt".into());

    let buf = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let f = GgufFile::parse(&buf).expect("gguf parse");
    let tok = Tokenizer::from_gguf(&f).expect("tokenizer loads");
    let corpus = std::fs::read_to_string(&corpus_path).expect("corpus reads");
    let ids = tok.encode(&corpus);
    println!("file    : {path}");
    println!("tokens  : {} (pin {H2_TOKENS})", ids.len());
    assert_eq!(
        ids.len(),
        H2_TOKENS,
        "corpus token count must match the banked pin"
    );

    // token_embd.weight: rows × emb, Q2_0 (34 B per 128-wide block).
    let emb_t = f
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd.weight present");
    assert_eq!(
        emb_t.ty,
        neuralos_rt::GGML_TYPE_Q2_0,
        "token_embd must be Q2_0 in this file"
    );
    assert_eq!(emb_t.dims.len(), 2, "token_embd is 2-D");
    let emb: usize = emb_t.dims[0] as usize; // dims[0] = width (model.rs:665 convention)
    let vocab: usize = emb_t.dims[1] as usize;
    assert_eq!(vocab, 151_669, "vocab pin (tensor-derived, ISA ISC-51)");
    let row_bytes = emb.div_ceil(128) * 34;
    let emb_data = f.tensor_data(emb_t).expect("emb slice");

    // blk.0.attn_norm.weight (f32) → milli — the same conversion the
    // in-vivo path runs (hybrid_invivo.rs:291-296).
    let norm_t = f
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_norm.weight")
        .expect("attn_norm tensor present");
    let norm_milli: Vec<i32> = {
        let d = f.tensor_data(norm_t).expect("norm slice");
        assert_eq!(d.len(), emb * 4, "attn_norm width == emb");
        d.chunks_exact(4)
            .map(|c| f32_bits_to_milli(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()
    };

    // Decode + attn_norm every needed token once (windows overlap).
    // The WRAPPED windows (r3/r4) read tokens [0,1589) again — the
    // corpus is fully covered by the union, so one pass over all
    // 4,411 tokens serves every window; wrap slicing indexes modulo.
    let n_tokens = ids.len();
    let mut h_norm: Vec<Vec<i32>> = Vec::with_capacity(n_tokens);
    let mut row_milli = vec![0_i32; emb];
    let mut normed = vec![0_i32; emb];
    for &t in &ids[..n_tokens] {
        let r = t as usize;
        let row = &emb_data[r * row_bytes..][..row_bytes];
        q2_0_row_to_milli(row, &mut row_milli).expect("emb row decode");
        rms_norm_milli(&row_milli, &norm_milli, &mut normed);
        h_norm.push(normed.clone());
    }

    for (off, name) in WINDOWS {
        // wrap-aware contiguous view: the drive order is the window's
        // own token order (tail of the corpus, then head) — exactly
        // what a wrapped run consumes, step for step.
        let win: Vec<&Vec<i32>> = (0..H2_STEPS)
            .map(|i| &h_norm[(off + i) % n_tokens])
            .collect();
        let total: u64 = (H2_STEPS * DRIVEN_DIMS) as u64;

        // k derivation — H2's frozen procedure (Rider A): RMS over driven
        // dims in NORM units, k = target / rms.
        let mut sum_sq = 0.0_f64;
        for row in &win {
            for &v in &row[..DRIVEN_DIMS] {
                sum_sq += (v as f64 / 1000.0).powi(2);
            }
        }
        let rms_units = (sum_sq / total as f64).sqrt();
        let k = TARGET_RMS_UA / rms_units;

        // The pre-clamp currents, exactly as hybrid_invivo.rs:343 builds
        // them: raw = milli × k (the milli-domain multiplication under
        // study). Also the corrected-domain twin: raw_c = milli/1000 × k.
        let mut abs_raw: Vec<f64> = Vec::with_capacity(total as usize);
        let mut rail_dim = vec![0u64; DRIVEN_DIMS];
        let mut hist = [0u64; 5];
        let mut clamped: u64 = 0;
        let mut corrected_clamped: u64 = 0;
        for row in &win {
            for (d, &v) in row[..DRIVEN_DIMS].iter().enumerate() {
                let raw = v as f64 * k;
                let a = raw.abs();
                abs_raw.push(a);
                if a >= 1000.0 {
                    clamped += 1;
                    rail_dim[d] += 1;
                }
                if (v as f64 / 1000.0 * k).abs() >= 1000.0 {
                    corrected_clamped += 1;
                }
                if a < 100.0 {
                    hist[0] += 1;
                } else if a < 150.0 {
                    hist[1] += 1;
                } else if a < 300.0 {
                    hist[2] += 1;
                } else if a <= 600.0 {
                    hist[3] += 1;
                } else {
                    hist[4] += 1;
                }
            }
        }
        abs_raw.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let end = off + H2_STEPS;
        if end <= n_tokens {
            println!("\n=== window {name} (tokens [{off}, {end})) ===");
        } else {
            let tail = n_tokens - off;
            println!(
                "\n=== window {name} (tokens [{off}, {n_tokens}) + [0, {}) — {tail}+{} = {H2_STEPS} steps) ===",
                end % n_tokens,
                end % n_tokens
            );
        }
        println!("scaling : rms {rms_units:.4} → k = {k:.2} µA/unit (target {TARGET_RMS_UA} µA)");
        println!(
            "clamp@±1000 : {clamped}/{total} = {:.3}%  hist {:?}",
            clamped as f64 / total as f64 * 100.0,
            hist
        );
        let (td, tc) = rail_dim
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(d, &c)| (d, c))
            .unwrap();
        println!("hottest dim: {td} railed {tc}×");

        let pct = |p: f64| -> f64 {
            let i = (((total as f64) * p).ceil() as usize).saturating_sub(1);
            abs_raw[i.min(abs_raw.len() - 1)]
        };
        println!(
            "|I| pre-clamp µA: p50 {:.1} · p75 {:.1} · p90 {:.1} · p95 {:.1} · p99 {:.1} · max {:.1}",
            pct(0.50), pct(0.75), pct(0.90), pct(0.95), pct(0.99), abs_raw[abs_raw.len() - 1]
        );
        for &c in &[1000.0, 2000.0, 3000.0, 10_000.0, 32_767.0] {
            let over = abs_raw.partition_point(|&a| a < c) as u64;
            let frac = (total - over) as f64 / total as f64 * 100.0;
            println!(
                "ceiling ±{c:>7.0}: clamped {frac:.3}%{}",
                if frac >= 50.0 { "  ⚠ fails §7" } else { "" }
            );
        }
        println!(
            "corrected-domain (k on norm units): clamp@±1000 = {:.3}% — RMS post-clamp {:.1} µA",
            corrected_clamped as f64 / total as f64 * 100.0,
            {
                // true RMS of the corrected pre-clamp drive:
                let mut s = 0.0_f64;
                for row in &win {
                    for &v in &row[..DRIVEN_DIMS] {
                        let r = v as f64 / 1000.0 * k;
                        s += r * r;
                    }
                }
                (s / total as f64).sqrt()
            }
        );

        if off == 0 {
            // Self-verification against the banked H2 record.
            assert!((k - H2_K).abs() < 0.005, "k {k} != banked {H2_K}");
            assert_eq!(clamped, H2_CLAMPED, "clamp count != banked");
            assert_eq!(total, H2_TOTAL);
            assert_eq!(hist, H2_HIST, "histogram != banked");
            assert_eq!(
                (td, tc),
                (H2_TOP_DIM, H2_TOP_DIM_RAILS),
                "hottest dim != banked"
            );
            println!("pins    : ALL H2 banked pins reproduced — probe == H2 as-run");
        }
    }
}
