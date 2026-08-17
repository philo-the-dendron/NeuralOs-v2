//! Stage 4, session 2: the first REAL forward compute on Bonsai weights.
//!
//! Slice (all integer, all real tensors from `Bonsai-1.7B-Q1_0.gguf`):
//!
//!   token id → `token_embd.weight` row (q1_0 → milli)
//!            → `blk.0.attn_norm` RMSNorm (f32 weights → milli at load)
//!            → Q/K/V projections (`q1_0_matvec` on real q1_0 tensors)
//!
//! Sanity gates per stage (nonzero output, bounded magnitudes); prints
//! stats and exits nonzero on any degenerate stage. Attention itself
//! (softmax, RoPE, GQA merge) is session 3 — this is the projection
//! proving the Q1_0 matvec eats real model weights.
//!
//! Usage: `cargo run -p neuralos-rt --example bonsai_forward -- [path]`

use neuralos_rt::{
    f32_bits_to_milli, matvec_scaled, q1_0_row_to_milli, rms_norm_milli, GgufFile, ModelConfig,
};

const TOKENS: &[u32] = &[0, 1, 42]; // first, second, arbitrary (last added at runtime)

fn stats(v: &[i32]) -> (i64, u32, usize) {
    let mean = v.iter().map(|x| i64::from(*x)).sum::<i64>() / v.len() as i64;
    let absmax = v.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
    let nonzero = v.iter().filter(|x| **x != 0).count();
    (mean, absmax, nonzero)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Bonsai-1.7B-Q1_0.gguf".into());
    let buf = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let f = GgufFile::parse(&buf).expect("container parses");
    let cfg = ModelConfig::from_gguf(&f).expect("config reads");
    let (emb, q_rows, kv_rows) = (
        cfg.emb,
        cfg.heads * cfg.head_dim,
        cfg.kv_heads * cfg.head_dim,
    );
    let emb_t = f.tensor("token_embd.weight").expect("token_embd");
    let emb_data = f.tensor_data(emb_t).expect("emb data");
    let n_vocab = *emb_t.dims.last().expect("2-D") as u32;
    let tokens: Vec<u32> = TOKENS.iter().copied().chain([n_vocab - 1]).collect();

    let norm_t = f.tensor("blk.0.attn_norm.weight").expect("attn_norm");
    let norm_data = f.tensor_data(norm_t).expect("norm data");
    let q_t = f.tensor("blk.0.attn_q.weight").expect("attn_q");
    let q_data = f.tensor_data(q_t).expect("q data");
    let k_t = f.tensor("blk.0.attn_k.weight").expect("attn_k");
    let k_data = f.tensor_data(k_t).expect("k data");
    let v_t = f.tensor("blk.0.attn_v.weight").expect("attn_v");
    let v_data = f.tensor_data(v_t).expect("v data");

    // f32 norm weights → milli (load edge; integer forever after).
    let mut w_norm = vec![0_i32; emb];
    for (i, slot) in w_norm.iter_mut().enumerate() {
        let bits = u32::from_le_bytes([
            norm_data[i * 4],
            norm_data[i * 4 + 1],
            norm_data[i * 4 + 2],
            norm_data[i * 4 + 3],
        ]);
        *slot = f32_bits_to_milli(bits);
    }
    let (w_mean, w_absmax, _) = stats(&w_norm);
    println!("attn_norm weights (milli): mean {w_mean}, absmax {w_absmax}");

    let row_bytes = emb / 128 * 18; // q1_0: (emb/128) blocks × 18 B per row
    let mut failures = 0_usize;

    for &tok in &tokens {
        // --- embedding row (q1_0 → milli).
        let row = &emb_data[usize::try_from(tok).unwrap() * row_bytes..][..row_bytes];
        let mut x = vec![0_i32; emb];
        q1_0_row_to_milli(row, &mut x).expect("embedding row decodes");
        let (x_mean, x_absmax, x_nz) = stats(&x);
        let x_var_ok = x_nz == emb; // every element is ±γ, γ > 0 in practice
        println!("tok {tok:>6}: emb milli mean {x_mean:>5}, absmax {x_absmax:>6}, nonzero {x_nz}/{emb}");

        // --- RMSNorm (integer).
        let mut h = vec![0_i32; emb];
        rms_norm_milli(&x, &w_norm, &mut h);
        let (h_mean, h_absmax, h_nz) = stats(&h);
        println!("           norm milli mean {h_mean:>5}, absmax {h_absmax:>6}, nonzero {h_nz}/{emb}");

        // --- QKV projections via the unit-chaining wrapper: milli in,
        // milli out (true units — the review flagged that this example's
        // old hand-rolled normalize printed raw partial-sum units under
        // a "milli" label).
        let mut q_out = vec![0_i32; q_rows];
        let mut k_out = vec![0_i32; kv_rows];
        let mut v_out = vec![0_i32; kv_rows];
        matvec_scaled(q_data, &h, q_rows, &mut q_out).expect("q matvec");
        matvec_scaled(k_data, &h, kv_rows, &mut k_out).expect("k matvec");
        matvec_scaled(v_data, &h, kv_rows, &mut v_out).expect("v matvec");
        let (q_mean, q_absmax, q_nz) = stats(&q_out);
        let (k_mean, k_absmax, k_nz) = stats(&k_out);
        let (v_mean, v_absmax, v_nz) = stats(&v_out);
        println!(
            "           q milli mean {q_mean:>6} absmax {q_absmax:>7} nz {q_nz}/{q_rows} | k {k_mean:>6}/{k_absmax:>7}/{k_nz}/{kv_rows} | v {v_mean:>6}/{v_absmax:>7}/{v_nz}/{kv_rows}"
        );

        // Sanity gates: stages mostly-nonzero (a single exact-cancelling
        // matvec element is arithmetic, not degeneracy) + bounded (under
        // the norm soundness rail, compared in u32 — an i32::MIN value
        // cannot wrap negative and pass).
        let bound: u32 = 66_600_000;
        let ok = x_var_ok
            && h_nz > emb / 2
            && q_nz >= q_rows * 95 / 100
            && k_nz >= kv_rows * 95 / 100
            && v_nz >= kv_rows * 95 / 100
            && q_absmax < bound
            && k_absmax < bound
            && v_absmax < bound;
        if !ok {
            println!("           STAGE DEGENERATE for token {tok}");
            failures += 1;
        }
    }

    if failures > 0 {
        println!("FORWARD: NO ({failures} degenerate tokens)");
        std::process::exit(1);
    }
    println!("FORWARD: OK — real tokens flow embedding -> RMSNorm -> QKV on real Q1_0 weights, integer-only");
}
