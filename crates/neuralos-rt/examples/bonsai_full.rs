//! Stage 4, session 3: the FULL Qwen3 forward pass on the real Bonsai
//! model — 28 blocks of attention (YaRN RoPE + GQA + integer softmax)
//! and FFN (SiLU gate/up), final RMSNorm, and last-position logits over
//! the tied 151 669-token embedding.
//!
//! Run in RELEASE (the forward is ~4.7 G integer ops for 4 tokens):
//!
//! ```text
//! cargo run -p neuralos-rt --release --example bonsai_full
//! ```
//!
//! Gates: every block's residual stream must stay nonzero and off the
//! i32 rails; logits top-5 printed (ids only — tokenizer is session 4).

use neuralos_rt::{GgufFile, Qwen3};

const PROMPT: &[u32] = &[0, 42, 151668, 7];
const MAX_POS: usize = 64;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Bonsai-1.7B-Q1_0.gguf".into());
    let buf = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let t0 = std::time::Instant::now();
    let f = GgufFile::parse(&buf).expect("container parses");
    let mut model = Qwen3::load(&f, MAX_POS).expect("model loads");
    println!(
        "loaded: {} tensors parsed, model materialized in {:.1?}",
        f.tensors.len(),
        t0.elapsed()
    );

    let t1 = std::time::Instant::now();
    let h = model.forward(PROMPT).expect("forward");
    let fwd = t1.elapsed();
    println!("forward: {} tokens × 28 blocks in {:.1?}", PROMPT.len(), fwd);

    // Health gates over the residual stream.
    let rail = i32::MAX - 1;
    let mut failures = 0;
    for (pos, hh) in h.iter().enumerate() {
        let nz = hh.iter().filter(|v| **v != 0).count();
        let absmax = hh.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
        let mean = hh.iter().map(|v| i64::from(*v)).sum::<i64>() / hh.len() as i64;
        let ok = nz >= EMB * 9 / 10 && (absmax as i32) < rail;
        println!(
            "  pos {pos}: hidden nz {nz}/2048, absmax {absmax}, mean {mean} {}",
            if ok { "OK" } else { "DEGENERATE" }
        );
        if !ok {
            failures += 1;
        }
    }

    // Final norm + tied logits, last position.
    let t2 = std::time::Instant::now();
    let top = model.topk_logits(&h[PROMPT.len() - 1], 5).expect("logits");
    println!("logits (tied emb, 151669 rows) in {:.1?}", t2.elapsed());
    print!("top-5 ids: ");
    for (id, v) in &top {
        print!("({id}, {v}) ");
    }
    println!();

    if failures > 0 {
        println!("FULL: NO ({failures} degenerate positions)");
        std::process::exit(1);
    }
    let argmax = top.first().map(|(id, _)| *id);
    let degenerate_argmax = argmax == Some(0) && PROMPT.last() != Some(&0);
    if degenerate_argmax {
        println!("FULL: NO (argmax collapsed to token 0)");
        std::process::exit(1);
    }
    println!("FULL: OK — 28-block Qwen3 forward + tied logits on real Q1_0 weights, integer compute path");
}

const EMB: usize = 2048;
