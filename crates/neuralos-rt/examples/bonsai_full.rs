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
//! Gates (strengthened by the 2026-08-15 adversarial review):
//! - per-LAYER liveness: every block's residual delta > 0 (a dead
//!   attention/FFN wiring shows delta 0 — the old final-hidden-only
//!   check could not see that);
//! - every layer's residual stays under the norm-soundness rail
//!   (6.66e7 milli — not the i32 rail, which sits 32× too high);
//! - final hidden per position: mostly nonzero, bounded;
//! - logits top-5 printed with distinct values (ids only — tokenizer
//!   is session 4); argmax must not collapse to token 0;
//! - total forward time under 5 min (the ISC-35 falsifier, now gated).

use neuralos_rt::{GgufFile, Qwen3, RESIDUAL_SOUND_MAX};

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
    let (h, health) = model
        .forward_with_health(PROMPT)
        .expect("forward");
    let fwd = t1.elapsed();
    let (emb, layers) = (model.config().emb, model.config().layers);
    let rail = model.residual_sound_max();
    println!("forward: {} tokens × {layers} layers in {fwd:?} (emb {emb}, rail {rail})", PROMPT.len());
    if fwd > std::time::Duration::from_secs(300) {
        println!("FULL: NO (forward exceeded 5 min: {fwd:?})");
        std::process::exit(1);
    }

    let mut failures = 0_usize;

    // Per-layer liveness + soundness gates.
    for (l, &delta) in health.per_layer_delta.iter().enumerate() {
        let live = delta > 0;
        println!(
            "  layer {l:>2}: residual delta {delta:>10} {}",
            if live { "OK" } else { "DEAD" }
        );
        if !live {
            failures += 1;
        }
    }
    let rail_u: u32 = u32::try_from(rail).unwrap_or(u32::MAX);
    let absmax_u = health.max_abs_residual.unsigned_abs();
    println!(
        "residual absmax {} (soundness rail {rail}, frozen 1.7B const {RESIDUAL_SOUND_MAX}) {}",
        health.max_abs_residual,
        if absmax_u < rail_u { "OK" } else { "OUT OF RANGE" }
    );
    if absmax_u >= rail_u {
        failures += 1;
    }

    // Final-hidden per-position gates (nonzero, bounded — compared in
    // u32 so an i32::MIN element cannot wrap negative and pass).
    let bound = rail_u;
    for (pos, hh) in h.iter().enumerate() {
        let nz = hh.iter().filter(|v| **v != 0).count();
        let absmax = hh.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
        let mean = hh.iter().map(|v| i64::from(*v)).sum::<i64>() / hh.len() as i64;
        let ok = nz >= emb * 9 / 10 && absmax < bound;
        println!(
            "  pos {pos}: hidden nz {nz}/{emb}, absmax {absmax}, mean {mean} {}",
            if ok { "OK" } else { "DEGENERATE" }
        );
        if !ok {
            failures += 1;
        }
    }

    // Final norm + tied logits, last position.
    let t2 = std::time::Instant::now();
    let top = model.topk_logits(&h[PROMPT.len() - 1], 5).expect("logits");
    let vocab = model.config().vocab;
    println!("logits (tied emb, {vocab} rows) in {:.1?}", t2.elapsed());
    print!("top-5 ids: ");
    for (id, v) in &top {
        print!("({id}, {v}) ");
    }
    println!();
    // Uniform-logit collapse (top1 == top5 value) is degenerate too.
    if let (Some((_, first)), Some((_, last))) = (top.first(), top.last()) {
        if first == last {
            println!("FULL: NO (uniform top-5 logits: {first})");
            std::process::exit(1);
        }
    }

    if failures > 0 {
        println!("FULL: NO ({failures} failed gates)");
        std::process::exit(1);
    }
    let argmax = top.first().map(|(id, _)| *id);
    let degenerate_argmax = argmax == Some(0) && PROMPT.last() != Some(&0);
    if degenerate_argmax {
        println!("FULL: NO (argmax collapsed to token 0)");
        std::process::exit(1);
    }
    println!("FULL: OK — {layers}-block Qwen3 forward + tied logits on real Q1_0 weights, integer compute path");
}
