//! Qwen3 forward on Bonsai Q1_0 weights — Stage 4, session 3.
//!
//! All compute is integer (milli-domain activations, i64 intermediates);
//! f64 appears only at the load edge (norm weights, rope tables — the
//! doctrine pinned in ISA Decisions).
//!
//! Block structure (Qwen3, per HF `modeling_qwen3` + the file's tensors):
//!
//! ```text
//! h += attn_output(W_o · Attention(attn_norm(h)))
//! Attention: Q/K/V → per-head q/k RMSNorm (head_dim=128) → YaRN RoPE →
//!            scores·(1/√128) → causal softmax → GQA-weighted V merge
//! h += ffn_down(W_d · SiLU(W_g · ffn_norm(h)) ⊙ (W_u · ffn_norm(h)))
//! ```
//!
//! GQA: 16 Q heads, 8 KV heads — Q head `h` attends KV head `h/2`.
//! Output: tied embeddings (the file has no `output.weight`; verified in
//! `bonsai_probe`) → `logits[t] = h·emb_t`.

use crate::gguf::GgufFile;
use crate::math::{MathKit, RopeTables};
use crate::norm::rms_norm_milli;
use crate::q1_0::{matvec_scaled, q1_0_row_to_milli};

const EMB: usize = 2048;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const FFN: usize = 6144;
const LAYERS: usize = 28;
/// milli view of 1/√128 (88.3883476…‰) — attention score scale, applied
/// as ×88 + ×3883/10000 (i.e. ×88.3883).
const SCORE_SCALE_MILLI: i64 = 88;
const SCORE_SCALE_EXTRA_NUM: i64 = 3883;
const VOCAB: usize = 151_669;

/// Errors from the model layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// A required tensor is missing from the file.
    MissingTensor(String),
    /// A tensor's byte size disagrees with its dims/type.
    BadTensorSize(String),
    /// Prompt position exceeded the rope table.
    PositionOutOfRange,
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingTensor(n) => write!(f, "tensor missing: {n}"),
            Self::BadTensorSize(n) => write!(f, "tensor byte size wrong: {n}"),
            Self::PositionOutOfRange => write!(f, "position beyond rope table"),
        }
    }
}

impl std::error::Error for ModelError {}

#[derive(Debug)]
struct LayerSlices {
    attn_norm: Vec<i32>,  // milli, EMB
    q: Vec<u8>,           // q1_0 rows: 2048 out × 2048 in
    k: Vec<u8>,           // 1024 out
    v: Vec<u8>,           // 1024 out
    q_norm: Vec<i32>,     // milli, HEAD_DIM
    k_norm: Vec<i32>,     // milli, HEAD_DIM
    out_w: Vec<u8>,       // 2048 out
    ffn_norm: Vec<i32>,   // milli, EMB
    gate: Vec<u8>,        // 6144 out
    up: Vec<u8>,          // 6144 out
    down: Vec<u8>,        // 2048 out
}

/// A loaded Qwen3 Bonsai model over a parsed GGUF buffer.
///
/// Owns converted f32 norm weights (milli) and copies of nothing else —
/// q1_0 tensors borrow the caller's buffer slices via owned `Vec<u8>`
/// copies made once at load (248 MB file; the q1_0 payload is ~240 MB,
/// accepted for a research runtime; the parse itself borrows).
#[derive(Debug)]
pub struct Qwen3 {
    layers: Vec<LayerSlices>,
    out_norm: Vec<i32>,
    emb: Vec<u8>,
    kit: MathKit,
    rope: RopeTables,
    max_pos: usize,
}

fn f32_tensor_milli(f: &GgufFile<'_>, name: &str, expect_len: usize) -> Result<Vec<i32>, ModelError> {
    let t = f.tensor(name).ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    let d = f.tensor_data(t).map_err(|_| ModelError::BadTensorSize(name.into()))?;
    if d.len() != expect_len * 4 {
        return Err(ModelError::BadTensorSize(name.into()));
    }
    Ok((0..expect_len)
        .map(|i| {
            let mut b = [0_u8; 4];
            b.copy_from_slice(&d[i * 4..i * 4 + 4]);
            crate::norm::f32_bits_to_milli(u32::from_le_bytes(b))
        })
        .collect())
}

fn q1_0_tensor(f: &GgufFile<'_>, name: &str, rows: usize) -> Result<Vec<u8>, ModelError> {
    let t = f.tensor(name).ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    let d = f.tensor_data(t).map_err(|_| ModelError::BadTensorSize(name.into()))?;
    let expect = rows * (EMB / 128) * 18;
    if d.len() != expect {
        return Err(ModelError::BadTensorSize(format!("{name} ({} B, want {expect})", d.len())));
    }
    Ok(d.to_vec())
}

impl Qwen3 {
    /// Load from a parsed file (config constants pinned from the real
    /// Bonsai-1.7B: heads 16/8, head_dim 128, ffn 6144, 28 layers).
    ///
    /// # Errors
    ///
    /// [`ModelError::MissingTensor`] / [`BadTensorSize`] on any mismatch.
    pub fn load(f: &GgufFile<'_>, max_pos: usize) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            layers.push(LayerSlices {
                attn_norm: f32_tensor_milli(f, &format!("blk.{l}.attn_norm.weight"), EMB)?,
                q: q1_0_tensor(f, &format!("blk.{l}.attn_q.weight"), EMB)?,
                k: q1_0_tensor(f, &format!("blk.{l}.attn_k.weight"), KV_HEADS * HEAD_DIM)?,
                v: q1_0_tensor(f, &format!("blk.{l}.attn_v.weight"), KV_HEADS * HEAD_DIM)?,
                q_norm: f32_tensor_milli(f, &format!("blk.{l}.attn_q_norm.weight"), HEAD_DIM)?,
                k_norm: f32_tensor_milli(f, &format!("blk.{l}.attn_k_norm.weight"), HEAD_DIM)?,
                out_w: q1_0_tensor(f, &format!("blk.{l}.attn_output.weight"), EMB)?,
                ffn_norm: f32_tensor_milli(f, &format!("blk.{l}.ffn_norm.weight"), EMB)?,
                gate: q1_0_tensor(f, &format!("blk.{l}.ffn_gate.weight"), FFN)?,
                up: q1_0_tensor(f, &format!("blk.{l}.ffn_up.weight"), FFN)?,
                down: q1_0_tensor(f, &format!("blk.{l}.ffn_down.weight"), FFN)?,
            });
        }
        Ok(Self {
            layers,
            out_norm: f32_tensor_milli(f, "output_norm.weight", EMB)?,
            emb: q1_0_tensor(f, "token_embd.weight", VOCAB)?,
            kit: MathKit::new(),
            // YaRN from the real file's KV: base 1e6, factor 4, orig 8192.
            rope: RopeTables::new_yarn(HEAD_DIM, max_pos, 1e6, 4.0, 8192, 32.0, 1.0),
            max_pos,
        })
    }

    /// Token embedding lookup (milli).
    fn embed(&self, token: u32, out: &mut [i32; EMB]) -> Result<(), ModelError> {
        let t = usize::try_from(token).map_err(|_| ModelError::PositionOutOfRange)?;
        let row_bytes = EMB / 128 * 18;
        let row = &self.emb[t * row_bytes..][..row_bytes];
        q1_0_row_to_milli(row, out).map_err(|_| ModelError::BadTensorSize("emb row".into()))
    }

    /// Full forward: `tokens` (length n ≤ max_pos) → per-position hidden
    /// states (milli) and, on demand, last-position logits.
    ///
    /// # Errors
    ///
    /// [`ModelError::PositionOutOfRange`] if tokens exceed the rope table;
    /// tensor errors surface as [`ModelError::BadTensorSize`].
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<[i32; EMB]>, ModelError> {
        let n = tokens.len();
        if n == 0 || n > self.max_pos {
            return Err(ModelError::PositionOutOfRange);
        }
        // Hidden states per position (milli).
        let mut h: Vec<[i32; EMB]> = Vec::with_capacity(n);
        for &t in tokens {
            let mut row = [0_i32; EMB];
            self.embed(t, &mut row)?;
            h.push(row);
        }
        // KV caches: [layer][pos][kv_head].
        let mut k_cache = vec![vec![[[0_i32; HEAD_DIM]; KV_HEADS]; n]; LAYERS];
        let mut v_cache = vec![vec![[[0_i32; HEAD_DIM]; KV_HEADS]; n]; LAYERS];

        for (l, layer) in self.layers.iter().enumerate() {
            // ---- Attention ----
            let mut q_raw = [0_i32; EMB];
            let mut k_raw = [0_i32; KV_HEADS * HEAD_DIM];
            let mut v_raw = [0_i32; KV_HEADS * HEAD_DIM];
            let mut normed = [0_i32; EMB];
            let mut attn_out = [0_i32; EMB];
            for pos in 0..n {
                rms_norm_milli(&h[pos], &layer.attn_norm, &mut normed);
                matvec_scaled(&layer.q, &normed, EMB, &mut q_raw)
                    .map_err(|_| ModelError::BadTensorSize("q".into()))?;
                matvec_scaled(&layer.k, &normed, KV_HEADS * HEAD_DIM, &mut k_raw)
                    .map_err(|_| ModelError::BadTensorSize("k".into()))?;
                matvec_scaled(&layer.v, &normed, KV_HEADS * HEAD_DIM, &mut v_raw)
                    .map_err(|_| ModelError::BadTensorSize("v".into()))?;

                // Per-head q/k norm + rope; stash KV.
                for kv in 0..KV_HEADS {
                    let (kn, vn) = (&mut k_cache[l][pos][kv], &mut v_cache[l][pos][kv]);
                    for d in 0..HEAD_DIM {
                        kn[d] = k_raw[kv * HEAD_DIM + d];
                        vn[d] = v_raw[kv * HEAD_DIM + d];
                    }
                    let mut k_head = [0_i32; HEAD_DIM];
                    rms_norm_milli(kn, &layer.k_norm, &mut k_head);
                    self.rope.apply(&mut k_head, pos); // pos < n ≤ max_pos, safe
                    *kn = k_head;
                }
                // Attention output for this position (context, milli).
                let mut ctx = [0_i32; EMB];
                for qh in 0..HEADS {
                    let mut q_head = [0_i32; HEAD_DIM];
                    for d in 0..HEAD_DIM {
                        q_head[d] = q_raw[qh * HEAD_DIM + d];
                    }
                    let mut q_normed = [0_i32; HEAD_DIM];
                    rms_norm_milli(&q_head, &layer.q_norm, &mut q_normed);
                    self.rope.apply(&mut q_normed, pos); // safe: pos < max_pos
                    let q_head = q_normed;
                    // Scores over t ≤ pos against kv head qh/2.
                    let kv = qh / (HEADS / KV_HEADS);
                    let mut scores = [0_i32; 64];
                    let mut probs = [0_i32; 64];
                    for t in 0..=pos {
                        let mut dot: i64 = 0; // Σ q_milli·k_milli = real×1e6
                        for d in 0..HEAD_DIM {
                            dot += i64::from(q_head[d]) * i64::from(k_cache[l][t][kv][d]);
                        }
                        // × 88.388 / 1000 → milli logits (i64, bounded:
                        // |q|,|k| ≲ 1e5 milli → |dot| ≤ 128e10 × 88.4 ≈ 1.1e15).
                        let s = dot * SCORE_SCALE_MILLI
                            + dot * SCORE_SCALE_EXTRA_NUM / 10_000;
                        scores[t] = ((s + if s >= 0 { 500 } else { -500 }) / 1000)
                            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                    }
                    self.kit.softmax_q12(&scores[..=pos], &mut probs[..=pos]);
                    // Context = Σ_t p_t × v_t / Q12.
                    let mut acc = [0_i64; HEAD_DIM];
                    for t in 0..=pos {
                        let p = i64::from(probs[t]);
                        for d in 0..HEAD_DIM {
                            acc[d] += p * i64::from(v_cache[l][t][kv][d]);
                        }
                    }
                    for d in 0..HEAD_DIM {
                        let a = acc[d];
                        let q = if a >= 0 {
                            (a + 2048) / 4096
                        } else {
                            -((-a + 2048) / 4096)
                        };
                        ctx[qh * HEAD_DIM + d] =
                            q.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                    }
                }
                // Output projection + residual.
                matvec_scaled(&layer.out_w, &ctx, EMB, &mut attn_out)
                    .map_err(|_| ModelError::BadTensorSize("out".into()))?;
                for d in 0..EMB {
                    h[pos][d] = h[pos][d].saturating_add(attn_out[d]);
                }
            }

            // ---- FFN (per position, after attention residual) ----
            let mut gate = [0_i32; FFN];
            let mut up = [0_i32; FFN];
            let mut down = [0_i32; EMB];
            for (_pos, hp) in h.iter_mut().enumerate().take(n) {
                rms_norm_milli(hp, &layer.ffn_norm, &mut normed);
                matvec_scaled(&layer.gate, &normed, FFN, &mut gate)
                    .map_err(|_| ModelError::BadTensorSize("gate".into()))?;
                matvec_scaled(&layer.up, &normed, FFN, &mut up)
                    .map_err(|_| ModelError::BadTensorSize("up".into()))?;
                // SiLU(gate) ⊙ up, milli: silu_milli × up / 1000.
                for i in 0..FFN {
                    let s = i64::from(self.kit.silu_milli(gate[i]));
                    let u = i64::from(up[i]);
                    gate[i] = ((s * u + 500) / 1000)
                        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                }
                matvec_scaled(&layer.down, &gate, EMB, &mut down)
                    .map_err(|_| ModelError::BadTensorSize("down".into()))?;
                for d in 0..EMB {
                    hp[d] = hp[d].saturating_add(down[d]);
                }
            }
        }
        Ok(h)
    }

    /// Final norm + logits for one hidden state (tied embeddings):
    /// `logits[t] = h·emb_t` in milli units. Returns top-`k` (index,
    /// value) sorted by value desc — full logits are 151 669 × i32 ≈
    /// 600 KB, materialized only in release examples that want it.
    ///
    /// # Errors
    ///
    /// Same as [`Qwen3::forward`].
    pub fn topk_logits(&self, h: &[i32; EMB], k: usize) -> Result<Vec<(u32, i32)>, ModelError> {
        let mut normed = [0_i32; EMB];
        rms_norm_milli(h, &self.out_norm, &mut normed);
        // h → i16 activations once; per-row partials reuse them.
        let amax = normed.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
        if amax == 0 {
            return Ok((0..k as u32).map(|t| (t, 0)).collect());
        }
        let acts: Vec<i16> = normed
            .iter()
            .map(|&v| {
                let num = i64::from(v) * 32_767;
                let den = i64::from(amax);
                let q = if num >= 0 { (num + den / 2) / den } else { -((-num + den / 2) / den) };
                q.clamp(i16::MIN as i64, i16::MAX as i64) as i16
            })
            .collect();
        let row_bytes = EMB / 128 * 18;
        let mut raw = [0_i32; 1];
        let mut all: Vec<(u32, i32)> = Vec::with_capacity(VOCAB);
        for t in 0..VOCAB {
            let row = &self.emb[t * row_bytes..][..row_bytes];
            crate::q1_0::q1_0_matvec(row, &acts, 1, &mut raw)
                .map_err(|_| ModelError::BadTensorSize("emb".into()))?;
            all.push((t as u32, raw[0]));
        }
        all.sort_by(|a, b| b.1.cmp(&a.1));
        all.truncate(k);
        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end attention-math reference: one tiny synthetic layer path
    /// (norm → rope → score → softmax → context) vs an f64 re-derivation.
    /// Uses the real math kit + rope tables on synthetic vectors.
    #[test]
    fn attention_pipeline_matches_f64_reference() {
        let kit = MathKit::new();
        let rope = RopeTables::new_yarn(HEAD_DIM, 16, 1e6, 4.0, 8192, 32.0, 1.0);
        let pos = 5;

        // Synthetic q/k heads (milli), one k per past position.
        let q_src: [i32; HEAD_DIM] =
            core::array::from_fn(|i| (i as i32 * 61) % 300 - 150);
        let k_src: [[i32; HEAD_DIM]; 6] = core::array::from_fn(|t| {
            core::array::from_fn(|i| ((i as i32 + t as i32 * 13) * 37) % 400 - 200)
        });
        let v: [[i32; HEAD_DIM]; 6] = core::array::from_fn(|t| {
            core::array::from_fn(|i| ((i as i32 + t as i32 * 7) * 53) % 200 - 100)
        });

        // Integer path: rope q and each k, dot × scale, softmax, merge v.
        let mut q = q_src;
        rope.apply(&mut q, pos);
        let mut kr = [[0_i32; HEAD_DIM]; 6];
        for (t, k) in k_src.iter().enumerate() {
            kr[t] = *k;
            rope.apply(&mut kr[t], t);
        }
        let mut scores = [0_i32; 6];
        for t in 0..=pos {
            let mut dot: i64 = 0;
            for d in 0..HEAD_DIM {
                dot += i64::from(q[d]) * i64::from(kr[t][d]);
            }
            let s = dot * SCORE_SCALE_MILLI + dot * SCORE_SCALE_EXTRA_NUM / 10_000;
            scores[t] = ((s + 500) / 1000) as i32;
        }
        let mut probs = [0_i32; 6];
        kit.softmax_q12(&scores[..=pos], &mut probs[..=pos]);
        let mut ctx = [0_i64; HEAD_DIM];
        for t in 0..=pos {
            for d in 0..HEAD_DIM {
                ctx[d] += i64::from(probs[t]) * i64::from(v[t][d]);
            }
        }

        // f64 reference: same pipeline in floats (rope via the same
        // tables' f64 equivalents would be circular — so recompute cos/sin
        // per the YaRN formulas independently).
        let corr = |n_rot: f64| -> f64 {
            128.0 * (8192.0 / (n_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * 1e6_f64.ln())
        };
        let (low, high) = (corr(32.0).floor().max(0.0), corr(1.0).ceil().min(127.0));
        let mscale = 1.0 + 0.1 * 4.0_f64.ln();
        let angle = |p: usize, i: usize| -> (f64, f64) {
            let extrap = p as f64 * 1e6_f64.powf(-(2.0 * i as f64) / 128.0);
            let interp = 0.25 * extrap;
            let y = ((i as f64 / 2.0 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
            let r = 1.0 - y;
            let th = interp * (1.0 - r) + extrap * r;
            (th.cos() * mscale, th.sin() * mscale)
        };
        let rot = |p: usize, v: &[i32]| -> Vec<f64> {
            let mut out = vec![0.0; HEAD_DIM];
            for i in 0..64 {
                let (c, s) = angle(p, i);
                let (x1, x2) = (f64::from(v[i]), f64::from(v[i + 64]));
                out[i] = x1 * c - x2 * s;
                out[i + 64] = x2 * c + x1 * s;
            }
            out
        };
        let qf = rot(pos, &q_src);
        let kf: Vec<Vec<f64>> = (0..=pos).map(|t| rot(t, &k_src[t])).collect();
        let maxs = (0..=pos)
            .map(|t| {
                // d = Σ q·k in milli² → milli (real×1000); score_milli =
                // d × 88.3883 (NO extra /1000 — d is already milli).
                let d: f64 = (0..HEAD_DIM).map(|i| qf[i] * kf[t][i]).sum::<f64>() / 1000.0;
                d * 88.388_347
            })
            .collect::<Vec<f64>>();
        let mx = maxs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = maxs.iter().map(|s| ((*s - mx) / 1000.0_f64).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for d in 0..HEAD_DIM {
            let want: f64 = (0..=pos)
                .map(|t| exps[t] / sum * f64::from(v[t][d]))
                .sum();
            let got = (ctx[d] as f64) / 4096.0;
            assert!(
                (got - want).abs() <= 4.0 + want.abs() * 0.02,
                "ctx[{d}] {got} vs {want}"
            );
        }
    }
}
