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

use crate::gguf::{GgufFile, MetadataValue};
use crate::math::{div_round_half_away, MathKit, RopeTables};
use crate::norm::rms_norm_milli;
use crate::q1_0::{matvec_scaled, q1_0_row_to_milli};

const EMB: usize = 2048;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const FFN: usize = 6144;
const LAYERS: usize = 28;
/// milli view of 1/√128 (88.3883476…‰) — attention score scale, applied
/// as ×88 + ×3883/10000 (i.e. ×88.3883). The split keeps each i64
/// product ≤ |dot|·3883 ≈ 5e15 at the documented |dot| ≤ 1.28e12 — an
/// unsplit ×8838835 would reach 1.13e19 and overflow i64.
const SCORE_SCALE_MILLI: i64 = 88;
const SCORE_SCALE_EXTRA_NUM: i64 = 3883;
const VOCAB: usize = 151_669;
/// Residual-stream soundness rail: `rms_norm_milli`'s `Σx²` accumulation
/// (checked) is guaranteed only while `2048·max(x²) < 2^63`, i.e.
/// `|x| < 6.66e7` milli — health gates check THIS, not the i32 rail
/// (which sits 32× higher and would let norms run garbage first).
pub const RESIDUAL_SOUND_MAX: i64 = 66_600_000;

/// Errors from the model layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// A required tensor is missing from the file.
    MissingTensor(String),
    /// A tensor's byte size or dims disagree with its type/expected shape.
    BadTensorSize(String),
    /// Prompt position exceeded the rope table.
    PositionOutOfRange,
    /// A token id is outside the vocabulary (`0..151669`).
    TokenOutOfRange,
    /// A metadata KV value disagrees with the pinned model config
    /// (e.g. `qwen3.block_count` ≠ 28).
    ConfigMismatch(String),
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingTensor(n) => write!(f, "tensor missing: {n}"),
            Self::BadTensorSize(n) => write!(f, "tensor byte size wrong: {n}"),
            Self::PositionOutOfRange => write!(f, "position beyond rope table"),
            Self::TokenOutOfRange => write!(f, "token id outside vocabulary"),
            Self::ConfigMismatch(msg) => write!(f, "config mismatch: {msg}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Per-layer liveness evidence from [`Qwen3::forward_with_health`] —
/// what the bonsai_full gate actually gates on (2026-08-15 review: the
/// old final-hidden-only check could not see a dead attention layer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardHealth {
    /// Largest absolute change each layer made to the residual stream
    /// (`max |h_after − h_before|` over positions × dims). A layer that
    /// contributed nothing (broken wiring, zeroed projection) shows 0.
    pub per_layer_delta: Vec<i32>,
    /// Largest absolute residual value seen at any layer boundary —
    /// must stay under [`RESIDUAL_SOUND_MAX`] for the norms to be sound.
    pub max_abs_residual: i32,
}

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

fn f32_tensor_milli(
    f: &GgufFile<'_>,
    name: &str,
    expect_len: usize,
) -> Result<Vec<i32>, ModelError> {
    let t = f.tensor(name).ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    // Shape check first: a 1-D tensor of exactly expect_len entries.
    if t.dims.len() != 1 || t.dims[0] != expect_len as u64 {
        return Err(ModelError::BadTensorSize(format!(
            "{name} dims {:?} (want [{expect_len}])",
            t.dims
        )));
    }
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

/// Load a Q1_0 tensor of `rows` rows, each `width` weights wide —
/// validating BOTH byte size and dims (a transposed tensor has identical
/// bytes but a silently wrong layout; only dims catch it).
fn q1_0_tensor(
    f: &GgufFile<'_>,
    name: &str,
    rows: usize,
    width: usize,
) -> Result<Vec<u8>, ModelError> {
    let t = f.tensor(name).ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    if t.dims.len() != 2 || t.dims[0] != width as u64 || t.dims[1] != rows as u64 {
        return Err(ModelError::BadTensorSize(format!(
            "{name} dims {:?} (want [{width}, {rows}])",
            t.dims
        )));
    }
    let d = f.tensor_data(t).map_err(|_| ModelError::BadTensorSize(name.into()))?;
    let expect = rows * (width / crate::q1_0::Q1_0_BLOCK) * crate::q1_0::Q1_0_BLOCK_BYTES;
    if d.len() != expect {
        return Err(ModelError::BadTensorSize(format!("{name} ({} B, want {expect})", d.len())));
    }
    Ok(d.to_vec())
}

/// Cross-check a metadata KV against the pinned config — loud on
/// mismatch, silent pass-through when the key is absent (defaults
/// documented in `load`).
fn expect_kv(f: &GgufFile<'_>, key: &str, want: i64) -> Result<(), ModelError> {
    let Some(v) = f.value(key) else {
        return Ok(());
    };
    let got: Option<i64> = match v {
        MetadataValue::U8(x) => Some(i64::from(*x)),
        MetadataValue::U16(x) => Some(i64::from(*x)),
        MetadataValue::U32(x) => Some(i64::from(*x)),
        MetadataValue::U64(x) => i64::try_from(*x).ok(),
        MetadataValue::I8(x) => Some(i64::from(*x)),
        MetadataValue::I16(x) => Some(i64::from(*x)),
        MetadataValue::I32(x) => Some(i64::from(*x)),
        MetadataValue::I64(x) => Some(*x),
        MetadataValue::F32(x) => {
            // freq_base arrives as f32; compare in milli to dodge f32
            // printing artifacts (1e6 is exactly representable anyway).
            let m = (f64::from(*x) * 1000.0).round() as i64;
            let w = want * 1000;
            return if m == w {
                Ok(())
            } else {
                Err(ModelError::ConfigMismatch(format!("{key} = {x}, want {want}")))
            };
        }
        _ => None,
    };
    match got {
        Some(g) if g == want => Ok(()),
        Some(g) => Err(ModelError::ConfigMismatch(format!("{key} = {g}, want {want}"))),
        None => Err(ModelError::ConfigMismatch(format!("{key}: non-numeric value"))),
    }
}

impl Qwen3 {
    /// Load from a parsed file (config constants pinned from the real
    /// Bonsai-1.7B: heads 16/8, head_dim 128, ffn 6144, 28 layers). The
    /// file's own config KVs (`general.architecture`, `qwen3.block_count`,
    /// `…head_count`, `…head_count_kv`, `…key_length`, `…embedding_length`,
    /// `…feed_forward_length`, `qwen3.rope.freq_base`,
    /// `qwen3.rope.original_context`) are cross-checked against those
    /// pins — a different Qwen3 shape (or a transposed tensor: dims are
    /// validated too, which byte counts alone cannot catch) is a loud
    /// [`ModelError::ConfigMismatch`] / [`ModelError::BadTensorSize`],
    /// never a silent misparse. Absent keys pass (defaults documented
    /// here: rope base 1e6, orig_ctx 8192, factor 4 — pinned from the
    /// fork's runtime config for this model).
    ///
    /// # Errors
    ///
    /// [`ModelError::MissingTensor`] / [`BadTensorSize`] /
    /// [`ConfigMismatch`] on any mismatch.
    pub fn load(f: &GgufFile<'_>, max_pos: usize) -> Result<Self, ModelError> {
        if let Some(MetadataValue::String(arch)) = f.value("general.architecture") {
            if arch != "qwen3" {
                return Err(ModelError::ConfigMismatch(format!(
                    "general.architecture = {arch}, want qwen3"
                )));
            }
        }
        expect_kv(f, "qwen3.embedding_length", EMB as i64)?;
        expect_kv(f, "qwen3.block_count", LAYERS as i64)?;
        expect_kv(f, "qwen3.attention.head_count", HEADS as i64)?;
        expect_kv(f, "qwen3.attention.head_count_kv", KV_HEADS as i64)?;
        expect_kv(f, "qwen3.attention.key_length", HEAD_DIM as i64)?;
        expect_kv(f, "qwen3.feed_forward_length", FFN as i64)?;
        expect_kv(f, "qwen3.rope.freq_base", 1_000_000)?;
        expect_kv(f, "qwen3.rope.original_context", 8192)?;
        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            layers.push(LayerSlices {
                attn_norm: f32_tensor_milli(f, &format!("blk.{l}.attn_norm.weight"), EMB)?,
                q: q1_0_tensor(f, &format!("blk.{l}.attn_q.weight"), EMB, EMB)?,
                k: q1_0_tensor(f, &format!("blk.{l}.attn_k.weight"), KV_HEADS * HEAD_DIM, EMB)?,
                v: q1_0_tensor(f, &format!("blk.{l}.attn_v.weight"), KV_HEADS * HEAD_DIM, EMB)?,
                q_norm: f32_tensor_milli(f, &format!("blk.{l}.attn_q_norm.weight"), HEAD_DIM)?,
                k_norm: f32_tensor_milli(f, &format!("blk.{l}.attn_k_norm.weight"), HEAD_DIM)?,
                out_w: q1_0_tensor(f, &format!("blk.{l}.attn_output.weight"), EMB, EMB)?,
                ffn_norm: f32_tensor_milli(f, &format!("blk.{l}.ffn_norm.weight"), EMB)?,
                gate: q1_0_tensor(f, &format!("blk.{l}.ffn_gate.weight"), FFN, EMB)?,
                up: q1_0_tensor(f, &format!("blk.{l}.ffn_up.weight"), FFN, EMB)?,
                down: q1_0_tensor(f, &format!("blk.{l}.ffn_down.weight"), EMB, FFN)?,
            });
        }
        Ok(Self {
            layers,
            out_norm: f32_tensor_milli(f, "output_norm.weight", EMB)?,
            emb: q1_0_tensor(f, "token_embd.weight", VOCAB, EMB)?,
            kit: MathKit::new(),
            // YaRN from the real file's KV: base 1e6, factor 4, orig 8192.
            rope: RopeTables::new_yarn(HEAD_DIM, max_pos, 1e6, 4.0, 8192, 32.0, 1.0),
            max_pos,
        })
    }

    /// Token embedding lookup (milli). Out-of-vocabulary ids (beyond the
    /// embedding's row count — `VOCAB` on any file that passed `load`)
    /// are a loud [`ModelError::TokenOutOfRange`], never a slice panic
    /// (the 2026-08-15 review found the old path panicked behind the
    /// Result).
    fn embed(&self, token: u32, out: &mut [i32; EMB]) -> Result<(), ModelError> {
        let row_bytes = EMB / crate::q1_0::Q1_0_BLOCK * crate::q1_0::Q1_0_BLOCK_BYTES;
        let t = usize::try_from(token).map_err(|_| ModelError::TokenOutOfRange)?;
        if t >= self.emb.len() / row_bytes {
            return Err(ModelError::TokenOutOfRange);
        }
        let row = &self.emb[t * row_bytes..][..row_bytes];
        q1_0_row_to_milli(row, out).map_err(|_| ModelError::BadTensorSize("emb row".into()))
    }

    /// Full forward: `tokens` (length n ≤ max_pos) → per-position hidden
    /// states (milli) and, on demand, last-position logits.
    ///
    /// # Errors
    ///
    /// [`ModelError::PositionOutOfRange`] if tokens exceed the rope table
    /// (or the prompt is empty); [`ModelError::TokenOutOfRange`] for an
    /// out-of-vocabulary id; tensor errors surface as
    /// [`ModelError::BadTensorSize`].
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<[i32; EMB]>, ModelError> {
        self.forward_inner(tokens, None)
    }

    /// [`Qwen3::forward`] plus per-layer liveness evidence — what the
    /// bonsai_full health gate consumes (every layer's residual delta
    /// and the max absolute residual against the norm soundness rail).
    ///
    /// # Errors
    ///
    /// Same as [`Qwen3::forward`].
    pub fn forward_with_health(
        &mut self,
        tokens: &[u32],
    ) -> Result<(Vec<[i32; EMB]>, ForwardHealth), ModelError> {
        let mut health = ForwardHealth {
            per_layer_delta: Vec::new(),
            max_abs_residual: 0,
        };
        let h = self.forward_inner(tokens, Some(&mut health))?;
        Ok((h, health))
    }

    fn forward_inner(
        &mut self,
        tokens: &[u32],
        mut health: Option<&mut ForwardHealth>,
    ) -> Result<Vec<[i32; EMB]>, ModelError> {
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
        // Score/prob scratch sized by n — the 2026-08-15 review found
        // fixed [i32; 64] arrays panicking on any prompt longer than 64
        // tokens whenever max_pos was loaded larger.
        let mut scores = vec![0_i32; n];
        let mut probs = vec![0_i32; n];

        for (l, layer) in self.layers.iter().enumerate() {
            let prev_h = h.clone();
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
                    for t in 0..=pos {
                        let mut dot: i64 = 0; // Σ q_milli·k_milli = real×1e6
                        for d in 0..HEAD_DIM {
                            dot += i64::from(q_head[d]) * i64::from(k_cache[l][t][kv][d]);
                        }
                        // × 88.3883 / 1000 → milli logits (i64, bounded:
                        // |q|,|k| ≲ 1e5 milli → |dot| ≤ 128e10; × 88.4
                        // ≈ 1.13e14 « i64::MAX).
                        let s = dot * SCORE_SCALE_MILLI
                            + dot * SCORE_SCALE_EXTRA_NUM / 10_000;
                        scores[t] = div_round_half_away(s, 1000)
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
                        ctx[qh * HEAD_DIM + d] = div_round_half_away(acc[d], 4096)
                            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
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
            for hp in h.iter_mut().take(n) {
                rms_norm_milli(hp, &layer.ffn_norm, &mut normed);
                matvec_scaled(&layer.gate, &normed, FFN, &mut gate)
                    .map_err(|_| ModelError::BadTensorSize("gate".into()))?;
                matvec_scaled(&layer.up, &normed, FFN, &mut up)
                    .map_err(|_| ModelError::BadTensorSize("up".into()))?;
                // SiLU(gate) ⊙ up, milli: silu_milli × up / 1000.
                for i in 0..FFN {
                    let s = i64::from(self.kit.silu_milli(gate[i]));
                    let u = i64::from(up[i]);
                    gate[i] = div_round_half_away(s * u, 1000)
                        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                }
                matvec_scaled(&layer.down, &gate, EMB, &mut down)
                    .map_err(|_| ModelError::BadTensorSize("down".into()))?;
                for d in 0..EMB {
                    hp[d] = hp[d].saturating_add(down[d]);
                }
            }

            if let Some(health) = health.as_deref_mut() {
                let mut delta: i64 = 0;
                let mut layer_max: i64 = 0;
                for (after, before) in h.iter().zip(prev_h.iter()) {
                    for (&a, &b) in after.iter().zip(before.iter()) {
                        delta = delta.max(i64::from(a.abs_diff(b)));
                        layer_max = layer_max.max(i64::from(a.unsigned_abs()));
                    }
                }
                health.per_layer_delta.push(
                    delta.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
                );
                health.max_abs_residual =
                    health.max_abs_residual.max(layer_max.min(i64::from(i32::MAX)) as i32);
            }
        }
        Ok(h)
    }

    /// Final norm + logits for one hidden state (tied embeddings):
    /// `logits[t] ≈ h·emb_t × 1000` — true milli units (the raw
    /// normalized-activation dot is rescaled by `amax/32767`, matching
    /// [`matvec_scaled`]; the 2026-08-15 review found the old version
    /// returned ranking-valid but unscaled units while the doc claimed
    /// milli). Returns top-`k` (index, value) sorted by value desc;
    /// values saturate at the i32 rails only for hostile magnitudes.
    /// A degenerate all-zero hidden returns `k` pairs of `(id, 0)` —
    /// downstream gates catch that upstream.
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
                div_round_half_away(i64::from(v) * 32_767, i64::from(amax))
                    .clamp(i16::MIN as i64, i16::MAX as i64) as i16
            })
            .collect();
        let row_bytes = EMB / crate::q1_0::Q1_0_BLOCK * crate::q1_0::Q1_0_BLOCK_BYTES;
        let mut raw = [0_i32; 1];
        let mut all: Vec<(u32, i32)> = Vec::with_capacity(VOCAB);
        for t in 0..VOCAB {
            let row = &self.emb[t * row_bytes..][..row_bytes];
            crate::q1_0::q1_0_matvec(row, &acts, 1, &mut raw)
                .map_err(|_| ModelError::BadTensorSize("emb".into()))?;
            // Rescale into true milli units (ranking is invariant — the
            // factor is global and positive).
            let milli = div_round_half_away(i64::from(raw[0]) * i64::from(amax), 32_767)
                .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            all.push((t as u32, milli));
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
            scores[t] = div_round_half_away(s, 1000) as i32;
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
            // Ramp over the PAIR index (i0 = 2·i element index → i0/2 = i)
            // — the 2026-08-15 review fixed the same i/2 transcription
            // slip here that lived in the implementation.
            let y = ((i as f64 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
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

    // ---- Contract regression tests (2026-08-15 adversarial review) ----
    //
    // A synthetic zero-weight model: full-size tensors (so every size
    // check passes) but γ=0 everywhere — embeddings are zero, every
    // matvec hits the amax==0 fast path, so a 65-token forward costs
    // milliseconds while still exercising the full control flow (the
    // old fixed [i32; 64] score arrays panicked at pos 64 on exactly
    // this path).

    fn synthetic_model(max_pos: usize, emb_rows: usize) -> Qwen3 {
        let row_bytes = |rows: usize, width: usize| rows * (width / 128) * 18;
        let layer = LayerSlices {
            attn_norm: vec![0_i32; EMB],
            q: vec![0_u8; row_bytes(EMB, EMB)],
            k: vec![0_u8; row_bytes(KV_HEADS * HEAD_DIM, EMB)],
            v: vec![0_u8; row_bytes(KV_HEADS * HEAD_DIM, EMB)],
            q_norm: vec![0_i32; HEAD_DIM],
            k_norm: vec![0_i32; HEAD_DIM],
            out_w: vec![0_u8; row_bytes(EMB, EMB)],
            ffn_norm: vec![0_i32; EMB],
            gate: vec![0_u8; row_bytes(FFN, EMB)],
            up: vec![0_u8; row_bytes(FFN, EMB)],
            down: vec![0_u8; row_bytes(EMB, FFN)],
        };
        Qwen3 {
            layers: vec![layer],
            out_norm: vec![0_i32; EMB],
            emb: vec![0_u8; row_bytes(emb_rows, EMB)],
            kit: MathKit::new(),
            rope: RopeTables::new_yarn(HEAD_DIM, max_pos, 1e6, 4.0, 8192, 32.0, 1.0),
            max_pos,
        }
    }

    #[test]
    fn forward_past_64_tokens_is_ok_not_panic() {
        // max_pos is caller-chosen; the old fixed score/probs arrays
        // panicked at pos 64 whenever max_pos ≥ 65.
        let mut model = synthetic_model(128, 128);
        let tokens: Vec<u32> = (0..65_u32).collect();
        let h = model.forward(&tokens).expect("65-token forward");
        assert_eq!(h.len(), 65);
        // Boundary probes: exactly max_pos works, one past errors.
        let n_tokens: Vec<u32> = (0..128_u32).collect();
        assert!(model.forward(&n_tokens).is_ok());
        let over: Vec<u32> = (0..129_u32).collect();
        assert_eq!(
            model.forward(&over),
            Err(ModelError::PositionOutOfRange)
        );
        assert_eq!(model.forward(&[]), Err(ModelError::PositionOutOfRange));
    }

    #[test]
    fn forward_out_of_vocabulary_token_is_err_not_panic() {
        // Synthetic model: 8 embedding rows (load() pins VOCAB on real
        // files; the guard uses the actual row count, which is the
        // honest bound for any constructed model).
        let mut model = synthetic_model(8, 8);
        assert_eq!(model.forward(&[8]), Err(ModelError::TokenOutOfRange));
        assert_eq!(
            model.forward(&[151_669]),
            Err(ModelError::TokenOutOfRange)
        );
        assert_eq!(
            model.forward(&[u32::MAX]),
            Err(ModelError::TokenOutOfRange)
        );
        // The last row within the tensor is fine.
        assert!(model.forward(&[7]).is_ok());
    }

    #[test]
    fn forward_with_health_reports_layer_evidence() {
        let mut model = synthetic_model(8, 8);
        let (h, health) = model
            .forward_with_health(&[0, 1, 2])
            .expect("forward with health");
        assert_eq!(h.len(), 3);
        assert_eq!(health.per_layer_delta.len(), 1);
        // The zero-weight model is genuinely dead — delta 0 is the
        // honest measurement (bonsai_full gates > 0 on real weights).
        assert_eq!(health.per_layer_delta[0], 0);
        assert_eq!(health.max_abs_residual, 0);
    }

    /// Minimal one-tensor GGUF builder (layout per gguf::tests' W).
    fn one_tensor_file(name: &str, dims: &[u64], ty: u32, data: &[u8]) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3_u32.to_le_bytes());
        b.extend_from_slice(&1_u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&0_u64.to_le_bytes()); // n_kv
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&0_u64.to_le_bytes()); // offset
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn loader_rejects_transposed_dims() {
        // Same byte count, swapped dims — only a dims check catches it.
        let bytes = 2048 * (6144 / 128) * 18;
        let data = vec![0_u8; bytes];
        // Correct [width, rows] passes…
        let good = one_tensor_file("t.weight", &[6144, 2048], 41, &data);
        let f = GgufFile::parse(&good).expect("parse");
        assert!(q1_0_tensor(&f, "t.weight", 2048, 6144).is_ok());
        // …the transpose errors.
        let bad = one_tensor_file("t.weight", &[2048, 6144], 41, &data);
        let f = GgufFile::parse(&bad).expect("parse");
        assert!(matches!(
            q1_0_tensor(&f, "t.weight", 2048, 6144),
            Err(ModelError::BadTensorSize(_))
        ));
        // Truncated data (right dims, wrong byte count) also errors.
        let short = one_tensor_file("t.weight", &[6144, 2048], 41, &data[..bytes - 18]);
        let f = GgufFile::parse(&short).expect("parse");
        assert!(matches!(
            q1_0_tensor(&f, "t.weight", 2048, 6144),
            Err(ModelError::BadTensorSize(_))
        ));
    }

    #[test]
    fn loader_config_kv_mismatches_are_loud() {
        let data = vec![0_u8; 2048 * 16 * 18];
        // block_count = 27 (not 28) → ConfigMismatch at load time, before
        // any tensor is touched.
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3_u32.to_le_bytes());
        b.extend_from_slice(&0_u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&1_u64.to_le_bytes()); // n_kv
        let kv = "qwen3.block_count";
        b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
        b.extend_from_slice(kv.as_bytes());
        b.extend_from_slice(&4_u32.to_le_bytes()); // U32
        b.extend_from_slice(&27_u32.to_le_bytes());
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        b.extend_from_slice(&data);
        let f = GgufFile::parse(&b).expect("parse");
        assert_eq!(
            expect_kv(&f, "qwen3.block_count", LAYERS as i64),
            Err(ModelError::ConfigMismatch(
                "qwen3.block_count = 27, want 28".into()
            ))
        );
        // Architecture mismatch caught at load.
        let _ = data;
    }

    #[test]
    fn expect_kv_passes_on_absent_and_matching() {
        let b = one_tensor_file("t", &[8, 1], 0, &[0_u8; 32]);
        let f = GgufFile::parse(&b).expect("parse");
        assert_eq!(expect_kv(&f, "qwen3.block_count", 28), Ok(()));
    }
}
