//! Qwen3 forward on Bonsai Q1_0 weights — Stage 4, sessions 3–4 + the
//! 4B generalization (config-driven, 2026-08-16).
//!
//! All compute is integer (milli-domain activations, i64 intermediates);
//! f64 appears only at the load edge (norm weights, rope tables, the
//! score-scale split — the doctrine pinned in ISA Decisions).
//!
//! Geometry comes from the FILE (`ModelConfig::from_gguf` reads the
//! fork's own `qwen3.*` KVs) — two real shapes drive it today:
//! Bonsai-1.7B (28 blocks, emb 2048, 16/8 heads, FFN 6144, rope base
//! 1e6) and Bonsai-4B (36 blocks, emb 2560, 32/8 heads, FFN 9728, rope
//! base 5e6). head_dim is 128 on both and the attention score scale
//! derives from it (`ModelConfig::score_scale`) — nothing in this file
//! assumes the 1.7B shape anymore.
//!
//! Block structure (Qwen3, per HF `modeling_qwen3` + the file's tensors):
//!
//! ```text
//! h += attn_output(W_o · Attention(attn_norm(h)))
//! Attention: Q/K/V → per-head q/k RMSNorm (head_dim) → YaRN RoPE →
//!            scores·(1/√head_dim) → causal softmax → GQA-weighted V merge
//! h += ffn_down(W_d · SiLU(W_g · ffn_norm(h)) ⊙ (W_u · ffn_norm(h)))
//! ```
//!
//! GQA: Q head `h` attends KV head `h/(heads/kv_heads)` (2 on 1.7B,
//! 4 on 4B). The attention context is `heads·head_dim` wide — equal to
//! `emb` on 1.7B (16×128) but 4096 ≠ 2560 on 4B, so `attn_q`/
//! `attn_output` are genuinely non-square there. Output: tied
//! embeddings (no `output.weight` in either file; verified in
//! `bonsai_probe`) → `logits[t] = h·emb_t`.
//!
//! Two execution paths share one arithmetic contract:
//! - [`Qwen3::forward`] — the reviewed session-3 full forward
//!   (recomputed per call; `forward_with_health` adds layer evidence).
//! - [`Qwen3::new_session`] / [`Session`] — append-only incremental
//!   decode: [`Qwen3::prefill`] the prompt once, then
//!   [`Qwen3::step`] one token at a time against the persistent KV
//!   cache. The per-position arithmetic is the SAME code shape in the
//!   SAME order (attention is position-local given the caches; the FFN
//!   is position-local), so integer exactness makes the two paths
//!   bit-identical — pinned by exact-equality tests (tolerance 0) on a
//!   synthetic nonzero model (CI) and on the real file.

use crate::gguf::{GgufFile, MetadataValue};
use crate::math::{div_round_half_away, MathKit, RopeTables};
use crate::norm::rms_norm_milli;
use crate::q1_0::{matvec_scaled, q1_0_row_to_milli};

/// Residual-stream soundness rail for the 1.7B shape (emb 2048):
/// `rms_norm_milli`'s `Σx²` accumulation (checked) is guaranteed only
/// while `2048·max(x²) < 2^63`, i.e. `|x| < 6.66e7` milli — health
/// gates check THIS, not the i32 rail (which sits 32× higher and would
/// let norms run garbage first). Kept as the frozen gate example's
/// rail; the exact per-model derivation is [`Qwen3::residual_sound_max`]
/// (emb 2048 → 67_108_864 — this const is the conservative rounding).
pub const RESIDUAL_SOUND_MAX: i64 = 66_600_000;

/// Per-file model geometry, read from the GGUF's own config KVs
/// ([`ModelConfig::from_gguf`]) — the 4B session replaced the pinned
/// 1.7B constants. Recorded shapes (probe evidence, 2026-08-16):
///
/// | KV | Bonsai-1.7B | Bonsai-4B |
/// |---|---|---|
/// | `qwen3.block_count` | 28 | 36 |
/// | `qwen3.embedding_length` | 2048 | 2560 |
/// | `qwen3.attention.head_count` | 16 | 32 |
/// | `qwen3.attention.head_count_kv` | 8 | 8 |
/// | `qwen3.attention.key_length` (= `value_length`) | 128 | 128 |
/// | `qwen3.feed_forward_length` | 6144 | 9728 |
/// | `qwen3.rope.freq_base` | 1e6 | **5e6** |
/// | `qwen3.rope.scaling.{type,factor,orig_ctx}` | yarn, 4.0, 8192 | same |
///
/// `vocab` is NOT a KV in these files — it is the `token_embd.weight`
/// row count (151 669 on both tiers), filled by [`Qwen3::load`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Hidden width (`qwen3.embedding_length`).
    pub emb: usize,
    /// Query heads (`qwen3.attention.head_count`).
    pub heads: usize,
    /// KV heads (`qwen3.attention.head_count_kv`) — GQA.
    pub kv_heads: usize,
    /// Per-head dimension (`qwen3.attention.key_length`; cross-checked
    /// against `value_length` when present). 128 on both real files.
    pub head_dim: usize,
    /// FFN inner width (`qwen3.feed_forward_length`).
    pub ffn: usize,
    /// Block count (`qwen3.block_count`).
    pub layers: usize,
    /// Embedding rows = vocabulary size (from the tensor, not a KV).
    pub vocab: usize,
    /// RoPE base (`qwen3.rope.freq_base`, REQUIRED — it differs per
    /// tier: 1e6 on 1.7B, 5e6 on 4B; a default here would run the
    /// wrong tables silently).
    pub rope_base: f64,
    /// YaRN factor (`qwen3.rope.scaling.factor`; default 4.0 when
    /// absent — both real files carry it).
    pub yarn_factor: f64,
    /// YaRN original context (`qwen3.rope.scaling.original_context_length`;
    /// default 8192 when absent — both real files carry it).
    pub orig_ctx: usize,
}

impl ModelConfig {
    /// Read the config from a parsed GGUF. Required KVs are loud when
    /// absent (`ConfigMismatch`) — a different tier must never silently
    /// default. Optional cross-checks: `attention.value_length` ==
    /// `key_length`, `rope.scaling.type` == `yarn`, and the RMS eps ==
    /// 1e-6 (the integer eps floor in `rms_norm_milli` is baked for
    /// it). `vocab` is left 0 for [`Qwen3::load`] to fill.
    ///
    /// # Errors
    ///
    /// [`ModelError::ConfigMismatch`] on any absent key, non-numeric
    /// value, or broken invariant (emb/ffn % 128, head_dim % 2,
    /// heads % kv_heads).
    pub fn from_gguf(f: &GgufFile<'_>) -> Result<Self, ModelError> {
        let emb = kv_usize(f, "qwen3.embedding_length")?;
        let layers = kv_usize(f, "qwen3.block_count")?;
        let heads = kv_usize(f, "qwen3.attention.head_count")?;
        let kv_heads = kv_usize(f, "qwen3.attention.head_count_kv")?;
        let head_dim = kv_usize(f, "qwen3.attention.key_length")?;
        let ffn = kv_usize(f, "qwen3.feed_forward_length")?;
        let rope_base = kv_f64(f, "qwen3.rope.freq_base")?;
        let inv = |what: &str| ModelError::ConfigMismatch(what.into());
        if emb == 0 || emb % crate::q1_0::Q1_0_BLOCK != 0 {
            return Err(inv("qwen3.embedding_length must be a positive multiple of 128"));
        }
        if ffn == 0 || ffn % crate::q1_0::Q1_0_BLOCK != 0 {
            return Err(inv("qwen3.feed_forward_length must be a positive multiple of 128"));
        }
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(inv("qwen3.attention.key_length must be positive and even (rope pairs)"));
        }
        if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 {
            return Err(inv("qwen3.attention.head_count must be a positive multiple of head_count_kv"));
        }
        // Optional cross-checks — loud on disagreement, absent = default.
        expect_kv(f, "qwen3.attention.value_length", head_dim as i64)?;
        if let Some(MetadataValue::String(ty)) = f.value("qwen3.rope.scaling.type") {
            if ty != "yarn" {
                return Err(inv(&format!(
                    "qwen3.rope.scaling.type = {ty}, want yarn (only pinned scheme)"
                )));
            }
        }
        // The milli integer eps floor in rms_norm_milli is derived for
        // 1e-6 (rounds to 0 at milli resolution) — a different eps
        // would need a different floor.
        if let Some(v) = f.value("qwen3.attention.layer_norm_rms_epsilon") {
            let eps = match v {
                MetadataValue::F32(x) => f64::from(*x),
                MetadataValue::F64(x) => *x,
                _ => f64::NAN,
            };
            if (eps - 1e-6).abs() > 1e-12 {
                return Err(inv(
                    "qwen3.attention.layer_norm_rms_epsilon != 1e-6 (rms_norm_milli's eps floor is baked for 1e-6)",
                ));
            }
        }
        let yarn_factor = match f.value("qwen3.rope.scaling.factor") {
            Some(v) => numeric_as_f64(v).ok_or_else(|| {
                inv("qwen3.rope.scaling.factor: non-numeric value")
            })?,
            None => 4.0,
        };
        let orig_ctx = match f.value("qwen3.rope.scaling.original_context_length") {
            Some(_) => kv_usize(f, "qwen3.rope.scaling.original_context_length")?,
            None => 8192,
        };
        Ok(Self {
            emb,
            heads,
            kv_heads,
            head_dim,
            ffn,
            layers,
            vocab: 0,
            rope_base,
            yarn_factor,
            orig_ctx,
        })
    }

    /// Attention score scale for a head dimension, as the (main,
    /// extra) split: milli scores = `dot × (main + extra/10⁴) / 10⁶`
    /// with `main + extra/10⁴ = 1000/√head_dim`. The split keeps each
    /// i64 product bounded: `|dot| ≤ head_dim·max|q·k|`; with milli
    /// activations ≲ 1e5, `|dot| ≲ head_dim·1e10` and the largest
    /// product is `|dot|·extra ≈ head_dim·1e13` « i64::MAX.
    /// head_dim 128 → **(88, 3883)** — bit-identical to the pinned
    /// 1.7B constants this replaces (pinned by test).
    #[must_use]
    pub fn score_scale(head_dim: usize) -> (i64, i64) {
        let inv = 1000.0 / (head_dim as f64).sqrt();
        let main = inv.floor() as i64;
        let extra = ((inv - main as f64) * 10_000.0).round() as i64;
        (main, extra)
    }
}

/// Errors from the model layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// A required tensor is missing from the file.
    MissingTensor(String),
    /// A tensor's byte size or dims disagree with its type/expected shape.
    BadTensorSize(String),
    /// Prompt position exceeded the rope table.
    PositionOutOfRange,
    /// A token id is outside the vocabulary (`0..vocab`).
    TokenOutOfRange,
    /// A metadata KV value disagrees with the file's own config, or a
    /// required config KV is absent/non-numeric (e.g. `qwen3.block_count`).
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
    /// must stay under the model's soundness rail
    /// ([`Qwen3::residual_sound_max`]) for the norms to be sound.
    pub max_abs_residual: i32,
}

#[derive(Debug)]
struct LayerSlices {
    attn_norm: Vec<i32>,  // milli, emb
    q: Vec<u8>,           // q1_0 rows: heads·head_dim out × emb in
    k: Vec<u8>,           // kv_heads·head_dim out
    v: Vec<u8>,           // kv_heads·head_dim out
    q_norm: Vec<i32>,     // milli, head_dim
    k_norm: Vec<i32>,     // milli, head_dim
    out_w: Vec<u8>,       // emb out × heads·head_dim in
    ffn_norm: Vec<i32>,   // milli, emb
    gate: Vec<u8>,        // ffn out × emb in
    up: Vec<u8>,          // ffn out × emb in
    down: Vec<u8>,        // emb out × ffn in
}

/// A persistent, append-only decode state: the per-layer KV caches plus
/// the residual-soundness witness (fog (g) from the 2026-08-15 review —
/// residual growth with prompt length is now carried evidence, not a
/// print in one example).
///
/// Positions enter via [`Qwen3::prefill`] / [`Qwen3::step`] and never
/// leave — each new token attends the whole history with no recompute
/// (the session-3 full forward is O(N²) over a prompt; this is O(N)).
/// Caches grow per position (2·layers·kv_heads·head_dim·4 B — 229 KB
/// on 1.7B, 288 KB on 4B), not by `max_pos` up front.
#[derive(Debug)]
pub struct Session {
    /// `[layer][pos]` — post-norm post-RoPE keys, milli, flat
    /// `kv_heads·head_dim`.
    k_cache: Vec<Vec<Vec<i32>>>,
    /// `[layer][pos]` — values, milli, flat `kv_heads·head_dim`.
    v_cache: Vec<Vec<Vec<i32>>>,
    n: usize,
    /// Largest absolute residual seen at any layer boundary (the
    /// soundness-rail witness).
    max_abs_residual: i32,
}

impl Session {
    /// Number of positions cached (prompt + generated so far).
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether no tokens have been processed yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Largest absolute residual at any layer boundary so far — compare
    /// against [`Qwen3::residual_sound_max`] (the norm-soundness rail;
    /// fog (g) evidence).
    #[must_use]
    pub fn max_abs_residual(&self) -> i32 {
        self.max_abs_residual
    }
}

/// A loaded Qwen3 Bonsai model over a parsed GGUF buffer.
///
/// Owns converted f32 norm weights (milli) and copies of nothing else —
/// q1_0 tensors borrow the caller's buffer slices via owned `Vec<u8>`
/// copies made once at load (248 MB / 572 MB files; the q1_0 payload is
/// ~240 MB / ~546 MB, accepted for a research runtime; the parse itself
/// borrows).
#[derive(Debug)]
pub struct Qwen3 {
    cfg: ModelConfig,
    layers: Vec<LayerSlices>,
    out_norm: Vec<i32>,
    emb: Vec<u8>,
    kit: MathKit,
    rope: RopeTables,
    /// (main, extra) score-scale split for `cfg.head_dim`.
    score: (i64, i64),
    max_pos: usize,
    /// Per-block hidden snapshots, filled only when `capture` is armed by
    /// [`Qwen3::forward_block_states`] (diagnostic path).
    block_states: Vec<Vec<Vec<i32>>>,
    capture: bool,
}

/// Required integer KV — loud when absent or non-numeric.
fn kv_usize(f: &GgufFile<'_>, key: &str) -> Result<usize, ModelError> {
    let v = f.value(key).ok_or_else(|| {
        ModelError::ConfigMismatch(format!("{key} absent (required config KV)"))
    })?;
    let n = match numeric_as_u64(v) {
        Some(n) => n,
        None => {
            return Err(ModelError::ConfigMismatch(format!(
                "{key}: non-integer value"
            )))
        }
    };
    usize::try_from(n)
        .map_err(|_| ModelError::ConfigMismatch(format!("{key} = {n}: out of range")))
}

/// Required float KV — loud when absent or non-numeric.
fn kv_f64(f: &GgufFile<'_>, key: &str) -> Result<f64, ModelError> {
    let v = f.value(key).ok_or_else(|| {
        ModelError::ConfigMismatch(format!("{key} absent (required config KV)"))
    })?;
    numeric_as_f64(v)
        .ok_or_else(|| ModelError::ConfigMismatch(format!("{key}: non-numeric value")))
}

fn numeric_as_u64(v: &MetadataValue) -> Option<u64> {
    match v {
        MetadataValue::U8(x) => Some(u64::from(*x)),
        MetadataValue::U16(x) => Some(u64::from(*x)),
        MetadataValue::U32(x) => Some(u64::from(*x)),
        MetadataValue::U64(x) => Some(*x),
        MetadataValue::I8(x) => u64::try_from(*x).ok(),
        MetadataValue::I16(x) => u64::try_from(*x).ok(),
        MetadataValue::I32(x) => u64::try_from(*x).ok(),
        MetadataValue::I64(x) => u64::try_from(*x).ok(),
        _ => None,
    }
}

fn numeric_as_f64(v: &MetadataValue) -> Option<f64> {
    match v {
        MetadataValue::F32(x) => Some(f64::from(*x)),
        MetadataValue::F64(x) => Some(*x),
        other => numeric_as_u64(other).map(|n| n as f64),
    }
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
    if d.len() < expect_len * 4 {
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
/// validating BOTH dims and byte size (a transposed tensor has identical
/// bytes but a silently wrong layout; only dims catch it). The data
/// slice may carry alignment padding beyond the formula size (the 4B's
/// `token_embd.weight` is 24 B short of a 32-byte boundary): the slice
/// must be exactly `rows·(width/128)·18` or that value rounded up to
/// the file's alignment; only the formula bytes are copied.
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
    let align = f.alignment.max(1);
    let padded = expect
        .checked_add(align as usize - 1)
        .map(|x| x / align as usize * align as usize)
        .unwrap_or(usize::MAX);
    if d.len() != expect && d.len() != padded {
        return Err(ModelError::BadTensorSize(format!(
            "{name} ({} B, want {expect} (or {padded} padded))",
            d.len()
        )));
    }
    Ok(d[..expect].to_vec())
}

/// Cross-check an optional metadata KV against a wanted integer —
/// loud on mismatch, silent pass-through when the key is absent
/// (defaults documented at the call sites).
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
            // Floats compare in milli to dodge f32 printing artifacts
            // (1e6 is exactly representable anyway).
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
    /// The model's exact norm-soundness rail: `rms_norm_milli`'s `Σx²`
    /// accumulation is guaranteed while `emb·max(x²) < 2^63`, i.e.
    /// `|x| < √(2^63/emb)` — 67_108_864 at emb 2048 (where the frozen
    /// [`RESIDUAL_SOUND_MAX`] const is the conservative 66.6 M
    /// rounding), 60_024_845 at emb 2560. Observed residuals on real
    /// prompts sit at 18–30 M — far under either form.
    #[must_use]
    pub fn residual_sound_max(&self) -> i64 {
        (9_223_372_036_854_775_808.0_f64 / self.cfg.emb as f64).sqrt() as i64
    }

    /// The loaded geometry (per-file, from the GGUF's own KVs).
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Load from a parsed file. The geometry comes from the file's own
    /// config KVs via [`ModelConfig::from_gguf`] (required keys are
    /// loud when absent — a different tier never silently defaults);
    /// `vocab` is the `token_embd.weight` row count; every tensor's
    /// dims are validated against the config (a transposed tensor has
    /// identical bytes but a silently wrong layout) and its byte size
    /// against `rows × (width/128) × 18` (alignment padding tolerated,
    /// [`q1_0_tensor`]). Rope: YaRN, base/factor/orig_ctx from the
    /// config; beta_fast 32 / beta_slow 1 stay pinned from the fork's
    /// runtime defaults (both real files' scaling KVs agree).
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
        let mut cfg = ModelConfig::from_gguf(f)?;
        let score = ModelConfig::score_scale(cfg.head_dim);
        let q_rows = cfg.heads * cfg.head_dim;
        let kv_rows = cfg.kv_heads * cfg.head_dim;
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            layers.push(LayerSlices {
                attn_norm: f32_tensor_milli(f, &format!("blk.{l}.attn_norm.weight"), cfg.emb)?,
                q: q1_0_tensor(f, &format!("blk.{l}.attn_q.weight"), q_rows, cfg.emb)?,
                k: q1_0_tensor(f, &format!("blk.{l}.attn_k.weight"), kv_rows, cfg.emb)?,
                v: q1_0_tensor(f, &format!("blk.{l}.attn_v.weight"), kv_rows, cfg.emb)?,
                q_norm: f32_tensor_milli(f, &format!("blk.{l}.attn_q_norm.weight"), cfg.head_dim)?,
                k_norm: f32_tensor_milli(f, &format!("blk.{l}.attn_k_norm.weight"), cfg.head_dim)?,
                out_w: q1_0_tensor(f, &format!("blk.{l}.attn_output.weight"), cfg.emb, q_rows)?,
                ffn_norm: f32_tensor_milli(f, &format!("blk.{l}.ffn_norm.weight"), cfg.emb)?,
                gate: q1_0_tensor(f, &format!("blk.{l}.ffn_gate.weight"), cfg.ffn, cfg.emb)?,
                up: q1_0_tensor(f, &format!("blk.{l}.ffn_up.weight"), cfg.ffn, cfg.emb)?,
                down: q1_0_tensor(f, &format!("blk.{l}.ffn_down.weight"), cfg.emb, cfg.ffn)?,
            });
        }
        // vocab = embedding row count (no vocab_size KV in these files).
        let emb_t = f
            .tensor("token_embd.weight")
            .ok_or_else(|| ModelError::MissingTensor("token_embd.weight".into()))?;
        if emb_t.dims.len() != 2 || emb_t.dims[0] != cfg.emb as u64 {
            return Err(ModelError::BadTensorSize(format!(
                "token_embd.weight dims {:?} (want [{}, rows])",
                emb_t.dims, cfg.emb
            )));
        }
        cfg.vocab = usize::try_from(emb_t.dims[1])
            .map_err(|_| ModelError::BadTensorSize("token_embd rows".into()))?;
        let emb = q1_0_tensor(f, "token_embd.weight", cfg.vocab, cfg.emb)?;
        let rope = RopeTables::new_yarn(
            cfg.head_dim,
            max_pos,
            cfg.rope_base,
            cfg.yarn_factor,
            cfg.orig_ctx,
            32.0,
            1.0,
        );
        Ok(Self {
            layers,
            out_norm: f32_tensor_milli(f, "output_norm.weight", cfg.emb)?,
            emb,
            kit: MathKit::new(),
            // YaRN from the file's own KVs (base 1e6/5e6, factor 4,
            // orig 8192 — probe-printed provenance).
            rope,
            score,
            cfg,
            max_pos,
            block_states: Vec::new(),
            capture: false,
        })
    }

    /// Token embedding lookup (milli). Out-of-vocabulary ids (beyond the
    /// embedding's row count) are a loud [`ModelError::TokenOutOfRange`],
    /// never a slice panic (the 2026-08-15 review found the old path
    /// panicked behind the Result).
    fn embed(&self, token: u32, out: &mut [i32]) -> Result<(), ModelError> {
        let row_bytes = self.emb_row_bytes();
        let t = usize::try_from(token).map_err(|_| ModelError::TokenOutOfRange)?;
        if t >= self.emb.len() / row_bytes {
            return Err(ModelError::TokenOutOfRange);
        }
        let row = &self.emb[t * row_bytes..][..row_bytes];
        q1_0_row_to_milli(row, out).map_err(|_| ModelError::BadTensorSize("emb row".into()))
    }

    fn emb_row_bytes(&self) -> usize {
        self.cfg.emb / crate::q1_0::Q1_0_BLOCK * crate::q1_0::Q1_0_BLOCK_BYTES
    }

    /// Full forward: `tokens` (length n ≤ max_pos) → per-position hidden
    /// states (milli, each `emb` wide).
    ///
    /// # Errors
    ///
    /// [`ModelError::PositionOutOfRange`] if tokens exceed the rope table
    /// (or the prompt is empty); [`ModelError::TokenOutOfRange`] for an
    /// out-of-vocabulary id; tensor errors surface as
    /// [`ModelError::BadTensorSize`].
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<Vec<i32>>, ModelError> {
        self.forward_inner(tokens, None)
    }

    /// [`Qwen3::forward`] plus per-layer liveness evidence — what the
    /// bonsai_full health gate consumes (every layer's residual delta
    /// and the max absolute residual against the soundness rail).
    ///
    /// # Errors
    ///
    /// Same as [`Qwen3::forward`].
    pub fn forward_with_health(
        &mut self,
        tokens: &[u32],
    ) -> Result<(Vec<Vec<i32>>, ForwardHealth), ModelError> {
        let mut health = ForwardHealth {
            per_layer_delta: Vec::new(),
            max_abs_residual: 0,
        };
        let h = self.forward_inner(tokens, Some(&mut health))?;
        Ok((h, health))
    }

    /// Diagnostic variant of [`Qwen3::forward`] that also returns the
    /// full hidden-state matrix at every substep boundary: entry 0 is
    /// the embedding layer, then per block `l` two entries — after
    /// block `l`'s attention residual and after its FFN residual (2·L+1
    /// entries total). All positions, milli units — the per-block drift
    /// microscope used by the reference-comparison harness (session
    /// C-core) and available for future profiling.
    pub fn forward_block_states(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<Vec<Vec<i32>>>, ModelError> {
        self.block_states = Vec::new();
        self.capture = true;
        let result = self.forward_inner(tokens, None);
        self.capture = false;
        let states = std::mem::take(&mut self.block_states);
        result.map(|_| states)
    }

    /// Start an incremental decode session (empty caches).
    #[must_use]
    pub fn new_session(&self) -> Session {
        Session {
            k_cache: vec![Vec::new(); self.layers.len()],
            v_cache: vec![Vec::new(); self.layers.len()],
            n: 0,
            max_abs_residual: 0,
        }
    }

    /// Prefill a session with the prompt, returning the hidden state
    /// after each token. Thin loop over [`Qwen3::step`] — prefill is
    /// just the first N steps (append-only; no recompute).
    ///
    /// # Errors
    ///
    /// Same as [`Qwen3::step`].
    pub fn prefill(
        &self,
        ses: &mut Session,
        tokens: &[u32],
    ) -> Result<Vec<Vec<i32>>, ModelError> {
        let mut out = Vec::with_capacity(tokens.len());
        for &t in tokens {
            out.push(self.step(ses, t)?);
        }
        Ok(out)
    }

    /// Run ONE token through all layers against the session's caches,
    /// appending its K/V and returning its final hidden state. The
    /// arithmetic mirrors `forward_inner`'s per-position body verbatim
    /// (same order, same rounding) — integer exactness makes the two
    /// paths bit-identical (pinned by tests at tolerance 0).
    ///
    /// # Errors
    ///
    /// [`ModelError::PositionOutOfRange`] when the session is already at
    /// `max_pos` positions (rope table exhausted); [`ModelError::TokenOutOfRange`]
    /// for an out-of-vocabulary id; tensor errors as
    /// [`ModelError::BadTensorSize`].
    pub fn step(&self, ses: &mut Session, token: u32) -> Result<Vec<i32>, ModelError> {
        let pos = ses.n;
        if pos >= self.max_pos {
            return Err(ModelError::PositionOutOfRange);
        }
        let (emb, heads, kv_heads, head_dim, ffn) =
            (self.cfg.emb, self.cfg.heads, self.cfg.kv_heads, self.cfg.head_dim, self.cfg.ffn);
        let kv_width = kv_heads * head_dim;
        let mut h = vec![0_i32; emb];
        self.embed(token, &mut h)?;

        let mut q_raw = vec![0_i32; heads * head_dim];
        let mut k_raw = vec![0_i32; kv_width];
        let mut v_raw = vec![0_i32; kv_width];
        let mut normed = vec![0_i32; emb];
        let mut attn_out = vec![0_i32; emb];
        let mut ctx = vec![0_i32; heads * head_dim];
        let mut q_head = vec![0_i32; head_dim];
        let mut k_head = vec![0_i32; head_dim];
        let mut q_normed = vec![0_i32; head_dim];
        let mut k_normed = vec![0_i32; head_dim];
        let mut acc = vec![0_i64; head_dim];
        let mut scores = vec![0_i32; pos + 1];
        let mut probs = vec![0_i32; pos + 1];

        for (l, layer) in self.layers.iter().enumerate() {
            // ---- Attention (this position against the cached history) ----
            rms_norm_milli(&h, &layer.attn_norm, &mut normed);
            matvec_scaled(&layer.q, &normed, heads * head_dim, &mut q_raw)
                .map_err(|_| ModelError::BadTensorSize("q".into()))?;
            matvec_scaled(&layer.k, &normed, kv_width, &mut k_raw)
                .map_err(|_| ModelError::BadTensorSize("k".into()))?;
            matvec_scaled(&layer.v, &normed, kv_width, &mut v_raw)
                .map_err(|_| ModelError::BadTensorSize("v".into()))?;

            ses.k_cache[l].push(vec![0_i32; kv_width]);
            ses.v_cache[l].push(vec![0_i32; kv_width]);
            for kv in 0..kv_heads {
                for d in 0..head_dim {
                    k_head[d] = k_raw[kv * head_dim + d];
                }
                rms_norm_milli(&k_head, &layer.k_norm, &mut k_normed);
                self.rope.apply(&mut k_normed, pos);
                let (kc, vc) = (&mut ses.k_cache[l][pos], &mut ses.v_cache[l][pos]);
                for d in 0..head_dim {
                    kc[kv * head_dim + d] = k_normed[d];
                    vc[kv * head_dim + d] = v_raw[kv * head_dim + d];
                }
            }

            for qh in 0..heads {
                for d in 0..head_dim {
                    q_head[d] = q_raw[qh * head_dim + d];
                }
                rms_norm_milli(&q_head, &layer.q_norm, &mut q_normed);
                self.rope.apply(&mut q_normed, pos);
                let kv = qh / (heads / kv_heads);
                for (t, score) in scores.iter_mut().enumerate().take(pos + 1) {
                    let mut dot: i64 = 0;
                    let kc = &ses.k_cache[l][t][kv * head_dim..];
                    for (&qd, &kd) in q_normed.iter().zip(kc.iter().take(head_dim)) {
                        dot += i64::from(qd) * i64::from(kd);
                    }
                    let s = dot * self.score.0 + dot * self.score.1 / 10_000;
                    // dot is milli^2 (real x 1e6); score is milli — so the
                    // conversion is x (1000/√head_dim) / 1e6, NOT /1e3 (the
                    // /1e3 variant made every score 1000x too large,
                    // saturating softmax into a hard argmax — the 15%
                    // block-0 drift found by the C-core microscope).
                    *score = div_round_half_away(s, 1_000_000)
                        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                }
                self.kit.softmax_q12(&scores, &mut probs);
                acc.iter_mut().for_each(|a| *a = 0);
                for (t, &p) in probs.iter().enumerate().take(pos + 1) {
                    let p = i64::from(p);
                    for (a, &v) in acc
                        .iter_mut()
                        .zip(ses.v_cache[l][t][kv * head_dim..].iter().take(head_dim))
                    {
                        *a += p * i64::from(v);
                    }
                }
                for (d, &a) in acc.iter().enumerate() {
                    ctx[qh * head_dim + d] = div_round_half_away(a, 4096)
                        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                }
            }
            matvec_scaled(&layer.out_w, &ctx, emb, &mut attn_out)
                .map_err(|_| ModelError::BadTensorSize("out".into()))?;
            for d in 0..emb {
                h[d] = h[d].saturating_add(attn_out[d]);
            }

            // ---- FFN ----
            rms_norm_milli(&h, &layer.ffn_norm, &mut normed);
            let mut gate = vec![0_i32; ffn];
            let mut up = vec![0_i32; ffn];
            matvec_scaled(&layer.gate, &normed, ffn, &mut gate)
                .map_err(|_| ModelError::BadTensorSize("gate".into()))?;
            matvec_scaled(&layer.up, &normed, ffn, &mut up)
                .map_err(|_| ModelError::BadTensorSize("up".into()))?;
            for i in 0..ffn {
                let s = i64::from(self.kit.silu_milli(gate[i]));
                let u = i64::from(up[i]);
                gate[i] = div_round_half_away(s * u, 1000)
                    .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            }
            let mut down = vec![0_i32; emb];
            matvec_scaled(&layer.down, &gate, emb, &mut down)
                .map_err(|_| ModelError::BadTensorSize("down".into()))?;
            for d in 0..emb {
                h[d] = h[d].saturating_add(down[d]);
            }
            let layer_max: i64 = h.iter().map(|v| i64::from(v.unsigned_abs())).max().unwrap_or(0);
            ses.max_abs_residual =
                ses.max_abs_residual.max(layer_max.min(i64::from(i32::MAX)) as i32);
        }
        ses.n += 1;
        Ok(h)
    }

    fn forward_inner(
        &mut self,
        tokens: &[u32],
        mut health: Option<&mut ForwardHealth>,
    ) -> Result<Vec<Vec<i32>>, ModelError> {
        let n = tokens.len();
        if n == 0 || n > self.max_pos {
            return Err(ModelError::PositionOutOfRange);
        }
        let (emb, heads, kv_heads, head_dim, ffn) =
            (self.cfg.emb, self.cfg.heads, self.cfg.kv_heads, self.cfg.head_dim, self.cfg.ffn);
        let kv_width = kv_heads * head_dim;
        // Hidden states per position (milli).
        let mut h: Vec<Vec<i32>> = Vec::with_capacity(n);
        for &t in tokens {
            let mut row = vec![0_i32; emb];
            self.embed(t, &mut row)?;
            h.push(row);
        }
        if self.capture {
            self.block_states.push(h.clone());
        }
        // KV caches: [layer][pos], flat kv_heads·head_dim per position.
        let mut k_cache = vec![vec![vec![0_i32; kv_width]; n]; self.layers.len()];
        let mut v_cache = vec![vec![vec![0_i32; kv_width]; n]; self.layers.len()];
        // Score/prob scratch sized by n — the 2026-08-15 review found
        // fixed [i32; 64] arrays panicking on any prompt longer than 64
        // tokens whenever max_pos was loaded larger.
        let mut scores = vec![0_i32; n];
        let mut probs = vec![0_i32; n];

        for (l, layer) in self.layers.iter().enumerate() {
            let prev_h = h.clone();
            // ---- Attention ----
            let mut q_raw = vec![0_i32; heads * head_dim];
            let mut k_raw = vec![0_i32; kv_width];
            let mut v_raw = vec![0_i32; kv_width];
            let mut normed = vec![0_i32; emb];
            let mut attn_out = vec![0_i32; emb];
            let mut ctx = vec![0_i32; heads * head_dim];
            let mut q_head = vec![0_i32; head_dim];
            let mut k_head = vec![0_i32; head_dim];
            let mut q_normed = vec![0_i32; head_dim];
            let mut k_normed = vec![0_i32; head_dim];
            let mut acc = vec![0_i64; head_dim];
            for pos in 0..n {
                rms_norm_milli(&h[pos], &layer.attn_norm, &mut normed);
                matvec_scaled(&layer.q, &normed, heads * head_dim, &mut q_raw)
                    .map_err(|_| ModelError::BadTensorSize("q".into()))?;
                matvec_scaled(&layer.k, &normed, kv_width, &mut k_raw)
                    .map_err(|_| ModelError::BadTensorSize("k".into()))?;
                matvec_scaled(&layer.v, &normed, kv_width, &mut v_raw)
                    .map_err(|_| ModelError::BadTensorSize("v".into()))?;

                // Per-head q/k norm + rope; stash KV.
                for kv in 0..kv_heads {
                    let (kn, vn) = (&mut k_cache[l][pos], &mut v_cache[l][pos]);
                    for d in 0..head_dim {
                        kn[kv * head_dim + d] = k_raw[kv * head_dim + d];
                        vn[kv * head_dim + d] = v_raw[kv * head_dim + d];
                    }
                    for d in 0..head_dim {
                        k_head[d] = kn[kv * head_dim + d];
                    }
                    rms_norm_milli(&k_head, &layer.k_norm, &mut k_normed);
                    self.rope.apply(&mut k_normed, pos); // pos < n ≤ max_pos, safe
                    for d in 0..head_dim {
                        kn[kv * head_dim + d] = k_normed[d];
                    }
                }
                // Attention output for this position (context, milli).
                for qh in 0..heads {
                    for d in 0..head_dim {
                        q_head[d] = q_raw[qh * head_dim + d];
                    }
                    rms_norm_milli(&q_head, &layer.q_norm, &mut q_normed);
                    self.rope.apply(&mut q_normed, pos); // safe: pos < max_pos
                    // Scores over t ≤ pos against kv head qh/(heads/kv_heads).
                    let kv = qh / (heads / kv_heads);
                    for t in 0..=pos {
                        let mut dot: i64 = 0; // Σ q_milli·k_milli = real×1e6
                        for d in 0..head_dim {
                            dot += i64::from(q_normed[d])
                                * i64::from(k_cache[l][t][kv * head_dim + d]);
                        }
                        // dot is milli^2; milli scores = dot × (main +
                        // extra/10⁴) / 1e6 with main+extra/10⁴ =
                        // 1000/√head_dim (bounded: |dot| ≲ head_dim·1e10;
                        // the largest split product ≈ head_dim·1e13 «
                        // i64::MAX).
                        let s = dot * self.score.0 + dot * self.score.1 / 10_000;
                        scores[t] = div_round_half_away(s, 1_000_000)
                            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                    }
                    self.kit.softmax_q12(&scores[..=pos], &mut probs[..=pos]);
                    // Context = Σ_t p_t × v_t / Q12.
                    acc.iter_mut().for_each(|a| *a = 0);
                    for t in 0..=pos {
                        let p = i64::from(probs[t]);
                        for d in 0..head_dim {
                            acc[d] += p * i64::from(v_cache[l][t][kv * head_dim + d]);
                        }
                    }
                    for d in 0..head_dim {
                        ctx[qh * head_dim + d] = div_round_half_away(acc[d], 4096)
                            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                    }
                }
                // Output projection + residual.
                matvec_scaled(&layer.out_w, &ctx, emb, &mut attn_out)
                    .map_err(|_| ModelError::BadTensorSize("out".into()))?;
                for d in 0..emb {
                    h[pos][d] = h[pos][d].saturating_add(attn_out[d]);
                }
            }

            // ---- FFN (per position, after attention residual) ----
            let mut gate = vec![0_i32; ffn];
            let mut up = vec![0_i32; ffn];
            let mut down = vec![0_i32; emb];
            if self.capture {
                // Substep snapshot: after this block's attention residual.
                self.block_states.push(h.clone());
            }
            for hp in h.iter_mut().take(n) {
                rms_norm_milli(hp, &layer.ffn_norm, &mut normed);
                matvec_scaled(&layer.gate, &normed, ffn, &mut gate)
                    .map_err(|_| ModelError::BadTensorSize("gate".into()))?;
                matvec_scaled(&layer.up, &normed, ffn, &mut up)
                    .map_err(|_| ModelError::BadTensorSize("up".into()))?;
                // SiLU(gate) ⊙ up, milli: silu_milli × up / 1000.
                for i in 0..ffn {
                    let s = i64::from(self.kit.silu_milli(gate[i]));
                    let u = i64::from(up[i]);
                    gate[i] = div_round_half_away(s * u, 1000)
                        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                }
                matvec_scaled(&layer.down, &gate, emb, &mut down)
                    .map_err(|_| ModelError::BadTensorSize("down".into()))?;
                for d in 0..emb {
                    hp[d] = hp[d].saturating_add(down[d]);
                }
            }

            if self.capture {
                self.block_states.push(h.clone());
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
    /// [`ModelError::BadTensorSize`] if the embedding tensor is corrupt
    /// (surfaces from the row scan).
    pub fn topk_logits(&self, h: &[i32], k: usize) -> Result<Vec<(u32, i32)>, ModelError> {
        let emb = self.cfg.emb;
        let mut normed = vec![0_i32; emb];
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
        let row_bytes = self.emb_row_bytes();
        let rows = self.emb.len() / row_bytes;
        let mut raw = [0_i32; 1];
        let mut all: Vec<(u32, i32)> = Vec::with_capacity(rows);
        for t in 0..rows {
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

    /// Greedy decoding's pick: the argmax logit over the tied embedding
    /// for one hidden state — O(vocab) single pass, no full sort (the
    /// generation loop calls this once per token). Ties break to the
    /// LOWEST id (first-seen wins on strict `>`), matching
    /// [`Qwen3::topk_logits`]'s stable-sort convention. Same units and
    /// zero/degenerate contract as [`Qwen3::topk_logits`].
    ///
    /// # Errors
    ///
    /// [`ModelError::BadTensorSize`] if the embedding tensor is corrupt
    /// (surfaces from the row scan).
    pub fn argmax_logit(&self, h: &[i32]) -> Result<(u32, i32), ModelError> {
        let emb = self.cfg.emb;
        let mut normed = vec![0_i32; emb];
        rms_norm_milli(h, &self.out_norm, &mut normed);
        let amax = normed.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
        if amax == 0 {
            return Ok((0, 0)); // degenerate hidden — sentinel, as documented
        }
        let acts: Vec<i16> = normed
            .iter()
            .map(|&v| {
                div_round_half_away(i64::from(v) * 32_767, i64::from(amax))
                    .clamp(i16::MIN as i64, i16::MAX as i64) as i16
            })
            .collect();
        let row_bytes = self.emb_row_bytes();
        let rows = self.emb.len() / row_bytes;
        let mut raw = [0_i32; 1];
        let mut best: (u32, i32) = (0, i32::MIN);
        for t in 0..rows {
            let row = &self.emb[t * row_bytes..][..row_bytes];
            crate::q1_0::q1_0_matvec(row, &acts, 1, &mut raw)
                .map_err(|_| ModelError::BadTensorSize("emb".into()))?;
            let milli = div_round_half_away(i64::from(raw[0]) * i64::from(amax), 32_767)
                .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            if milli > best.1 {
                best = (t as u32, milli);
            }
        }
        Ok(best)
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
        const HD: usize = 128; // head_dim on both real files
        let kit = MathKit::new();
        let rope = RopeTables::new_yarn(HD, 16, 1e6, 4.0, 8192, 32.0, 1.0);
        let (score_main, score_extra) = ModelConfig::score_scale(HD);
        assert_eq!((score_main, score_extra), (88, 3883)); // the 1.7B pin
        let pos = 5;

        // Synthetic q/k heads (milli), one k per past position.
        let q_src: [i32; HD] = core::array::from_fn(|i| (i as i32 * 61) % 300 - 150);
        let k_src: [[i32; HD]; 6] = core::array::from_fn(|t| {
            core::array::from_fn(|i| ((i as i32 + t as i32 * 13) * 37) % 400 - 200)
        });
        let v: [[i32; HD]; 6] = core::array::from_fn(|t| {
            core::array::from_fn(|i| ((i as i32 + t as i32 * 7) * 53) % 200 - 100)
        });

        // Integer path: rope q and each k, dot × scale, softmax, merge v.
        let mut q = q_src;
        rope.apply(&mut q, pos);
        let mut kr = [[0_i32; HD]; 6];
        for (t, k) in k_src.iter().enumerate() {
            kr[t] = *k;
            rope.apply(&mut kr[t], t);
        }
        let mut scores = [0_i32; 6];
        for t in 0..=pos {
            let mut dot: i64 = 0;
            for d in 0..HD {
                dot += i64::from(q[d]) * i64::from(kr[t][d]);
            }
            let s = dot * score_main + dot * score_extra / 10_000;
            // Production chain (fixed 2026-08-16): dot is milli^2, milli
            // scores = dot x 88.3883 / 1e6.
            scores[t] = div_round_half_away(s, 1_000_000) as i32;
        }
        let mut probs = [0_i32; 6];
        kit.softmax_q12(&scores[..=pos], &mut probs[..=pos]);
        let mut ctx = [0_i64; HD];
        for t in 0..=pos {
            for d in 0..HD {
                ctx[d] += i64::from(probs[t]) * i64::from(v[t][d]);
            }
        }

        // f64 reference in REAL units end-to-end (milli inputs /1000,
        // exact softmax) — no shared unit chain with the integer side
        // (the pre-fix version divided by 1000 twice on both sides and
        // the circularity hid the 1000x score-scale bug).
        let corr = |n_rot: f64| -> f64 {
            128.0 * (8192.0 / (n_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * 1e6_f64.ln())
        };
        let (low, high) = (corr(32.0).floor().max(0.0), corr(1.0).ceil().min(127.0));
        let mscale = 1.0 + 0.1 * 4.0_f64.ln();
        let angle = |p: usize, i: usize| -> (f64, f64) {
            let extrap = p as f64 * 1e6_f64.powf(-(2.0 * i as f64) / 128.0);
            let interp = 0.25 * extrap;
            let y = ((i as f64 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
            let r = 1.0 - y;
            let th = interp * (1.0 - r) + extrap * r;
            (th.cos() * mscale, th.sin() * mscale)
        };
        let rot = |p: usize, v: &[i32]| -> Vec<f64> {
            let mut out = vec![0.0; HD];
            for i in 0..64 {
                let (c, s) = angle(p, i);
                let (x1, x2) = (f64::from(v[i]) / 1000.0, f64::from(v[i + 64]) / 1000.0);
                out[i] = x1 * c - x2 * s;
                out[i + 64] = x2 * c + x1 * s;
            }
            out
        };
        let qf = rot(pos, &q_src);
        let kf: Vec<Vec<f64>> = (0..=pos).map(|t| rot(t, &k_src[t])).collect();
        let scores_real: Vec<f64> = (0..=pos)
            .map(|t| {
                let dot: f64 = (0..HD).map(|i| qf[i] * kf[t][i]).sum();
                dot / (HD as f64).sqrt()
            })
            .collect();
        let mx = scores_real.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores_real.iter().map(|s| (s - mx).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for d in 0..HD {
            let want: f64 = (0..=pos)
                .map(|t| exps[t] / sum * f64::from(v[t][d]) / 1000.0)
                .sum();
            // ctx accumulates probs(Q12) x v(milli): /4096 -> milli, /1000
            // -> real.
            let got = ctx[d] as f64 / 4096.0 / 1000.0;
            assert!(
                (got - want).abs() <= 0.004 + want.abs() * 0.02,
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
    //
    // The synthetic dims mirror the REAL 1.7B file (assertions below
    // were derived on it); synthetic_model_cfg lets tests build other
    // shapes.

    fn cfg_17b(vocab: usize) -> ModelConfig {
        ModelConfig {
            emb: 2048,
            heads: 16,
            kv_heads: 8,
            head_dim: 128,
            ffn: 6144,
            layers: 1,
            vocab,
            rope_base: 1e6,
            yarn_factor: 4.0,
            orig_ctx: 8192,
        }
    }

    fn synthetic_model_with(cfg: ModelConfig) -> Qwen3 {
        let row_bytes = |rows: usize, width: usize| rows * (width / 128) * 18;
        let layer = LayerSlices {
            attn_norm: vec![0_i32; cfg.emb],
            q: vec![0_u8; row_bytes(cfg.heads * cfg.head_dim, cfg.emb)],
            k: vec![0_u8; row_bytes(cfg.kv_heads * cfg.head_dim, cfg.emb)],
            v: vec![0_u8; row_bytes(cfg.kv_heads * cfg.head_dim, cfg.emb)],
            q_norm: vec![0_i32; cfg.head_dim],
            k_norm: vec![0_i32; cfg.head_dim],
            out_w: vec![0_u8; row_bytes(cfg.emb, cfg.heads * cfg.head_dim)],
            ffn_norm: vec![0_i32; cfg.emb],
            gate: vec![0_u8; row_bytes(cfg.ffn, cfg.emb)],
            up: vec![0_u8; row_bytes(cfg.ffn, cfg.emb)],
            down: vec![0_u8; row_bytes(cfg.emb, cfg.ffn)],
        };
        Qwen3 {
            cfg: cfg.clone(),
            layers: vec![layer],
            out_norm: vec![0_i32; cfg.emb],
            emb: vec![0_u8; row_bytes(cfg.vocab, cfg.emb)],
            kit: MathKit::new(),
            rope: RopeTables::new_yarn(
                cfg.head_dim,
                64,
                cfg.rope_base,
                cfg.yarn_factor,
                cfg.orig_ctx,
                32.0,
                1.0,
            ),
            score: ModelConfig::score_scale(cfg.head_dim),
            max_pos: 64,
            block_states: Vec::new(),
            capture: false,
        }
    }

    fn synthetic_model(max_pos: usize, emb_rows: usize) -> Qwen3 {
        let mut m = synthetic_model_with(cfg_17b(emb_rows));
        m.max_pos = max_pos;
        // Rebuild rope at the caller's max_pos (cheap, tiny tables).
        m.rope = RopeTables::new_yarn(m.cfg.head_dim, max_pos, 1e6, 4.0, 8192, 32.0, 1.0);
        m
    }

    #[test]
    fn forward_past_64_tokens_is_ok_not_panic() {
        // max_pos is caller-chosen; the old fixed score/probs arrays
        // panicked at pos 64 whenever max_pos ≥ 65.
        let mut model = synthetic_model(128, 128);
        let tokens: Vec<u32> = (0..65_u32).collect();
        let h = model.forward(&tokens).expect("65-token forward");
        assert_eq!(h.len(), 65);
        assert_eq!(h[0].len(), 2048);
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
        // Synthetic model: 8 embedding rows (load() derives vocab from
        // the tensor on real files; the guard uses the actual row
        // count, which is the honest bound for any constructed model).
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
    fn loader_accepts_alignment_padding_and_copies_exact_bytes() {
        // The 4B finding: token_embd sits 24 B short of a 32-byte
        // boundary, so its slice runs to the next tensor's aligned
        // offset. Expect-size must pass, and only the formula bytes
        // may be copied (padding never enters the model).
        let expect = 3 * 18; // 54 B: 3 rows × 1 block (width 128)
        let data = vec![0xA5_u8; expect + 32 - expect % 32]; // padded to 32
        let b = one_tensor_file("t.weight", &[128, 3], 41, &data);
        let f = GgufFile::parse(&b).expect("parse");
        let got = q1_0_tensor(&f, "t.weight", 3, 128).expect("padded slice loads");
        assert_eq!(got.len(), expect, "padding must not be copied");
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
            expect_kv(&f, "qwen3.block_count", LAYERS_FOR_KV_TEST as i64),
            Err(ModelError::ConfigMismatch(
                "qwen3.block_count = 27, want 28".into()
            ))
        );
        // Architecture mismatch caught at load.
        let _ = data;
    }

    /// The value the original pinned-constants loader asserted (28) —
    /// kept as a literal here because the config-driven loader no
    /// longer HAS a block-count constant to compare against.
    const LAYERS_FOR_KV_TEST: usize = 28;

    #[test]
    fn expect_kv_passes_on_absent_and_matching() {
        let b = one_tensor_file("t", &[8, 1], 0, &[0_u8; 32]);
        let f = GgufFile::parse(&b).expect("parse");
        assert_eq!(expect_kv(&f, "qwen3.block_count", 28), Ok(()));
    }

    /// Synthetic KV-only GGUF builder (n_tensors = 0).
    fn kv_file(kvs: &[(&str, MetadataValue)]) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3_u32.to_le_bytes());
        b.extend_from_slice(&0_u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            let (ty, payload): (u32, Vec<u8>) = match v {
                MetadataValue::U32(x) => (4, x.to_le_bytes().to_vec()),
                MetadataValue::F32(x) => (6, x.to_le_bytes().to_vec()),
                MetadataValue::String(s) => {
                    let mut p = (s.len() as u64).to_le_bytes().to_vec();
                    p.extend_from_slice(s.as_bytes());
                    (8, p)
                }
                _ => unimplemented!("test builder covers U32/F32/String"),
            };
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&payload);
        }
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        b
    }

    /// The full 4B config block, as the real file carries it (probe
    /// evidence) — the from_gguf acceptance vector.
    fn config_kvs_4b() -> Vec<(&'static str, MetadataValue)> {
        vec![
            ("qwen3.block_count", MetadataValue::U32(36)),
            ("qwen3.embedding_length", MetadataValue::U32(2560)),
            ("qwen3.feed_forward_length", MetadataValue::U32(9728)),
            ("qwen3.attention.head_count", MetadataValue::U32(32)),
            ("qwen3.attention.head_count_kv", MetadataValue::U32(8)),
            ("qwen3.attention.key_length", MetadataValue::U32(128)),
            ("qwen3.attention.value_length", MetadataValue::U32(128)),
            ("qwen3.rope.freq_base", MetadataValue::F32(5_000_000.0)),
            (
                "qwen3.rope.scaling.type",
                MetadataValue::String("yarn".into()),
            ),
            ("qwen3.rope.scaling.factor", MetadataValue::F32(4.0)),
            (
                "qwen3.rope.scaling.original_context_length",
                MetadataValue::U32(8192),
            ),
            (
                "qwen3.attention.layer_norm_rms_epsilon",
                MetadataValue::F32(1e-6),
            ),
        ]
    }

    #[test]
    fn from_gguf_reads_the_4b_config_block() {
        let b = kv_file(&config_kvs_4b());
        let f = GgufFile::parse(&b).expect("parse");
        let cfg = ModelConfig::from_gguf(&f).expect("4B config");
        assert_eq!(
            cfg,
            ModelConfig {
                emb: 2560,
                heads: 32,
                kv_heads: 8,
                head_dim: 128,
                ffn: 9728,
                layers: 36,
                vocab: 0, // filled by load() from the embedding rows
                rope_base: 5_000_000.0,
                yarn_factor: 4.0,
                orig_ctx: 8192,
            }
        );
    }

    #[test]
    fn from_gguf_is_loud_on_missing_and_broken() {
        // Empty KV set: the FIRST required key is named.
        let b = kv_file(&[]);
        let f = GgufFile::parse(&b).expect("parse");
        assert!(matches!(
            ModelConfig::from_gguf(&f),
            Err(ModelError::ConfigMismatch(msg)) if msg.contains("qwen3.embedding_length")
        ));
        // Drop key_length from the 4B block → named.
        let kvs: Vec<(&str, MetadataValue)> = config_kvs_4b()
            .into_iter()
            .filter(|(k, _)| *k != "qwen3.attention.key_length")
            .collect();
        let b = kv_file(&kvs);
        let f = GgufFile::parse(&b).expect("parse");
        assert!(matches!(
            ModelConfig::from_gguf(&f),
            Err(ModelError::ConfigMismatch(msg)) if msg.contains("key_length")
        ));
        // value_length disagreeing with key_length → loud.
        let mut kvs = config_kvs_4b();
        for (k, v) in &mut kvs {
            if *k == "qwen3.attention.value_length" {
                *v = MetadataValue::U32(64);
            }
        }
        let b = kv_file(&kvs);
        let f = GgufFile::parse(&b).expect("parse");
        assert!(matches!(
            ModelConfig::from_gguf(&f),
            Err(ModelError::ConfigMismatch(msg)) if msg.contains("value_length")
        ));
        // A non-yarn scaling type → loud.
        let mut kvs = config_kvs_4b();
        for (k, v) in &mut kvs {
            if *k == "qwen3.rope.scaling.type" {
                *v = MetadataValue::String("linear".into());
            }
        }
        let b = kv_file(&kvs);
        let f = GgufFile::parse(&b).expect("parse");
        assert!(matches!(
            ModelConfig::from_gguf(&f),
            Err(ModelError::ConfigMismatch(msg)) if msg.contains("yarn")
        ));
        // head_count not a multiple of head_count_kv → loud.
        let mut kvs = config_kvs_4b();
        for (k, v) in &mut kvs {
            if *k == "qwen3.attention.head_count" {
                *v = MetadataValue::U32(30);
            }
        }
        let b = kv_file(&kvs);
        let f = GgufFile::parse(&b).expect("parse");
        assert!(matches!(
            ModelConfig::from_gguf(&f),
            Err(ModelError::ConfigMismatch(msg)) if msg.contains("head_count")
        ));
    }

    #[test]
    fn score_scale_pins_the_1_7b_split() {
        // head_dim 128 (both real files) must reproduce the pinned
        // constants bit-identically — the 1.7B evidence depends on it.
        assert_eq!(ModelConfig::score_scale(128), (88, 3883));
        // Exact and half splits on power-of-two dims (derived, not
        // hand-waved: 1000/8 = 125 exactly; 1000/16 = 62.5).
        assert_eq!(ModelConfig::score_scale(64), (125, 0));
        assert_eq!(ModelConfig::score_scale(256), (62, 5000));
    }

    #[test]
    fn residual_rail_derives_per_model() {
        // emb 2048 → exactly 2^26 (2^63/2^11 = 2^52, √ = 2^26); the
        // frozen const stays the conservative rounding of it.
        let m = synthetic_model(8, 8);
        assert_eq!(m.residual_sound_max(), 67_108_864);
        assert!(m.residual_sound_max() >= RESIDUAL_SOUND_MAX);
        // Wider hidden → tighter rail (4B: emb 2560).
        let m4 = synthetic_model_with(ModelConfig { emb: 2560, ..cfg_17b(8) });
        assert!(m4.residual_sound_max() < m.residual_sound_max());
    }

    // ---- Session (incremental decode) + argmax — session 4 ----

    /// A NONZERO synthetic model: every q1_0 block carries fp16 scale
    /// 1.0 (0x3C00) and the 0xAA sign pattern; every norm weight is
    /// 1.0 (1000 milli). Real arithmetic flows (unlike the zero model,
    /// where equality is vacuous) at synthetic cost: 1 layer.
    fn synthetic_model_nonzero(max_pos: usize, emb_rows: usize) -> Qwen3 {
        let fill = |rows: usize, width: usize| -> Vec<u8> {
            let blocks = width / 128;
            let mut d = Vec::with_capacity(rows * blocks * 18);
            for _ in 0..rows * blocks {
                d.extend_from_slice(&0x3C00_u16.to_le_bytes());
                d.extend(std::iter::repeat_n(0xAA_u8, 16));
            }
            d
        };
        let cfg = cfg_17b(emb_rows);
        let layer = LayerSlices {
            attn_norm: vec![1000_i32; cfg.emb],
            q: fill(cfg.heads * cfg.head_dim, cfg.emb),
            k: fill(cfg.kv_heads * cfg.head_dim, cfg.emb),
            v: fill(cfg.kv_heads * cfg.head_dim, cfg.emb),
            q_norm: vec![1000_i32; cfg.head_dim],
            k_norm: vec![1000_i32; cfg.head_dim],
            out_w: fill(cfg.emb, cfg.heads * cfg.head_dim),
            ffn_norm: vec![1000_i32; cfg.emb],
            gate: fill(cfg.ffn, cfg.emb),
            up: fill(cfg.ffn, cfg.emb),
            down: fill(cfg.emb, cfg.ffn),
        };
        Qwen3 {
            cfg: cfg.clone(),
            layers: vec![layer],
            out_norm: vec![1000_i32; cfg.emb],
            emb: fill(cfg.vocab, cfg.emb),
            kit: MathKit::new(),
            rope: RopeTables::new_yarn(
                cfg.head_dim,
                max_pos,
                cfg.rope_base,
                cfg.yarn_factor,
                cfg.orig_ctx,
                32.0,
                1.0,
            ),
            score: ModelConfig::score_scale(cfg.head_dim),
            max_pos,
            block_states: Vec::new(),
            capture: false,
        }
    }

    #[test]
    fn incremental_matches_forward_synthetic_exact() {
        // The mission's equivalence falsifier, CI-runnable: the
        // incremental path must reproduce the full forward BIT-EXACTLY
        // (tolerance 0) on real (nonzero) arithmetic.
        let prompt: &[u32] = &[0, 42, 7, 3];
        let mut model = synthetic_model_nonzero(16, 64);
        let full = model.forward(prompt).expect("full forward");

        let mut ses = model.new_session();
        assert!(ses.is_empty());
        let inc = model.prefill(&mut ses, prompt).expect("prefill");
        assert_eq!(ses.len(), prompt.len());
        assert_eq!(inc.len(), full.len());
        for (pos, (a, b)) in inc.iter().zip(full.iter()).enumerate() {
            assert_eq!(a, b, "hidden mismatch at pos {pos}");
        }

        // Appending one more token must match a fresh full forward over
        // prompt+token — still bit-exact.
        let mut ses2 = model.new_session();
        model.prefill(&mut ses2, prompt).expect("prefill 2");
        let h_next = model.step(&mut ses2, 5).expect("step");
        let full2 = model.forward(&[0, 42, 7, 3, 5]).expect("full forward 2");
        assert_eq!(&h_next, full2.last().expect("nonempty"));

        // The residual witness is populated and sane on nonzero weights.
        assert!(ses2.max_abs_residual() > 0);
    }

    #[test]
    fn session_error_paths() {
        let model = synthetic_model_nonzero(2, 8);
        let mut ses = model.new_session();
        assert!(model.step(&mut ses, 0).is_ok());
        assert!(model.step(&mut ses, 1).is_ok());
        // Rope table exhausted at max_pos.
        assert_eq!(
            model.step(&mut ses, 0),
            Err(ModelError::PositionOutOfRange)
        );
        // Out-of-vocabulary ids are loud on the incremental path too.
        let mut ses2 = model.new_session();
        assert_eq!(
            model.step(&mut ses2, 8),
            Err(ModelError::TokenOutOfRange)
        );
        assert!(ses2.is_empty(), "failed step must not advance the session");
    }

    #[test]
    fn argmax_tie_breaks_lowest_and_finds_peak() {
        // Uniform logits (zero-weight model) → tie → lowest id (0).
        let mut model = synthetic_model(8, 8);
        let h = model.forward(&[0]).expect("forward");
        assert_eq!(model.argmax_logit(&h[0]).unwrap(), (0, 0));

        // A peak row wins: embedding row 3 with a large scale and all
        // + signs dominates every dot product.
        let mut model = synthetic_model_nonzero(4, 4);
        let row_bytes = 2048 / 128 * 18;
        for b in 0..2048 / 128 {
            let base = 3 * row_bytes + b * 18;
            model.emb[base..base + 2].copy_from_slice(&0x7BFF_u16.to_le_bytes()); // fp16 ≈ 6.5e4
        }
        let h = model.forward(&[0]).expect("forward");
        let (id, val) = model.argmax_logit(&h[0]).unwrap();
        assert_eq!(id, 3);
        assert!(val > 0);
        // And argmax agrees with topk's first element (convention pin).
        let top = model.topk_logits(&h[0], 3).unwrap();
        assert_eq!(top[0], (id, val));
    }

    #[test]
    #[ignore = "needs models/Bonsai-1.7B-Q1_0.gguf (gitignored, 248 MB)"]
    fn real_incremental_matches_forward_exact() {
        let Some(buf) = ["models/Bonsai-1.7B-Q1_0.gguf", "../../models/Bonsai-1.7B-Q1_0.gguf"]
            .iter()
            .find_map(|p| std::fs::read(p).ok())
        else {
            eprintln!("model file absent — skipping");
            return;
        };
        let f = GgufFile::parse(&buf).expect("container parses");
        let mut model = Qwen3::load(&f, 16).expect("model loads");
        let prompt: &[u32] = &[0, 42, 151_668, 7];
        let (full, health) = model.forward_with_health(prompt).expect("full forward");

        let mut ses = model.new_session();
        let inc = model.prefill(&mut ses, prompt).expect("prefill");
        assert_eq!(inc.len(), full.len());
        for (pos, (a, b)) in inc.iter().zip(full.iter()).enumerate() {
            assert_eq!(a, b, "REAL MODEL: hidden mismatch at pos {pos}");
        }
        // One appended token, still exact vs a 5-token full forward.
        let h5 = model.step(&mut ses, 9).expect("step");
        let full5 = model.forward(&[0, 42, 151_668, 7, 9]).expect("forward 5");
        assert_eq!(&h5, full5.last().unwrap());
        // Residual witness populated and under the norm-soundness rail.
        assert!(ses.max_abs_residual() > 0);
        assert!(
            i64::from(ses.max_abs_residual()) < model.residual_sound_max()
        );
        assert_eq!(health.per_layer_delta.len(), 28);
    }

    /// BOTH real files load through the config-driven loader with the
    /// geometry the probe pinned, and a 2-token forward produces sane
    /// (nonzero, under-rail) residuals on each — the 4B session's
    /// loader acceptance test.
    #[test]
    #[ignore = "needs models/Bonsai-{1.7B,4B}-Q1_0.gguf (gitignored)"]
    fn real_files_load_with_expected_configs() {
        let cases = [
            (
                "models/Bonsai-1.7B-Q1_0.gguf",
                ModelConfig {
                    emb: 2048,
                    heads: 16,
                    kv_heads: 8,
                    head_dim: 128,
                    ffn: 6144,
                    layers: 28,
                    vocab: 151_669,
                    rope_base: 1e6,
                    yarn_factor: 4.0,
                    orig_ctx: 8192,
                },
            ),
            (
                "models/Bonsai-4B-Q1_0.gguf",
                ModelConfig {
                    emb: 2560,
                    heads: 32,
                    kv_heads: 8,
                    head_dim: 128,
                    ffn: 9728,
                    layers: 36,
                    vocab: 151_669,
                    rope_base: 5e6,
                    yarn_factor: 4.0,
                    orig_ctx: 8192,
                },
            ),
        ];
        for (path, want) in cases {
            let Some(buf) = std::fs::read(path).ok().or_else(|| {
                std::fs::read(format!("../../{path}")).ok()
            }) else {
                panic!("{path} absent — this test requires both real files");
            };
            let f = GgufFile::parse(&buf).expect("container parses");
            let mut model = Qwen3::load(&f, 8).expect("model loads");
            assert_eq!(*model.config(), want, "{path}: config");
            assert!(model.residual_sound_max() > 0);
            let (h, health) = model.forward_with_health(&[0, 42]).expect("2-token forward");
            assert_eq!(h.len(), 2);
            assert_eq!(h[0].len(), want.emb);
            assert_eq!(health.per_layer_delta.len(), want.layers);
            assert!(health.per_layer_delta.iter().all(|&d| d > 0), "{path}: dead layer");
            assert!(
                i64::from(health.max_abs_residual) < model.residual_sound_max(),
                "{path}: residual {} vs rail {}",
                health.max_abs_residual,
                model.residual_sound_max()
            );
        }
    }
}
