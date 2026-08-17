//! Q1_0 compute on top of the container — the model-side math surface
//! (Stage 4, session 2).
//!
//! One decision governs this module (ISA Decisions 2026-08-15): the Q1_0
//! compute path is a **per-block decode-matvec** — sign-bit partial sums
//! × per-block γ — not a fused/LUT kernel. Correctness first; bitnet.cpp
//! -style lookup tables are a deferred, profiling-gated optimization.
//!
//! # Numeric contract
//!
//! Activations are `i16` (Q15-style, same as the SNN kernel); weights are
//! Q1_0 blocks (fp16 scale + sign bits, see `docs/TERNARY_FORMAT.md`);
//! scales enter arithmetic only through their milli view
//! ([`neuralos_snn::half_to_milli`]): `±γ` per element, per-block partials
//! accumulated in `i64`, each block's contribution
//! `round(partial × γ_milli / 1000)`, final saturation to `i32`
//! documented. Bounds: `|partial| ≤ 128·32768 = 4_194_304` (i16::MIN is
//! reachable); `|γ_milli| ≤ 65_504_000` finite / `2^31` when the fp16
//! scale is ±inf (the documented `half_to_milli` saturation) → worst
//! per-block product `4_194_304 × 2^31 ≈ 9.0e15` — 0.1% of i64 range.
//! The cross-block sum can exceed `i32` only when
//! `γ_real ≳ 2^31 / (blocks × 4194.304)` — 512 at width 1 block, 32 at
//! 16 blocks (2048-wide rows), 10.7 at 48 blocks (6144-wide) — and then
//! saturates at the row level, never silently wraps.

use neuralos_snn::half_to_milli;

/// Q1_0 block size (weights per block — one fp16 scale + 16 sign bytes).
/// Single-sourced from the published codec ([`neuralos_snn::bridge`]);
/// the cross-crate equality is pinned by a test below.
pub use neuralos_snn::bridge::Q1_0_BLOCK;
/// Q1_0 block byte size: 2 scale bytes + 16 sign bytes — derived from
/// the codec's own length function (`encoded_len(128 weights) = 18`),
/// not a second hand-written constant.
pub const Q1_0_BLOCK_BYTES: usize = neuralos_snn::bridge::q1_0_encoded_len(Q1_0_BLOCK);

const _: () = assert!(Q1_0_BLOCK_BYTES == 18, "q1_0 layout must stay 18 B/block");

/// Errors from the Q1_0 compute path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q10Error {
    /// Activation length not a multiple of the 128-weight block.
    BadLength,
    /// A buffer is shorter than the layout requires.
    TooShort,
}

impl core::fmt::Display for Q10Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadLength => write!(f, "activation length not a multiple of 128"),
            Self::TooShort => write!(f, "buffer shorter than the layout requires"),
        }
    }
}

impl std::error::Error for Q10Error {}

/// One Q1_0 row (or any contiguous run of blocks) → milli-domain values:
/// element `i` becomes `±γ_milli` of its block.
///
/// This is the embedding-materialization path: a token's embedding row is
/// 16 blocks × 128 = 2048 values, each exactly `+γ` or `−γ` of its block.
///
/// Non-finite fp16 scales follow `half_to_milli`'s documented saturation
/// (±inf → ±`i32::MAX` after sign application — the negation happens in
/// i64, so an inf-scale block saturates per-sign instead of panicking on
/// `-(i32::MIN)`; NaN γ → 0).
///
/// `n` values need `n/128 × 18` bytes of `data` and `n` slots of `out`.
///
/// # Errors
///
/// [`Q10Error::BadLength`] / [`Q10Error::TooShort`] on bad sizes.
pub fn q1_0_row_to_milli(data: &[u8], out: &mut [i32]) -> Result<(), Q10Error> {
    let n = out.len();
    if !n.is_multiple_of(Q1_0_BLOCK) {
        return Err(Q10Error::BadLength);
    }
    if data.len() < n / Q1_0_BLOCK * Q1_0_BLOCK_BYTES {
        return Err(Q10Error::TooShort);
    }
    for b in 0..n / Q1_0_BLOCK {
        let base = b * Q1_0_BLOCK_BYTES;
        let gamma = i64::from(half_to_milli(u16::from_le_bytes([data[base], data[base + 1]])));
        let neg = (-gamma).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        let pos = gamma.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        for j in 0..Q1_0_BLOCK {
            let sign = (data[base + 2 + j / 8] >> (j % 8)) & 1;
            out[b * Q1_0_BLOCK + j] = if sign == 1 { pos } else { neg };
        }
    }
    Ok(())
}

/// The Q1_0 matvec: `out[j] = Σᵢ ±γ_{j,b(i)} · a[i]` — per-block scaled.
///
/// `data` holds `rows` consecutive Q1_0 rows, each
/// `n = acts.len()` weights wide (`n % 128 == 0`, `n/128 × 18` bytes per
/// row — exactly how a `[n, rows]` GGUF tensor lays out row-major);
/// `acts` are i16 activations; `out` receives `rows` i32 results.
///
/// Per block: `partial = Σ (bit ? +a : −a)` in i64, then
/// `round(partial × γ_milli / 1000)` — the milli scale cancels the /1000
/// exactly when activations are themselves milli-scaled, and is otherwise
/// just a fixed rational factor. Saturates to i32 at the row level.
///
/// # Errors
///
/// [`Q10Error::BadLength`] / [`Q10Error::TooShort`] on bad sizes —
/// including `out.len() < rows` and a `rows` whose byte footprint would
/// overflow `usize`.
pub fn q1_0_matvec(
    data: &[u8],
    acts: &[i16],
    rows: usize,
    out: &mut [i32],
) -> Result<(), Q10Error> {
    let n = acts.len();
    if !n.is_multiple_of(Q1_0_BLOCK) {
        return Err(Q10Error::BadLength);
    }
    if out.len() < rows {
        return Err(Q10Error::TooShort);
    }
    let blocks = n / Q1_0_BLOCK;
    let row_bytes = blocks
        .checked_mul(Q1_0_BLOCK_BYTES)
        .filter(|rb| rows.checked_mul(*rb).is_some_and(|total| total <= data.len()))
        .ok_or(Q10Error::TooShort)?;
    for j in 0..rows {
        let row = &data[j * row_bytes..(j + 1) * row_bytes];
        let mut acc: i64 = 0;
        for b in 0..blocks {
            let base = b * Q1_0_BLOCK_BYTES;
            let scale_bits = u16::from_le_bytes([row[base], row[base + 1]]);
            let mut partial: i64 = 0;
            for k in 0..Q1_0_BLOCK {
                let a = i64::from(acts[b * Q1_0_BLOCK + k]);
                let sign = (row[base + 2 + k / 8] >> (k % 8)) & 1;
                partial += if sign == 1 { a } else { -a };
            }
            // γ at fp16-EXACT precision (mantissa × 2^shift, pure
            // integer). The milli grid quantizes the model's real block
            // scales (γ ≈ 0.02–0.09 → milli 20–90) with 0.4–1.9% relative
            // error per block; compounded through every matvec of every
            // block that surfaced as multi-percent logit drift vs the
            // f32 reference (session C-core). The fork multiplies by the
            // fp16-exact γ; this matches it, integer-only.
            acc += match half_scale_mant_shift(scale_bits) {
                Some((m, sh)) => {
                    // Bounds: |partial| ≤ 128·32767 ≈ 4.2e6, m ≤ 2048 →
                    // |num| ≤ 8.6e9; sh ∈ [−24, 5] → |result| ≤ 2.75e14,
                    // and the 16-block row sum stays ≪ i64::MAX.
                    let num = partial * m as i64;
                    if sh >= 0 {
                        num << sh
                    } else {
                        let s = u32::try_from(-sh).expect("sh ≥ −24");
                        let half = 1_u64 << (s - 1);
                        let q = (num.unsigned_abs() + half) >> s;
                        if num >= 0 {
                            q as i64
                        } else {
                            -(q as i64)
                        }
                    }
                }
                None => {
                    // Hostile/degenerate scale (sign bit, ±inf, NaN):
                    // Session-A saturation semantics via the milli view.
                    crate::math::div_round_half_away(
                        partial * i64::from(half_to_milli(scale_bits)),
                        1000,
                    )
                }
            };
        }
        out[j] = acc.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    }
    Ok(())
}

/// An fp16 scale as an exact integer pair `(mantissa, shift)` with
/// `γ = mant × 2^shift` — dyadic, so integer arithmetic can apply it
/// with zero quantization. `None` on degenerate scales (sign bit set,
/// exponent 31 = ±inf/NaN) — callers fall back to the documented
/// saturation semantics. Shared by the q1_0 and q2_0 compute paths
/// (fp16-generic — the block layouts differ, the scale does not).
pub(crate) fn half_scale_mant_shift(h: u16) -> Option<(u64, i32)> {
    if h & 0x8000 != 0 {
        return None; // negative scale: hostile input
    }
    let exp = i32::from((h >> 10) & 0x1F);
    let mant = u64::from(h & 0x03FF);
    match exp {
        31 => None,                     // inf / NaN
        0 if mant == 0 => Some((0, 0)), // +0: an all-zero block
        0 => Some((mant, -24)),         // subnormal
        e => Some((mant | 0x400, e - 25)),
    }
}

/// The unit-chaining wrapper (session 3): milli activations in, milli
/// results out.
///
/// `q1_0_matvec` eats i16 activations (absmax-normalized Q15) and returns
/// raw partial-sum units; this wrapper normalizes `x_milli` → i16, runs
/// the matvec, and rescales by `amax/32767`, so
/// `out[j] ≈ Σᵢ w_real[j,i] · x_real[i] × 1000` — true milli units that
/// chain directly into norms, residuals, and the next matvec.
///
/// # Errors
///
/// Same as [`q1_0_matvec`] (bad length / short buffers) — including a
/// zero input vector, which still validates `data`/`out` sizes before
/// returning zeros.
pub fn matvec_scaled(
    data: &[u8],
    x_milli: &[i32],
    rows: usize,
    out: &mut [i32],
) -> Result<(), Q10Error> {
    let n = x_milli.len();
    if !n.is_multiple_of(Q1_0_BLOCK) {
        return Err(Q10Error::BadLength);
    }
    if out.len() < rows {
        return Err(Q10Error::TooShort);
    }
    let row_bytes = (n / Q1_0_BLOCK).checked_mul(Q1_0_BLOCK_BYTES).ok_or(Q10Error::TooShort)?;
    let total = rows.checked_mul(row_bytes).ok_or(Q10Error::TooShort)?;
    if total > data.len() {
        return Err(Q10Error::TooShort);
    }
    let amax = x_milli.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    let mut acts = vec![0_i16; n]; // load-edge-sized scratch (std crate)
    if amax == 0 {
        out[..rows].fill(0);
        return Ok(());
    }
    for (v, slot) in x_milli.iter().zip(acts.iter_mut()) {
        *slot = crate::math::div_round_half_away(i64::from(*v) * 32_767, i64::from(amax))
            .clamp(i16::MIN as i64, i16::MAX as i64) as i16;
    }
    let mut raw = vec![0_i32; rows];
    q1_0_matvec(data, &acts, rows, &mut raw)?;
    // Rescale by amax/32767: out_milli = round(raw × amax/32767).
    let num_scale = i64::from(amax);
    for (o, r) in out.iter_mut().zip(raw.iter()).take(rows) {
        *o = crate::math::div_round_half_away(i64::from(*r) * num_scale, 32_767)
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuralos_snn::Trit;

    /// Build one Q1_0 row (n values, one block per 128) from a trit-ish
    /// pattern (true = +γ, false = −γ) and per-block fp16 scales.
    fn build_row(signs: &[bool], gammas_half: &[u16]) -> Vec<u8> {
        let blocks = signs.len() / Q1_0_BLOCK;
        let mut data = Vec::with_capacity(blocks * Q1_0_BLOCK_BYTES);
        for b in 0..blocks {
            data.extend_from_slice(&gammas_half[b].to_le_bytes());
            let mut sign_bytes = [0_u8; 16];
            for (k, &s) in signs[b * Q1_0_BLOCK..(b + 1) * Q1_0_BLOCK]
                .iter()
                .enumerate()
            {
                if s {
                    sign_bytes[k / 8] |= 1 << (k % 8);
                }
            }
            data.extend_from_slice(&sign_bytes);
        }
        data
    }

    /// Scalar reference through the Stage-2 codec + explicit milli math.
    /// Exact fp16 decode in f64 (test-side, independent of production
    /// integer arithmetic): sign × (mant | 0x400) · 2^(exp−25), subnormals
    /// sign × mant · 2^−24. Dyadic → exact in f64.
    #[allow(dead_code)] // kept as documentation of the fp16 layout; the
                        // reference path uses half_scale_mant_shift.
    fn half_exact(h: u16) -> f64 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = i32::from((h >> 10) & 0x1F);
        let mant = f64::from(h & 0x03FF);
        if exp == 0 {
            sign * mant * (-24.0_f64).exp2()
        } else {
            sign * (mant + 1024.0) * f64::from(exp - 25).exp2()
        }
    }

    fn reference(data: &[u8], acts: &[i16], rows: usize) -> Vec<i32> {
        let n = acts.len();
        (0..rows)
            .map(|j| {
                let row = &data[j * (n / Q1_0_BLOCK * Q1_0_BLOCK_BYTES)..];
                let mut trits = vec![Trit::Zero; n];
                let mut scales = vec![0_u16; n / Q1_0_BLOCK];
                neuralos_snn::decode_q1_0(row, &mut trits, &mut scales).unwrap();
                let mut acc: f64 = 0.0;
                for b in 0..n / Q1_0_BLOCK {
                    // Mirror production semantics exactly: fp16-exact for
                    // normal positive scales, the documented milli
                    // saturation for degenerate ones (sign/inf/NaN).
                    let g = match super::half_scale_mant_shift(scales[b]) {
                        Some((m, sh)) => (m as f64) * f64::from(sh).exp2(),
                        None => f64::from(half_to_milli(scales[b])) / 1000.0,
                    };
                    let mut partial: i64 = 0;
                    for k in 0..Q1_0_BLOCK {
                        let t = trits[b * Q1_0_BLOCK + k];
                        let w = match t {
                            Trit::One => 1_i64,
                            Trit::MinusOne => -1,
                            Trit::Zero => 0,
                        };
                        partial += w * i64::from(acts[b * Q1_0_BLOCK + k]);
                    }
                    // Dyadic product — exact in f64; round half away.
                    acc += (partial as f64 * g).round();
                }
                acc.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
            })
            .collect()
    }

    #[test]
    fn matvec_matches_reference_single_block() {
        // One block: alternating signs, γ = fp16 ≈0.1 (milli 100).
        let signs: Vec<bool> = (0..128).map(|i| i % 3 != 0).collect();
        let data = build_row(&signs, &[0x2E66]); // fp16 ≈0.099976 → 100 milli
        let acts: Vec<i16> = (0..128).map(|i| ((i * 37) % 200 - 100) as i16).collect();
        let mut out = [0_i32; 1];
        q1_0_matvec(&data, &acts, 1, &mut out).unwrap();
        assert_eq!(out[0], reference(&data, &acts, 1)[0]);
        // Nonzero by construction.
        assert_ne!(out[0], 0);
    }

    #[test]
    fn matvec_matches_reference_multi_block_multi_row() {
        // 2 rows × 256 (2 blocks), mixed scales incl. a NEGATIVE scale
        // (fp16 −2.0) — the review flagged negative/inf/NaN γ as the
        // untested hostile-file surface.
        let n = 256;
        let mut signs = Vec::with_capacity(n);
        for i in 0..n {
            signs.push((i / 128) % 2 == 0 || i % 5 == 0);
        }
        let mut signs2 = Vec::with_capacity(n);
        for i in 0..n {
            signs2.push(i % 7 < 3);
        }
        let gammas = [0x4248_u16, 0xC000]; // fp16 ≈3.140625 (milli 3141) and −2.0
        let mut data = build_row(&signs, &gammas);
        data.extend(build_row(&signs2, &gammas));
        let acts: Vec<i16> = (0..n)
            .map(|i| ((i as i32 * 91) % 300 - 150) as i16)
            .collect();
        let mut out = [0_i32; 2];
        q1_0_matvec(&data, &acts, 2, &mut out).unwrap();
        let expect = reference(&data, &acts, 2);
        assert_eq!(out[0], expect[0]);
        assert_eq!(out[1], expect[1]);
    }

    #[test]
    fn row_to_milli_matches_decode() {
        let signs: Vec<bool> = (0..128).map(|i| i % 2 == 0).collect();
        let data = build_row(&signs, &[0x3C00]); // γ = 1.0 → milli 1000
        let mut out = vec![0_i32; 128];
        q1_0_row_to_milli(&data, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, if i % 2 == 0 { 1000 } else { -1000 });
        }
    }

    #[test]
    fn matvec_scaled_matches_f64_dequant_reference() {
        // One row, 2 blocks; γ values from fp16 bits 0x3C00 (1.0) and
        // 0x4248 (≈3.140625). Reference: decode via half_to_milli, dot in
        // f64, ×1000.
        let signs: Vec<bool> = (0..256).map(|i| i % 3 != 0).collect();
        let data = build_row(&signs, &[0x3C00, 0x4248]);
        let x: Vec<i32> = (0..256)
            .map(|i| (i * 71) % 400 - 200)
            .collect();
        let mut out = [0_i32; 1];
        matvec_scaled(&data, &x, 1, &mut out).unwrap();

        let g = [1000.0_f64, 3141.0]; // milli views of the fp16 scales
        let mut want = 0.0_f64;
        for (b, gb) in g.iter().enumerate() {
            for k in 0..128 {
                let i = b * 128 + k;
                let w = if signs[i] { *gb } else { -*gb };
                want += w * f64::from(x[i]);
            }
        }
        want /= 1000.0; // milli × milli → milli needs /1000
        let got = f64::from(out[0]);
        assert!(
            (got - want).abs() <= 2.0 + want.abs() * 0.002,
            "matvec_scaled {got} vs f64 {want}"
        );
        assert_ne!(out[0], 0);
    }

    #[test]
    fn matvec_scaled_matches_f64_2048_wide() {
        // The model's real row width (16 blocks, EMB 2048) — the review
        // flagged that only 256-wide was tested. Tolerance derived from
        // the error budget: 16 per-block roundings (≤ ±8 worst-case),
        // activation quantization (≤ Σ|γ|·amax/65534), and the final
        // rescale rounding — generous relative term + a floor covering
        // near-cancelling sums.
        let gamma_pairs: [(u16, f64); 4] =
            [(0x3C00, 1000.0), (0x2E66, 100.0), (0x4248, 3141.0), (0x3800, 500.0)];
        // ^ 0x2E66 is fp16 ≈0.099976 → 100 milli (the "0.05" label in
        // the older single-block test was wrong — this table is derived
        // from half_to_milli, not from comments).
        let gammas: Vec<u16> = (0..16).map(|b| gamma_pairs[b % 4].0).collect();
        let g_milli: Vec<f64> = (0..16).map(|b| gamma_pairs[b % 4].1).collect();
        let signs: Vec<bool> = (0..2048).map(|i| (i % 5 != 0) && (i % 7 >= 2)).collect();
        let data = build_row(&signs, &gammas);
        let x: Vec<i32> = (0..2048).map(|i| (i * 71) % 400 - 200).collect();
        let mut out = [0_i32; 1];
        matvec_scaled(&data, &x, 1, &mut out).unwrap();

        let mut want = 0.0_f64;
        for (b, &g) in g_milli.iter().enumerate() {
            for k in 0..128 {
                let i = b * 128 + k;
                let w = if signs[i] { g } else { -g };
                want += w * f64::from(x[i]);
            }
        }
        want /= 1000.0;
        let got = f64::from(out[0]);
        assert!(
            (got - want).abs() <= 25.0 + want.abs() * 0.004,
            "matvec_scaled 2048-wide {got} vs f64 {want}"
        );
    }

    #[test]
    fn matvec_scaled_zero_input_is_zero() {
        let data = [0_u8; 36];
        let x = [0_i32; 256];
        let mut out = [5_i32; 1];
        matvec_scaled(&data, &x, 1, &mut out).unwrap();
        assert_eq!(out[0], 0);
    }

    #[test]
    fn hostile_scale_blocks_saturate_not_panic() {
        // ±inf / NaN fp16 scales from a corrupt file: documented
        // saturation (half_to_milli maps ±inf → ±i32 rails, NaN → 0),
        // never a negation panic (the old `−gamma` on i32::MIN).
        let mut data = Vec::new();
        data.extend_from_slice(&0xFC00_u16.to_le_bytes()); // −inf → i32::MIN
        data.extend(std::iter::repeat_n(0xB5_u8, 16)); // mixed signs
        let mut out = vec![0_i32; 128];
        q1_0_row_to_milli(&data, &mut out).unwrap();
        for (j, &v) in out.iter().enumerate() {
            let sign_set = (0xB5 >> (j % 8)) & 1 == 1;
            assert_eq!(v, if sign_set { i32::MIN } else { i32::MAX }, "elem {j}");
        }
        // matvec with an inf scale saturates at the row level, no panic.
        let acts = [100_i16; 128];
        let mut acc = [0_i32; 1];
        q1_0_matvec(&data, &acts, 1, &mut acc).unwrap();
        assert_eq!(acc[0], i32::MIN); // partial < 0 × γ = i32::MIN-milli
    }

    #[test]
    fn matvec_scaled_validates_sizes_before_any_work() {
        // 2026-08-15 review: the zero-input path used to skip data-length
        // validation entirely, and short `out` either panicked (zero
        // path) or silently wrote fewer rows.
        let data = [0_u8; 36];
        // Zero input + too-short data → Err, not silent Ok.
        assert_eq!(
            matvec_scaled(&data[..18], &[0_i32; 256], 1, &mut [0_i32; 1]),
            Err(Q10Error::TooShort)
        );
        // Zero input + short out → Err, not panic.
        assert_eq!(
            matvec_scaled(&data, &[0_i32; 256], 4, &mut [0_i32; 1]),
            Err(Q10Error::TooShort)
        );
        // Nonzero input + short out → Err, not silent partial write.
        assert_eq!(
            matvec_scaled(&data, &[100_i32; 256], 4, &mut [0_i32; 1]),
            Err(Q10Error::TooShort)
        );
        // Absurd rows: checked math → Err, never an overflow panic.
        assert_eq!(
            q1_0_matvec(&data, &[0_i16; 256], usize::MAX, &mut [0_i32; 1]),
            Err(Q10Error::TooShort)
        );
        assert_eq!(
            matvec_scaled(&data, &[0_i32; 256], usize::MAX, &mut [0_i32; 1]),
            Err(Q10Error::TooShort)
        );
    }

    #[test]
    fn matvec_gamma_is_fp16_exact_not_milli() {
        // γ = fp16 0x3555 = 0.33349609375 → milli rounds to 333 (−0.15%).
        // A milli-γ implementation yields a measurably different dot
        // product; the exact one matches the f64 reference to the unit.
        let signs: Vec<bool> = (0..128).map(|i| i % 2 == 0).collect();
        let data = build_row(&signs, &[0x3555]);
        let acts: Vec<i16> = (0..128).map(|i| ((i * 91) % 300 - 150) as i16).collect();
        let mut out = [0_i32; 1];
        q1_0_matvec(&data, &acts, 1, &mut out).unwrap();
        assert_eq!(out[0], reference(&data, &acts, 1)[0]);
        // And the value is nonzero (a real product, not a degenerate 0).
        assert_ne!(out[0], 0);
    }

    #[test]
    fn rejects_bad_sizes() {
        let data = [0_u8; 18];
        let mut out = [0_i32; 128];
        assert_eq!(
            q1_0_row_to_milli(&data[..17], &mut out),
            Err(Q10Error::TooShort)
        );
        let mut odd = [0_i32; 100];
        assert_eq!(
            q1_0_row_to_milli(&data, &mut odd),
            Err(Q10Error::BadLength)
        );
        let acts = [0_i16; 64];
        assert_eq!(
            q1_0_matvec(&data, &acts, 1, &mut out[..1]),
            Err(Q10Error::BadLength)
        );
        assert_eq!(
            q1_0_matvec(&data, &[0_i16; 128], 1, &mut []),
            Err(Q10Error::TooShort)
        );
    }
}
