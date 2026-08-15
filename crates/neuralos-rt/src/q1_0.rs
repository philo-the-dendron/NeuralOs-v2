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
//! ([`neuralos_snn::half_to_milli`): `±γ` per element, per-block partials
//! accumulated in `i64`, each block's contribution
//! `round(partial × γ_milli / 1000)`, final saturation to `i32`
//! documented. Bound: `|partial| ≤ 128·32767 ≈ 4.2e6`, `γ_milli ≤ 6.55e10`
//! → product ≤ 2.8e17 « i64::MAX per block; the cross-block sum can exceed
//! `i32` only for absurd scales (γ > ~50 in real units) and saturates.

use neuralos_snn::half_to_milli;
/// Q1_0 block size (weights per block — one fp16 scale + 16 sign bytes).
pub const Q1_0_BLOCK: usize = 128;
/// Q1_0 block byte size: 2 scale bytes + 16 sign bytes.
pub const Q1_0_BLOCK_BYTES: usize = 18;

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
        let gamma = half_to_milli(u16::from_le_bytes([data[base], data[base + 1]]));
        for j in 0..Q1_0_BLOCK {
            let sign = (data[base + 2 + j / 8] >> (j % 8)) & 1;
            out[b * Q1_0_BLOCK + j] = if sign == 1 { gamma } else { -gamma };
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
/// [`Q10Error::BadLength`] / [`Q10Error::TooShort`] on bad sizes.
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
    let blocks = n / Q1_0_BLOCK;
    let row_bytes = blocks * Q1_0_BLOCK_BYTES;
    if data.len() < rows * row_bytes || out.len() < rows {
        return Err(Q10Error::TooShort);
    }
    for j in 0..rows {
        let row = &data[j * row_bytes..(j + 1) * row_bytes];
        let mut acc: i64 = 0;
        for b in 0..blocks {
            let base = b * Q1_0_BLOCK_BYTES;
            let gamma_milli = i64::from(half_to_milli(u16::from_le_bytes([
                row[base],
                row[base + 1],
            ])));
            let mut partial: i64 = 0;
            for k in 0..Q1_0_BLOCK {
                let a = i64::from(acts[b * Q1_0_BLOCK + k]);
                let sign = (row[base + 2 + k / 8] >> (k % 8)) & 1;
                partial += if sign == 1 { a } else { -a };
            }
            // Round-half-away-from-zero on the milli→unit conversion.
            let contrib = if partial >= 0 {
                (partial * gamma_milli + 500) / 1000
            } else {
                -((-partial * gamma_milli + 500) / 1000)
            };
            acc += contrib;
        }
        out[j] = acc.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    }
    Ok(())
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
/// Same as [`q1_0_matvec`] (bad length / short buffers).
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
    let amax = x_milli.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    let mut acts = vec![0_i16; n]; // load-edge-sized scratch (std crate)
    if amax == 0 {
        out[..rows].fill(0);
        return Ok(());
    }
    for (v, slot) in x_milli.iter().zip(acts.iter_mut()) {
        let num = i64::from(*v) * 32_767;
        let den = i64::from(amax);
        let q = if num >= 0 {
            (num + den / 2) / den
        } else {
            -((-num + den / 2) / den)
        };
        *slot = q.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
    }
    let mut raw = vec![0_i32; rows];
    q1_0_matvec(data, &acts, rows, &mut raw)?;
    // Rescale by amax/32767: out_milli = round(raw × amax/32767).
    let num_scale = i64::from(amax);
    for (o, r) in out.iter_mut().zip(raw.iter()).take(rows) {
        let v = i64::from(*r);
        let mag = (v.unsigned_abs() as i64 * num_scale + 16_383) / 32_767;
        let signed = if v < 0 { -mag } else { mag };
        *o = signed.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
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
    fn reference(data: &[u8], acts: &[i16], rows: usize) -> Vec<i32> {
        let n = acts.len();
        (0..rows)
            .map(|j| {
                let row = &data[j * (n / Q1_0_BLOCK * Q1_0_BLOCK_BYTES)..];
                let mut trits = vec![Trit::Zero; n];
                let mut scales = vec![0_u16; n / Q1_0_BLOCK];
                neuralos_snn::decode_q1_0(row, &mut trits, &mut scales).unwrap();
                let mut acc: i64 = 0;
                for b in 0..n / Q1_0_BLOCK {
                    let g = i64::from(half_to_milli(scales[b]));
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
                    let c = if partial >= 0 {
                        (partial * g + 500) / 1000
                    } else {
                        -((-partial * g + 500) / 1000)
                    };
                    acc += c;
                }
                acc.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
            })
            .collect()
    }

    #[test]
    fn matvec_matches_reference_single_block() {
        // One block: alternating signs, γ = fp16 0.05 (milli 50).
        let signs: Vec<bool> = (0..128).map(|i| i % 3 != 0).collect();
        let data = build_row(&signs, &[0x2E66]); // fp16 0.05
        let acts: Vec<i16> = (0..128).map(|i| ((i * 37) % 200 - 100) as i16).collect();
        let mut out = [0_i32; 1];
        q1_0_matvec(&data, &acts, 1, &mut out).unwrap();
        assert_eq!(out[0], reference(&data, &acts, 1)[0]);
        // Nonzero by construction.
        assert_ne!(out[0], 0);
    }

    #[test]
    fn matvec_matches_reference_multi_block_multi_row() {
        // 2 rows × 256 (2 blocks), mixed scales incl. negative-sign scales.
        let n = 256;
        let mut signs = Vec::with_capacity(n);
        for i in 0..n {
            signs.push((i / 128) % 2 == 0 || i % 5 == 0);
        }
        let mut signs2 = Vec::with_capacity(n);
        for i in 0..n {
            signs2.push(i % 7 < 3);
        }
        let gammas = [0x4248_u16, 0x3C00]; // fp16 ≈3.140625 (milli 3141) and 1.0
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
    fn matvec_scaled_zero_input_is_zero() {
        let data = [0_u8; 36];
        let x = [0_i32; 256];
        let mut out = [5_i32; 1];
        matvec_scaled(&data, &x, 1, &mut out).unwrap();
        assert_eq!(out[0], 0);
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
