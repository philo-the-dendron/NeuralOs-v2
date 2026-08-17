//! Q2_0 compute on top of the container — the 2-bit ternary tier
//! (Stage 4, session D slice 1).
//!
//! Sibling of [`crate::q1_0`] at the same seam, over the re-pinned
//! layout (session D): one block per **128 weights, 34 bytes** — LE
//! fp16 scale (`d = max|w|`, TWN-style) + 32 bytes of LSB-first 2-bit
//! codes (`00`=−1 `01`=0 `10`=+1); code `11` is unreachable from the
//! reference quantizer and is a loud error here, never `+2·d`.
//!
//! # Numeric contract
//!
//! Same shape as the Q1_0 module: activations `i16` (absmax-normalized
//! Q15), per-block partial sums in `i64`, the fp16 scale applied at
//! **exact** precision (integer mantissa × 2^shift — the C-core
//! doctrine), saturating `i32` out. Bounds: Q2_0 partials are HALF the
//! Q1_0 reach (`|partial| ≤ 128·32768` still — 128 elements per block
//! on both formats; only the code width differs), `m ≤ 2048` →
//! `|num| ≤ 8.6e9`; `sh ∈ [−24, 5]` → `|block result| ≤ 2.75e14`. The
//! cross-block row sum stays ≪ `i64::MAX` at real widths (40 blocks ×
//! 2.75e14 ≈ 1.1e16 at 2560-wide — 0.1% of range). Q2_0 scales are
//! `max|w|` (larger than Q1_0's `mean|w|` for the same weights), but
//! the bound is driven by the largest FINITE fp16 (65504 → mant 2047,
//! shift 5 — exactly the worst case above), so the commentary holds at
//! the seam by construction, not by luck.

use neuralos_snn::half_to_milli;

/// Q2_0 block size (weights per block — one fp16 scale + 32 code
/// bytes). Single-sourced from the published codec
/// ([`neuralos_snn::bridge`]); re-pinned session D (was wrongly 64).
pub use neuralos_snn::bridge::Q2_0_BLOCK;
/// Q2_0 block byte size: 2 scale bytes + 32 code bytes — derived from
/// the codec's own length function (`encoded_len(128) = 34`), not a
/// second hand-written constant.
pub const Q2_0_BLOCK_BYTES: usize = neuralos_snn::bridge::q2_0_encoded_len(Q2_0_BLOCK);

const _: () = assert!(Q2_0_BLOCK_BYTES == 34, "q2_0 layout must stay 34 B/block");

/// Errors from the Q2_0 compute path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q20Error {
    /// Activation length not a multiple of the 128-weight block.
    BadLength,
    /// A buffer is shorter than the layout requires.
    TooShort,
    /// A 2-bit lane holds code 3 — unreachable from the reference
    /// quantizer; a loud error, never a silent `+2·d` clamp.
    UnsupportedCode,
}

impl core::fmt::Display for Q20Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadLength => write!(f, "activation length not a multiple of 128"),
            Self::TooShort => write!(f, "buffer shorter than the layout requires"),
            Self::UnsupportedCode => write!(f, "2-bit code 3 (unreachable from the reference quantizer)"),
        }
    }
}

impl std::error::Error for Q20Error {}

/// One Q2_0 row (or any contiguous run of blocks) → milli-domain
/// values: element `i` becomes `−d`, `0`, or `+d` (milli view) of its
/// block — the embedding-materialization path for q2_0 files.
///
/// Non-finite fp16 scales follow `half_to_milli`'s documented
/// saturation (±inf → ±`i32::MAX` after sign application — the
/// negation happens in `i64`, so an inf-scale block saturates per-sign
/// instead of panicking on `-(i32::MIN)`; NaN d → 0).
///
/// `n` values need `n/128 × 34` bytes of `data` and `n` slots of
/// `out`. All errors (bad length, short buffers, code 3) leave `out`
/// untouched — the code pre-scan runs before the first write.
///
/// # Errors
///
/// [`Q20Error::BadLength`] / [`Q20Error::TooShort`] /
/// [`Q20Error::UnsupportedCode`] on bad sizes or an impossible code.
pub fn q2_0_row_to_milli(data: &[u8], out: &mut [i32]) -> Result<(), Q20Error> {
    let n = out.len();
    if !n.is_multiple_of(Q2_0_BLOCK) {
        return Err(Q20Error::BadLength);
    }
    let blocks = n / Q2_0_BLOCK;
    if data.len() < blocks * Q2_0_BLOCK_BYTES {
        return Err(Q20Error::TooShort);
    }
    scan_for_code_three(data, blocks)?;
    for b in 0..blocks {
        let base = b * Q2_0_BLOCK_BYTES;
        let d = i64::from(half_to_milli(u16::from_le_bytes([data[base], data[base + 1]])));
        let neg = (-d).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        let pos = d.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        for j in 0..Q2_0_BLOCK {
            let code = (data[base + 2 + j / 4] >> (2 * (j % 4))) & 0b11;
            out[b * Q2_0_BLOCK + j] = match code {
                0 => neg,
                1 => 0,
                2 => pos,
                _ => return Err(Q20Error::UnsupportedCode), // scan makes this unreachable
            };
        }
    }
    Ok(())
}

/// Pre-scan `blocks` blocks for any code-3 lane, so every error path
/// fires before the first output write. A byte holds a code-3 lane iff
/// some adjacent bit pair is `11`: `byte & (byte >> 1) & 0x55 != 0`.
/// Stride-aware — the fp16 scale bytes are never scanned as codes.
fn scan_for_code_three(data: &[u8], blocks: usize) -> Result<(), Q20Error> {
    for b in 0..blocks {
        let base = b * Q2_0_BLOCK_BYTES;
        for &byte in &data[base + 2..base + Q2_0_BLOCK_BYTES] {
            if byte & (byte >> 1) & 0x55 != 0 {
                return Err(Q20Error::UnsupportedCode);
            }
        }
    }
    Ok(())
}

/// The Q2_0 matvec: `out[j] = Σᵢ c_{j,b(i)} · d_{j,b(i)} · a[i]` with
/// `c ∈ {−1, 0, +1}` — per-block scaled, scale at fp16-EXACT precision
/// (integer mantissa × 2^shift, same as the C-core q1_0 path).
///
/// `data` holds `rows` consecutive Q2_0 rows, each `n = acts.len()`
/// weights wide (`n % 128 == 0`, `n/128 × 34` bytes per row — exactly
/// how a `[n, rows]` GGUF tensor lays out row-major); `acts` are i16
/// activations; `out` receives `rows` i32 results, saturating at the
/// row level. Hostile scales (sign bit, ±inf, NaN) fall back to the
/// documented milli-view saturation semantics — same as Q1_0.
///
/// # Errors
///
/// [`Q20Error::BadLength`] / [`Q20Error::TooShort`] on bad sizes —
/// including `out.len() < rows` and a `rows` whose byte footprint
/// would overflow `usize`; [`Q20Error::UnsupportedCode`] from the
/// pre-scan (before any output write).
pub fn q2_0_matvec(
    data: &[u8],
    acts: &[i16],
    rows: usize,
    out: &mut [i32],
) -> Result<(), Q20Error> {
    let n = acts.len();
    if !n.is_multiple_of(Q2_0_BLOCK) {
        return Err(Q20Error::BadLength);
    }
    if out.len() < rows {
        return Err(Q20Error::TooShort);
    }
    let blocks = n / Q2_0_BLOCK;
    let row_bytes = blocks
        .checked_mul(Q2_0_BLOCK_BYTES)
        .filter(|rb| rows.checked_mul(*rb).is_some_and(|total| total <= data.len()))
        .ok_or(Q20Error::TooShort)?;
    scan_for_code_three(&data[..rows * row_bytes], rows * blocks)?;
    for j in 0..rows {
        let row = &data[j * row_bytes..(j + 1) * row_bytes];
        let mut acc: i64 = 0;
        for b in 0..blocks {
            let base = b * Q2_0_BLOCK_BYTES;
            let scale_bits = u16::from_le_bytes([row[base], row[base + 1]]);
            let mut partial: i64 = 0;
            for k in 0..Q2_0_BLOCK {
                let code = (row[base + 2 + k / 4] >> (2 * (k % 4))) & 0b11;
                let a = i64::from(acts[b * Q2_0_BLOCK + k]);
                partial += match code {
                    0 => -a,
                    1 => 0,
                    2 => a,
                    _ => unreachable!("code 3 excluded by the pre-scan"),
                };
            }
            // d at fp16-EXACT precision — same helper and rationale as
            // the q1_0 path (session C-core): the milli grid quantizes
            // real scales; the fork multiplies by the exact fp16 value.
            acc += match crate::q1_0::half_scale_mant_shift(scale_bits) {
                Some((m, sh)) => {
                    // Bounds: |partial| ≤ 128·32767 ≈ 4.2e6, m ≤ 2048 →
                    // |num| ≤ 8.6e9; sh ∈ [−24, 5] → |result| ≤ 2.75e14,
                    // and the 40-block (2560-wide) row sum stays ≪ i64::MAX.
                    let num = partial * i64::try_from(m).expect("m ≤ 2048");
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

/// The unit-chaining wrapper (same contract as
/// [`crate::q1_0::matvec_scaled`]): milli activations in, milli
/// results out — absmax-normalize to i16, `q2_0_matvec`, rescale by
/// `amax/32767`, so `out[j] ≈ Σᵢ w_real[j,i] · x_real[i] × 1000`.
///
/// # Errors
///
/// Same as [`q2_0_matvec`] (bad length / short buffers / code 3) —
/// including a zero input vector, which still validates `data`/`out`
/// sizes before returning zeros.
pub fn matvec_scaled(
    data: &[u8],
    x_milli: &[i32],
    rows: usize,
    out: &mut [i32],
) -> Result<(), Q20Error> {
    let n = x_milli.len();
    if !n.is_multiple_of(Q2_0_BLOCK) {
        return Err(Q20Error::BadLength);
    }
    if out.len() < rows {
        return Err(Q20Error::TooShort);
    }
    let row_bytes = (n / Q2_0_BLOCK).checked_mul(Q2_0_BLOCK_BYTES).ok_or(Q20Error::TooShort)?;
    let total = rows.checked_mul(row_bytes).ok_or(Q20Error::TooShort)?;
    if total > data.len() {
        return Err(Q20Error::TooShort);
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
    q2_0_matvec(data, &acts, rows, &mut raw)?;
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

    /// Build one Q2_0 row (n values, one block per 128) from a code
    /// pattern ({0,1,2} = {−1,0,+1}) and per-block fp16 scales.
    fn build_row(codes: &[u8], scales_half: &[u16]) -> Vec<u8> {
        let blocks = codes.len() / Q2_0_BLOCK;
        let mut data = Vec::with_capacity(blocks * Q2_0_BLOCK_BYTES);
        for b in 0..blocks {
            data.extend_from_slice(&scales_half[b].to_le_bytes());
            let mut code_bytes = [0_u8; 32];
            for (k, &c) in codes[b * Q2_0_BLOCK..(b + 1) * Q2_0_BLOCK]
                .iter()
                .enumerate()
            {
                assert!(c <= 2, "test builder emits no code 3");
                code_bytes[k / 4] |= c << (2 * (k % 4));
            }
            data.extend_from_slice(&code_bytes);
        }
        data
    }

    /// Independent reference through the PUBLISHED Stage-2 codec
    /// (`decode_q2_0` — a different code path from the matvec's inline
    /// lane extraction) + exact fp16 scales in f64 + per-block
    /// round-half-away, mirroring production semantics exactly for
    /// normal positive scales and the documented milli saturation for
    /// degenerate ones.
    fn reference(data: &[u8], acts: &[i16], rows: usize) -> Vec<i32> {
        let n = acts.len();
        (0..rows)
            .map(|j| {
                let row = &data[j * (n / Q2_0_BLOCK * Q2_0_BLOCK_BYTES)..];
                let mut trits = vec![Trit::Zero; n];
                let mut scales = vec![0_u16; n / Q2_0_BLOCK];
                neuralos_snn::decode_q2_0(row, &mut trits, &mut scales).unwrap();
                let mut acc: f64 = 0.0;
                for b in 0..n / Q2_0_BLOCK {
                    let d = match crate::q1_0::half_scale_mant_shift(scales[b]) {
                        Some((m, sh)) => f64::from(u32::try_from(m).unwrap()) * f64::from(sh).exp2(),
                        None => f64::from(half_to_milli(scales[b])) / 1000.0,
                    };
                    let mut partial: i64 = 0;
                    for k in 0..Q2_0_BLOCK {
                        let w = match trits[b * Q2_0_BLOCK + k] {
                            Trit::One => 1_i64,
                            Trit::MinusOne => -1,
                            Trit::Zero => 0,
                        };
                        partial += w * i64::from(acts[b * Q2_0_BLOCK + k]);
                    }
                    acc += (partial as f64 * d).round();
                }
                acc.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
            })
            .collect()
    }

    #[test]
    fn matvec_matches_reference_single_block() {
        // One block, period-3 codes (all three values exercised), d =
        // fp16 ≈0.1 (milli 100).
        let codes: Vec<u8> = (0..128).map(|i| (i % 3) as u8).collect();
        let data = build_row(&codes, &[0x2E66]);
        let acts: Vec<i16> = (0..128).map(|i| ((i * 37) % 200 - 100) as i16).collect();
        let mut out = [0_i32; 1];
        q2_0_matvec(&data, &acts, 1, &mut out).unwrap();
        assert_eq!(out[0], reference(&data, &acts, 1)[0]);
        assert_ne!(out[0], 0);
    }

    #[test]
    fn matvec_matches_reference_multi_block_multi_row() {
        // 2 rows × 256 (2 blocks), mixed scales incl. a NEGATIVE scale
        // (fp16 −2.0 — hostile-file surface, milli fallback semantics).
        let n = 256;
        let codes: Vec<u8> = (0..n).map(|i| ((i % 5) % 3) as u8).collect();
        let codes2: Vec<u8> = (0..n).map(|i| ((i % 7) % 3) as u8).collect();
        let scales = [0x4248_u16, 0xC000]; // fp16 ≈3.140625 and −2.0
        let mut data = build_row(&codes, &scales);
        data.extend(build_row(&codes2, &scales));
        let acts: Vec<i16> = (0..n)
            .map(|i| ((i * 91) % 300 - 150) as i16)
            .collect();
        let mut out = [0_i32; 2];
        q2_0_matvec(&data, &acts, 2, &mut out).unwrap();
        let expect = reference(&data, &acts, 2);
        assert_eq!(out[0], expect[0]);
        assert_eq!(out[1], expect[1]);
    }

    #[test]
    fn matvec_gamma_is_fp16_exact_not_milli() {
        // d = fp16 0x3555 = 0.33349609375 → milli rounds to 333 (−0.15%).
        // A milli-d implementation yields a measurably different dot
        // product; the exact one matches the reference to the unit.
        let codes: Vec<u8> = (0..128).map(|i| (i % 2) as u8).collect();
        let data = build_row(&codes, &[0x3555]);
        let acts: Vec<i16> = (0..128).map(|i| ((i * 91) % 300 - 150) as i16).collect();
        let mut out = [0_i32; 1];
        q2_0_matvec(&data, &acts, 1, &mut out).unwrap();
        assert_eq!(out[0], reference(&data, &acts, 1)[0]);
        assert_ne!(out[0], 0);
    }

    #[test]
    fn matvec_scaled_matches_f64_reference_2560_wide() {
        // The real Q2_0 row width (40 blocks, 2560 weights), f64
        // reference in REAL units from the exact fp16 scales — never
        // the milli grid. Tolerance covers activation quantization
        // (Σ|w|·amax/65534) + 40 per-block roundings + the final
        // rescale rounding: relative term + floor for near-cancelling
        // sums.
        let gamma_pairs: [(u16, f64); 4] =
            [(0x3C00, 1.0), (0x2E66, 0.0999755859375), (0x4248, 3.140625), (0x3800, 0.5)];
        let scales: Vec<u16> = (0..20).map(|b| gamma_pairs[b % 4].0).collect();
        let d_exact: Vec<f64> = (0..20).map(|b| gamma_pairs[b % 4].1).collect();
        let codes: Vec<u8> = (0..2560).map(|i| ((i % 7) % 3) as u8).collect();
        let data = build_row(&codes, &scales);
        let x: Vec<i32> = (0..2560).map(|i| (i * 71) % 400 - 200).collect();
        let mut out = [0_i32; 1];
        matvec_scaled(&data, &x, 1, &mut out).unwrap();

        // Tolerance covers activation quantization (Σ|w|·amax/65534) +
        // 20 per-block roundings + the final rescale rounding: relative
        // term + floor for near-cancelling sums.
        let mut want = 0.0_f64;
        for (b, &d) in d_exact.iter().enumerate() {
            for k in 0..Q2_0_BLOCK {
                let i = b * Q2_0_BLOCK + k;
                let w = match codes[i] {
                    0 => -1.0,
                    1 => 0.0,
                    _ => 1.0,
                };
                want += w * d * f64::from(x[i]);
            }
        }
        let got = f64::from(out[0]);
        assert!(
            (got - want).abs() <= 30.0 + want.abs() * 0.004,
            "matvec_scaled q2_0 2560-wide {got} vs f64 {want}"
        );
        assert_ne!(out[0], 0);
    }

    #[test]
    fn code_three_is_loud_and_leaves_out_untouched() {
        // Head lane AND tail lane of the code span; out untouched both
        // times (pre-scan runs before any write).
        let codes: Vec<u8> = vec![1; 128];
        let mut data = build_row(&codes, &[0x3C00]);
        let acts = [100_i16; 128];
        let mut out = [7_i32; 1];
        data[2] = 0x03; // element 0 lane = 3
        assert_eq!(
            q2_0_matvec(&data, &acts, 1, &mut out),
            Err(Q20Error::UnsupportedCode)
        );
        assert_eq!(out[0], 7, "head-lane code 3 must not write out");
        let mut data2 = build_row(&codes, &[0x3C00]);
        data2[33] = 0xC0; // element 127 lane = 3
        assert_eq!(
            q2_0_matvec(&data2, &acts, 1, &mut out),
            Err(Q20Error::UnsupportedCode)
        );
        assert_eq!(out[0], 7, "tail-lane code 3 must not write out");
        // row_to_milli: same contract.
        let mut row = [9_i32; 128];
        assert_eq!(
            q2_0_row_to_milli(&data, &mut row),
            Err(Q20Error::UnsupportedCode)
        );
        assert_eq!(row[0], 9);
    }

    #[test]
    fn hostile_scale_blocks_saturate_not_panic() {
        // ±inf / NaN fp16 scales from a corrupt file: documented
        // saturation (half_to_milli maps ±inf → ±i32 rails, NaN → 0),
        // never a negation panic. Codes here alternate −1 (even) /
        // 0 (odd): code 0 with d = −inf-milli (i32::MIN) gives
        // neg = −(i32::MIN) clamped → i32::MAX; code 1 gives exact 0.
        let codes: Vec<u8> = (0..128).map(|i| (i % 2) as u8).collect();
        let data = build_row(&codes, &[0xFC00]); // fp16 −inf
        let mut out = vec![0_i32; 128];
        q2_0_row_to_milli(&data, &mut out).unwrap();
        for (j, &v) in out.iter().enumerate() {
            let want = if j % 2 == 0 { i32::MAX } else { 0 };
            assert_eq!(v, want, "elem {j}");
        }
        // matvec with a +inf scale saturates at the row level, no
        // panic: partial = −64·100 (even elements are −1), × the
        // +inf milli rail / 1000 → deeply negative → i32::MIN.
        let data2 = build_row(&codes, &[0x7C00]); // fp16 +inf
        let acts = [100_i16; 128];
        let mut acc = [0_i32; 1];
        q2_0_matvec(&data2, &acts, 1, &mut acc).unwrap();
        assert_eq!(acc[0], i32::MIN);
    }

    #[test]
    fn row_to_milli_matches_decode() {
        // Period-3 codes, d = fp16 1.0 → milli ±1000/0 exactly.
        let codes: Vec<u8> = (0..128).map(|i| (i % 3) as u8).collect();
        let data = build_row(&codes, &[0x3C00]);
        let mut out = vec![0_i32; 128];
        q2_0_row_to_milli(&data, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            let want = match i % 3 {
                0 => -1000,
                1 => 0,
                _ => 1000,
            };
            assert_eq!(v, want, "elem {i}");
        }
    }

    #[test]
    fn lane_and_byte_order_golden() {
        // Period-5 codes across 256 elements (2 blocks, distinct
        // scales): 5 doesn't divide 4, so every byte differs — byte
        // AND lane order pinned; the block boundary (element 128)
        // breaks any block-order blindness.
        let codes: Vec<u8> = (0..256)
            .map(|i| if i % 5 == 0 { 2 } else if i % 5 == 1 { 0 } else { 1 })
            .collect();
        let data = build_row(&codes, &[0x3C00, 0x4248]);
        let acts: Vec<i16> = (0..256).map(|i| ((i * 13) % 100 - 50) as i16).collect();
        let mut out = [0_i32; 1];
        q2_0_matvec(&data, &acts, 1, &mut out).unwrap();
        assert_eq!(out[0], reference(&data, &acts, 1)[0]);
        // And the element-level path agrees with the code census —
        // derived from the constructed pattern, not hand-counted.
        let mut row = vec![0_i32; 256];
        q2_0_row_to_milli(&data, &mut row).unwrap();
        let zeros = row.iter().filter(|&&v| v == 0).count();
        assert_eq!(zeros, codes.iter().filter(|&&c| c == 1).count());
    }

    #[test]
    fn matvec_scaled_zero_input_is_zero_and_validates_first() {
        let data = [0_u8; 68]; // 2 blocks (256-wide)
        // Zero input + too-short data (1 block for a 256-wide vector)
        // → Err, not silent Ok.
        assert_eq!(
            matvec_scaled(&data[..34], &[0_i32; 256], 1, &mut [0_i32; 1]),
            Err(Q20Error::TooShort)
        );
        // Zero input + short out → Err, not panic.
        assert_eq!(
            matvec_scaled(&data, &[0_i32; 128], 4, &mut [0_i32; 1]),
            Err(Q20Error::TooShort)
        );
        // Happy zero path.
        let mut out = [5_i32; 1];
        matvec_scaled(&data, &[0_i32; 128], 1, &mut out).unwrap();
        assert_eq!(out[0], 0);
        // Absurd rows: checked math → Err, never an overflow panic.
        assert_eq!(
            q2_0_matvec(&data, &[0_i16; 128], usize::MAX, &mut [0_i32; 1]),
            Err(Q20Error::TooShort)
        );
        assert_eq!(
            matvec_scaled(&data, &[0_i32; 128], usize::MAX, &mut [0_i32; 1]),
            Err(Q20Error::TooShort)
        );
    }

    #[test]
    fn rejects_bad_sizes() {
        let data = [0_u8; 34];
        let mut out = [0_i32; 128];
        assert_eq!(
            q2_0_row_to_milli(&data[..33], &mut out),
            Err(Q20Error::TooShort)
        );
        let mut odd = [0_i32; 100];
        assert_eq!(
            q2_0_row_to_milli(&data, &mut odd),
            Err(Q20Error::BadLength)
        );
        let acts = [0_i16; 64];
        assert_eq!(
            q2_0_matvec(&data, &acts, 1, &mut out[..1]),
            Err(Q20Error::BadLength)
        );
        assert_eq!(
            q2_0_matvec(&data, &[0_i16; 128], 1, &mut []),
            Err(Q20Error::TooShort)
        );
    }
}
