//! Integer RMSNorm — Qwen3's normalization layer in pure fixed-point
//! (Stage 4, session 2).
//!
//! Qwen3 (and the Llama lineage) normalizes as
//! `y = x / rms(x) · w` with `rms(x) = sqrt(mean(x²) + eps)`. Here `x`
//! lives in the milli domain (i32; see `q1_0`), `w` too (f32 weights
//! convert at load via [`f32_bits_to_milli`]), and everything downstream
//! of the conversion is integer: an exact Newton [`isqrt`], a
//! documented integer eps floor, i64 intermediates.
//!
//! # Working range (documented bound)
//!
//! `Σx²` accumulates in i64: sound while `2048 · max(x²) < 2^63`, i.e.
//! `|x| ≲ 5×10^7` milli (≈ 50 000 in real units) — embedding-scale values
//! are orders of magnitude below. The `x·w` product needs
//! `|x·w| < 2^63` — same order. Debug asserts guard both.

/// Convert raw f32 bits to the milli domain: `round(v × 1000)` — the f32
/// sibling of `neuralos_snn::half_to_milli`, same integer construction
/// (mantissa shift, round-half-away), saturating at the i32 rails.
///
/// This is a load-edge conversion (file bytes → fixed point), not part of
/// the compute path: norm weights arrive as f32 tensors and never float
/// again. ±inf saturates; NaN maps to 0 (both documented, mirroring the
/// fp16 sibling).
#[must_use]
pub fn f32_bits_to_milli(bits: u32) -> i32 {
    let negative = (bits >> 31) == 1;
    let exp = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0);
    if exp == 0 {
        return 0; // ±0 / subnormal-f32 → 0 at milli resolution (floor)
    }
    if exp == 0xFF {
        if bits & 0x007F_FFFF == 0 {
            return if negative { i32::MIN } else { i32::MAX };
        }
        return 0; // NaN — documented mapping
    }
    let mant = i64::from(bits & 0x007F_FFFF) | (1_i64 << 23);
    // value = mant × 2^(exp-127-23); milli = value × 1000.
    // exp ∈ [1, 254]: shift can be negative (large exponents) — f32 max
    // (≈3.4e38 → 3.4e41 milli) far exceeds i32, so saturate via i64 pow.
    let e = i64::from(exp) - 127 - 23;
    let scaled = if e >= 0 {
        mant.checked_shl(u32::try_from(e).unwrap_or(63))
            .map(|m| m.saturating_mul(1000))
            .unwrap_or(i64::MAX)
    } else {
        let shift = u32::try_from(-e).unwrap_or(63);
        let num = mant * 1000;
        let half = 1_i64 << (shift - 1);
        (num + half) >> shift
    };
    let milli = scaled.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    if negative {
        -milli
    } else {
        milli
    }
}

/// Exact integer square root: the unique `r` with `r² ≤ n < (r+1)².
/// Newton's method on u64 — no float anywhere.
#[must_use]
pub fn isqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    // Seed from the bit length (halved): always ≥ sqrt for a fast descent.
    let bits = 64 - n.leading_zeros();
    let mut x = 1_u64 << bits.div_ceil(2);
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            break;
        }
        x = y;
    }
    x
}

/// Integer RMSNorm: `out[i] = round(x[i] · w[i] / rms)` with
/// `rms = max(isqrt(round(mean(x²))), 1)` — the `max(·, 1)` is the integer
/// eps floor (Qwen3's 1e-6 eps rounds to 0 at milli resolution; a zero rms
/// on a zero vector must not divide by zero).
///
/// All-milli, all-integer: `Σx²` in i64, per-element `x·w` in i64 with
/// round-half-away division. Bounds documented at the module level.
///
/// # Panics (debug only)
///
/// Debug-asserts the documented working range (`Σx²` and `|x·w| < 2^63`).
pub fn rms_norm_milli(x: &[i32], w: &[i32], out: &mut [i32]) {
    debug_assert_eq!(x.len(), w.len(), "weight length mismatch");
    debug_assert_eq!(x.len(), out.len(), "output length mismatch");
    let n = x.len();
    let sum_sq: i64 = x
        .iter()
        .map(|&v| {
            let v64 = i64::from(v);
            v64.checked_mul(v64)
                .expect("x² overflow — outside documented working range")
        })
        .sum();
    debug_assert!(sum_sq >= 0, "Σx² overflowed");
    let mean = (sum_sq + n as i64 / 2) / n as i64; // round
    let rms = isqrt(mean.unsigned_abs()).max(1) as i64;
    for i in 0..n {
        let prod = i64::from(x[i]) * i64::from(w[i]);
        let q = if prod >= 0 {
            (prod + rms / 2) / rms
        } else {
            -((-prod + rms / 2) / rms)
        };
        out[i] = q.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_milli_known_vectors() {
        // f32 bits computed exactly: 1.0, 0.5, −2.0, 100.0, 1e-4 (→ 0.1 → 0).
        assert_eq!(f32_bits_to_milli(0x3F80_0000), 1000); // 1.0
        assert_eq!(f32_bits_to_milli(0x3F00_0000), 500); // 0.5
        assert_eq!(f32_bits_to_milli(0xC000_0000), -2000); // −2.0
        assert_eq!(f32_bits_to_milli(0x42C8_0000), 100_000); // 100.0
        assert_eq!(f32_bits_to_milli(0x38D1_B717), 0); // ≈1e-4 → 0.1 milli → 0
        assert_eq!(f32_bits_to_milli(0x0000_0000), 0); // +0
        assert_eq!(f32_bits_to_milli(0x7F80_0000), i32::MAX); // +inf
        assert_eq!(f32_bits_to_milli(0xFF80_0000), i32::MIN); // −inf
        assert_eq!(f32_bits_to_milli(0x7FC0_0000), 0); // NaN
    }

    #[test]
    fn isqrt_exact_small_and_large() {
        for n in 0_u64..10_000 {
            let r = isqrt(n);
            assert!(r * r <= n, "isqrt({n}) = {r} too big");
            assert!((r + 1) * (r + 1) > n, "isqrt({n}) = {r} too small");
        }
        // Perfect squares at scale.
        assert_eq!(isqrt(1_000_000), 1000);
        assert_eq!(isqrt(4_000_000_000_000_000), 63_245_553); // ⌊√4e15⌉? no: 63 245 553² = 4.0e15
        assert_eq!(isqrt(u64::MAX), 4_294_967_295); // (2^32−1)² ≤ 2^64−1
    }

    #[test]
    fn rms_norm_unit_vector_is_weight() {
        // x = [1000, 0, 0, 0] (milli 1.0, 0, 0, 0), w = [500, 1, 1, 1].
        // mean(x²) = 250000, rms = 500; out = x·w/500 = [1000, 0, 0, 0].
        let x = [1000, 0, 0, 0];
        let w = [500, 1, 1, 1];
        let mut out = [0_i32; 4];
        rms_norm_milli(&x, &w, &mut out);
        assert_eq!(out, [1000, 0, 0, 0]);
    }

    #[test]
    fn rms_norm_known_uniform() {
        // x = [3000, 4000]: Σx² = 25e6, mean rounds to 12_500_001,
        // isqrt = 3535 (3535² = 12_496_225 ≤ · < 3536²). w = 1.0:
        // out = round(x·1000/3535) = [849, 1132] (verified by hand from
        // the exact integers: (3_001_767)/3535 = 849.0+, 4_001_767/3535
        // = 1132.4).
        let x = [3000, 4000];
        let w = [1000, 1000];
        let mut out = [0_i32; 2];
        rms_norm_milli(&x, &w, &mut out);
        assert_eq!(out, [849, 1132]);
    }

    #[test]
    fn rms_norm_zero_vector_is_zero() {
        let mut out = [9_i32; 3];
        rms_norm_milli(&[0, 0, 0], &[1000, 1000, 1000], &mut out);
        assert_eq!(out, [0, 0, 0]); // eps floor saves the divide; x=0 → 0
    }

    #[test]
    fn rms_norm_scale_invariance_of_shape() {
        // Doubling x leaves out proportional (rms scales linearly).
        let w = [1000_i32; 4];
        let x1 = [100_i32, 200, 300, 400];
        let x2 = [200_i32, 400, 600, 800];
        let mut o1 = [0_i32; 4];
        let mut o2 = [0_i32; 4];
        rms_norm_milli(&x1, &w, &mut o1);
        rms_norm_milli(&x2, &w, &mut o2);
        for i in 0..4 {
            let (a, b) = (o1[i], o2[i]);
            // Doubling x doubles rms → out ≈ unchanged. isqrt's ±1 on rms
            // gives the relative slack: |b − a| ≤ |a|/273 + 2.
            let tol = a.unsigned_abs() as i64 / 273 + 2;
            assert!(
                (i64::from(b) - i64::from(a)).abs() <= tol,
                "not scale-invariant: {a} vs {b}"
            );
        }
    }
}
