//! Stage 3 of the ternary bridge: the **shared kernel** (see
//! `docs/VISION.md`).
//!
//! One `no_std`, zero-alloc, integer-only ternary matmul primitive that an
//! SNN layer and a dense LLM-style layer both call — the mechanical proof
//! that the union composes. This is the BitNet-matmul analog: packed
//! ternary weights `{−1, 0, +1}` × Q15 activations, i32 accumulation.
//!
//! # Two formats, two roles (the Stage-3 decision)
//!
//! - **Wire format** = `BitNet` `i2_s` ([`crate::bridge`], transposed 4-lane
//!   packing) — what crosses systems.
//! - **Compute format** = this module's **sequential** packing — element
//!   `i` at byte `i/4`, shift `2·(i%4)`, LSB-first lanes, codes
//!   `{0,1,2} = {−1,0,+1}`. Sequential indexing is what a hot matmul loop
//!   wants (no lane transpose math per element).
//!
//! [`crate::bridge::repack_i2s_to_kernel`] is the seam: weights arrive as
//! `i2_s` bytes, repack once, compute in the native layout.
//!
//! # Numeric contract
//!
//! Activations are Q15 i16 (`−32768..=32767`, produced by
//! [`absmax_normalize_q15`] — the integer analog of `BitNet`'s per-token
//! absmax activation quantization). Accumulation is i32 with the documented
//! bound `|acc| ≤ n × 32767` — safe for any row width `n < 65 536`. No
//! float, no heap: every function is buffer-based.

#![allow(clippy::module_name_repetitions)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::bridge::BridgeError;
use crate::trit::Trit;

/// Sequential packing: 4 trits per byte.
pub const TRITS_PER_BYTE: usize = 4;

/// Q15 full scale (1.0 in the activation fixed-point).
pub const Q15_MAX: i16 = i16::MAX;

/// Pack a ternary slice into the kernel's sequential compute layout.
///
/// Element `i` → byte `i/4`, shift `2·(i%4)`; codes `{0,1,2}` =
/// `{−1,0,+1}` (the shared code table). Requires `trits.len() % 4 == 0` and
/// `out.len() >= trits.len()/4`. Returns bytes written.
///
/// # Errors
///
/// [`BridgeError::BadLength`] on a non-multiple-of-4 slice;
/// [`BridgeError::TooShort`] on an undersized `out`.
pub fn pack_trits(trits: &[Trit], out: &mut [u8]) -> Result<usize, BridgeError> {
    let n = trits.len();
    if !n.is_multiple_of(TRITS_PER_BYTE) {
        return Err(BridgeError::BadLength);
    }
    if out.len() < n / TRITS_PER_BYTE {
        return Err(BridgeError::TooShort);
    }
    out[..n / TRITS_PER_BYTE].fill(0);
    for (i, t) in trits.iter().enumerate() {
        let code = match t {
            Trit::MinusOne => 0,
            Trit::Zero => 1,
            Trit::One => 2,
        };
        out[i / TRITS_PER_BYTE] |= code << (2 * (i % TRITS_PER_BYTE));
    }
    Ok(n / TRITS_PER_BYTE)
}

/// Unpack element `i` from sequential kernel packing.
///
/// # Errors
///
/// [`BridgeError::UnsupportedCode`] on code 3 (corrupt data).
pub fn unpack_trit(packed: &[u8], i: usize) -> Result<Trit, BridgeError> {
    let code = (packed[i / TRITS_PER_BYTE] >> (2 * (i % TRITS_PER_BYTE))) & 0x03;
    match code {
        0 => Ok(Trit::MinusOne),
        1 => Ok(Trit::Zero),
        2 => Ok(Trit::One),
        _ => Err(BridgeError::UnsupportedCode),
    }
}

/// Normalize a vector into Q15 by its absmax — the integer analog of
/// `BitNet`'s per-token activation quantization.
///
/// `out[i] = round(value[i] / absmax × 32767)` in pure integer math
/// (round-half-away-from-zero). Returns the absmax itself — the scale the
/// caller needs to un-normalize downstream (for argmax consumers the scale
/// cancels; for honest dense outputs it does not). An all-zero vector
/// normalizes to zeros and returns 0.
///
/// `values.len()` must equal `out.len()`.
#[must_use]
pub fn absmax_normalize_q15(values: &[i16], out: &mut [i16]) -> i16 {
    debug_assert_eq!(values.len(), out.len(), "value/out length mismatch");
    let absmax = values.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    if absmax == 0 {
        out.fill(0);
        return 0;
    }
    for (v, slot) in values.iter().zip(out.iter_mut()) {
        // |v| ≤ absmax → |num| ≤ 32767 × absmax ≤ 32767×32768 < 2^31: exact.
        let num = i32::from(*v) * i32::from(Q15_MAX);
        let den = i32::from(absmax);
        let half = den / 2; // den > 0
        let q = if num >= 0 {
            (num + half) / den
        } else {
            -((-num + half) / den)
        };
        *slot = q.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    absmax as i16
}

/// The shared ternary matvec: `out[j] = Σᵢ w[j·n+i] · a[i]`.
///
/// `packed_weights` holds `rows` consecutive rows of sequentially-packed
/// ternary weights (each row `n = activations.len()` trits, `n/4` bytes —
/// pack with [`pack_trits`] or arrive via
/// [`crate::bridge::repack_i2s_to_kernel`]); `activations` are Q15 i16;
/// `out` receives `rows` i32 accumulators.
///
/// Accumulation bound: `|out[j]| ≤ n × 32767` — overflow-free for any
/// `n < 65 536` (documented, asserted in debug builds).
///
/// # Errors
///
/// [`BridgeError::BadLength`] if `n % 4 != 0`; [`BridgeError::TooShort`] if
/// `packed_weights.len() < rows·n/4` or `out.len() < rows`;
/// [`BridgeError::UnsupportedCode`] on any code-3 lane.
pub fn ternary_matvec(
    packed_weights: &[u8],
    activations: &[i16],
    rows: usize,
    out: &mut [i32],
) -> Result<(), BridgeError> {
    let n = activations.len();
    if !n.is_multiple_of(TRITS_PER_BYTE) {
        return Err(BridgeError::BadLength);
    }
    let row_bytes = n / TRITS_PER_BYTE;
    if packed_weights.len() < rows * row_bytes || out.len() < rows {
        return Err(BridgeError::TooShort);
    }
    debug_assert!(
        n <= 65_535,
        "i32 accumulator bound |acc| ≤ n·32767 requires n < 65 536"
    );
    for (j, acc) in out.iter_mut().enumerate().take(rows) {
        let row = &packed_weights[j * row_bytes..(j + 1) * row_bytes];
        let mut sum: i32 = 0;
        for (i, &a) in activations.iter().enumerate() {
            let code = (row[i / TRITS_PER_BYTE] >> (2 * (i % TRITS_PER_BYTE))) & 0x03;
            let w = match code {
                0 => -1_i32,
                1 => 0,
                2 => 1,
                _ => return Err(BridgeError::UnsupportedCode),
            };
            sum += w * i32::from(a);
        }
        *acc = sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::shadow_unrelated)]
    use super::*;
    use proptest::prelude::*;

    // ----- Packing -----

    #[test]
    fn pack_known_vector() {
        // [+1, −1, 0, 0] → codes 2,0,1,1 → byte 2 | 0<<2 | 1<<4 | 1<<6 = 0x52.
        let trits = [Trit::One, Trit::MinusOne, Trit::Zero, Trit::Zero];
        let mut out = [0_u8; 1];
        assert_eq!(pack_trits(&trits, &mut out), Ok(1));
        assert_eq!(out[0], 0x52);
        // Unpack round-trip.
        for (i, &t) in trits.iter().enumerate() {
            assert_eq!(unpack_trit(&out, i), Ok(t));
        }
    }

    #[test]
    fn pack_rejects_bad_input() {
        let mut out = [0_u8; 1];
        assert_eq!(
            pack_trits(&[Trit::One; 3], &mut out),
            Err(BridgeError::BadLength)
        );
        assert_eq!(
            pack_trits(&[Trit::One; 8], &mut out),
            Err(BridgeError::TooShort)
        );
    }

    // ----- absmax Q15 -----

    #[test]
    fn absmax_known_vector() {
        let vals = [10_i16, 5, 0, -10];
        let mut out = [0_i16; 4];
        let scale = absmax_normalize_q15(&vals, &mut out);
        assert_eq!(scale, 10);
        assert_eq!(out, [32_767, 16_384, 0, -32_767]); // 16383.5 rounds away → 16384
    }

    #[test]
    fn absmax_zero_vector_is_zeros() {
        let mut out = [7_i16; 3];
        assert_eq!(absmax_normalize_q15(&[0, 0, 0], &mut out), 0);
        assert_eq!(out, [0, 0, 0]);
    }

    // ----- Matvec -----

    #[test]
    fn matvec_known_vector() {
        // W = [[+1, 0, −1, 0], [−1, −1, −1, −1]], a = [1000, 5000, 200, 7].
        let row0 = [Trit::One, Trit::Zero, Trit::MinusOne, Trit::Zero];
        let row1 = [Trit::MinusOne; 4];
        let mut packed = [0_u8; 2];
        pack_trits(&row0, &mut packed[..1]).expect("pack");
        pack_trits(&row1, &mut packed[1..]).expect("pack");
        let a = [1000_i16, 5000, 200, 7];
        let mut out = [0_i32; 2];
        ternary_matvec(&packed, &a, 2, &mut out).expect("matvec");
        assert_eq!(out, [1000 - 200, -(1000 + 5000 + 200 + 7)]);
    }

    #[test]
    fn matvec_rejects_bad_input() {
        let mut out = [0_i32; 1];
        let mut packed = [0_u8; 1];
        // n % 4 != 0.
        assert_eq!(
            ternary_matvec(&packed, &[1_i16, 2, 3], 1, &mut out),
            Err(BridgeError::BadLength)
        );
        // packed too short for rows=2.
        assert_eq!(
            ternary_matvec(&packed, &[1_i16, 2, 3, 4], 2, &mut out),
            Err(BridgeError::TooShort)
        );
        // out too short for rows=2.
        pack_trits(&[Trit::One; 4], &mut packed).expect("pack");
        assert_eq!(
            ternary_matvec(&packed, &[1_i16, 2, 3, 4], 2, &mut out),
            Err(BridgeError::TooShort)
        );
    }

    // ----- Properties: matvec vs an unpacked scalar reference -----

    proptest! {
        /// For any packed weights and activations, ternary_matvec equals a
        /// naive Trit×i16 scalar reference — exact integer equality.
        #[test]
        fn prop_matvec_matches_scalar_reference(
            n4 in 1_usize..=16,        // row width in bytes (4·n4 trits)
            rows in 1_usize..=6,
            seed in any::<u64>(),
        ) {
            let n = n4 * 4;
            // xorshift64 trit generator.
            let mut x = seed | 1;
            let mut next_trit = || {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                match x % 3 { 0 => Trit::MinusOne, 1 => Trit::Zero, _ => Trit::One }
            };
            let trits: Vec<Vec<Trit>> =
                (0..rows).map(|_| (0..n).map(|_| next_trit()).collect()).collect();
            let acts: Vec<i16> = (0..n).map(|_| next_trit() as i16 * 10_000).collect();

            let mut packed = vec![0_u8; rows * n4];
            for (r, row) in trits.iter().enumerate() {
                pack_trits(row, &mut packed[r * n4..(r + 1) * n4]).unwrap();
            }
            let mut out = vec![0_i32; rows];
            ternary_matvec(&packed, &acts, rows, &mut out).unwrap();

            for (r, row) in trits.iter().enumerate() {
                let reference: i32 = row
                    .iter()
                    .zip(&acts)
                    .map(|(&t, &a)| i32::from(t.to_weight(1)) * i32::from(a))
                    .sum();
                prop_assert_eq!(out[r], reference, "row {}", r);
            }
        }

        /// absmax output: sign-preserving, |out| ≤ 32767, and the max
        /// magnitude element attains ±32767 exactly.
        #[test]
        fn prop_absmax_bounds_and_attainment(
            vals in prop::collection::vec(-3000i16..=3000, 1..=40),
        ) {
            let mut out = vec![0_i16; vals.len()];
            let scale = absmax_normalize_q15(&vals, &mut out);
            let max_abs = vals.iter().map(|v| v.unsigned_abs()).max().unwrap();
            prop_assert_eq!(scale, max_abs as i16);
            for (v, &o) in vals.iter().zip(out.iter()) {
                // Q15 by construction: the output can never be i16::MIN
                // (|out| ≤ 32767 — the clamp is unreachable for in-range math).
                prop_assert!(o != i16::MIN, "{o} escaped Q15");
                if *v == 0 { prop_assert_eq!(o, 0); }
                if *v > 0 { prop_assert!(o >= 0); }
                if *v < 0 { prop_assert!(o <= 0); }
            }
            if max_abs > 0 {
                prop_assert!(out.iter().any(|&o| o.abs() == 32_767));
            }
        }

        /// pack → unpack round-trips every element.
        #[test]
        fn prop_pack_unpack_round_trip(
            n4 in 1_usize..=32,
            seed in any::<u64>(),
        ) {
            let n = n4 * 4;
            let mut x = seed | 1;
            let trits: Vec<Trit> = (0..n).map(|_| {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                match x % 3 { 0 => Trit::MinusOne, 1 => Trit::Zero, _ => Trit::One }
            }).collect();
            let mut packed = vec![0_u8; n4];
            pack_trits(&trits, &mut packed).unwrap();
            for (i, &t) in trits.iter().enumerate() {
                prop_assert_eq!(unpack_trit(&packed, i), Ok(t));
            }
        }
    }
}
