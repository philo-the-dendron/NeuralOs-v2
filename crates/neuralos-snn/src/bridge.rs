//! Stage 2 of the ternary bridge: the **format bridge** (see `docs/VISION.md`
//! and `docs/TERNARY_FORMAT.md`).
//!
//! `NeuralOS` speaks the byte layouts of the two ternary ecosystems that
//! actually ship models:
//!
//! - **`BitNet` `i2_s`** (microsoft/BitNet, `utils/convert-hf-to-gguf-bitnet.py`
//!   `quantize_to_i2_s`): export **and** import. 2-bit codes `{0,1,2}` =
//!   `{−1, 0, +1}` packed 4-per-byte in a *transposed* layout (see
//!   [`encode_i2_s`]), plus a 32-byte tail whose first 4 bytes are the LE
//!   f32 scale. Scale semantics: BitNet-Round `γ = mean|w|` — the same
//!   convention as [`crate::trit::tensor_scale`].
//! - **Prism `q1_0`** (PrismML-Eng/llama.cpp `ggml-common.h` `block_q1_0`):
//!   import only. 18 bytes per 128 weights: LE fp16 scale (`γ = mean|w|`) +
//!   16 sign bytes (set bit → `+γ`). Binary `{−γ, +γ}` — no zero state; it
//!   embeds losslessly into ternary, so no information is lost on import.
//!   Export does not exist: ternary → binary would silently turn zeros into
//!   `+γ`.
//! - **Prism `q2_0`** (same fork, `block_q2_0`): import only. 34 bytes per
//!   128 weights: LE fp16 scale (`max|w|`) + 32 bytes of LSB-first 2-bit
//!   lanes, `00`=−1 `01`=0 `10`=+1. Code `11` (`+2·d`) cannot be produced by
//!   the reference quantizer and is rejected loudly here. (Re-pinned in
//!   session D from the fork's `ggml/src/ggml-common.h` — `QK2_0 = 128`,
//!   `qs[QK2_0/4]` — after the first real `q2_0` file measured 680 B per
//!   2560-wide row, refuting this module's earlier 18-B/64-weight layout.)
//!
//! # Integer-only, `no_std`, zero-alloc
//!
//! No float types anywhere: fp16/fp32 scales travel as raw bits ([`u16`] /
//! [`u32`]). [`half_to_f32_bits`] widens fp16→fp32 bit-exactly with pure
//! integer arithmetic; [`half_to_milli`] gives the fixed-point numeric view
//! (`round(v × 1000)`, saturating). All codecs are buffer-based — the caller
//! provides the output slice — so the module works on a bare `no_std` target
//! with no allocator. That is the RISC-V posture: decode Bonsai weights on
//! the edge device itself.
//!
//! # Length rules (honesty over permissiveness)
//!
//! - `i2_s` requires `n % 128 == 0`: the reference's transposed packing
//!   truncates output at `n/4` bytes, which silently drops elements whenever
//!   `n % 128 != 0`. A permissive codec would be silently lossy; this one
//!   refuses.
//! - `q1_0` requires `n % 128 == 0` (the C reference asserts the same).
//! - `q2_0` requires `n % 128 == 0` (ditto — `QK2_0` is 128, re-pinned
//!   session D; it is NOT 64).

#![allow(clippy::module_name_repetitions)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::trit::Trit;

/// The `i2_s` tail size in bytes: 4 bytes of LE f32 scale bits, then 28
/// zero bytes (the reference aligns each row to 32).
pub const I2_S_TAIL_BYTES: usize = 32;

/// `i2_s` block size in values (transposed packing operates on 4 lanes of
/// 32 elements).
pub const I2_S_BLOCK: usize = 128;

/// `q1_0` block size in values (one fp16 scale + 16 sign bytes).
pub const Q1_0_BLOCK: usize = 128;

/// `q2_0` block size in values: one fp16 scale + 32 bytes of 2-bit codes
/// (`QK2_0 = 128` in the fork's `ggml/src/ggml-common.h`; 34 B/block).
pub const Q2_0_BLOCK: usize = 128;

/// Errors from the format codecs. Loud by design — no decode path clamps,
/// pads, or guesses. A short buffer, a wrong length, or an impossible code
/// is an [`Err`], never best-effort output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeError {
    /// The trit slice length is not a multiple of the format's block size
    /// (`i2_s`/`q1_0`: 128, `q2_0`: 64).
    BadLength,
    /// The byte buffer is shorter than the layout requires.
    TooShort,
    /// A 2-bit code the reference quantizer can never produce (`3`) was
    /// found in the input — the data is malformed or not this format.
    UnsupportedCode,
}

impl core::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadLength => {
                write!(f, "tensor length is not a multiple of the block size")
            }
            Self::TooShort => write!(f, "byte buffer shorter than the layout requires"),
            Self::UnsupportedCode => {
                write!(f, "2-bit code 3 found (reference quantizer cannot emit it)")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BridgeError {}

// ---------------------------------------------------------------------------
// Code <-> Trit tables (shared 2-bit encoding: 00=-1, 01=0, 10=+1)
// ---------------------------------------------------------------------------

const fn trit_to_code(t: Trit) -> u8 {
    match t {
        Trit::MinusOne => 0,
        Trit::Zero => 1,
        Trit::One => 2,
    }
}

const fn code_to_trit(code: u8) -> Result<Trit, BridgeError> {
    match code {
        0 => Ok(Trit::MinusOne),
        1 => Ok(Trit::Zero),
        2 => Ok(Trit::One),
        _ => Err(BridgeError::UnsupportedCode),
    }
}

// ---------------------------------------------------------------------------
// BitNet i2_s — encode + decode
// ---------------------------------------------------------------------------

/// Encoded byte length of an `i2_s` tensor with `n` trits: `n/4` packed
/// bytes plus the 32-byte scale tail. `n` must be a multiple of 128.
#[must_use]
const fn i2_s_encoded_len(n: usize) -> usize {
    n / 4 + I2_S_TAIL_BYTES
}

/// The `i2_s` byte index holding element `i`'s code.
///
/// The reference packs 128 values into 32 bytes *transposed*: element `i`
/// lives at byte `(i/128)*32 + (i%32)` in lane `(i%128)/32`, counting from
/// the top bits. This is `numpy.reshape(n, 4, 32)` with lane 0 shifted `<<6`.
#[must_use]
const fn i2_s_byte_index(i: usize) -> usize {
    (i / I2_S_BLOCK) * 32 + (i % 32)
}

/// The 2-bit shift (within the byte) of element `i`'s lane: lane 0 → bits
/// 7-6, lane 1 → bits 5-4, lane 2 → bits 3-2, lane 3 → bits 1-0.
#[must_use]
const fn i2_s_lane_shift(i: usize) -> u32 {
    6 - 2 * (((i % I2_S_BLOCK) / 32) as u32)
}

/// Encode a ternary tensor into the `BitNet` `i2_s` byte layout.
///
/// `scale_bits` is the raw f32 bit pattern of the scale (BitNet-Round
/// `γ = mean|w|`; callers on the i16 substrate can derive it from
/// [`crate::trit::tensor_scale`]). Layout: `n/4` packed code bytes in the
/// transposed 4-lane packing, then 32 tail bytes — the first 4 the LE f32
/// scale bits, the rest zero, matching the reference.
///
/// Requires `trits.len() % 128 == 0` and `out.len() >= i2_s_encoded_len(n)`.
/// Returns the number of bytes written.
///
/// # Errors
///
/// [`BridgeError::BadLength`] if `n % 128 != 0`; [`BridgeError::TooShort`] if
/// `out` cannot hold the encoding.
pub fn encode_i2_s(trits: &[Trit], scale_bits: u32, out: &mut [u8]) -> Result<usize, BridgeError> {
    let n = trits.len();
    if !n.is_multiple_of(I2_S_BLOCK) {
        return Err(BridgeError::BadLength);
    }
    let written = i2_s_encoded_len(n);
    if out.len() < written {
        return Err(BridgeError::TooShort);
    }
    out[..written].fill(0);
    for (i, t) in trits.iter().enumerate() {
        out[i2_s_byte_index(i)] |= trit_to_code(*t) << i2_s_lane_shift(i);
    }
    out[n / 4..n / 4 + 4].copy_from_slice(&scale_bits.to_le_bytes());
    Ok(written)
}

/// Decode a `BitNet` `i2_s` byte stream back into ternary values.
///
/// `trits.len()` is the tensor length `n` (must be `n % 128 == 0`);
/// `bytes.len()` must be at least `i2_s_encoded_len(n)`. Returns the raw f32
/// scale bits from the tail — the exact inverse of [`encode_i2_s`]. The
/// 28 tail pad bytes after the scale are not validated (the reference
/// ignores them too). On [`BridgeError::UnsupportedCode`], `trits` may
/// hold partial output — do not use it.
///
/// # Errors
///
/// [`BridgeError::BadLength`] / [`BridgeError::TooShort`] on bad sizes;
/// [`BridgeError::UnsupportedCode`] if any 2-bit lane holds code 3 (the
/// reference encoder never emits it).
pub fn decode_i2_s(bytes: &[u8], trits: &mut [Trit]) -> Result<u32, BridgeError> {
    let n = trits.len();
    if !n.is_multiple_of(I2_S_BLOCK) {
        return Err(BridgeError::BadLength);
    }
    if bytes.len() < i2_s_encoded_len(n) {
        return Err(BridgeError::TooShort);
    }
    for (i, slot) in trits.iter_mut().enumerate() {
        let byte = bytes[i2_s_byte_index(i)];
        let code = (byte >> i2_s_lane_shift(i)) & 0x03;
        *slot = code_to_trit(code)?;
    }
    let mut scale = [0u8; 4];
    scale.copy_from_slice(&bytes[n / 4..n / 4 + 4]);
    Ok(u32::from_le_bytes(scale))
}

// ---------------------------------------------------------------------------
// Prism q1_0 — decode (import only)
// ---------------------------------------------------------------------------

/// Encoded byte length of a `q1_0` tensor with `n` weights: 18 bytes per
/// 128-weight block (2-byte LE fp16 scale + 16 sign bytes).
#[must_use]
pub const fn q1_0_encoded_len(n: usize) -> usize {
    (n / Q1_0_BLOCK) * 18
}

/// Decode a Prism `q1_0` tensor: per 128-weight block, a LE fp16 scale
/// (`γ = mean|w|`) followed by 16 sign bytes — element `j`'s sign is bit
/// `j%8` of byte `j/8` (LSB-first); set → `+γ`, clear → `−γ`.
///
/// The format is binary, so every decoded trit is [`Trit::One`] or
/// [`Trit::MinusOne`] — no zeros exist in `q1_0`. Returns the fp16 scale
/// bits of the first block (see `docs/TERNARY_FORMAT.md` for the
/// multi-block scale caveat: Prism stores one scale *per block*; this
/// decode returns per-block scales through `scale_bits_out` when provided,
/// and the first block's scale for the one-block case).
///
/// Requires `trits.len() % 128 == 0` and `bytes.len() >= q1_0_encoded_len(n)`.
///
/// # Errors
///
/// [`BridgeError::BadLength`] / [`BridgeError::TooShort`] on bad sizes.
pub fn decode_q1_0(
    bytes: &[u8],
    trits: &mut [Trit],
    scale_bits_out: &mut [u16],
) -> Result<(), BridgeError> {
    let n = trits.len();
    if !n.is_multiple_of(Q1_0_BLOCK) {
        return Err(BridgeError::BadLength);
    }
    let blocks = n / Q1_0_BLOCK;
    if bytes.len() < q1_0_encoded_len(n) {
        return Err(BridgeError::TooShort);
    }
    if scale_bits_out.len() < blocks {
        return Err(BridgeError::TooShort);
    }
    for b in 0..blocks {
        let base = b * 18;
        scale_bits_out[b] = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
        for j in 0..Q1_0_BLOCK {
            let sign = (bytes[base + 2 + j / 8] >> (j % 8)) & 1;
            trits[b * Q1_0_BLOCK + j] = if sign == 1 { Trit::One } else { Trit::MinusOne };
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Prism q2_0 — decode + encode
// ---------------------------------------------------------------------------

/// Encoded byte length of a `q2_0` tensor with `n` weights: 34 bytes per
/// 128-weight block (2-byte LE fp16 scale + 32 bytes of 2-bit codes).
#[must_use]
pub const fn q2_0_encoded_len(n: usize) -> usize {
    (n / Q2_0_BLOCK) * 34
}

/// Decode a Prism `q2_0` tensor: per 128-weight block, a LE fp16 scale
/// (`max|w|`) followed by 32 bytes of LSB-first 2-bit lanes — element `j`'s
/// code is bits `2*(j%4)..2*(j%4)+1` of byte `j/4`, with `00`=−1, `01`=0,
/// `10`=+1.
///
/// Code `11` decodes to `+2·d` in the reference dequantizer but *cannot be
/// emitted* by the reference quantizer (with `d = max|w|`, `round(w/d)` never
/// leaves `[-1, 1]`). This decoder rejects it loudly rather than inventing a
/// mapping the reference never produced.
///
/// Requires `trits.len() % 128 == 0`. Scale bits per block are written to
/// `scale_bits_out` (needs `n/128` entries).
///
/// # Errors
///
/// [`BridgeError::BadLength`] / [`BridgeError::TooShort`] on bad sizes;
/// [`BridgeError::UnsupportedCode`] on code 3.
pub fn decode_q2_0(
    bytes: &[u8],
    trits: &mut [Trit],
    scale_bits_out: &mut [u16],
) -> Result<(), BridgeError> {
    let n = trits.len();
    if !n.is_multiple_of(Q2_0_BLOCK) {
        return Err(BridgeError::BadLength);
    }
    let blocks = n / Q2_0_BLOCK;
    if bytes.len() < q2_0_encoded_len(n) {
        return Err(BridgeError::TooShort);
    }
    if scale_bits_out.len() < blocks {
        return Err(BridgeError::TooShort);
    }
    for b in 0..blocks {
        let base = b * 34;
        scale_bits_out[b] = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
        for j in 0..Q2_0_BLOCK {
            let code = (bytes[base + 2 + j / 4] >> (2 * (j % 4))) & 0x03;
            trits[b * Q2_0_BLOCK + j] = code_to_trit(code)?;
        }
    }
    Ok(())
}

/// Encode a ternary tensor into the Prism `q2_0` byte layout — the exact
/// byte-level inverse of [`decode_q2_0`] (session E: the loop-closer's
/// export path; ternary `{−1,0,+1}` is losslessly representable because the
/// quantizer's reachable codes are exactly `trit + 1`).
///
/// `scale_bits` carries one raw LE fp16 bit pattern per 128-trit block.
/// Callers re-exporting imported weights pass the bits `decode_q2_0` handed
/// them — magnitudes preserved bit-exactly through the substrate round-trip.
/// Callers quantizing fresh from i16 weights derive the scale themselves
/// (the reference convention is `max|w|`).
///
/// Layout mirrors the decoder: per block, 2-byte LE scale then 32 bytes of
/// LSB-first 2-bit lanes — element `j`'s code at byte `j/4`, shift
/// `2*(j%4)`, with `00`=−1, `01`=0, `10`=+1. Code `11` is unconstructible
/// here: every lane is written from a [`Trit`], so an encoder round trip can
/// never produce the code the reference quantizer cannot.
///
/// Requires `trits.len() % 128 == 0`, `scale_bits.len() >= n/128`, and
/// `out.len() >= `[`q2_0_encoded_len`]`(n)`.
///
/// # Errors
///
/// [`BridgeError::BadLength`] if `n % 128 != 0`; [`BridgeError::TooShort`]
/// if `scale_bits` or `out` cannot hold the encoding.
pub fn encode_q2_0(trits: &[Trit], scale_bits: &[u16], out: &mut [u8]) -> Result<(), BridgeError> {
    let n = trits.len();
    if !n.is_multiple_of(Q2_0_BLOCK) {
        return Err(BridgeError::BadLength);
    }
    let blocks = n / Q2_0_BLOCK;
    if scale_bits.len() < blocks {
        return Err(BridgeError::TooShort);
    }
    if out.len() < q2_0_encoded_len(n) {
        return Err(BridgeError::TooShort);
    }
    for b in 0..blocks {
        let base = b * 34;
        out[base..base + 2].copy_from_slice(&scale_bits[b].to_le_bytes());
        out[base + 2..base + 34].fill(0);
        for j in 0..Q2_0_BLOCK {
            let code = trit_to_code(trits[b * Q2_0_BLOCK + j]);
            out[base + 2 + j / 4] |= code << (2 * (j % 4));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Integer-only scale plumbing (fp16 / fp32 as raw bits)
// ---------------------------------------------------------------------------

/// Widen an IEEE-754 binary16 bit pattern to binary32, bit-exactly, with
/// pure integer arithmetic (no float hardware needed — the RISC-V
/// FPU-less story).
///
/// Normals: `E32 = e - 15 + 127`, mantissa shifted up 13 bits. Subnormals
/// are normalized against the fp16 exponent floor `2^-14`. Zeros, infinities
/// and NaNs map to their exact fp32 counterparts (NaN is canonicalized to
/// the quiet payload the mantissa implies).
#[must_use]
pub const fn half_to_f32_bits(h: u16) -> u32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return sign; // ±0
        }
        // Subnormal fp16: value = mant × 2^-24. Normalize: find the leading
        // one at bit k (0-indexed), then E = k - 24, fp32 exp field = E + 127
        // = k + 103, and the remaining mantissa bits shift into place.
        let mut k = 0_u32;
        let m = mant;
        while (m >> k) > 1 {
            k += 1;
        }
        // k = index of the leading set bit. Exponent field = k + 127 - 24.
        let e32 = k + 103;
        let m32 = (mant ^ (1 << k)) << (23 - k);
        return sign | (e32 << 23) | m32;
    }
    if exp == 0x1F {
        // Inf / NaN: exponent all ones carries over; mantissa (incl. the
        // quiet bit if set) shifts up 13.
        return sign | (0xFF_u32 << 23) | (mant << 13);
    }
    sign | ((exp + 112) << 23) | (mant << 13)
}

/// Fixed-point numeric view of an fp16 scale: `round(value × 1000)` as
/// [`i32`], computed entirely in integers from the bit pattern.
///
/// Documented special handling (tested, not silent): ±infinity saturates to
/// [`i32::MAX`] / [`i32::MIN`]; NaN maps to 0; every finite fp16 fits without
/// saturation (max finite 65504 → `65_504_000` < 2^31). Values below 0.0005
/// round to 0 — the milli grid's honest floor.
#[must_use]
pub fn half_to_milli(h: u16) -> i32 {
    let bits = half_to_f32_bits(h);
    let negative = (bits >> 31) == 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    if exp == 0 {
        return 0; // ±0 (fp16-origin values can never be f32-subnormal)
    }
    let mant = (bits & 0x007F_FFFF) | (0x0080_0000); // implicit leading 1 (finite only)
    if exp == 0xFF {
        if bits.trailing_zeros() >= 23 {
            // ±inf: documented saturation, by sign.
            return if negative { i32::MIN } else { i32::MAX };
        }
        return 0; // NaN — documented mapping
    }
    // value = mant × 2^(exp-127-23); milli = value × 1000, rounded.
    // For fp16-derived values exp ≤ 142, so the shift below is always right
    // (≥ 8); mant×1000 ≤ 2^24×1000 < 2^34 — no i64 overflow.
    let shift = 127 + 23 - exp; // ≥ 8 for anything fp16-origin
    let scaled = i64::from(mant) * 1000;
    let milli = (scaled + (1_i64 << (shift - 1))) >> shift;
    if negative {
        (-milli) as i32
    } else {
        milli as i32
    }
}

// ---------------------------------------------------------------------------
// Stage 3: wire γ → substrate domain, and wire → compute repacking
// ---------------------------------------------------------------------------

/// Stage-3 scale policy (resolves the Stage-2 fog item): map an imported
/// wire-format γ — in the milli fixed-point view of [`half_to_milli`] —
/// into the i16 substrate weight domain.
///
/// `substrate_γ = round(milli × `[`crate::synapse::SCALE`]` / 1000)`,
/// saturating at [`i16::MIN`] / [`i16::MAX`]. With `SCALE = 1000` this is
/// numerically the identity; the function exists so the milli↔SCALE
/// coupling has exactly one home (and one pinning test,
/// `scale_constant_is_pinned`) instead of a scattered magic 1000. If SCALE
/// ever changes, this is the single line that adjusts.
///
/// Negative inputs are nonsense for a scale but saturate rather than panic
/// — documented, tested, never silent.
#[must_use]
pub fn wire_gamma_to_substrate(gamma_milli: i32) -> i16 {
    let scaled = (i64::from(gamma_milli) * i64::from(crate::synapse::SCALE)
        + if gamma_milli >= 0 { 500 } else { -500 })
        / 1000; // round-half-away-from-zero
    scaled.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

/// Repack a `BitNet` `i2_s` **wire** stream into the kernel's **compute**
/// packing — element-by-element bit surgery, no intermediate trit buffer
/// (Stage 3's zero-alloc seam between the two formats).
///
/// `i2_s` is the wire layout (transposed 4-lane); the kernel wants
/// sequential 2-bit codes (4 trits per byte, LSB-first lanes — element `i`
/// at byte `i/4`, shift `2·(i%4)`). Both use the code table
/// `{0,1,2} = {−1,0,+1}`.
///
/// `n` is the tensor length (must be `n % 128 == 0` — the wire side's
/// rule); `out` needs `n/4` bytes. The 32-byte `i2_s` tail (scale) is not
/// copied — read it with [`decode_i2_s`] or from the `n/4` offset directly.
///
/// # Errors
///
/// [`BridgeError::BadLength`] if `n % 128 != 0`; [`BridgeError::TooShort`]
/// if either buffer is undersized; [`BridgeError::UnsupportedCode`] on a
/// code-3 lane. On error, `out` may hold partial writes — do not use
/// them.
pub fn repack_i2s_to_kernel(i2s: &[u8], n: usize, out: &mut [u8]) -> Result<(), BridgeError> {
    if !n.is_multiple_of(I2_S_BLOCK) {
        return Err(BridgeError::BadLength);
    }
    if i2s.len() < i2_s_encoded_len(n) || out.len() < n / 4 {
        return Err(BridgeError::TooShort);
    }
    out[..n / 4].fill(0);
    for i in 0..n {
        let code = (i2s[i2_s_byte_index(i)] >> i2_s_lane_shift(i)) & 0x03;
        if code == 3 {
            return Err(BridgeError::UnsupportedCode);
        }
        out[i / 4] |= code << (2 * (i % 4));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::shadow_unrelated)]
    use super::*;
    use proptest::prelude::*;

    // ----- i2_s known vector (derived by hand from the reference script) -----

    #[test]
    fn i2_s_known_vector() {
        // Pattern of 8 codes {2,0,1,2,1,1,0,2} (= {+1,-1,0,+1,0,0,-1,+1}),
        // repeated to 128 elements. Transposed packing: byte j of the block
        // holds elements j (lane<<6), j+32 (lane<<4), j+64 (lane<<2), j+96
        // (lane<<0). Since the pattern has period 8 and 32 % 8 == 0, all four
        // lanes of byte j share the same code → bytes 0..7 = AA 00 55 AA 55
        // 55 00 AA, repeated 4× across the 32-byte block.
        let pat = [
            Trit::One,
            Trit::MinusOne,
            Trit::Zero,
            Trit::One,
            Trit::Zero,
            Trit::Zero,
            Trit::MinusOne,
            Trit::One,
        ];
        let trits: Vec<Trit> = (0..I2_S_BLOCK).map(|i| pat[i % 8]).collect();
        let scale_bits: u32 = 0x4000_0000; // f32 2.0
        let mut buf = [0xFF_u8; 128]; // pre-poisoned: encode must fully own its bytes
        let written = encode_i2_s(&trits, scale_bits, &mut buf).expect("encode");
        assert_eq!(written, 32 + I2_S_TAIL_BYTES);

        let mut expected = [0_u8; 64];
        expected[..32].copy_from_slice(&[0xAA, 0x00, 0x55, 0xAA, 0x55, 0x55, 0x00, 0xAA].repeat(4));
        expected[32..36].copy_from_slice(&scale_bits.to_le_bytes());
        assert_eq!(
            &buf[..written],
            &expected,
            "i2_s bytes must match reference layout"
        );

        // Round-trip through the decoder.
        let mut back = [Trit::Zero; I2_S_BLOCK];
        let got_scale = decode_i2_s(&buf, &mut back).expect("decode");
        assert_eq!(got_scale, scale_bits);
        assert_eq!(&back, &trits[..]);
    }

    #[test]
    fn i2_s_lane_order_golden_vector() {
        // 2026-08-15 review: the known vector above has period 8, and
        // 32 % 8 == 0 makes all four lanes of every byte identical — ANY
        // lane permutation reproduces those bytes. This vector uses a
        // period-3 pattern (3 ∤ 32) so the four lanes of each byte carry
        // DIFFERENT codes; the expected bytes are built by an
        // independent column-major loop (element e = j + 32·lane, code
        // << (6 − 2·lane)) that never touches i2_s_byte_index — a lane
        // swap now scrambles every byte and fails the test.
        let code_of = |i: usize| -> u8 {
            let t = match i % 3 {
                0 => Trit::MinusOne,
                1 => Trit::Zero,
                _ => Trit::One,
            };
            trit_to_code(t)
        };
        let trits: Vec<Trit> = (0..I2_S_BLOCK)
            .map(|i| match i % 3 {
                0 => Trit::MinusOne,
                1 => Trit::Zero,
                _ => Trit::One,
            })
            .collect();
        let mut buf = [0_u8; 64];
        let written = encode_i2_s(&trits, 0, &mut buf).expect("encode");
        assert_eq!(written, 64);

        // Independent expected-bytes construction (column-major).
        let mut expected = [0_u8; 64];
        for (j, slot) in expected.iter_mut().enumerate().take(32) {
            for lane in 0..4_usize {
                let e = 32 * lane + j; // element index in the 128-block
                *slot |= code_of(e) << (6 - 2 * lane);
            }
        }
        // Bytes 32..36 are the scale (0.0 here) — compare packed region.
        assert_eq!(
            &buf[..32],
            &expected[..32],
            "i2_s lane order must match the reference transposition"
        );

        // And decoding the independent bytes returns the original trits.
        let mut back = [Trit::Zero; I2_S_BLOCK];
        decode_i2_s(&buf, &mut back).expect("decode");
        assert_eq!(&back, &trits[..]);
    }

    #[test]
    fn i2_s_rejects_bad_lengths() {
        let trits = [Trit::One; 8]; // not a multiple of 128
        let mut out = [0u8; 64];
        assert_eq!(
            encode_i2_s(&trits, 0, &mut out),
            Err(BridgeError::BadLength)
        );
        let mut back = [Trit::Zero; 8];
        assert_eq!(
            decode_i2_s(&[0; 64], &mut back),
            Err(BridgeError::BadLength)
        );
    }

    #[test]
    fn i2_s_rejects_short_buffers() {
        let trits = [Trit::One; I2_S_BLOCK];
        let mut out = [0u8; 10]; // needs 64
        assert_eq!(encode_i2_s(&trits, 0, &mut out), Err(BridgeError::TooShort));
        let mut back = [Trit::Zero; I2_S_BLOCK];
        assert_eq!(decode_i2_s(&[0; 12], &mut back), Err(BridgeError::TooShort));
    }

    #[test]
    fn i2_s_rejects_code_three() {
        // One full-size buffer, every code lane = 3 (0xFF everywhere).
        let mut back = [Trit::Zero; I2_S_BLOCK];
        assert_eq!(
            decode_i2_s(&[0xFF; 64], &mut back),
            Err(BridgeError::UnsupportedCode)
        );
    }

    // ----- i2_s property: round-trip -----

    proptest! {
        #[test]
        fn prop_i2_s_round_trip(
            blocks in 1_usize..=4,
            seed in any::<u32>(),
            scale_bits in any::<u32>(),
        ) {
            // xorshift32 trit generator — deterministic from seed.
            let n = blocks * I2_S_BLOCK;
            let mut x = seed | 1;
            let trits: Vec<Trit> = (0..n)
                .map(|_| {
                    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                    match x % 3 { 0 => Trit::MinusOne, 1 => Trit::Zero, _ => Trit::One }
                })
                .collect();
            let mut buf = vec![0_u8; i2_s_encoded_len(n) + 8];
            let written = encode_i2_s(&trits, scale_bits, &mut buf).unwrap();
            prop_assert_eq!(written, i2_s_encoded_len(n));
            let mut back = vec![Trit::Zero; n];
            let got = decode_i2_s(&buf[..written], &mut back).unwrap();
            prop_assert_eq!(got, scale_bits);
            prop_assert_eq!(&back, &trits[..]);
        }
    }

    // ----- q1_0 known vector -----

    #[test]
    fn q1_0_known_vector() {
        // 16 sign bytes of 0xB5 = 0b1011_0101 → LSB-first bits
        // 1,0,1,0,1,1,0,1 → elements [+1,−1,+1,−1,+1,+1,−1,+1], repeated 16×.
        let pat = [
            Trit::One,
            Trit::MinusOne,
            Trit::One,
            Trit::MinusOne,
            Trit::One,
            Trit::One,
            Trit::MinusOne,
            Trit::One,
        ];
        let mut bytes = Vec::with_capacity(18);
        bytes.extend_from_slice(&0x3C00_u16.to_le_bytes()); // fp16 1.0
        bytes.extend(std::iter::repeat_n(0xB5_u8, 16));
        let mut trits = [Trit::Zero; Q1_0_BLOCK];
        let mut scales = [0_u16; 1];
        decode_q1_0(&bytes, &mut trits, &mut scales).expect("decode");
        assert_eq!(scales[0], 0x3C00);
        let expected: Vec<Trit> = (0..Q1_0_BLOCK).map(|i| pat[i % 8]).collect();
        assert_eq!(&trits, &expected[..], "q1_0 sign bits must map LSB-first");
    }

    #[test]
    fn q1_0_byte_order_golden_vector() {
        // 2026-08-15 review: the uniform-bytes vector above pins BIT
        // order but not BYTE order (16 identical bytes are position-
        // blind). Byte j gets exactly one set bit at position j%8 →
        // element 8j + (j%8) is positive (an independently-derived set:
        // {0, 9, 18, 27, 36, 45, 54, 63, 72+1, …}); every other element
        // is −γ.
        let mut bytes = Vec::with_capacity(18);
        bytes.extend_from_slice(&0x3C00_u16.to_le_bytes());
        let mut sign_bytes = [0_u8; 16];
        for (j, slot) in sign_bytes.iter_mut().enumerate() {
            *slot = 1 << (j % 8);
        }
        bytes.extend_from_slice(&sign_bytes);
        let mut trits = [Trit::Zero; Q1_0_BLOCK];
        let mut scales = [0_u16; 1];
        decode_q1_0(&bytes, &mut trits, &mut scales).expect("decode");
        let positives: std::collections::HashSet<usize> =
            (0..16_usize).map(|j| 8 * j + j % 8).collect();
        for (i, t) in trits.iter().enumerate() {
            let want = if positives.contains(&i) {
                Trit::One
            } else {
                Trit::MinusOne
            };
            assert_eq!(*t, want, "element {i}");
        }
    }

    #[test]
    fn q1_0_rejects_bad_input() {
        let mut trits = [Trit::Zero; Q1_0_BLOCK];
        let mut scales = [0u16; 1];
        // Length not a multiple of 128.
        let mut short_trits = [Trit::Zero; 64];
        assert_eq!(
            decode_q1_0(&[0; 18], &mut short_trits, &mut scales),
            Err(BridgeError::BadLength)
        );
        // Byte buffer one byte short.
        assert_eq!(
            decode_q1_0(&[0; 17], &mut trits, &mut scales),
            Err(BridgeError::TooShort)
        );
        // Scale output buffer too small.
        let mut no_scales = [0u16; 0];
        assert_eq!(
            decode_q1_0(&[0; 18], &mut trits, &mut no_scales),
            Err(BridgeError::TooShort)
        );
    }

    // ----- q2_0 known vector -----

    /// The session-D re-pin, as an arithmetic witness: the fork's
    /// `ggml/src/ggml-common.h` defines `QK2_0 = 128` with
    /// `qs[QK2_0/4]` (34 B/block), and the real
    /// `Ternary-Bonsai-4B-Q2_0.gguf` measures exactly
    /// `2560/128 × 34 = 680 B` per embedding row. The pre-session-D
    /// layout (64 w / 18 B) predicted 720 B/row and failed every one of
    /// the file's 253 `q2_0` tensors.
    #[test]
    fn q2_0_block_geometry_is_pinned() {
        assert_eq!(Q2_0_BLOCK, 128);
        assert_eq!(q2_0_encoded_len(128), 34);
        assert_eq!(q2_0_encoded_len(2560), 680);
        assert_eq!(q2_0_encoded_len(128 * 3), 34 * 3);
    }

    #[test]
    fn q2_0_known_vector() {
        // All 32 code bytes 0xA4 = 0b10_10_01_00 → lanes (LSB first)
        // 00,01,10,10 → elements [−1,0,+1,+1] repeated 32×. Scale fp16 4.0
        // = 0x4400.
        let pat = [Trit::MinusOne, Trit::Zero, Trit::One, Trit::One];
        let mut bytes = Vec::with_capacity(34);
        bytes.extend_from_slice(&0x4400_u16.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0xA4_u8, 32));
        let mut trits = [Trit::Zero; Q2_0_BLOCK];
        let mut scales = [0u16; 1];
        decode_q2_0(&bytes, &mut trits, &mut scales).expect("decode");
        assert_eq!(scales[0], 0x4400);
        let expected: Vec<Trit> = (0..Q2_0_BLOCK).map(|i| pat[i % 4]).collect();
        assert_eq!(&trits, &expected[..]);
    }

    #[test]
    fn q2_0_code3_rejected() {
        // First element's lane = code 3 (0b11) — loud error.
        let mut bytes = vec![0_u8; 34];
        bytes[0..2].copy_from_slice(&0x4400_u16.to_le_bytes());
        bytes[2] = 0x03;
        let mut trits = [Trit::Zero; Q2_0_BLOCK];
        let mut scales = [0_u16; 1];
        assert_eq!(
            decode_q2_0(&bytes, &mut trits, &mut scales),
            Err(BridgeError::UnsupportedCode)
        );
        // Code 3 in the LAST lane of the LAST byte too — the full 32-byte
        // code span is checked, not just the head of the block.
        let mut bytes = vec![0_u8; 34];
        bytes[0..2].copy_from_slice(&0x4400_u16.to_le_bytes());
        bytes[33] = 0xC0;
        assert_eq!(
            decode_q2_0(&bytes, &mut trits, &mut scales),
            Err(BridgeError::UnsupportedCode)
        );
    }

    #[test]
    fn q2_0_byte_and_lane_order_golden_vector() {
        // Period-3 codes across 128 elements: every byte carries three
        // distinct lanes and consecutive bytes differ — byte order AND
        // lane order both pinned (uniform 0xA4 bytes are position-blind).
        // codes: i%3 == 0 → −1 (00), 1 → 0 (01), 2 → +1 (10).
        let code_of = |i: usize| -> u8 { (i % 3) as u8 };
        let mut bytes = Vec::with_capacity(34);
        bytes.extend_from_slice(&0x4400_u16.to_le_bytes()); // fp16 4.0
        for j in 0..32_usize {
            let mut b = 0_u8;
            for lane in 0..4_usize {
                b |= code_of(4 * j + lane) << (2 * lane);
            }
            bytes.push(b);
        }
        let mut trits = [Trit::Zero; Q2_0_BLOCK];
        let mut scales = [0_u16; 1];
        decode_q2_0(&bytes, &mut trits, &mut scales).expect("decode");
        for (i, t) in trits.iter().enumerate() {
            let want = match i % 3 {
                0 => Trit::MinusOne,
                1 => Trit::Zero,
                _ => Trit::One,
            };
            assert_eq!(*t, want, "element {i}");
        }
    }

    #[test]
    fn q2_0_rejects_bad_input() {
        let mut trits = [Trit::Zero; Q2_0_BLOCK];
        let mut scales = [0u16; 1];
        let mut odd = [Trit::Zero; 100];
        assert_eq!(
            decode_q2_0(&[0; 34], &mut odd, &mut scales),
            Err(BridgeError::BadLength)
        );
        assert_eq!(
            decode_q2_0(&[0; 33], &mut trits, &mut scales),
            Err(BridgeError::TooShort)
        );
        // Two blocks need two scale slots.
        let mut two = [Trit::Zero; Q2_0_BLOCK * 2];
        assert_eq!(
            decode_q2_0(&[0; 34 * 2], &mut two, &mut scales),
            Err(BridgeError::TooShort)
        );
    }

    // ----- q2_0 encode (session E: the export inverse) -----

    /// The REAL first block of `token_embd.weight` from
    /// Ternary-Bonsai-4B-Q2_0.gguf (`data_start` + tensor offset, extracted
    /// verbatim — the artifact the session-D re-pin was validated against).
    /// 34 bytes: fp16 scale 0x24C8 (~18.7 milli max|w|) + 32 code bytes
    /// decoding to census +37 / 0×43 / −48 of 128.
    const REAL_Q2_0_FIRST_BLOCK: [u8; 34] = [
        0xC8, 0x24, 0x14, 0x44, 0x45, 0x1A, 0x18, 0x68, 0x68, 0x61, 0x8A, 0xA8, 0x91, 0x66, 0x45,
        0x42, 0x91, 0x80, 0x11, 0x62, 0x18, 0x11, 0x29, 0x48, 0x61, 0x00, 0x1A, 0x94, 0x81, 0x24,
        0x54, 0x0A, 0x86, 0x84,
    ];

    #[test]
    fn q2_0_real_artifact_first_block_decodes_and_round_trips() {
        // Decode the real bytes: exact scale bits + the recorded census.
        let mut trits = [Trit::Zero; Q2_0_BLOCK];
        let mut scales = [0u16; 1];
        decode_q2_0(&REAL_Q2_0_FIRST_BLOCK, &mut trits, &mut scales)
            .expect("real bytes decode (code 3 would mean the pin is wrong)");
        assert_eq!(scales[0], 0x24C8);
        let census = trits.iter().fold((0, 0, 0), |(p, z, m), t| match t {
            Trit::One => (p + 1, z, m),
            Trit::Zero => (p, z + 1, m),
            Trit::MinusOne => (p, z, m + 1),
        });
        assert_eq!(
            census,
            (37, 43, 48),
            "recorded probe census: +37 / 0×43 / −48"
        );
        // The strongest artifact assertion: decode → encode is the IDENTITY
        // on real bytes — the export codec reproduces the file it came from.
        let mut back = [0u8; 34];
        encode_q2_0(&trits, &scales, &mut back).expect("encode real block");
        assert_eq!(&back[..], &REAL_Q2_0_FIRST_BLOCK[..]);
    }

    #[test]
    fn encode_q2_0_reproduces_known_vector_bytes() {
        // The decode known vector, inverted byte-for-byte: 0x4400 scale +
        // 32 × 0xA4 code bytes decode to the period-4 [−1,0,+1,+1] pattern;
        // re-encoding that pattern with the SAME scale bits must give back
        // the identical 34 bytes — the encoder is the byte-level inverse.
        let pat = [Trit::MinusOne, Trit::Zero, Trit::One, Trit::One];
        let trits: Vec<Trit> = (0..Q2_0_BLOCK).map(|i| pat[i % 4]).collect();
        let mut out = [0_u8; 34];
        encode_q2_0(&trits, &[0x4400], &mut out).expect("encode");
        let mut expected = Vec::with_capacity(34);
        expected.extend_from_slice(&0x4400_u16.to_le_bytes());
        expected.extend(std::iter::repeat_n(0xA4_u8, 32));
        assert_eq!(&out[..], &expected[..]);
    }

    #[test]
    fn encode_q2_0_reproduces_golden_vector_bytes() {
        // The lane+byte-order golden vector (period-3 codes), inverted:
        // decode → encode must be the identity on the full 34 bytes.
        let code_of = |i: usize| -> u8 { (i % 3) as u8 };
        let mut bytes = Vec::with_capacity(34);
        bytes.extend_from_slice(&0x4400_u16.to_le_bytes());
        for j in 0..32_usize {
            let mut b = 0_u8;
            for lane in 0..4_usize {
                b |= code_of(4 * j + lane) << (2 * lane);
            }
            bytes.push(b);
        }
        let mut trits = [Trit::Zero; Q2_0_BLOCK];
        let mut scales = [0u16; 1];
        decode_q2_0(&bytes, &mut trits, &mut scales).expect("decode");
        let mut out = [0_u8; 34];
        encode_q2_0(&trits, &scales, &mut out).expect("encode");
        assert_eq!(&out[..], &bytes[..]);
    }

    #[test]
    fn encode_q2_0_rejects_bad_input() {
        let scales = [0x4400_u16; 2];
        let odd = [Trit::Zero; 100];
        assert_eq!(
            encode_q2_0(&odd, &scales, &mut [0; 68]),
            Err(BridgeError::BadLength)
        );
        let two_blocks = [Trit::Zero; Q2_0_BLOCK * 2];
        // out too short for two blocks.
        assert_eq!(
            encode_q2_0(&two_blocks, &scales, &mut [0; 34]),
            Err(BridgeError::TooShort)
        );
        // scales too short for two blocks (per-block scale indexing is
        // load-bearing — the patched-file path relies on it).
        assert_eq!(
            encode_q2_0(&two_blocks, &scales[..1], &mut [0; 68]),
            Err(BridgeError::TooShort)
        );
    }

    // ----- q2_0 property: encode∘decode = identity -----

    proptest! {
        #[test]
        fn prop_q2_0_round_trip(
            blocks in 1_usize..=4,
            seed in any::<u32>(),
            scales_seed in any::<u64>(),
        ) {
            // xorshift32 trit generator (house style, deterministic from seed);
            // per-block scale bits from xorshift64 so blocks carry DISTINCT
            // scales — a scale-slot mixup cannot pass silently.
            let n = blocks * Q2_0_BLOCK;
            let mut x = seed | 1;
            let trits: Vec<Trit> = (0..n)
                .map(|_| {
                    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                    match x % 3 { 0 => Trit::MinusOne, 1 => Trit::Zero, _ => Trit::One }
                })
                .collect();
            let mut s = scales_seed | 1;
            let scales: Vec<u16> = (0..blocks)
                .map(|_| {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                    (s & 0xFFFF) as u16
                })
                .collect();
            let mut buf = vec![0xFF_u8; q2_0_encoded_len(n) + 8]; // dirty: untouched bytes must not leak in
            encode_q2_0(&trits, &scales, &mut buf).unwrap();
            // Every byte the layout owns was written; the dirty tail is
            // beyond the encoding and ignored by decode.
            let mut back = vec![Trit::Zero; n];
            let mut back_scales = vec![0u16; blocks];
            decode_q2_0(&buf[..q2_0_encoded_len(n)], &mut back, &mut back_scales).unwrap();
            prop_assert_eq!(&back_scales, &scales[..]);
            prop_assert_eq!(&back, &trits[..]);
        }
    }

    // ----- fp16 scale plumbing -----

    #[test]
    fn half_known_vectors() {
        // (fp16 bits, exact f32 bits, milli)
        let cases: &[(u16, u32, i32)] = &[
            (0x0000, 0x0000_0000, 0),          // +0
            (0x8000, 0x8000_0000, 0),          // −0
            (0x3C00, 0x3F80_0000, 1000),       // 1.0
            (0x3800, 0x3F00_0000, 500),        // 0.5
            (0xC000, 0xC000_0000, -2000),      // −2.0
            (0x4400, 0x4080_0000, 4000),       // 4.0
            (0x7BFF, 0x477F_E000, 65_504_000), // max finite 65504
            (0x03FF, 0x387F_C000, 0),          // max subnormal 6.09756e-5 → milli 0
            (0x0400, 0x3880_0000, 0),          // min normal 6.10352e-5 → milli 0
            (0x7C00, 0x7F80_0000, i32::MAX),   // +inf saturates
            (0xFC00, 0xFF80_0000, i32::MIN),   // −inf saturates
            (0x7E00, 0x7FC0_0000, 0),          // NaN → 0 (documented)
        ];
        for &(h, f32_bits, milli) in cases {
            assert_eq!(half_to_f32_bits(h), f32_bits, "f32 bits for fp16 {h:#06x}");
            assert_eq!(half_to_milli(h), milli, "milli for fp16 {h:#06x}");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn half_to_milli_exhaustive_vs_f64() {
        // 2026-08-15 review: the milli conversion is hostile-file input
        // (every fp16 bit pattern can arrive in a Q1_0/Q2_0 scale) —
        // pin ALL 65 536 of them against an f64 reference instead of
        // trusting construction. Runs in milliseconds.
        for h in 0..=u16::MAX {
            let f = f32::from_bits(half_to_f32_bits(h));
            let want: i64 = if f.is_nan() {
                0
            } else if f.is_infinite() {
                i64::from(if f > 0.0 { i32::MAX } else { i32::MIN })
            } else {
                let r = (f64::from(f) * 1000.0).round();
                if r >= f64::from(i32::MAX) {
                    i64::from(i32::MAX)
                } else if r <= f64::from(i32::MIN) {
                    i64::from(i32::MIN)
                } else {
                    r as i64
                }
            };
            assert_eq!(i64::from(half_to_milli(h)), want, "fp16 bits {h:#06x}");
        }
    }

    // ----- Consumer seam: decoded trits feed the Trit substrate -----

    #[test]
    fn decoded_trits_feed_trit_substrate() {
        // Import a q1_0 tensor, materialize i16 weights at γ=125 via the
        // existing Trit API, classify back — must stay exactly on-grid.
        let mut bytes = Vec::with_capacity(18);
        bytes.extend_from_slice(&0x3C00_u16.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0xB5_u8, 16));
        let mut trits = [Trit::Zero; Q1_0_BLOCK];
        let mut scales = [0u16; 1];
        decode_q1_0(&bytes, &mut trits, &mut scales).expect("decode");

        let gamma = 125_i16;
        for t in trits {
            let w = t.to_weight(gamma);
            assert!(
                w == gamma || w == 0 || w == -gamma,
                "imported trit produced off-grid weight {w}"
            );
            assert_eq!(
                Trit::from_weight(w, gamma),
                t,
                "classification must round-trip"
            );
        }
    }

    // ----- fp16 widening property: against an integer reference -----

    proptest! {
        /// half_to_f32_bits matches a slow reference built from the f32 of
        /// the same value via bit assembly (kept integer-side; std-only).
        #[cfg(feature = "std")]
        #[test]
        fn prop_half_widening_matches_f32_reference(h in any::<u16>()) {
            let f = f32::from_bits(half_to_f32_bits(h));
            let reference = half_to_f32_via_f32(h);
            // NaN payloads are implementation-defined on conversion; ours
            // preserves the fp16 payload verbatim (IEEE-sanctioned), the
            // float-math reference collapses to the canonical quiet NaN.
            // Equality is bit-exact everywhere else.
            if reference.is_nan() {
                prop_assert!(f.is_nan());
            } else {
                prop_assert_eq!(f.to_bits(), reference.to_bits());
            }
        }
    }

    // ----- Stage 3: γ policy + wire→compute repack -----

    #[test]
    fn wire_gamma_known_vectors() {
        // With SCALE = 1000 the milli view maps through unchanged…
        assert_eq!(wire_gamma_to_substrate(24), 24); // γ = 0.024
        assert_eq!(wire_gamma_to_substrate(0), 0);
        assert_eq!(wire_gamma_to_substrate(125), 125); // Stage-1.5 gate γ
                                                       // …and saturates at the i16 rails.
        assert_eq!(wire_gamma_to_substrate(65_504_000), i16::MAX); // fp16 max finite
        assert_eq!(wire_gamma_to_substrate(-40_000), i16::MIN);
        assert_eq!(wire_gamma_to_substrate(-40_000_000), i16::MIN);
        // Negative nonsense passes through (small magnitudes) — documented.
        assert_eq!(wire_gamma_to_substrate(-5), -5);
    }

    #[test]
    fn scale_constant_is_pinned() {
        // The milli↔SCALE coupling: milli = ×1000 and SCALE = 1000 must
        // stay in lockstep or wire_gamma_to_substrate stops being the
        // identity. This test is the tripwire.
        assert_eq!(crate::synapse::SCALE, 1000);
    }

    #[test]
    fn repack_known_vector() {
        // Pattern [+1,−1,0,+1, 0,0,+1,−1] (codes 2,0,1,2, 1,1,2,0) ×16 = 128.
        let pat = [
            Trit::One,
            Trit::MinusOne,
            Trit::Zero,
            Trit::One,
            Trit::Zero,
            Trit::Zero,
            Trit::One,
            Trit::MinusOne,
        ];
        let trits: Vec<Trit> = (0..I2_S_BLOCK).map(|i| pat[i % 8]).collect();
        let mut wire = [0_u8; 64];
        encode_i2_s(&trits, 0x4000_0000, &mut wire).expect("encode");

        let mut kernel = [0_u8; 32];
        repack_i2s_to_kernel(&wire, I2_S_BLOCK, &mut kernel).expect("repack");

        // Sequential unpack must reproduce the trits in order.
        for (i, &t) in trits.iter().enumerate() {
            let code = (kernel[i / 4] >> (2 * (i % 4))) & 0x03;
            let got = code_to_trit(code).expect("no code 3 in encode output");
            assert_eq!(got, t, "element {i} wrong after repack");
        }
    }

    #[test]
    fn repack_rejects_bad_input() {
        let mut out = [0_u8; 32];
        // n not a multiple of 128 (the wire side's rule).
        assert_eq!(
            repack_i2s_to_kernel(&[0; 64], 64, &mut out),
            Err(BridgeError::BadLength)
        );
        // Wire buffer too short for n = 128.
        assert_eq!(
            repack_i2s_to_kernel(&[0; 12], 128, &mut out),
            Err(BridgeError::TooShort)
        );
        // Out buffer too short.
        let mut tiny = [0_u8; 4];
        assert_eq!(
            repack_i2s_to_kernel(&[0; 64], 128, &mut tiny),
            Err(BridgeError::TooShort)
        );
        // Code 3 anywhere in the live lanes.
        let evil = [0xFF_u8; 64];
        let mut sink = [0_u8; 32];
        assert_eq!(
            repack_i2s_to_kernel(&evil, 128, &mut sink),
            Err(BridgeError::UnsupportedCode)
        );
    }

    proptest! {
        /// encode → repack → sequential-unpack reproduces the original
        /// trits bit-exactly for any 128-aligned tensor.
        #[test]
        fn prop_repack_round_trip(
            blocks in 1_usize..=3,
            seed in any::<u32>(),
        ) {
            let n = blocks * I2_S_BLOCK;
            let mut x = seed | 1;
            let trits: Vec<Trit> = (0..n)
                .map(|_| {
                    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                    match x % 3 { 0 => Trit::MinusOne, 1 => Trit::Zero, _ => Trit::One }
                })
                .collect();
            let mut wire = vec![0_u8; i2_s_encoded_len(n)];
            encode_i2_s(&trits, 0, &mut wire).unwrap();
            let mut kernel = vec![0_u8; n / 4];
            repack_i2s_to_kernel(&wire, n, &mut kernel).unwrap();
            for (i, &t) in trits.iter().enumerate() {
                let code = (kernel[i / 4] >> (2 * (i % 4))) & 0x03;
                prop_assert_eq!(code_to_trit(code).unwrap(), t, "element {}", i);
            }
        }
    }

    #[cfg(feature = "std")]
    fn half_to_f32_via_f32(h: u16) -> f32 {
        // Reference via float math (std only, test only): decode fp16 fields
        // and rebuild the value, then let f32 round — exact for all fp16.
        let sign = if (h >> 15) & 1 == 1 { -1.0_f32 } else { 1.0 };
        let exp = i32::from((h >> 10) & 0x1F);
        let mant = f32::from(h & 0x03FF);
        if exp == 0 {
            sign * mant * (2.0_f32).powi(-24)
        } else if exp == 0x1F {
            if mant == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        } else {
            sign * (1.0 + mant / 1024.0) * (2.0_f32).powi(exp - 15)
        }
    }
}
