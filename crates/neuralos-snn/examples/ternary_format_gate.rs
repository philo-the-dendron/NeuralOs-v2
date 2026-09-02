//! Ternary-bridge gate — Stage 2: the format bridge.
//!
//! Question: can we round-trip a ternary tensor through the byte layouts of
//! the two ecosystems that actually ship ternary models — BitNet `i2_s`
//! (encode + decode) and Prism `q1_0` / `q2_0` (import)?
//!
//! Checks, each pinned to a test vector derived from the reference sources
//! (microsoft/BitNet `quantize_to_i2_s`; PrismML-Eng/llama.cpp
//! `block_q1_0` / `block_q2_0`):
//!
//!   * **A — i2_s round-trip:** deterministic 256-trit tensor (xorshift32
//!     generator) + a scale bit pattern → encode → decode → trits and scale
//!     bits must be bit-exact.
//!   * **B — q1_0 import:** a two-block `block_q1_0` byte stream (scales
//!     fp16 1.0 and 2.0, distinct sign patterns) must decode to the exact
//!     expected trits, per-block scales included.
//!   * **C — q2_0 import:** a `block_q2_0` stream (scale fp16 4.0, code
//!     bytes 0xA4) must decode exactly; and a code-3 byte must be *rejected*
//!     loudly, not clamped.
//!
//! The lines this prints ARE the gate evidence. Gate = YES iff all three
//! checks hold with zero tolerance for mismatch. The YES/NO call is recorded
//! in the commit message and `docs/VISION.md`.

use neuralos_snn::{
    decode_i2_s, decode_q1_0, decode_q2_0, encode_i2_s, half_to_f32_bits, half_to_milli,
    BridgeError, Trit,
};

const I2_S_SCALE_BITS: u32 = 0x4000_0000; // f32 2.0 — BitNet-Round γ carried verbatim

/// Deterministic ternary generator (xorshift32 → mod 3).
fn trit_stream(seed: u32, n: usize) -> Vec<Trit> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            match x % 3 {
                0 => Trit::MinusOne,
                1 => Trit::Zero,
                _ => Trit::One,
            }
        })
        .collect()
}

fn check_a_i2_s_round_trip() -> Result<usize, String> {
    let trits = trit_stream(0xC0FFEE, 256);
    let n = trits.len();
    let mut encoded = vec![0xFF_u8; n / 4 + 32 + 8]; // poisoned tail: codec must not touch it
    let written = encode_i2_s(&trits, I2_S_SCALE_BITS, &mut encoded)
        .map_err(|e| format!("encode failed: {e}"))?;
    if written != n / 4 + 32 {
        return Err(format!("written {written} != expected {}", n / 4 + 32));
    }
    let mut decoded = vec![Trit::Zero; n];
    let scale_back = decode_i2_s(&encoded[..written], &mut decoded)
        .map_err(|e| format!("decode failed: {e}"))?;
    let mismatches = trits.iter().zip(&decoded).filter(|(a, b)| a != b).count();
    if mismatches != 0 || scale_back != I2_S_SCALE_BITS {
        return Err(format!(
            "{mismatches} trit mismatches, scale {scale_back:#010x} vs {I2_S_SCALE_BITS:#010x}"
        ));
    }
    println!("  A. i2_s round-trip     : 256/256 trits bit-exact, scale bits {I2_S_SCALE_BITS:#010x} preserved");
    Ok(n)
}

fn check_b_q1_0_import() -> Result<usize, String> {
    // Two 128-weight blocks: scale fp16 1.0 (0x3C00) with sign bytes 0xB5,
    // scale fp16 2.0 (0x4000) with sign bytes 0x00 (all −γ).
    let mut bytes = Vec::with_capacity(36);
    bytes.extend_from_slice(&0x3C00_u16.to_le_bytes());
    bytes.extend(std::iter::repeat_n(0xB5_u8, 16));
    bytes.extend_from_slice(&0x4000_u16.to_le_bytes());
    bytes.extend(std::iter::repeat_n(0x00_u8, 16));

    let mut trits = vec![Trit::Zero; 256];
    let mut scales = [0_u16; 2];
    decode_q1_0(&bytes, &mut trits, &mut scales).map_err(|e| format!("decode failed: {e}"))?;

    // Block 0 pattern: 0xB5 = 0b1011_0101 → LSB-first bits 1,0,1,0,1,1,0,1.
    let pat0 = [
        Trit::One,
        Trit::MinusOne,
        Trit::One,
        Trit::MinusOne,
        Trit::One,
        Trit::One,
        Trit::MinusOne,
        Trit::One,
    ];
    let bad0 = (0..128).filter(|&i| trits[i] != pat0[i % 8]).count();
    // Block 1: all bits clear → all −γ.
    let bad1 = (128..256).filter(|&i| trits[i] != Trit::MinusOne).count();
    let scales_ok = scales[0] == 0x3C00 && scales[1] == 0x4000;
    if bad0 != 0 || bad1 != 0 || !scales_ok {
        return Err(format!(
            "block0 {bad0} mismatches, block1 {bad1} mismatches, scales {scales:?}"
        ));
    }
    println!(
        "  B. q1_0 import         : 256/256 sign bits exact, per-block scales [1.0, 2.0] fp16 (milli: [{}, {}])",
        half_to_milli(scales[0]),
        half_to_milli(scales[1])
    );
    Ok(256)
}

fn check_c_q2_0_import() -> Result<usize, String> {
    // One 128-weight block (session-D re-pin: QK2_0 = 128, 34 B/block):
    // scale fp16 4.0 (0x4400), code bytes 0xA4 = 0b10_10_01_00 →
    // LSB-first codes 00,01,10,10 → [−1, 0, +1, +1] ×32.
    let mut bytes = Vec::with_capacity(34);
    bytes.extend_from_slice(&0x4400_u16.to_le_bytes());
    bytes.extend(std::iter::repeat_n(0xA4_u8, 32));

    let mut trits = vec![Trit::Zero; 128];
    let mut scales = [0_u16; 1];
    decode_q2_0(&bytes, &mut trits, &mut scales).map_err(|e| format!("decode failed: {e}"))?;

    let pat = [Trit::MinusOne, Trit::Zero, Trit::One, Trit::One];
    let bad = (0..128).filter(|&i| trits[i] != pat[i % 4]).count();
    if bad != 0 || scales[0] != 0x4400 {
        return Err(format!("{bad} mismatches, scale {:#06x}", scales[0]));
    }

    // Code 3 must be a loud error, never a silent clamp.
    let mut evil = vec![0_u8; 34];
    evil[0..2].copy_from_slice(&0x4400_u16.to_le_bytes());
    evil[2] = 0x03;
    let mut sink = vec![Trit::Zero; 128];
    let mut sink_scales = [0_u16; 1];
    let rejected = decode_q2_0(&evil, &mut sink, &mut sink_scales);
    if rejected != Err(BridgeError::UnsupportedCode) {
        return Err(format!("code 3 was not rejected: {rejected:?}"));
    }
    println!("  C. q2_0 import         : 128/128 codes exact (scale fp16 4.0), code-3 input rejected loudly");
    Ok(128)
}

fn main() {
    println!("=== Stage 2 gate — the format bridge ===");
    println!("  layouts pinned to: microsoft/BitNet quantize_to_i2_s, PrismML-Eng/llama.cpp block_q1_0/block_q2_0");
    println!(
        "  fp16 1.0 -> f32 bits {:#010x} (integer widening check)",
        half_to_f32_bits(0x3C00)
    );

    let results = [
        check_a_i2_s_round_trip(),
        check_b_q1_0_import(),
        check_c_q2_0_import(),
    ];

    println!();
    let failed: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
    if failed.is_empty() {
        println!("STAGE 2 GATE: YES — ternary tensor round-trips bit-exactly; both ecosystems import clean");
    } else {
        for f in &failed {
            eprintln!("FAILED: {f}");
        }
        println!("STAGE 2 GATE: NO");
        std::process::exit(1);
    }
}
