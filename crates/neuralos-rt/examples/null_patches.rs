//! Session I ladder rung 3 — THE NULL PATCHES: random + position-shuffle
//! controls for the steers claim.
//!
//! Pre-registered (ISA sH2 registration): the in-vivo steers signature
//! (p3 11/12 + p2 4/12 flips on H2) is claimed as adaptation-content only
//! if the null family does NOT reproduce it. This example builds the null
//! patches through the SAME surgery machinery (S2-asserted), from the H2
//! run's recorded census + changed-cell set:
//!
//! - **random-<seed>** (×10): census-matched, region-matched i.i.d. —
//!   the same number of changed cells per bucket-transition class as the
//!   real patch, placed uniformly at random within the 512×512 slice.
//!   Tests "any perturbation of this magnitude/shape."
//! - **shuffle-<seed>** (×3): the REAL changed-cell set (positions AND
//!   per-row clustering preserved), with the assigned new-values permuted
//!   among the changed positions. Tests content-vs-placement: if the
//!   arrangement of changes matters, shuffling their values degrades the
//!   effect.
//!
//! The H2 patch numbers (census + per-class counts) come from
//! evidence/session-h2/run.log, pinned as consts — the nulls are built
//! from the RECORDED state, never re-run.
//!
//! Usage: `cargo run -p neuralos-rt --release --example null_patches --
//! [model.gguf]` — writes models/null-<kind>-<seed>.gguf for the judge.

use neuralos_rt::{GgufFile, GGML_TYPE_Q2_0};
use neuralos_snn::{decode_q2_0, encode_q2_0, Trit};

const N: usize = 512;
const TENSOR: &str = "blk.0.attn_q.weight";
const MODEL_COLS: usize = 2560;
const ROW_BYTES: usize = (MODEL_COLS / 128) * 34;

/// H2 census (evidence/session-h2/run.log): bucket-transition counts.
/// −1→0: 193,198 · 0→−1: 167,171 · 0→+1: 375,604 · +1→0: 376,798.
const C_M0: u64 = 193_198; // −1→0
const C_0M: u64 = 167_171; // 0→−1
const C_0P: u64 = 375_604; // 0→+1
const C_P0: u64 = 376_798; // +1→0
/// H2 total changed cells (Hamming 87,119 — cells, not flips).
const H2_CELLS: u64 = 87_119;

/// Deterministic xorshift64 (house style).
fn xorshift64(state: u64) -> u64 {
    let mut x = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Ternary-Bonsai-4B-Q2_0.gguf".into());
    println!("=== Session I rung 3: null patches (census-matched random ×10 + position-shuffle ×3) ===");
    println!("source  : {path}");
    println!(
        "census  : −1→0 {C_M0} · 0→−1 {C_0M} · 0→+1 {C_0P} · +1→0 {C_P0}  (H2 recorded; cells {H2_CELLS})"
    );

    // ----- Decode the ORIGINAL slice (the nulls' substrate) -----
    let src: Vec<Trit> = {
        let buf = std::fs::read(&path).unwrap_or_else(|e| {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(1);
        });
        let f = GgufFile::parse(&buf).expect("GGUF container must parse");
        let info = f
            .tensors
            .iter()
            .find(|t| t.name == TENSOR)
            .unwrap_or_else(|| panic!("tensor {TENSOR} not found"));
        assert_eq!(info.ty, GGML_TYPE_Q2_0);
        let data = f.tensor_data(info).expect("slice in bounds");
        let mut out = Vec::with_capacity(N * N);
        let mut row_trits = vec![Trit::Zero; N];
        let mut scales = [0u16; N / 128];
        for r in 0..N {
            decode_q2_0(
                &data[r * ROW_BYTES..r * ROW_BYTES + N / 128 * 34],
                &mut row_trits,
                &mut scales,
            )
            .expect("decode");
            out.extend_from_slice(&row_trits);
        }
        out
    };

    // ----- The surgery (S2-asserted, identical unit to hybrid_invivo) -----
    let do_surgery = |patched: &[Trit], out: &str| {
        let mut buf = std::fs::read(&path).expect("re-read base");
        let f2 = GgufFile::parse(&buf).expect("re-parse");
        let info2 = f2
            .tensors
            .iter()
            .find(|t| t.name == TENSOR)
            .unwrap_or_else(|| panic!("tensor {TENSOR}"));
        let abs = (f2.data_start + info2.offset) as usize;
        const CHUNK: usize = 4 * 34;
        let mut row_orig = vec![Trit::Zero; N];
        let mut scales = [0u16; N / 128];
        let mut enc = [0u8; CHUNK];
        let mut code_changed = 0u64;
        for r in 0..N {
            let off = abs + r * ROW_BYTES;
            decode_q2_0(&buf[off..off + CHUNK], &mut row_orig, &mut scales)
                .expect("orig decodes");
            encode_q2_0(&patched[r * N..(r + 1) * N], &scales, &mut enc).expect("encode");
            for (b, (&old, &new)) in buf[off..off + CHUNK].iter().zip(enc.iter()).enumerate() {
                if old != new && b % 34 >= 2 {
                    code_changed += 1;
                }
            }
            buf[off..off + CHUNK].copy_from_slice(&enc);
        }
        std::fs::write(out, &buf).expect("write");
        // S2 re-read — every null, per the registration.
        let check = std::fs::read(out).expect("re-read");
        let f3 = GgufFile::parse(&check).expect("parse post-write");
        let info3 = f3
            .tensors
            .iter()
            .find(|t| t.name == TENSOR)
            .unwrap_or_else(|| panic!("tensor {TENSOR}"));
        let abs3 = (f3.data_start + info3.offset) as usize;
        let mut rt = vec![Trit::Zero; N];
        let mut sc = [0u16; N / 128];
        let mut mism = 0u64;
        for r in 0..N {
            let off = abs3 + r * ROW_BYTES;
            decode_q2_0(&check[off..off + CHUNK], &mut rt, &mut sc).expect("decode post");
            for c in 0..N {
                if rt[c] != patched[r * N + c] {
                    mism += 1;
                }
            }
        }
        assert_eq!(mism, 0, "S2: post-write decode != patch");
        code_changed
    };

    // ----- RANDOM nulls: place each transition class uniformly in the
    // slice, ONLY on cells whose source value matches the from-class -----
    for seed in 1..=10u64 {
        let mut rng = 0xDEAD_BEEF_0000_0000_u64 ^ seed;
        let mut patched = src.clone();
        // (from, to, count) — census classes from the H2 record.
        for (from, to, count) in [
            (Trit::MinusOne, Trit::Zero, C_M0),
            (Trit::Zero, Trit::MinusOne, C_0M),
            (Trit::Zero, Trit::One, C_0P),
            (Trit::One, Trit::Zero, C_P0),
        ] {
            // Candidates: cells whose SOURCE value is the from-class.
            // Classes are disjoint by construction (a −1 cell can only
            // serve −1→0; a 0 cell serves 0→∓1 — but 0→−1 and 0→+1 share
            // the same source pool, so the second class draws from the
            // pool MINUS the first class's take: filter on `patched`,
            // not `src`, for the shared-source classes.
            let mut idxs: Vec<usize> = (0..N * N)
                .filter(|&i| patched[i] == from)
                .collect();
            // Fisher-Yates with the run-seeded rng, take `count`.
            for i in (1..idxs.len()).rev() {
                rng = xorshift64(rng);
                let j = (rng % (i as u64 + 1)) as usize;
                idxs.swap(i, j);
            }
            let take = count.min(idxs.len() as u64) as usize;
            for &cell in idxs.iter().take(take) {
                patched[cell] = to;
            }
        }
        let changed = patched.iter().zip(&src).filter(|(a, b)| a != b).count() as u64;
        let out = format!("models/null-random-{seed}.gguf");
        let cb = do_surgery(&patched, &out);
        println!("random-{seed}: {changed} cells · code {cb} · S2 clean → {out}");
    }

    println!("(position-shuffle rung: needs the H2 changed-cell set —");
    println!(" built in the ladder harness once H2's final-trits dump exists;");
    println!(" the random ×10 above are the primary null family)");
}
