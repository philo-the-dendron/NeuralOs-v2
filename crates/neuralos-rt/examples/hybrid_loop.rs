//! Session E — THE LOOP-CLOSER: pretrained ternary LLM weights, imported
//! into the spiking substrate, adapted under local STDP, exported BACK as
//! Q2_0 bytes into a patched GGUF copy that foreign tooling can run.
//!
//! Phase 1 is the D-2 hybrid experiment VERBATIM (the shared gate phase,
//! `neuralos_rt::harness::run_gate_phase` since R4(iii); frozen original
//! at tag `examples-pre-extraction`): decode → G1 import integrity → G2
//! spiking fidelity → G3 selective adaptation. Its recorded verdict
//! numbers are asserted as preconditions before any byte is written —
//! the surgery must operate on exactly the recorded adapted state, or it
//! does not run.
//!
//! Phase 2 is the loop: the adapted 512×512 slice of
//! `blk.0.attn_q.weight` is re-encoded through `encode_q2_0` (session E's
//! export codec — the byte-level inverse of `decode_q2_0`) with the file's
//! ORIGINAL per-block fp16 scale bits passed through untouched (recorded
//! decision: the substrate adapted STRUCTURE at γ=125; magnitudes stay the
//! model's own), and spliced into a copy of the GGUF as 512 disjoint
//! 136-byte chunks — the first 4 blocks (512 of 2560 input cols) of each of
//! the first 512 output rows.
//!
//! # Surgery gates (stated up front)
//!
//! - **S1 CONTAINMENT** — no byte outside the declared chunks may differ
//!   between original and patched; inside the chunks, ONLY code bytes may
//!   differ (scale bytes pass through bit-exactly, asserted).
//! - **S2 DISK ROUND-TRIP** — the patched file re-read from disk parses as
//!   GGUF, and its first 512×512 trits decode back to EXACTLY the adapted
//!   slice. What was written is what loads.
//!
//! The behavior gate (does the foreign runtime's output change?) is NOT
//! this example's — it belongs to the fork judge (G3-of-loop), run
//! separately on the patched file with identical greedy-forced flags.
//!
//! Deterministic end to end (D-2's seeds + pure encode/decode arithmetic):
//! two runs produce byte-identical patched files.
//!
//! **Control mode** (3rd arg `control`): the surgery runs with the
//! UNADAPTED source trits instead of the adapted slice — same decode,
//! same encode, same splice, same gates. The written file must be
//! BYTE-IDENTICAL to the original (asserted against a fresh read):
//! the codec+surgery path is provably transparent, so any behavioral
//! delta measured on the adapted file is attributable to the STDP
//! trits and nothing else. This is the Stage-0 attribution control.
//!
//! Usage: `cargo run -p neuralos-rt --release --example hybrid_loop --
//! [src.gguf] [dst.gguf]` (defaults `models/Ternary-Bonsai-4B-Q2_0.gguf`,
//! `models/Ternary-Bonsai-4B-Q2_0-loop.gguf`). Memory box: holds src +
//! patched buffers simultaneously (~2.2 GB) after the nets drop; peak
//! printed.
//!
//! Session E gates map: G1 = encoder unit/property tests in
//! `neuralos-snn`; G2 = S1+S2 here; G3 = fork delta on the patched file;
//! G4 = double-run determinism (both sides).

#![allow(non_snake_case)] // phase-2 locals keep the frozen original's const names

use neuralos_rt::harness::{
    decode_slice, peak_rss_mb, run_gate_phase, splice_trits, tensor_abs, verify_disk_roundtrip,
    ExperimentParams,
};
use neuralos_rt::GgufFile;
use neuralos_snn::Trit;

// ----- D-2 pinned state (session F re-run on the live-wire substrate —
// the surgery runs only on the exact recorded adapted state). The pre-fix
// dead-wire state (35,157 ×3 · 321,571 · 57,005 · 16,183,885) is ISA
// history; these are THREE DISTINCT totals now — weights shape firing.
// These are VERDICT PINS (assertion values, not tuning knobs) — the only
// consts this example carries; everything behavioral lives in
// ExperimentParams since R4(iii). -----
const D2_SPIKES_IMP: u64 = 35_115;
const D2_SPIKES_CTL: u64 = 35_136;
const D2_SPIKES_ZERO: u64 = 35_157;
const D2_FLIPS: u64 = 708_029; // G3 bucket flips
const D2_HAMMING: u64 = 64_877; // G3 changed buckets (0.2480 of 261,632)
const D2_PLASTICITY_EVENTS: u64 = 18_817_891;
/// Intra mean Δ — print precision of the pinned log line (±5e-5).
const D2_INTRA_DELTA: f64 = 0.1075;

/// Phase-2 budget: the loop legitimately holds src + patched file buffers
/// simultaneously (~2.2 GB) — the D-2 single-buffer box does not apply.
/// (Print-only companion to `ExperimentParams::rss_budget_mb`.)
const LOOP_RSS_BUDGET_MB: u64 = 2560;

fn main() {
    let t0 = std::time::Instant::now();
    let p = ExperimentParams::default();
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Ternary-Bonsai-4B-Q2_0.gguf".into());
    let out_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/Ternary-Bonsai-4B-Q2_0-loop.gguf".into());
    let control_mode = std::env::args()
        .nth(3)
        .map(|a| a == "control")
        .unwrap_or(false);
    println!(
        "=== Session E: the loop-closer — Bonsai Q2_0 → Trit → SNN → STDP → Q2_0 → patched GGUF ==="
    );
    println!("src     : {path}");
    println!("dst     : {out_path}");
    if control_mode {
        println!(
            "mode    : CONTROL — export carries the UNADAPTED source trits (attribution control)"
        );
    }
    println!(
        "host    : {}, first {}×{} slice (col j = pre → row i = post, matvec dataflow)",
        p.tensor, p.n, p.n
    );
    println!("substrate: γ={} (the proven 1.5x constant; LLM fp16 block scales unused — recorded), full-minus-diagonal density, 1.5c drive verbatim", p.gamma);
    println!("phase 1 : D-2 hybrid experiment re-run — recorded numbers asserted as preconditions");
    println!("phase 2 : export via encode_q2_0 + chunked GGUF surgery (S1 containment, S2 disk round-trip)");
    println!();

    // ----- Decode (scoped: file buffer drops before any network is built) -----
    let src = decode_slice(&path, &p);
    println!(
        "decode  : {} trits from real Q2_0 bytes (peak RSS so far is the file buffer)",
        src.len()
    );

    // ----- Phase 1: the D-2 gate phase, verbatim (prints G1–G3 + verdict) -----
    let o = run_gate_phase(&src, &p, true);
    if !o.pass {
        println!("surgery NOT run (phase-1 gates failed)");
        std::process::exit(1);
    }

    // ----- D-2 recorded numbers: asserted BEFORE any write -----
    println!();
    println!("--- D-2 preconditions (the surgery operates only on the recorded adapted state) ---");
    assert_eq!(
        o.imported_stats.total_spikes, D2_SPIKES_IMP,
        "D-2 G2 imported spikes"
    );
    assert_eq!(
        o.control_stats.total_spikes, D2_SPIKES_CTL,
        "D-2 G2 control spikes"
    );
    assert_eq!(
        o.zero_stats.total_spikes, D2_SPIKES_ZERO,
        "D-2 G2 zero-w spikes"
    );
    assert_eq!(
        o.hybrid.plasticity_events, D2_PLASTICITY_EVENTS,
        "D-2 plasticity events"
    );
    assert_eq!(o.hybrid.flips, D2_FLIPS, "D-2 bucket flips");
    assert_eq!(o.hamming, D2_HAMMING, "D-2 Hamming count");
    assert!(
        (o.d_intra - D2_INTRA_DELTA).abs() < 5e-5,
        "D-2 intra mean Δ was +0.1075 (clamp-rectified; got {})",
        o.d_intra
    );
    assert!(
        o.d_inter.abs() < f64::EPSILON,
        "D-2 inter mean Δ was exactly 0.0000 (schedule geometry; got {})",
        o.d_inter
    );
    println!("  spikes {D2_SPIKES_IMP}/{D2_SPIKES_CTL}/{D2_SPIKES_ZERO} · events {D2_PLASTICITY_EVENTS} · flips {D2_FLIPS} · Hamming {D2_HAMMING} · intra Δ +0.1075 (clamp-rectified) — all reproduced");

    // ----- Phase 2: export + surgery -----
    let N = p.n;
    let ROW_BYTES = p.row_bytes();
    let CHUNK_BYTES = p.chunk_bytes();
    let TENSOR_BYTES = p.tensor_bytes();
    println!();
    println!("--- LOOP: EXPORT + SURGERY (encode_q2_0 → patched GGUF copy) ---");
    // Reconstruct the adapted row-major slice: the diagonal never carried a
    // synapse (full-minus-diagonal build) → keeps its source trit; every
    // off-diagonal takes the final bucket of its synapse. The k-walk is the
    // exact inverse of build order (j outer, i inner, i≠j).
    let mut adapted = vec![Trit::Zero; N * N];
    let mut k = 0usize;
    for j in 0..N {
        for i in 0..N {
            if i != j {
                adapted[i * N + j] = o.hybrid.final_trits[k];
                k += 1;
            }
        }
    }
    // The diagonal carried no synapse (full-minus-diagonal build) → the
    // file's own trit stays (first run left it at Zero and the Hamming
    // cross-assert caught 295 phantom changes — the assert stays).
    for i in 0..N {
        adapted[i * N + i] = src[i * N + i];
    }
    assert_eq!(
        k,
        o.hybrid.final_trits.len(),
        "synapse walk covered exactly"
    );
    let changed_cells = adapted.iter().zip(&src).filter(|(a, b)| a != b).count() as u64;
    assert_eq!(
        changed_cells, o.hamming,
        "cell deltas == D-2 Hamming (diagonal untouched by construction)"
    );
    println!(
        "  adapted slice: {N}×{N}, {changed_cells} cells changed (= Hamming), {} retained",
        (N * N) as u64 - changed_cells
    );
    // Control mode exports the UNADAPTED source trits — same machinery.
    let export_trits: &[Trit] = if control_mode { &src } else { &adapted };

    // Re-read the file into the mutable patch buffer; compute the tensor's
    // absolute window from data_start + offset, sizes from DIMS (never from
    // inferred slice ends).
    let mut buf: Vec<u8> = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot re-read {path}: {e}");
        std::process::exit(1);
    });
    let f2 = GgufFile::parse(&buf).expect("re-parse");
    let abs = tensor_abs(&f2, &p);
    assert!(abs + TENSOR_BYTES <= buf.len(), "tensor window inside file");
    println!("  tensor window: abs {abs} + {TENSOR_BYTES} B (dims-derived, not slice-inferred)");

    // Splice via the shared surgery unit (R4-extracted): re-encode the
    // first 4 blocks from the EXPORT trits (adapted, or source in
    // control mode) with the ORIGINAL fp16 scale bits (magnitudes stay
    // the model's own — recorded decision). `expect_src` carries the
    // loop's chunk==slice codec-transparency assert; scales pass
    // through bit-exactly (asserted in the unit).
    let (code_bytes_changed, scale_bytes_changed) =
        splice_trits(&mut buf, export_trits, Some(src.as_slice()), &p);
    println!(
        "  spliced: {N} chunks × {CHUNK_BYTES} B = {} B declared region",
        N * CHUNK_BYTES
    );
    println!(
        "  bytes changed: code {code_bytes_changed} of {} · scale {scale_bytes_changed} of {} (must be 0)",
        N * (CHUNK_BYTES - 8),
        N * 8
    );
    if control_mode {
        assert_eq!(
            code_bytes_changed, 0,
            "control: encode(src) must reproduce every original byte"
        );
    }

    // S1 CONTAINMENT: compare against a fresh read of the original — every
    // differing byte must sit inside a declared chunk.
    drop(o.imported);
    drop(o.control);
    drop(o.zeronet);
    let orig = std::fs::read(&path).expect("re-read original for containment");
    assert_eq!(orig.len(), buf.len());
    let mut inside = 0u64;
    let mut outside = 0u64;
    for (pos, (a, b)) in buf.iter().zip(orig.iter()).enumerate() {
        if a != b {
            let rel = pos.wrapping_sub(abs);
            let in_chunk = pos >= abs
                && rel < TENSOR_BYTES
                && rel / ROW_BYTES < N
                && rel % ROW_BYTES < CHUNK_BYTES;
            if in_chunk {
                inside += 1;
            } else {
                outside += 1;
            }
        }
    }
    println!(
        "  S1 containment: {inside} differing bytes, ALL inside declared chunks; outside = {outside}"
    );
    assert_eq!(outside, 0, "bytes changed outside the declared chunks");
    if control_mode {
        assert_eq!(inside, 0, "control: the whole file must be byte-identical");
    } else {
        assert_eq!(
            inside, code_bytes_changed,
            "inside == code bytes (scales untouched)"
        );
    }
    drop(orig);

    // Write the patched copy, then S2 via the shared unit: re-read
    // from disk and prove what was written is what loads.
    std::fs::write(&out_path, &buf).unwrap_or_else(|e| {
        eprintln!("cannot write {out_path}: {e}");
        std::process::exit(1);
    });
    println!("  wrote: {out_path} ({} B)", buf.len());
    drop(buf);
    verify_disk_roundtrip(&out_path, export_trits, &p);
    println!(
        "  S2 disk round-trip: patched file parses; {} trits decoded from disk vs export trits — 0 mismatches",
        N * N
    );
    if control_mode {
        // The full-file money assert: control output == original, byte for
        // byte, re-read from disk after the write.
        let check = std::fs::read(&out_path).expect("re-read written control");
        let orig_full = std::fs::read(&path).expect("re-read original for control identity");
        assert!(
            check == orig_full,
            "control file must be byte-identical to the original"
        );
        println!("  CONTROL IDENTITY: written file == original file, byte for byte");
    }

    println!();
    println!(
        "LOOP EXPORT: CLEAN — adapted slice exported bit-exactly, containment held, disk round-trip exact"
    );
    println!(
        "next: fork judge on {out_path} (greedy-forced, NEURALOS_DUMP on, double-run) vs baseline"
    );
    println!(
        "wall {:.1}s   peak RSS {} MB (loop budget {LOOP_RSS_BUDGET_MB}; phase-1 box {} was single-buffer)",
        t0.elapsed().as_secs_f64(),
        peak_rss_mb(),
        p.rss_budget_mb
    );
}
