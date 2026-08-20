//! Session I ladder rung 3 v2 — THE NULL FAMILY: dose-matched primary ×10
//! + position-shuffle ×3, both derived from the H2 TERMINAL PATCH.
//!
//! Pre-registered (ISA sI amendment 2026-08-20, committed before the stress
//! chain's results were read): the PRIMARY family carries EXACTLY the H2
//! patch's changed-cell count (87,119), composition from the H2 terminal
//! diff (decode the H2 patched GGUF's slice vs the original). The earlier
//! census-transition family (~178k cells) is the STRESS ARM, report-only.
//!
//! - **dose-<seed>** (×10, seeds 1–10; escalation seeds 11–20
//!   pre-generated): exactly `H2_CELLS` changed cells, per-class
//!   composition matching the H2 terminal diff, placed uniformly over
//!   cells whose source value matches the from-class (seeded FY shuffle).
//! - **shuffle-<seed>** (×3): the REAL changed-cell set (positions and
//!   per-row clustering preserved); the assigned new-VALUES permuted
//!   among the changed positions, CONDITIONED on new ≠ source (re-draw
//!   on collision); a shrunk dose ABORTS loudly, never silently.
//!   Pre-stated bar: "reproduces the signature" = ≥10/12 p3 flips OR the
//!   p3 step-1 knife-edge crossing.
//!
//! Usage: `cargo run -p neuralos-rt --release --example null_patches --
//! [orig.gguf] [h2-patched.gguf]` — writes models/null-dose-<s>.gguf and
//! models/null-shuffle-<s>.gguf, all S2-asserted.

use neuralos_rt::{GgufFile, GGML_TYPE_Q2_0};
use neuralos_snn::{decode_q2_0, encode_q2_0, Trit};

const N: usize = 512;
const TENSOR: &str = "blk.0.attn_q.weight";
const MODEL_COLS: usize = 2560;
const ROW_BYTES: usize = (MODEL_COLS / 128) * 34;

fn xorshift64(state: u64) -> u64 {
    let mut x = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Decode the 512×512 slice from a GGUF (original or patched).
fn decode_slice(path: &str) -> Vec<Trit> {
    let buf = std::fs::read(path).unwrap_or_else(|e| {
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
    assert_eq!(data.len(), 4096 * ROW_BYTES);
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
}

fn main() {
    let orig_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Ternary-Bonsai-4B-Q2_0.gguf".into());
    let h2_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/Ternary-Bonsai-4B-Q2_0-invivo.gguf".into());
    println!("=== Null family v2: dose-matched ×10 + shuffle ×3 (from the H2 terminal patch) ===");
    println!("orig    : {orig_path}");
    println!("H2      : {h2_path}");

    let src = decode_slice(&orig_path);
    let h2 = decode_slice(&h2_path);

    // ----- The H2 terminal diff: exact changed-cell set + composition -----
    let changed_idx: Vec<usize> = (0..N * N).filter(|&i| h2[i] != src[i]).collect();
    let h2_cells = changed_idx.len();
    // per (from,to) class composition of the terminal diff
    let mut comp: [u64; 9] = [0; 9]; // (from*3 + to), values 0..2 minus/zero/one
    let tix = |t: Trit| -> usize {
        match t {
            Trit::MinusOne => 0,
            Trit::Zero => 1,
            Trit::One => 2,
        }
    };
    for &i in &changed_idx {
        comp[tix(src[i]) * 3 + tix(h2[i])] += 1;
    }
    println!("H2 diff : {h2_cells} changed cells (pre-registered 87,119)");
    assert_eq!(h2_cells, 87_119, "the H2 terminal diff must carry exactly the recorded cell count");
    println!(
        "comp    : −1→0 {} · 0→−1 {} · 0→+1 {} · +1→0 {} · (others {})",
        comp[1],
        comp[3],
        comp[5],
        comp[7],
        comp.iter().enumerate().filter(|(k, _)| {
            !matches!(*k, 1 | 3 | 5 | 7) // off-diagonal minus identity mappings
        }).count()
    );

    // ----- The surgery unit (S2-asserted, identical to hybrid_invivo) -----
    let do_surgery = |patched: &[Trit], out: &str| {
        let mut buf = std::fs::read(&orig_path).expect("re-read base");
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
        for r in 0..N {
            let off = abs + r * ROW_BYTES;
            decode_q2_0(&buf[off..off + CHUNK], &mut row_orig, &mut scales)
                .expect("orig decodes");
            encode_q2_0(&patched[r * N..(r + 1) * N], &scales, &mut enc).expect("encode");
            buf[off..off + CHUNK].copy_from_slice(&enc);
        }
        std::fs::write(out, &buf).expect("write");
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
    };

    // ----- PRIMARY: dose-matched ×10 (exact H2 composition, uniform cells) -----
    // class list: ALL nonzero (from,to) classes of the terminal diff —
    // the exact-dose assert caught that crossovers (−1→+1 etc., ~21k
    // cells) exist beyond the four census classes; dose-matching must
    // carry the FULL composition or the dose silently shrinks.
    let tr_of = |i: usize| -> Trit {
        match i {
            0 => Trit::MinusOne,
            1 => Trit::Zero,
            _ => Trit::One,
        }
    };
    let classes: Vec<(Trit, Trit, u64)> = (0..9)
        .filter(|&k| comp[k] > 0 && k / 3 != k % 3)
        .map(|k| (tr_of(k / 3), tr_of(k % 3), comp[k]))
        .collect();
    println!("classes : {} (full composition incl. crossovers)", classes.len());
    for seed in 1..=10u64 {
        let mut rng = 0xD05E_0000_0000_0001_u64 ^ seed;
        let mut patched = src.clone();
        // changed-once flag: pools draw from SRC-class cells that are
        // still unchanged — a cell consumed by one class can never be
        // redrawn by another (the double-draw bug the exact-dose assert
        // caught: 66,161 ≠ 87,119).
        let mut used = vec![false; N * N];
        for (from, to, count) in &classes {
            let mut idxs: Vec<usize> = (0..N * N)
                .filter(|&i| !used[i] && src[i] == *from)
                .collect();
            for i in (1..idxs.len()).rev() {
                rng = xorshift64(rng);
                let j = (rng % (i as u64 + 1)) as usize;
                idxs.swap(i, j);
            }
            let take = (*count).min(idxs.len() as u64) as usize;
            for &cell in idxs.iter().take(take) {
                patched[cell] = *to;
                used[cell] = true;
            }
        }
        let changed = patched.iter().zip(&src).filter(|(a, b)| a != b).count();
        assert!(
            changed == h2_cells,
            "dose-{seed}: changed {changed} != target {h2_cells} — dose must match EXACTLY"
        );
        let out = format!("models/null-dose-{seed}.gguf");
        do_surgery(&patched, &out);
        println!("dose-{seed}: {changed} cells (exact) · S2 clean → {out}");
    }

    // ----- SHUFFLE ×3: the real cell set, permuted new-values, new≠source -----
    // Perfect-dose constraint: assign the H2 VALUE MULTISET onto the H2
    // changed positions, no value on a same-valued source, EXACT dose.
    // Structure: 3 source-classes × 3 value-classes; a valid assignment
    // is a transportation problem. The terminal diff has pool = {0, +1}
    // (no −1s among new-values), so: −1-sourced cells take 0s or +1s;
    // 0-sourced take −1/ +1 (pool has +1 and 0 — but 0-on-0 forbidden →
    // 0-sourced cells need +1 or −1; pool's −1 count is 0 → they need +1);
    // +1-sourced take 0s or −1s (pool: 0s). Solve the 3×3 counts by
    // greedy most-constrained, then place uniformly within each bucket
    // with the seeded rng. Any infeasibility ABORTS loudly.
    let new_vals: Vec<Trit> = changed_idx.iter().map(|&i| h2[i]).collect();
    let tix = |t: Trit| -> usize {
        match t {
            Trit::MinusOne => 0,
            Trit::Zero => 1,
            Trit::One => 2,
        }
    };
    let tr_of = |i: usize| -> Trit {
        match i {
            0 => Trit::MinusOne,
            1 => Trit::Zero,
            _ => Trit::One,
        }
    };
    // src-class counts among changed cells; value-pool counts
    let mut src_count = [0usize; 3];
    for &i in &changed_idx {
        src_count[tix(src[i])] += 1;
    }
    let mut pool = [0usize; 3];
    for &v in &new_vals {
        pool[tix(v)] += 1;
    }
    // assignment[i][j] = # of src-class-i cells receiving value-class-j
    // (i≠j per the constraint). Greedy: repeatedly assign the largest
    // feasible (remaining_src, remaining_pool) pair avoiding i==j.
    let mut rem_src = src_count;
    let mut rem_pool = pool;
    let mut assign = [[0usize; 3]; 3];
    loop {
        let total_left: usize = rem_src.iter().sum();
        if total_left == 0 {
            break;
        }
        // Most-constrained-first: pick the src class whose allowed pool
        // (values != its own class) is tightest; assign to that pool's
        // largest bucket. Infeasibility aborts loudly (pre-registered).
        let mut best: Option<(usize, usize, i64)> = None; // (src_i, val_j, tightness)
        for i in 0..3 {
            if rem_src[i] == 0 {
                continue;
            }
            let allowed: Vec<usize> = (0..3).filter(|&j| j != i && rem_pool[j] > 0).collect();
            if allowed.is_empty() {
                panic!("shuffle: src-class {i} unassignable — infeasible; ABORT (pre-registered)");
            }
            let allowed_total: usize = allowed.iter().map(|&j| rem_pool[j]).sum();
            if allowed_total < rem_src[i] {
                panic!("shuffle: src-class {i} under-supplied ({allowed_total} < {}) — infeasible; ABORT", rem_src[i]);
            }
            // tightness: allowed surplus over demand (smaller = tighter)
            let tight = (allowed_total - rem_src[i]) as i64;
            if best.is_none() || tight < best.unwrap().2 {
                let jbest = *allowed.iter().max_by_key(|&&j| rem_pool[j]).unwrap();
                best = Some((i, jbest, tight));
            }
        }
        let (i, j, _tight) = best.expect("feasible step exists");
        let amt = rem_src[i].min(rem_pool[j]);
        assign[i][j] += amt;
        rem_src[i] -= amt;
        rem_pool[j] -= amt;
    }
    // place: per (i,j) bucket, draw amt cells from src-class-i changed cells
    for seed in 1..=3u64 {
        let mut rng = 0x5EED_FEED_0000_0003_u64 ^ seed;
        let mut cells_by_class: Vec<Vec<usize>> = vec![Vec::new(); 3];
        for &i in &changed_idx {
            cells_by_class[tix(src[i])].push(i);
        }
        for v in &mut cells_by_class {
            for k in (1..v.len()).rev() {
                rng = xorshift64(rng);
                let j2 = (rng % (k as u64 + 1)) as usize;
                v.swap(k, j2);
            }
        }
        let mut patched = src.clone();
        let mut cursors = [0usize; 3];
        let mut placed = 0usize;
        for i in 0..3 {
            for j in 0..3 {
                let amt = assign[i][j];
                for _ in 0..amt {
                    let cell = cells_by_class[i][cursors[i]];
                    cursors[i] += 1;
                    patched[cell] = tr_of(j);
                    placed += 1;
                }
            }
        }
        let changed = patched.iter().zip(&src).filter(|(a, b)| a != b).count();
        assert_eq!(changed, h2_cells, "shuffle-{seed}: dose must stay exact");
        assert_eq!(placed, h2_cells, "shuffle-{seed}: all cells placed");
        let out = format!("models/null-shuffle-{seed}.gguf");
        do_surgery(&patched, &out);
        println!("shuffle-{seed}: {changed} cells (exact, real positions, class-flow permuted values) · S2 clean → {out}");
    }
    println!("done — judge chains next (queued behind the stress arm).");
}
