//! Integer math for the runtime's nonlinear core — softmax, SiLU, RoPE
//! (Stage 4, session 3).
//!
//! Doctrine: the *compute path* is integer (milli/Q12 fixed point); the
//! *transcendental constants* (the `2^frac` table, YaRN cos/sin tables)
//! are generated once at the load edge with `std` f64 — the same
//! conversion doctrine as f32 norm weights (`f32_bits_to_milli`):
//! constants at load, integers forever after.
//!
//! # Fixed-point conventions
//!
//! - Activations/hidden: **milli** (i32; real × 1000).
//! - Softmax/SiLU exponentials: **Q12** (0..=4096 ≈ 0..1).
//! - `exp(x) = 2^(x·log2(e))`: decompose the exponent into an integer
//!   shift and a 10-bit fractional table lookup; clamp below 2^-11 → 0
//!   (the floor is hit at `|x| ≥ 7624` milli: n = ceil(7624·1443/1e6) = 12).

/// Q12 full scale.
pub const Q12: i64 = 4096;
/// Table resolution: 1024 entries over [0, 1).
const EXP2_ENTRIES: usize = 1024;
/// milli-domain log2(e) × 1000 (round(1.4426950408889634 × 1000)).
const LOG2_E_MILLI: i64 = 1443;

/// Divide with round-half-away-from-zero: the shared rounding convention
/// of every fixed-point hop in this crate (milli↔unit, Q12, rescale).
///
/// `den` must be positive (debug-asserted). For `num ≥ 0` this is
/// `(num + den/2) / den`; negatives round symmetrically away from zero —
/// never the trunc-toward-zero Rust `/` default.
///
/// Tested once, against an f64 reference, in [`crate::math::tests`].
#[must_use]
pub fn div_round_half_away(num: i64, den: i64) -> i64 {
    debug_assert!(den > 0, "divisor must be positive");
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -((-num + den / 2) / den)
    }
}

/// The integer nonlinear kit: one exp2 table, many uses (softmax, SiLU).
#[derive(Debug, Clone)]
pub struct MathKit {
    /// `table[i] = round(2^(i/1024) × 4096)` — Q12 values in [4096, 8192).
    exp2: [u16; EXP2_ENTRIES],
}

impl MathKit {
    /// Build the tables (load edge — uses std f64 for constants).
    #[must_use]
    pub fn new() -> Self {
        let mut exp2 = [0_u16; EXP2_ENTRIES];
        for (i, slot) in exp2.iter_mut().enumerate() {
            let v = ((i as f64 / EXP2_ENTRIES as f64).exp2() * Q12 as f64).round();
            *slot = v.clamp(0.0, u16::MAX as f64) as u16;
        }
        Self { exp2 }
    }

    /// `2^(x_milli/1000)` in Q12 for `x_milli ≤ 0` (callers max-subtract
    /// first; a positive input saturates to [`Q12`]). Returns 0 once the
    /// exponent falls below the Q12 floor (`x_milli ≤ −7624`, i.e.
    /// 2^−11 — one quantum above resolution). Input bound: the internal
    /// `× 1443` product must stay inside i64, so `|x_milli| ≲ 6.4e15`.
    #[must_use]
    pub fn exp2_q12(&self, x_milli: i64) -> i64 {
        if x_milli >= 0 {
            return Q12;
        }
        // y = x·log2(e): milli × milli → micro-exponent (y_real × 1e6).
        let y_micro = x_milli * LOG2_E_MILLI; // ≤ 0; |y| < 2^63 for |x| < 3e12
        // Decompose y = −n + f, n integer ≥ 0, f ∈ [0,1):
        //   n = ceil(−y) = (−y_micro + 1e6 − 1) / 1e6 (floor div on negatives)
        let neg_micro = -y_micro;
        let n = (neg_micro + 999_999) / 1_000_000; // ceil div (stable)
        if n >= 12 {
            return 0; // below Q12 resolution
        }
        // f = n + y (in micro): f_micro = n·1e6 − neg_micro ∈ [0, 1e6).
        let f_micro = n * 1_000_000 - neg_micro;
        let idx = ((f_micro * EXP2_ENTRIES as i64) / 1_000_000) as usize;
        let frac_q12 = i64::from(self.exp2[idx.min(EXP2_ENTRIES - 1)]);
        frac_q12 >> n
    }

    /// Integer softmax over milli logits → Q12 probabilities summing to
    /// [`Q12`] **exactly, for any length** — floor quotas plus the
    /// largest-remainder correction (the `rem` units left by flooring go
    /// to the elements with the largest fractional residuals).
    ///
    /// `out.len()` must equal `logits_milli.len()`.
    ///
    /// # Panics
    ///
    /// Panics (release too, via indexing) if `out` is shorter than
    /// `logits_milli`; a longer `out` leaves its tail unwritten.
    pub fn softmax_q12(&self, logits_milli: &[i32], out: &mut [i32]) {
        debug_assert_eq!(logits_milli.len(), out.len(), "softmax length mismatch");
        if logits_milli.is_empty() {
            return;
        }
        let max = i64::from(*logits_milli.iter().max().expect("nonempty"));
        let mut exps = Vec::with_capacity(logits_milli.len());
        let mut sum: i64 = 0;
        for &l in logits_milli {
            let e = self.exp2_q12(i64::from(l) - max);
            exps.push(e);
            sum += e;
        }
        if sum == 0 {
            // Unreachable through this API (the max element contributes
            // exp2_q12(0) = Q12), kept as a hard guard: all mass on the max.
            let argmax = logits_milli
                .iter()
                .enumerate()
                .max_by_key(|(_, &l)| l)
                .map(|(i, _)| i)
                .unwrap_or(0);
            out.fill(0);
            out[argmax] = Q12 as i32;
            return;
        }
        let n = exps.len();
        // Floor quotas: p_i = floor(e_i·Q12/sum) — all non-negative here.
        let mut floors = vec![0_i64; n];
        let mut rems = vec![0_i64; n];
        let mut acc: i64 = 0;
        for i in 0..n {
            let scaled = exps[i] * Q12;
            floors[i] = scaled / sum;
            rems[i] = scaled % sum;
            acc += floors[i];
        }
        // Σfloor ≤ Q12 < Σfloor + n — distribute the remainder one unit
        // at a time, largest residual first (ties → larger e), so the sum
        // is exactly Q12 for every input.
        let mut rem = Q12 - acc;
        debug_assert!(rem >= 0 && (rem as usize) < n, "remainder out of range");
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| rems[a].cmp(&rems[b]).then(exps[a].cmp(&exps[b])));
        let mut k = n;
        while rem > 0 {
            k -= 1;
            floors[order[k]] += 1;
            rem -= 1;
        }
        for (slot, &p) in out.iter_mut().zip(floors.iter()) {
            *slot = p.clamp(0, Q12) as i32;
        }
    }

    /// SiLU (swish): `x·sigmoid(x)` in milli, all integer via the exp
    /// table. Sign-correct by construction; zero at zero. Accuracy:
    /// within ~2 milli + 2% of the f64 reference for `|x| < 7.6`;
    /// for `x ≤ −7624` the output is 0 (the exp table's Q12 floor),
    /// a documented absolute error of at most ~4 milli at the cliff.
    #[must_use]
    pub fn silu_milli(&self, x_milli: i32) -> i32 {
        // sigmoid(x) = 1/(1+e^{−x}), in Q12.
        let sig_q12 = if x_milli >= 0 {
            // e^{−x} ≤ 1 → Q12 cleanly; sig = 4096·4096/(4096+e).
            let e = self.exp2_q12(-i64::from(x_milli));
            div_round_half_away(Q12 * Q12, Q12 + e)
        } else {
            // e^{x} ≤ 1; sig = e·4096/(4096+e).
            let e = self.exp2_q12(i64::from(x_milli));
            div_round_half_away(e * Q12, Q12 + e)
        };
        // silu = x · sig / 4096 (milli × Q12 → milli), round-half-away.
        div_round_half_away(i64::from(x_milli) * sig_q12, Q12) as i32
    }
}

impl Default for MathKit {
    fn default() -> Self {
        Self::new()
    }
}

/// YaRN rotary tables — pinned verbatim from the Prism fork's
/// `ggml-cpu/ops.cpp` (`rope_yarn` + `rope_yarn_ramp`) and `ggml.c`
/// (`ggml_rope_yarn_corr_dims`).
///
/// Model config (from the real file): base 1e6, factor 4 (freq_scale
/// 0.25), orig_ctx 8192, head_dim 128 (64 rotary pairs, half-split /
/// "neox" style), beta_fast 32, beta_slow 1, ext_factor 1, attn_factor
/// 1.0. `mscale_total = 1.0·(1 + 0.1·ln(1/0.25)) ≈ 1.1386`.
///
/// Fidelity notes (2026-08-15 adversarial review): mscale is baked into
/// cos AND sin — the llama.cpp lineage convention — so attention scores
/// carry mscale² ≈ 1.2965; the YaRN *paper* scales scores by mscale
/// once, but the fork (our pinned reference) does it this way.
/// `ext_factor`/`attn_factor` are fixed at 1 (not parameters).
///
/// cos/sin live in milli, computed at the load edge; [`RopeTables::apply`]
/// is pure integer afterward.
///
/// # Panics (in [`RopeTables::new_yarn`], debug)
///
/// Panics if `head_dim` is odd (rotation pairs need an even dimension).
#[derive(Debug, Clone)]
pub struct RopeTables {
    max_pos: usize,
    head_dim: usize,
    /// `cos_milli[pos·(head_dim/2) + i]`.
    cos_milli: Vec<i32>,
    sin_milli: Vec<i32>,
}

impl RopeTables {
    /// Build tables for positions `0..max_pos` (load edge).
    #[must_use]
    pub fn new_yarn(
        head_dim: usize,
        max_pos: usize,
        freq_base: f64,
        yarn_factor: f64,
        orig_ctx: usize,
        beta_fast: f64,
        beta_slow: f64,
    ) -> Self {
        assert!(head_dim.is_multiple_of(2), "head_dim must be even (rotation pairs)");
        let pairs = head_dim / 2;
        let freq_scale = 1.0 / yarn_factor;
        // corr_dim(n_rot) = n_dims·ln(orig/(n_rot·2π))/(2·ln(base)).
        let corr_dim = |n_rot: f64| -> f64 {
            (head_dim as f64) * (orig_ctx as f64 / (n_rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * freq_base.ln())
        };
        let low = corr_dim(beta_fast).floor().max(0.0);
        let high = corr_dim(beta_slow).ceil().min((head_dim - 1) as f64);
        // The pinned reference formula: ramp(i0) with i0 the ELEMENT
        // index (the reference iterates i0 += 2), so i0/2 is the PAIR
        // index. Callers below pass i0 = 2·pair. (2026-08-15 review:
        // passing the pair index directly shifted the interpolation
        // window from pairs [low, high] to [2·low, 2·high] — one octave
        // high; this call site is the fix, pinned by the i0=2i test.)
        let ramp = |i0: f64| -> f64 {
            let y = (i0 / 2.0 - low) / (high - low).max(0.001);
            1.0 - y.clamp(0.0, 1.0)
        };
        let mscale = 1.0_f64 * (1.0 + 0.1 * (1.0 / freq_scale).ln());

        let mut cos_milli = vec![0_i32; max_pos * pairs];
        let mut sin_milli = vec![0_i32; max_pos * pairs];
        // theta_base for pair i: base^(−2i/head_dim), iterated by
        // theta_scale = base^(−2/head_dim) (fork's cache_init loop).
        let theta_scale = freq_base.powf(-2.0 / head_dim as f64);
        for pos in 0..max_pos {
            let mut theta_base = 1.0_f64; // pos × base^0 at i=0… fork: theta = pos·theta_i
            for i in 0..pairs {
                let theta_extrap = pos as f64 * theta_base;
                let theta_interp = freq_scale * theta_extrap;
                let r = ramp(2.0 * i as f64); // ext_factor = 1; i0 = element index
                let theta = theta_interp * (1.0 - r) + theta_extrap * r;
                cos_milli[pos * pairs + i] = (theta.cos() * mscale * 1000.0).round() as i32;
                sin_milli[pos * pairs + i] = (theta.sin() * mscale * 1000.0).round() as i32;
                theta_base *= theta_scale;
            }
        }
        Self { max_pos, head_dim, cos_milli, sin_milli }
    }

    /// Apply RoPE in place to one head's `head_dim` milli values at
    /// `pos`, half-split pairing: `(v[i], v[i+pairs])` rotated by
    /// `(cos_i, sin_i)`. Pure integer (i64 products, round-half-away,
    /// clamped into i32 — hostile magnitudes saturate, never wrap).
    ///
    /// # Panics
    ///
    /// Panics if `pos >= max_pos` or the slice length ≠ `head_dim`.
    pub fn apply(&self, head: &mut [i32], pos: usize) {
        assert_eq!(head.len(), self.head_dim, "head dim mismatch");
        assert!(pos < self.max_pos, "position beyond rope table");
        let pairs = self.head_dim / 2;
        for i in 0..pairs {
            let c = i64::from(self.cos_milli[pos * pairs + i]);
            let s = i64::from(self.sin_milli[pos * pairs + i]);
            let x1 = i64::from(head[i]);
            let x2 = i64::from(head[i + pairs]);
            let r1 = div_round_half_away(x1 * c - x2 * s, 1000);
            let r2 = div_round_half_away(x2 * c + x1 * s, 1000);
            head[i] = r1.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            head[i + pairs] = r2.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // f64 references — the doctrine for numeric tests (ISA Learning,
    // session 2): compare against an independent float path, not
    // hand-derived integers.

    #[test]
    fn div_round_half_away_matches_f64() {
        // Ties round away from zero on both signs; otherwise nearest.
        assert_eq!(div_round_half_away(1500, 1000), 2);
        assert_eq!(div_round_half_away(-1500, 1000), -2);
        assert_eq!(div_round_half_away(500, 1000), 1);
        assert_eq!(div_round_half_away(-500, 1000), -1);
        assert_eq!(div_round_half_away(499, 1000), 0);
        assert_eq!(div_round_half_away(-499, 1000), 0);
        assert_eq!(div_round_half_away(0, 7), 0);
        // Odd divisor: half still rounds away exactly.
        assert_eq!(div_round_half_away(5, 10), 1); // 0.5
        assert_eq!(div_round_half_away(-5, 10), -1);
        // Randomized sweep vs the f64 reference (round = half away).
        let mut x = 0x2545_F491_4F6C_DD1D_u64;
        for _ in 0..2000 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let num = (x >> 1) as i64 % 1_000_003 - 500_001;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let den = ((x >> 1) % 4093 + 1) as i64;
            let want = ((num as f64) / (den as f64)).round() as i64;
            assert_eq!(div_round_half_away(num, den), want, "{num}/{den}");
        }
    }

    #[test]
    fn exp2_table_basics() {
        let kit = MathKit::new();
        assert_eq!(kit.exp2_q12(0), Q12); // e^0 = 1
        // e^x in Q12 vs f64 — the function computes e^x (2^(x·log2e)).
        for x_milli in [-1000_i64, -693, -2000, -3010, -500, -1200] {
            let want = ((x_milli as f64 / 1000.0).exp() * Q12 as f64).round();
            let got = kit.exp2_q12(x_milli);
            assert!(
                (got as f64 - want).abs() <= 3.0,
                "e^({x_milli}m) = {got} vs {want}"
            );
        }
        // Q12 floor at the 2^-11 boundary (n = ceil(|x|·1443/1e6) = 12).
        assert_eq!(kit.exp2_q12(-7623), 2); // last surviving value
        assert_eq!(kit.exp2_q12(-7624), 0); // first floored value
        assert_eq!(kit.exp2_q12(-20_000), 0);
        // Monotone nonincreasing over the full live domain [−7623, 0].
        let mut prev = Q12;
        for step in 0..1200 {
            let x = -i64::from(step) * 7;
            let e = kit.exp2_q12(x);
            assert!(e <= prev, "exp2 not monotone at {x}");
            prev = e;
        }
    }

    #[test]
    fn softmax_matches_f64_and_sums_exact() {
        let kit = MathKit::new();
        let mut cases: Vec<Vec<i32>> = vec![
            vec![0, 0, 0, 0],
            vec![1000, 0, -1000],
            vec![5000, 4997, -200, -9000],
            vec![-3000, -3000, 12000, 11000, 0],
            // 2026-08-15 review: constructed breakers of the old
            // last-element remainder-carry (summed 4097/4098).
            vec![0, -1386, -1386, -7624],
            vec![0, 0, 0, 0, 0, 0, -7624], // 6-way tie at max + floored tail
        ];
        // Every tie-count from 2..=20 with a floored tail, plus a
        // 64-way tie (the in-model window).
        for k in [2_usize, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 26, 31, 63] {
            let mut v = vec![0_i32; k];
            v.push(-7624);
            cases.push(v);
        }
        for logits in cases {
            let mut out = vec![0_i32; logits.len()];
            kit.softmax_q12(&logits, &mut out);
            assert_eq!(
                out.iter().map(|p| i64::from(*p)).sum::<i64>(),
                Q12,
                "Q12 sum must be exact for {logits:?}"
            );
            let max_l = logits.iter().copied().max().unwrap() as f64 / 1000.0;
            let exps: Vec<f64> = logits
                .iter()
                .map(|l| (( *l as f64 / 1000.0) - max_l).exp())
                .collect();
            let sum: f64 = exps.iter().sum();
            for (i, p) in out.iter().enumerate() {
                let want = exps[i] / sum * Q12 as f64;
                assert!(
                    (i64::from(*p) as f64 - want).abs() <= 3.0,
                    "probs[{i}] {} vs f64 {want:.2} (logits {logits:?})",
                    p
                );
            }
        }
        // Exact-sum holds for a long tail-only spread too (no ties).
        let spread: Vec<i32> = (0..64).map(|i| -i * 120).collect();
        let mut out = vec![0_i32; 64];
        kit.softmax_q12(&spread, &mut out);
        assert_eq!(out.iter().map(|p| i64::from(*p)).sum::<i64>(), Q12);
    }

    #[test]
    fn softmax_underflow_guard_puts_mass_somewhere() {
        let kit = MathKit::new();
        let logits = vec![-500_000, -400_000]; // huge gap after max-subtract
        let mut out = vec![0_i32; 2];
        kit.softmax_q12(&logits, &mut out);
        assert_eq!(out.iter().sum::<i32>(), Q12 as i32);
        assert_eq!(out[0], 0); // e^-100 → 0
        assert_eq!(out[1], Q12 as i32); // max element carries all mass
    }

    #[test]
    fn silu_matches_f64() {
        let kit = MathKit::new();
        for x_milli in -9000..=9000i32 {
            let x = f64::from(x_milli) / 1000.0;
            let want = x / (1.0 + (-x).exp()) * 1000.0;
            let got = f64::from(kit.silu_milli(x_milli));
            // Relative+absolute slack: table quantization (2^-10 on the
            // exponent) + /1000 rounding; beyond the x ≤ −7624 cliff the
            // output is 0 by design — a documented ≤4 milli absolute
            // error (see silu_milli docs).
            let cliff = x_milli <= -7624;
            assert!(
                (got - want).abs() <= 2.0 + want.abs() * 0.02 + if cliff { 4.5 } else { 0.0 },
                "silu({x_milli}) = {got} vs {want}"
            );
        }
        assert_eq!(kit.silu_milli(0), 0);
        assert_eq!(kit.silu_milli(-8000), 0); // cliff: documented zero
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let rope = RopeTables::new_yarn(128, 16, 1e6, 4.0, 8192, 32.0, 1.0);
        let mut head: Vec<i32> = (0..128).map(|i| i * 7 - 400).collect();
        let before = head.clone();
        rope.apply(&mut head, 0);
        // mscale ≈ 1.1386 ≠ 1 → not an exact identity; tolerance covers it.
        for (b, a) in before.iter().zip(&head) {
            assert!((a - b).abs() <= b.unsigned_abs() as i32 / 5 + 2, "{b} -> {a}");
        }
    }

    #[test]
    fn rope_preserves_norm_approximately() {
        let rope = RopeTables::new_yarn(128, 16, 1e6, 4.0, 8192, 32.0, 1.0);
        let head: Vec<i32> = (0..128).map(|i| i * 37 % 200 - 100).collect();
        let norm2 = |v: &[i32]| -> i64 { v.iter().map(|x| i64::from(*x) * i64::from(*x)).sum() };
        let n0 = norm2(&head);
        for pos in 0..8 {
            let mut h = head.clone();
            rope.apply(&mut h, pos);
            let n1 = norm2(&h);
            // mscale² ≈ 1.297 tolerance ±5%: rotation preserves, mscale
            // scales by ≈1.1386² = 1.2965.
            let ratio = n1 as f64 / n0 as f64;
            assert!(
                (ratio - 1.2965).abs() < 0.06,
                "pos {pos}: norm ratio {ratio}"
            );
        }
    }

    #[test]
    fn rope_matches_f64_reference() {
        // Independent f64 re-derivation of the fork's YaRN at a few
        // (pos, i) pairs — catches table-layout, pairing, AND ramp-window
        // mistakes. The ramp argument is the ELEMENT index (i0 = 2·pair),
        // per the pinned formula `1−clamp((i0/2−low)/(high−low))` — the
        // 2026-08-15 review caught the code (and this test's earlier
        // copy) feeding the pair index straight in, which shifted the
        // interpolation window one octave; pairs 17..34 are checked
        // explicitly to pin the window.
        let rope = RopeTables::new_yarn(128, 16, 1e6, 4.0, 8192, 32.0, 1.0);
        let pairs = 64_usize;
        let corr = |n_rot: f64| -> f64 {
            128.0_f64 * (8192.0 / (n_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * 1e6_f64.ln())
        };
        let low = corr(32.0).floor().max(0.0);
        let high = corr(1.0).ceil().min(127.0);
        let mscale = 1.0 + 0.1 * (4.0_f64).ln();
        for pos in [0_usize, 1, 7, 15] {
            let mut head: Vec<i32> = (0..128).map(|i| i * 53 % 300 - 150).collect();
            let original = head.clone();
            rope.apply(&mut head, pos);
            for i in [0_usize, 1, 10, 17, 22, 32, 34, 63] {
                let theta_base = 1e6_f64.powf(-(2.0 * i as f64) / 128.0);
                let extrap = pos as f64 * theta_base;
                let interp = 0.25 * extrap;
                let y = ((i as f64 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
                let r = 1.0 - y;
                let theta = interp * (1.0 - r) + extrap * r;
                let c = theta.cos() * mscale;
                let s = theta.sin() * mscale;
                let x1 = f64::from(original[i]);
                let x2 = f64::from(original[i + pairs]);
                let want1 = x1 * c - x2 * s;
                let want2 = x2 * c + x1 * s;
                assert!(
                    (f64::from(head[i]) - want1).abs() <= 2.0 + want1.abs() * 0.002,
                    "pos {pos} i {i}: {} vs {want1}",
                    head[i]
                );
                assert!(
                    (f64::from(head[i + pairs]) - want2).abs() <= 2.0 + want2.abs() * 0.002,
                    "pos {pos} i {i} (y2): {} vs {want2}",
                    head[i + pairs]
                );
            }
        }
    }
}
