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
//!   shift and a 10-bit fractional table lookup; clamp below 2^-12 → 0.

/// Q12 full scale.
pub const Q12: i64 = 4096;
/// Table resolution: 1024 entries over [0, 1).
const EXP2_ENTRIES: usize = 1024;
/// milli-domain log2(e) × 1000 (round(1.4426950408889634 × 1000)).
const LOG2_E_MILLI: i64 = 1443;

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
    /// first). Returns 0 below the Q12 floor (exponent < −12).
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
    /// [`Q12`] exactly (the last element carries the rounding remainder).
    pub fn softmax_q12(&self, logits_milli: &[i32], out: &mut [i32]) {
        debug_assert_eq!(logits_milli.len(), out.len());
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
            // Everything underflowed (impossible with the max element
            // contributing ≥ Q12·2^0… kept as a hard guard).
            let last = out.len() - 1;
            out.fill(0);
            out[last] = Q12 as i32;
            return;
        }
        let mut acc: i64 = 0;
        let last = out.len() - 1;
        for (i, &e) in exps.iter().enumerate() {
            if i == last {
                out[i] = (Q12 - acc).clamp(0, Q12) as i32;
            } else {
                // p_i = round(e·4096/sum), half-away.
                let p = (e * Q12 + sum / 2) / sum;
                acc += p;
                out[i] = p.clamp(0, Q12) as i32;
            }
        }
    }

    /// SiLU (swish): `x·sigmoid(x)` in milli, all integer via the exp
    /// table. Sign-correct by construction; zero at zero.
    #[must_use]
    pub fn silu_milli(&self, x_milli: i32) -> i32 {
        // sigmoid(x) = 1/(1+e^{−x}), in Q12.
        let sig_q12 = if x_milli >= 0 {
            // e^{−x} ≤ 1 → Q12 cleanly; sig = 4096·4096/(4096+e).
            let e = self.exp2_q12(-i64::from(x_milli));
            (Q12 * Q12 + (Q12 + e) / 2) / (Q12 + e)
        } else {
            // e^{x} ≤ 1; sig = e·4096/(4096+e).
            let e = self.exp2_q12(i64::from(x_milli));
            (e * Q12 + (Q12 + e) / 2) / (Q12 + e)
        };
        // silu = x · sig / 4096 (milli × Q12 → milli), round-half-away.
        let prod = i64::from(x_milli) * sig_q12;
        if prod >= 0 {
            ((prod + Q12 / 2) / Q12) as i32
        } else {
            (-((-prod + Q12 / 2) / Q12)) as i32
        }
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
/// cos/sin live in milli, computed at the load edge; [`RopeTables::apply`]
/// is pure integer afterward.
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
        let pairs = head_dim / 2;
        let freq_scale = 1.0 / yarn_factor;
        // corr_dim(n_rot) = n_dims·ln(orig/(n_rot·2π))/(2·ln(base)).
        let corr_dim = |n_rot: f64| -> f64 {
            (head_dim as f64) * (orig_ctx as f64 / (n_rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * freq_base.ln())
        };
        let low = corr_dim(beta_fast).floor().max(0.0);
        let high = corr_dim(beta_slow).ceil().min((head_dim - 1) as f64);
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
                let r = ramp(i as f64); // ext_factor = 1
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
    /// `(cos_i, sin_i)`. Pure integer (i64 products, /1000).
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
            head[i] = ((x1 * c - x2 * s + 500) / 1000) as i32;
            head[i + pairs] = ((x2 * c + x1 * s + 500) / 1000) as i32;
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
        assert_eq!(kit.exp2_q12(-20_000), 0); // e^-20 < Q12 floor
        // Monotone nonincreasing for x ≤ 0.
        let mut prev = Q12;
        for step in 0..200 {
            let x = -i64::from(step) * 7;
            let e = kit.exp2_q12(x);
            assert!(e <= prev, "exp2 not monotone at {x}");
            prev = e;
        }
    }

    #[test]
    fn softmax_matches_f64_and_sums_exact() {
        let kit = MathKit::new();
        let cases: Vec<Vec<i32>> = vec![
            vec![0, 0, 0, 0],
            vec![1000, 0, -1000],
            vec![5000, 4997, -200, -9000],
            vec![-3000, -3000, 12000, 11000, 0],
        ];
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
        for x_milli in -6000..=6000i32 {
            let x = f64::from(x_milli) / 1000.0;
            let want = x / (1.0 + (-x).exp()) * 1000.0;
            let got = f64::from(kit.silu_milli(x_milli));
            // Relative+absolute slack: table quantization (2^-10 on the
            // exponent) + /1000 rounding.
            assert!(
                (got - want).abs() <= 2.0 + want.abs() * 0.02,
                "silu({x_milli}) = {got} vs {want}"
            );
        }
        assert_eq!(kit.silu_milli(0), 0);
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
        // (pos, i) pairs — catches table-layout and pairing mistakes.
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
            for i in [0_usize, 1, 10, 32, 63] {
                let theta_base = 1e6_f64.powf(-(2.0 * i as f64) / 128.0);
                let extrap = pos as f64 * theta_base;
                let interp = 0.25 * extrap;
                let y = ((i as f64 / 2.0 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
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
