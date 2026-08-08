//! Ternary weight representation `{-1, 0, +1} × per-tensor scale` — Stage 1 of
//! the ternary bridge (see `docs/VISION.md`).
//!
//! BitNet-Round quantization: `γ = mean|w|` over the tensor; each weight snaps
//! to the nearest of `{-γ, 0, +γ}`. Pure integer math (`i64` accumulation for
//! the mean), `no_std`-compatible — pairs with `lif_neuron` and `synapse` as a
//! hot-path-clean primitive.
//!
//! # What this module is for
//!
//! Stage 1 asks: does an SNN whose synapse weights are *constrained* to three
//! levels still spike and learn? This module is the quantizer; the network
//! methods `ternarize_weights` / `reproject_ternary` apply it; the
//! `ternary_gate` example runs the falsifier and prints the verdict evidence.
//!
//! Stage 1.5b adds [`stochastic_ternary_flip`]: a pure, `no_std` Bernoulli
//! bucket-flip driven by the STDP residual. Under deterministic per-step
//! re-projection (Stage 1), STDP deltas max ±5 cannot cross the γ/2 ≈ 62
//! bucket boundary → 0 flips. The stochastic rule dissolves that boundary:
//! crossing becomes a probabilistic event whose rate reflects STDP evidence,
//! not a magnitude contest. Literature: Wu-Saxena 1801.02797, Mohan 2103.01271,
//! Camuñas-Mesa 2209.06068, `ReStoCNet` 1902.04161.
//!
//! # Stored representation
//!
//! The library's synapse weight stays `i16`. A ternary weight is simply an
//! `i16` whose value is one of `{-γ, 0, +γ}`. [`Trit`] is the *conceptual*
//! bucket classifier — used to count learning-induced state transitions
//! without changing how weights are stored or propagated.

#![allow(clippy::module_name_repetitions)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

/// A ternary value `{-1, 0, +1}` — the bucket class of a constrained synapse
/// weight. The stored weight remains `i16` (= [`Trit::to_weight`] at the
/// per-tensor scale); this enum exists to classify and count transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trit {
    /// The negative bucket — stored weight `−γ`.
    MinusOne,
    /// The silent bucket — stored weight `0`.
    Zero,
    /// The positive bucket — stored weight `+γ`.
    One,
}

impl Trit {
    /// Map this trit to its stored `i16` weight value at the given per-tensor
    /// `scale` (γ).
    #[must_use]
    pub const fn to_weight(self, scale: i16) -> i16 {
        match self {
            Self::MinusOne => -scale,
            Self::Zero => 0,
            Self::One => scale,
        }
    }

    /// Classify a stored weight into its ternary bucket using nearest-of-three
    /// projection: boundaries at `±⌈γ/2⌉`. Returns [`Trit::Zero`] if `scale ≤ 0`
    /// (degenerate — no meaningful quantization).
    #[must_use]
    pub fn from_weight(weight: i16, scale: i16) -> Self {
        if scale <= 0 {
            return Self::Zero;
        }
        // Nearest-of-three decision boundary: |w| ≥ ceil(γ/2) → nonzero bucket.
        // ceil(γ/2) for positive γ via (γ+1)/2 in i32 (avoids i16 overflow at MAX
        // and sidesteps unstable signed `div_ceil`).
        let threshold = (i32::from(scale) + 1) / 2;
        let w = i32::from(weight);
        if w >= threshold {
            Self::One
        } else if w <= -threshold {
            Self::MinusOne
        } else {
            Self::Zero
        }
    }
}

/// Compute the per-tensor scale `γ = round(mean|w|)` (BitNet-Round).
///
/// Returns `0` for an empty slice (degenerate — caller must guard).
#[must_use]
pub fn tensor_scale(weights: &[i16]) -> i16 {
    if weights.is_empty() {
        return 0;
    }
    let len = weights.len() as i64;
    let sum_abs: i64 = weights.iter().map(|w| i64::from(*w).abs()).sum();
    // Round-half-up to nearest integer. sum_abs ≥ 0, so no sign issues.
    let floor = sum_abs / len;
    let mean = if (sum_abs % len) * 2 >= len {
        floor + 1
    } else {
        floor
    };
    // Bounded by max|w| ≤ i16::MAX; cast is safe.
    mean.min(i64::from(i16::MAX)) as i16
}

/// Snap a single weight to the nearest of `{-scale, 0, +scale}`. Identity on a
/// weight already on-grid. Returns `0` if `scale ≤ 0`.
#[must_use]
pub fn project_to_ternary(weight: i16, scale: i16) -> i16 {
    Trit::from_weight(weight, scale).to_weight(scale)
}

/// BitNet-Round in-place ternarization: compute `γ = mean|w|`, then snap each
/// weight to the nearest of `{-γ, 0, +γ}`. Sign-preserving. Returns `γ`
/// (or `0` if the slice is empty).
///
/// No float — `γ` is computed in `i64` and the per-weight decision uses the
/// `±⌈γ/2⌉` boundary. This is a one-shot quantizer; keeping weights on-grid
/// during a live STDP run is the caller's job (re-project after each step).
///
/// Not idempotent: snapping changes the magnitude distribution, so a second
/// call computes a different `γ`. The gate holds `γ` fixed by calling this
/// once, then [`project_to_ternary`] with the returned `γ` for ongoing constraint.
#[must_use]
pub fn ternarize(weights: &mut [i16]) -> i16 {
    let scale = tensor_scale(weights);
    if scale == 0 {
        return 0;
    }
    for w in weights.iter_mut() {
        *w = project_to_ternary(*w, scale);
    }
    scale
}

/// Base rate for stochastic ternary bucket-flips (Stage 1.5b).
///
/// `P(flip) = |residual| × STOCHASTIC_FLIP_RATE / 65536`. The residual is the
/// off-grid push from one step's STDP delta (max ±5 under the default rule),
/// so this constant converts STDP evidence strength into a Bernoulli rate.
/// At 3000: δ=1 → P≈4.6%, δ=3 → P≈13.7%, δ=5 → P≈22.9%.
///
/// Tunable — raise for more aggressive learning, lower for stability. The
/// Stage 1.5b gate tunes this against the flip-rate / spiking-stability tradeoff.
pub const STOCHASTIC_FLIP_RATE: u32 = 3000;

/// Stochastic ternary bucket-flip (Stage 1.5b).
///
/// Given a current on-grid ternary weight, the per-tensor scale `γ`, the STDP
/// residual (the off-grid push from this step's plasticity — sign = direction,
/// magnitude = evidence strength), and a uniform random `draw ∈ [0, 65535]`:
///
/// - With probability `|residual| × `[`STOCHASTIC_FLIP_RATE`]` / 65536`, flip
///   the weight one bucket toward the residual's sign (LTP → +γ, LTD → −γ),
///   saturating at the extreme bucket.
/// - Otherwise, snap back to the current bucket (identity for an on-grid input).
///
/// Returns a value in `{-γ, 0, +γ}`. The caller should additionally clamp to
/// the synapse's `[min_weight, max_weight]` — e.g. excitatory synapses
/// (`min_weight = 0`) cannot go negative, so a LTD flip from the zero bucket
/// is a no-op after clamping.
///
/// # No latent state
///
/// The stored weight is **genuinely ternary** at all times. Unlike the
/// deferred 1.5a path (latent i16 accumulation + periodic re-quantize), there
/// is no shadow weight — the Bernoulli draw replaces the boundary crossing,
/// not the representation.
///
/// # Integer-only
///
/// No float in the hot path. The probability is realized as a fixed-point
/// threshold compared against a 16-bit LFSR draw.
#[must_use]
pub fn stochastic_ternary_flip(
    current_weight: i16,
    gamma: i16,
    residual: i16,
    draw: u16,
) -> i16 {
    if gamma <= 0 || residual == 0 {
        return project_to_ternary(current_weight, gamma);
    }
    let threshold = u32::from(residual.unsigned_abs())
        .saturating_mul(STOCHASTIC_FLIP_RATE)
        .min(0xFFFF) as u16;
    if draw >= threshold {
        return project_to_ternary(current_weight, gamma);
    }
    // Flip one bucket toward sign(residual), saturating at the extreme.
    if residual > 0 {
        current_weight.saturating_add(gamma).min(gamma)
    } else {
        current_weight.saturating_sub(gamma).max(-gamma)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::shadow_unrelated)]
    use super::*;
    use proptest::prelude::*;

    // ----- Unit tests -----

    #[test]
    fn to_weight_round_trips_all_three() {
        for scale in [1_i16, 50, 125, 1000] {
            assert_eq!(Trit::MinusOne.to_weight(scale), -scale);
            assert_eq!(Trit::Zero.to_weight(scale), 0);
            assert_eq!(Trit::One.to_weight(scale), scale);
        }
    }

    #[test]
    fn from_weight_boundaries_at_half_gamma() {
        let g = 125_i16; // boundary = ceil(125/2) = 63
        assert_eq!(Trit::from_weight(62, g), Trit::Zero);
        assert_eq!(Trit::from_weight(63, g), Trit::One);
        assert_eq!(Trit::from_weight(-62, g), Trit::Zero);
        assert_eq!(Trit::from_weight(-63, g), Trit::MinusOne);
        assert_eq!(Trit::from_weight(125, g), Trit::One);
        assert_eq!(Trit::from_weight(-125, g), Trit::MinusOne);
        assert_eq!(Trit::from_weight(0, g), Trit::Zero);
    }

    #[test]
    fn from_weight_degenerate_scale_is_zero() {
        assert_eq!(Trit::from_weight(100, 0), Trit::Zero);
        assert_eq!(Trit::from_weight(100, -5), Trit::Zero);
    }

    #[test]
    fn odd_gamma_boundary_rounds_up() {
        // γ=5 → ceil(5/2)=3: |w|≥3 → nonzero.
        assert_eq!(Trit::from_weight(2, 5), Trit::Zero);
        assert_eq!(Trit::from_weight(3, 5), Trit::One);
    }

    #[test]
    fn project_is_identity_on_grid() {
        let g = 125_i16;
        for w in [g, 0, -g] {
            assert_eq!(project_to_ternary(w, g), w);
        }
    }

    #[test]
    fn project_snaps_off_grid_to_nearest() {
        let g = 125_i16;
        // +130 is closer to +125 than to 0 or +250(clamped) → +125.
        assert_eq!(project_to_ternary(130, g), 125);
        // +60 is closer to 0 than to +125 → 0.
        assert_eq!(project_to_ternary(60, g), 0);
        // -200 clamps to -125 (only 3 levels).
        assert_eq!(project_to_ternary(-200, g), -125);
    }

    #[test]
    fn ternarize_balanced_weights_to_three_levels() {
        // Balanced-topology weight classes: {80, 150, 200, 120, -200, -120}.
        let mut w = [80_i16, 150, 200, 120, -200, -120];
        let g = ternarize(&mut w);
        assert!(g > 0);
        for &x in &w {
            assert!(
                x == g || x == 0 || x == -g,
                "ternarized weight {x} not in {{-{g}, 0, {g}}}"
            );
        }
        // Sign preserved.
        assert!(w[0] > 0 && w[3] > 0);
        assert!(w[4] < 0 && w[5] < 0);
    }

    #[test]
    fn ternarize_then_project_at_same_gamma_is_stable() {
        // ternarize computes γ and snaps. Re-projecting each weight at that SAME
        // γ is then identity — weights are stable at their bucket. (Note: calling
        // ternarize AGAIN is NOT idempotent, because the quantized distribution's
        // mean|w| differs from the input's — expected for mean-based quantization.
        // The gate holds γ fixed via reproject_ternary, not by re-ternarizing.)
        let mut w = [80_i16, 150, 200, -180, -120, 60];
        let gamma = ternarize(&mut w);
        for &x in &w {
            assert_eq!(
                project_to_ternary(x, gamma),
                x,
                "on-grid weight must be fixed by project at its own γ"
            );
        }
    }

    #[test]
    fn ternarize_empty_returns_zero() {
        let mut w: [i16; 0] = [];
        assert_eq!(ternarize(&mut w), 0);
    }

    #[test]
    fn tensor_scale_rounds_half_up() {
        // mean|w| = (10+15)/2 = 12.5 → round to 13 (half-up).
        assert_eq!(tensor_scale(&[10, 15]), 13);
        // mean|w| = (10+14)/2 = 12 → 12.
        assert_eq!(tensor_scale(&[10, 14]), 12);
    }

    // ----- Stochastic ternary flip (Stage 1.5b) -----

    #[test]
    fn stochastic_flip_zero_residual_is_no_op() {
        let g = 125_i16;
        for &w in &[g, 0, -g] {
            assert_eq!(stochastic_ternary_flip(w, g, 0, 0), w);
            assert_eq!(stochastic_ternary_flip(w, g, 0, 65535), w);
        }
    }

    #[test]
    fn stochastic_flip_zero_gamma_is_no_op() {
        assert_eq!(stochastic_ternary_flip(100, 0, 5, 0), 0);
        assert_eq!(stochastic_ternary_flip(100, -1, 5, 0), 0);
    }

    #[test]
    fn stochastic_flip_draw_zero_always_flips() {
        // draw = 0 is below any nonzero threshold → always flips.
        let g = 125_i16;
        assert_eq!(stochastic_ternary_flip(-g, g, 1, 0), 0);   // -γ → 0 (LTP)
        assert_eq!(stochastic_ternary_flip(0, g, 1, 0), g);     // 0 → +γ (LTP)
        assert_eq!(stochastic_ternary_flip(g, g, 1, 0), g);     // +γ → +γ (saturate)
        assert_eq!(stochastic_ternary_flip(g, g, -1, 0), 0);    // +γ → 0 (LTD)
        assert_eq!(stochastic_ternary_flip(0, g, -1, 0), -g);   // 0 → -γ (LTD)
        assert_eq!(stochastic_ternary_flip(-g, g, -1, 0), -g);  // -γ → -γ (saturate)
    }

    #[test]
    fn stochastic_flip_draw_max_never_flips() {
        // draw = 65535 ≥ any threshold (max threshold = 5×3000 = 15000 < 65535).
        let g = 125_i16;
        for &w in &[g, 0, -g] {
            assert_eq!(stochastic_ternary_flip(w, g, 5, 65535), w);
            assert_eq!(stochastic_ternary_flip(w, g, -5, 65535), w);
        }
    }

    #[test]
    fn stochastic_flip_sign_correctness() {
        let g = 125_i16;
        // LTP (residual > 0): target ≥ current bucket.
        assert!(stochastic_ternary_flip(0, g, 3, 0) >= 0);
        assert_eq!(stochastic_ternary_flip(0, g, 3, 0), g);
        // LTD (residual < 0): target ≤ current bucket.
        assert!(stochastic_ternary_flip(0, g, -3, 0) <= 0);
        assert_eq!(stochastic_ternary_flip(0, g, -3, 0), -g);
    }

    #[test]
    fn stochastic_flip_saturates_at_extremes() {
        let g = 125_i16;
        // +γ with LTP → stays +γ.
        assert_eq!(stochastic_ternary_flip(g, g, 5, 0), g);
        // -γ with LTD → stays -γ.
        assert_eq!(stochastic_ternary_flip(-g, g, -5, 0), -g);
    }

    #[test]
    fn stochastic_flip_output_always_on_grid() {
        let g = 125_i16;
        for &w in &[g, 0, -g] {
            for &res in &[0_i16, 1, 3, 5, -1, -3, -5] {
                for &draw in &[0_u16, 1, 100, 1000, 10000, 50000, 65535] {
                    let result = stochastic_ternary_flip(w, g, res, draw);
                    assert!(
                        result == g || result == 0 || result == -g,
                        "off-grid result {result} for w={w}, res={res}, draw={draw}"
                    );
                }
            }
        }
    }

    // ----- Property tests -----

    proptest! {
        /// Every ternarized weight lands in {-γ, 0, +γ}.
        #[test]
        fn prop_ternarize_output_on_grid(
            n in 1_usize..=200,
            seed_w in -500i16..=500,
        ) {
            let mut w: Vec<i16> = (0..n).map(|i| seed_w.wrapping_add(i as i16)).collect();
            let g = ternarize(&mut w);
            prop_assert!(g >= 0);
            for &x in &w {
                prop_assert!(x == g || x == 0 || x == -g, "off-grid: {x}, γ={g}");
            }
        }

        /// Ternarization preserves each weight's sign (or zeroes it).
        #[test]
        fn prop_ternarize_preserves_sign(
            weights in prop::collection::vec(-1000i16..=1000, 1..=50),
        ) {
            let mut w = weights.clone();
            let _g = ternarize(&mut w);
            for (orig, &quantized) in weights.iter().zip(w.iter()) {
                if *orig > 0 {
                    prop_assert!(quantized >= 0, "positive weight flipped negative");
                } else if *orig < 0 {
                    prop_assert!(quantized <= 0, "negative weight flipped positive");
                }
            }
        }

        /// from_weight ∘ to_weight round-trips for any trit + positive scale.
        #[test]
        fn prop_trit_round_trip(trit in prop_oneof![
            Just(Trit::MinusOne), Just(Trit::Zero), Just(Trit::One),
        ], scale in 1_i16..=2000) {
            let w = trit.to_weight(scale);
            prop_assert_eq!(Trit::from_weight(w, scale), trit);
        }

        /// project_to_ternary is idempotent.
        #[test]
        fn prop_project_idempotent(w in -2000i16..=2000, scale in 1_i16..=2000) {
            let once = project_to_ternary(w, scale);
            let twice = project_to_ternary(once, scale);
            prop_assert_eq!(once, twice);
        }

        /// tensor_scale is always in [0, max|w|].
        #[test]
        fn prop_scale_bounded(
            weights in prop::collection::vec(-1000i16..=1000, 1..=50),
        ) {
            let g = tensor_scale(&weights);
            let max_abs = weights.iter().map(|w| w.abs()).max().unwrap_or(0);
            prop_assert!(g >= 0);
            prop_assert!(g <= max_abs, "γ {g} exceeds max|w| {max_abs}");
        }

        /// Stochastic flip output is always in {-γ, 0, +γ}.
        #[test]
        fn prop_stochastic_flip_output_on_grid(
            bucket in prop_oneof![Just(-1_i16), Just(0), Just(1)],
            gamma in 1_i16..=2000,
            residual in -5i16..=5,
            draw in 0u16..=65535,
        ) {
            let w = bucket * gamma;
            let result = stochastic_ternary_flip(w, gamma, residual, draw);
            prop_assert!(
                result == gamma || result == 0 || result == -gamma,
                "off-grid: {result}, γ={gamma}"
            );
        }

        /// Stochastic flip sign-correctness: LTP (residual > 0) never moves
        /// toward -γ; LTD (residual < 0) never moves toward +γ.
        #[test]
        fn prop_stochastic_flip_sign_correct(
            bucket in prop_oneof![Just(-1_i16), Just(0), Just(1)],
            gamma in 1_i16..=2000,
            residual_abs in 1i16..=5,
            draw in 0u16..=65535,
        ) {
            let w = bucket * gamma;
            // LTP direction.
            let ltp = stochastic_ternary_flip(w, gamma, residual_abs, draw);
            prop_assert!(ltp >= w, "LTP must not decrease weight: {ltp} < {w}");
            // LTD direction.
            let ltd = stochastic_ternary_flip(w, gamma, -residual_abs, draw);
            prop_assert!(ltd <= w, "LTD must not increase weight: {ltd} > {w}");
        }

        /// Zero residual is always a no-op (identity projection).
        #[test]
        fn prop_stochastic_flip_zero_residual_noop(
            bucket in prop_oneof![Just(-1_i16), Just(0), Just(1)],
            gamma in 1_i16..=2000,
            draw in 0u16..=65535,
        ) {
            let w = bucket * gamma;
            prop_assert_eq!(stochastic_ternary_flip(w, gamma, 0, draw), w);
        }

        /// P(flip) is in [0, 1): the threshold never exceeds 65535, so draw=max
        /// never flips, and draw=0 always flips (for nonzero residual).
        #[test]
        fn prop_stochastic_flip_p_range(
            gamma in 1_i16..=2000,
            residual_abs in 1i16..=5,
        ) {
            // draw=max → never flips (threshold ≤ 5×3000 = 15000 < 65535).
            for &w in &[gamma, 0, -gamma] {
                prop_assert_eq!(
                    stochastic_ternary_flip(w, gamma, residual_abs, 65535),
                    w
                );
            }
        }
    }
}
