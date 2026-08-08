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
    }
}
