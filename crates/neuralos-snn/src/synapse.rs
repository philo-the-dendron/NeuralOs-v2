//! Synapse model — clean port from v0.1 `libneuralos/src/core/spiking_neural_network/synapse.rs`.
//!
//! # Audit context
//!
//! The audit flagged this as **Gem #3** — *"the better-engineered STDP rule impl"*
//! (vs `stdp_plasticity.rs` which had the LTP timing filter bug). Port this version.
//!
//! # Invariants (testable)
//!
//! - `Synapse::new(id, id, _)` always returns `Err(Error::InvalidParameter)` (no self-connections).
//! - Weight stays within `[min_weight, max_weight]` after every `update_weight` call.
//! - `STDPRule::calculate_weight_change(dt)` returns 0 for `|dt| ≥ 10·tau` (decay floor).
//! - LTP (pre before post, dt < 0) returns a non-negative delta.
//! - LTD (post before pre, dt > 0) returns a non-positive delta.
//!
//! # Fixed-point conventions
//!
//! - Weight: `i16`, fixed-point 1000 = 1.0 (range ~[-2000, 2000] typically).
//! - Time: `u32` microseconds.
//!
//! # `no_std`
//!
//! Uses `crate::Error` (no `String`, no alloc) for self-connection rejection.
//! Fully `no_std`-compatible — pairs with `lif_neuron`.

#![allow(clippy::module_name_repetitions)]
// Fixed-point design: narrow types (i16 weight, u32 μs) with
// physics-bounded values. Casts between them are intentional and bounded.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::Error;

/// Fixed-point scale: 1000 = 1.0.
///
/// Public since Stage 3: `bridge::wire_gamma_to_substrate` maps imported
/// wire-format scales into the substrate through exactly this constant —
/// the coupling has one home and a pinning test, not a magic 1000.
pub const SCALE: i32 = 1000;

/// Synapse type — biological neurotransmitter classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SynapseType {
    /// Excitatory (glutamate / AMPA). Weight ≥ 0.
    #[default]
    Excitatory,
    /// Inhibitory (GABA). Weight < 0.
    Inhibitory,
    /// Modulatory (dopamine, serotonin). Slower, lower amplitude.
    Modulatory,
}

/// Synapse with fixed-point weight and STDP instrumentation counters.
///
/// Biologically-modeled synaptic transmission with fixed-point math throughout.
/// Pairs with [`crate::lif_neuron::LIFNeuron`] for full integrate-and-fire + STDP.
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Presynaptic (source) neuron ID.
    pub pre_neuron_id: u16,
    /// Postsynaptic (target) neuron ID.
    pub post_neuron_id: u16,
    /// Neurotransmitter type — determines `tau_rise`, `tau_decay`, weight bounds.
    pub synapse_type: SynapseType,

    /// Synaptic weight (fixed-point, 1000 = 1.0). Negative = inhibitory.
    pub weight: i16,
    /// Maximum weight (plasticity clamp).
    pub max_weight: i16,
    /// Minimum weight (plasticity clamp).
    pub min_weight: i16,

    /// Conductance rise time constant (μs) — biological: AMPA 0.5ms, GABA 0.3ms.
    pub tau_rise_us: u16,
    /// Conductance decay time constant (μs) — biological: AMPA 5ms, GABA 10ms.
    pub tau_decay_us: u16,

    /// Cumulative RAW STDP delta applied by `update_weight` (session G
    /// instrumentation — the mechanism-label evidence): the signed sum of
    /// every plasticity delta BEFORE clamping. Read it per class (intra vs
    /// inter assemblies) to decide whether a realized bucket movement was
    /// pairing-driven or clamp/flip-machinery-driven.
    pub raw_stdp_delta: i64,
    /// Cumulative delta ABSORBED by the `[min_weight, max_weight]` clamp:
    /// `Σ(delta − applied)`. When |absorbed| is large relative to
    /// [`raw_stdp_delta`], bounds asymmetry — not pairing — shaped the
    /// trajectory.
    pub absorbed_delta: i64,
}

impl Synapse {
    /// New synapse between `pre_id` and `post_id` with initial `weight`.
    ///
    /// Returns `Err(Error::InvalidParameter)` on self-connection (`pre_id == post_id`).
    /// Synapse type is inferred from the sign of `weight`; biological parameters
    /// (`tau_rise`, `tau_decay`, weight bounds) are set from the type.
    pub fn new(pre_id: u16, post_id: u16, weight: i16) -> Result<Self, Error> {
        if pre_id == post_id {
            return Err(Error::InvalidParameter);
        }

        let synapse_type = if weight >= 0 {
            SynapseType::Excitatory
        } else {
            SynapseType::Inhibitory
        };
        let (tau_rise_us, tau_decay_us, max_weight, min_weight) = biological_params(synapse_type);

        Ok(Self {
            pre_neuron_id: pre_id,
            post_neuron_id: post_id,
            synapse_type,
            weight,
            max_weight,
            min_weight,
            tau_rise_us,
            tau_decay_us,
            raw_stdp_delta: 0,
            absorbed_delta: 0,
        })
    }

    /// Apply a plastic weight delta. Clamps to `[min_weight, max_weight]`.
    /// Tracks the raw delta and the clamp-absorbed remainder (session G).
    pub fn update_weight(&mut self, delta_weight: i16) {
        let target = self
            .weight
            .saturating_add(delta_weight)
            .clamp(self.min_weight, self.max_weight);
        let applied = target - self.weight;
        self.raw_stdp_delta += i64::from(delta_weight);
        self.absorbed_delta += i64::from(delta_weight) - i64::from(applied);
        self.weight = target;
    }

    /// Weight as a percentage of `max_weight` (for visualization).
    #[must_use]
    pub fn normalized_weight(&self) -> i16 {
        let abs_max = self.max_weight.unsigned_abs();
        if abs_max > 0 {
            (i32::from(self.weight) * 100 / i32::from(abs_max)) as i16
        } else {
            0
        }
    }
}

impl Default for Synapse {
    fn default() -> Self {
        // Default = excitatory synapse from neuron 0 to neuron 1, weight 100.
        Self::new(0, 1, 100).unwrap_or(Self {
            pre_neuron_id: 0,
            post_neuron_id: 1,
            synapse_type: SynapseType::Excitatory,
            weight: 100,
            max_weight: 2000,
            min_weight: 0,
            tau_rise_us: 500,
            tau_decay_us: 5_000,
            raw_stdp_delta: 0,
            absorbed_delta: 0,
        })
    }
}

/// Biological parameter tuple (`tau_rise_us`, `tau_decay_us`, `max_weight`, `min_weight`)
/// per neurotransmitter type.
fn biological_params(t: SynapseType) -> (u16, u16, i16, i16) {
    match t {
        SynapseType::Excitatory => (500, 5_000, 2000, 0),      // AMPA
        SynapseType::Inhibitory => (300, 10_000, 0, -2000),    // GABA
        SynapseType::Modulatory => (1000, 50_000, 1000, -1000), // Dopamine etc.
    }
}

/// Spike-Timing-Dependent Plasticity rule.
///
/// The audit flagged this as **the better STDP implementation** (vs `stdp_plasticity.rs`):
/// real exp-decay via fixed-point linear approximation, sign-correct (positive dt → LTD,
/// negative dt → LTP).
#[derive(Debug, Clone)]
pub struct STDPRule {
    /// LTP time constant (μs) — pre-before-post window.
    pub tau_plus_us: u32,
    /// LTD time constant (μs) — post-before-pre window.
    pub tau_minus_us: u32,
    /// LTP amplitude (fixed-point, 1000 = 1.0).
    pub a_plus: i16,
    /// LTD amplitude (negative).
    pub a_minus: i16,
    /// Overall learning rate (fixed-point, 1000 = 1.0).
    pub learning_rate: u16,
}

impl STDPRule {
    /// Default biological STDP: 20ms/20ms tau, asymmetric amplitudes (LTD slightly larger).
    #[must_use]
    pub fn new() -> Self {
        Self {
            tau_plus_us: 20_000,
            tau_minus_us: 20_000,
            a_plus: 50,
            a_minus: -53, // Slightly larger |LTD| than LTP — asymmetry is biological
            learning_rate: 100,
        }
    }

    /// Compute the weight delta for a given spike-time difference.
    ///
    /// # Parameters
    ///
    /// - `dt_us = pre_spike_time - post_spike_time` (positive = post fired first → LTD).
    ///
    /// # Returns
    ///
    /// Weight delta (signed): non-negative for LTP, non-positive for LTD, zero outside the window.
    ///
    /// # Invariants
    ///
    /// - `dt > 0` → returns ≤ 0 (LTD)
    /// - `dt < 0` → returns ≥ 0 (LTP)
    /// - `|dt| ≥ 10·tau` → returns 0 (decay floor)
    ///
    /// # Bug fix vs v0.1
    ///
    /// v0.1 used `factor = SCALE - decay` without clamping. When `|dt| > tau`,
    /// `decay > SCALE` and `factor` goes negative, breaking the sign convention
    /// (LTP returns negative, LTD returns positive). The property test
    /// `prop_stdp_sign_convention` catches this at `dt = -24000`. Fix: clamp
    /// `factor` to `≥ 0` — the linear exp approximation naturally floors at zero.
    #[must_use]
    pub fn calculate_weight_change(&self, dt_us: i32) -> i16 {
        if dt_us > 0 {
            // Post fired before pre → LTD (depress the synapse).
            let decay = (dt_us.abs() * SCALE) / self.tau_minus_us as i32;
            if decay < 10_000 {
                let factor = (SCALE - decay).max(0); // Clamp to ≥ 0
                ((i32::from(self.a_minus) * factor * i32::from(self.learning_rate))
                    / (SCALE * SCALE)) as i16
            } else {
                0
            }
        } else {
            // Pre fired before post (or simultaneous) → LTP (potentiate).
            let decay = (dt_us.abs() * SCALE) / self.tau_plus_us as i32;
            if decay < 10_000 {
                let factor = (SCALE - decay).max(0); // Clamp to ≥ 0
                ((i32::from(self.a_plus) * factor * i32::from(self.learning_rate))
                    / (SCALE * SCALE)) as i16
            } else {
                0
            }
        }
    }
}

impl Default for STDPRule {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for custom synapse configurations.
#[derive(Debug)]
pub struct SynapseBuilder {
    synapse: Synapse,
}

impl SynapseBuilder {
    pub fn new(pre_id: u16, post_id: u16, weight: i16) -> Result<Self, Error> {
        Ok(Self {
            synapse: Synapse::new(pre_id, post_id, weight)?,
        })
    }

    #[must_use]
    pub fn tau_decay_us(mut self, tau: u16) -> Self {
        self.synapse.tau_decay_us = tau;
        self
    }

    #[must_use]
    pub fn weight_bounds(mut self, min: i16, max: i16) -> Self {
        self.synapse.min_weight = min;
        self.synapse.max_weight = max;
        self
    }

    #[must_use]
    pub fn synapse_type(mut self, t: SynapseType) -> Self {
        let (tau_rise, tau_decay, max_w, min_w) = biological_params(t);
        self.synapse.synapse_type = t;
        self.synapse.tau_rise_us = tau_rise;
        self.synapse.tau_decay_us = tau_decay;
        self.synapse.max_weight = max_w;
        self.synapse.min_weight = min_w;
        self
    }

    #[must_use]
    pub fn build(self) -> Synapse {
        self.synapse
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::shadow_unrelated)]
    use super::*;
    use proptest::prelude::*;

    // ----- Unit tests -----

    #[test]
    fn new_excitatory_synapse_has_default_params() {
        let s = Synapse::new(1, 2, 100).expect("valid ids");
        assert_eq!(s.pre_neuron_id, 1);
        assert_eq!(s.post_neuron_id, 2);
        assert_eq!(s.weight, 100);
        assert_eq!(s.synapse_type, SynapseType::Excitatory);
        assert_eq!(s.tau_rise_us, 500); // AMPA
        assert_eq!(s.tau_decay_us, 5_000);
        assert_eq!(s.max_weight, 2000);
        assert_eq!(s.min_weight, 0);
    }

    #[test]
    fn negative_weight_makes_inhibitory() {
        let s = Synapse::new(1, 2, -100).expect("valid ids");
        assert_eq!(s.synapse_type, SynapseType::Inhibitory);
        assert_eq!(s.tau_rise_us, 300); // GABA
        assert_eq!(s.tau_decay_us, 10_000);
        assert_eq!(s.max_weight, 0);
        assert_eq!(s.min_weight, -2000);
    }

    #[test]
    fn self_connection_rejected() {
        let err = Synapse::new(5, 5, 100).unwrap_err();
        assert_eq!(err, Error::InvalidParameter);
    }

    #[test]
    fn builder_overrides_params() {
        let s = SynapseBuilder::new(1, 2, 150)
            .expect("valid ids")
            .weight_bounds(-500, 500)
            .build();
        assert_eq!(s.max_weight, 500);
        assert_eq!(s.min_weight, -500);
    }

    #[test]
    fn builder_sets_modulatory_type() {
        let s = SynapseBuilder::new(1, 2, 50)
            .expect("valid ids")
            .synapse_type(SynapseType::Modulatory)
            .build();
        assert_eq!(s.synapse_type, SynapseType::Modulatory);
        assert_eq!(s.tau_decay_us, 50_000);
    }

    #[test]
    fn weight_clamped_at_bounds() {
        let mut s = Synapse::new(1, 2, 100).expect("valid ids");
        s.update_weight(10_000); // Way over max
        assert_eq!(s.weight, s.max_weight);
        s.update_weight(-10_000); // Way under min
        assert_eq!(s.weight, s.min_weight);
    }

    // ----- STDP rule tests -----

    #[test]
    fn stdp_ltp_when_pre_before_post() {
        // dt < 0 means pre fired before post → LTP (positive delta).
        let rule = STDPRule::new();
        let dt_us: i32 = -5_000; // Pre 5ms before post
        let delta = rule.calculate_weight_change(dt_us);
        assert!(delta >= 0, "pre-before-post must produce LTP (>=0), got {delta}");
    }

    #[test]
    fn stdp_ltd_when_post_before_pre() {
        // dt > 0 means post fired before pre → LTD (negative delta).
        let rule = STDPRule::new();
        let dt_us: i32 = 5_000;
        let delta = rule.calculate_weight_change(dt_us);
        assert!(delta <= 0, "post-before-pre must produce LTD (<=0), got {delta}");
    }

    #[test]
    fn stdp_zero_outside_window() {
        let rule = STDPRule::new();
        // 10× tau_plus = 200_000 μs → outside window
        let far_dt: i32 = -200_000;
        assert_eq!(rule.calculate_weight_change(far_dt), 0);
        let far_dt_pos: i32 = 200_000;
        assert_eq!(rule.calculate_weight_change(far_dt_pos), 0);
    }

    #[test]
    fn stdp_zero_at_zero_dt() {
        // dt = 0 → simultaneous; LTP branch with decay = 0 → factor = SCALE → a_plus · lr.
        let rule = STDPRule::new();
        let delta = rule.calculate_weight_change(0);
        let expected = (i32::from(rule.a_plus) * SCALE * i32::from(rule.learning_rate))
            / (SCALE * SCALE);
        assert_eq!(delta, expected as i16);
    }

    #[test]
    fn stdp_decay_monotonic_with_abs_dt() {
        // Larger |dt| → smaller |delta| (within window).
        let rule = STDPRule::new();
        let small = rule.calculate_weight_change(-1_000).abs();
        let large = rule.calculate_weight_change(-10_000).abs();
        assert!(
            large <= small,
            "|delta| must decay with |dt|: small={small}, large={large}"
        );
    }

    // ----- Property tests (Cardano-grade rigor) -----

    proptest! {
        /// Self-connection always rejected, regardless of weight.
        #[test]
        fn prop_self_connection_always_rejected(id in 0u16..=1000, weight in -3000i16..=3000) {
            let result = Synapse::new(id, id, weight);
            prop_assert!(result.is_err());
        }

        /// Weight stays within `[min_weight, max_weight]` after any update.
        #[test]
        fn prop_weight_clamped(
            weight in -2000i16..=2000,
            delta in -5000i16..=5000,
        ) {
            let mut s = Synapse::new(1, 2, weight).unwrap_or_else(|_| {
                Synapse::new(1, 2, 0).expect("fallback synapse")
            });
            s.update_weight(delta);
            prop_assert!(s.weight >= s.min_weight);
            prop_assert!(s.weight <= s.max_weight);
        }

        /// STDP sign convention holds across random dt.
        #[test]
        fn prop_stdp_sign_convention(dt_us in -200_000i32..=200_000) {
            let rule = STDPRule::new();
            let delta = rule.calculate_weight_change(dt_us);
            if dt_us > 0 {
                prop_assert!(delta <= 0, "post-before-pre must produce LTD");
            } else if dt_us < 0 {
                prop_assert!(delta >= 0, "pre-before-post must produce LTP");
            }
            // dt == 0 → LTP branch (>= 0), tested separately.
        }

        /// STDP decays to zero outside 10× tau window.
        #[test]
        fn prop_stdp_zero_outside_window(multiplier in 11u32..=100) {
            let rule = STDPRule::new();
            let far_dt_pos = (rule.tau_minus_us * multiplier) as i32;
            let far_dt_neg = -((rule.tau_plus_us * multiplier) as i32);
            prop_assert_eq!(rule.calculate_weight_change(far_dt_pos), 0);
            prop_assert_eq!(rule.calculate_weight_change(far_dt_neg), 0);
        }
    }
}
