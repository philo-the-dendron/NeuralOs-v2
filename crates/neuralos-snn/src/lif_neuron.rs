//! Leaky Integrate-and-Fire (LIF) neuron — clean port from v0.1.
//!
//! # Invariants (testable)
//!
//! - After [`LIFNeuron::reset`], membrane potential = resting potential, no currents,
//!   no refractory, no history.
//! - After a spike, the neuron is refractory for exactly `tau_refractory_us`.
//! - Membrane potential is always in `[``MEMBRANE_MV_MIN``, ``MEMBRANE_MV_MAX``]` mV.
//! - Firing rate cannot exceed `1_000_000 / tau_refractory_us` Hz (refractory bound).
//! - With zero input and large dt, membrane → resting (leak convergence).
//!
//! # Fixed-point convention
//!
//! - Membrane potential: `i16` in the neuron's voltage quanta — mV in
//!   [`VoltageResolution::Millivolt`] (default, `-100..=50`), centi-mV in
//!   [`VoltageResolution::CentiMillivolt`] (`-10_000..=5_000`)
//! - Time: `u32` microseconds (no float in hot path)
//! - Current: `i16` μA
//! - Resistance: `u16` MΩ
//! - Capacitance: `u16` pF
//!
//! # The voltage grid and the dead zone (why resolution is configurable)
//!
//! `delta_v = dt_over_tau · (leak + R·I/1000) / 1000` truncates to whole
//! quanta. On the default mV grid at default params (dt=1 ms, R=100 MΩ), a
//! steady total current at rest below ~**200 μA (E, τ=20 ms)** or ~**100 μA
//! (I, τ=10 ms)** — the dead zone scales as `τ`/`dt_over_tau` — moves the
//! membrane exactly zero, forever. The ternary substrate's recurrent pulses
//! (weight/divisor, default 10 = ±12 μA at γ=125) sit deep inside that blindness.
//! [`VoltageResolution::CentiMillivolt`] shrinks both dead zones 100× with
//! the same `i16`, no float, and bit-identical arithmetic shape; the mV
//! default exists so every recorded result in the lineage keeps its exact
//! numbers. (Session F addendum: coherent multi-spike pulse SUMS can cross
//! a sub-threshold margin even on the mV grid — the dead zone binds single
//! small currents, not stacked ones; see the post-fix sweep evidence. The
//! operative firing barrier is a CLIMB condition, not the rest dead zone:
//! the last 1 mV step into threshold needs a ~20 mV gap to `V_ss` (E-type,
//! dt=1 ms) — sustained effective current ≳ 340 μA — which is why the
//! sweep's firing cliff sits between 450 and 300 μA. And at a sticking
//! point truncation RECTIFIES: one +12 μA pulse ratchets the membrane
//! +1 mV while a −12 μA pulse is absorbed — pinned by
//! `rectification_at_the_sticking_point`.)
//!
//! # `no_std`
//!
//! Uses `heapless::Vec` for spike history (compile-time capacity, no allocator).
//! Fully `no_std`-compatible for embedded RISC-V deployment.
//!
//! # Bug fixes vs v0.1
//!
//! - **Time ownership**: v0.1's `current_time()` returned `last_spike_time_us`,
//!   so `firing_rate_hz` windows never advanced between spikes. Now the caller
//!   passes `current_time_us` to `integrate_and_fire`. The neuron stores it as
//!   `last_update_time_us`, used as the firing-rate reference.
//! - **Noise seed**: v0.1 seeded the LFSR only by `id`, so every neuron with the
//!   same id produced identical "noise" forever. Now seeded by `id XOR time`.

#![allow(clippy::module_name_repetitions)]
// Fixed-point design: types are intentionally narrow (i16 mV, u32 μs, i16 μA, u8 amplitude).
// Casts between them are part of the design and bounded by physics:
//   - dt_us ≤ ~2^31 μs (35 min) — realistic sim step ceiling
//   - membrane_potential in [-100, 50] — clamped after every integration
//   - spike intervals bounded by u32 — sim runs ≤ ~71 min before any risk
// Allow clippy's cast lints for the hot path; every narrowing cast sits after a clamp
// or against a documented bound.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use heapless::Vec;

/// Maximum spike history length (compile-time, no allocator).
pub const MAX_SPIKE_HISTORY: usize = 64;

/// Membrane potential lower bound (mV) — biological floor.
pub const MEMBRANE_MV_MIN: i16 = -100;

/// Membrane potential upper bound (mV) — biological ceiling.
pub const MEMBRANE_MV_MAX: i16 = 50;

/// Voltage-domain resolution of a neuron's stored potentials.
///
/// The membrane/threshold/resting/reset fields hold quanta of this grid.
/// See the module doc's "dead zone" section for why this exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VoltageResolution {
    /// 1 mV grid — the historical default. Dead zone ≈ 200 μA at rest.
    #[default]
    Millivolt,
    /// 0.01 mV (10 μV) grid — 100× finer. Dead zone ≈ 2 μA; the ternary
    /// substrate's ±12 μA recurrent pulses become visible membrane motion.
    CentiMillivolt,
}

impl VoltageResolution {
    /// Quanta per millivolt: 1 (mV grid) or 100 (centi-mV grid).
    #[must_use]
    pub const fn scale(self) -> i32 {
        match self {
            Self::Millivolt => 1,
            Self::CentiMillivolt => 100,
        }
    }
}

/// Biological neuron classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NeuronType {
    /// Excitatory pyramidal neuron (~80% of cortical neurons).
    #[default]
    Excitatory,
    /// Inhibitory interneuron (~20% of cortical neurons).
    Inhibitory,
}

/// LIF neuron — fixed-point math, `no_std`-compatible, bounded spike history.
#[derive(Debug, Clone)]
pub struct LIFNeuron {
    /// Unique identifier within the network.
    pub id: u16,
    /// Biological classification (excitatory/inhibitory).
    pub neuron_type: NeuronType,

    /// Current membrane potential, in this neuron's voltage quanta (mV or
    /// centi-mV — see [`voltage_resolution`](Self::voltage_resolution)).
    /// Bounded by `[MEMBRANE_MV_MIN, MEMBRANE_MV_MAX]` scaled to the grid.
    pub membrane_potential: i16,
    /// Resting potential (mV in Millivolt mode, centi-mV in `CentiMillivolt`).
    /// Typically `-70`.
    pub resting_potential: i16,
    /// Spike threshold (quanta per the resolution). `-55` mV excitatory,
    /// `-50` mV inhibitory.
    pub threshold: i16,
    /// Reset potential after spike (quanta per the resolution). `-80` mV.
    pub reset_potential: i16,

    /// The voltage grid the four potentials above live on.
    pub voltage_resolution: VoltageResolution,

    /// Membrane time constant (μs). Determines integration speed.
    pub tau_membrane_us: u32,
    /// Refractory period (μs). Prevents immediate re-firing.
    pub tau_refractory_us: u32,
    /// Remaining refractory time (μs). `0` when not refractory.
    pub refractory_time_us: u32,
    /// Last simulation time the neuron was updated (μs).
    /// Owned by the caller via `integrate_and_fire(current_time_us)`.
    pub last_update_time_us: u32,
    /// Timestamp of the last spike (μs).
    pub last_spike_time_us: u32,
    /// Accumulated synaptic current (μA).
    pub synaptic_current_ua: i16,

    /// Membrane capacitance (pF).
    pub capacitance_pf: u16,
    /// Membrane resistance (MΩ).
    pub resistance_mohm: u16,

    /// Noise amplitude (μA). `0` = silent.
    pub noise_amplitude_ua: u8,
    /// Spike-frequency adaptation current (μA). Decays between spikes, jumps on each spike.
    pub adaptation_current_ua: i16,

    /// Spike timestamps (chronological order, oldest at index 0). Bounded by `MAX_SPIKE_HISTORY`.
    spike_history: Vec<u32, MAX_SPIKE_HISTORY>,
}

impl LIFNeuron {
    /// New excitatory neuron with default biological parameters.
    #[must_use]
    pub fn new(id: u16) -> Self {
        Self::new_with_type(id, NeuronType::Excitatory)
    }

    /// New neuron with specific biological type. Type sets threshold, tau, capacitance.
    /// Voltage grid = mV (the historical default).
    #[must_use]
    pub fn new_with_type(id: u16, neuron_type: NeuronType) -> Self {
        Self::new_with_type_resolution(id, neuron_type, VoltageResolution::Millivolt)
    }

    /// New neuron with a chosen biological type AND voltage resolution.
    ///
    /// All four potentials are constructed on the chosen grid (e.g. resting
    /// `-70` mV / `-7_000` centi-mV). Currents, times, and history are
    /// grid-independent.
    #[must_use]
    pub fn new_with_type_resolution(
        id: u16,
        neuron_type: NeuronType,
        resolution: VoltageResolution,
    ) -> Self {
        let (threshold_mv, tau_membrane_us, capacitance_pf) = match neuron_type {
            NeuronType::Excitatory => (-55, 20_000, 100), // 20 ms tau, pyramidal
            NeuronType::Inhibitory => (-50, 10_000, 80), // 10 ms tau, interneuron
        };
        let s = resolution.scale();
        Self {
            id,
            neuron_type,
            membrane_potential: (-70 * s) as i16,
            resting_potential: (-70 * s) as i16,
            threshold: (threshold_mv * s) as i16,
            reset_potential: (-80 * s) as i16,
            voltage_resolution: resolution,
            tau_membrane_us,
            tau_refractory_us: 2_000,
            refractory_time_us: 0,
            last_update_time_us: 0,
            last_spike_time_us: 0,
            synaptic_current_ua: 0,
            capacitance_pf,
            resistance_mohm: 100,
            noise_amplitude_ua: 5,
            adaptation_current_ua: 0,
            spike_history: Vec::new(),
        }
    }

    /// Switch the voltage grid in place, rescaling the four stored potentials
    /// (values are preserved exactly: ×100 or ÷100 on whole-quanta values).
    /// Call before stepping; spikes/history/currents are grid-independent.
    pub fn set_voltage_resolution(&mut self, resolution: VoltageResolution) {
        if resolution == self.voltage_resolution {
            return;
        }
        let new_s = resolution.scale();
        let old_s = self.voltage_resolution.scale();
        let rescale = |v: i16| -> i16 { ((i32::from(v) * new_s) / old_s) as i16 };
        self.membrane_potential = rescale(self.membrane_potential);
        self.resting_potential = rescale(self.resting_potential);
        self.threshold = rescale(self.threshold);
        self.reset_potential = rescale(self.reset_potential);
        self.voltage_resolution = resolution;
    }

    /// Integrate the membrane equation for one time step, return `true` if a spike fired.
    ///
    /// # Parameters
    ///
    /// - `input_current_ua`: external input current (μA)
    /// - `dt_us`: time step (μs)
    /// - `current_time_us`: simulation time (μs) — **owned by the caller**, not the neuron
    ///
    /// # Bug fix vs v0.1
    ///
    /// v0.1's `current_time()` returned `last_spike_time_us`, so firing-rate windows
    /// never advanced between spikes. Time is now passed in explicitly.
    pub fn integrate_and_fire(
        &mut self,
        input_current_ua: i16,
        dt_us: u32,
        current_time_us: u32,
    ) -> bool {
        self.last_update_time_us = current_time_us;

        // Refractory period: count down, no integration.
        if self.refractory_time_us > 0 {
            self.refractory_time_us = self.refractory_time_us.saturating_sub(dt_us);
            return false;
        }

        // Combine input + synaptic + noise − adaptation (all saturating).
        let noise = self.generate_noise(current_time_us);
        let total_current = input_current_ua
            .saturating_add(self.synaptic_current_ua)
            .saturating_add(noise)
            .saturating_sub(self.adaptation_current_ua);

        // LIF equation: dV/dt = (V_rest − V + R·I) / τ
        // Discretized:   ΔV = (dt/τ) · (V_rest − V + R·I)
        // Fixed-point:   dt_over_tau = (dt · 1000) / τ      (scaled by 1000)
        //                delta_v      = dt_over_tau · (leak + current) / 1000
        // Voltage quanta: leak is stored-native; the current term converts
        // μA → quanta (R·I/1000 mV, ×scale for the grid). At scale = 1 the
        // expression sequence is byte-identical to the historical arithmetic.
        let s = self.voltage_resolution.scale();
        let dt_over_tau = ((dt_us as i32) * 1000) / self.tau_membrane_us as i32;
        let leak_term = i32::from(self.resting_potential - self.membrane_potential);
        let current_term = (i32::from(total_current) * i32::from(self.resistance_mohm) * s) / 1000;
        let delta_v = (dt_over_tau * (leak_term + current_term)) / 1000;

        let new_v = i32::from(self.membrane_potential)
            .saturating_add(delta_v)
            .clamp(i32::from(MEMBRANE_MV_MIN) * s, i32::from(MEMBRANE_MV_MAX) * s);
        self.membrane_potential = new_v as i16;

        if self.membrane_potential >= self.threshold {
            self.spike(current_time_us);
            true
        } else {
            false
        }
    }

    /// Record a spike: reset potential, enter refractory, store timestamp, adapt.
    fn spike(&mut self, current_time_us: u32) {
        self.membrane_potential = self.reset_potential;
        self.refractory_time_us = self.tau_refractory_us;
        self.last_spike_time_us = current_time_us;
        // Drop oldest if at capacity (we want recent spikes, not the first ones).
        if self.spike_history.is_full() {
            self.spike_history.remove(0);
        }
        let _ = self.spike_history.push(current_time_us);
        // Spike-frequency adaptation accumulates per spike.
        self.adaptation_current_ua = self.adaptation_current_ua.saturating_add(2);
    }

    /// Accumulate synaptic current (μA). Saturating — never overflows.
    pub fn add_synaptic_current(&mut self, current_ua: i16) {
        self.synaptic_current_ua = self.synaptic_current_ua.saturating_add(current_ua);
    }

    /// Zero out synaptic current. The network calls this AFTER integration
    /// has read the accumulator (session F's clear-after-read fix) — Phase 2
    /// then injects fresh pulses for the next step to integrate.
    pub fn clear_synaptic_current(&mut self) {
        self.synaptic_current_ua = 0;
    }

    /// Decay the spike-frequency adaptation current by 1 μA (linear).
    ///
    /// Called once per step by the network BEFORE integration — without it,
    /// adaptation accumulates (+2/spike) unboundedly and silences the
    /// network. Phase-identical to the historical opening-loop position.
    ///
    /// (Session F split: the historical `decay_synaptic_current` also
    /// decayed the synaptic accumulator, but that half was structurally
    /// dead in `step` since the port — the accumulator was always cleared
    /// before its only read. Post-fix the network clears after the read, so
    /// there is nothing to decay: pulses live exactly one step at full
    /// amplitude, then are integrated and zeroed. The decorative
    /// `tau_synapse_us` field was removed at alpha.3 — F5a.)
    pub fn decay_adaptation_current(&mut self) {
        if self.adaptation_current_ua > 0 {
            self.adaptation_current_ua = self.adaptation_current_ua.saturating_sub(1);
        }
    }

    /// Reset neuron to its initial state (resting potential, no currents, no history).
    pub fn reset(&mut self) {
        self.membrane_potential = self.resting_potential;
        self.refractory_time_us = 0;
        self.last_update_time_us = 0;
        self.last_spike_time_us = 0;
        self.synaptic_current_ua = 0;
        self.adaptation_current_ua = 0;
        self.spike_history.clear();
    }

    /// Is the neuron currently in its refractory period?
    #[must_use]
    pub fn is_refractory(&self) -> bool {
        self.refractory_time_us > 0
    }

    /// Firing rate within a recent window, in **millihertz** (no float in hot path).
    ///
    /// Returns `spikes/sec × 1000`. For Hz: `rate_mhz as f32 / 1000.0`.
    ///
    /// Reference time = `last_update_time_us` (set by `integrate_and_fire`).
    ///
    /// # Bug fix vs v0.1
    ///
    /// v0.1 used `last_spike_time_us` as reference, which never advanced between
    /// spikes, breaking the window filter. Now uses `last_update_time_us`.
    #[must_use]
    pub fn firing_rate_mhz(&self, window_us: u32) -> u32 {
        if self.spike_history.is_empty() || window_us == 0 {
            return 0;
        }
        let window_start = self.last_update_time_us.saturating_sub(window_us);
        let count = self
            .spike_history
            .iter()
            .filter(|&&t| t >= window_start)
            .count() as u32;
        // spikes/sec × 1000 = (count × 1_000_000_000) / window_us
        count
            .saturating_mul(1_000_000_000)
            .checked_div(window_us)
            .unwrap_or(0)
    }

    /// Inter-spike-interval statistics: `(mean_us, std_dev_us)`, or `None` if `< 2` spikes.
    ///
    /// Pure integer math (no float, `no_std`-safe). Std-dev via the sum-of-squares
    /// formula; for very long spike trains, consider computing stats at network level.
    #[must_use]
    pub fn isi_stats_us(&self) -> Option<(u32, u32)> {
        if self.spike_history.len() < 2 {
            return None;
        }
        let n = self.spike_history.len();
        let mut intervals_sum: u64 = 0;
        let mut intervals_sqsum: u64 = 0;
        let mut count: u64 = 0;
        for i in 1..n {
            let prev = self.spike_history[i - 1];
            let curr = self.spike_history[i];
            if curr >= prev {
                let d = u64::from(curr - prev);
                intervals_sum += d;
                intervals_sqsum += d * d;
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        let mean = intervals_sum / count;
        // Variance = E[x²] − E[x]²
        let mean_sq = mean * mean;
        let sq_mean = intervals_sqsum / count;
        let variance = sq_mean.saturating_sub(mean_sq);
        let std = isqrt_u64(variance);
        Some((mean as u32, std as u32))
    }

    /// Iterate over spike timestamps (oldest first).
    pub fn spikes(&self) -> impl Iterator<Item = &u32> {
        self.spike_history.iter()
    }

    /// Number of recorded spikes (bounded by `MAX_SPIKE_HISTORY`).
    #[must_use]
    pub fn spike_count(&self) -> usize {
        self.spike_history.len()
    }

    /// Deterministic LFSR noise seeded by `id XOR current_time_us`.
    ///
    /// # Bug fix vs v0.1
    ///
    /// v0.1 seeded only by `id`, so every neuron with the same id produced
    /// identical "noise" forever (deterministic, not noise). Now seeded by
    /// `id XOR current_time_us` — different each step, still reproducible for tests.
    fn generate_noise(&self, current_time_us: u32) -> i16 {
        if self.noise_amplitude_ua == 0 {
            return 0;
        }
        let mut lfsr = u32::from(self.id) ^ current_time_us;
        // 16-bit Galois LFSR, taps 0xB400, period 65_535. Iterate 4x for diffusion.
        for _ in 0..4 {
            lfsr = (lfsr >> 1) ^ (if lfsr & 1 != 0 { 0xB400_u32 } else { 0 });
        }
        let raw = (lfsr & 0xFF) as i16 - 128; // -128..=127
        (raw * i16::from(self.noise_amplitude_ua)) / 128
    }
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Builder for custom neuron configurations.
#[derive(Debug)]
pub struct NeuronBuilder {
    neuron: LIFNeuron,
}

impl NeuronBuilder {
    #[must_use]
    pub fn new(id: u16) -> Self {
        Self {
            neuron: LIFNeuron::new(id),
        }
    }

    #[must_use]
    pub fn neuron_type(mut self, t: NeuronType) -> Self {
        let id = self.neuron.id;
        self.neuron = LIFNeuron::new_with_type(id, t);
        self
    }

    /// Threshold in mV — stored on the neuron's grid (×scale), so the
    /// builder's contract stays mV in every resolution.
    #[must_use]
    pub fn threshold_mv(mut self, threshold: i16) -> Self {
        let s = self.neuron.voltage_resolution.scale();
        self.neuron.threshold = threshold * s as i16;
        self
    }

    /// Switch the voltage grid, rescaling all four stored potentials
    /// (preserves values; order-independent with the other setters).
    #[must_use]
    pub fn voltage_resolution(mut self, resolution: VoltageResolution) -> Self {
        self.neuron.set_voltage_resolution(resolution);
        self
    }

    #[must_use]
    pub fn tau_membrane_us(mut self, tau: u32) -> Self {
        self.neuron.tau_membrane_us = tau;
        self
    }

    #[must_use]
    pub fn tau_refractory_us(mut self, tau: u32) -> Self {
        self.neuron.tau_refractory_us = tau;
        self
    }

    #[must_use]
    pub fn capacitance_pf(mut self, c: u16) -> Self {
        self.neuron.capacitance_pf = c;
        self
    }

    #[must_use]
    pub fn resistance_mohm(mut self, r: u16) -> Self {
        self.neuron.resistance_mohm = r;
        self
    }

    #[must_use]
    pub fn noise_amplitude_ua(mut self, a: u8) -> Self {
        self.neuron.noise_amplitude_ua = a;
        self
    }

    #[must_use]
    pub fn build(self) -> LIFNeuron {
        self.neuron
    }
}

/// Integer square root (`no_std`-safe, no float).
#[inline]
fn isqrt_u64(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = u64::midpoint(x, n / x);
    }
    x
}

#[cfg(test)]
mod tests {
    #![allow(clippy::shadow_unrelated)]
    use super::*;
    use proptest::prelude::*;

    // ----- Unit tests (ported + fixed from v0.1) -----

    #[test]
    fn new_excitatory_neuron_has_default_params() {
        let n = LIFNeuron::new(1);
        assert_eq!(n.id, 1);
        assert_eq!(n.neuron_type, NeuronType::Excitatory);
        assert_eq!(n.membrane_potential, -70);
        assert_eq!(n.threshold, -55);
        assert_eq!(n.tau_membrane_us, 20_000);
    }

    #[test]
    fn inhibitory_neuron_has_different_params() {
        let n = LIFNeuron::new_with_type(2, NeuronType::Inhibitory);
        assert_eq!(n.threshold, -50);
        assert_eq!(n.tau_membrane_us, 10_000);
    }

    #[test]
    fn builder_overrides_params() {
        let n = NeuronBuilder::new(3)
            .neuron_type(NeuronType::Inhibitory)
            .threshold_mv(-45)
            .tau_membrane_us(15_000)
            .build();
        assert_eq!(n.id, 3);
        assert_eq!(n.neuron_type, NeuronType::Inhibitory);
        assert_eq!(n.threshold, -45);
        assert_eq!(n.tau_membrane_us, 15_000);
    }

    #[test]
    fn builder_threshold_is_mv_in_every_resolution() {
        let n = NeuronBuilder::new(9)
            .voltage_resolution(VoltageResolution::CentiMillivolt)
            .threshold_mv(-45)
            .build();
        assert_eq!(n.threshold, -4_500, "−45 mV stored as centi-mV");
        assert_eq!(n.resting_potential, -7_000);
    }

    #[test]
    fn resolution_switch_rescales_stored_potentials() {
        let mut n = LIFNeuron::new_with_type_resolution(
            10,
            NeuronType::Excitatory,
            VoltageResolution::CentiMillivolt,
        );
        n.membrane_potential = -6_000;
        n.set_voltage_resolution(VoltageResolution::Millivolt);
        assert_eq!(n.membrane_potential, -60);
        assert_eq!(n.threshold, -55);
        assert_eq!(n.reset_potential, -80);
    }

    #[test]
    fn positive_current_raises_membrane_potential() {
        let mut n = LIFNeuron::new(4);
        let initial = n.membrane_potential;
        // 10 ms step, 100 μA — biologically realistic depolarization.
        // (1 ms / 50 μA loses precision in the fixed-point math: dt_over_tau=50,
        // current_term=5, delta_v=(50*5)/1000=0. Documented behavior.)
        let spiked = n.integrate_and_fire(100, 10_000, 10_000);
        assert!(
            n.membrane_potential > initial,
            "membrane should rise with sustained positive input (was {}, now {})",
            initial,
            n.membrane_potential
        );
        assert!(!spiked, "single 10 ms step at 100 μA should not spike");
    }

    // ----- Voltage resolution: the dead zone, demonstrated and pinned -----

    /// Noise-off helper (deterministic traces for hand-derived expectations).
    fn quiet_neuron(id: u16, r: VoltageResolution) -> LIFNeuron {
        let mut n = LIFNeuron::new_with_type_resolution(id, NeuronType::Excitatory, r);
        n.noise_amplitude_ua = 0;
        n
    }

    #[test]
    fn millivolt_trace_is_pinned_to_the_historical_arithmetic() {
        // Hand-derived from the ORIGINAL formula (mV grid, E neuron,
        // dt=1000 μs, τ=20 ms → dt_over_tau=50, R=100 → ct=I/10):
        //   I=600: m −70 → ct 60 → ΔV=50·60/1000=3 → −67 | leak −3 → 2 → −65
        //   → −63 (leak −7 → 2) → −61 → −59 → −57 → −55 ⇒ SPIKE at step 7.
        let mut n = quiet_neuron(11, VoltageResolution::Millivolt);
        let expected = [-67, -65, -63, -61, -59, -57];
        for &want in &expected {
            let spiked = n.integrate_and_fire(600, 1000, 0);
            assert!(!spiked);
            assert_eq!(n.membrane_potential, want);
        }
        let spiked = n.integrate_and_fire(600, 1000, 0);
        assert!(spiked, "7th 600 μA step crosses −55");
        assert_eq!(n.membrane_potential, -80, "spike reset");
    }

    #[test]
    fn the_dead_zone_12ua_pair() {
        // THE demonstration: the ternary substrate's recurrent pulse
        // (±12 μA) is invisible on the mV grid and 6 quanta on centi-mV.
        let mut mv = quiet_neuron(12, VoltageResolution::Millivolt);
        mv.integrate_and_fire(12, 1000, 0);
        assert_eq!(mv.membrane_potential, -70, "mV grid: 12 μA ⇒ ΔV = 0");

        let mut cmv = quiet_neuron(13, VoltageResolution::CentiMillivolt);
        cmv.integrate_and_fire(12, 1000, 0);
        // ct = 12·100·100/1000 = 120 cV; ΔV = 50·120/1000 = 6 cV.
        assert_eq!(cmv.membrane_potential, -7_000 + 6, "centi grid: 12 μA ⇒ 6 cV");
    }

    #[test]
    fn above_threshold_current_blind_on_mv_spikes_on_centi() {
        // 160 μA > the ~150 μA E threshold current (V_ss = −54 mV), yet the
        // mV grid truncates every step to 0 from rest — silent forever.
        // Centi-mV climbs to −54 mV and fires. The ruler, not the cable.
        let mut mv = quiet_neuron(14, VoltageResolution::Millivolt);
        let mut mv_spikes = 0;
        for t in 0..100 {
            if mv.integrate_and_fire(160, 1000, t) {
                mv_spikes += 1;
            }
        }
        assert_eq!(mv_spikes, 0, "mV grid is blind to 160 μA from rest");
        assert_eq!(mv.membrane_potential, -70);

        let mut cmv = quiet_neuron(15, VoltageResolution::CentiMillivolt);
        let mut cmv_spikes = 0;
        for t in 0..100 {
            if cmv.integrate_and_fire(160, 1000, t) {
                cmv_spikes += 1;
            }
        }
        assert!(cmv_spikes >= 1, "centi grid fires on 160 μA (V_ss −54 mV)");
    }

    #[test]
    fn centi_mode_constants_and_bounds() {
        let n = quiet_neuron(16, VoltageResolution::CentiMillivolt);
        assert_eq!(n.membrane_potential, -7_000);
        assert_eq!(n.resting_potential, -7_000);
        assert_eq!(n.threshold, -5_500);
        assert_eq!(n.reset_potential, -8_000);
        // Clamp bounds scale with the grid (−10_000..=5_000).
        let mut m = quiet_neuron(17, VoltageResolution::CentiMillivolt);
        m.membrane_potential = -9_999;
        m.integrate_and_fire(-30_000, 1000, 0); // huge hyperpolarizing step
        assert!(m.membrane_potential >= -10_000, "clamped at scaled floor");
    }

    /// The climb barrier + rectification (session F, review-derived):
    /// a 300 μA-driven E neuron (mV grid) climbs +1 mV/step only while the
    /// gap to `V_ss` (−40 mV) is ≥ 20 mV — it STICKS at −59 (gap 19, delta
    /// truncates to 0), 4 mV short of threshold, forever. At the sticking
    /// point one +12 μA recurrent pulse ratchets +1 mV (19+1.2 = 20.2 → 1);
    /// one −12 μA pulse is absorbed (17.8 → 0). Positive excursions bind,
    /// negative ones are free — the rectifier that lets coherent volleys
    /// (and even incoherent positive noise) recruit neurons below the
    /// drive-only cliff.
    #[test]
    fn rectification_at_the_sticking_point() {
        let mut n = quiet_neuron(18, VoltageResolution::Millivolt);
        // Climb: delta = (leak + 30)/20; ≥1 while leak ≥ −10 (V ≤ −60);
        // at −59 leak = −11 → 19/20 = 0. Stuck.
        for t in 0..50 {
            n.integrate_and_fire(300, 1000, t);
        }
        assert_eq!(n.membrane_potential, -59, "sticks 4 mV under threshold");
        n.integrate_and_fire(300, 1000, 100);
        assert_eq!(n.membrane_potential, -59, "stays stuck under drive alone");

        // +12 μA pulse (a ternary +γ wire transmitting): +1 mV.
        n.add_synaptic_current(12);
        n.integrate_and_fire(300, 1000, 101);
        assert_eq!(n.membrane_potential, -58, "positive pulse ratchets +1 mV");

        // −12 μA pulse at the NEW sticking point (gap 18 → 18.8/20 → 0):
        // absorbed, no ratchet down.
        n.add_synaptic_current(-12);
        n.integrate_and_fire(300, 1000, 102);
        assert_eq!(n.membrane_potential, -58, "negative pulse absorbed");
    }

    #[test]
    fn large_current_triggers_spike_and_refractory() {
        let mut n = NeuronBuilder::new(5).threshold_mv(-65).build();
        // 200 μA over repeated 10 ms steps must eventually trigger a spike.
        let mut spiked = false;
        let mut t = 0_u32;
        for _ in 0..10 {
            if n.integrate_and_fire(200, 10_000, t) {
                spiked = true;
                break;
            }
            t = t.saturating_add(10_000);
        }
        assert!(spiked, "200 μA over 10 ms steps should eventually spike");
        assert_eq!(n.membrane_potential, n.reset_potential);
        assert!(n.is_refractory());
    }

    #[test]
    fn refractory_blocks_repeated_spikes() {
        let mut n = LIFNeuron::new(6);
        n.membrane_potential = n.threshold + 1;
        let spiked = n.integrate_and_fire(0, 1000, 1000);
        assert!(spiked);
        // Same step tries again — must be blocked by refractory.
        let again = n.integrate_and_fire(1000, 1000, 2000);
        assert!(!again, "must not spike during refractory");
    }

    #[test]
    fn synaptic_current_accumulates_then_clears() {
        let mut n = LIFNeuron::new(7);
        n.add_synaptic_current(30);
        n.add_synaptic_current(20);
        assert_eq!(n.synaptic_current_ua, 50);
        n.clear_synaptic_current();
        assert_eq!(n.synaptic_current_ua, 0);
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut n = LIFNeuron::new(8);
        n.membrane_potential = -50;
        n.synaptic_current_ua = 100;
        n.adaptation_current_ua = 10;
        // Force a spike via the private method by going through integrate path.
        n.spike(5000);
        n.spike(6000);
        assert_eq!(n.spike_count(), 2);
        n.reset();
        assert_eq!(n.membrane_potential, n.resting_potential);
        assert_eq!(n.synaptic_current_ua, 0);
        assert_eq!(n.adaptation_current_ua, 0);
        assert!(!n.is_refractory());
        assert_eq!(n.spike_count(), 0);
    }

    #[test]
    fn firing_rate_uses_last_update_time_not_last_spike() {
        // Regression for v0.1 bug: current_time() returned last_spike_time_us,
        // so the firing-rate window never advanced. Now last_update_time_us drives it.
        let mut n = LIFNeuron::new(9);
        // Spike at t = 1000.
        n.membrane_potential = n.threshold + 1;
        let _ = n.integrate_and_fire(0, 1000, 1000);
        // Step forward to t = 10_000_000 (10 sec) with no input.
        for step in 2..=10_000 {
            let _ = n.integrate_and_fire(0, 1000, step * 1000);
        }
        // 1-second window ending at t = 10_000_000: no spikes in last 1 s.
        let rate_mhz = n.firing_rate_mhz(1_000_000);
        assert_eq!(
            rate_mhz, 0,
            "no spikes in last 1 s — rate must be 0 (v0.1 would have returned > 0)"
        );
    }

    #[test]
    fn isi_stats_none_with_fewer_than_two_spikes() {
        let n = LIFNeuron::new(10);
        assert!(n.isi_stats_us().is_none());
    }

    #[test]
    fn isi_stats_computed_with_two_spikes() {
        let mut n = LIFNeuron::new(11);
        n.spike(1_000);
        n.spike(2_000); // ISI = 1000 μs
        let (mean, std) = n.isi_stats_us().expect("two spikes → Some");
        assert_eq!(mean, 1000);
        assert_eq!(std, 0);
    }

    #[test]
    fn noise_is_zero_when_amplitude_is_zero() {
        let n = NeuronBuilder::new(12).noise_amplitude_ua(0).build();
        for t in 0..1000_u32 {
            assert_eq!(n.generate_noise(t), 0);
        }
    }

    #[test]
    fn noise_varies_with_time_for_same_id() {
        // Regression for v0.1 bug: noise seeded only by id → constant per neuron.
        let n = LIFNeuron::new(13);
        let samples: [i16; 8] = [
            n.generate_noise(100),
            n.generate_noise(101),
            n.generate_noise(102),
            n.generate_noise(103),
            n.generate_noise(104),
            n.generate_noise(105),
            n.generate_noise(106),
            n.generate_noise(107),
        ];
        // At least two distinct values across 8 samples (not all identical).
        let first = samples[0];
        let all_same = samples.iter().all(|&s| s == first);
        assert!(
            !all_same,
            "noise must vary with time — got identical values across 8 samples (all = {first})"
        );
    }

    // ----- Property tests (Cardano-grade rigor) -----

    proptest! {
        /// After reset, membrane = resting, no currents, no refractory, no history.
        #[test]
        fn prop_reset_clears_all_state(id in 0u16..=1000) {
            let mut n = LIFNeuron::new(id);
            n.add_synaptic_current(50);
            n.adaptation_current_ua = 20;
            n.spike(5000);
            n.spike(6000);
            n.reset();
            prop_assert_eq!(n.membrane_potential, n.resting_potential);
            prop_assert_eq!(n.synaptic_current_ua, 0);
            prop_assert_eq!(n.adaptation_current_ua, 0);
            prop_assert_eq!(n.refractory_time_us, 0);
            prop_assert_eq!(n.spike_count(), 0);
        }

        /// Membrane potential always within biological bounds.
        #[test]
        fn prop_membrane_potential_stays_bounded(
            id in 0u16..=100,
            input in -1000i16..=1000,
            dt in 1u32..=10_000,
            start_t in 0u32..=1_000_000,
        ) {
            let mut n = LIFNeuron::new(id);
            let mut t = start_t;
            for _ in 0..100 {
                let _ = n.integrate_and_fire(input, dt, t);
                t = t.saturating_add(dt);
            }
            prop_assert!(n.membrane_potential >= MEMBRANE_MV_MIN);
            prop_assert!(n.membrane_potential <= MEMBRANE_MV_MAX);
        }

        /// Firing rate ≤ 1_000_000 / tau_refractory_us (refractory physical bound, +5% tolerance).
        #[test]
        fn prop_firing_rate_respects_refractory_bound(
            id in 0u16..=50,
            tau_ref in 500u32..=10_000,
        ) {
            let mut n = NeuronBuilder::new(id).tau_refractory_us(tau_ref).build();
            n.threshold = -100; // force spiking with any input
            let mut t = 0_u32;
            for _ in 0..100 {
                let _ = n.integrate_and_fire(1000, 1000, t);
                t += 1000;
            }
            let max_rate_mhz = 1_000_000_000_u32 / tau_ref;
            let ceiling = max_rate_mhz + max_rate_mhz / 20; // +5% for noise jitter
            let actual = n.firing_rate_mhz(t);
            prop_assert!(
                actual <= ceiling,
                "rate {} mHz exceeds refractory ceiling {} mHz",
                actual,
                ceiling
            );
        }

        /// Spike history never exceeds MAX_SPIKE_HISTORY.
        #[test]
        fn prop_history_never_exceeds_capacity(id in 0u16..=10) {
            let mut n = LIFNeuron::new(id);
            n.threshold = -100;
            for i in 0..1000_u32 {
                let _ = n.integrate_and_fire(1000, 1000, i * 1000);
            }
            prop_assert!(n.spike_count() <= MAX_SPIKE_HISTORY);
        }
    }

    // ----- 2026-08-20 substrate-audit pins (F1): the adaptation-decay
    // contract and leak convergence, previously claimed-but-untested. -----

    #[test]
    fn adaptation_decay_is_exact_minus_one_with_floor_zero() {
        let mut n = quiet_neuron(21, VoltageResolution::Millivolt);
        n.adaptation_current_ua = 3;
        n.decay_adaptation_current();
        assert_eq!(n.adaptation_current_ua, 2);
        n.decay_adaptation_current();
        assert_eq!(n.adaptation_current_ua, 1);
        n.decay_adaptation_current();
        assert_eq!(n.adaptation_current_ua, 0);
        n.decay_adaptation_current();
        assert_eq!(n.adaptation_current_ua, 0, "floors at zero, never negative");
    }

    #[test]
    fn spike_adds_exactly_two_adaptation_quanta() {
        let mut n = quiet_neuron(22, VoltageResolution::Millivolt);
        n.threshold = -100; // force a spike this step
        let spiked = n.integrate_and_fire(1000, 1000, 0);
        assert!(spiked);
        assert_eq!(n.adaptation_current_ua, 2, "+2 per spike");
    }

    #[test]
    fn leak_convergence_large_dt_lands_on_rest_exactly() {
        // The module-doc invariant: zero input + large dt (dt = tau) gives
        // delta_v = leak exactly (1000·leak/1000 — no truncation), so the
        // membrane lands ON rest in one step, from either side, mV grid.
        let mut n = quiet_neuron(23, VoltageResolution::Millivolt);
        n.membrane_potential = -90;
        let _ = n.integrate_and_fire(0, 20_000, 20_000);
        assert_eq!(n.membrane_potential, n.resting_potential, "from below");
        n.membrane_potential = -20;
        let _ = n.integrate_and_fire(0, 20_000, 40_000);
        assert_eq!(n.membrane_potential, n.resting_potential, "from above");
        let _ = n.integrate_and_fire(0, 20_000, 60_000);
        assert_eq!(n.membrane_potential, n.resting_potential, "stays at rest");
    }
}
