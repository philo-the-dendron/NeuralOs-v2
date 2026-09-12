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
//! Spike history is a ring over `[u32; MAX_SPIKE_HISTORY]` (compile-time
//! capacity, no allocator, no dependency): see [`SpikeRing`].
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

/// Maximum spike history length (compile-time, no allocator).
pub const MAX_SPIKE_HISTORY: usize = 64;
// `SpikeRing::push` indexes with `& (MAX_SPIKE_HISTORY - 1)`, the same slot
// as `% MAX_SPIKE_HISTORY` only for a power of two.
const _: () = assert!(MAX_SPIKE_HISTORY.is_power_of_two());

/// Bounded spike-timestamp history: a ring over
/// `[u32; MAX_SPIKE_HISTORY]` with the index of the oldest entry and a
/// length. Push is O(1) — once full, the oldest entry is overwritten
/// (recent spikes are what the consumers want). Iteration and indexing
/// run oldest to newest, the order the ISI walk depends on. Replaced
/// `heapless::Vec` (2026-09-10): its `remove(0)` shifted the whole
/// buffer on every spike past the 64th.
#[derive(Clone)]
struct SpikeRing {
    buf: [u32; MAX_SPIKE_HISTORY],
    /// Index of the oldest live entry. Meaningless while `len == 0`.
    head: usize,
    len: usize,
}

impl SpikeRing {
    const fn new() -> Self {
        Self {
            buf: [0; MAX_SPIKE_HISTORY],
            head: 0,
            len: 0,
        }
    }

    /// Append `t` as the newest entry; drop the oldest when full.
    ///
    /// Both stores index with `& (MAX_SPIKE_HISTORY - 1)`: the same slot as
    /// `%` and as the bare `head` (a power of two, asserted beside the
    /// constant; `head` stays below it), and a bound the optimizer reads from
    /// the index itself. The bare `self.buf[self.head]` was the one index
    /// whose bound did not follow from its own expression, and its bounds
    /// check changed how LLVM treated the whole neuron (ISA round 26,
    /// record-only (d); round 27, item 4).
    fn push(&mut self, t: u32) {
        debug_assert!(self.head < MAX_SPIKE_HISTORY);
        if self.len < MAX_SPIKE_HISTORY {
            self.buf[(self.head + self.len) & (MAX_SPIKE_HISTORY - 1)] = t;
            self.len += 1;
        } else {
            self.buf[self.head & (MAX_SPIKE_HISTORY - 1)] = t;
            self.head = (self.head + 1) % MAX_SPIKE_HISTORY;
        }
    }

    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Read side, test-gated like its only callers (the introspection
    /// trio below).
    #[cfg(test)]
    fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The `i`-th entry, oldest first. Panics when `i >= len()`, like a
    /// slice index would.
    fn get(&self, i: usize) -> u32 {
        assert!(
            i < self.len,
            "SpikeRing index {i} out of range {}",
            self.len
        );
        self.buf[(self.head + i) % MAX_SPIKE_HISTORY]
    }

    /// Entries oldest to newest.
    fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.len).map(move |i| self.get(i))
    }
}

/// Only the live entries, oldest first — never the stale slots.
impl core::fmt::Debug for SpikeRing {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Membrane potential lower bound (mV) — biological floor.
pub const MEMBRANE_MV_MIN: i16 = -100;

/// Membrane potential upper bound (mV) — biological ceiling.
pub const MEMBRANE_MV_MAX: i16 = 50;

/// The `dt/τ` scaling factor, scaled by 1000: `(dt_us * 1000) / tau_us`.
///
/// **Exact over the whole `u32 × u32` domain, and never saturated here.** The
/// batch kernel cannot accept every value this returns — its `i32`
/// intermediates overflow above `simd::DT_OVER_TAU_MAX` — so `simd` applies
/// that bound at its own boundary. The bound belongs to the kernel that needs
/// it; it is not a property of `dt/τ`, and a neuron must not inherit it. See
/// [`LIFNeuron::integrate_and_fire`].
///
/// This is the ONE definition of the formula. `simd::dt_over_tau` is this
/// function plus the batch's clamp, so the two cannot drift.
///
/// The wide formula computes the product and the division in `u64`.
/// Computing them with `as i32` casts wrapped for `dt_us > i32::MAX` and for
/// `tau_membrane_us > i32::MAX`: `dt_over_tau(2_147_484, u32::MAX)` returned
/// `-2_147_483_647`, a value that overflows every downstream multiply, and
/// `dt_over_tau(1000, u32::MAX)` returned `-1_000_000` — a single step that
/// drove the membrane to the floor. Both inputs are `u32`, so `u64` covers the
/// whole domain exactly, the result is always non-negative, and `i64` holds it
/// with room to spare (the maximum is `u32::MAX * 1000 = 4_294_967_295_000`).
///
/// `u64` only where it is needed: when `dt_us <= u32::MAX / 1000`
/// (4,294,967 µs; the physical dt is 1,000) the product fits `u32` and the
/// division is one hardware divide on a 32-bit core; above it,
/// `dt_over_tau_wide` computes the `u64` formula as it always did, out of
/// line. The guard is "fits `u32`", a property of the value like
/// `div_1000`'s "fits `i32`", never a bound on `dt/τ`. With dt a run-time
/// value, the `u64` division was a call to the ROM's `__udivdi3` on every
/// integrating step (ISA round 27, item 1).
///
/// # Examples
/// ```
/// # use neuralos_snn::lif_neuron::dt_over_tau;
/// assert_eq!(dt_over_tau(1_000, 20_000), 50);    // the physical default
/// assert_eq!(dt_over_tau(40_000, 20_000), 2_000); // dt/tau = 2, exact
/// assert_eq!(dt_over_tau(1_000, u32::MAX), 0);   // no wrap to a negative
/// ```
#[inline]
#[must_use]
pub fn dt_over_tau(dt_us: u32, tau_membrane_us: u32) -> i64 {
    if tau_membrane_us == 0 {
        // Division-by-zero guard, and nothing more. This comment claimed "the
        // network rejects tau == 0 at construction" until 2026-09-01; no such
        // rejection exists anywhere in the crate. `tau_membrane_us` is a `pub`
        // field on `LIFNeuron`, and `SpikingNeuralNetwork::new` validates only
        // `neuron_count` and `time_step_us`. So a zero time constant is
        // reachable, and it lands here rather than on a panic.
        //
        // Returning 0 means "this step does not move the membrane", which is
        // the safe reading: a zero time constant is not physical, and refusing
        // to move is better than dividing by zero or inventing an infinity.
        // Construction-time validation is deliberately deferred to the
        // resistance-seam change (PR #1 scope note), which is where the
        // signature moves anyway.
        return 0;
    }
    // One hardware divide when the product fits `u32`, which covers every
    // physical step; the guard is "fits", not a bound (doc above).
    if dt_us <= u32::MAX / 1000 {
        i64::from(dt_us * 1000 / tau_membrane_us)
    } else {
        dt_over_tau_wide(dt_us, tau_membrane_us)
    }
}

/// The wide half of [`dt_over_tau`], for `dt_us > u32::MAX / 1000`: the
/// `u64` formula as it always was. Out of line and cold, like
/// [`div_1000_wide`].
#[cold]
#[inline(never)]
fn dt_over_tau_wide(dt_us: u32, tau_membrane_us: u32) -> i64 {
    let raw = (u64::from(dt_us) * 1000) / u64::from(tau_membrane_us);
    // u32::MAX * 1000 < i64::MAX, so this is lossless for every input.
    raw as i64
}

/// `x / 1000`, exact: the same truncating division, done in `i32` when the
/// value fits there.
///
/// The two divisions by 1000 in [`LIFNeuron::integrate_and_fire`] (the
/// current term and `delta_v`) work in `i64`, and on a 32-bit core with no
/// 64-bit divide each was a call to a software division (the ROM's
/// `__divdi3` on the ESP32-C3, ISA round 23). In the physical regime both
/// values fit `i32`, where a division by the constant 1000 is a
/// multiply-high and shifts. So the fast path narrows, divides and widens;
/// a value that does not fit takes [`div_1000_wide`], the `i64` division as
/// it always was.
///
/// The guard is "fits `i32`", a property of the value, never a bound on the
/// model: nothing is clamped or refused here. It is not the batch kernel's
/// `simd::DT_OVER_TAU_MAX`, a limit the kernel's `i32` intermediates put on
/// its inputs; the neuron stays the exact reference semantic over its whole
/// domain. `i32::try_from`, never `as i32`: the cast wraps `i32::MAX + 1`
/// to `i32::MIN` (pinned by `div_1000_is_exact_on_both_sides_of_the_guard`).
#[inline]
fn div_1000(x: i64) -> i64 {
    if let Ok(y) = i32::try_from(x) {
        i64::from(y / 1000)
    } else {
        div_1000_wide(x)
    }
}

/// The wide half of [`div_1000`]. Out of line so that no inlined call site
/// carries the 64-bit divide (`#[inline(never)]`: LLVM may inline a
/// function this small even when it is marked cold); cold so the fast path
/// is laid out straight through.
#[cold]
#[inline(never)]
fn div_1000_wide(x: i64) -> i64 {
    x / 1000
}

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

    /// Spike timestamps (chronological order, oldest first). Bounded by
    /// `MAX_SPIKE_HISTORY`; the oldest is overwritten once full.
    spike_history: SpikeRing,
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
    fn new_with_type(id: u16, neuron_type: NeuronType) -> Self {
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
            NeuronType::Inhibitory => (-50, 10_000, 80),  // 10 ms tau, interneuron
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
            spike_history: SpikeRing::new(),
        }
    }

    /// Switch the voltage grid in place, rescaling the four stored potentials
    /// (values are preserved exactly: ×100 or ÷100 on whole-quanta values).
    /// Call before stepping; spikes/history/currents are grid-independent.
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
        // The whole chain is `i64` and EXACT: no input in the full
        // `u32 dt × u32 τ × i16 current × u16 resistance × either scale` domain
        // can wrap or panic, and no correct result is approximated. The batch
        // kernel's `DT_OVER_TAU_MAX` bound is NOT applied here — it is the
        // kernel's `i32` limit, not a property of the model, and the neuron is
        // the reference semantic the batch approximates.
        //                delta_v      = dt_over_tau · (leak + current) / 1000
        // Voltage quanta: leak is stored-native; the current term converts
        // μA → quanta (R·I/1000 mV, ×scale for the grid). At scale = 1 the
        // expression sequence is byte-identical to the historical arithmetic.
        // Both divisions by 1000 go through `div_1000`: the same truncation,
        // in `i32` when the value fits (the physical regime), in `i64` when
        // it does not.
        let s = i64::from(self.voltage_resolution.scale());
        let dt_over_tau = dt_over_tau(dt_us, self.tau_membrane_us);
        // Both sides widened before the subtraction: in `i16` this overflows for
        // off-grid states (resting = i16::MAX against membrane = i16::MIN) and
        // panicked in debug.
        let leak_term = i64::from(self.resting_potential) - i64::from(self.membrane_potential);
        let current_term = div_1000(i64::from(total_current) * i64::from(self.resistance_mohm) * s);
        // The ONE multiply in this chain that cannot fit `i64` for every legal
        // input. Worst case `|leak + current_term|` is
        // `32_768 * 65_535 * scale / 1000 + 65_535`: 214_810_623 on the
        // centi-mV grid, and 2_212_985 on the mV grid. Against
        // `dt_over_tau`'s maximum of 4_294_967_295_000 those give about 9.2e20
        // and 9.5e18 — BOTH past an `i64::MAX` of 9.22e18, so this is not a
        // centi-mV-only concern. Saturating is exact rather than approximate
        // here, and that is a proof, not a hope: saturating requires
        // `|delta_v| > i64::MAX / 1000 ≈ 9.2e15`, and every such delta is
        // already millions of times outside the voltage clamp below, which
        // preserves the sign. So the clamped result is identical to the one an
        // unbounded integer would give. Pinned over the whole domain against an
        // `i128` reference by `prop_integrate_and_fire_is_exact_over_the_whole_domain`.
        let delta_v = div_1000(dt_over_tau.saturating_mul(leak_term + current_term));

        let new_v = i64::from(self.membrane_potential)
            .saturating_add(delta_v)
            .clamp(
                i64::from(MEMBRANE_MV_MIN) * s,
                i64::from(MEMBRANE_MV_MAX) * s,
            );
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
        // The ring drops the oldest once full (we want recent spikes, not
        // the first ones).
        self.spike_history.push(current_time_us);
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

/// Grid-switch rescaling (was the builder's `voltage_resolution`
/// step; test-only since the builder deletion — the resolution a
/// neuron lives on is fixed at construction).
#[cfg(test)]
impl LIFNeuron {
    fn set_voltage_resolution(&mut self, resolution: VoltageResolution) {
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

    /// Test-only introspection quartet (was pub; zero non-test
    /// callers — the R8 census deferral, ruled in the 2026-08-22
    /// consolidation). `spikes()` was deleted outright: zero
    /// callers anywhere, and `spike_count()` + the quartet cover
    /// every read the tests need.
    fn is_refractory(&self) -> bool {
        self.refractory_time_us > 0
    }

    /// Firing rate within a recent window, in millihertz. Reference
    /// time = `last_update_time_us` (the v0.1 bug fix: window filter
    /// keyed on advancing time, not last-spike).
    fn firing_rate_mhz(&self, window_us: u32) -> u32 {
        if self.spike_history.is_empty() || window_us == 0 {
            return 0;
        }
        let window_start = self.last_update_time_us.saturating_sub(window_us);
        let count = self
            .spike_history
            .iter()
            .filter(|&t| t >= window_start)
            .count() as u32;
        // spikes/sec × 1000 = (count × 1_000_000_000) / window_us
        count
            .saturating_mul(1_000_000_000)
            .checked_div(window_us)
            .unwrap_or(0)
    }

    /// Inter-spike-interval statistics: `(mean_us, std_dev_us)`, or
    /// `None` if `< 2` spikes. Pure integer math; std-dev via
    /// sum-of-squares.
    fn isi_stats_us(&self) -> Option<(u32, u32)> {
        if self.spike_history.len() < 2 {
            return None;
        }
        let n = self.spike_history.len();
        // u128 accumulators (alpha.6): an interval is at most u32::MAX,
        // its square below 2^64, and up to MAX_SPIKE_HISTORY - 1 of them
        // are summed. In u64 the sum of squares overflowed once three
        // intervals exceeded 2^32 / sqrt(3), about 2.48e9 µs, which the
        // non-monotonic filter below makes reachable (found 2026-09-10
        // by the differential proptest).
        let mut intervals_sum: u128 = 0;
        let mut intervals_sqsum: u128 = 0;
        let mut count: u128 = 0;
        for i in 1..n {
            let prev = self.spike_history.get(i - 1);
            let curr = self.spike_history.get(i);
            if curr >= prev {
                let d = u128::from(curr - prev);
                intervals_sum += d;
                intervals_sqsum += d * d;
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        // mean <= u32::MAX, so mean_sq and sq_mean are below 2^64 and
        // the variance fits u64 exactly.
        let mean = intervals_sum / count;
        // Variance = E[x²] − E[x]²
        let mean_sq = mean * mean;
        let sq_mean = intervals_sqsum / count;
        let variance = u64::try_from(sq_mean.saturating_sub(mean_sq))
            .expect("variance of u32 intervals is below 2^64");
        let std = isqrt_u64(variance);
        Some((mean as u32, std as u32))
    }

    fn spike_count(&self) -> usize {
        self.spike_history.len()
    }
}

/// Integer square root (`no_std`-safe, no float). Only consumer is
/// the test-gated `isi_stats_us`.
#[cfg(test)]
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
    fn neuron_fields_directly_configurable() {
        let mut n = LIFNeuron::new_with_type(3, NeuronType::Inhibitory);
        n.threshold = -45;
        n.tau_membrane_us = 15_000;
        assert_eq!(n.id, 3);
        assert_eq!(n.neuron_type, NeuronType::Inhibitory);
        assert_eq!(n.threshold, -45);
        assert_eq!(n.tau_membrane_us, 15_000);
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
        assert_eq!(
            cmv.membrane_potential,
            -7_000 + 6,
            "centi grid: 12 μA ⇒ 6 cV"
        );
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
        let mut n = LIFNeuron::new(5);
        n.threshold = -65;
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
        let mut n = LIFNeuron::new(12);
        n.noise_amplitude_ua = 0;
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

    /// The `saturating_mul` in the delta chain, exercised deterministically —
    /// because the property test above essentially never reaches it.
    ///
    /// Saturation needs `dt/τ` near 4.3e12 AND a near-maximal
    /// `|leak + current_term|` in the SAME draw, and uniform draws over
    /// `u32 × u32 × i16 × u16` deliver that combination with vanishing
    /// probability: a reviewer sampled two million draws from that
    /// distribution and not one saturated. So the proptest proves exactness
    /// over the domain it actually reaches, and THESE rows prove the
    /// saturating branch is exact. Neither claim stands on the other.
    ///
    /// Each row saturates the multiply (verified: `|product|` is 9.50e18 on
    /// the mV grid and 9.23e20 on centi, against an `i64::MAX` of 9.22e18) and
    /// each lands exactly where an unbounded integer would put it — on the
    /// voltage bound, because a delta that large has nowhere else to go.
    #[test]
    fn the_saturating_multiply_lands_where_an_unbounded_integer_would() {
        // (name, membrane, resting, input, resistance, centi, expected)
        let rows: [(&str, i16, i16, i16, u16, bool, i16); 4] = [
            ("mV, positive", -32768, 32767, 32767, 65535, false, 50),
            ("mV, negative", 32767, -32768, -32768, 65535, false, -100),
            ("centi, positive", -32768, 32767, 32767, 65535, true, 5000),
            (
                "centi, negative",
                32767,
                -32768,
                -32768,
                65535,
                true,
                -10000,
            ),
        ];

        for (name, mp, rp, input, resistance, centi, expected) in rows {
            let resolution = if centi {
                VoltageResolution::CentiMillivolt
            } else {
                VoltageResolution::Millivolt
            };
            let mut n = LIFNeuron::new(0);
            n.voltage_resolution = resolution;
            n.membrane_potential = mp;
            n.resting_potential = rp;
            n.resistance_mohm = resistance;
            n.threshold = i16::MAX;
            // The shortest legal time constant, so dt/tau is maximal. Without
            // this the neuron keeps `LIFNeuron::new`'s 20_000 us default, the
            // product lands near 4.6e16, and NOTHING saturates — the rows
            // would pass while proving the opposite of what they claim.
            n.tau_membrane_us = 1;
            n.noise_amplitude_ua = 0;
            n.synaptic_current_ua = 0;
            n.adaptation_current_ua = 0;
            n.refractory_time_us = 0;
            let _ = n.integrate_and_fire(input, u32::MAX, 0);

            // The same step with no width limit at all, reading the neuron's
            // own tau rather than assuming it.
            let s = i128::from(resolution.scale());
            let dtot = (i128::from(u32::MAX) * 1000) / i128::from(n.tau_membrane_us);
            let leak = i128::from(rp) - i128::from(mp);
            let current_term = (i128::from(input) * i128::from(resistance) * s) / 1000;
            let product = dtot * (leak + current_term);
            assert!(
                product > i128::from(i64::MAX) || product < i128::from(i64::MIN),
                "{name}: this row must actually saturate i64, |product| = {}",
                product.abs()
            );
            let unbounded = (i128::from(mp) + product / 1000).clamp(
                i128::from(MEMBRANE_MV_MIN) * s,
                i128::from(MEMBRANE_MV_MAX) * s,
            );

            assert_eq!(i128::from(n.membrane_potential), unbounded, "{name}");
            assert_eq!(n.membrane_potential, expected, "{name}");
        }
    }

    /// `div_1000` on both sides of its guard, and on the rounding: every row
    /// against its value written out, and the row itself checked against
    /// `/`, so a typo in the table cannot pass.
    ///
    /// `i32::MAX + 1` is the row that catches a wrapping `as i32` in place of
    /// `i32::try_from`: the cast sends it to `i32::MIN`, and the helper would
    /// answer `-2_147_483` where `2_147_483` is right. `i32::MIN - 1` is the
    /// same trap from below.
    #[test]
    fn div_1000_is_exact_on_both_sides_of_the_guard() {
        let (min, max) = (i64::from(i32::MIN), i64::from(i32::MAX));
        let rows: [(i64, i64); 12] = [
            (min - 1, -2_147_483),
            (min, -2_147_483),
            (max, 2_147_483),
            (max + 1, 2_147_483),
            (i64::MIN, -9_223_372_036_854_775),
            (i64::MAX, 9_223_372_036_854_775),
            (999, 0),
            (-999, 0),
            (1000, 1),
            (-1000, -1),
            (1001, 1),
            (-1001, -1),
        ];
        for (x, expected) in rows {
            assert_eq!(x / 1000, expected, "the table: {x} / 1000");
            assert_eq!(div_1000(x), expected, "div_1000({x})");
        }
    }

    proptest! {
        /// `div_1000` equals the `i64` division it replaced, on both paths.
        /// Half the draws come from `i32`, widened, and run the fast path;
        /// half from all of `i64`, which almost always runs the wide one. A
        /// random `i64` fits `i32` with probability 2⁻³², so drawing from
        /// `i64` alone would never reach the fast path, the code the helper
        /// adds.
        #[test]
        fn prop_div_1000_equals_i64_division(
            x in prop_oneof![any::<i32>().prop_map(i64::from), any::<i64>()],
        ) {
            prop_assert_eq!(div_1000(x), x / 1000, "x = {}", x);
        }
    }

    proptest! {
        /// `integrate_and_fire` is EXACT over the whole input domain, against an
        /// independent `i128` reference that cannot overflow: every `u32` step,
        /// every `u32` time constant, every `i16` current, every `u16`
        /// resistance, both voltage grids, membrane and resting anywhere in
        /// `i16`.
        ///
        /// Three separate claims, and the third is the one worth having:
        ///
        /// - it never panics (debug builds are the falsifier — this whole class
        ///   is what the `i32` chain got wrong: `dt_us as i32 * 1000`,
        ///   `resting - membrane` in `i16`, and
        ///   `total_current * resistance_mohm * scale` at the centi-mV scale
        ///   all overflowed for reachable inputs);
        /// - the result stays inside the voltage grid;
        /// - the result EQUALS the unbounded-integer answer, computed here in
        ///   `i128`, which cannot overflow.
        ///
        /// What this test does NOT prove, and must not be cited for: that the
        /// `saturating_mul` in the delta chain is exact. Uniform draws over
        /// this domain essentially never reach it — saturation needs a `dt/τ`
        /// near 4.3e12 AND a near-maximal `|leak + current_term|` in the same
        /// draw, and two million sampled draws produced none. That branch is
        /// proved by `the_saturating_multiply_lands_where_an_unbounded_integer_would`,
        /// which reaches it deterministically on four rows.
        #[test]
        fn prop_integrate_and_fire_is_exact_over_the_whole_domain(
            mp in any::<i16>(),
            rp in any::<i16>(),
            input in any::<i16>(),
            resistance in any::<u16>(),
            tau_us in 1u32..=u32::MAX,
            dt_us in 0u32..=u32::MAX,
            centi in any::<bool>(),
        ) {
            let resolution = if centi {
                VoltageResolution::CentiMillivolt
            } else {
                VoltageResolution::Millivolt
            };
            let mut n = LIFNeuron::new(0);
            n.voltage_resolution = resolution;
            n.membrane_potential = mp;
            n.resting_potential = rp;
            n.tau_membrane_us = tau_us;
            n.resistance_mohm = resistance;
            n.threshold = i16::MAX; // unreachable: no spike, no reset
            n.noise_amplitude_ua = 0;
            n.synaptic_current_ua = 0;
            n.adaptation_current_ua = 0;
            n.refractory_time_us = 0;

            let _ = n.integrate_and_fire(input, dt_us, 0);

            // The same equation with no width limit anywhere. i128 holds it:
            // the largest product is about 9.2e20 against an i128::MAX of 1.7e38.
            let s = i128::from(resolution.scale());
            let dtot = (i128::from(dt_us) * 1000) / i128::from(tau_us);
            let leak = i128::from(rp) - i128::from(mp);
            let current_term = (i128::from(input) * i128::from(resistance) * s) / 1000;
            let delta = (dtot * (leak + current_term)) / 1000;
            let expected = (i128::from(mp) + delta)
                .clamp(i128::from(MEMBRANE_MV_MIN) * s, i128::from(MEMBRANE_MV_MAX) * s);

            prop_assert_eq!(
                i128::from(n.membrane_potential), expected,
                "membrane {} != exact {} (mp={} rp={} input={} resistance={} dt={} tau={} centi={})",
                n.membrane_potential, expected, mp, rp, input, resistance, dt_us, tau_us, centi
            );
            let bound = i128::from(MEMBRANE_MV_MAX) * s;
            prop_assert!(
                i128::from(n.membrane_potential).abs() <= bound.max(i128::from(MEMBRANE_MV_MIN).abs() * s),
                "membrane {} left the voltage grid", n.membrane_potential
            );
        }

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
            let mut n = LIFNeuron::new(id);
            n.tau_refractory_us = tau_ref;
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

    // ----- The dt/tau formula, and the neuron's exactness over the whole domain -----

    /// `integrate_and_fire` used to inline
    /// `((dt_us as i32) * 1000) / self.tau_membrane_us as i32`. Both casts
    /// wrapped. `tau = u32::MAX` became `-1`, so the quotient was `-dt * 1000`
    /// and a single 1 ms step drove the membrane from rest to the floor —
    /// silently, no panic in either profile.
    #[test]
    fn a_time_constant_longer_than_i32_no_longer_inverts_the_step() {
        assert_eq!(dt_over_tau(1000, u32::MAX), 0, "tau >> dt is a zero step");

        let mut n = quiet_neuron(24, VoltageResolution::Millivolt);
        n.tau_membrane_us = u32::MAX;
        n.threshold = i16::MAX; // unreachable: read the raw membrane
        let before = n.membrane_potential;
        let _ = n.integrate_and_fire(1000, 1000, 0);
        // Old inline: dt_over_tau = 1_000_000 / -1 = -1_000_000, so
        // delta_v = -1_000_000 * (0 + 100) / 1000 = -100_000 -> clamped to -100.
        assert_eq!(
            n.membrane_potential, before,
            "a 4295 s time constant must not move the membrane in a 1 ms step"
        );
    }

    /// The other half of the same cast bug: `dt_us as i32 * 1000` overflowed
    /// `i32` for `dt_us >= 2_147_484` — a debug panic, a release wrap. The
    /// formula computes in `u64` and returns `i64`, which holds the whole
    /// domain: `u32::MAX * 1000` is about 4.29e12.
    #[test]
    fn a_time_step_past_the_i32_product_no_longer_overflows() {
        assert_eq!(dt_over_tau(2_147_484, u32::MAX), 0);
        assert_eq!(dt_over_tau(i32::MAX as u32, u32::MAX), 499);
        assert_eq!(
            dt_over_tau(u32::MAX, 1),
            4_294_967_295_000,
            "exact, not clamped"
        );
        assert_eq!(
            dt_over_tau(1000, 20_000),
            50,
            "the physical default is untouched"
        );

        let mut n = quiet_neuron(25, VoltageResolution::Millivolt);
        n.membrane_potential = MEMBRANE_MV_MIN;
        n.threshold = i16::MAX;
        // Old inline: (2_147_484 as i32) * 1000 = 2_147_484_000 > i32::MAX.
        let _ = n.integrate_and_fire(0, 2_147_484, 0);
        // Exact: dt_over_tau = 107_374_200, leak = 30, so the delta is enormous
        // and the clamp — not a bound on dt/tau — is what bounds the result.
        assert_eq!(n.membrane_potential, MEMBRANE_MV_MAX);
    }

    /// `dt_over_tau` on both sides of its `u32` guard: 4,294,967 is the
    /// largest dt whose product fits `u32`, 4,294,968 the smallest that
    /// takes the wide path. Every row against its value written out, and the
    /// value against the `u64` formula the guard sits in front of, so a typo
    /// in the table cannot pass. The `τ = u32::MAX` pair is the sharpest: 0
    /// on one side of the guard, 1 on the other.
    #[test]
    fn dt_over_tau_is_exact_on_both_sides_of_the_guard() {
        let m = u32::MAX;
        // (dt_us, tau_us, expected)
        let rows: [(u32, u32, i64); 11] = [
            (4_294_967, 1, 4_294_967_000),
            (4_294_968, 1, 4_294_968_000),
            (4_294_967, 7, 613_566_714),
            (4_294_968, 7, 613_566_857),
            (4_294_967, m, 0),
            (4_294_968, m, 1),
            (m, 1, 4_294_967_295_000),
            (m, m, 1_000),
            (1_000, 20_000, 50),
            (0, 20_000, 0),
            (1_000, 0, 0),
        ];
        for (dt, tau, expected) in rows {
            let formula = if tau == 0 {
                0
            } else {
                (u64::from(dt) * 1000 / u64::from(tau)) as i64
            };
            assert_eq!(formula, expected, "the table: ({dt}, {tau})");
            assert_eq!(dt_over_tau(dt, tau), expected, "dt_over_tau({dt}, {tau})");
        }
    }

    proptest! {
        /// `dt_over_tau` equals the `u64` formula it computed before the
        /// guard. dt is drawn half from `0..=4_294_967`, the fast path, and
        /// half from all of `u32`, which almost always takes the wide one: a
        /// random `u32` is at most 4,294,967 in 0.1 % of draws. τ comes from
        /// all of `u32`; zero, which a draw almost never hits, is a row of
        /// the test above.
        #[test]
        fn prop_dt_over_tau_equals_the_u64_formula(
            dt in prop_oneof![0u32..=4_294_967, any::<u32>()],
            tau in any::<u32>(),
        ) {
            let formula = if tau == 0 {
                0
            } else {
                (u64::from(dt) * 1000 / u64::from(tau)) as i64
            };
            prop_assert_eq!(dt_over_tau(dt, tau), formula, "dt = {} tau = {}", dt, tau);
        }
    }

    /// The neuron is EXACT where the batch kernel clamps, and this is the row
    /// that made the shared-bound design wrong.
    ///
    /// `dt = 40_000 µs` into `τ = 20_000 µs` is a ratio of 2. Nothing
    /// overflows, every value is physical, and the discretisation
    /// `dt_over_tau = 2000` is the correct one. Borrowing the batch's
    /// `DT_OVER_TAU_MAX = 1884` moved this step from −40 mV to −44 mV: a wrong
    /// answer, silently, in a domain a caller can reach through
    /// `SpikingNeuralNetwork::new(n, 40_000, ..)`.
    #[test]
    fn the_neuron_is_exact_where_the_batch_clamps() {
        const BATCH_BOUND: i64 = 1884; // simd::DT_OVER_TAU_MAX, not imported: no_std

        let mut n = quiet_neuron(26, VoltageResolution::Millivolt);
        n.membrane_potential = -100;
        n.resting_potential = -70;
        n.tau_membrane_us = 20_000;
        n.threshold = i16::MAX; // unreachable: read the raw membrane

        let exact = dt_over_tau(40_000, 20_000);
        assert_eq!(exact, 2000, "dt/tau = 2, exactly");
        assert!(
            exact > BATCH_BOUND,
            "the witness must be above the batch's bound"
        );

        let _ = n.integrate_and_fire(0, 40_000, 0);

        // leak = -70 - (-100) = 30, no current.
        //   neuron: 2000 * 30 / 1000 = +60  ->  -40
        //   batch:  1884 * 30 / 1000 = +56  ->  -44
        assert_eq!(n.membrane_potential, -40, "the neuron takes the exact step");
        let clamped = -100 + i16::try_from(BATCH_BOUND * 30 / 1000).expect("fits i16");
        assert_eq!(clamped, -44, "what the batch's bound would have given");
        assert_ne!(
            n.membrane_potential, clamped,
            "the divergence above dt/tau = 1.884 is real and documented, not a bug"
        );
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

    // ----- 2026-09-10: the ring replaced heapless::Vec. The oracle is the
    // pre-ring implementation kept verbatim: heapless::Vec with remove(0)
    // when full, and the three consumers' formulas over it. Same spike
    // times into both; iteration order and every consumer must agree. -----

    /// The pre-ring history and its consumers, as they were.
    struct HeaplessOracle {
        history: heapless::Vec<u32, MAX_SPIKE_HISTORY>,
    }

    impl HeaplessOracle {
        fn new() -> Self {
            Self {
                history: heapless::Vec::new(),
            }
        }

        fn push(&mut self, t: u32) {
            if self.history.is_full() {
                self.history.remove(0);
            }
            let _ = self.history.push(t);
        }

        fn firing_rate_mhz(&self, last_update_time_us: u32, window_us: u32) -> u32 {
            if self.history.is_empty() || window_us == 0 {
                return 0;
            }
            let window_start = last_update_time_us.saturating_sub(window_us);
            let count = self.history.iter().filter(|&&t| t >= window_start).count() as u32;
            count
                .saturating_mul(1_000_000_000)
                .checked_div(window_us)
                .unwrap_or(0)
        }

        fn isi_stats_us(&self) -> Option<(u32, u32)> {
            if self.history.len() < 2 {
                return None;
            }
            let n = self.history.len();
            // Same u128 accumulators as the ring side (alpha.6): the
            // oracle is the pre-ring HISTORY kept verbatim; the
            // consumers' arithmetic moves on both sides together.
            let mut intervals_sum: u128 = 0;
            let mut intervals_sqsum: u128 = 0;
            let mut count: u128 = 0;
            for i in 1..n {
                let prev = self.history[i - 1];
                let curr = self.history[i];
                if curr >= prev {
                    let d = u128::from(curr - prev);
                    intervals_sum += d;
                    intervals_sqsum += d * d;
                    count += 1;
                }
            }
            if count == 0 {
                return None;
            }
            let mean = intervals_sum / count;
            let mean_sq = mean * mean;
            let sq_mean = intervals_sqsum / count;
            let variance = u64::try_from(sq_mean.saturating_sub(mean_sq))
                .expect("variance of u32 intervals is below 2^64");
            let std = isqrt_u64(variance);
            Some((mean as u32, std as u32))
        }
    }

    proptest! {
        /// Differential: the neuron's ring against the heapless oracle on
        /// the same spike times — contents in order, then every consumer.
        /// Up to 300 spikes so the ring wraps several times; unordered
        /// times so non-monotonic sequences exercise the ISI filter.
        /// Times are arbitrary u32: the 2^27 bound this test carried on
        /// 2026-09-10 worked around the u64 overflow of `isi_stats_us`
        /// it had found, and the u128 accumulators (alpha.6) removed it.
        #[test]
        fn prop_ring_matches_the_heapless_oracle(
            times in proptest::collection::vec(any::<u32>(), 0..300),
            last_update_time_us in any::<u32>(),
            window_us in any::<u32>(),
        ) {
            let mut n = LIFNeuron::new(1);
            let mut oracle = HeaplessOracle::new();
            for &t in &times {
                n.spike(t);
                oracle.push(t);
            }
            let ring: std::vec::Vec<u32> = n.spike_history.iter().collect();
            let vec: std::vec::Vec<u32> = oracle.history.iter().copied().collect();
            prop_assert_eq!(ring, vec, "iteration order and contents");

            n.last_update_time_us = last_update_time_us;
            prop_assert_eq!(n.spike_count(), oracle.history.len());
            prop_assert_eq!(
                n.firing_rate_mhz(window_us),
                oracle.firing_rate_mhz(last_update_time_us, window_us)
            );
            prop_assert_eq!(n.isi_stats_us(), oracle.isi_stats_us());
        }

        /// The ring alone: after any number of pushes it holds the newest
        /// `min(pushes, MAX_SPIKE_HISTORY)` entries, oldest first, by
        /// iteration and by index; clear empties it and it works again.
        #[test]
        fn prop_ring_keeps_the_newest_in_order(
            times in proptest::collection::vec(any::<u32>(), 0..300),
        ) {
            let mut ring = SpikeRing::new();
            for &t in &times {
                ring.push(t);
            }
            let live = times.len().min(MAX_SPIKE_HISTORY);
            prop_assert_eq!(ring.len(), live);
            prop_assert_eq!(ring.is_empty(), live == 0);
            let expected = &times[times.len() - live..];
            let got: std::vec::Vec<u32> = ring.iter().collect();
            prop_assert_eq!(got.as_slice(), expected);
            for (i, &t) in expected.iter().enumerate() {
                prop_assert_eq!(ring.get(i), t);
            }

            ring.clear();
            prop_assert!(ring.is_empty());
            prop_assert_eq!(ring.iter().count(), 0);
            ring.push(7);
            prop_assert_eq!(ring.len(), 1);
            prop_assert_eq!(ring.get(0), 7);
        }
    }

    /// Three intervals of `u32::MAX` through the non-monotonic filter: in
    /// u64 the sum of their squares overflowed (three above 2^32/sqrt(3)
    /// suffice; the earlier note said "near 2^31", which is below the
    /// threshold). Both sides agree on the exact answer.
    #[test]
    fn isi_stats_survive_three_maximal_intervals() {
        let times = [0, u32::MAX, 0, u32::MAX, 0, u32::MAX];
        let mut n = LIFNeuron::new(1);
        let mut oracle = HeaplessOracle::new();
        for &t in &times {
            n.spike(t);
            oracle.push(t);
        }
        assert_eq!(n.isi_stats_us(), Some((u32::MAX, 0)));
        assert_eq!(oracle.isi_stats_us(), Some((u32::MAX, 0)));
    }

    #[test]
    fn ring_debug_shows_live_entries_only() {
        let mut ring = SpikeRing::new();
        assert_eq!(std::format!("{ring:?}"), "[]");
        ring.push(1);
        ring.push(2);
        assert_eq!(std::format!("{ring:?}"), "[1, 2]");
        // Wrap: 70 pushes of 0..70 keep 6..70, and only those print.
        let mut ring = SpikeRing::new();
        for t in 0..70_u32 {
            ring.push(t);
        }
        let shown = std::format!("{ring:?}");
        assert!(shown.starts_with("[6, 7, 8,"), "{shown}");
        assert!(shown.ends_with(" 69]"), "{shown}");
        assert_eq!(ring.iter().count(), MAX_SPIKE_HISTORY);
    }

    #[test]
    #[should_panic(expected = "SpikeRing index 2 out of range 2")]
    fn ring_get_past_len_panics_like_a_slice() {
        let mut ring = SpikeRing::new();
        ring.push(10);
        ring.push(20);
        let _ = ring.get(2);
    }
}
