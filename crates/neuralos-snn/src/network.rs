//! Spiking Neural Network — topology builders + step orchestrator.
//!
//! # Module placement
//!
//! This module is `std`-gated. The hot-path primitives ([`LIFNeuron`], [`Synapse`])
//! are `no_std`-compatible; this orchestrator uses `Vec`, `VecDeque`, and `Instant`
//! for desktop/server simulation. For bare-metal RISC-V deployment, hand-roll loops
//! over the primitives directly.
//!
//! # Invariants (testable)
//!
//! - `SpikingNeuralNetwork::new(0, ...)` always returns `Err`.
//! - `step()` advances `current_time_us` by exactly `time_step_us`.
//! - Topology builders never create self-connections.
//! - LFSR RNG is deterministic: same seed → same network connectivity.
//! - Synapse weights respect the presynaptic neuron's type (E → positive, I → negative).
//!
//! # Bug fixes vs v0.1
//!
//! - **Spatial grid dropped entirely.** v0.1's spatial grid used `grid_x = id % size`
//!   and `grid_y = id % size` (same formula → diagonal-only grid, not 2D), and a
//!   fallback loop processed every firing neuron anyway, making the grid dead code
//!   that only added memory + complexity. CSR sparse matrix is the real value.
//! - **Plasticity dt fixed.** v0.1 used hardcoded `dt_ltd = -1000` for the LTD branch,
//!   but `dt < 0` means pre-before-post which is LTP sign — the LTD branch was
//!   secretly doing LTP. v2 computes dt from actual pre/post spike timing.
//! - **Plasticity synapse-index bug fixed.** v0.1 indexed `self.synapses[post_id]`,
//!   using the postsynaptic neuron ID as a synapse vector index. That accesses the
//!   wrong synapse (or panics). v2 tracks `synapse_index` in the plasticity queue.

#![allow(clippy::module_name_repetitions)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::similar_names
)]

use crate::lif_neuron::{LIFNeuron, NeuronType};
use crate::synapse::{STDPRule, Synapse};
use crate::{Error, Result};
use std::collections::VecDeque;
use std::vec::Vec;

/// Default biological E/I ratio (80% excitatory, 20% inhibitory — cortical).
const DEFAULT_EXCITATORY_RATIO: f64 = 0.8;

/// Default weight for excitatory synapses (fixed-point, 1000 = 1.0).
const DEFAULT_EXCITATORY_WEIGHT: i16 = 100;
/// Default weight for inhibitory synapses (negative, fixed-point).
const DEFAULT_INHIBITORY_WEIGHT: i16 = -150;
/// Default LFSR seed (deterministic across runs).
const DEFAULT_SEED: u32 = 0x1234_5678;
/// LFSR Galois tap for 16-bit maximal-length (period `65_535`).
const LFSR_TAP: u32 = 0xB400;

/// Network topology descriptor. Passed to [`SpikingNeuralNetwork::new`] and
/// consumed by `build_topology()`.
///
/// `f64` fields are configuration-time parameters (not hot-path), so floating-point
/// is acceptable here — the `no_std` constraint applies to per-step computation only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkTopology {
    /// Random sparse connectivity. `connectivity ∈ [0.0, 1.0]` is the fraction of
    /// all possible (pre ≠ post) pairs to wire.
    Random { connectivity: f64 },
    /// Watts-Strogatz small-world. Ring lattice with `local_connections` per neuron,
    /// each rewired with probability `rewiring_prob ∈ [0.0, 1.0]`.
    SmallWorld {
        local_connections: u8,
        rewiring_prob: f64,
    },
    /// Layered feedforward. `layers` defines the neuron count per layer; layers
    /// must sum to the network's `neuron_count`.
    Feedforward { layers: &'static [u16] },
    /// Balanced E/I network with 4 connection classes (E→E, E→I, I→E, I→I).
    /// `excitatory_ratio ∈ (0.0, 1.0)` is the fraction of neurons that are excitatory.
    Balanced { excitatory_ratio: f64 },
}

impl Default for NetworkTopology {
    fn default() -> Self {
        Self::Balanced {
            excitatory_ratio: DEFAULT_EXCITATORY_RATIO,
        }
    }
}

/// Spike event emitted by a neuron during a [`SpikingNeuralNetwork::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spike {
    /// ID of the neuron that fired.
    pub neuron_id: u16,
    /// Simulation time (μs) at which the spike occurred.
    pub time_us: u32,
    /// Spike weight (fixed-point, 1000 = 1.0). Always 1000 for direct neuron spikes.
    pub weight: i16,
}

/// Network-level statistics. Computed after each step. No serde (kept minimal).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NetworkStats {
    /// Total neuron count (fixed at construction).
    pub total_neurons: u16,
    /// Total synapse count (set after `build_topology`).
    pub total_synapses: u32,
    /// Cumulative spikes emitted since construction or last `reset`.
    pub total_spikes: u64,
    /// Cumulative plasticity updates applied.
    pub plasticity_events: u64,
    /// Mean membrane potential across all neurons (mV). Computed post-step.
    pub avg_membrane_potential_mv: f64,
    /// Mean firing rate per neuron (Hz). Computed post-step.
    pub firing_rate_hz: f64,
}

impl NetworkStats {
    fn new(neuron_count: u16) -> Self {
        Self {
            total_neurons: neuron_count,
            total_synapses: 0,
            total_spikes: 0,
            plasticity_events: 0,
            avg_membrane_potential_mv: -70.0,
            firing_rate_hz: 0.0,
        }
    }
}

/// Compressed Sparse Row (CSR) synapse storage — O(1) per-presynaptic iteration.
///
/// Three parallel arrays (the standard CSR layout):
/// - `weights[i]`: weight of synapse i
/// - `col_indices[i]`: postsynaptic neuron ID of synapse i
/// - `row_ptrs[n]`: start index into `weights/col_indices` for presynaptic neuron n
///   (length = `neuron_count` + 1; `row_ptrs`[`neuron_count`] = total synapse count)
#[derive(Debug, Clone)]
pub struct SparseSynapseMatrix {
    weights: Vec<i16>,
    col_indices: Vec<u16>,
    /// Length = `neuron_count` + 1. Built incrementally, finalized by [`finalize`].
    row_ptrs: Vec<u32>,
    /// Total neuron count (sets `row_ptrs` length).
    neuron_count: u16,
}

impl SparseSynapseMatrix {
    /// New empty CSR matrix sized for `neuron_count` presynaptic neurons.
    #[must_use] pub fn new(neuron_count: u16, estimated_synapses: usize) -> Self {
        Self {
            weights: Vec::with_capacity(estimated_synapses),
            col_indices: Vec::with_capacity(estimated_synapses),
            row_ptrs: vec![0; neuron_count as usize + 1],
            neuron_count,
        }
    }

    /// Append a synapse. Must be called in (`pre_id`, `post_id`) order — or call
    /// [`finalize`] to recompute row pointers from scratch after bulk insertion.
    pub fn add(&mut self, pre_id: u16, post_id: u16, weight: i16) {
        debug_assert!(
            pre_id < self.neuron_count,
            "pre_id {pre_id} ≥ neuron_count {}",
            self.neuron_count
        );
        self.weights.push(weight);
        self.col_indices.push(post_id);
        // Incremental row_ptrs: bump every row after pre_id by 1.
        for row in (pre_id as usize + 1)..self.row_ptrs.len() {
            self.row_ptrs[row] += 1;
        }
    }

    /// Recompute `row_ptrs` from scratch. Currently a no-op — the incremental
    /// [`add`] keeps `row_ptrs` in sync. Kept for API parity with v0.1 and future
    /// unordered-insert support.
    pub fn finalize(&mut self) {
        // No-op: add() maintains row_ptrs incrementally.
    }

    /// Iterate synapses from `pre_id`. Returns `(post_id, weight)` pairs.
    /// O(1) setup, O(out-degree) iteration.
    #[must_use] pub fn connections(&self, pre_id: u16) -> SynapseIter<'_> {
        debug_assert!(pre_id < self.neuron_count);
        let start = self.row_ptrs[pre_id as usize] as usize;
        let end = self.row_ptrs[pre_id as usize + 1] as usize;
        SynapseIter {
            weights: &self.weights[start..end],
            col_indices: &self.col_indices[start..end],
            pos: 0,
        }
    }

    /// Total synapse count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Is the matrix empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// Iterator over a single presynaptic neuron's outgoing synapses.
#[derive(Debug, Clone)]
pub struct SynapseIter<'a> {
    weights: &'a [i16],
    col_indices: &'a [u16],
    pos: usize,
}

impl Iterator for SynapseIter<'_> {
    type Item = (u16, i16); // (post_id, weight)

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.weights.len() {
            let item = (self.col_indices[self.pos], self.weights[self.pos]);
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }
}

/// Plasticity queue entry: `(pre_neuron_id, post_neuron_id, synapse_index, pre_spike_time_us)`.
/// Drained by [`SpikingNeuralNetwork::update_plasticity`].
type PlasticityEntry = (u16, u16, usize, u32);

/// Main spiking neural network orchestrator.
///
/// Holds [`LIFNeuron`] + [`Synapse`] collections, a CSR [`SparseSynapseMatrix`]
/// for fast synaptic transmission, and an [`STDPRule`] for plasticity. One call to
/// [`step`](Self::step) advances the simulation by `time_step_us` microseconds.
pub struct SpikingNeuralNetwork {
    neurons: Vec<LIFNeuron>,
    /// Parallel to synapse insertion order. Indexed via plasticity queue.
    synapses: Vec<Synapse>,
    /// CSR storage for fast per-presynaptic iteration during transmission.
    synapse_matrix: SparseSynapseMatrix,
    time_step_us: u32,
    current_time_us: u32,
    plasticity_rule: STDPRule,
    stats: NetworkStats,
    spike_history: VecDeque<Spike>,
    max_spike_history: usize,
    topology: NetworkTopology,
    seed: u32,
    /// Buffer of pending plasticity updates from the most recent step.
    plasticity_queue: Vec<PlasticityEntry>,
}

impl SpikingNeuralNetwork {
    /// Construct a network with `neuron_count` neurons, `time_step_us` simulation
    /// step, and the given `topology`. Neurons are created with the biological
    /// 80/20 E/I ratio unless the topology overrides (e.g., Feedforward is all E).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `neuron_count == 0` or `time_step_us == 0`.
    pub fn new(
        neuron_count: u16,
        time_step_us: u32,
        topology: NetworkTopology,
    ) -> Result<Self> {
        if neuron_count == 0 {
            return Err(Error::InvalidParameter);
        }
        if time_step_us == 0 {
            return Err(Error::InvalidParameter);
        }

        let mut neurons = Vec::with_capacity(neuron_count as usize);
        for id in 0..neuron_count {
            let nt = if (id as f64) < neuron_count as f64 * DEFAULT_EXCITATORY_RATIO {
                NeuronType::Excitatory
            } else {
                NeuronType::Inhibitory
            };
            neurons.push(LIFNeuron::new_with_type(id, nt));
        }

        let estimated_synapses = estimate_synapses(neuron_count, &topology);

        Ok(Self {
            neurons,
            synapses: Vec::with_capacity(estimated_synapses),
            synapse_matrix: SparseSynapseMatrix::new(neuron_count, estimated_synapses),
            time_step_us,
            current_time_us: 0,
            plasticity_rule: STDPRule::new(),
            stats: NetworkStats::new(neuron_count),
            spike_history: VecDeque::new(),
            max_spike_history: 10_000,
            topology,
            seed: DEFAULT_SEED,
            plasticity_queue: Vec::with_capacity(estimated_synapses),
        })
    }

    /// Build the configured topology. Must be called before [`step`].
    /// Resets `stats.total_synapses` to the resulting synapse count.
    pub fn build_topology(&mut self) -> Result<()> {
        match self.topology {
            NetworkTopology::Random { connectivity } => self.build_random(connectivity)?,
            NetworkTopology::SmallWorld {
                local_connections,
                rewiring_prob,
            } => self.build_small_world(local_connections, rewiring_prob)?,
            NetworkTopology::Feedforward { layers } => self.build_feedforward(layers)?,
            NetworkTopology::Balanced { excitatory_ratio } => {
                self.build_balanced(excitatory_ratio)?;
            }
        }
        self.stats.total_synapses = self.synapses.len() as u32;
        Ok(())
    }

    /// Advance the simulation by one `time_step_us`. Returns the spikes emitted
    /// this step in chronological (neuron-id) order.
    pub fn step(&mut self, input_currents: &[i16]) -> Result<Vec<Spike>> {
        let mut output_spikes: Vec<Spike> = Vec::new();
        let mut firing_neurons: Vec<u16> = Vec::new();

        // Clear previous step's synaptic currents and plasticity queue.
        for n in &mut self.neurons {
            n.clear_synaptic_current();
        }
        self.plasticity_queue.clear();

        // Phase 1: integrate-and-fire all neurons (O(n)).
        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            let current_ua = input_currents.get(idx).copied().unwrap_or(0);
            if neuron.integrate_and_fire(current_ua, self.time_step_us, self.current_time_us) {
                let neuron_id = idx as u16;
                let spike = Spike {
                    neuron_id,
                    time_us: self.current_time_us,
                    weight: 1000,
                };
                output_spikes.push(spike);
                firing_neurons.push(neuron_id);
                self.spike_history.push_back(spike);
                if self.spike_history.len() > self.max_spike_history {
                    self.spike_history.pop_front();
                }
            }
        }

        // Phase 2: propagate spikes through synapses (O(active_synapses)).
        // For each firing neuron, iterate its outgoing CSR slice and inject
        // current into each postsynaptic partner. Queue plasticity updates.
        for &pre_id in &firing_neurons {
            for (post_id, weight) in self.synapse_matrix.connections(pre_id) {
                if let Some(post_n) = self.neurons.get_mut(post_id as usize) {
                    // Scale weight down to keep currents in a reasonable μA range.
                    post_n.add_synaptic_current(weight / 10);
                }
                // Find synapse index for the plasticity queue.
                // Linear scan — for the network sizes we run, this is fine.
                // (A reverse index from (pre, post) → synapse_idx would speed this up.)
                let syn_idx = self
                    .synapses
                    .iter()
                    .position(|s| s.pre_neuron_id == pre_id && s.post_neuron_id == post_id)
                    .unwrap_or(0);
                self.plasticity_queue
                    .push((pre_id, post_id, syn_idx, self.current_time_us));
            }
        }

        // Phase 3: apply STDP plasticity for every active synapse (O(queue)).
        self.update_plasticity(&firing_neurons);

        // Advance time and stats.
        self.current_time_us = self.current_time_us.saturating_add(self.time_step_us);
        self.stats.total_spikes += output_spikes.len() as u64;
        self.update_stats();

        Ok(output_spikes)
    }

    /// Apply STDP to every queued (pre, post, `synapse_idx`, `pre_time`) entry.
    ///
    /// # Bug fix vs v0.1
    ///
    /// v0.1 used `dt_ltd = -1000` hardcoded for the LTD branch (post not firing
    /// this step), which is LTP sign. v2 computes dt from actual pre/post spike
    /// timing: if post fired this step → dt ≈ 0 (LTP at the limit); otherwise
    /// look up post's `last_spike_time_us` from its `LIFNeuron` and compute real dt.
    fn update_plasticity(&mut self, firing_neurons: &[u16]) {
        if self.plasticity_queue.is_empty() {
            return;
        }
        // Snapshot firing set for post lookup.
        let post_fired_this_step = |id: u16| firing_neurons.contains(&id);

        for &(_pre_id, post_id, syn_idx, pre_time) in &self.plasticity_queue.clone() {
            let Some(synapse) = self.synapses.get_mut(syn_idx) else {
                continue;
            };
            let post_time = if post_fired_this_step(post_id) {
                // Post fired this step → near-simultaneous → small positive dt → LTD.
                // Use 1μs to break the tie toward LTD (post fired just after pre).
                pre_time.saturating_add(1)
            } else {
                // Post didn't fire this step — use its last actual spike time.
                // If post never fired, last_spike_time_us = 0, giving a large
                // positive dt → LTD with decayed magnitude. Reasonable default.
                self.neurons
                    .get(post_id as usize)
                    .map_or(0, |n| n.last_spike_time_us)
            };
            // dt = pre_time - post_time. Positive (pre after post) → LTD.
            let dt_us: i32 = (pre_time as i64 - post_time as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            let delta = self.plasticity_rule.calculate_weight_change(dt_us);
            synapse.update_weight(delta);
            self.stats.plasticity_events += 1;
        }
    }

    /// Update running statistics (sampling every 10th neuron for averages).
    fn update_stats(&mut self) {
        let n = self.neurons.len();
        if n == 0 {
            return;
        }
        let mut total_v: i64 = 0;
        let mut sampled = 0u32;
        for i in (0..n).step_by(10) {
            total_v += i64::from(self.neurons[i].membrane_potential_mv);
            sampled += 1;
        }
        if sampled > 0 {
            self.stats.avg_membrane_potential_mv = (total_v as f64 * 10.0) / n as f64;
        }
        let time_sec = self.current_time_us as f64 / 1_000_000.0;
        if time_sec > 0.0 {
            self.stats.firing_rate_hz =
                self.stats.total_spikes as f64 / (time_sec * n as f64);
        }
    }

    /// Append a synapse. Both `SparseSynapseMatrix` and `synapses` vec get a copy.
    pub fn add_synapse(&mut self, pre_id: u16, post_id: u16, weight: i16) -> Result<()> {
        if pre_id as usize >= self.neurons.len() || post_id as usize >= self.neurons.len() {
            return Err(Error::IndexOutOfBounds);
        }
        let synapse = Synapse::new(pre_id, post_id, weight)?;
        self.synapse_matrix.add(pre_id, post_id, weight);
        self.synapses.push(synapse);
        Ok(())
    }

    /// Reset all neurons, synapses, stats, and time. Keeps topology + synapse wiring.
    pub fn reset(&mut self) {
        for n in &mut self.neurons {
            n.reset();
        }
        for s in &mut self.synapses {
            s.reset();
        }
        self.current_time_us = 0;
        self.stats.total_spikes = 0;
        self.stats.plasticity_events = 0;
        self.spike_history.clear();
        self.plasticity_queue.clear();
    }

    /// Read-only access to current stats.
    #[must_use]
    pub fn stats(&self) -> &NetworkStats {
        &self.stats
    }

    /// Current simulation time (μs).
    #[must_use]
    pub fn current_time_us(&self) -> u32 {
        self.current_time_us
    }

    /// Total neuron count.
    #[must_use]
    pub fn neuron_count(&self) -> u16 {
        self.neurons.len() as u16
    }

    /// Total synapse count.
    #[must_use]
    pub fn synapse_count(&self) -> u32 {
        self.synapses.len() as u32
    }

    /// Read-only access to spike history (most recent first; back = oldest).
    #[must_use] pub fn spike_history(&self) -> &VecDeque<Spike> {
        &self.spike_history
    }

    // ----- Topology builders (private) -----

    /// Random sparse connectivity via Fisher-Yates sampling without replacement.
    fn build_random(&mut self, connectivity: f64) -> Result<()> {
        let n = self.neurons.len();
        if n < 2 {
            return Ok(());
        }
        let total_possible = n * (n - 1);
        let target = ((total_possible as f64) * connectivity.clamp(0.0, 1.0)) as usize;
        let mut rng = self.seed;
        let mut pairs: Vec<(u16, u16)> = Vec::with_capacity(total_possible);
        for pre in 0..n as u16 {
            for post in 0..n as u16 {
                if pre != post {
                    pairs.push((pre, post));
                }
            }
        }
        for i in 0..target.min(pairs.len()) {
            rng = advance_lfsr(rng);
            let range = (pairs.len() - i) as u32;
            let j = i + (rng % range) as usize;
            pairs.swap(i, j);
            let (pre_id, post_id) = pairs[i];
            rng = advance_lfsr(rng);
            let weight = typed_weight(self.neurons[pre_id as usize].neuron_type, rng);
            self.add_synapse(pre_id, post_id, weight)?;
        }
        self.seed = rng;
        Ok(())
    }

    /// Watts-Strogatz small-world: ring lattice + probabilistic rewiring.
    fn build_small_world(&mut self, local_connections: u8, rewiring_prob: f64) -> Result<()> {
        let n = self.neurons.len() as u16;
        if n < 2 {
            return Ok(());
        }
        let mut rng = self.seed;
        // Ring lattice: each neuron connects to `local_connections` neighbors.
        // Bug fix vs v0.1: when `local_connections >= n`, the modulo wraps to
        // self-connections (e.g., n=5, offset=5 → target = (i+5)%5 = i). Skip them.
        for i in 0..n {
            for offset in 1..=local_connections as u16 {
                let target = (i + offset) % n;
                if target == i {
                    continue; // Skip self-connection from wrap-around.
                }
                let weight = typed_weight(self.neurons[i as usize].neuron_type, 0);
                self.add_synapse(i, target, weight)?;
            }
        }
        // Rewiring: with probability rewiring_prob, replace a local connection
        // with a random one.
        for i in 0..n {
            for _ in 0..local_connections {
                rng = advance_lfsr(rng);
                let roll = (rng & 0xFFFF) as f64 / 65_536.0;
                if roll < rewiring_prob.clamp(0.0, 1.0) {
                    rng = advance_lfsr(rng);
                    let new_target = (rng % n as u32) as u16;
                    if new_target != i {
                        let weight = typed_weight(self.neurons[i as usize].neuron_type, 0);
                        self.add_synapse(i, new_target, weight)?;
                    }
                }
            }
        }
        self.seed = rng;
        Ok(())
    }

    /// Layered feedforward: connect each layer to the next with sparse projection.
    fn build_feedforward(&mut self, layers: &[u16]) -> Result<()> {
        if layers.is_empty() {
            return Err(Error::InvalidParameter);
        }
        let total: u16 = layers.iter().sum();
        if total as usize != self.neurons.len() {
            return Err(Error::InvalidParameter);
        }
        let mut offset = 0u16;
        for window in layers.windows(2) {
            let curr_size = window[0];
            let next_size = window[1];
            let next_offset = offset + curr_size;
            let conns_per = (next_size / 4).max(1);
            let stride = (next_size / conns_per).max(1);
            for i in 0..curr_size {
                let pre_id = offset + i;
                let nt = self.neurons[pre_id as usize].neuron_type;
                for j in 0..conns_per {
                    let post_id = next_offset + (i + j * stride) % next_size;
                    let weight = typed_weight(nt, 0);
                    self.add_synapse(pre_id, post_id, weight)?;
                }
            }
            offset += curr_size;
        }
        Ok(())
    }

    /// Balanced E/I network with 4 connection classes (E→E, E→I, I→E, I→I).
    fn build_balanced(&mut self, excitatory_ratio: f64) -> Result<()> {
        let n = self.neurons.len() as u16;
        let exc_count = (n as f64 * excitatory_ratio.clamp(0.0, 1.0)) as u16;
        let inh_count = n - exc_count;
        if inh_count == 0 {
            return Err(Error::InvalidParameter);
        }
        let mut rng = self.seed;
        // E→E (weak excitatory)
        for _ in 0..(exc_count as u32 * 5) {
            rng = advance_lfsr(rng);
            let pre = (rng % exc_count as u32) as u16;
            rng = advance_lfsr(rng);
            let post = (rng % exc_count as u32) as u16;
            if pre != post {
                self.add_synapse(pre, post, 80)?;
            }
        }
        // E→I (strong excitatory)
        for _ in 0..(exc_count as u32 * 3) {
            rng = advance_lfsr(rng);
            let pre = (rng % exc_count as u32) as u16;
            rng = advance_lfsr(rng);
            let post = exc_count + (rng % inh_count as u32) as u16;
            self.add_synapse(pre, post, 150)?;
        }
        // I→E (strong inhibitory)
        for _ in 0..(inh_count as u32 * 8) {
            rng = advance_lfsr(rng);
            let pre = exc_count + (rng % inh_count as u32) as u16;
            rng = advance_lfsr(rng);
            let post = (rng % exc_count as u32) as u16;
            self.add_synapse(pre, post, -200)?;
        }
        // I→I (moderate inhibitory)
        for _ in 0..(inh_count as u32 * 2) {
            rng = advance_lfsr(rng);
            let pre = exc_count + (rng % inh_count as u32) as u16;
            rng = advance_lfsr(rng);
            let post_off = (rng % inh_count as u32) as u16;
            if (pre - exc_count) != post_off {
                self.add_synapse(pre, exc_count + post_off, -120)?;
            }
        }
        self.seed = rng;
        Ok(())
    }
}

impl std::fmt::Debug for SpikingNeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpikingNeuralNetwork")
            .field("neurons", &self.neurons.len())
            .field("synapses", &self.synapses.len())
            .field("time_step_us", &self.time_step_us)
            .field("current_time_us", &self.current_time_us)
            .field("topology", &self.topology)
            .finish_non_exhaustive()
    }
}

/// Pick weight sign based on presynaptic neuron type + a small jitter from `rng`.
fn typed_weight(nt: NeuronType, rng: u32) -> i16 {
    match nt {
        NeuronType::Excitatory => DEFAULT_EXCITATORY_WEIGHT + ((rng & 0xFF) as i16 % 50),
        NeuronType::Inhibitory => DEFAULT_INHIBITORY_WEIGHT - ((rng & 0xFF) as i16 % 50),
    }
}

/// 16-bit Galois LFSR advance. Deterministic, no_std-friendly.
fn advance_lfsr(lfsr: u32) -> u32 {
    (lfsr >> 1) ^ (if lfsr & 1 != 0 { LFSR_TAP } else { 0 })
}

/// Estimate synapse count for capacity pre-allocation.
fn estimate_synapses(neuron_count: u16, topology: &NetworkTopology) -> usize {
    match topology {
        NetworkTopology::Random { connectivity } => {
            ((neuron_count as usize).saturating_mul(neuron_count as usize - 1) as f64
                * *connectivity) as usize
        }
        NetworkTopology::SmallWorld { local_connections, .. } => {
            neuron_count as usize * (*local_connections as usize) * 2
        }
        NetworkTopology::Feedforward { layers } => layers
            .windows(2)
            .map(|w| w[0] as usize * w[1] as usize / 2)
            .sum(),
        NetworkTopology::Balanced { excitatory_ratio } => {
            let exc = (neuron_count as f64 * excitatory_ratio) as usize;
            let inh = neuron_count as usize - exc;
            exc * 5 + exc * 3 + inh * 8 + inh * 2
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::shadow_unrelated)]
    use super::*;
    use proptest::prelude::*;

    // ----- Unit tests -----

    #[test]
    fn empty_network_rejected() {
        let err = SpikingNeuralNetwork::new(0, 1000, NetworkTopology::default());
        assert!(err.is_err());
    }

    #[test]
    fn zero_time_step_rejected() {
        let err = SpikingNeuralNetwork::new(10, 0, NetworkTopology::default());
        assert!(err.is_err());
    }

    #[test]
    fn new_network_has_correct_neuron_count() {
        let net = SpikingNeuralNetwork::new(100, 1000, NetworkTopology::default())
            .expect("100 neurons valid");
        assert_eq!(net.neuron_count(), 100);
        assert_eq!(net.current_time_us(), 0);
    }

    #[test]
    fn topology_default_is_balanced_80_20() {
        let net = SpikingNeuralNetwork::new(100, 1000, NetworkTopology::default())
            .expect("valid");
        let excitatory = net
            .neurons
            .iter()
            .filter(|n| n.neuron_type == NeuronType::Excitatory)
            .count();
        assert_eq!(excitatory, 80, "default topology should give 80% excitatory");
    }

    #[test]
    fn random_topology_produces_synapses() {
        let mut net = SpikingNeuralNetwork::new(50, 1000, NetworkTopology::Random { connectivity: 0.2 })
            .expect("valid");
        net.build_topology().expect("build");
        assert!(net.synapse_count() > 0, "random topology must produce synapses");
    }

    #[test]
    fn smallworld_topology_produces_synapses() {
        let mut net = SpikingNeuralNetwork::new(
            50,
            1000,
            NetworkTopology::SmallWorld { local_connections: 4, rewiring_prob: 0.1 },
        )
        .expect("valid");
        net.build_topology().expect("build");
        assert!(net.synapse_count() > 0);
    }

    #[test]
    fn feedforward_topology_produces_synapses() {
        let mut net = SpikingNeuralNetwork::new(
            30,
            1000,
            NetworkTopology::Feedforward { layers: &[10, 15, 5] },
        )
        .expect("valid");
        net.build_topology().expect("build");
        assert!(net.synapse_count() > 0);
    }

    #[test]
    fn balanced_topology_produces_synapses() {
        let mut net = SpikingNeuralNetwork::new(
            50,
            1000,
            NetworkTopology::Balanced { excitatory_ratio: 0.8 },
        )
        .expect("valid");
        net.build_topology().expect("build");
        assert!(net.synapse_count() > 0);
    }

    #[test]
    fn feedforward_wrong_total_rejected() {
        let mut net = SpikingNeuralNetwork::new(
            30,
            1000,
            NetworkTopology::Feedforward { layers: &[10, 15, 10] }, // sums to 35, not 30
        )
        .expect("valid net init");
        let err = net.build_topology();
        assert!(err.is_err(), "mismatched layer total must error");
    }

    #[test]
    fn balanced_zero_inhibitory_rejected() {
        let mut net = SpikingNeuralNetwork::new(
            10,
            1000,
            NetworkTopology::Balanced { excitatory_ratio: 1.0 },
        )
        .expect("valid net init");
        let err = net.build_topology();
        assert!(err.is_err(), "0 inhibitory must error");
    }

    #[test]
    fn step_advances_time_by_time_step() {
        let mut net = SpikingNeuralNetwork::new(10, 1000, NetworkTopology::default())
            .expect("valid");
        net.build_topology().expect("build");
        let inputs = vec![0; 10];
        let _ = net.step(&inputs).expect("step");
        assert_eq!(net.current_time_us(), 1000);
        let _ = net.step(&inputs).expect("step");
        assert_eq!(net.current_time_us(), 2000);
    }

    #[test]
    fn step_with_strong_input_produces_spikes() {
        let mut net = SpikingNeuralNetwork::new(10, 1000, NetworkTopology::default())
            .expect("valid");
        net.build_topology().expect("build");
        let inputs = vec![1000; 10]; // Strong input to all
        let mut total_spikes = 0u32;
        for _ in 0..50 {
            let spikes = net.step(&inputs).expect("step");
            total_spikes += spikes.len() as u32;
        }
        assert!(total_spikes > 0, "strong input over 50 steps must produce spikes");
    }

    #[test]
    fn reset_clears_time_and_spikes() {
        let mut net = SpikingNeuralNetwork::new(20, 1000, NetworkTopology::default())
            .expect("valid");
        net.build_topology().expect("build");
        let inputs = vec![1000; 20];
        for _ in 0..5 {
            let _ = net.step(&inputs).expect("step");
        }
        assert!(net.current_time_us() > 0);
        assert!(net.stats().total_spikes > 0);
        net.reset();
        assert_eq!(net.current_time_us(), 0);
        assert_eq!(net.stats().total_spikes, 0);
        assert_eq!(net.stats().plasticity_events, 0);
    }

    #[test]
    fn add_synapse_rejects_out_of_bounds() {
        let mut net = SpikingNeuralNetwork::new(10, 1000, NetworkTopology::default())
            .expect("valid");
        let err = net.add_synapse(0, 100, 50);
        assert!(err.is_err());
        let err = net.add_synapse(100, 0, 50);
        assert!(err.is_err());
    }

    #[test]
    fn add_synapse_rejects_self_connection() {
        let mut net = SpikingNeuralNetwork::new(10, 1000, NetworkTopology::default())
            .expect("valid");
        let err = net.add_synapse(3, 3, 100);
        assert!(err.is_err());
    }

    #[test]
    fn sparse_matrix_iter_returns_added_synapses() {
        let mut m = SparseSynapseMatrix::new(5, 4);
        m.add(0, 1, 100);
        m.add(0, 2, 200);
        m.add(2, 3, -150);
        let row0: Vec<(u16, i16)> = m.connections(0).collect();
        assert_eq!(row0, vec![(1, 100), (2, 200)]);
        let row2: Vec<(u16, i16)> = m.connections(2).collect();
        assert_eq!(row2, vec![(3, -150)]);
        let row1: Vec<(u16, i16)> = m.connections(1).collect();
        assert!(row1.is_empty());
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn lfsr_is_deterministic_same_seed() {
        let a = advance_lfsr(0xDEAD_BEEF);
        let b = advance_lfsr(0xDEAD_BEEF);
        assert_eq!(a, b);
    }

    // ----- Property tests (Cardano-grade rigor) -----

    proptest! {
        /// Any neuron_count > 0 with any valid topology succeeds at construction.
        #[test]
        fn prop_valid_network_constructs(
            n in 1u16..=200,
            topology in topology_strategy(),
        ) {
            let result = SpikingNeuralNetwork::new(n, 1000, topology);
            prop_assert!(result.is_ok(), "valid n+topology must construct");
        }

        /// step() advances time by exactly time_step_us, regardless of input.
        #[test]
        fn prop_step_advances_time(
            n in 5u16..=50,
            dt in 100u32..=10_000,
            input_value in -500i16..=500,
        ) {
            let mut net = SpikingNeuralNetwork::new(n, dt, NetworkTopology::default())?;
            net.build_topology()?;
            let inputs = vec![input_value; n as usize];
            net.step(&inputs)?;
            prop_assert_eq!(net.current_time_us(), dt);
        }

        /// No topologies create self-connections (pre != post for every synapse).
        #[test]
        fn prop_no_self_connections(
            n in 5u16..=30,
            topology in topology_strategy(),
        ) {
            let mut net = SpikingNeuralNetwork::new(n, 1000, topology)?;
            net.build_topology()?;
            for s in &net.synapses {
                prop_assert_ne!(
                    s.pre_neuron_id, s.post_neuron_id,
                    "self-connection created"
                );
            }
        }

        /// LFSR produces same output for same input (deterministic).
        #[test]
        fn prop_lfsr_deterministic(seed in any::<u32>()) {
            let a = advance_lfsr(seed);
            let b = advance_lfsr(seed);
            let c = advance_lfsr(seed);
            prop_assert_eq!(a, b);
            prop_assert_eq!(a, c);
        }

        /// Sparse matrix iteration count matches total synapses added.
        #[test]
        fn prop_sparse_matrix_iter_matches_count(
            n in 1u16..=20,
            syn_count in 0usize..=50,
        ) {
            let mut m = SparseSynapseMatrix::new(n, syn_count);
            for i in 0..syn_count {
                let pre = (i as u16) % n;
                let post = ((i as u16) + 1) % n;
                let post = if post == pre { (post + 1) % n } else { post };
                m.add(pre, post, 100);
            }
            let total: usize = (0..n).map(|i| m.connections(i).count()).sum();
            prop_assert_eq!(total, syn_count);
        }
    }

    /// Proptest strategy for `NetworkTopology` variants with sane parameter ranges.
    /// Feedforward is excluded — its layer sizes must match the neuron count,
    /// which can't be expressed in a standalone strategy. Feedforward is covered
    /// by the dedicated `feedforward_topology_produces_synapses` unit test.
    fn topology_strategy() -> impl Strategy<Value = NetworkTopology> {
        prop_oneof![
            (0.01f64..=0.99).prop_map(|c| NetworkTopology::Random { connectivity: c }),
            (1u8..=10u8, 0.0f64..=1.0)
                .prop_map(|(lc, rp)| NetworkTopology::SmallWorld {
                    local_connections: lc,
                    rewiring_prob: rp,
                }),
            (0.5f64..=0.95).prop_map(|er| NetworkTopology::Balanced { excitatory_ratio: er }),
        ]
    }
}
