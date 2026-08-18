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
//! - **Full pairwise STDP (Stage 1.5d).** Through Stage 1.5c `update_plasticity`
//!   visited only pre-firing events, so `dt ≥ 0` always and the LTP branch
//!   (`dt < 0`) was unreachable — the rule was structurally LTD-only. v2 now runs
//!   a second, disjoint post-firing LTP pass via a reverse CSR.
//! - **CSR `set_weight` inverse-permutation fix (Stage 1.5d).** The counting-sort
//!   `finalize()` reorders `weights[]` by `pre_id`, so the prior `set_weight(syn_idx)`
//!   (which indexed `weights[syn_idx]` directly) wrote deltas to the wrong slots,
//!   desynchronizing transmission weights from `synapses[].weight`. v2 routes the
//!   write through an inverse permutation built in `finalize()`.

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

use crate::lif_neuron::{LIFNeuron, NeuronType, VoltageResolution};
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
/// LFSR seed for stochastic ternary flips (Stage 1.5b). Independent of the
/// topology seed so plasticity randomness doesn't correlate with wiring.
const TERNARY_FLIP_SEED: u32 = 0xA5A5_5A5A;
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
/// Forward direction: row key = presynaptic neuron. [`connections`]`(pre_id)`
/// iterates that neuron's outgoing edges.
/// Reverse direction: row key = postsynaptic neuron. [`incoming`]`(post_id)`
/// iterates that neuron's incoming edges (used by the post-firing LTP half of
/// pairwise STDP).
///
/// Forward parallel arrays (sorted by `pre_id` after [`finalize`]):
/// - `weights[k]`: weight of the edge at sorted position `k`
/// - `col_indices[k]`: postsynaptic neuron ID of that edge
/// - `synapse_indices[k]`: index of that edge in `SpikingNeuralNetwork::synapses`
/// - `row_ptrs[n]`: start index into the arrays for presynaptic neuron `n`
///   (length `neuron_count` + 1; last entry = total edge count)
///
/// Reverse parallel arrays (sorted by `post_id` after [`finalize`]):
/// - `rv_pre_ids[k]`: presynaptic neuron ID of the edge at reverse-sorted `k`
/// - `rv_syn_indices[k]`: that edge's index in `SpikingNeuralNetwork::synapses`
/// - `rv_row_ptrs[n]`: start index for postsynaptic neuron `n`
///
/// `weight_index_of[synapse_index]` is the **inverse permutation** of
/// `synapse_indices`: given a synapse index, it returns the forward-sorted
/// position holding that synapse's weight. [`set_weight`] routes through it so a
/// plasticity update hits the correct CSR slot even after [`finalize`] reorders
/// the arrays (without it, `set_weight(syn_idx)` would write `weights[syn_idx]`,
/// which after the counting sort holds a *different* edge's weight).
///
/// **CSR correctness requires [`finalize`].** [`add`] appends edges in insertion
/// order and bumps `row_ptrs` incrementally — which only yields correct
/// [`connections`] slices when edges happen to be added in non-decreasing `pre_id`
/// order. `build_topology` inserts in arbitrary order, so it MUST call
/// [`finalize`] after all `add`s. [`finalize`] does a counting-sort CSR build
/// (O(edges + neurons), stable) that reorders the forward arrays by `pre_id`,
/// builds the reverse arrays by `post_id`, and inverts the synapse-index
/// permutation into `weight_index_of`. Calling [`connections`] before
/// [`finalize`] on unsorted input returns slices with the right *count* but the
/// wrong *members* — a silent correctness bug (was the prior state: propagation
/// injected current into the wrong targets and STDP updated the wrong synapses).
///
/// [`finalize`]: SparseSynapseMatrix::finalize
/// [`connections`]: SparseSynapseMatrix::connections
/// [`incoming`]: SparseSynapseMatrix::incoming
/// [`add`]: SparseSynapseMatrix::add
/// [`set_weight`]: SparseSynapseMatrix::set_weight
#[derive(Debug, Clone)]
pub struct SparseSynapseMatrix {
    weights: Vec<i16>,
    col_indices: Vec<u16>,
    synapse_indices: Vec<usize>,
    /// Presynaptic neuron ID of each edge (insertion order; input to
    /// [`finalize`]'s counting sort).
    ///
    /// [`finalize`]: SparseSynapseMatrix::finalize
    pre_ids: Vec<u16>,
    /// Length = `neuron_count` + 1. Authoritative only after [`finalize`];
    /// between `add`s it is a best-effort incremental estimate (correct only
    /// for sorted insertion).
    ///
    /// [`finalize`]: SparseSynapseMatrix::finalize
    row_ptrs: Vec<u32>,
    /// Inverse permutation: `weight_index_of[synapse_index]` = forward-sorted
    /// position of that synapse's weight slot. Identity before [`finalize`]
    /// (insertion order); rebuilt as the inverse of `synapse_indices` by
    /// [`finalize`]. Lets [`set_weight`] stay O(1) and correct post-sort.
    ///
    /// [`finalize`]: SparseSynapseMatrix::finalize
    /// [`set_weight`]: SparseSynapseMatrix::set_weight
    weight_index_of: Vec<usize>,
    /// Reverse CSR (by postsynaptic neuron). Built alongside the forward sort
    /// in [`finalize`]. Empty until then; [`incoming`] returns nothing.
    ///
    /// [`finalize`]: SparseSynapseMatrix::finalize
    /// [`incoming`]: SparseSynapseMatrix::incoming
    rv_row_ptrs: Vec<u32>,
    rv_pre_ids: Vec<u16>,
    rv_syn_indices: Vec<usize>,
    /// Total neuron count (sets `row_ptrs` length).
    neuron_count: u16,
}

impl SparseSynapseMatrix {
    /// New empty CSR matrix sized for `neuron_count` presynaptic neurons.
    #[must_use]
    pub fn new(neuron_count: u16, estimated_synapses: usize) -> Self {
        Self {
            weights: Vec::with_capacity(estimated_synapses),
            col_indices: Vec::with_capacity(estimated_synapses),
            synapse_indices: Vec::with_capacity(estimated_synapses),
            pre_ids: Vec::with_capacity(estimated_synapses),
            row_ptrs: vec![0; neuron_count as usize + 1],
            weight_index_of: Vec::with_capacity(estimated_synapses),
            rv_row_ptrs: Vec::new(),
            rv_pre_ids: Vec::with_capacity(estimated_synapses),
            rv_syn_indices: Vec::with_capacity(estimated_synapses),
            neuron_count,
        }
    }

    /// Append a synapse. Edges are stored in insertion order; `row_ptrs` is
    /// bumped incrementally as a best-effort estimate (correct only when edges
    /// are added in non-decreasing `pre_id` order). For arbitrary insertion
    /// order — including `build_balanced` — call [`finalize`] before reading
    /// [`connections`], or the slices will hold the wrong edges.
    ///
    /// [`finalize`]: Self::finalize
    /// [`connections`]: Self::connections
    pub fn add(&mut self, pre_id: u16, post_id: u16, weight: i16, synapse_index: usize) {
        debug_assert!(
            pre_id < self.neuron_count,
            "pre_id {pre_id} ≥ neuron_count {}",
            self.neuron_count
        );
        self.weights.push(weight);
        self.col_indices.push(post_id);
        self.synapse_indices.push(synapse_index);
        self.pre_ids.push(pre_id);
        // Pre-finalize inverse permutation is identity: insertion order means
        // synapse i lives at weights[i]. finalize() overwrites it post-sort.
        self.weight_index_of.push(synapse_index);
        // Incremental row_ptrs: bump every row after pre_id by 1. This is only
        // authoritative for sorted insertion; finalize() overwrites it for the
        // general case.
        for row in (pre_id as usize + 1)..self.row_ptrs.len() {
            self.row_ptrs[row] += 1;
        }
    }

    /// Build the CSR layout authoritatively via a stable counting sort by
    /// `pre_id`. O(edges + neurons). **Must be called after all [`add`]s and
    /// before any [`connections`] read whenever edges were added out of
    /// `pre_id` order** (the default for `build_topology`). Reorders the four
    /// forward parallel arrays so each presynaptic neuron's outgoing edges are
    /// contiguous, rebuilds `row_ptrs` from real per-neuron out-degrees,
    /// builds the inverse permutation `weight_index_of` (so [`set_weight`]
    /// stays correct post-sort), and builds the reverse CSR (by `post_id`) so
    /// [`incoming`] works.
    ///
    /// [`add`]: Self::add
    /// [`connections`]: Self::connections
    /// [`incoming`]: Self::incoming
    /// [`set_weight`]: Self::set_weight
    pub fn finalize(&mut self) {
        let n = self.neuron_count as usize;
        let m = self.weights.len();

        // At entry, weights/col_indices/synapse_indices/pre_ids are all in
        // insertion order and mutually aligned. Capture the post + syn copies
        // now (before the forward scatter reassigns col_indices/synapse_indices)
        // so the reverse CSR can be counting-sorted by post_id from the same
        // insertion-order source.
        let ins_post: Vec<u16> = self.col_indices.clone();
        let ins_syn: Vec<usize> = self.synapse_indices.clone();

        // ----- Forward sort by pre_id -----
        // Count out-degree per presynaptic neuron.
        let mut degree = vec![0u32; n];
        for &pre in &self.pre_ids {
            degree[pre as usize] += 1;
        }
        // Prefix sum → row_ptrs (length n+1, last entry = total edges).
        self.row_ptrs = vec![0; n + 1];
        for (k, &deg) in degree.iter().enumerate() {
            self.row_ptrs[k + 1] = self.row_ptrs[k] + deg;
        }
        // Stable scatter into sorted order.
        let mut new_weights = vec![0i16; m];
        let mut new_col = vec![0u16; m];
        let mut new_syn = vec![0usize; m];
        let mut cursor = self.row_ptrs.clone();
        for (i, &pre) in self.pre_ids.iter().enumerate() {
            let pos = cursor[pre as usize] as usize;
            new_weights[pos] = self.weights[i];
            new_col[pos] = self.col_indices[i];
            new_syn[pos] = self.synapse_indices[i];
            cursor[pre as usize] += 1;
        }
        self.weights = new_weights;
        self.col_indices = new_col;
        self.synapse_indices = new_syn;

        // ----- Inverse permutation: weight_index_of[syn_idx] = sorted position -----
        // After the scatter, synapse_indices[pos] holds the synapse index living
        // at sorted position pos. Invert it so set_weight(syn_idx) lands on the
        // right slot. Without this, set_weight(syn_idx) would write weights[syn_idx]
        // — which after the sort holds a *different* edge's weight (the bd5b098
        // latent bug: plasticity deltas were written to the wrong CSR slots,
        // corrupting propagation weights).
        let mut inv = vec![0usize; m];
        for (pos, &syn_idx) in self.synapse_indices.iter().enumerate() {
            if syn_idx < m {
                inv[syn_idx] = pos;
            }
        }
        self.weight_index_of = inv;

        // ----- Reverse CSR by post_id (for the LTP post-firing path) -----
        // Stable counting-sort the same edges by postsynaptic neuron, carrying
        // pre_id (still insertion-order in self.pre_ids) and synapse index
        // (ins_syn, captured above). ins_post is the sort key.
        let mut post_degree = vec![0u32; n];
        for &post in &ins_post {
            let p = post as usize;
            if p < n {
                post_degree[p] += 1;
            }
        }
        let mut rv_row_ptrs = vec![0u32; n + 1];
        for (k, &deg) in post_degree.iter().enumerate() {
            rv_row_ptrs[k + 1] = rv_row_ptrs[k] + deg;
        }
        let mut rv_pre_ids = vec![0u16; m];
        let mut rv_syn_indices = vec![0usize; m];
        let mut rcursor = rv_row_ptrs.clone();
        for i in 0..m {
            let post = ins_post[i] as usize;
            if post < n {
                let pos = rcursor[post] as usize;
                rv_pre_ids[pos] = self.pre_ids[i];
                rv_syn_indices[pos] = ins_syn[i];
                rcursor[post] += 1;
            }
        }
        self.rv_row_ptrs = rv_row_ptrs;
        self.rv_pre_ids = rv_pre_ids;
        self.rv_syn_indices = rv_syn_indices;
    }

    /// Iterate synapses from `pre_id`. Returns `(post_id, weight, synapse_index)` tuples.
    /// O(1) setup, O(out-degree) iteration.
    #[must_use]
    pub fn connections(&self, pre_id: u16) -> SynapseIter<'_> {
        debug_assert!(pre_id < self.neuron_count);
        let start = self.row_ptrs[pre_id as usize] as usize;
        let end = self.row_ptrs[pre_id as usize + 1] as usize;
        SynapseIter {
            weights: &self.weights[start..end],
            col_indices: &self.col_indices[start..end],
            synapse_indices: &self.synapse_indices[start..end],
            pos: 0,
        }
    }

    /// Iterate **incoming** synapses to `post_id` (reverse CSR). Returns
    /// `(pre_id, synapse_index)` tuples. O(1) setup, O(in-degree) iteration.
    /// Empty until [`finalize`] is called.
    ///
    /// [`finalize`]: Self::finalize
    #[must_use]
    pub fn incoming(&self, post_id: u16) -> IncomingIter<'_> {
        if self.rv_row_ptrs.is_empty() {
            return IncomingIter {
                pre_ids: &[],
                syn_indices: &[],
                pos: 0,
            };
        }
        debug_assert!(post_id < self.neuron_count);
        let start = self.rv_row_ptrs[post_id as usize] as usize;
        let end = self.rv_row_ptrs[post_id as usize + 1] as usize;
        IncomingIter {
            pre_ids: &self.rv_pre_ids[start..end],
            syn_indices: &self.rv_syn_indices[start..end],
            pos: 0,
        }
    }

    /// Update the stored weight for a synapse index. Routes through
    /// `weight_index_of` (the inverse permutation) so the write lands on the
    /// correct forward-sorted slot even after [`finalize`] reordered the arrays.
    ///
    /// [`finalize`]: Self::finalize
    pub fn set_weight(&mut self, synapse_index: usize, weight: i16) {
        if let Some(&pos) = self.weight_index_of.get(synapse_index) {
            if let Some(slot) = self.weights.get_mut(pos) {
                *slot = weight;
            }
        }
    }

    /// Drop all stored edges, keeping the neuron-count sizing and capacity.
    /// Used by [`SpikingNeuralNetwork::build_topology`] to make rebuilds
    /// idempotent.
    pub fn clear(&mut self) {
        self.weights.clear();
        self.col_indices.clear();
        self.synapse_indices.clear();
        self.pre_ids.clear();
        self.weight_index_of.clear();
        self.rv_row_ptrs.clear();
        self.rv_pre_ids.clear();
        self.rv_syn_indices.clear();
        self.row_ptrs.iter_mut().for_each(|p| *p = 0);
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
    synapse_indices: &'a [usize],
    pos: usize,
}

impl Iterator for SynapseIter<'_> {
    type Item = (u16, i16, usize); // (post_id, weight, synapse_index)

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.weights.len() {
            let item = (
                self.col_indices[self.pos],
                self.weights[self.pos],
                self.synapse_indices[self.pos],
            );
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }
}

/// Iterator over a single postsynaptic neuron's incoming synapses (reverse CSR).
#[derive(Debug, Clone)]
pub struct IncomingIter<'a> {
    pre_ids: &'a [u16],
    syn_indices: &'a [usize],
    pos: usize,
}

impl Iterator for IncomingIter<'_> {
    type Item = (u16, usize); // (pre_id, synapse_index)

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.pre_ids.len() {
            let item = (self.pre_ids[self.pos], self.syn_indices[self.pos]);
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }
}

/// Plasticity queue entry: `(pre_neuron_id, post_neuron_id, synapse_index, pre_spike_time_us)`.
///
/// `synapse_index` is the shared stable index into both `self.synapses` and the
/// corresponding CSR weight slot, so plasticity can update the same synapse the
/// propagation pass just used.
/// Drained by [`SpikingNeuralNetwork::update_plasticity`].
type PlasticityEntry = (u16, u16, usize, u32);

/// Main spiking neural network orchestrator.
///
/// Holds [`LIFNeuron`] + [`Synapse`] collections, a CSR [`SparseSynapseMatrix`]
/// for fast synaptic transmission, and an [`STDPRule`] for plasticity. One call to
/// [`step`](Self::step) advances the simulation by `time_step_us` microseconds.
pub struct SpikingNeuralNetwork {
    neurons: Vec<LIFNeuron>,
    /// Parallel to CSR insertion order. Indexed via the plasticity queue and the
    /// stable indices carried inside `synapse_matrix`.
    synapses: Vec<Synapse>,
    /// CSR storage for fast per-presynaptic iteration during transmission.
    ///
    /// Invariant: `synapse_matrix.synapse_indices[k]` identifies the same logical
    /// synapse as `synapses[synapse_matrix.synapse_indices[k]]`, and the weight in
    /// the corresponding CSR slot is kept in sync after every STDP update.
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
    /// Whether STDP weight updates are applied each step. Default `true`
    /// (preserves library behavior). The visualizer disables this for
    /// sustained-firing mode and toggles it on to watch learning happen.
    plasticity_enabled: bool,
    /// LFSR state for stochastic ternary bucket-flips (Stage 1.5b). Advanced
    /// once per active synapse in [`stochastic_ternary_step`]. Independent of
    /// `seed` (topology) so plasticity randomness decorrelates from wiring.
    ternary_flip_lfsr: u32,
    /// Voltage grid every neuron was constructed on (see
    /// [`LIFNeuron::voltage_resolution`]). Kept at network level so stats can
    /// convert native quanta back to mV.
    voltage_resolution: VoltageResolution,
}

impl SpikingNeuralNetwork {
    /// Construct a network with `neuron_count` neurons, `time_step_us` simulation
    /// step, and the given `topology`. Neurons are created with the biological
    /// 80/20 E/I ratio unless the topology overrides (e.g., Feedforward is all E).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `neuron_count == 0` or `time_step_us == 0`.
    pub fn new(neuron_count: u16, time_step_us: u32, topology: NetworkTopology) -> Result<Self> {
        Self::new_with_voltage_resolution(
            neuron_count,
            time_step_us,
            topology,
            VoltageResolution::Millivolt,
        )
    }

    /// [`new`](Self::new) with an explicit voltage grid for every neuron —
    /// [`VoltageResolution::CentiMillivolt`] opens the sub-mV regime (dead
    /// zone ≈ 2 μA) where the ternary substrate's ±12 μA recurrent pulses
    /// move the membrane. All downstream behavior (drive, STDP, plasticity)
    /// is grid-independent; only membrane arithmetic granularity changes.
    ///
    /// # Errors
    ///
    /// Same as [`new`](Self::new).
    pub fn new_with_voltage_resolution(
        neuron_count: u16,
        time_step_us: u32,
        topology: NetworkTopology,
        resolution: VoltageResolution,
    ) -> Result<Self> {
        if neuron_count == 0 {
            return Err(Error::InvalidParameter);
        }
        if time_step_us == 0 {
            return Err(Error::InvalidParameter);
        }

        // Neuron-type assignment honors the topology's ratio when `Balanced`
        // parameterizes it; otherwise the biological 80/20 default. Keeping
        // this in sync with `build_balanced`'s E/I partition is what upholds
        // the module-doc sign invariant (E → positive outgoing weight,
        // I → negative). Previously this was hardcoded to 0.8, so a Balanced
        // topology with a different ratio silently mismatched the wiring
        // partition and broke the invariant.
        let excitatory_ratio = match topology {
            NetworkTopology::Balanced { excitatory_ratio } => excitatory_ratio.clamp(0.0, 1.0),
            _ => DEFAULT_EXCITATORY_RATIO,
        };
        // Same truncation as `build_balanced`'s `exc_count` — the partition
        // must be bit-identical or the sign invariant breaks at non-integer
        // `n * ratio` boundaries.
        let exc_count = (neuron_count as f64 * excitatory_ratio) as u16;

        let mut neurons = Vec::with_capacity(neuron_count as usize);
        for id in 0..neuron_count {
            let nt = if id < exc_count {
                NeuronType::Excitatory
            } else {
                NeuronType::Inhibitory
            };
            neurons.push(LIFNeuron::new_with_type_resolution(id, nt, resolution));
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
            plasticity_enabled: true,
            ternary_flip_lfsr: TERNARY_FLIP_SEED,
            voltage_resolution: resolution,
        })
    }

    /// Build the configured topology. Must be called before [`step`].
    ///
    /// Idempotent: a second call clears any existing synapses, CSR state, and
    /// pending plasticity entries before rebuilding, so repeated calls produce
    /// a single topology rather than accumulating. Runtime spike counters are
    /// preserved (use [`reset`](Self::reset) to clear those too).
    /// Resets `stats.total_synapses` to the resulting synapse count.
    pub fn build_topology(&mut self) -> Result<()> {
        self.synapses.clear();
        self.synapse_matrix.clear();
        self.plasticity_queue.clear();
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
        // Authoritative CSR build: the topology builders add edges in arbitrary
        // pre_id order, so the incremental row_ptrs is wrong until this counting
        // sort reorders the parallel arrays by presynaptic neuron. Without it,
        // connections(pre) returns slices with the right count but the wrong
        // edges — corrupting both propagation and STDP targeting.
        self.synapse_matrix.finalize();
        self.stats.total_synapses = self.synapses.len() as u32;
        Ok(())
    }

    /// Advance the simulation by one `time_step_us`. Returns the spikes emitted
    /// this step in chronological (neuron-id) order.
    pub fn step(&mut self, input_currents: &[i16]) -> Result<Vec<Spike>> {
        let mut output_spikes: Vec<Spike> = Vec::new();
        let mut firing_neurons: Vec<u16> = Vec::new();

        // Clear previous step's synaptic currents and decay adaptation.
        // Synaptic current is cleared (instantaneous-synapse model: only this
        // step's spikes contribute). Adaptation MUST decay each step or it
        // accumulates (+2/spike) without bound and silences the network — a
        // bug that previously made every sustained run self-quench after ~3 s.
        for n in &mut self.neurons {
            n.clear_synaptic_current();
            n.decay_synaptic_current(self.time_step_us);
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
            for (post_id, weight, syn_idx) in self.synapse_matrix.connections(pre_id) {
                if let Some(post_n) = self.neurons.get_mut(post_id as usize) {
                    // Scale weight down to keep currents in a reasonable μA range.
                    post_n.add_synaptic_current(weight / 10);
                }
                self.plasticity_queue
                    .push((pre_id, post_id, syn_idx, self.current_time_us));
            }
        }

        // Phase 3: apply pairwise STDP plasticity (LTD + LTP passes).
        // Gated so callers (e.g. the visualizer) can run in a sustained-firing
        // mode with fixed weights, toggling learning on to observe it.
        if self.plasticity_enabled {
            self.update_plasticity(&firing_neurons);
        }

        // Advance time and stats.
        self.current_time_us = self.current_time_us.saturating_add(self.time_step_us);
        self.stats.total_spikes += output_spikes.len() as u64;
        self.update_stats();

        Ok(output_spikes)
    }

    /// Apply pairwise STDP for this step — both halves of the rule.
    ///
    /// **LTD pass (pre-firing, post-before-pre):** for every queued presynaptic
    /// spike, pair it with the postsynaptic neuron's most recent spike. If post
    /// fired this step too, the a9a2679 same-step tie-break yields `dt = +1` →
    /// LTD (the documented bias toward depression on coincidence); otherwise
    /// `dt = pre_time − post.last_spike ≥ 0` → LTD branch. This is the
    /// historically-existing path; left intact.
    ///
    /// **LTP pass (post-firing, pre-before-post):** for every postsynaptic
    /// neuron that fired this step, pair it with each incoming synapse's
    /// presynaptic partner. When that presynaptic neuron fired *earlier* (a
    /// previous step, within the STDP window) but *not this step*, `dt =
    /// pre.last_spike − post_time < 0` → LTP. This is the half that was missing
    /// through Stage 1.5c — the prior substrate was structurally LTD-only
    /// (`update_plasticity` only visited pre-firing events, so `dt ≥ 0` always
    /// and the LTP branch of `calculate_weight_change` was unreachable). It is
    /// now reachable via the reverse CSR ([`SparseSynapseMatrix::incoming`]).
    ///
    /// **No double-counting.** The two passes are disjoint per synapse per step:
    /// a synapse whose pre and post both fire this step is handled by the LTD
    /// pass (tie-break) and explicitly skipped by the LTP pass's `pre didn't
    /// fire this step` guard — preserving the a9a2679 same-step invariant.
    ///
    /// # Bug fix vs v0.1
    ///
    /// v0.1 used `dt_ltd = -1000` hardcoded for the LTD branch (post not firing
    /// this step), which is LTP sign. v2 computes dt from actual pre/post spike
    /// timing. And through Stage 1.5c v2 only computed the LTD half (pre-firing
    /// events); the LTP half is added here.
    ///
    /// [`SparseSynapseMatrix::incoming`]: SparseSynapseMatrix::incoming
    fn update_plasticity(&mut self, firing_neurons: &[u16]) {
        if self.plasticity_queue.is_empty() && firing_neurons.is_empty() {
            return;
        }
        let fired_this_step = |id: u16| firing_neurons.contains(&id);

        // ----- LTD pass: pre-firing events (post-before-pre) -----
        // dt ≥ 0 by construction (pre fires this step; post's reference spike is
        // ≤ pre_time), so only the LTD branch of the rule fires here.
        if !self.plasticity_queue.is_empty() {
            for &(_pre_id, post_id, syn_idx, pre_time) in &self.plasticity_queue.clone() {
                let Some(synapse) = self.synapses.get_mut(syn_idx) else {
                    continue;
                };
                let post_time = if fired_this_step(post_id) {
                    // Post fired this step → near-simultaneous → small positive
                    // dt → LTD. Use 1μs to break the tie toward LTD (treat pre
                    // as just after post). (a9a2679 same-step fix.)
                    pre_time.saturating_sub(1)
                } else {
                    // Post didn't fire this step — use its last actual spike.
                    // If post never fired, last_spike_time_us = 0, giving a
                    // large positive dt → LTD with decayed magnitude (~0 outside
                    // the window once the sim has run a while).
                    self.neurons
                        .get(post_id as usize)
                        .map_or(0, |n| n.last_spike_time_us)
                };
                // dt = pre_time - post_time. Positive (pre after post) → LTD.
                let dt_us: i32 = (pre_time as i64 - post_time as i64)
                    .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                let delta = self.plasticity_rule.calculate_weight_change(dt_us);
                synapse.update_weight(delta);
                debug_assert!(syn_idx < self.synapse_matrix.len());
                self.synapse_matrix.set_weight(syn_idx, synapse.weight);
                self.stats.plasticity_events += 1;
            }
        }

        // ----- LTP pass: post-firing events (pre-before-post) -----
        // For each firing post, pair with each incoming presynaptic spike that
        // happened strictly earlier (a previous step, within the window). The
        // guard `!fired_this_step(pre_id)` keeps this disjoint from the LTD pass
        // (same-step co-fire is LTD-only, preserving a9a2679) and prevents
        // double-counting. `last_spike_time_us > 0` excludes neurons that never
        // fired (their time-0 sentinel would otherwise fake a recent pre spike
        // and spuriously potentiate silent partners early in the run).
        let post_time = self.current_time_us;
        // Collect work items first: incoming() borrows synapse_matrix
        // immutably, but applying the delta mutates synapses + synapse_matrix.
        let mut ltp_work: Vec<(usize, i32)> = Vec::new();
        for &post_id in firing_neurons {
            for (pre_id, syn_idx) in self.synapse_matrix.incoming(post_id) {
                if fired_this_step(pre_id) {
                    continue; // same-step co-fire handled by LTD pass.
                }
                let Some(pre_n) = self.neurons.get(pre_id as usize) else {
                    continue;
                };
                let pre_time = pre_n.last_spike_time_us;
                if pre_time == 0 {
                    continue; // pre never fired — no real pre-before-post pair.
                }
                // dt = pre_time - post_time < 0 (pre fired earlier) → LTP branch.
                let dt_us: i32 = (pre_time as i64 - post_time as i64)
                    .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                if dt_us >= 0 {
                    continue; // defensive: only the LTP (dt<0) branch belongs here.
                }
                ltp_work.push((syn_idx, dt_us));
            }
        }
        for (syn_idx, dt_us) in ltp_work {
            let Some(synapse) = self.synapses.get_mut(syn_idx) else {
                continue;
            };
            let delta = self.plasticity_rule.calculate_weight_change(dt_us);
            if delta == 0 {
                continue; // outside the window — no event to count.
            }
            synapse.update_weight(delta);
            debug_assert!(syn_idx < self.synapse_matrix.len());
            self.synapse_matrix.set_weight(syn_idx, synapse.weight);
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
            total_v += i64::from(self.neurons[i].membrane_potential);
            sampled += 1;
        }
        if sampled > 0 {
            // Convert native quanta → mV for the stat (÷scale; identity on
            // the default grid, so historical numbers are unchanged).
            let s = self.voltage_resolution.scale();
            self.stats.avg_membrane_potential_mv =
                (total_v as f64 * 10.0) / (n as f64 * f64::from(s));
        }
        let time_sec = self.current_time_us as f64 / 1_000_000.0;
        if time_sec > 0.0 {
            self.stats.firing_rate_hz = self.stats.total_spikes as f64 / (time_sec * n as f64);
        }
    }

    /// Append a synapse. Both `SparseSynapseMatrix` and `synapses` vec get a copy.
    pub fn add_synapse(&mut self, pre_id: u16, post_id: u16, weight: i16) -> Result<()> {
        if pre_id as usize >= self.neurons.len() || post_id as usize >= self.neurons.len() {
            return Err(Error::IndexOutOfBounds);
        }
        let synapse = Synapse::new(pre_id, post_id, weight)?;
        let synapse_index = self.synapses.len();
        debug_assert_eq!(synapse_index, self.synapse_matrix.len());
        self.synapse_matrix
            .add(pre_id, post_id, weight, synapse_index);
        self.synapses.push(synapse);
        Ok(())
    }

    /// Rebuild the CSR layout (authoritative forward sort + reverse CSR +
    /// inverse permutation) after wiring the network externally via
    /// [`add_synapse`].
    ///
    /// [`build_topology`] already does this for its own builders; this method
    /// is the path for callers that construct synapse wiring themselves —
    /// e.g. importing a pretrained weight matrix edge by edge. Without it:
    ///
    /// - `connections(pre)` returns slices with the right count but the wrong
    ///   members whenever edges were added out of `pre_id` order (the
    ///   incremental `row_ptrs` is only correct for sorted insertion), and
    /// - `incoming(post)` returns nothing (the reverse CSR is empty until a
    ///   finalize), silently regressing plasticity to the pre-1.5d LTD-only
    ///   substrate — the post-firing LTP pass becomes unreachable.
    ///
    /// Contract: call **exactly once**, after all `add_synapse` calls and
    /// before the first [`step`]. Like [`SparseSynapseMatrix::finalize`], it
    /// is not idempotent on an already-finalized matrix (the counting sort's
    /// source arrays hold insertion order); rebuilding external wiring means
    /// clearing and re-adding. Also refreshes `stats.total_synapses`.
    ///
    /// [`step`]: Self::step
    /// [`SparseSynapseMatrix::finalize`]: SparseSynapseMatrix::finalize
    pub fn finalize_synapses(&mut self) {
        self.synapse_matrix.finalize();
        self.stats.total_synapses = self.synapses.len() as u32;
    }

    /// Enable or disable STDP weight updates. When disabled, `step()` still
    /// propagates spikes and advances time, but synapse weights stay fixed —
    /// useful for sustained-firing visualization or as a control baseline.
    pub fn set_plasticity_enabled(&mut self, enabled: bool) {
        self.plasticity_enabled = enabled;
    }

    /// Whether STDP weight updates are currently applied each step.
    #[must_use]
    pub fn plasticity_enabled(&self) -> bool {
        self.plasticity_enabled
    }

    /// One-shot BitNet-Round ternarization of all synapse weights: `γ = mean|w|`,
    /// each weight snapped to the nearest of `{-γ, 0, +γ}`. Syncs the CSR
    /// transmission weights. Returns `γ` (or `0` if there are no synapses).
    ///
    /// This is a *one-shot* quantizer for the ternary-bridge Stage 1 gate — it
    /// does **not** install an ongoing ternary mode. Keeping weights on-grid
    /// during a live STDP run is the caller's job: call [`reproject_ternary`]
    /// after each [`step`](Self::step).
    ///
    /// [`reproject_ternary`]: Self::reproject_ternary
    #[must_use]
    pub fn ternarize_weights(&mut self) -> i16 {
        // Collect into a temp so the immutable γ scan borrow ends before the
        // mutable snap pass (borrow-checker hygiene; one-shot op, alloc is fine).
        let weights: Vec<i16> = self.synapses.iter().map(|s| s.weight).collect();
        let gamma = crate::trit::tensor_scale(&weights);
        if gamma == 0 {
            return 0;
        }
        for (idx, s) in self.synapses.iter_mut().enumerate() {
            s.weight = crate::trit::project_to_ternary(s.weight, gamma);
            self.synapse_matrix.set_weight(idx, s.weight);
        }
        gamma
    }

    /// Re-project every synapse weight onto `{-gamma, 0, +gamma}` (nearest of
    /// three) and sync the CSR. Returns the count of weights whose *stored i16
    /// value* changed this call (i.e. STDP had pushed them off-grid).
    ///
    /// Note: this counts i16 snaps, not ternary *bucket* transitions — a snap
    /// from `+130` back to `+125` is a nonzero return but NOT a bucket flip.
    /// Callers running the Stage 1 gate should track bucket flips separately
    /// (classify with [`Trit::from_weight`] before/after) for the learning
    /// signal; that's the metric that determines the gate verdict.
    ///
    /// [`Trit::from_weight`]: crate::trit::Trit::from_weight
    pub fn reproject_ternary(&mut self, gamma: i16) -> u32 {
        if gamma == 0 {
            return 0;
        }
        let mut snapped = 0u32;
        for (idx, s) in self.synapses.iter_mut().enumerate() {
            let new_w = crate::trit::project_to_ternary(s.weight, gamma);
            if new_w != s.weight {
                snapped += 1;
            }
            s.weight = new_w;
            self.synapse_matrix.set_weight(idx, s.weight);
        }
        snapped
    }

    /// Stochastic ternary bucket-flip step (Stage 1.5b).
    ///
    /// For every synapse, the STDP delta applied during [`step`](Self::step)
    /// pushed the weight slightly off-grid (the weight was on-grid before the
    /// step). This method:
    ///
    /// 1. Measures the off-grid residual (= this step's STDP delta, since
    ///    `weight − project(weight) = delta` for small deltas relative to γ/2).
    /// 2. Draws a Bernoulli trial with `P(flip) ∝ |residual|` via the network's
    ///    independent LFSR.
    /// 3. On success, flips the ternary bucket one step toward the delta's sign
    ///    (LTP → +γ, LTD → −γ), saturating at the extreme bucket and respecting
    ///    each synapse's `[min_weight, max_weight]` bounds.
    /// 4. Snaps all weights back onto `{-γ, 0, +γ}` regardless.
    ///
    /// The stored weight is **genuinely ternary** at all times — no latent or
    /// shadow state. Call after each `step()` in the Stage 1.5b regime, the
    /// same way [`reproject_ternary`] is called in the Stage 1 (deterministic)
    /// regime.
    ///
    /// Returns the number of ternary bucket transitions (flips) this call.
    ///
    /// [`reproject_ternary`]: Self::reproject_ternary
    pub fn stochastic_ternary_step(&mut self, gamma: i16) -> u32 {
        if gamma == 0 {
            return 0;
        }
        let mut flips = 0u32;
        let mut lfsr = self.ternary_flip_lfsr;
        for (idx, s) in self.synapses.iter_mut().enumerate() {
            let projected = crate::trit::project_to_ternary(s.weight, gamma);
            let residual = s.weight - projected;
            if residual == 0 {
                continue; // No STDP event this step (or delta was zero).
            }
            lfsr = advance_lfsr(lfsr);
            let draw = (lfsr & 0xFFFF) as u16;
            let target =
                crate::trit::stochastic_ternary_flip(s.weight, gamma, residual, draw);
            let clamped = target.clamp(s.min_weight, s.max_weight);
            if clamped != projected {
                flips += 1;
            }
            s.weight = clamped;
            self.synapse_matrix.set_weight(idx, s.weight);
        }
        self.ternary_flip_lfsr = lfsr;
        flips
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
        self.ternary_flip_lfsr = TERNARY_FLIP_SEED;
    }

    /// Read-only access to the synapse collection (for analysis / visualization).
    /// Useful for demos, stats, and weight-evolution tracking.
    #[must_use]
    pub fn synapses(&self) -> &[Synapse] {
        &self.synapses
    }

    /// Read-only access to the neuron collection (for analysis / visualization).
    #[must_use]
    pub fn neurons(&self) -> &[LIFNeuron] {
        &self.neurons
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
    #[must_use]
    pub fn spike_history(&self) -> &VecDeque<Spike> {
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

    /// Watts-Strogatz small-world: ring lattice with probabilistic **rewiring**.
    ///
    /// For each directed local edge `(i, i+offset)`, with probability
    /// `rewiring_prob` the target is replaced by a uniformly random node `≠ i`;
    /// otherwise the local edge is kept. This is true Watts-Strogatz rewiring
    /// (keep-or-replace), not shortcut augmentation — total edge count is
    /// conserved at `~n × local_connections` (minus any wrap-around self-skips).
    fn build_small_world(&mut self, local_connections: u8, rewiring_prob: f64) -> Result<()> {
        let n = self.neurons.len() as u16;
        if n < 2 {
            return Ok(());
        }
        let p = rewiring_prob.clamp(0.0, 1.0);
        let mut rng = self.seed;
        for i in 0..n {
            for offset in 1..=local_connections as u16 {
                let local_target = (i + offset) % n;
                // Bug fix vs v0.1: when `local_connections >= n`, the modulo
                // wraps to self-connections (e.g., n=5, offset=5 → target = i). Skip them.
                if local_target == i {
                    continue;
                }
                rng = advance_lfsr(rng);
                let roll = (rng & 0xFFFF) as f64 / 65_536.0;
                let target = if roll < p {
                    // Rewire: pick a uniformly random target ≠ i.
                    rng = advance_lfsr(rng);
                    let mut new_target = (rng % n as u32) as u16;
                    if new_target == i {
                        new_target = (i + 1) % n;
                    }
                    new_target
                } else {
                    local_target
                };
                let weight = typed_weight(self.neurons[i as usize].neuron_type, 0);
                self.add_synapse(i, target, weight)?;
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
        NetworkTopology::SmallWorld {
            local_connections,
            ..
        } => neuron_count as usize * (*local_connections as usize),
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
    use crate::trit::Trit;
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
        let net = SpikingNeuralNetwork::new(100, 1000, NetworkTopology::default()).expect("valid");
        let excitatory = net
            .neurons
            .iter()
            .filter(|n| n.neuron_type == NeuronType::Excitatory)
            .count();
        assert_eq!(
            excitatory, 80,
            "default topology should give 80% excitatory"
        );
    }

    #[test]
    fn random_topology_produces_synapses() {
        let mut net =
            SpikingNeuralNetwork::new(50, 1000, NetworkTopology::Random { connectivity: 0.2 })
                .expect("valid");
        net.build_topology().expect("build");
        assert!(
            net.synapse_count() > 0,
            "random topology must produce synapses"
        );
    }

    #[test]
    fn smallworld_topology_produces_synapses() {
        let mut net = SpikingNeuralNetwork::new(
            50,
            1000,
            NetworkTopology::SmallWorld {
                local_connections: 4,
                rewiring_prob: 0.1,
            },
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
            NetworkTopology::Feedforward {
                layers: &[10, 15, 5],
            },
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
            NetworkTopology::Balanced {
                excitatory_ratio: 0.8,
            },
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
            NetworkTopology::Feedforward {
                layers: &[10, 15, 10],
            }, // sums to 35, not 30
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
            NetworkTopology::Balanced {
                excitatory_ratio: 1.0,
            },
        )
        .expect("valid net init");
        let err = net.build_topology();
        assert!(err.is_err(), "0 inhibitory must error");
    }

    #[test]
    fn step_advances_time_by_time_step() {
        let mut net =
            SpikingNeuralNetwork::new(10, 1000, NetworkTopology::default()).expect("valid");
        net.build_topology().expect("build");
        let inputs = vec![0; 10];
        let _ = net.step(&inputs).expect("step");
        assert_eq!(net.current_time_us(), 1000);
        let _ = net.step(&inputs).expect("step");
        assert_eq!(net.current_time_us(), 2000);
    }

    #[test]
    fn step_with_strong_input_produces_spikes() {
        let mut net =
            SpikingNeuralNetwork::new(10, 1000, NetworkTopology::default()).expect("valid");
        net.build_topology().expect("build");
        let inputs = vec![1000; 10]; // Strong input to all
        let mut total_spikes = 0u32;
        for _ in 0..50 {
            let spikes = net.step(&inputs).expect("step");
            total_spikes += spikes.len() as u32;
        }
        assert!(
            total_spikes > 0,
            "strong input over 50 steps must produce spikes"
        );
    }

    #[test]
    fn reset_clears_time_and_spikes() {
        let mut net =
            SpikingNeuralNetwork::new(20, 1000, NetworkTopology::default()).expect("valid");
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
        let mut net =
            SpikingNeuralNetwork::new(10, 1000, NetworkTopology::default()).expect("valid");
        let err = net.add_synapse(0, 100, 50);
        assert!(err.is_err());
        let err = net.add_synapse(100, 0, 50);
        assert!(err.is_err());
    }

    #[test]
    fn add_synapse_rejects_self_connection() {
        let mut net =
            SpikingNeuralNetwork::new(10, 1000, NetworkTopology::default()).expect("valid");
        let err = net.add_synapse(3, 3, 100);
        assert!(err.is_err());
    }

    #[test]
    fn sparse_matrix_iter_returns_added_synapses() {
        let mut m = SparseSynapseMatrix::new(5, 4);
        m.add(0, 1, 100, 0);
        m.add(0, 2, 200, 1);
        m.add(2, 3, -150, 2);
        let row0: Vec<(u16, i16, usize)> = m.connections(0).collect();
        assert_eq!(row0, vec![(1, 100, 0), (2, 200, 1)]);
        let row2: Vec<(u16, i16, usize)> = m.connections(2).collect();
        assert_eq!(row2, vec![(3, -150, 2)]);
        let row1: Vec<(u16, i16, usize)> = m.connections(1).collect();
        assert!(row1.is_empty());
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn csr_finalize_recovers_correct_members_for_unsorted_insertion() {
        // Regression: add() in arbitrary pre_id order used to make connections(pre)
        // return slices with the right COUNT but the wrong MEMBERS (the incremental
        // row_ptrs only works for sorted insertion). finalize() counting-sorts by
        // pre_id so connections returns each neuron's actual outgoing edges.
        let mut m = SparseSynapseMatrix::new(5, 4);
        m.add(0, 1, 100, 0);
        m.add(2, 3, 200, 1);
        m.add(0, 4, 150, 2); // unsorted: pre=0 reappears after pre=2
        m.finalize();
        let mut row0: Vec<(u16, i16, usize)> = m.connections(0).collect();
        row0.sort_by_key(|t| t.0);
        assert_eq!(
            row0, vec![(1, 100, 0), (4, 150, 2)],
            "connections(0) must return only pre=0's edges"
        );
        let row2: Vec<(u16, i16, usize)> = m.connections(2).collect();
        assert_eq!(row2, vec![(3, 200, 1)], "connections(2) must return pre=2's edge only");
        // Empty rows stay empty.
        assert_eq!(m.connections(1).count(), 0);
        assert_eq!(m.connections(3).count(), 0);
        assert_eq!(m.connections(4).count(), 0);
        // Total conserved.
        let total: usize = (0..5).map(|i| m.connections(i).count()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn csr_connections_pre_id_consistent_after_build() {
        // Network-level regression for the unsorted-insertion CSR bug. After
        // build_topology (which now finalizes), connections(pre) must return
        // only synapses whose presynaptic neuron IS `pre`. The old incremental
        // row_ptrs violated this for build_balanced's arbitrary insertion order
        // — corrupting propagation targeting and STDP synapse selection.
        let mut net =
            SpikingNeuralNetwork::new(64, 1000, NetworkTopology::default()).expect("valid");
        net.build_topology().expect("build");
        for pre in 0..net.neuron_count() {
            for (_post, _w, syn_idx) in net.synapse_matrix.connections(pre) {
                assert_eq!(
                    net.synapses[syn_idx].pre_neuron_id, pre,
                    "CSR returned a synapse under connections({pre}) whose presynaptic neuron is {} — finalize() must group by pre_id",
                    net.synapses[syn_idx].pre_neuron_id
                );
            }
        }
    }

    #[test]
    fn finalize_synapses_sorts_external_adds_and_builds_both_csrs() {
        // Session D-2: the public external-wiring path. `add_synapse` in
        // arbitrary pre_id order + `finalize_synapses()` must produce the
        // authoritative forward CSR (right members, stable within-row order),
        // the reverse CSR (right incoming edges), and refreshed stats.
        let mut net = SpikingNeuralNetwork::new(
            4,
            1000,
            NetworkTopology::Random { connectivity: 0.0 },
        )
        .expect("valid");
        net.build_topology().expect("build");
        assert_eq!(net.synapse_count(), 0, "zero-connectivity start is empty");
        // Deliberately unsorted pre order (pre 2, pre 0, pre 2 again, pre 1).
        net.add_synapse(2, 0, -125).expect("add");
        net.add_synapse(0, 1, 125).expect("add");
        net.add_synapse(2, 3, -125).expect("add");
        net.add_synapse(1, 3, 0).expect("add");
        net.finalize_synapses();
        assert_eq!(net.synapse_count(), 4);
        assert_eq!(net.stats().total_synapses, 4, "stats must refresh");

        // Forward CSR: each row holds exactly its presynaptic neuron's edges,
        // stable order among same-pre adds (0 was added before 3 for pre 2).
        let row0: Vec<(u16, i16)> = net
            .synapse_matrix
            .connections(0)
            .map(|(p, w, _)| (p, w))
            .collect();
        assert_eq!(row0, vec![(1, 125)]);
        let row1: Vec<(u16, i16)> = net
            .synapse_matrix
            .connections(1)
            .map(|(p, w, _)| (p, w))
            .collect();
        assert_eq!(row1, vec![(3, 0)]);
        let row2: Vec<(u16, i16)> = net
            .synapse_matrix
            .connections(2)
            .map(|(p, w, _)| (p, w))
            .collect();
        assert_eq!(row2, vec![(0, -125), (3, -125)]);
        assert_eq!(net.synapse_matrix.connections(3).count(), 0);

        // Reverse CSR: post 3 receives from pre 1 and pre 2 (stable insertion
        // order among same-post adds: pre 2's edge was added before pre 1's).
        let inc3: Vec<u16> = net.synapse_matrix.incoming(3).map(|(p, _)| p).collect();
        assert_eq!(inc3, vec![2, 1]);
        let inc0: Vec<u16> = net.synapse_matrix.incoming(0).map(|(p, _)| p).collect();
        assert_eq!(inc0, vec![2]);
    }

    #[test]
    fn finalize_synapses_makes_ltp_reachable_on_external_wiring() {
        // Session D-2: the reason `finalize_synapses` must exist. The LTP
        // (post-firing) plasticity pass iterates the reverse CSR; on external
        // wiring it is empty until a finalize, so pre-before-post firing can
        // NEVER potentiate — the pre-1.5d LTD-only regression, silently. Same
        // drive on two nets: the finalized one potentiates, the unfinalized
        // one cannot (LTD clamps the zero-born excitatory weight at 0).
        let mut net = SpikingNeuralNetwork::new(
            4,
            1000,
            NetworkTopology::Random { connectivity: 0.0 },
        )
        .expect("valid");
        net.build_topology().expect("build");
        net.neurons[1].noise_amplitude_ua = 0;
        net.neurons[2].noise_amplitude_ua = 0;
        net.add_synapse(1, 2, 0).expect("add");
        net.finalize_synapses();

        let mut raw = SpikingNeuralNetwork::new(
            4,
            1000,
            NetworkTopology::Random { connectivity: 0.0 },
        )
        .expect("valid");
        raw.build_topology().expect("build");
        raw.neurons[1].noise_amplitude_ua = 0;
        raw.neurons[2].noise_amplitude_ua = 0;
        raw.add_synapse(1, 2, 0).expect("add");

        // Drive: pre (neuron 1) steps 0..6, then post (neuron 2) steps 7..17.
        // Pre fires ≈step 6 (integer LIF integration needs 7 driven steps to
        // reach −55 mV from rest), post ≈step 13 → pre-before-post within the
        // STDP window → LTP is the only rule that can raise this weight.
        for step in 0..18u32 {
            let mut inp = vec![0_i16; 4];
            if step < 7 {
                inp[1] = 600;
            } else {
                inp[2] = 600;
            }
            net.step(&inp).expect("step");
            raw.step(&inp).expect("step");
        }
        assert!(
            net.neurons[1].last_spike_time_us > 0 && net.neurons[2].last_spike_time_us > 0,
            "drive must make both pre and post fire (pre @ {}μs, post @ {}μs)",
            net.neurons[1].last_spike_time_us,
            net.neurons[2].last_spike_time_us
        );
        assert!(
            net.synapses[0].weight > 0,
            "finalized external wiring must allow LTP: weight {} (started 0)",
            net.synapses[0].weight
        );
        assert_eq!(
            raw.synapses[0].weight, 0,
            "without finalize the reverse CSR is empty — LTP unreachable, weight frozen at 0"
        );
    }

    #[test]
    fn lfsr_is_deterministic_same_seed() {
        let a = advance_lfsr(0xDEAD_BEEF);
        let b = advance_lfsr(0xDEAD_BEEF);
        assert_eq!(a, b);
    }

    #[test]
    fn smallworld_rewiring_conserves_edge_count() {
        // True Watts-Strogatz rewiring keeps-or-replaces each local edge; it
        // never augments. With `local_connections` well below `n` (no wrap-around
        // self-skip), the count must equal exactly `n × local_connections`.
        let mut net = SpikingNeuralNetwork::new(
            50,
            1000,
            NetworkTopology::SmallWorld {
                local_connections: 4,
                rewiring_prob: 0.3,
            },
        )
        .expect("valid");
        net.build_topology().expect("build");
        assert_eq!(
            net.synapse_count(),
            50 * 4,
            "rewiring must conserve edge count (keep-or-replace, not augment)"
        );
    }

    #[test]
    fn build_topology_is_idempotent_no_accumulation() {
        // Regression for the silent-accumulation footgun: a second build must
        // REPLACE the topology, not stack on top of the first.
        let mut net = SpikingNeuralNetwork::new(
            50,
            1000,
            NetworkTopology::Random { connectivity: 0.3 },
        )
        .expect("valid");
        net.build_topology().expect("first build");
        let count_first = net.synapse_count();
        assert!(count_first > 0, "first build must produce synapses");

        net.build_topology().expect("second build");
        let count_second = net.synapse_count();
        assert_ne!(
            count_second,
            2 * count_first,
            "second build must not double the synapse count"
        );
        // Random topology's count is formula-determined (connectivity ×
        // total_possible), so it is also stable across rebuilds.
        assert_eq!(count_second, count_first);
    }

    #[test]
    fn ternary_gate_stage1_deterministic_is_frozen() {
        // Stage 1 DETERMINISTIC baseline — pins the ruled-out negative result.
        // Under per-step re-projection with BitNet-Round γ = mean|w|, STDP
        // deltas (max ±5) are ~12× smaller than the bucket boundary γ/2 (≈62),
        // so no weight ever crosses a ternary threshold → 0 flips. This is the
        // known-dead regime the field identified; Stage 1.5b (stochastic flips)
        // is the reopen path. See docs/VISION.md "Stage 1" + "Stage 1.5 reopen
        // paths."
        let mut net =
            SpikingNeuralNetwork::new(128, 1000, NetworkTopology::default()).expect("valid");
        net.build_topology().expect("build");
        let gamma = net.ternarize_weights();
        assert!(gamma > 0, "balanced net must produce a nonzero γ");

        let mut prev: Vec<Trit> = net
            .synapses()
            .iter()
            .map(|s| Trit::from_weight(s.weight, gamma))
            .collect();
        net.set_plasticity_enabled(true);
        let inputs = vec![600_i16; 128];
        let mut flips = 0u64;
        for _ in 0..200 {
            let _ = net.step(&inputs).expect("step");
            net.reproject_ternary(gamma);
            for (i, s) in net.synapses().iter().enumerate() {
                let cur = Trit::from_weight(s.weight, gamma);
                if cur != prev[i] {
                    flips += 1;
                    prev[i] = cur;
                }
            }
        }
        assert_eq!(
            flips, 0,
            "Stage 1 deterministic: ternary learning must be frozen under per-step re-projection; got {flips} flips"
        );
    }

    #[test]
    fn ternary_gate_stage1_5b_stochastic_unfreezes_learning() {
        // Stage 1.5b CANARY — stochastic bucket-flips reopen ternary learning.
        // Under deterministic re-projection (Stage 1), STDP deltas can't cross
        // the γ/2 boundary → 0 flips. The stochastic rule dissolves that: each
        // STDP event does a Bernoulli(∝|δ|) draw to flip one bucket in the
        // delta's direction. The stored weight stays genuinely ternary — no
        // latent/shadow state.
        //
        // This test asserts nonzero bucket movement over a balanced 128-net run
        // — the signal that the regime changed from Stage 1's frozen baseline.
        let mut net =
            SpikingNeuralNetwork::new(128, 1000, NetworkTopology::default()).expect("valid");
        net.build_topology().expect("build");
        let gamma = net.ternarize_weights();
        assert!(gamma > 0, "balanced net must produce a nonzero γ");

        let mut prev: Vec<Trit> = net
            .synapses()
            .iter()
            .map(|s| Trit::from_weight(s.weight, gamma))
            .collect();
        net.set_plasticity_enabled(true);
        let inputs = vec![600_i16; 128];
        let mut flips = 0u64;
        for _ in 0..200 {
            let _ = net.step(&inputs).expect("step");
            net.stochastic_ternary_step(gamma);
            for (i, s) in net.synapses().iter().enumerate() {
                let cur = Trit::from_weight(s.weight, gamma);
                if cur != prev[i] {
                    flips += 1;
                    prev[i] = cur;
                }
            }
        }
        assert!(
            flips > 0,
            "Stage 1.5b: stochastic flips must produce nonzero bucket movement; got 0"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn ternary_gate_stage1_5c_selectivity_under_structured_input() {
        // Stage 1.5c CANARY — ternary STDP discriminates by correlation.
        //
        // 1.5b proved the stochastic-flip mechanism moves weights (802 flips)
        // but under *uniform synchronous* drive — a degenerate one-directional
        // collapse. This canary runs the real test: under *structured* input
        // (correlated groups, gapped rotation) where the i16 baseline
        // discriminates (intra-group E→E depress via co-fire LTD, inter-group
        // don't), ternary-stochastic must PRESERVE that differential (intra
        // mean < inter mean) with nonzero flips and non-collapsed spiking.
        // Gates whether the Stage 2 format bridge is worth building.
        // See examples/ternary_selectivity.rs for the full diagnostic.
        let neurons: u16 = 128;
        let exc = (f64::from(neurons) * 0.8) as u16;
        let groups: u16 = 4;
        let active_on = 60u32;
        let off_gap = 40u32;
        let slot_len = active_on + off_gap;
        let init_steps = (slot_len * u32::from(groups)) as usize;
        let learn_steps = 600usize;
        let total = init_steps + learn_steps;
        let i_active = 600_i16;
        let i_idle = 0_i16;
        let i_inh = 600_i16;
        let group_of = |nid: u16| -> u16 {
            let g = (u32::from(nid) * u32::from(groups) / u32::from(exc)) as u16;
            g.min(groups - 1)
        };
        // Gapped rotating drive: one group active per slot, silent gap between.
        let mut inputs: Vec<Vec<i16>> = Vec::with_capacity(total);
        for step in 0..total {
            let within = (step as u32) % (slot_len * u32::from(groups));
            let slot = within / slot_len;
            let within_slot = within % slot_len;
            let active_group =
                if slot < u32::from(groups) && within_slot < active_on { slot as u16 } else { groups };
            let mut inp = vec![i_inh; neurons as usize];
            for n in 0..exc {
                inp[n as usize] = if group_of(n) == active_group { i_active } else { i_idle };
            }
            inputs.push(inp);
        }
        let classify = |net: &SpikingNeuralNetwork| -> (Vec<usize>, Vec<usize>) {
            let mut intra = Vec::new();
            let mut inter = Vec::new();
            for (i, s) in net.synapses().iter().enumerate() {
                if s.pre_neuron_id < exc && s.post_neuron_id < exc {
                    if group_of(s.pre_neuron_id) == group_of(s.post_neuron_id) {
                        intra.push(i);
                    } else {
                        inter.push(i);
                    }
                }
            }
            (intra, inter)
        };
        let mean = |w: &[i16], idx: &[usize]| -> f64 {
            if idx.is_empty() {
                0.0
            } else {
                idx.iter().map(|&i| f64::from(w[i])).sum::<f64>() / idx.len() as f64
            }
        };

        // (1) i16 baseline control — MUST discriminate.
        let mut net =
            SpikingNeuralNetwork::new(neurons, 1000, NetworkTopology::default()).expect("valid");
        net.build_topology().expect("build");
        let (intra, inter) = classify(&net);
        net.set_plasticity_enabled(false);
        for inp in &inputs[..init_steps] {
            let _ = net.step(inp).expect("init");
        }
        net.set_plasticity_enabled(true);
        for inp in &inputs[init_steps..] {
            let _ = net.step(inp).expect("learn");
        }
        let i16_w: Vec<i16> = net.synapses().iter().map(|s| s.weight).collect();
        let i16_intra = mean(&i16_w, &intra);
        let i16_inter = mean(&i16_w, &inter);
        assert!(
            i16_inter - i16_intra > 20.0,
            "i16 control must discriminate by >20: inter={i16_inter:.2} intra={i16_intra:.2}"
        );

        // (2) Ternary stochastic — must preserve the differential + flip.
        let mut tnet =
            SpikingNeuralNetwork::new(neurons, 1000, NetworkTopology::default()).expect("valid");
        tnet.build_topology().expect("build");
        let (tintra, tinter) = classify(&tnet);
        tnet.set_plasticity_enabled(false);
        for inp in &inputs[..init_steps] {
            let _ = tnet.step(inp).expect("init");
        }
        let gamma = tnet.ternarize_weights();
        let mut prev: Vec<Trit> = tnet
            .synapses()
            .iter()
            .map(|s| Trit::from_weight(s.weight, gamma))
            .collect();
        tnet.set_plasticity_enabled(true);
        let mut flips = 0u64;
        for inp in &inputs[init_steps..] {
            let _ = tnet.step(inp).expect("learn");
            tnet.stochastic_ternary_step(gamma);
            for (i, s) in tnet.synapses().iter().enumerate() {
                let cur = Trit::from_weight(s.weight, gamma);
                if cur != prev[i] {
                    flips += 1;
                    prev[i] = cur;
                }
            }
        }
        let tern_w: Vec<i16> = tnet.synapses().iter().map(|s| s.weight).collect();
        let tern_intra = mean(&tern_w, &tintra);
        let tern_inter = mean(&tern_w, &tinter);
        assert!(flips > 0, "ternary must produce bucket flips; got 0");
        assert!(
            tern_inter - tern_intra > f64::from(gamma) / 2.0,
            "ternary must discriminate by >γ/2: inter={tern_inter:.2} intra={tern_intra:.2} γ={gamma}"
        );
    }

    #[test]
    fn same_step_cofire_biases_toward_ltd() {
        let mut net = SpikingNeuralNetwork::new(
            2,
            1000,
            NetworkTopology::Balanced {
                excitatory_ratio: 0.5,
            },
        )
        .expect("valid");
        net.neurons[0].noise_amplitude_ua = 0;
        net.neurons[1].noise_amplitude_ua = 0;
        net.add_synapse(0, 1, 100).expect("synapse");

        net.neurons[0].last_spike_time_us = 1_000;
        net.neurons[1].last_spike_time_us = 1_000;
        net.plasticity_queue.push((0, 1, 0, 1_000));
        net.update_plasticity(&[0, 1]);

        assert!(
            net.synapses[0].weight < 100,
            "same-step tie should depress weight"
        );
    }

    #[test]
    fn plasticity_updated_weight_affects_future_propagation() {
        let mut net = SpikingNeuralNetwork::new(
            2,
            1000,
            NetworkTopology::Balanced {
                excitatory_ratio: 0.5,
            },
        )
        .expect("valid");
        net.neurons[0].noise_amplitude_ua = 0;
        net.neurons[1].noise_amplitude_ua = 0;
        net.add_synapse(0, 1, 100).expect("synapse");

        net.neurons[0].last_spike_time_us = 1_000;
        net.neurons[1].last_spike_time_us = 1_000;
        net.plasticity_queue.push((0, 1, 0, 1_000));
        net.update_plasticity(&[0, 1]);
        let updated_weight = net.synapses[0].weight;
        assert_ne!(updated_weight, 100, "plasticity should change weight");

        net.neurons[1].clear_synaptic_current();
        for (post_id, weight, _) in net.synapse_matrix.connections(0) {
            if post_id == 1 {
                net.neurons[1].add_synaptic_current(weight / 10);
            }
        }
        assert_eq!(
            net.neurons[1].synaptic_current_ua,
            updated_weight / 10,
            "propagation must use the updated synapse weight"
        );
    }

    #[test]
    fn csr_weight_stays_in_sync_with_synapse_after_plasticity() {
        let mut net = SpikingNeuralNetwork::new(
            2,
            1000,
            NetworkTopology::Balanced {
                excitatory_ratio: 0.5,
            },
        )
        .expect("valid");
        net.neurons[0].noise_amplitude_ua = 0;
        net.neurons[1].noise_amplitude_ua = 0;
        net.add_synapse(0, 1, 100).expect("synapse");

        let _ = net.step(&[1000, 1000]).expect("plasticity step");
        let synapse_weight = net.synapses[0].weight;
        let csr_edge = net
            .synapse_matrix
            .connections(0)
            .next()
            .expect("csr edge must exist");
        assert_eq!(csr_edge.2, 0, "first inserted synapse should keep index 0");
        assert_eq!(
            csr_edge.1, synapse_weight,
            "CSR transmission weight must mirror synapse weight after STDP"
        );
    }

    // ----- Full pairwise STDP (LTP + LTD) regression suite -----
    //
    // Through Stage 1.5c the substrate was structurally LTD-only:
    // `update_plasticity` visited only pre-firing events, so `dt ≥ 0` always and
    // the LTP branch (`dt < 0`) was unreachable. The post-firing LTP path added
    // in this change makes the rule genuinely bidirectional. These tests pin
    // both halves and the invariants that keep them from interfering.

    #[test]
    fn ltp_post_firing_strengthens_synapse_when_pre_fired_earlier() {
        // The focused proof that LTP is now reachable in orchestration: a
        // postsynaptic spike paired with a recent presynaptic spike (pre-before-
        // post, dt < 0) potentiates the synapse. This was impossible before the
        // post-firing LTP pass existed.
        let mut net = SpikingNeuralNetwork::new(
            2,
            1000,
            NetworkTopology::Balanced {
                excitatory_ratio: 0.5,
            },
        )
        .expect("valid");
        net.neurons[0].noise_amplitude_ua = 0;
        net.neurons[1].noise_amplitude_ua = 0;
        net.add_synapse(0, 1, 100).expect("synapse");
        // finalize() builds the reverse CSR so incoming(post) resolves the
        // pre→post edge for the LTP pass.
        net.synapse_matrix.finalize();
        assert_eq!(
            net.synapse_matrix.incoming(1).count(),
            1,
            "reverse CSR must list the one incoming edge to post 1"
        );

        let start_weight = net.synapses[0].weight;
        // Pre fired 3 ms ago; post fires "now" at current_time_us = 10_000.
        net.neurons[0].last_spike_time_us = 7_000;
        net.current_time_us = 10_000;
        // Only the post fired this step; plasticity_queue stays empty (no pre
        // spike this step) so the LTD pass is inert — only LTP can fire.
        net.update_plasticity(&[1]);

        assert!(
            net.synapses[0].weight > start_weight,
            "pre-before-post (dt = 7000−10000 = −3000μs) must potentiate: was {start_weight}, now {}",
            net.synapses[0].weight
        );
        // And the CSR slot must mirror the strengthened synapse (inverse-perm fix).
        let csr_w = net
            .synapse_matrix
            .connections(0)
            .next()
            .expect("csr edge")
            .1;
        assert_eq!(
            csr_w, net.synapses[0].weight,
            "CSR weight must mirror potentiated synapse weight"
        );
    }

    #[test]
    fn ltp_pass_does_not_double_count_same_step_cofire() {
        // a9a2679 invariant under full STDP: when pre and post both fire in the
        // same step, the LTD pass applies its +1μs tie-break (→ LTD) and the LTP
        // pass must SKIP the edge (pre fired this step) — it must not cancel the
        // depression. Net effect on a same-step co-fire is still depression.
        let mut net = SpikingNeuralNetwork::new(
            2,
            1000,
            NetworkTopology::Balanced {
                excitatory_ratio: 0.5,
            },
        )
        .expect("valid");
        net.neurons[0].noise_amplitude_ua = 0;
        net.neurons[1].noise_amplitude_ua = 0;
        net.add_synapse(0, 1, 100).expect("synapse");
        net.synapse_matrix.finalize();

        let start_weight = net.synapses[0].weight;
        net.neurons[0].last_spike_time_us = 1_000;
        net.neurons[1].last_spike_time_us = 1_000;
        net.current_time_us = 1_000;
        // Both fire this step; queue the pre-firing event as step() would.
        net.plasticity_queue.push((0, 1, 0, 1_000));
        net.update_plasticity(&[0, 1]);

        assert!(
            net.synapses[0].weight < start_weight,
            "same-step co-fire must still depress under full STDP: was {start_weight}, now {}",
            net.synapses[0].weight
        );
    }

    #[test]
    fn ltd_pre_after_post_still_depresses_under_full_stdp() {
        // LTD half still works under the now-bidirectional rule: pre fires this
        // step, post fired earlier → dt > 0 → depression. The LTP pass must not
        // fire here (post did not fire this step), so the net is pure LTD.
        let mut net = SpikingNeuralNetwork::new(
            2,
            1000,
            NetworkTopology::Balanced {
                excitatory_ratio: 0.5,
            },
        )
        .expect("valid");
        net.neurons[0].noise_amplitude_ua = 0;
        net.neurons[1].noise_amplitude_ua = 0;
        net.add_synapse(0, 1, 100).expect("synapse");
        net.synapse_matrix.finalize();

        let start_weight = net.synapses[0].weight;
        // Post fired 3 ms ago; pre fires now.
        net.neurons[1].last_spike_time_us = 7_000;
        net.current_time_us = 10_000;
        net.plasticity_queue.push((0, 1, 0, 10_000));
        // Only pre fired this step.
        net.update_plasticity(&[0]);

        assert!(
            net.synapses[0].weight < start_weight,
            "post-before-pre (dt = +3000μs) must depress under full STDP: was {start_weight}, now {}",
            net.synapses[0].weight
        );
    }

    #[test]
    fn full_stdp_is_bidirectional_in_orchestration() {
        // Orchestration-level proof that both branches are reachable in a real
        // (non-hand-primed) run. Under sustained drive, recurrent timing jitter
        // produces both pre-after-post (LTD) and pre-before-post (LTP) pairs, so
        // some weights must increase AND some decrease. Through 1.5c `up` was
        // always 0 (depression-only); this pins that the substrate no longer is.
        let mut net =
            SpikingNeuralNetwork::new(64, 1000, NetworkTopology::default()).expect("valid");
        net.build_topology().expect("build");
        let init: Vec<i16> = net.synapses().iter().map(|s| s.weight).collect();
        net.set_plasticity_enabled(true);
        let inputs = vec![600_i16; 64];
        for _ in 0..400 {
            let _ = net.step(&inputs).expect("step");
        }
        let (mut up, mut down) = (0u32, 0u32);
        for (s, &w0) in net.synapses().iter().zip(init.iter()) {
            if s.weight > w0 {
                up += 1;
            } else if s.weight < w0 {
                down += 1;
            }
        }
        assert!(up > 0, "LTP must be reachable: got 0 weights increased");
        assert!(down > 0, "LTD must still fire: got 0 weights decreased");
    }

    #[test]
    fn csr_weight_index_of_keeps_slots_synced_in_multi_synapse_net() {
        // Regression for the bd5b098 latent bug: finalize()'s counting sort
        // reorders `weights` by pre_id, so the old `set_weight(syn_idx)` (which
        // indexed weights[syn_idx] directly) wrote deltas to the WRONG slots —
        // desynchronizing the CSR transmission weights from `synapses[].weight`.
        // The inverse-permutation `weight_index_of` routes each write to the
        // correct sorted position. This runs a real balanced net (finalize
        // reorders heavily) and asserts every CSR slot matches its synapse.
        let mut net =
            SpikingNeuralNetwork::new(64, 1000, NetworkTopology::default()).expect("valid");
        net.build_topology().expect("build");
        net.set_plasticity_enabled(true);
        let inputs = vec![600_i16; 64];
        for _ in 0..50 {
            let _ = net.step(&inputs).expect("step");
        }
        let mut mismatches = 0u32;
        for pre in 0..net.neuron_count() {
            for (_post, csr_w, syn_idx) in net.synapse_matrix.connections(pre) {
                if csr_w != net.synapses[syn_idx].weight {
                    mismatches += 1;
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "every CSR forward slot must mirror its synapse weight after plasticity"
        );
        // Reverse CSR consistency: every incoming entry must name a synapse whose
        // post_neuron_id matches the queried post (pins the reverse counting sort).
        let mut rv_mismatches = 0u32;
        for post in 0..net.neuron_count() {
            for (pre, syn_idx) in net.synapse_matrix.incoming(post) {
                let s = &net.synapses[syn_idx];
                if s.pre_neuron_id != pre || s.post_neuron_id != post {
                    rv_mismatches += 1;
                }
            }
        }
        assert_eq!(
            rv_mismatches, 0,
            "every reverse-CSR edge must match its synapse's (pre, post)"
        );
    }

    #[test]
    fn reverse_csr_incoming_lists_correct_edges_after_finalize() {
        // Focused unit test for the reverse CSR: incoming(post) must return
        // exactly the edges whose post_neuron_id == post, in a net where
        // build_topology inserts in arbitrary order.
        let mut net =
            SpikingNeuralNetwork::new(48, 1000, NetworkTopology::default()).expect("valid");
        net.build_topology().expect("build");
        let mut by_post: Vec<Vec<(u16, usize)>> = (0..net.neuron_count())
            .map(|_| Vec::new())
            .collect();
        for (syn_idx, s) in net.synapses().iter().enumerate() {
            by_post[s.post_neuron_id as usize].push((s.pre_neuron_id, syn_idx));
        }
        for post in 0..net.neuron_count() {
            let mut got: Vec<(u16, usize)> = net.synapse_matrix.incoming(post).collect();
            got.sort_unstable();
            let mut want = by_post[post as usize].clone();
            want.sort_unstable();
            assert_eq!(
                got, want,
                "incoming({post}) must return post's actual incoming edges"
            );
        }
    }

    // ----- Voltage resolution propagation (the finer ruler) -----

    #[test]
    fn network_propagates_centimillivolt_resolution_to_every_neuron() {
        let mut net = SpikingNeuralNetwork::new_with_voltage_resolution(
            50,
            1000,
            NetworkTopology::Random { connectivity: 0.0 },
            VoltageResolution::CentiMillivolt,
        )
        .expect("constructs");
        net.build_topology().expect("empty build");
        for n in &net.neurons {
            assert_eq!(n.voltage_resolution, VoltageResolution::CentiMillivolt);
            assert_eq!(n.resting_potential, -7_000);
        }
        // E threshold −5_500 / I −5_000 on the scaled grid.
        assert_eq!(net.neurons[0].threshold, -5_500);
        assert_eq!(net.neurons[49].threshold, -5_000);
    }

    #[test]
    fn sub_dead_zone_drive_spikes_only_in_centimillivolt_network() {
        // Network-level dead-zone proof: 160 μA sustained (above the ~150 μA
        // E threshold current, below the mV grid's 200 μA dead zone from
        // rest). Same drive, same noise seeds — only the grid differs.
        // 5 neurons ⇒ exc_count = 4 (neuron 0 is Excitatory; the I neurons
        // sit at V_ss −54 mV < their −50 threshold and stay silent in both
        // grids). Zero connectivity ⇒ every neuron is driven independently.
        let drive = || vec![160_i16; 5];
        let mut mv =
            SpikingNeuralNetwork::new(5, 1000, NetworkTopology::Random { connectivity: 0.0 })
                .expect("constructs");
        mv.build_topology().expect("build");
        for n in &mut mv.neurons {
            n.noise_amplitude_ua = 0;
        }
        let mut mv_spikes = 0;
        for _ in 0..100 {
            mv_spikes += mv.step(&drive()).expect("step").len();
        }
        assert_eq!(mv_spikes, 0, "mV grid: blind to 160 μA from rest");

        let mut cmv = SpikingNeuralNetwork::new_with_voltage_resolution(
            5,
            1000,
            NetworkTopology::Random { connectivity: 0.0 },
            VoltageResolution::CentiMillivolt,
        )
        .expect("constructs");
        cmv.build_topology().expect("build");
        for n in &mut cmv.neurons {
            n.noise_amplitude_ua = 0;
        }
        let mut cmv_spikes = 0;
        for _ in 0..100 {
            cmv_spikes += cmv.step(&drive()).expect("step").len();
        }
        assert!(cmv_spikes >= 1, "centi grid: 160 μA fires the E neurons");
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

        /// Presynaptic-type sign invariant (module-doc claim): excitatory
        /// presynaptic → strictly positive weight; inhibitory → strictly negative.
        /// Pins the contract every topology builder relies on via `typed_weight`.
        #[test]
        fn prop_topology_weights_respect_presynaptic_type(
            n in 5u16..=30,
            topology in topology_strategy(),
        ) {
            let mut net = SpikingNeuralNetwork::new(n, 1000, topology)?;
            net.build_topology()?;
            // The sign invariant holds vacuously when a sparse topology wires
            // nothing (e.g. tiny n × tiny connectivity); proptest still explores
            // plenty of non-empty cases across the strategy range.
            for s in &net.synapses {
                let pre_type = net.neurons[s.pre_neuron_id as usize].neuron_type;
                match pre_type {
                    NeuronType::Excitatory => prop_assert!(
                        s.weight > 0,
                        "excitatory presynaptic must yield positive weight, got {}",
                        s.weight
                    ),
                    NeuronType::Inhibitory => prop_assert!(
                        s.weight < 0,
                        "inhibitory presynaptic must yield negative weight, got {}",
                        s.weight
                    ),
                }
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
                m.add(pre, post, 100, i);
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
            (1u8..=10u8, 0.0f64..=1.0).prop_map(|(lc, rp)| NetworkTopology::SmallWorld {
                local_connections: lc,
                rewiring_prob: rp,
            }),
            (0.5f64..=0.95).prop_map(|er| NetworkTopology::Balanced {
                excitatory_ratio: er
            }),
        ]
    }
}
