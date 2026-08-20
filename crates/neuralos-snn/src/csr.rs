//! Compressed Sparse Row (CSR) synapse storage — O(1) per-presynaptic
//! iteration, forward + reverse.
//!
//! Split from `network.rs` at R4(ii) (2026-08-20). Re-exported through
//! `network` — the published paths
//! (`neuralos_snn::network::SparseSynapseMatrix` etc.) are unchanged.

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
