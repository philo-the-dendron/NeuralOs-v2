//! Network-level statistics — computed after each [`step`].
//!
//! Split from `network.rs` at R4(ii) (2026-08-20): the orchestrator file
//! held three responsibilities; this module owns the statistics surface.
//! Re-exported through `network` — the published path
//! `neuralos_snn::network::NetworkStats` is unchanged.

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
    /// STDP pairing histogram (session F instrumentation): in-window
    /// pairings by category — the Hebbian-attribution evidence.
    /// `same_step` = a9a2679 co-fire tie-break (dt = +1, LTD branch);
    /// `post_leads` = post-before-pre within the window (LTD branch);
    /// `pre_leads` = pre-before-post within the window (LTP branch —
    /// the Hebbian channel live transmission opened in session F).
    /// Out-of-window pairings contribute delta 0 and are not counted.
    pub stdp_pairs_same_step: u64,
    pub stdp_pairs_post_leads: u64,
    pub stdp_pairs_pre_leads: u64,
    /// Mean membrane potential across all neurons (mV). Computed post-step.
    pub avg_membrane_potential_mv: f64,
    /// Mean firing rate per neuron (Hz). Computed post-step.
    pub firing_rate_hz: f64,
}

impl NetworkStats {
    pub(crate) fn new(neuron_count: u16) -> Self {
        Self {
            total_neurons: neuron_count,
            total_synapses: 0,
            total_spikes: 0,
            plasticity_events: 0,
            stdp_pairs_same_step: 0,
            stdp_pairs_post_leads: 0,
            stdp_pairs_pre_leads: 0,
            avg_membrane_potential_mv: -70.0,
            firing_rate_hz: 0.0,
        }
    }
}
