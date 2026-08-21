#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_debug_implementations)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![doc = include_str!("../README.md")]

pub mod bridge;
pub mod kernel;
pub mod lif_neuron;
pub mod nir;
pub mod synapse;
pub mod trit;
#[cfg(feature = "std")]
pub mod network;
#[cfg(feature = "std")]
pub mod csr;
#[cfg(feature = "std")]
pub mod stats;
#[cfg(feature = "simd")]
pub mod simd;

pub use bridge::{
    decode_i2_s, decode_q1_0, decode_q2_0, encode_i2_s, encode_q2_0, half_to_f32_bits,
    half_to_milli, repack_i2s_to_kernel, wire_gamma_to_substrate, BridgeError,
};
pub use kernel::{
    absmax_normalize_q15, pack_trits, ternary_matvec, unpack_trit, Q15_MAX, TRITS_PER_BYTE,
};
pub use lif_neuron::{
    LIFNeuron, NeuronBuilder, NeuronType, VoltageResolution, MEMBRANE_MV_MAX, MEMBRANE_MV_MIN,
    MAX_SPIKE_HISTORY,
};
pub use nir::{
    nir_export, nir_import, nir_scan, NirBuffers, NirError, NirImportOptions, NirLif, NirLinear,
    NirNode, NirNodeKind, NirNote, NirReport, NirScan, EXPORT_VERSION, NIR_NOTE_KINDS, NIR_REF_SHA,
};
pub use synapse::{STDPRule, Synapse, SynapseBuilder, SynapseType, SCALE};
pub use trit::{project_to_ternary, stochastic_ternary_flip, tensor_scale, ternarize, Trit};
#[cfg(feature = "std")]
pub use network::{
    NetworkStats, NetworkTopology, SparseSynapseMatrix, SpikingNeuralNetwork, Spike,
};

/// Crate-level error type.
///
/// Fallible public functions return `Result<T>` via this type; infallible
/// accessors return plain values. No `unwrap()` / `expect()` outside tests
/// (v0.1 lesson).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Invalid parameter (e.g. zero time constant, zero neuron count).
    InvalidParameter,
    /// Index out of bounds (e.g. neuron id ≥ network size).
    IndexOutOfBounds,
    /// Missing spike-history entry (e.g. no presynaptic spike recorded inside
    /// the STDP window).
    SpikeHistoryMissing,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameter => write!(f, "invalid parameter"),
            Self::IndexOutOfBounds => write!(f, "index out of bounds"),
            Self::SpikeHistoryMissing => write!(f, "spike history entry missing"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Crate-wide `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;
