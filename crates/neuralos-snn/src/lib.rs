#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]
#![cfg_attr(feature = "simd", deny(unsafe_code))]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![doc = include_str!("../README.md")]

pub mod bridge;
#[cfg(feature = "std")]
pub mod csr;
pub mod fixed;
pub mod kernel;
pub mod lif_neuron;
#[cfg(feature = "std")]
pub mod network;
pub mod nir;
#[cfg(feature = "simd")]
#[allow(unsafe_code)]
pub mod simd;
pub mod spike_recorder;
#[cfg(feature = "std")]
pub mod stats;
pub mod synapse;
pub mod trit;

pub use bridge::{
    decode_i2_s, decode_q1_0, decode_q2_0, encode_i2_s, encode_q2_0, half_to_f32_bits,
    half_to_milli, repack_i2s_to_kernel, wire_gamma_to_substrate, BridgeError,
};
pub use fixed::{FixedNetwork, FixedSynapse};
pub use kernel::{
    absmax_normalize_q15, pack_trits, ternary_matvec, unpack_trit, Q15_MAX, TRITS_PER_BYTE,
};
pub use lif_neuron::{LIFNeuron, NeuronType, VoltageResolution, MEMBRANE_MV_MAX, MEMBRANE_MV_MIN};
#[cfg(feature = "std")]
pub use network::{
    NetworkStats, NetworkTopology, SparseSynapseMatrix, Spike, SpikingNeuralNetwork,
};
pub use nir::{
    nir_export, nir_import, nir_scan, quantize_lif, quantize_linear, NirBuffers, NirError,
    NirImportOptions, NirLif, NirLifParams, NirLifPopulation, NirLinear, NirNode, NirNodeKind,
    NirNote, NirReport, NirScan, EXPORT_VERSION, NIR_NOTE_KINDS, NIR_REF_SHA,
};
pub use spike_recorder::{SpikeRecorder, MAX_SPIKE_HISTORY};
#[cfg(feature = "unstable-stdp")]
pub use synapse::STDPRule;
pub use synapse::{Synapse, SynapseType, SCALE};
pub use trit::{project_to_ternary, stochastic_ternary_flip, tensor_scale, ternarize, Trit};

/// Crate-level error type.
///
/// Fallible public functions return `Result<T>` via this type; infallible
/// accessors return plain values. No `unwrap()` / `expect()` outside tests
/// (v0.1 lesson).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Invalid parameter (e.g. zero time constant, zero neuron count).
    InvalidParameter,
    /// Index out of bounds (e.g. neuron id ≥ network size).
    IndexOutOfBounds,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameter => write!(f, "invalid parameter"),
            Self::IndexOutOfBounds => write!(f, "index out of bounds"),
        }
    }
}

impl core::error::Error for Error {}

// `core` and not `std`: the trait moved to `core` in 1.81, below the 1.92
// floor, so the impl holds in a `no_std` build — the target this library
// exists for. The assertion is its pin: without the impl it is `E0277`, in
// every configuration.
const _: fn() = || {
    fn a<T: core::error::Error>() {}
    a::<Error>();
};

/// Crate-wide `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;
