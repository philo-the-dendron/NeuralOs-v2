#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]
#![cfg_attr(feature = "simd", deny(unsafe_code))]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![doc = include_str!("../README.md")]

#[cfg(feature = "unstable-bridge")]
pub mod bridge;
#[cfg(feature = "std")]
mod csr;
pub mod fixed;
#[cfg(feature = "unstable-bridge")]
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
mod stats;
pub mod synapse;
pub mod trit;

pub use fixed::{FixedNetwork, FixedSynapse};
pub use lif_neuron::{LIFNeuron, NeuronType, VoltageResolution};
#[cfg(feature = "std")]
pub use network::{NetworkTopology, SpikingNeuralNetwork};
#[cfg(feature = "std")]
pub use nir::NirImport;
pub use nir::NirImportOptions;
#[cfg(feature = "unstable-stdp")]
pub use synapse::STDPRule;
pub use synapse::Synapse;

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
