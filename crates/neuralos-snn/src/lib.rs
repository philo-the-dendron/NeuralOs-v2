#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_debug_implementations)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![doc = include_str!("../README.md")]

pub mod lif_neuron;
pub mod synapse;
#[cfg(feature = "std")]
pub mod network;

pub use lif_neuron::{
    LIFNeuron, NeuronBuilder, NeuronType, MEMBRANE_MV_MAX, MEMBRANE_MV_MIN, MAX_SPIKE_HISTORY,
};
pub use synapse::{STDPRule, Synapse, SynapseBuilder, SynapseType};
#[cfg(feature = "std")]
pub use network::{
    NetworkStats, NetworkTopology, SparseSynapseMatrix, SpikingNeuralNetwork, Spike,
};

/// Crate-level error type.
///
/// Toutes les fonctions publiques retournent `Result<T>` via ce type.
/// Pas de `unwrap()` / `expect()` en dehors des tests (leçon v0.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Paramètre invalide (ex: constante de temps nulle, capacité nulle).
    InvalidParameter,
    /// Index hors bornes (ex: ID de neurone ≥ taille du réseau).
    IndexOutOfBounds,
    /// Entrée d'historique de spike manquante (ex: pas de spike pré-synaptique
    /// enregistré dans la fenêtre STDP).
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

/// Alias de Result pour toute la crate.
pub type Result<T> = core::result::Result<T, Error>;
