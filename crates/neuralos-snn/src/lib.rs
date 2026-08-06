#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_debug_implementations)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![doc = include_str!("../README.md")]

/// Crate-level error type.
///
/// Toutes les fonctions publiques retournent `Result<T>` via ce type.
/// Pas de `unwrap()` / `expect()` en dehors des tests (leçon v0.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Paramètre invalide (ex: constante de temps négative, capacité nulle).
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
            Self::InvalidParameter => write!(f, "paramètre invalide"),
            Self::IndexOutOfBounds => write!(f, "index hors bornes"),
            Self::SpikeHistoryMissing => write!(f, "entrée d'historique de spike manquante"),
        }
    }
}

/// Alias de Result pour toute la crate.
pub type Result<T> = core::result::Result<T, Error>;

// Phase 0 (prochaine session) ajoute:
//   pub mod lif_neuron;
//   pub mod synapse;
//   pub mod stdp;
//   pub mod topology;
//
// Pour l'instant, la crate scaffold compile et expose son error type.
// Pas de module vide (leçon v0.1: "noms impressionnants, corps vides").
