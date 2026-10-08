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
pub mod trace;
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
///
/// One variant per rule: a refusal returns the variant of the rule it
/// breaks, and a function's `# Errors` names the variants it returns. Not
/// exhaustive: a new rule may add a variant in a minor release. The
/// variants' order and their integer values (an `as` cast) are not
/// promised. A call that breaks two rules returns the variant of one of
/// them, and which one is not promised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A synapse from a neuron to itself ([`Synapse::new`], and
    /// `SpikingNeuralNetwork::add_synapse` through it).
    SelfConnection,
    /// A network of no neuron, or of more than 65,535: ids are `u16`
    /// (`SpikingNeuralNetwork::new` and `from_neurons`).
    NeuronCountOutOfRange,
    /// A time step of zero (`SpikingNeuralNetwork::new` and
    /// `from_neurons`).
    ZeroTimeStep,
    /// A neuron id not below the network's neuron count
    /// (`SpikingNeuralNetwork::add_synapse`).
    NeuronIdOutOfRange,
    /// A synaptic input divisor of 0 or above 32,767
    /// (`SpikingNeuralNetwork::set_synaptic_input_divisor`): the step
    /// divides by it as an `i16`.
    DivisorOutOfRange,
    /// A `Feedforward` layer list with a layer of no neuron, or whose sizes
    /// do not sum to the neuron count
    /// (`SpikingNeuralNetwork::build_topology`).
    BadLayerSizes,
    /// A `Balanced` ratio that leaves the network no excitatory or no
    /// inhibitory neuron (`SpikingNeuralNetwork::build_topology`).
    MissingNeuronType,
    /// A network with plasticity enabled, which a [`FixedNetwork`] does not
    /// have (`FixedNetwork::try_from`).
    PlasticityEnabled,
    /// A network whose neuron count is not the [`FixedNetwork`]'s `N`
    /// (`FixedNetwork::try_from`).
    NeuronCountMismatch,
    /// A network whose synapse count is not the [`FixedNetwork`]'s `S`
    /// (`FixedNetwork::try_from`).
    SynapseCountMismatch,
    /// A network whose CSR does not deliver each synapse under its own
    /// `pre`, in the order added (`FixedNetwork::try_from`).
    StaleCsr,
    /// Neurons on more than one voltage grid (`trace::header`).
    MixedVoltageGrids,
    /// A trace case name that is empty or holds anything but lowercase
    /// ASCII letters, digits and `-` (`trace::header`).
    BadCaseName,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::SelfConnection => "a synapse from a neuron to itself",
            Self::NeuronCountOutOfRange => "a neuron count outside 1 to 65,535",
            Self::ZeroTimeStep => "a time step of zero",
            Self::NeuronIdOutOfRange => "a neuron id not below the neuron count",
            Self::DivisorOutOfRange => "a synaptic input divisor outside 1 to 32,767",
            Self::BadLayerSizes => {
                "a layer of no neuron, or layer sizes that do not sum to the neuron count"
            }
            Self::MissingNeuronType => {
                "a balanced network with no excitatory or no inhibitory neuron"
            }
            Self::PlasticityEnabled => "a network with plasticity enabled",
            Self::NeuronCountMismatch => "a neuron count other than the fixed network's N",
            Self::SynapseCountMismatch => "a synapse count other than the fixed network's S",
            Self::StaleCsr => "a CSR that does not deliver the synapses in the order added",
            Self::MixedVoltageGrids => "neurons on more than one voltage grid",
            Self::BadCaseName => "a case name that is empty or outside [a-z0-9-]",
        })
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

// One byte, every variant a unit. The assertion is its pin: a variant
// that carries data grows the type and fails the build here, in every
// configuration.
const _: () = assert!(core::mem::size_of::<Error>() == 1);

/// Crate-wide `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::Error;
    use std::collections::BTreeSet;

    /// Each variant's text: its own, lowercase and with no trailing
    /// punctuation, as the API guidelines ask of an error's `Display`.
    #[test]
    fn each_variant_prints_its_rule() {
        let texts = [
            (Error::SelfConnection, "a synapse from a neuron to itself"),
            (
                Error::NeuronCountOutOfRange,
                "a neuron count outside 1 to 65,535",
            ),
            (Error::ZeroTimeStep, "a time step of zero"),
            (
                Error::NeuronIdOutOfRange,
                "a neuron id not below the neuron count",
            ),
            (
                Error::DivisorOutOfRange,
                "a synaptic input divisor outside 1 to 32,767",
            ),
            (
                Error::BadLayerSizes,
                "a layer of no neuron, or layer sizes that do not sum to the neuron count",
            ),
            (
                Error::MissingNeuronType,
                "a balanced network with no excitatory or no inhibitory neuron",
            ),
            (
                Error::PlasticityEnabled,
                "a network with plasticity enabled",
            ),
            (
                Error::NeuronCountMismatch,
                "a neuron count other than the fixed network's N",
            ),
            (
                Error::SynapseCountMismatch,
                "a synapse count other than the fixed network's S",
            ),
            (
                Error::StaleCsr,
                "a CSR that does not deliver the synapses in the order added",
            ),
            (
                Error::MixedVoltageGrids,
                "neurons on more than one voltage grid",
            ),
            (
                Error::BadCaseName,
                "a case name that is empty or outside [a-z0-9-]",
            ),
        ];
        for (error, text) in texts {
            // No wildcard: a new variant does not compile here until it has
            // its line above.
            match error {
                Error::SelfConnection
                | Error::NeuronCountOutOfRange
                | Error::ZeroTimeStep
                | Error::NeuronIdOutOfRange
                | Error::DivisorOutOfRange
                | Error::BadLayerSizes
                | Error::MissingNeuronType
                | Error::PlasticityEnabled
                | Error::NeuronCountMismatch
                | Error::SynapseCountMismatch
                | Error::StaleCsr
                | Error::MixedVoltageGrids
                | Error::BadCaseName => assert_eq!(error.to_string(), text),
            }
            assert!(text.starts_with(char::is_lowercase), "{text}");
            assert!(!text.ends_with(['.', '!', '?']), "{text}");
        }
        let distinct: BTreeSet<&str> = texts.iter().map(|&(_, text)| text).collect();
        assert_eq!(distinct.len(), texts.len(), "two variants print alike");
    }
}
