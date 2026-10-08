//! `neuralos-trace v1`, the trace format: a network's run as text, a
//! header line then one row per step, which a host and a board write
//! alike and compare byte for byte.
//!
//! ```text
//! # neuralos-trace v1 case=<name> kind=<kind> n=<neurons> dt_us=<dt> res=<mV|cmV> plasticity=<on|off> divisor=<d> steps=<N> rows=<all|spikes>
//! <step> <time_us> | <spiking neuron ids> | <membrane of every neuron, i16 quanta>
//! ```
//!
//! The header line (`header`, with `std`): these fields in this order, one
//! space apart. `case` names the trace: one or more lowercase ASCII
//! letters, digits and `-`. `kind` says where it comes from ([`Kind`], each
//! variant with its word); a reader reads the rows the same whatever the
//! kind. `n` is the number of the network's neurons (every row carries one
//! membrane per neuron), `dt_us` its time step in microseconds,
//! `plasticity` whether it learns, and `divisor` its synaptic input divisor
//! (`SpikingNeuralNetwork::synaptic_input_divisor`). `res` is the grid
//! every neuron stores its potentials on, one for the whole network, which
//! is how a reader reads the membranes: a quantum is 1 mV on `mV`, 0.01 mV
//! on `cmV`. `steps` is the run, and `rows` says which of its steps get a
//! row ([`Rows`]).
//!
//! A row ([`row`]): one per step (`rows=all`), or one per step where a
//! neuron fired (`rows=spikes`). A row is written after its step: `time_us`
//! is the step's own timestamp, the one its spikes carry; the ids are the
//! neurons that fired, ascending, one space apart, and a silent step leaves
//! nothing between the bars but their two spaces (`|  |`); the membranes
//! are every neuron's after the step, so a neuron that fired shows its
//! reset potential. Everything is decimal.
//!
//! A replay that prints several traces over one serial port ends with
//! `# neuralos-trace end`: the ESP32-C3 firmware and the QEMU replay print
//! it, and `tools/esp32c3_trace_diff.py` reads it.
//!
//! The library's own traces are `tests/traces/`, one file per case, each
//! compared line for line with the code by the trace tests.

use core::fmt::{self, Write};

use crate::lif_neuron::LIFNeuron;
#[cfg(feature = "std")]
use crate::lif_neuron::VoltageResolution;
#[cfg(feature = "std")]
use crate::network::SpikingNeuralNetwork;
#[cfg(feature = "std")]
use crate::{Error, Result};

/// The format's name and version, the header line's two words after `#`.
#[cfg(feature = "std")]
const FORMAT: &str = "neuralos-trace v1";

/// Where a trace comes from, the header's `kind=`.
///
/// No reader needs it to read the rows, which read the same whatever the
/// kind, so a reader that meets a kind it does not know reads them as any
/// other. Not exhaustive: a new kind may come in a minor release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Kind {
    /// `kind=regression`: this code's own output, compared with itself
    /// across commits, so a trace that moves is a behavior change.
    Regression,
    /// `kind=reference`: also checked against an oracle that is not this
    /// code; in the library's tests, the unbounded-integer neuron equation.
    Reference,
    /// `kind=stranger`: a stranger's graph, converted and run on the host
    /// by `neuralos-nir2json --freeze`, for a board's run to be compared
    /// with.
    Stranger,
}

/// Which steps of the run get a row, the header's `rows=`.
///
/// Exhaustive on purpose: a reader reads the rows by this value, so a third
/// one would change how a file reads, a new version of the format. A caller
/// may match a `Rows` without a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rows {
    /// Every step, `rows=all`.
    All,
    /// The steps where a neuron fired, `rows=spikes`.
    Spikes,
}

/// The header line of `net`'s trace, without its newline: what the network
/// says of itself, and the run the caller names.
///
/// `case` names the trace, `kind` says where it comes from, `steps` is the
/// run and `rows` which of its steps get a row; `n`, `dt_us`, `res`,
/// `plasticity` and `divisor` are read from `net` (the module doc).
///
/// # Errors
///
/// [`Error::MixedVoltageGrids`] when `net`'s neurons do not all store their
/// potentials on one grid, since `res=` names one for the whole trace;
/// [`Error::BadCaseName`] when `case` is empty or holds anything but
/// lowercase ASCII letters, digits and `-`.
#[cfg(feature = "std")]
pub fn header(
    net: &SpikingNeuralNetwork,
    case: &str,
    kind: Kind,
    steps: u32,
    rows: Rows,
) -> Result<String> {
    let Some(grid) = one_grid(net) else {
        return Err(Error::MixedVoltageGrids);
    };
    if !is_case_name(case) {
        return Err(Error::BadCaseName);
    }
    Ok(format!(
        "# {FORMAT} case={case} kind={} n={} dt_us={} res={} plasticity={} divisor={} steps={steps} rows={}",
        match kind {
            Kind::Regression => "regression",
            Kind::Reference => "reference",
            Kind::Stranger => "stranger",
        },
        net.neurons().len(),
        net.time_step_us(),
        match grid {
            VoltageResolution::Millivolt => "mV",
            VoltageResolution::CentiMillivolt => "cmV",
        },
        if net.plasticity_enabled() { "on" } else { "off" },
        net.synaptic_input_divisor(),
        match rows {
            Rows::All => "all",
            Rows::Spikes => "spikes",
        },
    ))
}

/// The grid every neuron of `net` stores its potentials on, when they all
/// share one; `None` when they do not, or when `net` has no neuron, which
/// no constructor builds.
#[cfg(feature = "std")]
pub(crate) fn one_grid(net: &SpikingNeuralNetwork) -> Option<VoltageResolution> {
    let mut grids = net.neurons().iter().map(|n| n.voltage_resolution);
    let first = grids.next()?;
    grids.all(|g| g == first).then_some(first)
}

/// A trace's case name: non-empty, lowercase ASCII letters, digits and `-`.
#[cfg(feature = "std")]
pub(crate) fn is_case_name(case: &str) -> bool {
    !case.is_empty()
        && case
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// One row of `neuralos-trace v1` (the module doc), its newline included:
/// `<step> <time_us> | <ids of the neurons that fired, ascending> |
/// <membrane of every neuron>`, all decimal; when no neuron fired, nothing
/// stands between the bars but their two spaces. `time_us` is a `u64`: it
/// prints whole, whatever its size. The one writer of a frozen network's
/// rows: the trace tests write them into a `String`, the ESP32-C3 firmware
/// and the QEMU replay to a serial port.
///
/// # Errors
///
/// When the writer refuses a write.
// The body of the trace tests' `row.rs`, moved verbatim: `sep` beside
// `step`.
#[allow(clippy::similar_names)]
pub fn row(
    out: &mut impl Write,
    step: u32,
    time_us: u64,
    fired: &[bool],
    neurons: &[LIFNeuron],
) -> fmt::Result {
    write!(out, "{step} {time_us} | ")?;
    let mut sep = "";
    for (id, &f) in fired.iter().enumerate() {
        if f {
            write!(out, "{sep}{id}")?;
            sep = " ";
        }
    }
    out.write_str(" | ")?;
    let mut sep = "";
    for n in neurons {
        write!(out, "{sep}{}", n.membrane_potential)?;
        sep = " ";
    }
    out.write_char('\n')
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::lif_neuron::NeuronType;

    /// A network of one neuron per grid given, no synapse, a 0.5 ms step:
    /// every trace runs at 1 ms or 0.1 ms, so a step off both shows
    /// `dt_us` is read.
    fn on_grids(grids: &[VoltageResolution]) -> SpikingNeuralNetwork {
        let neurons = (0u16..)
            .zip(grids)
            .map(|(id, &grid)| {
                LIFNeuron::new_with_type_resolution(id, NeuronType::Excitatory, grid)
            })
            .collect();
        SpikingNeuralNetwork::from_neurons(neurons, 500).expect("neurons given")
    }

    /// The time prints whole past the `u32` range.
    #[test]
    fn a_time_past_the_u32_range_prints_whole() {
        let mut out = String::new();
        row(&mut out, 7, 5_000_000_000, &[false], &[LIFNeuron::new(0)]).expect("a String");
        assert!(out.starts_with("7 5000000000 | "), "{out:?}");
        assert!(out.ends_with('\n'), "{out:?}");
    }

    /// Every field of the line, each kind's and each rows' word: the
    /// network's own values, at a divisor no trace uses, and the run the
    /// caller names.
    #[test]
    fn the_header_is_what_the_network_says_and_the_run_it_is_given() {
        let mut net = on_grids(&[VoltageResolution::CentiMillivolt; 2]);
        net.set_synaptic_input_divisor(3).expect("in 1 to 32,767");
        for (kind, word) in [
            (Kind::Regression, "regression"),
            (Kind::Reference, "reference"),
            (Kind::Stranger, "stranger"),
        ] {
            for (rows, rows_word) in [(Rows::All, "all"), (Rows::Spikes, "spikes")] {
                assert_eq!(
                    header(&net, "two-lif-neurons", kind, 150, rows),
                    Ok(format!(
                        "# neuralos-trace v1 case=two-lif-neurons kind={word} n=2 dt_us=500 \
                         res=cmV plasticity=off divisor=3 steps=150 rows={rows_word}"
                    ))
                );
            }
        }
        let mv = on_grids(&[VoltageResolution::Millivolt]);
        assert!(header(&mv, "a", Kind::Regression, 1, Rows::All)
            .is_ok_and(|h| h.contains(" n=1 dt_us=500 res=mV ")));
    }

    /// `res=` names one grid for the whole trace, so a network whose
    /// neurons do not share one is refused rather than misdescribed.
    #[test]
    fn a_network_on_two_grids_is_refused() {
        let mixed = on_grids(&[
            VoltageResolution::Millivolt,
            VoltageResolution::CentiMillivolt,
        ]);
        assert_eq!(
            header(&mixed, "mixed", Kind::Regression, 1, Rows::All),
            Err(Error::MixedVoltageGrids)
        );
        let one = on_grids(&[VoltageResolution::Millivolt; 2]);
        assert!(header(&one, "one", Kind::Regression, 1, Rows::All).is_ok());
    }

    /// `n=` is the length of the neuron list the rows carry: 65,535 neurons
    /// on one grid, the most a network holds.
    #[test]
    fn n_counts_every_neuron_at_the_cap() {
        let neurons = (0..u16::MAX)
            .map(|id| {
                LIFNeuron::new_with_type_resolution(
                    id,
                    NeuronType::Excitatory,
                    VoltageResolution::Millivolt,
                )
            })
            .collect();
        let net = SpikingNeuralNetwork::from_neurons(neurons, 1_000).expect("neurons given");
        assert!(header(&net, "big", Kind::Regression, 1, Rows::All)
            .is_ok_and(|h| h.contains(" n=65535 ")));
    }

    /// The case name's rule: what every name in the tree and every name
    /// `neuralos-nir2json` writes fits, and nothing a reader could split.
    #[test]
    fn a_case_name_is_lowercase_letters_digits_and_dashes() {
        let net = on_grids(&[VoltageResolution::Millivolt]);
        for good in ["a", "0", "-", "chain-3", "snntorch-two-layer"] {
            assert!(
                header(&net, good, Kind::Regression, 1, Rows::All).is_ok(),
                "{good:?}"
            );
        }
        for bad in [
            "",
            "Chain-3",
            "chain_3",
            "chain 3",
            "chain\n3",
            "cha\u{ee}ne",
        ] {
            assert_eq!(
                header(&net, bad, Kind::Regression, 1, Rows::All),
                Err(Error::BadCaseName),
                "{bad:?}"
            );
        }
    }
}
