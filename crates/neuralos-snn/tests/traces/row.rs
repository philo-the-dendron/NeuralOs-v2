//! The one writer of a trace row, over `core::fmt::Write`. A frozen
//! network's rows go through it: on the host into a `String`
//! (`tests/traces.rs`), and since it needs only `core` and `neuralos-snn`,
//! firmware can include it by `#[path]` and write to a serial port. The
//! std network's rows are `cases::render`'s, a second writer: the compare
//! holds each against the files instead of trusting either.

use core::fmt::{self, Write};

use neuralos_snn::LIFNeuron;

/// One row of `neuralos-trace v1` (`cases.rs`), its newline included:
/// `<step> <time_us> | <ids of the neurons that fired, ascending> |
/// <membrane of every neuron>`, all decimal, nothing between the bars
/// when no neuron fired.
///
/// # Errors
///
/// When the writer refuses a write.
pub fn row(
    out: &mut impl Write,
    step: u32,
    time_us: u32,
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
