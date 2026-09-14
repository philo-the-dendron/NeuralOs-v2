//! The link-time proof that `FixedNetwork::step` has no panic path
//! (build.sh runs it, under two profiles).
//!
//! `_start` steps a `FixedNetwork<8, 6>`, the shape of `feedforward-8` in
//! `crates/neuralos-snn/tests/traces/cases.rs` (layers 3-3-2: six
//! excitatory neurons and two inhibitory, six synapses), forever. The
//! network and the input go through `black_box` at every step, so the
//! optimizer may assume nothing about the neurons, the synapses, the time
//! or the drive: what it compiles is the step for any state of that shape.
//! The panic handler (`bare.rs`) is an undefined reference, so the link is
//! green only if no panic is reachable from that step. Linked, never run.

#![no_std]
#![no_main]

mod bare;

use core::hint::black_box;
use neuralos_snn::{FixedNetwork, FixedSynapse, LIFNeuron, NeuronType, VoltageResolution};

/// `feedforward-8`'s synapses: layer 0 (ids 0 to 2) to layer 1 (3 to 5),
/// then layer 1 to layer 2 (6 and 7), weight 100 at divisor 1.
const SYNAPSES: [FixedSynapse; 6] = [
    FixedSynapse {
        pre: 0,
        post: 3,
        pulse_ua: 100,
    },
    FixedSynapse {
        pre: 1,
        post: 4,
        pulse_ua: 100,
    },
    FixedSynapse {
        pre: 2,
        post: 5,
        pulse_ua: 100,
    },
    FixedSynapse {
        pre: 3,
        post: 6,
        pulse_ua: 100,
    },
    FixedSynapse {
        pre: 4,
        post: 7,
        pulse_ua: 100,
    },
    FixedSynapse {
        pre: 5,
        post: 6,
        pulse_ua: 100,
    },
];

fn neuron(id: u16, kind: NeuronType) -> LIFNeuron {
    LIFNeuron::new_with_type_resolution(id, kind, VoltageResolution::CentiMillivolt)
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let (e, i) = (NeuronType::Excitatory, NeuronType::Inhibitory);
    let mut net = FixedNetwork::new(
        [
            neuron(0, e),
            neuron(1, e),
            neuron(2, e),
            neuron(3, e),
            neuron(4, e),
            neuron(5, e),
            neuron(6, i),
            neuron(7, i),
        ],
        SYNAPSES,
        1_000,
    );
    let input: [i16; 8] = [600, 600, 600, 148, 148, 148, 199, 199];
    let mut fired = [false; 8];
    loop {
        black_box(&mut net).step(black_box(&input), &mut fired);
        black_box(&mut fired);
    }
}
