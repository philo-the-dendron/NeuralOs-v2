//! A network of fixed size: `N` neurons and `S` synapses in arrays, no
//! heap, no plasticity, the std step's order verbatim.
//!
//! [`FixedNetwork::step`] is `SpikingNeuralNetwork::step` without the
//! plasticity passes and the stats, in the same order:
//! the adaptation decay of every neuron; integrate-and-fire, which reads
//! the pulses the previous step delivered; the clear; then the pulses of
//! this step's spikes, in the synapse array's order. The step has no
//! `Result`, no index and no division of its own; the neuron it calls
//! divides only by a constant or a `NonZero`. An id past the arrays is no
//! spike and no target.
//!
//! On the host, `FixedNetwork::try_from(&net)` converts a plasticity-off
//! `SpikingNeuralNetwork` of the same size whose CSR delivers its
//! synapses, and the two step alike, bit for bit: the tests below pin it
//! on the network of `chain-3` (`tests/traces/cases.rs`), and
//! `tests/traces.rs` on every plasticity-off trace.
//!
//! A frozen network's rows go through [`row`], the one writer of a
//! `neuralos-trace v1` row: the trace tests on the host, the ESP32-C3
//! firmware and the QEMU replay. `freeze` (`std`) writes the arrays
//! themselves as Rust source.
//!
//! # `no_std`
//!
//! No allocator, no dependency. `FixedSynapse::from_network`, the
//! conversion and `freeze` read a `SpikingNeuralNetwork` or write a
//! `String`, so they are `std`; [`row`] writes to any `core::fmt::Write`.

use core::fmt::{self, Write};

use crate::lif_neuron::LIFNeuron;
#[cfg(feature = "std")]
use crate::network::SpikingNeuralNetwork;
#[cfg(feature = "std")]
use crate::{Error, Result};

#[cfg(feature = "std")]
pub mod freeze;

/// One synapse of a [`FixedNetwork`]: a spike of `pre` adds `pulse_ua` to
/// the synaptic current of `post`, which `post`'s next step integrates.
///
/// Six bytes, `Copy`. The pulse is the std network's per-spike
/// `weight / synaptic_input_divisor`, computed once, when the synapse is
/// made (`from_network`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedSynapse {
    /// Presynaptic neuron id: its spike sends the pulse.
    pub pre: u16,
    /// Postsynaptic neuron id: its synaptic current takes the pulse.
    pub post: u16,
    /// The pulse, μA.
    pub pulse_ua: i16,
}

#[cfg(feature = "std")]
impl FixedSynapse {
    /// The synapses of `net` by `pre`, ascending, and within one `pre` in
    /// the order they were added: a stable sort of
    /// [`SpikingNeuralNetwork::synapses`] by `pre`, the order the counting
    /// sort of [`SpikingNeuralNetwork::finalize_synapses`] gives the CSR.
    ///
    /// That is the order `net`'s own step delivers them in exactly when
    /// its CSR delivers each synapse under its own `pre`, in the order
    /// added — see `FixedNetwork::try_from` § Errors. On a network
    /// where it does not, this list is the sorted one, the std step
    /// reads the stale CSR, and the two can part. Returning a `Vec`,
    /// this cannot refuse such a network; `FixedNetwork::try_from`
    /// does, and is the way to convert one.
    ///
    /// Each pulse is the std step's own expression, `weight / divisor as
    /// i16`, so a divisor above 32,767 wraps negative here as it does there.
    ///
    /// # Panics
    ///
    /// When the divisor is 65,535 (`-1` as `i16`) and a weight is
    /// `i16::MIN`: the division overflows here, at construction, where the
    /// std step panics when that synapse's `pre` fires.
    #[must_use]
    // `divisor as i16` is the std step's expression, verbatim (doc above).
    #[allow(clippy::cast_possible_wrap)]
    pub fn from_network(net: &SpikingNeuralNetwork) -> Vec<Self> {
        let divisor = net.synaptic_input_divisor() as i16;
        let mut synapses: Vec<Self> = net
            .synapses()
            .iter()
            .map(|s| Self {
                pre: s.pre_neuron_id,
                post: s.post_neuron_id,
                pulse_ua: s.weight / divisor,
            })
            .collect();
        // Stable: synapses of one `pre` keep the order they were added in.
        synapses.sort_by_key(|s| s.pre);
        synapses
    }
}

/// `N` neurons and `S` synapses in arrays: a network with no heap and no
/// plasticity (module doc).
///
/// Built by [`new`](Self::new) from any arrays, or on the host by
/// `FixedNetwork::try_from(&net)` from a plasticity-off
/// `SpikingNeuralNetwork` of the same size whose CSR agrees with its
/// synapse list — exactly what that conversion refuses is its § Errors.
#[derive(Debug, Clone)]
pub struct FixedNetwork<const N: usize, const S: usize> {
    neurons: [LIFNeuron; N],
    synapses: [FixedSynapse; S],
    dt_us: u32,
    time_us: u32,
}

impl<const N: usize, const S: usize> FixedNetwork<N, S> {
    /// A network of these neurons and synapses, stepping `dt_us`, at time
    /// 0.
    ///
    /// Total: any arrays are a network. A synapse whose `pre` or `post` is
    /// not below `N` delivers nothing, and the array order is the delivery
    /// order ([`synapses`](Self::synapses)).
    #[must_use]
    pub const fn new(neurons: [LIFNeuron; N], synapses: [FixedSynapse; S], dt_us: u32) -> Self {
        Self {
            neurons,
            synapses,
            dt_us,
            time_us: 0,
        }
    }

    /// Advance by one `dt_us`, in the std step's order without plasticity:
    /// decay every neuron's adaptation current; integrate each with
    /// `input[i]` at this step's time, writing `fired[i]`; clear every
    /// synaptic current; deliver each synapse whose `pre` fired, in array
    /// order; advance the time, saturating.
    ///
    /// A pulse is read by the next step's integration: the one-step
    /// synaptic delay of the std network. A spike at step t reaches `post`
    /// at step t + 1. A `post` that is refractory at step t + 1 integrates
    /// nothing, and the clear drops the pulse: it is lost, not deferred.
    ///
    /// ```
    /// use neuralos_snn::{FixedNetwork, FixedSynapse, LIFNeuron};
    ///
    /// let quiet = |id| {
    ///     let mut n = LIFNeuron::new(id);
    ///     n.noise_amplitude_ua = 0;
    ///     n
    /// };
    /// let synapse = [FixedSynapse { pre: 0, post: 1, pulse_ua: 400 }];
    /// let mut fired = [false; 2];
    ///
    /// // `pre` fires at step 0 (3,000 μA: +15 mV, onto the threshold).
    /// let mut net = FixedNetwork::new([quiet(0), quiet(1)], synapse, 1_000);
    /// net.step(&[3_000, 0], &mut fired);
    /// assert_eq!(fired, [true, false]);
    /// assert_eq!(net.neurons()[1].membrane_potential, -70); // not yet
    /// net.step(&[0, 0], &mut fired);
    /// assert_eq!(net.neurons()[1].membrane_potential, -68); // one step later
    ///
    /// // The same, `post` refractory through step 1.
    /// let mut post = quiet(1);
    /// post.refractory_time_us = 2_000;
    /// let mut net = FixedNetwork::new([quiet(0), post], synapse, 1_000);
    /// net.step(&[3_000, 0], &mut fired);
    /// assert_eq!(net.neurons()[1].synaptic_current_ua, 400); // delivered
    /// net.step(&[0, 0], &mut fired); // not read, then cleared
    /// assert_eq!(net.neurons()[1].synaptic_current_ua, 0);
    /// assert_eq!(net.neurons()[1].refractory_time_us, 0);
    /// net.step(&[0, 0], &mut fired); // integrates, and nothing is left
    /// assert_eq!(net.neurons()[1].membrane_potential, -70);
    /// ```
    pub fn step(&mut self, input: &[i16; N], fired: &mut [bool; N]) {
        for n in &mut self.neurons {
            n.decay_adaptation_current();
        }
        for ((n, &current_ua), spiked) in self.neurons.iter_mut().zip(input).zip(fired.iter_mut()) {
            *spiked = n.integrate_and_fire(current_ua, self.dt_us, self.time_us);
        }
        for n in &mut self.neurons {
            n.clear_synaptic_current();
        }
        for s in &self.synapses {
            if fired.get(usize::from(s.pre)) == Some(&true) {
                if let Some(post) = self.neurons.get_mut(usize::from(s.post)) {
                    post.add_synaptic_current(s.pulse_ua);
                }
            }
        }
        self.time_us = self.time_us.saturating_add(self.dt_us);
    }

    /// The neurons, by id.
    #[must_use]
    pub const fn neurons(&self) -> &[LIFNeuron; N] {
        &self.neurons
    }

    /// The synapses, in delivery order.
    ///
    /// From `FixedSynapse::from_network` this is the CSR order: by `pre`,
    /// ascending, stable within one `pre`. It must stay so for this network
    /// and the std one to agree: the std step visits its firing neurons in
    /// ascending id order, and saturating adds are order-dependent (from 0,
    /// +30,000 twice then −30,000 ends at 2,767; −30,000 first ends at
    /// 30,000). The step itself delivers in whatever order the array holds.
    #[must_use]
    pub const fn synapses(&self) -> &[FixedSynapse; S] {
        &self.synapses
    }

    /// The simulation step, μs.
    #[must_use]
    pub const fn dt_us(&self) -> u32 {
        self.dt_us
    }

    /// The time of the next step, μs: the timestamp its integration and its
    /// spikes carry. Starts at 0 from [`new`](Self::new), at the network's
    /// current time from `try_from`.
    #[must_use]
    pub const fn time_us(&self) -> u32 {
        self.time_us
    }
}

#[cfg(feature = "std")]
impl<const N: usize, const S: usize> TryFrom<&SpikingNeuralNetwork> for FixedNetwork<N, S> {
    type Error = Error;

    /// `net` as it stands: its neurons cloned into the array, its synapses
    /// by [`FixedSynapse::from_network`], its time step and its current
    /// time.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] when `net` has plasticity enabled (a
    /// fixed network has none); when it has not exactly `N` neurons and
    /// `S` synapses; or when its CSR does not deliver each synapse under
    /// its own `pre`, in the order added. A caller gets there, for
    /// example, by adding edges out of `pre` order with no
    /// `finalize_synapses` after them, or by finalizing such edges a
    /// second time (`finalize_synapses` is not idempotent, and
    /// `build_topology` already finalizes). That network steps through a
    /// stale CSR, delivering a pulse under another synapse's edge, and
    /// this one, always sorted, can part from it in silence.
    ///
    /// A network starts with plasticity off, and only a build with the
    /// `unstable-stdp` feature can turn it on: without the feature the
    /// first refusal is never met.
    fn try_from(net: &SpikingNeuralNetwork) -> Result<Self> {
        if net.plasticity_enabled()
            || usize::from(net.neuron_count()) != N
            || usize::try_from(net.synapse_count()) != Ok(S)
            || !net.csr_delivers_its_synapses()
        {
            return Err(Error::InvalidParameter);
        }
        let neurons: [LIFNeuron; N] = net
            .neurons()
            .to_vec()
            .try_into()
            .map_err(|_| Error::InvalidParameter)?;
        let synapses: [FixedSynapse; S] = FixedSynapse::from_network(net)
            .try_into()
            .map_err(|_| Error::InvalidParameter)?;
        Ok(Self {
            neurons,
            synapses,
            dt_us: net.time_step_us(),
            time_us: net.current_time_us(),
        })
    }
}

/// One row of `neuralos-trace v1` (the library's `tests/traces/cases.rs`),
/// its newline included: `<step> <time_us> | <ids of the neurons that
/// fired, ascending> | <membrane of every neuron>`, all decimal, nothing
/// between the bars when no neuron fired. The one writer of a frozen
/// network's rows: the trace tests write them into a `String`, the
/// ESP32-C3 firmware and the QEMU replay to a serial port.
///
/// Provisional until the pub walk.
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

// After the last `impl` of this module, and not before one: a v0 symbol
// carries its `impl` block's index in its module, so a block inserted
// earlier renumbers the blocks after it and relays the firmware's
// `.text` with no code change (measured, ISA round 43).
impl FixedSynapse {
    /// A synapse from its three values: a spike of `pre` adds `pulse_ua`
    /// to the synaptic current of `post`.
    ///
    /// `const`, and not `std`: a frozen network's synapse array is a
    /// `const` of these calls, written by `freeze::module` and
    /// compiled outside this crate — by the firmware, which has no
    /// `std`. The call is positional with the two ids side by side, so
    /// which is which is pinned by a test, not by the types: the frozen
    /// cases' synapse line in `tests/traces.rs`.
    ///
    /// ```
    /// use neuralos_snn::FixedSynapse;
    ///
    /// let s = FixedSynapse::new(0, 1, 400);
    /// assert_eq!((s.pre, s.post, s.pulse_ua), (0, 1, 400));
    /// ```
    #[must_use]
    pub const fn new(pre: u16, post: u16, pulse_ua: i16) -> Self {
        Self {
            pre,
            post,
            pulse_ua,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use crate::csr::SparseSynapseMatrix;
    #[cfg(feature = "std")]
    use crate::lif_neuron::{NeuronType, VoltageResolution};

    /// An excitatory neuron on the mV grid, no noise.
    fn quiet(id: u16) -> LIFNeuron {
        let mut n = LIFNeuron::new(id);
        n.noise_amplitude_ua = 0;
        n
    }

    /// Every field a step writes.
    #[cfg(feature = "std")]
    fn state(n: &LIFNeuron) -> (i16, u32, u32, u32, i16, i16) {
        (
            n.membrane_potential,
            n.refractory_time_us,
            n.last_update_time_us,
            n.last_spike_time_us,
            n.synaptic_current_ua,
            n.adaptation_current_ua,
        )
    }

    #[test]
    fn a_synapse_is_six_bytes() {
        assert_eq!(core::mem::size_of::<FixedSynapse>(), 6);
    }

    /// Saturating adds do not commute, so the array order is the delivery
    /// order, sorted or not. Neurons 1 and 2 fire at step 0 on 3,000 μA
    /// (+15 mV, from rest to the −55 mV threshold) and pulse neuron 0,
    /// which is quiet: −30,000, then +30,000 twice, ends at 30,000. By
    /// `pre`, the two +30,000 would come first, saturate at 32,767, and end
    /// at 2,767.
    #[test]
    fn an_unsorted_array_delivers_in_array_order() {
        let mut net = FixedNetwork::new(
            [quiet(0), quiet(1), quiet(2)],
            [
                FixedSynapse {
                    pre: 2,
                    post: 0,
                    pulse_ua: -30_000,
                },
                FixedSynapse {
                    pre: 1,
                    post: 0,
                    pulse_ua: 30_000,
                },
                FixedSynapse {
                    pre: 1,
                    post: 0,
                    pulse_ua: 30_000,
                },
            ],
            1_000,
        );
        let mut fired = [false; 3];
        net.step(&[0, 3_000, 3_000], &mut fired);
        assert_eq!(fired, [false, true, true]);
        assert_eq!(net.neurons()[0].synaptic_current_ua, 30_000);
    }

    /// Total: an id past the arrays is no spike and no target, and the step
    /// goes on.
    #[test]
    fn an_id_past_the_arrays_is_no_spike_and_no_target() {
        let mut net = FixedNetwork::new(
            [quiet(0), quiet(1)],
            [
                FixedSynapse {
                    pre: 0,
                    post: 9,
                    pulse_ua: 500,
                },
                FixedSynapse {
                    pre: 7,
                    post: 1,
                    pulse_ua: 500,
                },
            ],
            1_000,
        );
        let mut fired = [false; 2];
        net.step(&[3_000, 0], &mut fired);
        assert_eq!(fired, [true, false]);
        assert_eq!(net.neurons()[1].synaptic_current_ua, 0);
        assert_eq!(net.time_us(), 1_000);
    }

    /// The network of `chain-3` (`tests/traces/cases.rs`) without its
    /// edges: three excitatory neurons, centi-mV, no noise, divisor 1.
    #[cfg(feature = "std")]
    fn chain_3() -> SpikingNeuralNetwork {
        let neurons = (0..3)
            .map(|id| {
                let mut n = LIFNeuron::new_with_type_resolution(
                    id,
                    NeuronType::Excitatory,
                    VoltageResolution::CentiMillivolt,
                );
                n.noise_amplitude_ua = 0;
                n
            })
            .collect();
        let mut net =
            SpikingNeuralNetwork::from_neurons(neurons, 1_000).expect("three neurons, 1 ms");
        net.set_synaptic_input_divisor(1).expect("nonzero");
        net
    }

    /// The two networks side by side for 100 steps, neuron 0 driven at 600
    /// μA: every spike, every field a step writes, and the time, at each
    /// step. Returns how many times each neuron fired.
    #[cfg(feature = "std")]
    #[must_use]
    fn steps_alike(net: &mut SpikingNeuralNetwork, fixed: &mut FixedNetwork<3, 2>) -> [u32; 3] {
        let mut spiked = [false; 3];
        let mut counts = [0u32; 3];
        for step in 0..100 {
            let want: Vec<u16> = net
                .step(&[600, 0, 0])
                .expect("a built network steps")
                .iter()
                .map(|s| s.neuron_id)
                .collect();
            fixed.step(&[600, 0, 0], &mut spiked);
            let got: Vec<u16> = (0..3u16).filter(|&id| spiked[usize::from(id)]).collect();
            assert_eq!(got, want, "step {step}: the spikes");
            for (f, s) in fixed.neurons().iter().zip(net.neurons()) {
                assert_eq!(state(f), state(s), "step {step}: neuron {}", s.id);
            }
            assert_eq!(
                fixed.time_us(),
                net.current_time_us(),
                "step {step}: the time"
            );
            for &id in &got {
                counts[usize::from(id)] += 1;
            }
        }
        counts
    }

    /// The fixed step against the std step on the network of `chain-3`
    /// (`tests/traces/cases.rs`): three excitatory neurons in a line,
    /// centi-mV, no noise, 5,000-weight edges at divisor 1, neuron 0 driven
    /// at 600 μA. Every spike and every field a step writes, at each of the
    /// 100 steps, and all three neurons fire.
    #[cfg(feature = "std")]
    #[test]
    fn the_step_is_the_std_step_on_chain_3() {
        let mut net = chain_3();
        net.add_synapse(0, 1, 5_000).expect("ids in range");
        net.add_synapse(1, 2, 5_000).expect("ids in range");
        net.finalize_synapses();
        let mut fixed =
            FixedNetwork::<3, 2>::try_from(&net).expect("plasticity off, 3 neurons, 2 synapses");

        let counts = steps_alike(&mut net, &mut fixed);
        assert!(
            counts.iter().all(|&n| n > 0),
            "all three fire, 1 and 2 through the chain: {counts:?}"
        );
    }

    /// `try_from` refuses a network whose CSR does not deliver its
    /// synapses. One way to lose that property: the same chain-3, its two
    /// edges added out of `pre` order and never finalized. Its std step
    /// then reads slot 0 for neuron 0, which holds the 1→2 edge, and this
    /// pair does part — the fields at step 5, the spike lists at step 6.
    /// One `finalize_synapses` and the two are the same network again,
    /// for 100 steps.
    #[cfg(feature = "std")]
    #[test]
    fn try_from_refuses_a_stale_csr() {
        let mut net = chain_3();
        net.add_synapse(1, 2, 5_000).expect("ids in range");
        net.add_synapse(0, 1, 5_000).expect("ids in range");

        assert_eq!(
            FixedNetwork::<3, 2>::try_from(&net).err(),
            Some(Error::InvalidParameter),
            "the edges were added out of pre order and never finalized"
        );

        net.finalize_synapses();
        let mut fixed =
            FixedNetwork::<3, 2>::try_from(&net).expect("finalized: 3 neurons, 2 synapses");
        let counts = steps_alike(&mut net, &mut fixed);
        assert!(
            counts.iter().all(|&n| n > 0),
            "all three fire, 1 and 2 through the chain: {counts:?}"
        );
    }

    /// The order clause of `csr_delivers_its_synapses`, the half of that
    /// method no test reached until now (round 42 left it open): a CSR
    /// that holds every synapse under its own `pre` and one `pre`'s two
    /// slots in the wrong ORDER.
    ///
    /// The witness, on four default neurons: `add(3→2, 100)`,
    /// `add(2→3, 200)`, finalize, `add(2→3, 300)`, finalize, finalize.
    /// `finalize` is not idempotent, and the last two walk a list whose
    /// order they no longer match, so pre 2 ends up holding its own two
    /// synapses newest first. The mirror below is what makes that a
    /// measured fact of this test and not a claim of its comment: every
    /// slot sits under its own `pre`, 3 of 3, so every other clause of
    /// the method passes and the order clause alone refuses the network
    /// — and `FixedNetwork::try_from`, which reads the method, refuses
    /// it with them.
    #[cfg(feature = "std")]
    #[test]
    fn slots_out_of_order_under_one_pre_are_refused() {
        let edges: [(u16, u16, i16); 3] = [(3, 2, 100), (2, 3, 200), (2, 3, 300)];
        let mut net =
            SpikingNeuralNetwork::from_neurons((0..4).map(LIFNeuron::new).collect(), 1_000)
                .expect("four neurons, 1 ms");
        net.add_synapse(3, 2, 100).expect("ids in range");
        net.add_synapse(2, 3, 200).expect("ids in range");
        net.finalize_synapses();
        net.add_synapse(2, 3, 300).expect("ids in range");
        net.finalize_synapses();
        net.finalize_synapses();

        // The same calls on a bare matrix — `add_synapse` is `add` with
        // the running synapse index, `finalize_synapses` is `finalize` —
        // so the slots this witness produces are read, not assumed.
        let mut csr = SparseSynapseMatrix::new(4, edges.len());
        for (i, &(pre, post, weight)) in edges.iter().enumerate().take(2) {
            csr.add(pre, post, weight, i);
        }
        csr.finalize();
        csr.add(2, 3, 300, 2);
        csr.finalize();
        csr.finalize();
        let slots: Vec<Vec<(u16, i16, usize)>> =
            (0..4).map(|pre| csr.connections(pre).collect()).collect();
        assert_eq!(
            slots,
            vec![
                vec![],
                vec![],
                vec![(3, 300, 2), (3, 200, 1)],
                vec![(2, 100, 0)],
            ],
            "every slot under its own pre, 3 of 3, and pre 2's two descending"
        );

        assert!(
            !net.csr_delivers_its_synapses(),
            "pre 2 delivers its own synapses, in the reverse of the order they were added"
        );
        assert_eq!(
            FixedNetwork::<4, 3>::try_from(&net).err(),
            Some(Error::InvalidParameter),
            "and try_from refuses the network for it"
        );
    }

    /// `from_network`'s order is the CSR's. Edges added out of `pre` order,
    /// two sharing pre 2 and two sharing pre 0, pre 0's added out of `post`
    /// order (0→2 before 0→1), and a divisor that truncates (3): a
    /// `SparseSynapseMatrix` built from the same edges and finalized lists,
    /// pre by pre, the same synapses in the same order. Within one `pre`
    /// that is the order they were added, not the `post` order, so a sort
    /// by `(pre, post)` fails here.
    #[cfg(feature = "std")]
    #[test]
    fn from_network_is_the_csr_order() {
        let edges: [(u16, u16, i16); 5] = [
            (2, 0, -125),
            (0, 2, 7),
            (2, 3, -250),
            (1, 3, 40),
            (0, 1, 125),
        ];
        let mut net =
            SpikingNeuralNetwork::from_neurons((0..4).map(LIFNeuron::new).collect(), 1_000)
                .expect("four neurons, 1 ms");
        for &(pre, post, weight) in &edges {
            net.add_synapse(pre, post, weight)
                .expect("ids in range, no self edge");
        }
        net.finalize_synapses();
        net.set_synaptic_input_divisor(3).expect("nonzero");
        assert!(
            !net.synapses()
                .windows(2)
                .all(|w| w[0].pre_neuron_id <= w[1].pre_neuron_id),
            "the edges were added out of pre order, or the test proves nothing"
        );

        let mut csr = SparseSynapseMatrix::new(4, edges.len());
        for (i, &(pre, post, weight)) in edges.iter().enumerate() {
            csr.add(pre, post, weight, i);
        }
        csr.finalize();
        let want: Vec<FixedSynapse> = (0..4)
            .flat_map(|pre| {
                csr.connections(pre)
                    .map(move |(post, weight, _)| FixedSynapse {
                        pre,
                        post,
                        pulse_ua: weight / 3,
                    })
            })
            .collect();
        assert_eq!(FixedSynapse::from_network(&net), want);
        let pre_0: Vec<u16> = FixedSynapse::from_network(&net)
            .iter()
            .filter(|s| s.pre == 0)
            .map(|s| s.post)
            .collect();
        assert_eq!(
            pre_0,
            [2, 1],
            "pre 0's synapses in the order added, not by post"
        );
    }

    /// `try_from` refuses what a fixed network cannot hold, and carries the
    /// time step and the current time of what it can.
    #[cfg(feature = "std")]
    #[test]
    fn try_from_refuses_plasticity_and_other_sizes() {
        let mut net =
            SpikingNeuralNetwork::from_neurons((0..3).map(LIFNeuron::new).collect(), 1_000)
                .expect("three neurons, 1 ms");
        net.add_synapse(0, 1, 100).expect("ids in range");
        net.add_synapse(1, 2, 100).expect("ids in range");
        net.finalize_synapses();
        #[cfg(feature = "unstable-stdp")]
        {
            net.set_plasticity_enabled(true);
            assert_eq!(
                FixedNetwork::<3, 2>::try_from(&net).err(),
                Some(Error::InvalidParameter),
                "plasticity on"
            );
            net.set_plasticity_enabled(false);
        }
        assert_eq!(
            FixedNetwork::<4, 2>::try_from(&net).err(),
            Some(Error::InvalidParameter),
            "four neurons"
        );
        assert_eq!(
            FixedNetwork::<3, 1>::try_from(&net).err(),
            Some(Error::InvalidParameter),
            "one synapse"
        );
        for _ in 0..5 {
            net.step(&[0, 0, 0]).expect("a built network steps");
        }
        let fixed = FixedNetwork::<3, 2>::try_from(&net).expect("plasticity off, 3 and 2");
        assert_eq!((fixed.dt_us(), fixed.time_us()), (1_000, 5_000));
    }
}
