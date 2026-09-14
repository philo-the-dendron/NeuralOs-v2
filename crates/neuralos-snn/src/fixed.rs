//! A network of fixed size: `N` neurons and `S` synapses in arrays, no
//! heap, no plasticity, the std step's order verbatim.
//!
//! [`FixedNetwork::step`] is `SpikingNeuralNetwork::step` without the
//! plasticity passes, the stats and the spike history, in the same order:
//! the adaptation decay of every neuron; integrate-and-fire, which reads
//! the pulses the previous step delivered; the clear; then the pulses of
//! this step's spikes, in the synapse array's order. The step has no
//! `Result`, no index and no division of its own; the neuron it calls
//! divides only by a constant or a `NonZero`. An id past the arrays is no
//! spike and no target.
//!
//! On the host, `FixedNetwork::try_from(&net)` converts a plasticity-off,
//! finalized `SpikingNeuralNetwork` of the same size, and the two step
//! alike, bit for bit: the tests below pin it on the network of `chain-3`
//! (`tests/traces/cases.rs`), and `tests/traces.rs` on every
//! plasticity-off trace.
//!
//! # `no_std`
//!
//! No allocator, no dependency. `FixedSynapse::from_network` and the
//! conversion read a `SpikingNeuralNetwork`, so they are `std`, as it is.

use crate::lif_neuron::LIFNeuron;
#[cfg(feature = "std")]
use crate::network::SpikingNeuralNetwork;
#[cfg(feature = "std")]
use crate::{Error, Result};

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
    /// The synapses of `net` in the order its step delivers them: by `pre`,
    /// ascending, and within one `pre` in the order they were added. That
    /// is a stable sort of [`SpikingNeuralNetwork::synapses`] by `pre`, the
    /// order the counting sort of
    /// [`SparseSynapseMatrix::finalize`](crate::network::SparseSynapseMatrix::finalize)
    /// gives the CSR.
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
/// `SpikingNeuralNetwork` of the same size.
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
    /// synaptic delay of the std network.
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
    /// Assumes a finalized network (`finalize_synapses`, or
    /// `build_topology`), as `net`'s own step does: on synapses added out
    /// of `pre` order and never finalized, the std step delivers through a
    /// stale CSR and the two networks part.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] when `net` has plasticity enabled (a
    /// fixed network has none), or not exactly `N` neurons and `S`
    /// synapses.
    fn try_from(net: &SpikingNeuralNetwork) -> Result<Self> {
        if net.plasticity_enabled()
            || usize::from(net.neuron_count()) != N
            || usize::try_from(net.synapse_count()) != Ok(S)
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use crate::lif_neuron::{NeuronType, VoltageResolution};
    #[cfg(feature = "std")]
    use crate::network::SparseSynapseMatrix;

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

    /// The fixed step against the std step on the network of `chain-3`
    /// (`tests/traces/cases.rs`): three excitatory neurons in a line,
    /// centi-mV, no noise, 5,000-weight edges at divisor 1, neuron 0 driven
    /// at 600 μA. Every spike and every field a step writes, at each of the
    /// 100 steps, and all three neurons fire.
    #[cfg(feature = "std")]
    #[test]
    fn the_step_is_the_std_step_on_chain_3() {
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
        net.add_synapse(0, 1, 5_000).expect("ids in range");
        net.add_synapse(1, 2, 5_000).expect("ids in range");
        net.finalize_synapses();
        net.set_plasticity_enabled(false);
        net.set_synaptic_input_divisor(1).expect("nonzero");
        let mut fixed =
            FixedNetwork::<3, 2>::try_from(&net).expect("plasticity off, 3 neurons, 2 synapses");

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
        assert!(
            counts.iter().all(|&n| n > 0),
            "all three fire, 1 and 2 through the chain: {counts:?}"
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
        assert_eq!(
            FixedNetwork::<3, 2>::try_from(&net).err(),
            Some(Error::InvalidParameter),
            "plasticity on, the default"
        );
        net.set_plasticity_enabled(false);
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
