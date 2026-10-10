//! The constructors' rules, case by case: each case is run here against
//! the rule the docs state, written in this file apart from the library.
//! A case a rule refuses is refused under that rule's name; every other
//! case builds, and a topology case's network runs 25 steps; no case
//! panics. When a case breaks two rules either name will do: the library
//! promises no order between them.
//!
//! The topology cases at 65,535 neurons that build take over a minute in
//! a debug build (the CSR's insert walks every later row), so a debug
//! build skips their test and a release build runs it. Part of the test
//! gate (`cargo test --workspace`); alone,
//! `cargo test -p neuralos-snn --test constructors`.

use neuralos_snn::trace::{self, Kind, Rows};
use neuralos_snn::{
    Error, FixedNetwork, FixedSynapse, LIFNeuron, NetworkTopology, NeuronType,
    SpikingNeuralNetwork, VoltageResolution,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// What the rules say of a case.
#[derive(Debug)]
enum Rule {
    Builds,
    /// Refused under one of these names, two when a case breaks two rules.
    Refused(&'static [Error]),
}

/// The cases run, and the ones whose outcome is not their rule's.
#[derive(Default)]
struct Cases {
    run: usize,
    wrong: Vec<String>,
}

impl Cases {
    fn check(&mut self, id: &str, rule: Rule, case: impl FnOnce() -> Result<(), Error>) {
        self.run += 1;
        let got = catch_unwind(AssertUnwindSafe(case));
        let kept = match (&rule, &got) {
            (Rule::Builds, Ok(Ok(()))) => true,
            (Rule::Refused(names), Ok(Err(e))) => names.contains(e),
            _ => false,
        };
        if !kept {
            let got = match got {
                Ok(Ok(())) => "built".to_string(),
                Ok(Err(e)) => format!("refused {e:?}"),
                Err(_) => "panicked".to_string(),
            };
            self.wrong
                .push(format!("{id}: the rule says {rule:?}, got {got}"));
        }
    }

    fn assert_kept(self, cases: usize) {
        assert!(
            self.wrong.is_empty(),
            "{} of {} cases break their rule:\n{}",
            self.wrong.len(),
            self.run,
            self.wrong.join("\n")
        );
        assert_eq!(self.run, cases, "the cases run");
    }
}

/// 25 steps, a quarter of the neurons driven at 3,000 μA each step, chosen
/// by a fixed generator.
fn drive(net: &mut SpikingNeuralNetwork) -> Result<(), Error> {
    let mut lcg: u32 = 12_345;
    let mut input = vec![0i16; usize::from(net.neuron_count())];
    for _ in 0..25 {
        for current in &mut input {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *current = if (lcg >> 24).is_multiple_of(4) {
                3000
            } else {
                0
            };
        }
        net.step(&input)?;
    }
    Ok(())
}

fn topology_case(n: u16, dt: u32, topology: NetworkTopology) -> Result<(), Error> {
    let mut net = SpikingNeuralNetwork::new(n, dt, topology)?;
    net.build_topology()?;
    drive(&mut net)
}

/// `new`'s rules, then the topology's own (`build_topology`).
fn topology_rule(n: u16, dt: u32, topology: NetworkTopology) -> Rule {
    if n == 0 && dt == 0 {
        return Rule::Refused(&[Error::NeuronCountOutOfRange, Error::ZeroTimeStep]);
    }
    if n == 0 {
        return Rule::Refused(&[Error::NeuronCountOutOfRange]);
    }
    if dt == 0 {
        return Rule::Refused(&[Error::ZeroTimeStep]);
    }
    match topology {
        // A probability from 0 to 1, NaN refused, at any neuron count.
        NetworkTopology::Random { connectivity: p }
        | NetworkTopology::SmallWorld {
            rewiring_prob: p, ..
        } => {
            if (0.0..=1.0).contains(&p) {
                Rule::Builds
            } else {
                Rule::Refused(&[Error::ProbabilityOutOfRange])
            }
        }
        // Every layer has a neuron, and the sizes sum to the count.
        NetworkTopology::Feedforward { layers } => {
            let sum: usize = layers.iter().map(|&size| usize::from(size)).sum();
            if layers.contains(&0) || sum != usize::from(n) {
                Rule::Refused(&[Error::BadLayerSizes])
            } else {
                Rule::Builds
            }
        }
        // The count times the ratio, truncated, is the excitatory count,
        // and a network needs a neuron of each type.
        NetworkTopology::Balanced { excitatory_ratio } => {
            let excitatory = (f64::from(n) * excitatory_ratio).trunc();
            if excitatory >= 1.0 && excitatory <= f64::from(n) - 1.0 {
                Rule::Builds
            } else {
                Rule::Refused(&[Error::MissingNeuronType])
            }
        }
        other => panic!("no rule written here for {other:?}"),
    }
}

fn topology(cases: &mut Cases, id: &str, n: u16, dt: u32, topology: NetworkTopology) {
    cases.check(id, topology_rule(n, dt, topology), || {
        topology_case(n, dt, topology)
    });
}

/// `layers` for a `Feedforward` case, kept for the test's run.
fn leak(layers: &[u16]) -> &'static [u16] {
    Box::leak(layers.to_vec().into_boxed_slice())
}

/// `new` and `build_topology` at 1 to 30 neurons, every topology, the
/// values at and just past each rule's edges, -0.0 and the infinities;
/// and every `Feedforward` list of 0 to 4 layers over the sizes 0, 1, 2
/// and 5, the empty list included, so a layer of no neuron sits at every
/// place, on its sum and one off it each way.
fn section_a(cases: &mut Cases) {
    for n in [1, 2, 3, 4, 5, 8, 10, 30] {
        topology(
            cases,
            &format!("A/default/n={n}"),
            n,
            1000,
            NetworkTopology::default(),
        );
        for r in [
            -1.0,
            0.0,
            0.01,
            0.05,
            0.2,
            0.5,
            0.8,
            0.9,
            0.95,
            0.99,
            1.0_f64.next_down(),
            1.0,
            1.5,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let balanced = NetworkTopology::Balanced {
                excitatory_ratio: r,
            };
            topology(cases, &format!("A/bal/n={n}/r={r:?}"), n, 1000, balanced);
        }
        // The lower edge of the ratio: 1/n and the value below it.
        let lower = 1.0 / f64::from(n);
        for r in [lower.next_down(), lower] {
            let balanced = NetworkTopology::Balanced {
                excitatory_ratio: r,
            };
            topology(
                cases,
                &format!("A/bal-edge/n={n}/r={r:?}"),
                n,
                1000,
                balanced,
            );
        }
        // The probability rule at and just past each edge, -0.0, far
        // values and the infinities.
        let (below, above) = (0.0_f64.next_down(), 1.0_f64.next_up());
        for c in [
            -1.0,
            below,
            -0.0,
            0.0,
            0.1,
            0.5,
            1.0,
            above,
            2.0,
            1e30,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let random = NetworkTopology::Random { connectivity: c };
            topology(cases, &format!("A/rand/n={n}/c={c:?}"), n, 1000, random);
        }
        for k in [0u8, 1, 2, 3, 4, 10, 255] {
            for p in [
                -1.0,
                below,
                -0.0,
                0.0,
                0.5,
                1.0,
                above,
                5.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ] {
                let small_world = NetworkTopology::SmallWorld {
                    local_connections: k,
                    rewiring_prob: p,
                };
                topology(
                    cases,
                    &format!("A/sw/n={n}/k={k}/p={p:?}"),
                    n,
                    1000,
                    small_world,
                );
            }
        }
    }
    for dt in [0, 1, 1000, u32::MAX] {
        topology(
            cases,
            &format!("A/dt={dt}"),
            10,
            dt,
            NetworkTopology::default(),
        );
    }
    topology(cases, "A/n=0", 0, 1000, NetworkTopology::default());
    topology(cases, "A/n=0/dt=0", 0, 0, NetworkTopology::default());
    let sizes = [0u16, 1, 2, 5];
    for len in 0..=4u32 {
        for code in 0..sizes.len().pow(len) {
            let mut c = code;
            let mut list = Vec::new();
            for _ in 0..len {
                list.push(sizes[c % sizes.len()]);
                c /= sizes.len();
            }
            let sum: u16 = list.iter().sum();
            let layers = leak(&list);
            for n in [sum.checked_sub(1), Some(sum), Some(sum + 1)] {
                if let Some(n @ 1..) = n {
                    let feedforward = NetworkTopology::Feedforward { layers };
                    topology(cases, &format!("A/ff/{list:?}/n={n}"), n, 1000, feedforward);
                }
            }
        }
    }
}

/// `from_neurons` at 0 to 3 neurons and around 65,535, with and without a
/// time step, then every order of 1 to 3 neurons over the two grids.
fn section_b(cases: &mut Cases) {
    for n in [0usize, 1, 2, 3, 65_534, 65_535, 65_536, 65_537] {
        for dt in [0u32, 1000] {
            let rule = match (n == 0 || n > 65_535, dt == 0) {
                (true, true) => Rule::Refused(&[Error::NeuronCountOutOfRange, Error::ZeroTimeStep]),
                (true, false) => Rule::Refused(&[Error::NeuronCountOutOfRange]),
                (false, true) => Rule::Refused(&[Error::ZeroTimeStep]),
                (false, false) => Rule::Builds,
            };
            cases.check(&format!("B/from/n={n}/dt={dt}"), rule, || {
                let neurons = (0..n).map(|id| LIFNeuron::new(id as u16)).collect();
                let mut net = SpikingNeuralNetwork::from_neurons(neurons, dt)?;
                let mut input = vec![0i16; n];
                input[n - 1] = 3000;
                net.step(&input).map(|_| ())
            });
        }
    }
    let grids = [
        VoltageResolution::Millivolt,
        VoltageResolution::CentiMillivolt,
    ];
    for len in 1..=3usize {
        for code in 0..(1usize << len) {
            let order: Vec<VoltageResolution> = (0..len).map(|i| grids[(code >> i) & 1]).collect();
            let rule = if order.iter().all(|&g| g == order[0]) {
                Rule::Builds
            } else {
                Rule::Refused(&[Error::MixedVoltageGrids])
            };
            cases.check(&format!("B/grid/{order:?}"), rule, || {
                let neurons = (0u16..)
                    .zip(&order)
                    .map(|(id, &grid)| {
                        LIFNeuron::new_with_type_resolution(id, NeuronType::Excitatory, grid)
                    })
                    .collect();
                let mut net = SpikingNeuralNetwork::from_neurons(neurons, 1000)?;
                net.step(&[]).map(|_| ())
            });
        }
    }
}

/// `set_synaptic_input_divisor` on both sides of each edge, each with six
/// weights: one step sends a pulse, and the fixed path converts the
/// synapse.
fn section_c(cases: &mut Cases) {
    for d in [
        0u16, 1, 2, 3, 10, 100, 1000, 32_766, 32_767, 32_768, 32_769, 40_000, 65_534, 65_535,
    ] {
        for w in [2000i16, -2000, 1, 0, i16::MAX, i16::MIN] {
            let rule = if d == 0 || d > 32_767 {
                Rule::Refused(&[Error::DivisorOutOfRange])
            } else {
                Rule::Builds
            };
            cases.check(&format!("C/div/d={d}/w={w}"), rule, || {
                let neurons = (0..2).map(LIFNeuron::new).collect();
                let mut net = SpikingNeuralNetwork::from_neurons(neurons, 1000)?;
                net.set_synaptic_input_divisor(d)?;
                net.add_synapse(0, 1, w)?;
                net.finalize_synapses();
                let _ = FixedSynapse::from_network(&net);
                net.step(&[3000, 0]).map(|_| ())
            });
        }
    }
}

/// Three neurons, wired, and finalized when asked.
fn net3(edges: &[(u16, u16, i16)], finalize: bool) -> Result<SpikingNeuralNetwork, Error> {
    let neurons = (0..3).map(LIFNeuron::new).collect();
    let mut net = SpikingNeuralNetwork::from_neurons(neurons, 1000)?;
    for &(pre, post, weight) in edges {
        net.add_synapse(pre, post, weight)?;
    }
    if finalize {
        net.finalize_synapses();
    }
    Ok(net)
}

/// `add_synapse` at the ids' edges on three neurons.
fn section_d(cases: &mut Cases) {
    for (pre, post) in [
        (0u16, 0u16),
        (0, 1),
        (0, 2),
        (0, 3),
        (3, 0),
        (2, 2),
        (65_535, 0),
        (0, 65_535),
        (65_535, 65_535),
    ] {
        let rule = match (pre >= 3 || post >= 3, pre == post) {
            (true, true) => Rule::Refused(&[Error::NeuronIdOutOfRange, Error::SelfConnection]),
            (true, false) => Rule::Refused(&[Error::NeuronIdOutOfRange]),
            (false, true) => Rule::Refused(&[Error::SelfConnection]),
            (false, false) => Rule::Builds,
        };
        cases.check(&format!("D/add/{pre}->{post}"), rule, || {
            net3(&[], true)?.add_synapse(pre, post, 1000)
        });
    }
}

fn fixed<const N: usize, const S: usize>(net: &SpikingNeuralNetwork) -> Result<(), Error> {
    FixedNetwork::<N, S>::try_from(net).map(|_| ())
}

/// `FixedNetwork::try_from`: the right sizes, the wrong ones, a stale CSR,
/// one finalized out of order, and, with `unstable-stdp`, plasticity on.
fn section_e(cases: &mut Cases) {
    let edges = [(0, 1, 5000), (1, 2, 5000)];
    let net = net3(&edges, true).expect("three neurons");
    cases.check("E/3-2", Rule::Builds, || fixed::<3, 2>(&net));
    cases.check(
        "E/n=4",
        Rule::Refused(&[Error::NeuronCountMismatch]),
        || fixed::<4, 2>(&net),
    );
    cases.check(
        "E/n=2",
        Rule::Refused(&[Error::NeuronCountMismatch]),
        || fixed::<2, 2>(&net),
    );
    cases.check(
        "E/s=1",
        Rule::Refused(&[Error::SynapseCountMismatch]),
        || fixed::<3, 1>(&net),
    );
    cases.check(
        "E/s=3",
        Rule::Refused(&[Error::SynapseCountMismatch]),
        || fixed::<3, 3>(&net),
    );
    let stale = net3(&[(1, 2, 5000), (0, 1, 5000)], false).expect("three neurons");
    cases.check("E/stale-csr", Rule::Refused(&[Error::StaleCsr]), || {
        fixed::<3, 2>(&stale)
    });
    let finalized = net3(&[(1, 2, 5000), (0, 1, 5000)], true).expect("three neurons");
    cases.check("E/out-of-order-finalized", Rule::Builds, || {
        fixed::<3, 2>(&finalized)
    });
    #[cfg(feature = "unstable-stdp")]
    {
        let mut on = net3(&edges, true).expect("three neurons");
        on.set_plasticity_enabled(true);
        cases.check(
            "E/plasticity-on",
            Rule::Refused(&[Error::PlasticityEnabled]),
            || fixed::<3, 2>(&on),
        );
    }
}

/// `trace::header` with twelve case names: one or more of `[a-z0-9-]`.
fn section_f(cases: &mut Cases) {
    let net = net3(&[], true).expect("three neurons");
    for name in [
        "", "ok", "ok-1", "a", "Ok", "a_b", "a b", "\u{e9}", "-", "a--b", "0", "A",
    ] {
        let fits = !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
        let rule = if fits {
            Rule::Builds
        } else {
            Rule::Refused(&[Error::BadCaseName])
        };
        cases.check(&format!("F/case/{name:?}"), rule, || {
            trace::header(&net, name, Kind::Regression, 10, Rows::All).map(|_| ())
        });
    }
}

/// Layer lists whose sizes sum past 65,535, refused by the layer rule
/// before any synapse.
fn section_g_refused(cases: &mut Cases) {
    for (list, n) in [
        (&[65_535u16, 11][..], 10u16),
        (&[65_535, 2], 1),
        (&[65_534, 4], 2),
        (&[65_535, 1], 65_535),
        (&[32_768, 32_768], 65_535),
    ] {
        let feedforward = NetworkTopology::Feedforward { layers: leak(list) };
        topology(cases, &format!("G/ff/{list:?}/n={n}"), n, 1000, feedforward);
    }
}

#[test]
fn every_constructor_case_follows_its_rule() {
    let mut cases = Cases::default();
    section_a(&mut cases);
    section_b(&mut cases);
    section_c(&mut cases);
    section_d(&mut cases);
    section_e(&mut cases);
    section_f(&mut cases);
    section_g_refused(&mut cases);
    let plasticity = usize::from(cfg!(feature = "unstable-stdp"));
    cases.assert_kept(1889 + 30 + 84 + 9 + 7 + plasticity + 12 + 5);
}

/// The cases at 65,535 neurons that build: two layer lists that sum to
/// it, and `SmallWorld` with no local connection and with two, near the
/// cap. In release an overflow wraps, so this test sees a panic or an
/// abort, never a wrong edge: the ring is
/// `smallworld_near_the_cap_wires_the_ring`'s.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "65,535 neurons: slow in a debug build; release runs it"
)]
fn the_cases_at_65_535_neurons_build() {
    let mut cases = Cases::default();
    for list in [&[65_534u16, 1][..], &[1, 65_534]] {
        let feedforward = NetworkTopology::Feedforward { layers: leak(list) };
        topology(
            &mut cases,
            &format!("G/ff/{list:?}/n=65535"),
            65_535,
            1000,
            feedforward,
        );
    }
    for k in [0, 2] {
        let small_world = NetworkTopology::SmallWorld {
            local_connections: k,
            rewiring_prob: 0.0,
        };
        topology(
            &mut cases,
            &format!("G/sw/n=65535/k={k}/p=0.0"),
            65_535,
            1000,
            small_world,
        );
    }
    cases.assert_kept(4);
}
