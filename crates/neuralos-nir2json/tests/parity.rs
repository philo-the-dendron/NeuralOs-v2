//! Parity with snnTorch, spike train for spike train. Each graph in
//! `fixtures/parity/` is snnTorch 1.0's own export, and `reference.json`
//! holds snnTorch's own spikes for it: zero reset, 0.1 ms steps, and the
//! graph's input, 1 on every feature at every step or one bitmask a step
//! (`tools/gen_snnTorch_parity.py`, the directory's README). This test
//! converts each graph as a stranger's `--sim-units` does, at snnTorch's
//! step, steps it on the same input, and holds every neuron's train to
//! snnTorch's: the first LIF layer as it is, each later layer one step
//! late, since a spike reaches the next layer on the next step
//! (`SpikingNeuralNetwork::step` § Order), over the steps both runs can
//! show. A known miss is named with its reason and pinned by its counts,
//! so a fix shows here as a change.

use std::path::{Path, PathBuf};

use neuralos_nir2json::{SIM_DT_US, convert_file_opts, effective_options};
use neuralos_snn::nir::{NirImport, NirImportOptions};

/// The graphs the bridge does not yet run spike for spike: name, the
/// spike count of each neuron here, and why.
const KNOWN_MISSES: &[(&str, &[usize], &str)] = &[
    (
        "w2.0_1.0_b0.98",
        &[1_250, 624],
        "the substrate's adaptation adds 2 a spike and NIR's LIF has none: \
         a neuron that should fire on every step fires less often",
    ),
    (
        "fan_in_16_8_4_b0.9_random",
        &[148, 342, 124, 429, 67, 0, 610, 87, 543, 0, 64, 27],
        "fan-in under random input: five trains differ, each count within one \
         of snnTorch's; the substrate's adaptation moves two of them, and three \
         differ without it",
    ),
];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity")
}

/// A config's input, one value per feature a step: `"ones"`, 1 on every
/// feature at every step, or one bitmask a step, bit i feature i.
fn input(config: &serde_json::Value, features: usize, steps: u32) -> Vec<Vec<i16>> {
    let steps = usize::try_from(steps).expect("fits");
    match &config["input"] {
        serde_json::Value::String(s) if s == "ones" => vec![vec![1; features]; steps],
        serde_json::Value::Array(masks) => {
            assert_eq!(masks.len(), steps, "one bitmask a step");
            masks
                .iter()
                .map(|m| {
                    let m = m.as_u64().expect("a bitmask");
                    (0..features).map(|i| i16::from(m >> i & 1 == 1)).collect()
                })
                .collect()
        }
        other => panic!("an input: {other}"),
    }
}

/// Each neuron's spike steps, in the network's order, over the steps of
/// `input`, one value per feature a step.
fn run(nir: &Path, input: &[Vec<i16>]) -> Vec<Vec<u32>> {
    let opts = NirImportOptions {
        dt_us: SIM_DT_US,
        ..NirImportOptions::default()
    };
    let c = convert_file_opts(nir, opts, true).expect("converts under --sim-units");
    let graph = NirImport::from_json(&c.json, effective_options(opts, true)).expect("imports");
    let (mut net, enc, _) = graph.build_network().expect("the graph builds");
    let mut trains = vec![Vec::new(); net.neurons().len()];
    for (step, x) in (0u32..).zip(input) {
        let spikes = net.step(&enc.encode(&[x.as_slice()])).expect("steps");
        for spike in spikes {
            trains[usize::from(spike.neuron_id)].push(step);
        }
    }
    trains
}

#[test]
fn a_converted_snntorch_graph_fires_snntorch_s_spikes() {
    let text = std::fs::read_to_string(fixtures().join("reference.json")).expect("the references");
    let reference: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    let steps = u32::try_from(reference["steps"].as_u64().expect("steps")).expect("fits");
    let mut exact = 0;
    for config in reference["configs"].as_array().expect("configs") {
        let name = config["name"].as_str().expect("name");
        let features =
            usize::try_from(config["features"].as_u64().expect("features")).expect("fits");
        let layers: Vec<usize> = config["layers"]
            .as_array()
            .expect("layers")
            .iter()
            .map(|n| usize::try_from(n.as_u64().expect("a size")).expect("fits"))
            .collect();
        let want: Vec<Vec<u32>> = config["spikes"]
            .as_array()
            .expect("spikes")
            .iter()
            .map(|t| {
                t.as_array()
                    .expect("a train")
                    .iter()
                    .map(|s| u32::try_from(s.as_u64().expect("a step")).expect("fits"))
                    .collect()
            })
            .collect();
        let have = run(
            &fixtures().join(config["nir"].as_str().expect("file")),
            &input(config, features, steps),
        );
        assert_eq!(have.len(), want.len(), "{name}: neurons");

        // layer L lags L steps; snnTorch's last L steps are past our run
        let mut same = true;
        let mut neuron = 0;
        for (lag, size) in layers.iter().enumerate() {
            let lag = u32::try_from(lag).expect("fits");
            for _ in 0..*size {
                let ours: Vec<u32> = have[neuron]
                    .iter()
                    .map(|&t| {
                        t.checked_sub(lag).unwrap_or_else(|| {
                            panic!(
                                "{name}: neuron {neuron} fires at step {t}, before layer {lag} can"
                            )
                        })
                    })
                    .collect();
                let theirs: Vec<u32> = want[neuron]
                    .iter()
                    .copied()
                    .filter(|t| t + lag < steps)
                    .collect();
                same &= ours == theirs;
                neuron += 1;
            }
        }
        match KNOWN_MISSES.iter().find(|(miss, ..)| *miss == name) {
            Some((_, counts, why)) => {
                let ours: Vec<usize> = have.iter().map(Vec::len).collect();
                assert!(
                    !same,
                    "{name} runs spike for spike now: take it off KNOWN_MISSES"
                );
                assert_eq!(ours, *counts, "{name}, a known miss ({why}), moved");
            }
            None => {
                assert!(same, "{name}: ours {have:?}\n  snnTorch {want:?}");
                exact += 1;
            }
        }
    }
    assert_eq!(
        exact + KNOWN_MISSES.len(),
        reference["configs"].as_array().map_or(0, Vec::len)
    );
}
