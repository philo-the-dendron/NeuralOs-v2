//! Parity with snnTorch, spike train for spike train. Each graph in
//! `fixtures/parity/` is snnTorch 1.0's own export, and `reference.json`
//! holds snnTorch's own spikes for each run of it: zero reset, 0.1 ms
//! steps, and the run's input, 1 on every feature at every step or one
//! bitmask a step (`tools/gen_snnTorch_parity.py`, the directory's
//! README). `digits.json` holds the same for a model snnTorch trained,
//! each test digit's 8-bit pixels held as a current
//! (`tools/gen_snnTorch_digits.py`). This test converts each graph as a
//! stranger's `--sim-units` does, at snnTorch's step, steps it on the
//! same input, each bias input the conversion adds driven as `--freeze`
//! drives it, and holds every neuron's train to snnTorch's: the first LIF
//! layer as it is, each later layer one step late per layer, since a
//! spike reaches the next layer on the next step
//! (`SpikingNeuralNetwork::step` § Order), over the steps both runs can
//! show. A known miss is named with its reason and pinned by its counts,
//! the trained model's by each layer's count of exact trains, so a fix
//! shows here as a change.

use std::path::{Path, PathBuf};

use neuralos_nir2json::{Inputs, SIM_DT_US, convert_file_opts, effective_options};
use neuralos_snn::nir::{NirImport, NirImportOptions, Thousandths};

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

/// The digits model, `digits.json`: per test digit, its label and how many
/// trains of each layer, of 32 and 10, run spike for spike. Most exact
/// trains are silent on both sides: of the trains snnTorch fires on these
/// ten digits, 101 of 155 in layer 1 and 1 of 25 in layer 2 run exact.
/// Every digit misses in some, and none for the bridge's logic: above all
/// for the substrate's adaptation, which adds 2 a spike where NIR's LIF
/// adds nothing, then for its resolution, the weights at 1/1000 of a unit,
/// a learned β's leak rate at 1/1000 of a step, so a neuron up to one part
/// in `1000 · (1 − β)` slower, and `r` in whole MΩ. A closer substrate
/// shows here as a count that rises.
const DIGITS: &[(u64, [usize; 2])] = &[
    (0, [28, 8]),
    (1, [31, 9]),
    (2, [30, 8]),
    (3, [25, 6]),
    (4, [24, 6]),
    (5, [28, 7]),
    (6, [23, 6]),
    (7, [25, 7]),
    (8, [29, 8]),
    (9, [23, 8]),
];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity")
}

/// A config's input, one value per feature a step: `"ones"`, 1 on every
/// feature at every step; one bitmask a step, bit i feature i; or
/// `{"pixels": [...]}`, one 8-bit pixel a feature held at every step, `p`
/// read as `p/255` in thousandths, rounded.
fn input(config: &serde_json::Value, features: usize, steps: u32) -> Vec<Vec<Thousandths>> {
    let steps = usize::try_from(steps).expect("fits");
    match &config["input"] {
        serde_json::Value::String(s) if s == "ones" => {
            vec![vec![Thousandths(1_000); features]; steps]
        }
        serde_json::Value::Array(masks) => {
            assert_eq!(masks.len(), steps, "one bitmask a step");
            masks
                .iter()
                .map(|m| {
                    let m = m.as_u64().expect("a bitmask");
                    (0..features)
                        .map(|i| Thousandths(i32::from(m >> i & 1 == 1) * 1_000))
                        .collect()
                })
                .collect()
        }
        serde_json::Value::Object(form) => {
            let pixels: Vec<Thousandths> = form["pixels"]
                .as_array()
                .expect("pixels")
                .iter()
                .map(|p| {
                    let p = p.as_u64().filter(|&p| p <= 255).expect("an 8-bit pixel");
                    let p = i32::try_from(p).expect("fits");
                    // p · 1,000 / 255, rounded: no pixel falls on a half
                    Thousandths((p * 2_000 + 255) / 510)
                })
                .collect();
            assert_eq!(pixels.len(), features, "one pixel a feature");
            vec![pixels; steps]
        }
        other => panic!("an input: {other}"),
    }
}

/// Each neuron's spike steps, in the network's order, over the steps of
/// `input`, one value per feature of the graph's own Inputs a step; each
/// bias input the conversion adds is driven as `freeze` drives it
/// (`Inputs`).
fn run(nir: &Path, input: &[Vec<Thousandths>]) -> Vec<Vec<u32>> {
    let opts = NirImportOptions {
        dt_us: SIM_DT_US,
        ..NirImportOptions::default()
    };
    let c = convert_file_opts(nir, opts, true).expect("converts under --sim-units");
    let graph = NirImport::from_json(&c.json, effective_options(opts, true)).expect("imports");
    let (mut net, enc, _) = graph.build_network().expect("the graph builds");
    let inputs = Inputs::new(&graph, &enc, &c.stamp.bias).expect("the conversion's bias");
    let mut trains = vec![Vec::new(); net.neurons().len()];
    // a held input gives the same currents at every step: encoded once,
    // as `freeze` encodes each run of its drive
    let mut held: Option<(Vec<Vec<Thousandths>>, Vec<i16>)> = None;
    for (step, x) in (0u32..).zip(input) {
        let per_input = inputs.at(step, x);
        if held.as_ref().is_none_or(|(values, _)| *values != per_input) {
            let slices: Vec<&[Thousandths]> = per_input.iter().map(Vec::as_slice).collect();
            let currents = enc.encode(&slices);
            held = Some((per_input, currents));
        }
        let (_, currents) = held.as_ref().expect("encoded");
        let spikes = net.step(currents).expect("steps");
        for spike in spikes {
            trains[usize::from(spike.neuron_id)].push(step);
        }
    }
    trains
}

/// A config's LIF layers, their sizes in network order.
fn layers(config: &serde_json::Value) -> Vec<usize> {
    config["layers"]
        .as_array()
        .expect("layers")
        .iter()
        .map(|n| usize::try_from(n.as_u64().expect("a size")).expect("fits"))
        .collect()
}

/// Each neuron's train beside snnTorch's, in network order.
type Trains = Vec<(Vec<u32>, Vec<u32>)>;

/// Every run of `file`, a reference in `fixtures/parity/`: its config,
/// and its [`Trains`]. Layer L lags L steps, so its spikes move L steps
/// back, and snnTorch's last L steps are past our run.
fn compared(file: &str) -> Vec<(serde_json::Value, Trains)> {
    let text = std::fs::read_to_string(fixtures().join(file)).expect("the references");
    let reference: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    let steps = u32::try_from(reference["steps"].as_u64().expect("steps")).expect("fits");
    let configs = reference["configs"].as_array().expect("configs");
    configs
        .iter()
        .map(|config| {
            let name = config["name"].as_str().expect("name");
            let features =
                usize::try_from(config["features"].as_u64().expect("features")).expect("fits");
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
            let mut pairs = Vec::with_capacity(have.len());
            for (lag, size) in (0u32..).zip(layers(config)) {
                for _ in 0..size {
                    let neuron = pairs.len();
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
                    pairs.push((ours, theirs));
                }
            }
            (config.clone(), pairs)
        })
        .collect()
}

#[test]
fn a_converted_snntorch_graph_fires_snntorch_s_spikes() {
    let runs = compared("reference.json");
    let mut exact = 0;
    for (config, pairs) in &runs {
        let name = config["name"].as_str().expect("name");
        let same = pairs.iter().all(|(ours, theirs)| ours == theirs);
        match KNOWN_MISSES.iter().find(|(miss, ..)| *miss == name) {
            Some((_, counts, why)) => {
                let ours: Vec<usize> = pairs.iter().map(|(ours, _)| ours.len()).collect();
                assert!(
                    !same,
                    "{name} runs spike for spike now: take it off KNOWN_MISSES"
                );
                assert_eq!(ours, *counts, "{name}, a known miss ({why}), moved");
            }
            None => {
                let (ours, theirs): (Vec<_>, Vec<_>) = pairs.iter().cloned().unzip();
                assert!(same, "{name}: ours {ours:?}\n  snnTorch {theirs:?}");
                exact += 1;
            }
        }
    }
    assert_eq!(exact + KNOWN_MISSES.len(), runs.len());
}

/// The one reference whose weights nobody chose, the digits model: on
/// every test digit the class, the last layer's neuron with the most
/// spikes (the first of a tie), is snnTorch's, and each layer's count of
/// exact trains is the one `DIGITS` pins.
#[test]
fn a_trained_snntorch_model_gives_snntorch_s_class() {
    let runs = compared("digits.json");
    let mut table = Vec::new();
    for (config, pairs) in &runs {
        let name = config["name"].as_str().expect("name");
        let sizes = layers(config);
        let last = &pairs[pairs.len() - sizes.last().copied().expect("a layer")..];
        let ours: Vec<usize> = last.iter().map(|(ours, _)| ours.len()).collect();
        let theirs: Vec<usize> = last.iter().map(|(_, theirs)| theirs.len()).collect();
        let class = |counts: &[usize]| counts.iter().position(|n| Some(n) == counts.iter().max());
        assert_eq!(
            class(&ours),
            class(&theirs),
            "{name}: the class, ours {ours:?}, snnTorch {theirs:?}"
        );
        let mut exact = Vec::new();
        let mut from = 0;
        for size in sizes {
            let layer = &pairs[from..from + size];
            exact.push(layer.iter().filter(|(ours, theirs)| ours == theirs).count());
            from += size;
        }
        let exact: [usize; 2] = exact.try_into().expect("two layers");
        table.push((config["label"].as_u64().expect("a label"), exact));
    }
    assert_eq!(table, DIGITS, "the digits model's exact trains moved");
}
