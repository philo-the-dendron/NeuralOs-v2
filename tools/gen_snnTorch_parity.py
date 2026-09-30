#!/usr/bin/env python3
"""The parity references: snnTorch's own spikes, for the NIR bridge's test.

`crates/neuralos-nir2json/tests/parity.rs` converts each graph here as a
stranger's `--sim-units` does, at snnTorch's 0.1 ms, and holds every neuron's
spike train to snnTorch's, step for step. This script makes both halves with
snnTorch 1.0 itself: each graph through its own `export_to_nir`
(snntorch/export_nir.py: `Leaky` → `nir.LIF`, `tau = dt/(1-β)` at its
hard-coded `dt = 1e-4`, `r = 1/(1-β)`), and each run in snnTorch with
`reset_mechanism="zero"`, what NIR's LIF means (the exporter writes
`v_reset = 0` for every model), for STEPS steps. Each run's input is 1 on
every feature at every step, or, for a run that says so, one 0 or 1 per
feature a step, which `reference.json` holds as one bitmask a step (bit i,
feature i); a graph runs once or twice, each run a config.

The set: the two-layer witness shape

    Linear(1,1) → Leaky → Linear(1,1) → Leaky        (β 0.98, threshold 1.0)

at five weight pairs, each one that a change of the bridge, taken out alone,
turns red: (1.0, 1.0) the firing rule and the refractory period, (0.3, 1.0)
the true scale of a drive, (0.05, 1.0) the voltage scale, (1.0, 0.3) the true
scale of a spiking edge, (2.0, 1.0) the substrate's adaptation, a known miss;
(0.005, 0.1) at threshold 0.1, which spikes in float exactly as (0.05, 1.0)
does and holds the voltage scale to the threshold, since its drive is weak;
a 4 → 3 → 2 graph with fan-in, torch's own initialisation (seed 0) times
3, per-neuron β 0.9; and a 16 → 8 → 4 graph, torch's own initialisation
(seed 1) times 3, per-neuron β 0.9, under random input, 0 or 1 at p 0.3
(torch seed 7), a known miss the test pins by its counts. Weights are OUR
values through THEIR pipeline, the pre-authorized class of
gen_snnTorch_stranger.py.

Then the biased graphs: snnTorch exports a biased `nn.Linear` as `Affine`,
and the converter makes its bias an input of its own, 1 from its
population's depth, since a spike reaches the next layer one step late. The
witness shape with its second Linear biased, at six (w1, w2, bias, β), each
under all ones and under random input (0 or 1 at p 0.3, torch seed 11); a
biased 4 → 3 → 2, torch's own initialisation (seed 2) times 2, per-neuron
β 0.9, both layers biased, under all ones and random input (torch seed 12);
and a three-layer chain of unit weights, β 0.98, whose third Linear's bias,
−0.2, starts at step 2. The witness at (1.0, 1.0, −0.3, 0.98), the
4 → 3 → 2 and the chain are there for their start: a bias one step early
or late breaks their trains.

The exporter writes the edge list in the order Python's hash seed gives, so
the script runs itself under `PYTHONHASHSEED=0`: a second run writes every
file byte for byte, the `.nir` files and `reference.json`, all pinned by the
fixture directory's SHA256SUMS.

Stack (of record): the repo's .nirenv, snntorch 1.0.0 · nir
1.0.9.dev1+g7883c3c85 · torch 2.13.0+cpu. Run:

    .nirenv/bin/python3 tools/gen_snnTorch_parity.py <out dir>

Out, as committed: crates/neuralos-nir2json/tests/fixtures/parity/
"""
import json
import os
import sys

import nir
import snntorch as snn
import torch
import torch.nn as nn
from snntorch.export_nir import export_to_nir

STEPS = 1500
WITNESS = [(1.0, 1.0, 1.0), (0.3, 1.0, 1.0), (0.05, 1.0, 1.0), (1.0, 0.3, 1.0),
           (2.0, 1.0, 1.0), (0.005, 0.1, 0.1)]
# w1, w2, the second Linear's bias, β
BIASED = [(1.0, 0.5, 0.02, 0.98), (1.0, 0.5, -0.02, 0.98), (0.3, 1.0, 0.05, 0.98),
          (1.0, 1.0, -0.3, 0.98), (0.05, 2.0, 0.01, 0.98), (1.0, 0.5, 0.05, 0.9)]


def leaky(n, beta, init_hidden, threshold=1.0):
    return snn.Leaky(beta=torch.full((n,), beta), threshold=torch.full((n,), threshold),
                     init_hidden=init_hidden, reset_mechanism="zero")


def witness(w1, w2, thr, init_hidden):
    net = nn.Sequential(nn.Linear(1, 1, bias=False), leaky(1, 0.98, init_hidden, thr),
                        nn.Linear(1, 1, bias=False), leaky(1, 0.98, init_hidden, thr))
    with torch.no_grad():
        net[0].weight.fill_(w1)
        net[2].weight.fill_(w2)
    return net


def fan_in(sizes, seed, init_hidden):
    torch.manual_seed(seed)
    a, b, c = sizes
    net = nn.Sequential(nn.Linear(a, b, bias=False), leaky(b, 0.9, init_hidden),
                        nn.Linear(b, c, bias=False), leaky(c, 0.9, init_hidden))
    with torch.no_grad():
        net[0].weight.mul_(3.0)
        net[2].weight.mul_(3.0)
    return net


def biased(w1, w2, b2, beta, init_hidden):
    """The witness shape, its second Linear biased: an Affine after a
    spiking layer."""
    net = nn.Sequential(nn.Linear(1, 1, bias=False), leaky(1, beta, init_hidden),
                        nn.Linear(1, 1), leaky(1, beta, init_hidden))
    with torch.no_grad():
        net[0].weight.fill_(w1)
        net[2].weight.fill_(w2)
        net[2].bias.fill_(b2)
    return net


def fan_in_biased(sizes, seed, init_hidden):
    """PyTorch's default Linear, biased, torch's own initialisation times 2."""
    torch.manual_seed(seed)
    a, b, c = sizes
    net = nn.Sequential(nn.Linear(a, b), leaky(b, 0.9, init_hidden),
                        nn.Linear(b, c), leaky(c, 0.9, init_hidden))
    with torch.no_grad():
        for linear in (net[0], net[2]):
            linear.weight.mul_(2.0)
            linear.bias.mul_(2.0)
    return net


def deep(b3, init_hidden):
    """Three layers, unit weights, the third Linear biased: an Affine two
    spiking layers deep."""
    net = nn.Sequential(nn.Linear(1, 1, bias=False), leaky(1, 0.98, init_hidden),
                        nn.Linear(1, 1, bias=False), leaky(1, 0.98, init_hidden),
                        nn.Linear(1, 1), leaky(1, 0.98, init_hidden))
    with torch.no_grad():
        for i in (0, 2, 4):
            net[i].weight.fill_(1.0)
        net[4].bias.fill_(b3)
    return net


def random_drive(features, seed):
    """One 0 or 1 per feature a step, 1 at p 0.3."""
    torch.manual_seed(seed)
    return (torch.rand(STEPS, features) < 0.3).float()


def run(net, drive):
    """snnTorch's own step: each Leaky's spikes feed the next Linear."""
    layers = list(zip(net[0::2], net[1::2]))
    mems = [lif.init_leaky() for _, lif in layers]
    spikes = [[] for _ in range(sum(lin.out_features for lin, _ in layers))]
    with torch.no_grad():
        for t in range(STEPS):
            x, fired = drive[t], []
            for j, (lin, lif) in enumerate(layers):
                x, mems[j] = lif(lin(x), mems[j])
                fired += list(x)
            for i, k in enumerate(fired):
                if k.item():
                    spikes[i].append(t)
    return spikes


def main():
    # the exporter's edge list follows the hash seed: 0 makes it one order
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  {**os.environ, "PYTHONHASHSEED": "0"})
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    # each graph: its name, how to make it, its features and layers, and
    # its runs, each a suffix to the name and a drive (None: all ones)
    graphs = [(f"w{w1}_{w2}_b0.98" + ("" if thr == 1.0 else f"_thr{thr}"),
               lambda h, a=w1, b=w2, c=thr: witness(a, b, c, h), 1, [1, 1], [("", None)])
              for w1, w2, thr in WITNESS]
    graphs.append(("fan_in_4_3_2_b0.9", lambda h: fan_in((4, 3, 2), 0, h), 4, [3, 2],
                   [("", None)]))
    graphs.append(("fan_in_16_8_4_b0.9_random", lambda h: fan_in((16, 8, 4), 1, h), 16,
                   [8, 4], [("", random_drive(16, 7))]))
    graphs += [(f"w{w1}_{w2}_bias{b2}_b{beta}",
                lambda h, a=w1, b=w2, c=b2, d=beta: biased(a, b, c, d, h), 1, [1, 1],
                [("", None), ("_random", random_drive(1, 11))])
               for w1, w2, b2, beta in BIASED]
    graphs.append(("fan_in_4_3_2_bias_b0.9", lambda h: fan_in_biased((4, 3, 2), 2, h), 4,
                   [3, 2], [("", None), ("_random", random_drive(4, 12))]))
    graphs.append(("w1.0_1.0_1.0_bias-0.2_b0.98", lambda h: deep(-0.2, h), 1, [1, 1, 1],
                   [("", None)]))
    configs = []
    for name, make, features, layers, runs in graphs:
        nir.write(f"{out}/{name}.nir", export_to_nir(make(True), torch.zeros(features)))
        for suffix, drive in runs:
            if drive is None:
                given, drive = "ones", torch.ones(STEPS, features)
            else:
                given = [sum(1 << i for i, v in enumerate(x) if v) for x in drive.tolist()]
            configs.append({"name": name + suffix, "nir": f"{name}.nir", "features": features,
                            "layers": layers, "input": given,
                            "spikes": run(make(False), drive)})
    reference = {"generator": "tools/gen_snnTorch_parity.py", "snntorch": snn.__version__,
                 "torch": torch.__version__, "nir": nir.__version__, "dt_s": 1e-4,
                 "steps": STEPS, "reset": "zero",
                 "input": "per config: \"ones\", 1 on every feature at every step, or one "
                          "bitmask a step, bit i feature i",
                 "configs": configs}
    # one graph a line: the spike lists are data, not prose to diff
    with open(f"{out}/reference.json", "w") as f:
        f.write("{\n")
        for key, value in reference.items():
            if key != "configs":
                f.write(f" {json.dumps(key)}: {json.dumps(value)},\n")
        f.write(' "configs": [\n')
        f.write(",\n".join(f"  {json.dumps(c)}" for c in configs))
        f.write("\n ]\n}\n")
    print(f"{len(configs)} graphs, {STEPS} steps, snntorch {snn.__version__}")


main()
