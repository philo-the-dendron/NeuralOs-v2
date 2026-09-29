#!/usr/bin/env python3
"""The parity references: snnTorch's own spikes, for the NIR bridge's test.

`crates/neuralos-nir2json/tests/parity.rs` converts each graph here as a
stranger's `--sim-units` does, at snnTorch's 0.1 ms, and holds every neuron's
spike train to snnTorch's, step for step. This script makes both halves with
snnTorch 1.0 itself: each graph through its own `export_to_nir`
(snntorch/export_nir.py: `Leaky` → `nir.LIF`, `tau = dt/(1-β)` at its
hard-coded `dt = 1e-4`, `r = 1/(1-β)`), and each run in snnTorch with
`reset_mechanism="zero"`, what NIR's LIF means (the exporter writes
`v_reset = 0` for every model), input 1 on every feature at every step, for
STEPS steps.

The set: the two-layer witness shape

    Linear(1,1) → Leaky → Linear(1,1) → Leaky        (β 0.98, threshold 1.0)

at five weight pairs, each one that a change of the bridge, taken out alone,
turns red: (1.0, 1.0) the firing rule and the refractory period, (0.3, 1.0)
the true scale of a drive, (0.05, 1.0) the voltage scale, (1.0, 0.3) the true
scale of a spiking edge, (2.0, 1.0) the substrate's adaptation, a known miss;
(0.005, 0.1) at threshold 0.1, which spikes in float exactly as (0.05, 1.0)
does and holds the voltage scale to the threshold, since its drive is weak;
and a 4 → 3 → 2 graph with fan-in, torch's own initialisation (seed 0) times
3, per-neuron β 0.9. Weights are OUR values through THEIR pipeline, the
pre-authorized class of gen_snnTorch_stranger.py. No bias anywhere: a biased
`nn.Linear` exports as `Affine`, which the converter refuses.

The emissions are not byte-stable across runs: the exporter writes the edge
list in another order each run (two runs, two shas, one graph), so the
committed files are one emission, pinned by the fixture directory's
SHA256SUMS; `reference.json` is byte-stable.

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


def fan_in(init_hidden):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 3, bias=False), leaky(3, 0.9, init_hidden),
                        nn.Linear(3, 2, bias=False), leaky(2, 0.9, init_hidden))
    with torch.no_grad():
        net[0].weight.mul_(3.0)
        net[2].weight.mul_(3.0)
    return net


def run(net, features):
    """snnTorch's own step: each Leaky's spikes feed the next Linear."""
    lin1, lif1, lin2, lif2 = net
    m1, m2 = lif1.init_leaky(), lif2.init_leaky()
    x = torch.ones(features)
    spikes = [[] for _ in range(lin1.out_features + lin2.out_features)]
    with torch.no_grad():
        for t in range(STEPS):
            k1, m1 = lif1(lin1(x), m1)
            k2, m2 = lif2(lin2(k1), m2)
            for i, k in enumerate(list(k1) + list(k2)):
                if k.item():
                    spikes[i].append(t)
    return spikes


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    graphs = [(f"w{w1}_{w2}_b0.98" + ("" if thr == 1.0 else f"_thr{thr}"),
               lambda h, a=w1, b=w2, c=thr: witness(a, b, c, h), 1, [1, 1])
              for w1, w2, thr in WITNESS]
    graphs.append(("fan_in_4_3_2_b0.9", fan_in, 4, [3, 2]))
    configs = []
    for name, make, features, layers in graphs:
        nir.write(f"{out}/{name}.nir", export_to_nir(make(True), torch.zeros(features)))
        configs.append({"name": name, "nir": f"{name}.nir", "features": features,
                        "layers": layers, "spikes": run(make(False), features)})
    reference = {"generator": "tools/gen_snnTorch_parity.py", "snntorch": snn.__version__,
                 "torch": torch.__version__, "nir": nir.__version__, "dt_s": 1e-4,
                 "steps": STEPS, "reset": "zero", "input": "1 on every feature, every step",
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
