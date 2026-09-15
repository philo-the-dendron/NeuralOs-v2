#!/usr/bin/env python3
"""The two corners of a capacity bar (ROADMAP § 0.1.0 check 8), by snnTorch's exporter.

The witness's pipeline, tools/gen_snnTorch_two_layer.py, with two ints and
per-neuron tensors: snnTorch's own `export_to_nir` in the repo's .nirenv,
every weight 1.0, β 0.98 and threshold 1.0, OUR values through THEIR
pipeline, the pre-authorized class of that script. β and threshold are
`[n]` tensors, one value per neuron: a `[1]` tensor makes the exporter
type-check the layer as one neuron.

The sizes come from a byte count, the bar's `44·N + 6·S` (a neuron is 44
bytes of the frozen arrays, a synapse 6), so the corners regenerate from
that number alone:

    neurons  Linear(1,N) → Leaky(N): N = bytes // 44, S = 0, the most
             neurons (the first Linear is the drive, not a synapse);
    dense    Linear(1,n) → Leaky(n) → Linear(n,n) → Leaky(n): N = 2n,
             S = n², the largest n with 88·n + 6·n² ≤ bytes, the most
             synapses.

A step costs a·N + b·S + c, linear in both, so over the graphs under the
bar its largest value sits at one of these two.

The emission is not byte-stable across runs (HDF5 ordering, as
PROVENANCE.md says of the witness), so the frozen module and its trace
are what a record pins. The .nir goes under
firmware/esp32c3/target/stranger/ (gitignored), never committed. Run:

    .nirenv/bin/python3 tools/gen_snnTorch_corner.py neurons|dense <bytes> <out.nir>

It prints N, S and 44·N + 6·S. Exit: 0 written · 2 usage.
"""
import sys

USAGE = "usage: tools/gen_snnTorch_corner.py neurons|dense <bytes> <out.nir>"


def widths(corner, budget):
    """The LIF layers' widths for a corner under `budget` bytes of arrays."""
    if corner == "neurons":
        return [budget // 44]
    n = 0
    while 88 * (n + 1) + 6 * (n + 1) ** 2 <= budget:
        n += 1
    return [n, n]


def main():
    args = sys.argv[1:]
    if len(args) != 3 or args[0] not in ("neurons", "dense") or not args[1].isdigit():
        print(USAGE, file=sys.stderr)
        return 2
    corner, budget, out = args[0], int(args[1]), args[2]
    layers = widths(corner, budget)
    if layers[0] < 1:
        print(f"gen_snnTorch_corner: {budget} bytes hold no {corner} graph", file=sys.stderr)
        return 2

    import torch, torch.nn as nn, snntorch as snn, nir
    from snntorch.export_nir import export_to_nir

    modules, fan_in = [], 1
    for n in layers:
        modules.append(nn.Linear(fan_in, n, bias=False))
        modules.append(
            snn.Leaky(beta=torch.full((n,), 0.98), threshold=torch.full((n,), 1.0), init_hidden=True)
        )
        fan_in = n
    net = nn.Sequential(*modules)
    with torch.no_grad():
        for m in net:
            if isinstance(m, nn.Linear):
                m.weight.fill_(1.0)
    g = export_to_nir(net, torch.zeros(1))  # a (1,1) sample fails type inference
    nir.write(out, g)

    n_all = sum(layers)
    s_all = layers[0] * layers[1] if corner == "dense" else 0
    print(f"{corner}: N {n_all}, S {s_all}, 44·N + 6·S = {44 * n_all + 6 * s_all} bytes of {budget}: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
