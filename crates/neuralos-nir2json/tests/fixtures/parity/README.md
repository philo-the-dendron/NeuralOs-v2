# Parity fixtures

snnTorch 1.0's own exports and runs, for `tests/parity.rs`: it converts
each graph as a stranger's `--sim-units` does, at snnTorch's 0.1 ms,
and holds every neuron's spike train to snnTorch's.

Written by `tools/gen_snnTorch_parity.py`, whose header gives the set
and why each graph is in it, in the repo's `.nirenv` (snntorch 1.0.0,
nir 1.0.9.dev1+g7883c3c85, torch 2.13.0+cpu):

    .nirenv/bin/python3 tools/gen_snnTorch_parity.py \
      crates/neuralos-nir2json/tests/fixtures/parity

`reference.json` holds snnTorch's spikes: `reset_mechanism="zero"`,
what NIR's LIF means (snnTorch's exporter writes `v_reset = 0` for
every model), input 1 on every feature at every step, 1,500 steps of
0.1 ms. A second run writes it byte for byte.

The `.nir` files do not come out byte for byte: the exporter writes the
edge list in another order each run, and the nodes, their parameters
and the set of edges stay the same (a second run's files pass the
test). The committed emission is pinned by `SHA256SUMS`, from this
directory:

    sha256sum -c SHA256SUMS
