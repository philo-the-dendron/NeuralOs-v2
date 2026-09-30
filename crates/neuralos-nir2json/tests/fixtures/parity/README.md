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
every model), 1,500 steps of 0.1 ms, and each run's input: 1 on every
feature at every step, or a random 0 or 1 per feature at p 0.3, held
as one bitmask a step. snnTorch exports a biased `nn.Linear` as an
`Affine`, whose bias the converter makes an input of its own; the test
drives it as `--freeze` does, 1 from its population's depth.

The exporter writes the edge list in the order Python's hash seed
gives, so the script runs itself under `PYTHONHASHSEED=0`: a second run
writes every file byte for byte, the `.nir` files and `reference.json`.
`SHA256SUMS` pins them, from this directory:

    sha256sum -c SHA256SUMS
