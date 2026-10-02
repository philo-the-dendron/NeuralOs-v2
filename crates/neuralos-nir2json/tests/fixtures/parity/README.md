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

`digits.nir` and `digits.json` are a model snnTorch trains, the one
reference whose weights nobody chose, written by
`tools/gen_snnTorch_digits.py`, whose header gives how, with numpy
2.5.2 too:

    .nirenv/bin/python3 tools/gen_snnTorch_digits.py \
      crates/neuralos-nir2json/tests/fixtures/parity

A 784 → 32 → 10 model trained on 28 × 28 digits the script draws, and
snnTorch's own run of the trained model on ten test digits, one a
class, each held for 1,500 steps of 0.1 ms. `digits.json` holds each
digit's 8-bit pixels, its label and the spikes; the test reads a pixel
`p` as `p/255` in thousandths, asserts the class snnTorch gives, and
pins, per digit, how many trains of each layer run spike for spike,
most of them silent on both sides.

The exporter writes the edge list in the order Python's hash seed
gives, so both scripts run themselves under `PYTHONHASHSEED=0`, and
the digits script trains on one thread: a second run on the same
machine writes every file byte for byte, the `.nir` files and the two
references. `SHA256SUMS` pins them, from this directory:

    sha256sum -c SHA256SUMS
