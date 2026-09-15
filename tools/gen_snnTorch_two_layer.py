#!/usr/bin/env python3
"""The two-layer framework witness (stranger-file ladder rung ii, pre-authorized).

ROADMAP § 0.1.0 check 6 asks for a two-layer snnTorch, norse or rockpool
graph that builds (LIF→Linear→LIF). The paper corpus has one graph with a
LIF→Linear edge inside the four-kind subset, two_lif_neurons.nir, and the
`nir` library wrote it, not a framework. This script makes the framework's:
snnTorch's own `export_to_nir` (snntorch/export_nir.py: `Leaky` → `nir.LIF`,
`tau = dt/(1-β)` at its hard-coded `dt = 1e-4`, `r = 1/(1-β)`) on

    Linear(1,1) → Leaky → Linear(1,1) → Leaky

with both weights filled with 1.0, β = 0.98 and threshold 1.0: OUR values,
THEIR pipeline, the pre-authorized class of gen_snnTorch_stranger.py. It
emits input → 0 → 1 → 2 → 3 → output, LIF nodes of τ 5 ms, r 50 Ω, leak 0,
threshold 1.0, which neuralos-nir2json converts under --sim-units (r 50,000
MΩ, inside the u16 ceiling; β 0.99 would put 100,000 MΩ over it).

The emission is not byte-stable across runs (HDF5 ordering: two runs, two
shas, one graph), so the committed fixture is one emission, pinned by its
sha in PROVENANCE.md. The seed line is moot with the weights filled and is
kept for the form.

Stack (of record, in PROVENANCE.md): the repo's .nirenv, snntorch 1.0.0 ·
nirtorch 2.6 · nir 1.0.9.dev1+g7883c3c85 · torch 2.13.0+cpu. Run:

    .nirenv/bin/python3 tools/gen_snnTorch_two_layer.py <out.nir>

Out, as committed:
crates/neuralos-nir2json/tests/fixtures/community/snnTorch_two_layer.nir
"""
import torch, torch.nn as nn, snntorch as snn, nir, sys
from snntorch.export_nir import export_to_nir
torch.manual_seed(0)
net = nn.Sequential(
    nn.Linear(1, 1, bias=False),
    snn.Leaky(beta=torch.tensor([0.98]), threshold=torch.tensor([1.0]), init_hidden=True),
    nn.Linear(1, 1, bias=False),
    snn.Leaky(beta=torch.tensor([0.98]), threshold=torch.tensor([1.0]), init_hidden=True),
)
with torch.no_grad():
    net[0].weight.fill_(1.0); net[2].weight.fill_(1.0)
g = export_to_nir(net, torch.zeros(1))   # a (1,1) sample or a scalar beta fails type inference
nir.write(sys.argv[1], g)
