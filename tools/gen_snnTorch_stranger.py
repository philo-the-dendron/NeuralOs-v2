#!/usr/bin/env python3
"""The pre-authorized fallback emission (stranger-file ladder rung ii).

The sealed in-repo nomination (neuromorphs/NIR paper corpus) landed on
the parameter-convention wall: every third-party LIF emission in the
corpus carries the simulation-unit convention (r = 1..24 Ω), below the
substrate's biological quantization floor (r ≥ 1 MΩ) — the corpus-v2
finding of record. The sealed ladder's rung (ii) anticipated exactly
this: "generated with snnTorch installed — emitter-skew preserved
(snnTorch's to_nir path IS the audience's door)".

This script emits a LINEAR-ONLY graph (an encoder/readout head — no
LIF, so no r-convention anywhere) through snnTorch's own extractor
(`snntorch.export_nir._extract_snntorch_module` + `nirtorch` graph
extraction) and the `nir` writer. Weights are seeded and STATED in the
provenance: OUR values, THEIR pipeline — the pre-authorized class,
recorded openly. The weights land float32 (torch's default) — the f32
widening path exercised by a genuine stranger emission.

Stack (of record, in PROVENANCE.md): the CURRENT stranger stack from
PyPI — the pinned 2025-era nir clone cannot host snnTorch 1.0's
export path (version-skew demonstrated three ways during the session;
that too is the emitter reality). Run in a throwaway venv:

    python3 -m venv /tmp/snn-venv
    /tmp/snn-venv/bin/pip install torch --index-url \\
        https://download.pytorch.org/whl/cpu
    /tmp/snn-venv/bin/pip install snntorch nirtorch nir
    /tmp/snn-venv/bin/python3 tools/gen_snnTorch_stranger.py

Out: crates/neuralos-nir2json/tests/fixtures/community/snnTorch_linear_head.nir
"""
import os

import nir
import numpy as np  # noqa: F401  (nir.Input construction below uses arrays)
import torch
from nirtorch.to_nir import extract_nir_graph
from snntorch.export_nir import _extract_snntorch_module

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(
    ROOT, "crates/neuralos-nir2json/tests/fixtures/community/snnTorch_linear_head.nir"
)

torch.manual_seed(7883)  # stated in PROVENANCE.md — values ours, pipeline theirs
head = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False))
sample = torch.randn(64)  # UNBATCHED — the canonical NIR input shape; batched
# samples fail the old-stack type inference on the Input->Linear edge
# (recorded: the torch-batch artifact is not a NIR shape).

graph = extract_nir_graph(head, _extract_snntorch_module, sample, model_name="snnTorch_linear_head")
graph.infer_types()
nir.write(OUT, graph)
import snntorch  # noqa: E402

print("nodes:", {k: type(v).__name__ for k, v in graph.nodes.items()})
print("edges:", graph.edges)
print("weight:", graph.nodes["0"].weight.dtype, tuple(graph.nodes["0"].weight.shape))
print(
    f"wrote {OUT} ({os.path.getsize(OUT)} B) · "
    f"snntorch {snntorch.__version__} · torch {torch.__version__} · "
    f"nirtorch {__import__('nirtorch').version.VERSION} · "
    f"nir {getattr(nir, '__version__', 'unknown')}"
)
