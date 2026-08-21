#!/usr/bin/env python3
"""Generate the NIR import fixtures from the PINNED reference itself.

Doctrine (AGENTS.md): authoritative layouts come from the reference
sources VERBATIM — test vectors derive from the reference's own
outputs, never from our re-implementation of ourselves.

Usage (numpy required; the repo pins the reference at nir-ref/):
    /path/to/venv-with-numpy/bin/python3 tools/gen_nir_fixtures.py

Reference: neuromorphs/NIR @ 7883c3c85f1be27ed113ccc9e8d6ab47ab541df4
(clone at nir-ref/, gitignored). Positive fixtures are emitted by the
reference's own classes + to_dict(); the negative fixtures are
MUTATIONS of that emission (each mutation is the minimal surgery for
one error class — the reference cannot emit malformed documents by
construction, so these are the only honest source).

Outputs: crates/neuralos-snn/tests/nir_fixtures/*.json
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nir-ref"))

import numpy as np  # noqa: E402
import nir  # noqa: E402  (the pinned clone)

OUT = ROOT / "crates/neuralos-snn/tests/nir_fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    return o


def dump(name, doc):
    path = OUT / name
    path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} B)")


# --- positive 1: the canonical chain, reference-emitted -------------
# Dyadic weights/params so the substrate quantization is predictable
# (0.5 -> 16384, 1.0 -> 32767 at scale 1/32767; -70/-55/-80 mV quanta).
# NOTE: the reference's type checker requires Linear rows == LIF
# population size — a LIF node is a POPULATION (per-neuron arrays).
# Slice 1 imports length-1 populations only (per-neuron expansion is
# the named slice-2 work), so the canonical chain here is 1x3 -> 1.
lin = nir.Linear(weight=np.array([[0.5, -1.0, 0.25]]))
lif = nir.LIF(
    tau=np.array([0.02]),
    r=np.array([1e8]),
    v_leak=np.array([-0.07]),
    v_threshold=np.array([-0.055]),
    v_reset=np.array([-0.08]),
)
g = nir.NIRGraph.from_list(lin, lif)
doc = {"version": "1.0.0", "node": to_jsonable(g.to_dict())}
dump("chain.json", doc)

# --- positive 2: absent v_reset + lossy weights ---------------------
# The absent-v_reset shape is reference-legal: serialization.read_node
# constructs it when the dataset is missing, and dict2NIRNode accepts
# it (from_dict fills zeros). We validate it THROUGH the reference
# loader, then emit it in the serialization shape (no v_reset key).
lif2_dict = {
    "type": "LIF",
    "tau": np.array([0.01]),
    "r": np.array([2e8]),
    "v_leak": np.array([-0.065]),
    "v_threshold": np.array([-0.05]),
}
node2 = nir.dict2NIRNode(lif2_dict)  # must construct (proves legality)
assert float(node2.v_reset[0]) == 0.0
lin2 = nir.Linear(weight=np.array([[0.1, 0.3, -0.2]]))
g2 = nir.NIRGraph.from_list(lin2, node2)
g2_dict = to_jsonable(g2.to_dict())
# to_dict carries the FILLED v_reset; emit the serialization shape
# (absent) that read_node produces for datasets written without it:
for n in g2_dict["nodes"].values():
    if n.get("type") == "LIF":
        del n["v_reset"]
dump("chain_vreset_absent.json", {"version": "1.0.0", "node": g2_dict})

# --- negative: minimal mutations of the positive emission -----------
base = json.dumps(doc, separators=(",", ":"))


def mutate(name, fn):
    d = json.loads(base)
    fn(d)
    dump(name, d)


def nodes(d):
    return d["node"]["nodes"]


mutate(
    "neg_affine.json",
    lambda d: nodes(d).__setitem__(
        "lif", {"type": "Affine", "weight": [[1.0]], "bias": [0.5]}
    ),
)
mutate(
    "neg_unknown_kind.json",
    lambda d: nodes(d).__setitem__("lif", {"type": "CubaLIF", "tau_syn": [0.01]}),
)
mutate(
    "neg_tau_zero.json",
    lambda d: nodes(d)["lif"].__setitem__("tau", [0.0]),
)
mutate(
    "neg_threshold_zero_quant.json",
    lambda d: nodes(d)["lif"].__setitem__("v_threshold", [-0.0004]),
)
mutate(
    "neg_tau_below_dt.json",
    lambda d: nodes(d)["lif"].__setitem__("tau", [0.0005]),
)
mutate(
    "neg_potential_out_of_range.json",
    lambda d: nodes(d)["lif"].__setitem__("v_leak", [0.06]),
)
mutate(
    "neg_missing_field.json",
    lambda d: nodes(d)["lif"].pop("v_threshold"),
)
mutate(
    "neg_unknown_endpoint.json",
    lambda d: d["node"]["edges"].__setitem__(0, ["input", "ghost"]),
)
mutate(
    "neg_duplicate_edge.json",
    lambda d: d["node"]["edges"].__setitem__(1, d["node"]["edges"][0]),
)
mutate(
    "neg_ragged_weight.json",
    lambda d: nodes(d)["linear"].__setitem__(
        "weight", [[0.5, -1.0, 0.25], [0.1, 0.2, 0.3, 0.4]]
    ),
)
mutate(
    "neg_param_length.json",
    lambda d: nodes(d)["lif"].__setitem__("tau", [0.02, 0.03]),
)
mutate(
    "neg_missing_version.json",
    lambda d: d.pop("version"),
)

# string-level mutations the json model cannot express:
escaped = base.replace('"lif"', '"l\\u0069f1"', 1)
(OUT / "neg_escaped_name.json").write_text(escaped + "\n")
print(f"wrote {OUT.relative_to(ROOT)}/neg_escaped_name.json")

print("done — reference:", nir.version if hasattr(nir, "version") else "?")
