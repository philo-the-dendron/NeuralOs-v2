#!/usr/bin/env python3
"""Generate the NIR import fixtures from the PINNED reference itself.

Doctrine (AGENTS.md): authoritative layouts come from the reference
sources VERBATIM — test vectors derive from the reference's own
outputs, never from our re-implementation of ourselves.

Usage (the repo-local .nirenv carries numpy+h5py and the pinned clone
pip-installed, so `nir.version` metadata resolves):
    .nirenv/bin/python3 tools/gen_nir_fixtures.py            # fixtures
    .nirenv/bin/python3 tools/gen_nir_fixtures.py --verify <our.nir>
                                       # load OUR HDF5 export with the
                                       # reference's own read() and
                                       # assert structural + numeric
                                       # equality (the interop leg)

Reference: neuromorphs/NIR @ 7883c3c85f1be27ed113ccc9e8d6ab47ab541df4
(clone at nir-ref/, gitignored). Positive fixtures are emitted by the
reference's own classes + to_dict()/write(); the negative fixtures are
MUTATIONS of that emission (each mutation is the minimal surgery for
one error class — the reference cannot emit malformed documents by
construction, so these are the only honest source).

Outputs: crates/neuralos-snn/tests/nir_fixtures/*.json
         crates/neuralos-rt/tests/nir_fixtures/*.nir
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nir-ref"))

import numpy as np  # noqa: E402
import nir  # noqa: E402  (the pinned clone)

OUT = ROOT / "crates" / "neuralos-snn" / "tests" / "nir_fixtures"
OUT.mkdir(parents=True, exist_ok=True)

# the frozen population values (both containers share them — the
# cross-container comparability contract)
POP_WEIGHT = np.array([[0.5, -1.0, 0.25], [-0.25, 1.0, 0.5]])
POP_LIF = dict(
    tau=np.array([0.02, 0.03]),
    r=np.array([1e8, 2e8]),
    v_leak=np.array([-0.07, -0.065]),
    v_threshold=np.array([-0.055, -0.05]),
    v_reset=np.array([-0.08, -0.075]),
)


def verify_our_export(path: pathlib.Path) -> int:
    """Load OUR .nir export with the reference's own read() + assert.

    The export writes DEQUANTIZED schema values (w' = q·scale,
    potentials/τ/R rendered from the records) — the reference must see
    a plain NIRGraph equal to the frozen population source within
    quantization error (≤ scale/2 for weights, exact for dyadic ones).
    """
    g = nir.read(path)
    names = set(g.nodes)
    assert names == {"input", "linear", "lif", "output"}, f"nodes: {names}"
    assert sorted(g.edges) == sorted(
        [("input", "linear"), ("linear", "lif"), ("lif", "output")]
    ), f"edges: {g.edges}"
    lin = g.nodes["linear"]
    assert lin.weight.shape == (2, 3), lin.weight.shape
    # our export writes dequantized weights (q·scale); the source
    # values are dyadic so the reconstruction is exact modulo the
    # quantizer's half-step bound
    scale = float(np.abs(POP_WEIGHT).max()) / 32767.0
    dequant_err = float(np.abs(lin.weight - POP_WEIGHT).max())
    assert dequant_err <= scale / 2 + 1e-12, f"weights off by {dequant_err} (scale {scale})"
    lif = g.nodes["lif"]
    for field, expect in POP_LIF.items():
        got = getattr(lif, field)
        tol = 5e-6 if field == "tau" else 5e-4  # record-rendered (us/mV grids)
        assert np.allclose(got, expect, atol=tol, rtol=0), f"{field}: {got} vs {expect}"
    print(f"VERIFY: PASS — the reference's own read() loads our export:")
    print(f"  nodes {sorted(names)} · edges {sorted(g.edges)}")
    print(f"  weight max |Δ| vs source {dequant_err:.3e} (≤ scale/2 = {scale / 2:.3e})")
    print(f"  LIF params within record-render tolerance (tau ≤ 5e-6 s, V ≤ 5e-4 V)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--verify":
        raise SystemExit(verify_our_export(pathlib.Path(sys.argv[2])))


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
# The canonical chain stays 1x3 -> 1 (the slice-1 shape);
# chain_population.json carries the 2-neuron expansion.
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

# --- positive 1b: a 2-neuron LIF population -------------------------
# Per-neuron expansion: distinct params per neuron prove per-neuron
# quantization (not one record cloned). Dyadic values keep the
# expected quanta predictable.
lin_pop = nir.Linear(weight=np.array([[0.5, -1.0, 0.25], [-0.25, 1.0, 0.5]]))
lif_pop = nir.LIF(
    tau=np.array([0.02, 0.03]),
    r=np.array([1e8, 2e8]),
    v_leak=np.array([-0.07, -0.065]),
    v_threshold=np.array([-0.055, -0.05]),
    v_reset=np.array([-0.08, -0.075]),
)
g_pop = nir.NIRGraph.from_list(lin_pop, lif_pop)
dump("chain_population.json", {"version": "1.0.0", "node": to_jsonable(g_pop.to_dict())})

# --- HDF5 fixtures (slice 2) — the reference's own write() ----------
# Same doctrine, one container over: positive .nir files are the
# pinned reference's own `nir.write()` emissions. The population chain
# REUSES chain_population.json's exact values so the frozen quanta are
# cross-container comparable (same dyadic weights, same per-neuron
# params). Negative container: lzf — write(compression="lzf") is
# reference-legal, censused out by our reader (stated policy).
# SZIP: unemittable by this toolchain (libaec rejects every legal ppb
# against NIR-scale chunks) — documented census rejection, no fixture.
# Empty edges: unemittable — NIRGraph auto-wires input_<n>/<n>_output
# junctions into edges=[] (probe finding of record, 2026-08-21).
OUT_H5 = ROOT / "crates" / "neuralos-rt" / "tests" / "nir_fixtures"
OUT_H5.mkdir(parents=True, exist_ok=True)


def dump_h5(name, graph, **write_kwargs):
    import hashlib

    path = OUT_H5 / name
    nir.write(path, graph, **write_kwargs)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} B, sha256 {sha[:16]}…)")


dump_h5("chain_population.nir", g_pop)  # gzip default (deflate id 1)
dump_h5("chain_population_uncompressed.nir", g_pop, compression=None)
dump_h5("neg_filter_lzf.nir", g_pop, compression="lzf")

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

# --- positive 3: general graphs (assembly slice, R18 P0) -------------
# Doctrine: constructed with EXPLICIT Input/Output nodes — the pinned
# trap is that infer_types silently auto-wires input_<n>/<n>_output
# junctions around any implicit boundary, rewriting merges into
# fork-shaped emissions. Every graph below self-asserts its exact
# node/edge set after construction (the reference must not have
# rewritten anything) before the bytes land in a fixture.
#
# The frozen POP values ride branch's first LIF population so quanta
# stay cross-fixture comparable; the recurrent pops pin the D1
# contract point (tau=20 ms, R=100 MOhm -> dt_over_tau=50 at dt=1 ms).

POP_W1 = np.array([[0.5, -1.0, 0.25], [-0.25, 1.0, 0.5]])  # 3->2


def emit_graph(name, nodes, edges, expect_edges):
    g = nir.NIRGraph(nodes=nodes, edges=edges)  # type_check=True (default)
    d = to_jsonable(g.to_dict())
    assert set(d["nodes"]) == set(nodes), f"{name}: node set rewritten!"
    got = sorted(tuple(e) for e in d["edges"])
    assert got == sorted(expect_edges), f"{name}: edges rewritten: {got}"
    dump(name, {"version": "1.0.0", "node": d})


# branch: 1 Input -> (Linear->Linear fused) and (Linear) -> 2 LIF pops
# -> 2 Outputs. Branch 2 (input->l3->lif2) is the plain chain shape;
# branch 1 carries the Linear->Linear chain the assembly fuses.
emit_graph(
    "branch.json",
    {
        "input": nir.Input(input_type=np.array([3])),
        "l1": nir.Linear(weight=POP_W1),                       # 3->2
        "l2": nir.Linear(weight=np.array([[1.0, 0.0], [0.0, -0.5]])),  # 2->2
        "l3": nir.Linear(weight=np.array([[0.25, 0.5, -0.25], [0.5, -0.25, 1.0]])),  # 3->2
        "lif1": nir.LIF(**{k: v for k, v in POP_LIF.items()}),  # pop 2 (frozen values)
        "lif2": nir.LIF(
            tau=np.array([0.01, 0.02]),
            r=np.array([2e8, 1e8]),
            v_leak=np.array([-0.065, -0.07]),
            v_threshold=np.array([-0.05, -0.055]),
            v_reset=np.array([-0.075, -0.08]),
        ),
        "out1": nir.Output(output_type=np.array([2])),
        "out2": nir.Output(output_type=np.array([2])),
    },
    [
        ("input", "l1"), ("l1", "l2"), ("l2", "lif1"),
        ("input", "l3"), ("l3", "lif2"),
        ("lif1", "out1"), ("lif2", "out2"),
    ],
    [
        ("input", "l1"), ("l1", "l2"), ("l2", "lif1"),
        ("input", "l3"), ("l3", "lif2"),
        ("lif1", "out1"), ("lif2", "out2"),
    ],
)

# merge: 2 Inputs -> 2 Linears -> ONE LIF population -> 1 Output.
# Weights 0.25 => q=8192 => x=100 feature current encodes to 81 uA:
# one branch alone stalls below threshold climb (V_ss-rest = 810
# centi-quanta < the 1500 gap), the summed fan-in (162 uA, V_ss-rest
# 1620 >= 1520 climb bound) fires — the gate's sum-fires pin.
MERGE_LIF = dict(
    tau=np.array([0.02, 0.02]),
    r=np.array([1e8, 1e8]),
    v_leak=np.array([-0.07, -0.07]),
    v_threshold=np.array([-0.055, -0.055]),
    v_reset=np.array([-0.08, -0.08]),
)
emit_graph(
    "merge.json",
    {
        "in1": nir.Input(input_type=np.array([2])),
        "in2": nir.Input(input_type=np.array([2])),
        "la": nir.Linear(weight=np.array([[0.25, 0.0], [0.0, 0.25]])),
        "lb": nir.Linear(weight=np.array([[0.25, 0.0], [0.0, 0.25]])),
        "lif": nir.LIF(**MERGE_LIF),
        "out": nir.Output(output_type=np.array([2])),
    },
    [("in1", "la"), ("in2", "lb"), ("la", "lif"), ("lb", "lif"), ("lif", "out")],
    [("in1", "la"), ("in2", "lb"), ("la", "lif"), ("lb", "lif"), ("lif", "out")],
)

# recurrent: 2 LIF populations mutually edged behind an explicit
# Input/Output (cycle tolerated by the reference exactly when an
# Input-rooted path and an Output leaf exist — both present here).
REC_LIF = dict(
    tau=np.array([0.02, 0.02]),
    r=np.array([1e8, 1e8]),
    v_leak=np.array([-0.07, -0.07]),
    v_threshold=np.array([-0.055, -0.055]),
    v_reset=np.array([-0.08, -0.08]),
)
recurrent_graph = nir.NIRGraph(
    nodes={
        "input": nir.Input(input_type=np.array([2])),
        "linear": nir.Linear(weight=np.array([[0.5, 0.0], [0.0, 0.5]])),
        "lif_a": nir.LIF(**REC_LIF),
        "lif_b": nir.LIF(**REC_LIF),
        "output": nir.Output(output_type=np.array([2])),
    },
    edges=[
        ("input", "linear"), ("linear", "lif_a"),
        ("lif_a", "lif_b"), ("lif_b", "lif_a"), ("lif_b", "output"),
    ],
)
recurrent_doc = to_jsonable(recurrent_graph.to_dict())
assert set(recurrent_doc["nodes"]) == {"input", "linear", "lif_a", "lif_b", "output"}
assert sorted(tuple(e) for e in recurrent_doc["edges"]) == sorted(
    [("input", "linear"), ("linear", "lif_a"), ("lif_a", "lif_b"),
     ("lif_b", "lif_a"), ("lif_b", "output")]
)
dump("recurrent.json", {"version": "1.0.0", "node": recurrent_doc})

# HDF5 pair for the merge graph (cross-container leg, R18 P4)
dump_h5("merge.nir", nir.NIRGraph(
    nodes={
        "in1": nir.Input(input_type=np.array([2])),
        "in2": nir.Input(input_type=np.array([2])),
        "la": nir.Linear(weight=np.array([[0.25, 0.0], [0.0, 0.25]])),
        "lb": nir.Linear(weight=np.array([[0.25, 0.0], [0.0, 0.25]])),
        "lif": nir.LIF(**MERGE_LIF),
        "out": nir.Output(output_type=np.array([2])),
    },
    edges=[("in1", "la"), ("in2", "lb"), ("la", "lif"), ("lb", "lif"), ("lif", "out")],
))  # gzip default
dump_h5("merge_uncompressed.nir", nir.NIRGraph(
    nodes={
        "in1": nir.Input(input_type=np.array([2])),
        "in2": nir.Input(input_type=np.array([2])),
        "la": nir.Linear(weight=np.array([[0.25, 0.0], [0.0, 0.25]])),
        "lb": nir.Linear(weight=np.array([[0.25, 0.0], [0.0, 0.25]])),
        "lif": nir.LIF(**MERGE_LIF),
        "out": nir.Output(output_type=np.array([2])),
    },
    edges=[("in1", "la"), ("in2", "lb"), ("la", "lif"), ("lb", "lif"), ("lif", "out")],
), compression=None)

# --- negative: ASSEMBLY-class rejections ----------------------------
# The reference-legal shapes our assembly rejects BY NAME this slice.
# Constructible ones are reference emissions (explicit NIRGraph, type
# check on); the reference's own constructor rejects the rest, so
# those are mutations — same doctrine as the format negatives.

# Input -> Output pass-through: reference-legal (types match).
emit_graph(
    "neg_asm_passthrough.json",
    {"input": nir.Input(input_type=np.array([1])),
     "output": nir.Output(output_type=np.array([1]))},
    [("input", "output")],
    [("input", "output")],
)
# Input -> LIF direct drive: legal NIR, no Linear between.
emit_graph(
    "neg_asm_direct_drive.json",
    {"input": nir.Input(input_type=np.array([1])),
     "lif": nir.LIF(tau=np.array([0.02]), r=np.array([1e8]),
                    v_leak=np.array([-0.07]), v_threshold=np.array([-0.055]),
                    v_reset=np.array([-0.08])),
     "output": nir.Output(output_type=np.array([1]))},
    [("input", "lif"), ("lif", "output")],
    [("input", "lif"), ("lif", "output")],
)
# Input -> Linear -> Output: legal, but nothing fires (encoder-only).
emit_graph(
    "neg_asm_no_lif.json",
    {"input": nir.Input(input_type=np.array([1])),
     "linear": nir.Linear(weight=np.array([[0.5]])),
     "output": nir.Output(output_type=np.array([1]))},
    [("input", "linear"), ("linear", "output")],
    [("input", "linear"), ("linear", "output")],
)
# Input -> LIF -> Linear -> Output: the readout edge (LIF -> Linear).
emit_graph(
    "neg_asm_lif_to_linear.json",
    {"input": nir.Input(input_type=np.array([1])),
     "lif": nir.LIF(tau=np.array([0.02]), r=np.array([1e8]),
                    v_leak=np.array([-0.07]), v_threshold=np.array([-0.055]),
                    v_reset=np.array([-0.08])),
     "linear": nir.Linear(weight=np.array([[0.5]])),
     "output": nir.Output(output_type=np.array([1]))},
    [("input", "lif"), ("lif", "linear"), ("linear", "output")],
    [("input", "lif"), ("lif", "linear"), ("linear", "output")],
)
# LIF self-loop: legal NIR (types trivially match), forbidden by the
# substrate (no self-synapse).
emit_graph(
    "neg_asm_self_loop.json",
    {"input": nir.Input(input_type=np.array([1])),
     "lif": nir.LIF(tau=np.array([0.02]), r=np.array([1e8]),
                    v_leak=np.array([-0.07]), v_threshold=np.array([-0.055]),
                    v_reset=np.array([-0.08])),
     "output": nir.Output(output_type=np.array([1]))},
    [("input", "lif"), ("lif", "lif"), ("lif", "output")],
    [("input", "lif"), ("lif", "lif"), ("lif", "output")],
)
# empty graph: zero nodes, zero edges (infer_types returns early).
emit_graph("neg_asm_empty.json", {}, [], [])

# mid-graph shape mismatch: l1 mutated 2x3 -> 2x4 (cols != Input [3]).
# The reference constructor would reject the type break, so this is a
# mutation of branch.json's emission.
with open(OUT / "branch.json") as f:
    _branch_doc = json.load(f)
_branch_doc["node"]["nodes"]["l1"]["weight"] = [
    [0.5, -1.0, 0.25, 0.125],
    [-0.25, 1.0, 0.5, 0.5],
]
dump("neg_asm_shape_mismatch.json", _branch_doc)

# cycle with no Output leaf: recurrent minus output (+ its edge) —
# the reference constructor rejects it ("No output nodes found").
with open(OUT / "recurrent.json") as f:
    _rec_doc = json.load(f)
del _rec_doc["node"]["nodes"]["output"]
_rec_doc["node"]["edges"] = [e for e in _rec_doc["node"]["edges"] if "output" not in e]
dump("neg_asm_cycle_no_output.json", _rec_doc)

# no Input at all: recurrent minus input+linear (+ their edges) — the
# reference constructor rejects it ("No input nodes found").
del _rec_doc["node"]["nodes"]["input"]
del _rec_doc["node"]["nodes"]["linear"]
_rec_doc["node"]["edges"] = [
    e for e in _rec_doc["node"]["edges"]
    if "input" not in e and "linear" not in e
]
dump("neg_asm_no_input.json", _rec_doc)

print("done — reference:", nir.version if hasattr(nir, "version") else "?")
