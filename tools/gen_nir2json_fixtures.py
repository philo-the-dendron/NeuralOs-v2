#!/usr/bin/env python3
"""Generate the neuralos-nir2json corpus-v2 fixtures (deterministic).

Inputs : existing rt fixtures (chain_population.nir, merge.nir,
         neg_filter_lzl.nir → REUSED for the lzf-refusal test) — the
         f32 twins cast their F64 datasets down; the big graph is
         synthesized with a fixed integer formula (no RNG anywhere).
Output : crates/neuralos-nir2json/tests/fixtures/
         - chain_population_f32.nir, merge_f32.nir   (twins)
         - big_linear.nir                            (1M-weight blowup probe)
         - affine_two_layer.nir                      (two biased Affines)
         - affine_bias_length.nir                    (a bias of the wrong length)
         - affine_dangling.nir                       (an Affine that feeds nothing)
         - weight_1d.nir                             (a Linear's weight of one dimension)
Run    : .nirenv/bin/python3 tools/gen_nir2json_fixtures.py

The three Affine graphs are the layout snnTorch writes for a biased
nn.Linear, float32 as torch's, with OUR values and no RNG:
input(2) → a (Affine 3×2) → l1 (LIF 3) → c (Affine 2×3) → l2 (LIF 2) →
output, the LIFs in simulation units (β 0.9 at 0.1 ms: τ 1 ms, r 10,
threshold 1). The second file gives c a bias of three values for its
two rows; the third adds input → d (Affine 1×2), which feeds nothing.

The community (.nir stranger) fixtures are NOT generated here — they
are copied verbatim from the pinned clone with provenance (see
tests/fixtures/community/PROVENANCE.md). Anti-circularity: emissions
we didn't cause are the entire point of that leg.
"""
import pathlib

import h5py

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "crates/neuralos-rt/tests/nir_fixtures"
OUT = ROOT / "crates/neuralos-nir2json/tests/fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def copy_tree_f32(src: h5py.Group, dst: h5py.Group) -> None:
    """Recursively mirror the NIR layout, casting F64 datasets to F32."""
    for key in src:
        obj = src[key]
        if isinstance(obj, h5py.Group):
            copy_tree_f32(obj, dst.create_group(key))
        elif h5py.check_dtype(vlen=obj.dtype) or obj.dtype.kind == "O":
            dst.create_dataset(key, data=obj[()], dtype=h5py.string_dtype(encoding="utf-8"))
        elif obj.dtype == "float64":
            dst.create_dataset(key, data=obj[()].astype("float32"), dtype="float32", compression="gzip")
        else:
            dst.create_dataset(key, data=obj[()], compression="gzip")


def twin(name: str) -> None:
    src = SRC / name
    dst = OUT / name.replace(".nir", "_f32.nir")
    with h5py.File(src, "r") as a, h5py.File(dst, "w") as b:
        copy_tree_f32(a, b)
    print(f"twin   : {dst.name} ({dst.stat().st_size} B)")


def big_linear(n: int = 1024) -> None:
    dst = OUT / "big_linear.nir"
    with h5py.File(dst, "w") as f:
        f.create_dataset("version", data="1.0.9.dev1+g7883c3c85", dtype=h5py.string_dtype(encoding="utf-8"))
        node = f.create_group("node")
        node.create_dataset("type", data="NIRGraph", dtype=h5py.string_dtype(encoding="utf-8"))
        nodes = node.create_group("nodes")
        inp = nodes.create_group("input")
        inp.create_dataset("type", data="Input", dtype=h5py.string_dtype(encoding="utf-8"))
        inp.create_dataset("shape", data=[n], compression="gzip")
        lin = nodes.create_group("linear")
        lin.create_dataset("type", data="Linear", dtype=h5py.string_dtype(encoding="utf-8"))
        # deterministic |w| ≤ 1, no RNG
        w = [[((i * 7919 + j * 104729) % 2001 - 1000) / 1000.0 for j in range(n)] for i in range(n)]
        lin.create_dataset("weight", data=w, dtype="float64", compression="gzip")
        out = nodes.create_group("output")
        out.create_dataset("type", data="Output", dtype=h5py.string_dtype(encoding="utf-8"))
        out.create_dataset("shape", data=[n], compression="gzip")
        node.create_dataset(
            "edges",
            data=[["input", "linear"], ["linear", "output"]],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    print(f"big    : {dst.name} ({dst.stat().st_size} B, {n}×{n} weights)")


def affine(name: str, c_bias: list, dangling: bool = False) -> None:
    dst = OUT / name
    s = h5py.string_dtype(encoding="utf-8")

    def f32(group, key, data):
        group.create_dataset(key, data=data, dtype="float32", compression="gzip")

    with h5py.File(dst, "w") as f:
        f.create_dataset("version", data="1.0.9.dev1+g7883c3c85", dtype=s)
        node = f.create_group("node")
        node.create_dataset("type", data="NIRGraph", dtype=s)
        nodes = node.create_group("nodes")
        for key, kind in (("input", "Input"), ("output", "Output")):
            g = nodes.create_group(key)
            g.create_dataset("type", data=kind, dtype=s)
            g.create_dataset("shape", data=[2], compression="gzip")
        affines = [
            ("a", [[0.5, -0.25], [0.125, 0.75], [-0.5, 0.25]], [0.05, -0.02, 0.0]),
            ("c", [[0.3, -0.2, 0.1], [0.05, 0.4, -0.15]], c_bias),
        ]
        edges = [["input", "a"], ["a", "l1"], ["l1", "c"], ["c", "l2"], ["l2", "output"]]
        if dangling:  # input → d, and d feeds nothing
            affines.append(("d", [[0.3, 0.3]], [0.04]))
            edges.append(["input", "d"])
        for key, weight, bias in affines:
            g = nodes.create_group(key)
            g.create_dataset("type", data="Affine", dtype=s)
            f32(g, "weight", weight)
            f32(g, "bias", bias)
        for key, n in (("l1", 3), ("l2", 2)):
            g = nodes.create_group(key)
            g.create_dataset("type", data="LIF", dtype=s)
            f32(g, "tau", [1e-3] * n)
            f32(g, "r", [10.0] * n)
            f32(g, "v_leak", [0.0] * n)
            f32(g, "v_threshold", [1.0] * n)
            f32(g, "v_reset", [0.0] * n)
        node.create_dataset("edges", data=edges, dtype=s)
    print(f"affine : {dst.name} ({dst.stat().st_size} B)")


def weight_1d() -> None:
    """input(2) → linear → output, the Linear's weight of one dimension."""
    dst = OUT / "weight_1d.nir"
    s = h5py.string_dtype(encoding="utf-8")
    with h5py.File(dst, "w") as f:
        f.create_dataset("version", data="1.0.9.dev1+g7883c3c85", dtype=s)
        node = f.create_group("node")
        node.create_dataset("type", data="NIRGraph", dtype=s)
        nodes = node.create_group("nodes")
        for key, kind in (("input", "Input"), ("output", "Output")):
            g = nodes.create_group(key)
            g.create_dataset("type", data=kind, dtype=s)
            g.create_dataset("shape", data=[2], compression="gzip")
        g = nodes.create_group("linear")
        g.create_dataset("type", data="Linear", dtype=s)
        g.create_dataset("weight", data=[0.5, 0.25], dtype="float64")
        node.create_dataset("edges", data=[["input", "linear"], ["linear", "output"]], dtype=s)
    print(f"1-D    : {dst.name} ({dst.stat().st_size} B)")


if __name__ == "__main__":
    twin("chain_population.nir")
    twin("merge.nir")
    big_linear()
    affine("affine_two_layer.nir", [0.03, -0.01])
    affine("affine_bias_length.nir", [0.03, -0.01, 0.02])
    affine("affine_dangling.nir", [0.03, -0.01], dangling=True)
    weight_1d()
    print("done — community fixtures are copied, not generated (PROVENANCE.md)")
