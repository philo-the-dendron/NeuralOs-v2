#!/usr/bin/env python3
"""Generate the neuralos-nir2json corpus-v2 fixtures (deterministic).

Inputs : existing rt fixtures (chain_population.nir, merge.nir,
         neg_filter_lzl.nir → REUSED for the lzf-refusal test) — the
         f32 twins cast their F64 datasets down; the big graph is
         synthesized with a fixed integer formula (no RNG anywhere).
Output : crates/neuralos-nir2json/tests/fixtures/
         - chain_population_f32.nir, merge_f32.nir   (twins)
         - big_linear.nir                            (1M-weight blowup probe)
Run    : .nirenv/bin/python3 tools/gen_nir2json_fixtures.py

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


if __name__ == "__main__":
    twin("chain_population.nir")
    twin("merge.nir")
    big_linear()
    print("done — community fixtures are copied, not generated (PROVENANCE.md)")
