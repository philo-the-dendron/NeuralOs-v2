# neuralos-nir2json

The inbound bridge: a stranger's NIR `.nir` file (HDF5) converted into
the JSON schema [`neuralos-snn`]'s `nir_import` consumes — **pure
Rust, no C toolchain** (`hdf5-pure`; dependencies reduce to byteorder
+ miniz_oxide via flate2's rust backend).

```
neuralos-nir2json <input.nir> <output.json>
```

Exit codes: `0` converted (a sidecar `<output>.meta.json` carries the
audit stamp) · `1` usage/IO · `2` named refusal.

Single-writer by construction: HDF5 → typed values → snn's own
`NirBuilder` (the quantizer) → snn's own `nir_export` (the schema
writer). The tool never writes JSON itself; there is no second
implementation to drift.

## Install

```bash
cargo install --git https://gitea.com/Caramoussin/NeuralOs-v2
```

or grab a prebuilt static binary from the releases (linux-x86_64).

## What converts, what refuses — and why

- **Node kinds:** `Input`, `LIF`, `Linear`, `Output` convert. Anything
  else (e.g. `Affine`, `Conv`, RNN blocks) is refused **loudly with the
  node's name and kind** — a recorded result, never a partial file.
- **Filters:** none or gzip (deflate) — the reference emission
  conventions. `lzf`, `szip`, anything else: refused by name before a
  single byte decodes (a filter we cannot decode is a silent-corruption
  hazard).
- **float32:** stranger files (snnTorch exports default to fp32) are
  widened to f64 bit-exactly; every widened dataset is listed in the
  sidecar's `f32_widened` array. No durability illusion: snn re-exports
  drop the stamp by construction.
- **Parameter conventions (the honest wall + the exact bridge):** the
  substrate quantizes **biological-scale** LIF parameters — membrane
  potentials on the mV grid within **[−100, +50] mV**, `r ∈ [1, 65535]
  MΩ`. The wider ecosystem's mainstream exports (snnTorch, norse,
  rockpool defaults) carry the **simulation-unit convention**:
  `r ≈ 1–24 Ω`, dimensionless voltages — and the wall is **double**:
  `r` refuses first, the voltages (e.g. `v_th 0.1` read as 0.1 V =
  100 mV) right behind it. snnTorch's LIF is dimensionless
  (β/threshold, no R), so no snnTorch export carries a biological `r`.
  **`--sim-units`** is the exact, stamped bridge: `r × 1000 → MΩ`,
  voltages read as mV, **centi grid forced** (without it the
  0.1-threshold family dies at ThresholdZero — 0.1 mV rounds to 0 on
  the default grid). The substrate couples only the product `r·I`, so
  dynamics are preserved when you drive your dimensionless currents as
  µA numerics (see the sidecar stamp). The transform is opt-in and
  sidecar-stamped — an interpretive act is never silent, and nothing
  is ever auto-rescaled. The flag is **all-or-nothing per file**:
  mixed-convention graphs (bio `r ≥ 1 MΩ` beside sim `r`) transform
  every `r` of the node — and refuse loudly via the r ceiling (a bio
  5 MΩ becomes 5×10⁹ MΩ > 65,535), never silent corruption.
  Linear-only graphs (encoders, readout heads) convert cleanly from
  any emitter, no flag needed.

## Development

```bash
cargo test -p neuralos-nir2json   # corpus v2: fixtures live in tests/fixtures/
```

Fixture provenance (the stranger files, sha-pinned): see
`tests/fixtures/community/PROVENANCE.md`. Regenerate the derived
fixtures (f32 twins, big graph): `.nirenv/bin/python3
tools/gen_nir2json_fixtures.py`. Regenerate the snnTorch fallback
emission: `tools/gen_snnTorch_stranger.py` (throwaway venv; the script
header carries the exact stack).

[`neuralos-snn`]: https://crates.io/crates/neuralos-snn
