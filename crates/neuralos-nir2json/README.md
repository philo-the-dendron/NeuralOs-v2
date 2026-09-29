# neuralos-nir2json

The inbound bridge: a stranger's NIR `.nir` file (HDF5) converted into
the JSON schema [`neuralos-snn`]'s `nir_import` consumes — **pure
Rust, no C toolchain** (`hdf5-pure` and what its `std` and `deflate`
features bring, flate2's rust backend among them;
`cargo tree -p neuralos-nir2json -e normal` lists them).

```
neuralos-nir2json [--sim-units] [--dt µs] [--freeze <out.rs> [--steps N] [--input v1,v2,…]] <input.nir> <output.json>
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
  **`--sim-units`** is the stamped bridge. Each LIF node gets a voltage
  scale, `V = min(10 / largest |threshold|, 45 / largest |potential|)`
  mV per unit, on the **centi grid**: the node's largest threshold is
  then 1,000 quanta, fewer only where a potential past 4.5 times it
  lowers `V`, its others in proportion, and the −100 mV floor ten times
  it or more down; the node's `r` becomes `r · V` MΩ. Linear weights
  keep their true scale in the library (`NirUnits::Simulation`, 1,000
  current quanta per unit), so the product `r·I` keeps the source's
  scale. The import follows NIR's LIF: it fires on `v > v_threshold`
  and has no refractory period. The step is 0.1 ms unless `--dt` gives
  another: snnTorch's exporter assumes it, and a `.nir` file carries
  none. The transform is opt-in and sidecar-stamped, each node's `V`
  with it: an interpretive act is never silent. What does not fit is
  refused by name: a weight past ±32,767 quanta at true scale, an `r`
  that `V` takes past 65,535 MΩ, a `tau` past 1,000 steps. Limits,
  named: `r` is whole MΩ, so a node whose `r · V` is near 1 rounds
  coarsely; the current a neuron receives in one step, from the input
  or from spikes, is not refused at import and saturates past 32.767
  units; a membrane driven below the −100 mV floor stops there, so a
  strongly inhibited neuron recovers sooner than snnTorch's; each step
  moves a membrane by whole quanta, so it stops up to about τ/dt quanta
  short of the value it settles to: a neuron that settles just above
  its threshold can stay silent, and one that nears it slowly fires
  later than in floating point; the substrate's spike-frequency
  adaptation stays on, which slows a neuron that fires at a high rate;
  and snnTorch's default reset, subtraction, does not survive NIR (its
  exporter writes `v_reset = 0` for every model), so a model runs as
  trained only with `reset_mechanism="zero"`. The flag is
  **all-or-nothing per file**: mixed-convention graphs (bio
  `r ≥ 1 MΩ` beside sim `r`) transform every `r` of the node — and
  refuse loudly via the r ceiling (a bio 5 MΩ becomes 5×10⁶·V MΩ),
  never silent corruption.
  Linear-only graphs (encoders, readout heads) convert cleanly from
  any emitter, no flag needed.

## Freeze: the arrays a `FixedNetwork` steps

```
neuralos-nir2json [--sim-units] [--dt µs] --freeze <out.rs> [--steps N] [--input v1,v2,…] <input.nir> <output.json>
```

After the conversion, `--freeze` builds the network the JSON describes
(the library's `NirImport::from_json` under the conversion's own
options, then `build_network`) and writes, beside the JSON and its
sidecar:

- **`<out.rs>`**, one `pub mod` named after the file's stem (lowercased,
  anything but a letter, digit or `_` made `_`; a letter or `_` first,
  not a Rust keyword): `N` neurons and `S`
  synapses as the arrays a `neuralos_snn::FixedNetwork<N, S>` steps,
  the time step, the run, the trace's header line (`kind=stranger`),
  and `DRIVE`, one run of `--steps` steps (default 150) of the currents
  the graph's own encoder gives for `--input`: one integer per input
  feature, in Input order, default 1 for every feature. The library's
  freezer writes it (`neuralos_snn::fixed::freeze::module`), the one
  that writes the library's own frozen traces.
- **`<out>.trace`**, that run on the host in `neuralos-trace v1`: the
  header line, then one row per step, the spikes and every membrane,
  written by `neuralos_snn::trace::row`, the row writer the firmware
  uses. The host steps the library's std network with plasticity off,
  the network `FixedNetwork::try_from` converts; the two step alike,
  bit for bit, on every network it converts (the library's trace
  tests), and this crate's test builds a module and steps it to the
  same rows.

Refused by name, exit 2, nothing written: a graph that does not
assemble (the library names why, e.g. a readout to Output or a graph
with no LIF), plasticity on (never, from NIR), more than 65,535 neurons
(a neuron id is a `u16`). A `--input` of the wrong length, a `--steps`
of 0, or a stem that gives no module name is a usage error, exit 1.

What the module is for: the ESP32-C3 firmware's slot for a stranger's
graph, filled by one command, `firmware/esp32c3/stranger.sh` (its
README; `docs/ROADMAP.md` § 0.1.0, check 7): the board steps the
module's arrays, and its rows must equal `<out>.trace`. The
module needs only `core` and `neuralos-snn`, so it compiles `no_std`.

## Build (the release artifact)

The prebuilt binary on the Gitea release is this, nothing more:

```bash
rustup target add x86_64-unknown-linux-musl          # once
crates/neuralos-nir2json/build.sh                     # → dist/neuralos-nir2json-v0.1.0-x86_64-unknown-linux-musl
```

The script is the bare cargo line (`cargo build -p neuralos-nir2json
--release --locked --target x86_64-unknown-linux-musl`) plus what the
bring-up binary lacked: rustc's path remap for the four source roots,
so no home path or registry path of the builder reaches the binary
(the published bring-up binary carries 38, left standing by ruling,
ISA rounds 23 and 26); a gate that refuses the artifact on any hit;
the `.text` sha; and the trim-paths canary. `tools/remap.sh` carries
the why, one header. Static-pie, statically linked, no libc dependency
(`ldd` says so). The musl target defaults to `crt-static`, so no extra
flags. `dist/` is gitignored: the binary lives on the release, its pin
lives in the record (`docs/releases/`, ISA round-20); the script never
overwrites a different binary under the same name. At the tree that
record names, under that record's toolchain (1.92.0), the sha is stable
across clean builds; with the remap it no longer depends on the clone
path either. A rebuild on another tree or toolchain gives another
sha, which is not a finding; the `.text` sha is what a rebuild is
compared against.

Verify a binary by execution, both exits:

```bash
B=crates/neuralos-nir2json/dist/neuralos-nir2json-v0.1.0-x86_64-unknown-linux-musl
$B crates/neuralos-nir2json/tests/fixtures/chain_population_f32.nir /tmp/chain.json   # exit 0, sidecar written
$B crates/neuralos-nir2json/tests/fixtures/neg_filter_lzf.nir /tmp/lzf.json           # exit 2, lzf refused by name
```

## Development

```bash
cargo test -p neuralos-nir2json   # corpus v2: fixtures live in tests/fixtures/
```

Fixture provenance (the stranger files, sha-pinned): see
`tests/fixtures/community/PROVENANCE.md`. Regenerate the derived
fixtures (f32 twins, big graph): `.nirenv/bin/python3
tools/gen_nir2json_fixtures.py`. Regenerate the snnTorch fallback
emission: `tools/gen_snnTorch_stranger.py` (throwaway venv; the script
header carries the exact stack), and the two-layer witness of D8:
`tools/gen_snnTorch_two_layer.py` (the repo's `.nirenv`; not
byte-stable across runs, so the committed emission is pinned by its
sha in PROVENANCE.md).

[`neuralos-snn`]: https://crates.io/crates/neuralos-snn
