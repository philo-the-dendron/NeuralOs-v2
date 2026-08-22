# neuralos-snn

> `no_std`, i16 fixed-point spiking neural networks for edge and RISC-V
> silicon — LIF neurons, pairwise STDP, CSR synapses, ternary weight
> codecs, and an AVX2 batch kernel.

Published on crates.io as `0.1.0-alpha.4`; the tree carries the alpha.5 changes (AGPL-3.0-or-later).

## What this crate is

A spiking-neural-network **substrate**: the neuron/synapse/network core
you can run bare-metal. Integer-only hot path (i16 millivolts, i16
microamps, u32 microseconds) — no floating point, no allocator in the
core, `no_std` by default. Designed for FPU-less edge targets
(ESP32-C3, `HiFive`, QEMU `riscv64gc`) and validated against the IEEE 2025
"Full-Integer SNN Inference with RISC-V ISA" design axis.

## Modules

| Module | What it holds |
|---|---|
| `lif_neuron` | Leaky-Integrate-and-Fire neuron, fixed-point, per-neuron voltage grid (`VoltageResolution`: mV default, opt-in centi-mV), bounded spike history |
| `synapse` | Synapse + pairwise STDP rule (a₊ 50 / a₋ −53 / lr 100), weight scale `SCALE = 1000` |
| `network` *(std)* | `SpikingNeuralNetwork` orchestration (`step()`), CSR `SparseSynapseMatrix` with forward + reverse iteration, 4 topology builders (Random, Small-World, Feedforward, Balanced E/I), plasticity passes (LTD + LTP), per-step stats |
| `trit` | Ternary weight type `{-1, 0, +1}` + scale, ternarizer, stochastic bucket-flip (LFSR, integer-only) |
| `bridge` | `BitNet` `i2_s` encode/decode (bit-exact round-trip), Prism `q1_0`/`q2_0` import + `q2_0` export, integer fp16 widening — layouts pinned from reference sources, loud errors on impossible input |
| `nir` | NIR (Neuromorphic Intermediate Representation) slice 1 — JSON import/export of `Input`/`Linear`/`LIF`/`Output` graphs, explicit per-node quantization records, loud lossiness, byte-stable export. Schema pinned verbatim to the reference implementation (`neuromorphs/NIR` @ `7883c3c`); fixtures are the reference's own emissions (`tools/gen_nir_fixtures.py`) |
| `kernel` | Shared `no_std` ternary matvec: sequential 2-bit packed trits × Q15 activations → i32, absmax normalization, wire→compute repack seam |
| `simd` *(feature)* | AVX2 batch LIF integration (`x86_64`, ~1.6–2.2× vs scalar, ±2 mV tolerance) |

## The voltage grid story

`delta_v = dt_over_tau · (leak + R·I/1000) / 1000` truncates to whole
quanta. On the default mV grid a steady sub-threshold current inside the
~200 μA dead zone moves the membrane exactly zero — recorded, tested,
and the reason `VoltageResolution::CentiMillivolt` exists (100× finer
dead zone, same i16, bit-identical arithmetic shape). The mV default
keeps every historically recorded result bit-exact.

## Features

- `std` *(default)* — enables the `network` orchestration module
- `simd` — implies `std`, x86_64-only AVX2 batch kernel

Without `std` the crate builds `no_std` (neurons, synapses, trit,
bridge, kernel, nir) — the embedded posture CI enforces.

## Usage sketch

```text
use neuralos_snn::{SpikingNeuralNetwork, NetworkTopology};

let mut net = SpikingNeuralNetwork::new_with_voltage_resolution(
    128, 1_000, NetworkTopology::Balanced { excitatory_ratio: 0.8 },
    Default::default(),
)?;
net.build_topology()?;
loop {
    let spikes = net.step(&inputs)?;   // decay → integrate → clear → propagate
    // spikes: Vec<Spike>; plasticity applies pairwise STDP when enabled
}
```

## Anti-scope

No I/O, no persistence, no UI, no drivers, no "OS", no crypto, no LLM
runtime — the library is a library. (The research runtime that proved a
ternary SNN↔LLM bridge on this substrate lives in the workspace's
`neuralos-rt`, `publish = false`.)

## Since alpha.2

- **F1 — adaptation-decay contract pinned**: unit tests pin the exact
  −1/step decay with floor 0 and the +2/spike jump; a 6,000-step live test
  proves a driven net stays firing and adaptation equilibrates. Leak
  convergence pinned too (dt = τ lands on rest exactly, both directions).
- **The coupling knob** — `synaptic_input_divisor` (new API): the recurrent
  pulse is `weight / divisor` μA. Default 10 = the historical pulse
  byte-for-byte; 0 rejected; pinned by default + doubling tests.
- **`network.rs` split into `csr.rs` + `stats.rs`** — every published path
  unchanged, CSR build/equivalence now pinned by dedicated tests
  (unsorted insertion, external adds, reverse-CSR incoming, plasticity
  weight sync).
- **F5a — the DECORATIVE-in-orchestration machinery removed** (alpha semver
  window): `Synapse`'s transmission/eligibility state (`delay_us`,
  `conductance`, `transmission_buffer`, `eligibility_trace`,
  `recent_activity`, synapse-side `last_spike_time_us`) and its dead
  methods (`transmit`, `receive_spike`, `is_active`, `set_delay_us`,
  `reset`), plus `LIFNeuron::tau_synapse_us`. Orchestration never read any
  of it — proven output-neutral by byte-exact re-pins of the bridge
  examples (gate verdict, export sha, 13/13 null patches).
- **Pub-API census**: five module-internal fns de-pubbed
  (`LIFNeuron::new_with_type`, `set_voltage_resolution`, the three
  `i2_s` layout helpers); introspection accessors and builders stay pub —
  see the alpha.3 audit record in the repo's `ISA.md`.
- SIMD gate runs in CI; the batch kernel is documented mV-grid-only.

## Since alpha.4 (the alpha.5 notes)

- **General graph assembly — `NirImport::build_network`**: any
  reference-emitted `Input`/`Linear`/`LIF`/`Output` graph assembles
  onto a real `SpikingNeuralNetwork` and fires. Every LIF
  population becomes neurons (its own quantized params); every
  LIF→LIF edge becomes an `EDGE_PULSE_QUANTA` synapse pair (the
  ratified D1 contract: 200 → a 20 μA pulse at the default divisor
  10, +10 centi-quanta exactly one step later — exact-pinned on
  both grids, the mV grid dead 10× over, which is why recurrent
  graphs reject mV options BY NAME with the copy-pasteable re-import
  remedy); the Linear DAG folds symbolically at setup and quantizes
  ONCE per (drive Linear, root Input) stage (D2 fusion — no
  hop-by-hop i16 composition), merged saturating into the global
  per-step current vector (D5 multi-Input). Plasticity frozen at
  assembly (NIR has no plasticity term). New public surface:
  `build_network`, `NirGraphEncoder`, `NirAssemblyReport`,
  `LinearFusedRecord`, `EDGE_PULSE_QUANTA`, and
  `NirError::EdgeShapeMismatch` (per-edge reference type-check
  parity). Every out-of-slice shape is a NAMED rejection (readout,
  direct drive, encoder-only, self-loop, Output-as-source, Linear
  cycles, pass-through, empty/no-Input/no-Output, the D7
  population bound); Input-unreachable structure assembles with
  structural `UndrivenPopulation` notes — silence documented, never
  silent. The frozen chain is proven bit-exact through both
  builders (`nir_assembly_gate`, 6/6, evidence of record).
- **Honest-claim language of record** (assembly): structure,
  per-edge type shapes, and quantization records are exact w.r.t.
  the pinned document; dynamics are named substrate conventions
  (`EDGE_PULSE_QUANTA`, divisor, grid, one-step delay, frozen
  plasticity) pinned by exact-value tests; the reference defines
  no execution semantics at this sha.
- **The banked consolidation breaks ride here** (2026-08-22): the
  `NeuronBuilder`/`SynapseBuilder` types and the setter surface
  deleted, the test-only introspection quartet relocated under
  `#[cfg(test)]`, `LIFNeuron::spikes()` deleted (zero callers),
  `Synapse::normalized_weight` test-relocated — API breaks with no
  known consumers.

## Since alpha.3 (the alpha.4 notes — PUBLISHED 2026-08-22)

- **NIR slice 1 shipped *in* alpha.3** (missing from the notes
  above): `neuralos_snn::nir` — JSON import/export of
  `Input`/`Linear`/`LIF`/`Output` graphs, schema pinned to
  `neuromorphs/NIR@7883c3c`, reference-emitted fixtures +
  `nir_format_gate`, explicit per-node quantization records.
- **Fresh-eyes review fixes (findings R1-R8)**: `skip_value`
  depth-capped at 64 (adversarial nesting is a loud `Json` error,
  not a stack overflow); trailing content after the root rejected
  (Python `json.loads` parity); denormal-`absmax` weights (e.g.
  `5e-324`) are a loud `BadNumber` — the underflowed `scale = 0`
  would have zeroed exports silently and broken idempotence;
  `round_half_away` rewritten as truncate-compare (the classic
  add-±0.5 idiom misrounds values 1 ulp below a half — reachable
  via `r` in MΩ: 499999.99999999994 Ω imported as 1 MΩ);
  `ChainEncoder::encode` accumulates in i64 (i32 silently wrapped
  negative at cols ≥ 3 with full-scale weights); scan rejects
  1-D/empty/3-D weight arrays up front; export rejects dangling
  edge indices (no `"?"` placeholders).
- **Structured-entry seam (post-review)**: `quantize_linear` and
  `quantize_lif` are public — callers holding materialized f64
  values (HDF5 import, builders) quantize without a JSON document,
  the same contract and errors, arena placement included. The 4×
  arena-scratch trick is gone: import stages source weights in a
  typed `NirBuffers::scratch` f64 slice; arena and scratch each
  hold the weight-cell count exactly. Breaking buffer-API change,
  alpha.4-bound.
- **Per-neuron LIF populations + `NirBuilder`**: a LIF node is the
  reference's population — per-neuron param arrays quantize to
  per-neuron records in a `lifs` buffer (`NirLifPopulation` views);
  assembly enforces Linear rows == population size (reference
  type-check parity), one neuron per row with its own params.
  `NirBuilder` (std) assembles graphs in memory over the quantizer
  seam; export renders per-neuron arrays; the printable-ASCII gate
  is a JSON-container property only (new
  `NonAsciiNodeName` at export, never on the typed surface).
  Reference-emitted `chain_population.json` fixture pins the
  expansion end-to-end (imports → assembles → fires → re-imports
  state-identical).
- **Honesty note**: findings R1-R5 were present in the published
  alpha.3 binary. No consumers are known (the module shipped within
  the day); all are fixed here, ahead of any consumer, targeted
  for alpha.4.
- **The HDF5 `.nir` container** ships workspace-side, in
  `neuralos-rt` behind its `hdf5` feature (vendored static HDF5,
  pre-read filter census, the reference's own fixtures +
  `nir_hdf5_gate`) — the seam it feeds is the structured entry above.

## Status

**`0.1.0-alpha.4` is live on crates.io** (published 2026-08-22): NIR
structured entry (pub quantizers, `NirBuilder`, per-neuron
populations) + the R9 review fixes. Alpha.5 adds the general
four-kind graph assembly (see "Since alpha.4"). The alpha.3 record: the
adaptation-decay contract pinned by unit + live tests (equilibrates,
never silences the net); `synaptic_input_divisor` — **the coupling
knob**, new public API (default 10 = the historical weight/10 pulse;
0 rejected); `network.rs` split into `csr.rs` + `stats.rs` with every
published path unchanged; the simd batch kernel doc'd mV-grid-only.

**The tree is ahead of the published crate** (rides `alpha.5`): the
general assembly above + the consolidation breaks (builders deleted,
introspection relocated). 306 offline unit/property tests (3 app,
199 snn, 104 rt) + 208 simd-gated + 121 hdf5-gated (in
`neuralos-rt`) + 5 model-gated `#[ignore]`; the API may still move
within alpha semver.

NIR itself: Pedersen et al., Nature Communications 15, 4962 (2024),
DOI 10.1038/s41467-024-52259-9 — this crate's `nir` module speaks
the schema pinned to `neuromorphs/NIR@7883c3c`.

## License

AGPL-3.0-or-later — see the workspace root `LICENSE`.
