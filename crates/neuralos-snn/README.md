# neuralos-snn

> `no_std`, i16 fixed-point spiking neural networks for edge and RISC-V
> silicon — LIF neurons, pairwise STDP, CSR synapses, ternary weight
> codecs, and an AVX2 batch kernel.

Published on crates.io as `0.1.0-alpha.3` (AGPL-3.0-or-later).

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

## Status

`0.1.0-alpha.3` — the substrate-hardening release: the
adaptation-decay contract pinned by unit + live tests (equilibrates,
never silences the net); `synaptic_input_divisor` — **the coupling
knob**, new public API (default 10 = the historical weight/10 pulse;
0 rejected); `network.rs` split into `csr.rs` + `stats.rs` with every
published path unchanged; the simd batch kernel doc'd mV-grid-only.
160 offline unit/property tests + 3 simd-gated; the API may still
move within alpha semver.

## License

AGPL-3.0-or-later — see the workspace root `LICENSE`.
