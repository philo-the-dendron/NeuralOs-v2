# NIR HDF5 gate — slice 2 evidence (2026-08-21)

The evidence gate of the HDF5 milestone: the pinned reference's own
`.nir` emissions, read end-to-end in pure Rust, fired on the
substrate, censused, exported, and read back — plus the interop leg:
the reference's own `nir.read()` loads OUR export.

## Files

| File | What |
|---|---|
| `gate.log` | `nir_hdf5_gate` full run — 5/5 gates |
| `verify.log` | reference-side `--verify` of our export |
| `gate_export.nir` | our HDF5 export (dumped by the gate) |
| `SHA256SUMS` | fixture + export pins |

## Rebuild (all in-repo, no /tmp)

```bash
# one-time env: .nirenv (numpy+h5py+cmake + pinned clone installed)
python3 -m venv .nirenv && .nirenv/bin/pip install numpy h5py cmake ./nir-ref

# fixtures (regenerates JSON 15 byte-identical + the 3 .nir files)
.nirenv/bin/python3 tools/gen_nir_fixtures.py

# the gate (vendored HDF5 needs cmake from .nirenv)
PATH="$PWD/.nirenv/bin:$PATH" \
  NEURALOS_NIR_HDF5_OUT=evidence/nir-hdf5-gate/gate_export.nir \
  cargo run -p neuralos-rt --features hdf5 --example nir_hdf5_gate \
  > evidence/nir-hdf5-gate/gate.log 2>&1

# the interop leg — the reference's own read() over our export
.nirenv/bin/python3 tools/gen_nir_fixtures.py --verify \
  evidence/nir-hdf5-gate/gate_export.nir
```

## Verdicts of record

- gate 1: reference emission quantizes exactly, cross-container
  (LIF (20k,−70,−55,−80,100MΩ,200pF)/(30k,−65,−50,−75,200MΩ,150pF);
  weights [16384,−32767,8192,−8192,32767,16384])
- gate 2: 9 spikes / 100 steps, first at step 6 — identical to the
  JSON format gate's frozen verdict (same source values, same drive)
- gate 3: lzf censused out before any read (dataset
  `node/nodes/input/shape`, filter named, policy stated)
- gate 4: export read-back semantically identical (idempotence =
  record equality — the named decision)
- gate 5: JSON export byte-stable (941 B) + state-identical re-import
- verify: reference loads our export; weights within the quantizer
  half-step bound (1.526e-05 ≤ scale/2 = 1.526e-05), LIF params
  within record-render tolerance

## alpha.4 dry-run (verbatim, from commit ff0be22)

```
     Updating crates.io index
    Packaging neuralos-snn v0.1.0-alpha.4 (/home/student/projets/NeuralOs-v2/crates/neuralos-snn)
     Updating crates.io index
    Packaged 43 files, 488.9KiB (128.5KiB compressed)
   Verifying neuralos-snn v0.1.0-alpha.4 (/home/student/projets/NeuralOs-v2/crates/neuralos-snn)
   Compiling neuralos-snn v0.1.0-alpha.4 (/home/student/projets/NeuralOs-v2/target/package/neuralos-snn-0.1.0-alpha.4)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 2.99s
   Uploading neuralos-snn v0.1.0-alpha.4 (/home/student/projets/NeuralOs-v2/crates/neuralos-snn)
warning: aborting upload due to dry run
```

`publish_dryrun_alpha4.log` is the raw capture. The real publish is
the principal's call.
