# NIR fixtures

Every `.json` here but the two `*_sim.json` is a reference emission, written by `tools/gen_nir_fixtures.py` from the pinned reference (`nir-ref/` @ `7883c3c`), whose header says how each positive and negative file is made.
The two `*_sim.json` are the witnesses of D8, the spiking Linear edge (`src/nir.rs`, `LINEAR_GAIN_DIVISOR`): `neuralos-nir2json --sim-units` output, the converter's own bytes (its corpus test holds them to it).
`two_lif_neurons_sim.json` from `crates/neuralos-nir2json/tests/fixtures/community/two_lif_neurons.nir`, sha `806c7c1cfae72b5be92ce15a670f37014b82a5d4fba8a077efce46d193c21a82`: the NIR paper's own file (that directory's PROVENANCE.md).
`snntorch_two_layer_sim.json` from `crates/neuralos-nir2json/tests/fixtures/community/snnTorch_two_layer.nir`, sha `3f1834a74f442c582336a84449d9089c5f2b85293b694fdda5443ed218ef1942`: snnTorch 1.0's own export (PROVENANCE.md, the fallback emission).
