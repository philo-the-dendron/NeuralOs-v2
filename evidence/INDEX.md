# Evidence Index

> Raw judge/experiment logs backing the bridge record. Verdicts and
> decisions live in `ISA.md`; the distilled narrative in
> `docs/RESEARCH_LOG.md`; the article in `paper/`. The judge is
> rebuildable: `bash tools/build_fork.sh` (pin `9ca265a`, dump patch
> in `tools/neuralos_dump.patch`).

| Directory / file | Session | Claim it backs | Key files |
|---|---|---|---|
| `corpus_readme_pinned.txt` | E stage 0 | The sha-pinned KLD corpus (12,148 bytes / 4,411 tokens; first 2,000 driven) | — |
| `session-f-judge/` | F (2026-08-19) | Loop closure on the live-wire substrate: 60/60 judged steps moved, max \|Δ\| 0.42, 0/60 argmax flips, continuations byte-identical; run1 == run2 | `p*_run{1,2}.{log,err}` (dumps = `NEURALOS_DUMP` lines), `README.md`, `SHA256SUMS` |
| `session-h-invivo/` | H (2026-08-19) | In-vivo gate: T1 PASS (G0 tables, clamp caveat 69.8%, class-dissolution finding); export 67,309 cells / 40,126 code bytes; the p3 continuation change, double-run | `invivo.log`, `invivo_export.log`, `p3_run{1,2}.{log,err}` |
| `session-h2/` | H2 (2026-08-19) | Corrected-corpus rerun: P1' PASS, P3' split (1/12 argmax flip), 2nd continuation change | per-prompt run pairs, `README.md`, `SHA256SUMS` |
| `session-i-primary/` | I (2026-08-20) | **The adjudication.** Dose-matched nulls ×10 + flow-shuffle ×3, all S2-clean. Conjunct: (a) FAIL 0.0798 < 0.2254 · (b) FAIL 8/10 ≫ 1/10 · (c) moot · (d) MIXED band → **BRANCH B: unattributed perturbation** (ISC-84) | `null-d1..d10/`, `null-f1..f3/`, `README.md` (the ruling table), `SHA256SUMS` |
| `session-i-stress/` | I (2026-08-20) | Stress arm (report-only): ~2× dose nulls are LOUD — p3 9/10, p4 9/10 (new flip family) — the p3 knife-edge is fragile to perturbation magnitude | `null-r1..r10/`, `README.md`, `SHA256SUMS` |
| `r4-baselines/` | R4(i) (2026-08-20) | Pre-refactor re-pin baselines for the frozen example family (the R4(iv) contract); judge p0 leg closed by r4-closeout | `README.md` (the protocol + verdict table), `*_run{1,2}.log` |
| `r4-closeout/` | R4(iii/iv) (2026-08-21) | **R4 closed.** Leg-3 re-pins: null_patches 13/13 byte-identical; judge p0 0/12 flips max \|Δ\| +0.4207 exact; stale H1 invivo bar root-caused; H2 re-pin **byte-identical** (21,065.6 s, export tier by-design not run) | `README.md`, `h2_invivo_r4iii.log`, `p0_{base,loop}_run{1,2}.{log,err}`, `SHA256SUMS`, `null_*`, `h1_invivo_r4iii.log` |
| `nir-hdf5-gate/` | NIR slice 2 (2026-08-21) | **The HDF5 evidence gate.** Reference-written `.nir` read end-to-end in pure Rust (5/5: exact frozen quanta cross-container · fires 9/100@6 · lzf censused out named · export read-back semantically identical · JSON byte-stability untouched); interop leg: the reference's own `read()` loads our export (weights ≤ scale/2) | `README.md` (rebuild), `gate.log`, `verify.log`, `SHA256SUMS` |
| `nir-assembly-gate/` | NIR general assembly (2026-08-22) | **The general four-kind graph gate.** Any reference-emitted Input/Linear/LIF/Output graph assembles and fires (6/6: branch first-spike pins `[3,14,1,3]` · merge summed-fan-in fires where single stalls (step 52 exact) · recurrent D1 pulse assert (−6990 = +10 quanta) · L→L fusion = once-quantized f64 product · 10 named rejections · frozen chain byte-identical through both builders); dynamics are named substrate conventions, no numeric-parity claim | `README.md` (rebuild), `gate.log`, `SHA256SUMS` |
| `qemu-riscv-gate/` | QEMU proof (2026-08-21) | **The riscv64gc no_std proof, both postures.** Leg A bare-metal none-elf: 175/175 cited checks, exit 0 (harness: `proofs/qemu-riscv-leg-a/`). Leg B user-mode musl: the REAL full suite on riscv64gc — 195/195 incl. the transmission trio + both Leg-C pins, exit 0, ~14 s | `README.md` (corrected pre-flight + rebuild), `leg-a.log`, `leg-b.log`, `SHA256SUMS` |
| `step5-readout/burn/` | Step-5 readout benchmark (2026-08-24→27) | **RAW EVIDENCE — ADJUDICATED 2026-08-27, PRE-REGISTRATION-UNDEFINED (ruling in ISA.md; cite that, not these logs).** The follow-on readout benchmark named in the paper's limitations: calibration gate PASS 8/8, DOMAIN-CORRECTED covariate arm, ON×5 + OFF driven + 2 identity tripwires + 50 dose-matched nulls + 3 FREE checkpoints. Mechanical aggregator printed `1/5 SEPARATED · 2 MIXED` (n=5, escalation spent); the trailing escalation prompt is a stale n=3 fallback string, not a ruling. Window overlap (r4/r0 79%) disclosed pre-burn | `README.md` (what ran + the two halts), `logs/leg-*.log`, per-dir `SHA256SUMS` (62 families clean) |

Seeds of record (pinned as constants in the frozen examples): census-
matched control `0x5EED_C0DE_0000_0002` (Fisher–Yates); primary dose
nulls `0xD05E_0000_0000_0001 XOR seed`, seed = 1..10.

Mechanical summary line of record: 1.7B-Q1_0 **NO 3/5** · 4B-Q1_0
**NO 4/5** · 4B-Q2_0 **YES 5/5** — all fork-attributed; hybrid seam
**ADAPTS**; loop **CLOSED** as capability; final downstream ruling
**unattributed perturbation**.

## Models manifest (provenance table, 2026-08-22)

Every file in `models/` (gitignored by design): 33 GGUFs — 4 HF
bases (9.3 GiB) + 29 derived (29.0 GiB) — plus one log. Falsifier: a
stranger with network + this repo reconstructs all 41,119,565,056
bytes from this table alone — four HF fetches for the bases, then one
deterministic command per derived family (the derived chain closes:
base Q2_0 → `hybrid_invivo export` → the invivo terminal →
`null_patches`). Full sha256s, not prefixes — byte-identity is the
check. Hashed 2026-08-22 from the working tree.

### Base files (4) — HF fetch

| File | Source (repo @ revision sha; live-verified via HF API 2026-08-22) | Size (B) | SHA-256 |
|---|---|---|---|
| `Bonsai-1.7B-Q1_0.gguf` | `prism-ml/Bonsai-1.7B-gguf` @ `210a9e99f79cb184909d49595906526eb2b3dd9a` | 248,302,272 | `3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3` |
| `Bonsai-4B-Q1_0.gguf` | `prism-ml/Bonsai-4B-gguf` @ `78f2c2bacd0904ffaba24b4873ed975e5818354a` | 572,270,624 | `4524b3f997f0f06444e568d1f26e2efd69effa3218c7ad3047432fb171e42168` |
| `Ternary-Bonsai-4B-F16.gguf` | `prism-ml/Ternary-Bonsai-4B-gguf` @ `a3eb42bafe873f9686bc97486c43b72ef7d75ec8` | 8,049,911,840 | `36bb7f8277a715eeb7ab306fd27d9d4e9abb078c92717856c3d3415777362f5c` |
| `Ternary-Bonsai-4B-Q2_0.gguf` | `prism-ml/Ternary-Bonsai-4B-gguf` @ `a3eb42bafe873f9686bc97486c43b72ef7d75ec8` | 1,074,969,344 | `4e0bf8b737b0431552f8c2c97695ab7c0cb214c94bcdeb4f5f267e67ddf28b8b` |

Cross-check banked: the 1.7B file's 248,302,272 B matches the
RESEARCH_LOG record ("248 MB, from HF `prism-ml/Bonsai-1.7B-gguf`"),
and every HF-listed size matches the local byte count exactly.
Fetch shape: `hf.co/<repo>/resolve/<revision-sha>/<file>`.

### Derived family 1 — `null-dose-1..10` + `null-flip-1..3` (13)

Regenerate (HEAD tool, deterministic seeds — primary dose `0xD05E_…
XOR seed`, seed 1..10; flow-flip hold-out ×3; byte-regeneration
already proven once: `evidence/r4-closeout/` re-pin 13/13
byte-identical):

```bash
cargo run -p neuralos-rt --release --example null_patches
# defaults: orig models/Ternary-Bonsai-4B-Q2_0.gguf
#         + H2   models/Ternary-Bonsai-4B-Q2_0-invivo.gguf
```

| File | Size (B) | SHA-256 |
|---|---|---|
| `null-dose-1.gguf` | 1,074,969,344 | `c2472e7cb9bad3a461ac9025a097fd52b7cd857c853504ac34b3fddaef6d3ccb` |
| `null-dose-2.gguf` | 1,074,969,344 | `27e55dab533a4b4655ebc4a50b5d28fe96db208273090ebbe961941e7dab1fae` |
| `null-dose-3.gguf` | 1,074,969,344 | `38b148345ae64965ce25de2bf1e83e3ecd8c83b6728fb0e3a58ba3a80140d7e0` |
| `null-dose-4.gguf` | 1,074,969,344 | `33489b80c5e0beafaae66d3b0feae34a6326b6cdf8063e8e45a62b86b20bc69f` |
| `null-dose-5.gguf` | 1,074,969,344 | `8c9898233c00ad06334e7e806b8142a3b003c5da14e5c7f88128e65ae926b8d9` |
| `null-dose-6.gguf` | 1,074,969,344 | `338d43b92dd9de4dbd4524c2403757075e47b35b6d14d2f02e7304d9070c289b` |
| `null-dose-7.gguf` | 1,074,969,344 | `847c3884193f4b0af0b2bc9aad77b3c82e1e557b09c72397198a53a2c378e2e8` |
| `null-dose-8.gguf` | 1,074,969,344 | `0287f1698ff97a008306afa8d7c4b73e515057c1f943fecdbbef53b6a78c7ae3` |
| `null-dose-9.gguf` | 1,074,969,344 | `cd3fda094f1e70b91beac8266ec3a008a02da05db33572846233a49edb146b78` |
| `null-dose-10.gguf` | 1,074,969,344 | `96271a0e458c34b8e6d1a69f3cd0ff9cf5ba16436292a35b060ea769532aa139` |
| `null-flip-1.gguf` | 1,074,969,344 | `e7dec5083ba5a8e261dfe7579f5b6a89a66611433b6afa1693331c4e3fb65c92` |
| `null-flip-2.gguf` | 1,074,969,344 | `358429a0f4ddc538a8a9dc39a593b344b773563864002363d13a420ac53062d5` |
| `null-flip-3.gguf` | 1,074,969,344 | `df0614a95ca2561efc293ff6337ece8266bf5a2b8d6932f48fc857d9ddfa7738` |

### Derived family 2 — `null-random-1..10` (10, the stress arm)

Honest pin: emitted by the **v1 census-transition tool** at commit
`4007027` — see `git show 4007027:crates/neuralos-rt/examples/null_patches.rs`
(single-arg usage, H2 census consts baked). HEAD's
`null_patches` emits the dose/flip families only — these ten are the
~178k-cell over-dosed stress arm (report-only,
`evidence/session-i-stress/`). Rebuild = check out the v1 example at
that commit and run it with the base Q2_0.

| File | Size (B) | SHA-256 |
|---|---|---|
| `null-random-1.gguf` | 1,074,969,344 | `991e824ace6296f43686dabf1ab612e9e7cd817b2a341676e5d27ee578b7bd31` |
| `null-random-2.gguf` | 1,074,969,344 | `4e0056f5c9b972b3635dbfdc1d82b98c28911320fe856c1d3d824503858fadad` |
| `null-random-3.gguf` | 1,074,969,344 | `27c4f42d4175554d2937a6b5b35c3fbfdae565bb162169d0226c574b74d3b9eb` |
| `null-random-4.gguf` | 1,074,969,344 | `32c24c6220d1b6653dc574c1868661eb112bd116260f174b0dbc96f30a1f42a8` |
| `null-random-5.gguf` | 1,074,969,344 | `02396834d89d391528c9b54af9aca6920f1a3cd0c731698eb537e90d9010adad` |
| `null-random-6.gguf` | 1,074,969,344 | `52b3472785e0ba6c455de38c56adfc50bc3a98b57407e983062769db020585e4` |
| `null-random-7.gguf` | 1,074,969,344 | `9a88eb90398a4d9700951de48ff462b88ec3db1b4293685e1c3fdb8be37e12e2` |
| `null-random-8.gguf` | 1,074,969,344 | `23608425c99ea80922c68cb31bc1582f247962ccaa32b6c95236a8fda8bf9c59` |
| `null-random-9.gguf` | 1,074,969,344 | `2c5c1e93de896b56de60fdf84a6328aea415cb38d1c6bd581eb36c7ada03660e` |
| `null-random-10.gguf` | 1,074,969,344 | `a622f292d2cdd896be758b5b6dc5ae56bc80dd861b28d9e49876797e37eaef70` |

### Derived family 3 — the in-vivo export set (4)

Regenerate (~6–8 h on the session box; checkpoints = learn-phase
quarters of the 1,600-step learn tier; terminal sha matches the
session-h2 README pin `71f2518a…`):

```bash
cargo run -p neuralos-rt --release --example hybrid_invivo -- \
  models/Ternary-Bonsai-4B-Q2_0.gguf export
# corpus is NOT argv — the pinned corpus (evidence/corpus_readme_pinned.txt,
# sha 18fb5452…, first 2000 tokens, single truncated pass) is baked in.
```

| File | Size (B) | SHA-256 |
|---|---|---|
| `Ternary-Bonsai-4B-Q2_0-invivo.gguf` | 1,074,969,344 | `71f2518a2d783cb409a3c06907a20bed1f1b5688378fc7ea7e8a0f6e16d9749b` |
| `Ternary-Bonsai-4B-Q2_0-invivo-ck400.gguf` | 1,074,969,344 | `7378e978d76fa743618799926988b5f62fa65632ef347a6c1aeb5915a952c58d` |
| `Ternary-Bonsai-4B-Q2_0-invivo-ck800.gguf` | 1,074,969,344 | `c985656c2b2f9efa9e21cc33f5f41d3f413e25654f96351159c39f3c800666c6` |
| `Ternary-Bonsai-4B-Q2_0-invivo-ck1200.gguf` | 1,074,969,344 | `dccc79fcc265330378615c02dcffa32170d6d0e7d4380d271b7804519a40e4eb` |

### Derived family 4 + 5 — loop export + attribution control (2)

```bash
cargo run -p neuralos-rt --release --example hybrid_loop   # → …-loop.gguf
cargo run -p neuralos-rt --release --example hybrid_loop -- \
  models/Ternary-Bonsai-4B-Q2_0.gguf \
  models/Ternary-Bonsai-4B-Q2_0-control.gguf control       # → control
```

| File | Size (B) | SHA-256 |
|---|---|---|
| `Ternary-Bonsai-4B-Q2_0-loop.gguf` | 1,074,969,344 | `24ffe5f3d334051746ddf425f22b54dd7ffa8e707007b99ec21eedb63203f951` |
| `Ternary-Bonsai-4B-Q2_0-control.gguf` | 1,074,969,344 | `4e0bf8b737b0431552f8c2c97695ab7c0cb214c94bcdeb4f5f267e67ddf28b8b` |

The loop sha matches the r4-closeout export pin (`24ffe5f3…`). The
control sha is **identical to the base Q2_0 by construction** — the
Stage-0 attribution control exports the UNADAPTED source trits and
asserts byte-identity against a fresh read; the codec+surgery path
is provably transparent (`hybrid_loop.rs`).

### Non-model artifact

| File | Size (B) | SHA-256 | What |
|---|---|---|---|
| `dl.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | empty download log |
