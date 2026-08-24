# Step-5 prep — evidence pins + rebuild (2026-08-23)

Prep session artifacts, all sha-pinned. The burn window reads
PREREG.md as authority; this file pins what prep verified.

## Model-file pins (sha256, this session)

| file | sha256 | status |
|---|---|---|
| Ternary-Bonsai-4B-Q2_0.gguf | 4e0bf8b737b0431552f8c2c97695ab7c0cb214c94bcdeb4f5f267e67ddf28b8b | == ISC-68 control-identity pin ✓ |
| Ternary-Bonsai-4B-Q2_0-invivo.gguf | 71f2518a2d783cb409a3c06907a20bed1f1b5688378fc7ea7e8a0f6e16d9749b | == H2 final-export pin ✓ |
| Ternary-Bonsai-4B-Q2_0-loop.gguf | 24ffe5f3d334051746ddf425f22b54dd7ffa8e707007b99ec21eedb63203f951 | == sF re-export pin ✓ |
| …-invivo-ck400.gguf | 7378e978d76fa743618799926988b5f62fa65632ef347a6c1aeb5915a952c58d | fresh prep pin (H2 banked cell counts 61,210, not shas — gap named) |
| …-invivo-ck800.gguf | c985656c2b2f9efa9e21cc33f5f41d3f413e25654f96351159c39f3c800666c6 | fresh prep pin (71,381 cells) |
| …-invivo-ck1200.gguf | dccc79fcc265330378615c02dcffa32170d6d0e7d4380d271b7804519a40e4eb | fresh prep pin (80,391 cells) |

ck cell counts re-verifiable before judging via the surgery decode
(harness splice machinery, control mode).

## The drive-domain probe (Rider B → measurement)

`clamp_probe.log` (sha 61e04bbb6dc30da56ece2c1b74cb572ae22077e21798722f9ed5ee0afa2d3f88)
— instrument `crates/neuralos-rt/examples/step5_clamp_probe.rs`.

Rebuild:

```bash
cargo run -p neuralos-rt --release --example step5_clamp_probe \
  > evidence/step5-readout/clamp_probe.log
```

Findings of record (all H2 pins reproduced first — k 10060.46,
4411 tokens, 69.477%, hist, dim 199 ×1786):

- pre-clamp |I| (window r0): p50 311,874 μA · p90 523,144 · p99
  1,881,305 · max 4,839,080 — ≈1000× the 450 μA target
  (milli/norm-unit domain mismatch, hybrid_invivo.rs:343 vs :325);
- clamped fraction IDENTICAL (69.477%) at ceilings
  ±1000/2000/3000/10000/32767 — the ceiling ladder is dead within
  the i16 input path;
- corrected domain (k on norm units): 2.74% clamped, realized RMS
  exactly 450.0 μA — the sH registration's stated intent;
- per-window k (Rider A): r0 10060.46 · r1 10101.90 · r2 10007.65.

## Null seeds

`null_seeds.txt` — 201–230 (+ 231–240 escalation), pre-committed;
the NULL generator reads, never mints.

## Calibration (the §8 artifact of record)

`calibrate.log` (sha 4558674f510a0788a3aca7e3f8952974aedec12124a0a8da1a666499c62ccb5e)
— run from the final tool, post checklist-fixes:

```bash
cargo run -p neuralos-rt --release --example step5_aggregate -- --calibrate \
  > evidence/step5-readout/calibrate.log
```

GATE PASS on banked logs (loud d1/d3 classify, quiet d7/loop silent);
M3 provenance pins test-enforced in judge.rs.

## Authority

`PREREG.md` (ratified 2026-08-23). ISA entries: sequencing + Tier-3
criteria; drive-domain finding + CLAMP-RELAXED→DOMAIN-CORRECTED
ruling; paper pre-submission fix (invivo.tex ×2, limitations.tex).
