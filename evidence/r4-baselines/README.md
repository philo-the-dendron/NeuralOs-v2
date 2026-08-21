# R4 baselines — the pre-refactor re-pin record (banked 2026-08-20)

The (a)-verification + R4(i) banking: every recorded verdict of the
frozen example family, re-run on THIS tree (`2fb7c5b`, post-audit-prelude
— prelude added tests/docs/CI only, zero numeric changes) immediately
before harness extraction. Post-refactor, R4(iv) must reproduce these
numbers exactly; any divergence is a refactor bug (stop rule: report,
no tuning).

| Run | Log(s) | Re-pinned verdict |
|---|---|---|
| `hybrid_gate` ×2 | `gate_run{1,2}.log` | **HYBRID GATE: ADAPTS**; intra +0.1075 / inter +0.0000; `[PAIRING-SELECTIVE, CLAMP-RECTIFIED]`; double-run identical (modulo wall line) |
| `hybrid_loop` ×2 | `loop_run{1,2}.log` | **LOOP EXPORT: CLEAN**; D-2 preconditions asserted in-run; export sha `24ffe5f3d3340517` — **byte-identical to the paper's recorded artifact** (app-repro §shas) AND to the pre-existing banked file, both runs |
| `hybrid_sweep` | `sweep_mv_run1.log` | **A\* = 600 μA** (mV); live-wire triple 35115/35136/35157 at 600 μA; E-cliff between 450 and 300 |
| `hybrid_sweep_cmv` | `sweep_cmv_run1.log` | **A\* = 600 μA** (centi); E-cliff at 150 μA (25750 = I-only floor); first-divergence row matches the session-F record |
| judge p0, base + loop ×2 | `judge_p0_{base,loop}_run{1,2}.{log,err}` | 0/12 argmax flips, max \|Δ\| **+0.4207** (recorded: 0.42), mean 0.0779 (recorded range 0.057–0.151); double-run dump lines byte-identical; **continuations byte-identical** base vs loop |
| `hybrid_invivo` | not re-run — see below | tier-1 (release compile) ✓ via the build; full-tier baseline = `evidence/session-h-invivo/` (session H ran on this same substrate; the prelude changed no numerics — proven by every other baseline re-pinning bit-exact and the 239-test quartet) |
| `null_patches` | not re-run | regenerates `models/null-*.gguf` deterministically on demand; the families' judge evidence lives in `evidence/session-i-{primary,stress}/` |

## Re-pin protocol for R4(iv)

1. `git tag examples-pre-extraction` holds the pre-refactor sources —
   the paper's "run them unmodified" contract points there.
2. Re-run gate + loop (+ sweeps) on the extracted harness; assert:
   gate verdict line identical, loop export sha `24ffe5f3…`, sweep
   A\*/cliff rows identical.
  3. `hybrid_invivo` bar (adjudicated): default-gain bit-identity —
    rewritten run must reproduce the banked T1 pins exactly
    (G0 H(i,c)=41,190 > H(i,z)=30,216; clamp fraction 69.8%; export
    67,309 cells / 40,126 code bytes per `evidence/INDEX.md`); the
    full 2,298 s record stays banked, not re-derived.
    **[CLOSED 2026-08-21 — see `evidence/r4-closeout/`: the H1 bar
    above was STALE when written (the sH2 single-pass drive redesign
    predates this banking; a 332-token corpus cannot reach the learn
    tier). Re-pinned instead against the H2 record at default args:
    byte-identical, wall/RSS + the by-design-not-run export tier
    excepted.]**
  4. Judge legs: one double-run pair (p0) via `tools/build_fork.sh` +
    the flags of record; `judge_delta` (the Rust port of delta.py —
    `cargo run -p neuralos-rt --release --example judge_delta -- <base.err>
    <patched.err>`) must read 0 flips, max |Δ| ≈ 0.4207,
    continuations byte-identical. **[CLOSED 2026-08-21 —
    `evidence/r4-closeout/`: 0/12 flips, max +0.4207, mean 0.0779,
    exact.]**
5. One-command subset: `AUTO_SMOKE=1 bash tools/hybrid_smoke.sh`
   (gate verdict + loop CLEAN + export sha; ~1 min with models).
