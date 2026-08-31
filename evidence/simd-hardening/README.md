# SIMD hardening — benchmark evidence (2026-08-31)

Backs the ISA close-out claim that the rounding fix (`div1024_toward_zero`,
commit `e52623e` on `work/simd-hardening`) costs the vector path roughly
12–15 % at N=1024 and leaves the scalar path unchanged.

**Species:** sha-pinned machine output. `bench_alternated.log` is immutable;
this README is the rebuild recipe.

**Rebuild.** Two binaries, alternated so the machine's load drifts equally
over both:

```bash
git checkout f82fc1e   # main before the branch
cargo build --release --example bench_simd --features simd -p neuralos-snn
cp target/release/examples/bench_simd /tmp/bench_before
git checkout cfe38f9   # the tip when this ran; the binary is byte-identical at 8a63d58 (sha256 0cd66b43…)
cargo build --release --example bench_simd --features simd -p neuralos-snn
cp target/release/examples/bench_simd /tmp/bench_after
for r in 1 2 3; do /tmp/bench_before; /tmp/bench_after; done
```

**Reading it.** The scalar column is the control (same code in both
binaries); its spread is the machine's noise. At N=1024, run 2: simd
before 1590 / 1506 / 1476 ns/step (mean 1524), after 1732 / 1699 / 1711
(mean 1714): +12.5 % mean, +15.1 % min-to-min; speedup 1.79× → 1.60×. The
reviewer's independent scalar-controlled run the same night gave +15.3 %
(1.71× → 1.47×). Direction is certain (after is slower in all six
alternated pairs); the magnitude is 12–15 % on a loaded box. A quiet-box
number is still owed.
