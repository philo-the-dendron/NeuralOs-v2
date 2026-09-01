# SIMD hardening — benchmark evidence (2026-08-31)

Backs the ISA close-out claim that the rounding fix (`div1024_toward_zero`,
commit `e52623e` on `work/simd-hardening`) costs the vector path **+11.7 %**
at N=1024 — the scalar-controlled figure, identical in both runs — and leaves
the scalar path unchanged.

**Species:** sha-pinned machine output. `bench_alternated.log` is immutable;
this README is the rebuild recipe, and `SHA256SUMS` re-pins it on every
correction (git is its changelog).

**Rebuild.** Two binaries, alternated so the machine's load drifts equally
over both:

```bash
git checkout f82fc1e   # main before the branch
cargo build --release --example bench_simd --features simd -p neuralos-snn
cp target/release/examples/bench_simd /tmp/bench_before
git checkout cfe38f9   # see "The after-binary's commit" below
cargo build --release --example bench_simd --features simd -p neuralos-snn
cp target/release/examples/bench_simd /tmp/bench_after
for r in 1 2 3; do /tmp/bench_before; /tmp/bench_after; done
```

**The after-binary's commit.** The log's header cites
`work/simd-hardening@1c5615a`, and that citation is dangling: `1c5615a` is
reachable from no ref. It is a real object in this clone — the pre-amend
version of `cfe38f9`, same parent `4eb2719`, kept alive only by the reflog
until it is garbage-collected — so `git rev-parse` still resolves it here
and `git log --all` cannot find it. In a fresh clone it does not exist at
all. The log is immutable and keeps the citation it was written with; this
is the correction.

`cfe38f9` is the honest substitute, and the substitution is verified, not
assumed:

- `1c5615a` and `cfe38f9` share the parent `4eb2719`.
- Every hunk of `git diff 1c5615a cfe38f9 -- crates/neuralos-snn/src/simd.rs`
  falls after line 583, which is `mod tests {` at both revisions. Same for
  `cfe38f9..8a63d58`. `mod tests` is `#[cfg(test)]`, which `--example
  bench_simd` never compiles.
- The non-test part of `simd.rs` (lines 1–582) hashes identically at
  `1c5615a`, `cfe38f9`, `8a63d58` and `137088d`:
  `sha256 061fee96d222bf47…`. That is what makes the binaries byte-identical
  (`sha256 0cd66b436eeaa01c`), and it is the claim the log's own header rests
  on.

The pin holds through `137088d` and no further, and it ends on BOTH
follow-up branches. Commits are named by subject as well as pinned, because
a sha is branch-local once work is cherry-picked and a bare sha is the
failure being corrected here:

- on `fix/simd-fixture-and-record`, `e79909e` — `docs(simd): two module-doc
  comments said things that are not true` — is the first commit to change
  the non-test region, taking it to `sha256 32e5bc3910beaf81`;
- on `fix/lif-scalar-domain`, `fix(lif): one dt_over_tau definition,
  saturating, shared with the batch` takes it to `sha256 2a5d18d6cad46298`.

Either way, a binary built past `137088d` is NOT the one measured here, and
re-running these numbers on a follow-up branch tip is a new experiment, not
a rebuild.

**Reading it.** The scalar column is the control (same code in both
binaries); its spread is the machine's noise. At N=1024, means over the
three alternated rounds of each run:

| | simd before | simd after | raw | scalar control | scalar-controlled |
|---|---|---|---|---|---|
| run 1 | 1665.3 | 1802.6 | **+8.2 %** | 2872.9 → 2783.0 (−3.1 %) | **+11.7 %** |
| run 2 | 1523.9 | 1714.3 | **+12.5 %** | 2720.0 → 2738.3 (+0.7 %) | **+11.7 %** |

Direction is certain: after is slower in all six alternated pairs. The
honest cost is **+11.7 % scalar-controlled, identical in both runs**; the
raw spread is 8–13 % and the min-to-min spread 7–15 %, which is the loaded
box, not the code. The reviewer's independent scalar-controlled run the same
night gave +15.3 % (1.71× → 1.47×). A quiet-box number is still owed.

(This section read "12–15 %" until 2026-09-01. That band was run 2 only —
its +12.5 % raw and +15.1 % min-to-min — quoted as if it covered both runs.
Run 1's raw figure is +8.2 %, outside the band on the low side. Controlling
for the scalar column, which is what the control is for, both runs agree
exactly at +11.7 %.)
