# spike-path-bench

Host timing of the LIF spike path, public API only, one harness for
two trees. The question it answers: did the alpha.6 ring (O(1) push)
change the per-step cost against the alpha.5 `heapless::Vec` history
(`remove(0)` = a 64-entry shift on every spike past the 64th)? The
harness prints numbers; the comparison and the box are recorded in
`evidence/esp32c3-bringup/README.md` § Host bench.

## Run (from the repo root)

```bash
# arm B: the alpha.6 tree (this checkout)
(cd proofs/spike-path-bench && cargo run --release --locked) \
  | tee evidence/esp32c3-bringup/host-bench-alpha6.log

# arm A: the alpha.5 tree, same source, path dependency re-resolved
git worktree add --detach bench-rebuild/alpha5 371e8c6
cp -r proofs/spike-path-bench bench-rebuild/alpha5/proofs/
(cd bench-rebuild/alpha5/proofs/spike-path-bench && cargo run --release --offline) \
  | tee evidence/esp32c3-bringup/host-bench-alpha5.log
git worktree remove --force bench-rebuild/alpha5
```

The block is the round-23 run as recorded, when this checkout was the
alpha.6 tree, and its two log names are pinned in
`evidence/esp32c3-bringup/SHA256SUMS`: a re-run writes to a new file
name and is compared against the pinned log, never written over it
(round 27's pair is `host-bench-r27-alpha6.log` and
`host-bench-r27-alpha7.log`, commands in that evidence README).

The committed `Cargo.lock` named the spine at alpha.6 for that run
(alpha.7 since round 27); in the alpha.5
worktree cargo re-resolves the path dependency to alpha.5 (no
`--locked` there, by design). Same profile (release, fat LTO, one
codegen unit), same box, same command. Cargo refuses two
semver-compatible versions of one crate in one graph, which is why
the two arms are two runs and not two dependencies.
