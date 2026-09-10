#!/usr/bin/env bash
# tools/percommit.sh [base..head]
#
# The per-commit CI job, run locally before the first push: every commit
# of the range (default main..HEAD) gets the full gate set in a scratch
# worktree with its own target dir, so a red commit is found here and
# reworked in place, not by a later commit on top of it (AGENTS.md
# § Session protocol, Git discipline).
#
# Source of the gate list: the `per-commit` job in .gitea/workflows/ci.yml
# (mirrored in .github/workflows/ci.yml). The loop body below copies that
# job's step list line for line, guards included; an edit to one without
# the other is drift, and the amendment grep sweep is what catches it.
# One deliberate difference from the job: it runs in a scratch worktree
# and target dir (the clean-build proof; the job builds in place on one
# cache). The environment is the job's: RUSTFLAGS="-D warnings" from
# both workflow files' top-level env block, NEURALOS_REQUIRE_AVX2=1
# from the per-commit step's own env, so a rustc warning is red here
# exactly as it is there.
#
# Scratch: $NEURALOS_PERCOMMIT_SCRATCH, default .percommit/ at the repo
# root (gitignored). The worktree is removed on exit; the target dir is
# kept so the next run is incremental. Every gate's output goes to
# $scratch/gate.out; on red the last 40 lines are printed and the script
# exits 1 with the worktree still removed.
#
# Needs what CI needs: the pinned toolchain (rust-toolchain.toml is in
# the worktree, rustup picks it up), the riscv32imc-unknown-none-elf
# target, and cmake on PATH for the vendored HDF5 (.nirenv/bin from the
# main clone is prepended).
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
range=${1:-main..HEAD}
scratch=${NEURALOS_PERCOMMIT_SCRATCH:-$repo/.percommit}
wt=$scratch/wt
mkdir -p "$scratch"

export CARGO_TARGET_DIR=$scratch/target
export RUSTFLAGS="-D warnings"
export NEURALOS_REQUIRE_AVX2=1
export PATH="$repo/.nirenv/bin:$PATH"

commits=$(git -C "$repo" rev-list --reverse "$range")
n=$(printf '%s\n' "$commits" | grep -c . || true)
echo "== $n commit(s) in $range  $(date +%T)"
if [ "$n" -eq 0 ]; then
  echo "nothing to check: $range is empty"
  exit 1
fi

# A previous run interrupted before its trap leaves $wt behind: registered
# (remove handles it) or a bare directory git no longer knows (rm does).
if [ -e "$wt" ]; then
  git -C "$repo" worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
fi
git -C "$repo" worktree prune
git -C "$repo" worktree add --quiet --detach "$wt" HEAD
trap 'git -C "$repo" worktree remove --force "$wt" 2>/dev/null || true' EXIT
cd "$wt"

# One gate: log it, run it, on red print the tail and stop.
run() {
  echo "--- $* --- $(date +%T)"
  if ! "$@" >"$scratch/gate.out" 2>&1; then
    echo "RED $c: $*"
    tail -40 "$scratch/gate.out"
    exit 1
  fi
}

for c in $commits; do
  git checkout --quiet "$c"
  echo "===== $c $(git log -1 --format=%s)  $(date +%T)"
  if grep -qE '^ +- name: cargo fmt --check$' .gitea/workflows/ci.yml .github/workflows/ci.yml; then
    run cargo fmt --all -- --check
  else
    echo "(fmt --check is not a gate at this commit)"
  fi
  run cargo check --workspace --all-targets
  run cargo test --workspace
  run cargo clippy --workspace --all-targets -- -D warnings
  run cargo build --no-default-features -p neuralos-snn
  run cargo test -p neuralos-snn --features simd
  run cargo clippy -p neuralos-snn --features simd --all-targets -- -D warnings
  run cargo test -p neuralos-snn --release --features simd -- --include-ignored
  run cargo test -p neuralos-rt --features hdf5
  run cargo run -p neuralos-rt --features hdf5 --example nir_hdf5_gate
  run cargo clippy -p neuralos-rt --features hdf5 --all-targets -- -D warnings
  # firmware/esp32c3 is its own workspace, invisible to every
  # --workspace line above (PR #13: this job was green while the
  # firmware job was red). Guarded like fmt: commits before the
  # crate existed stay valid.
  if [ -d firmware/esp32c3 ]; then
    (cd firmware/esp32c3 && run cargo fmt -- --check)
    (cd firmware/esp32c3 && run cargo build --release --locked)
    (cd firmware/esp32c3 && run cargo clippy --release --locked -- -D warnings)
  else
    echo "(no firmware/esp32c3 at this commit)"
  fi
  echo "===== $c GREEN  $(date +%T)"
done
echo "== all $n commit(s) green on their own  $(date +%T)"
