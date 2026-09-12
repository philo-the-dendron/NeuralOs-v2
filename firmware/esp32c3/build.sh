#!/usr/bin/env bash
# firmware/esp32c3/build.sh [build|clippy]
#
# The firmware build, with the path remap, the personal-string gate and
# the trim-paths canary (tools/remap.sh carries the why). `build`
# (default) is what every caller runs — CI's firmware job, the
# per-commit loop, tools/percommit.sh, the evidence README § Rebuild +
# run, the release build — and prints the ELF sha and the .text sha.
# `clippy` runs the lint under the same flags: clippy never shares
# build artifacts (it checks, it does not build), but same flags keep
# its own cache valid across the per-commit loop and make the lints
# see the same paths the artifact carries.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
# shellcheck source=../../tools/remap.sh
source "$(git rev-parse --show-toplevel)/tools/remap.sh"
remap_env

# The stamp's unit. esp-bootloader-esp-idf 0.5.0 (build.rs line 16)
# parses SOURCE_DATE_EPOCH with `Timestamp::from_microsecond`; the
# reproducible-builds spec says seconds, and the spec'd value stamped
# the descriptor `1970-01-01 00:29:49` (ISA round 26). So the seconds
# remap_env exported become microseconds here, for that version only:
# the lock is checked, and a bump refuses until the unit is re-read.
# That crate is the only locked dependency reading the variable.
bootloader_version=$(awk '$0=="name = \"esp-bootloader-esp-idf\""{getline; gsub(/version = |"/,""); print}' Cargo.lock)
if [ "$bootloader_version" = "0.5.0" ]; then
  SOURCE_DATE_EPOCH=$((SOURCE_DATE_EPOCH * 1000000))
  export SOURCE_DATE_EPOCH
else
  echo "build.sh: esp-bootloader-esp-idf is $bootloader_version, not 0.5.0: re-read its build.rs for the SOURCE_DATE_EPOCH unit before building (header)" >&2
  exit 1
fi

# CARGO_TARGET_DIR is honored (tools/percommit.sh sets it to its scratch).
elf=${CARGO_TARGET_DIR:-$here/target}/riscv32imc-unknown-none-elf/release/neuralos-esp32c3
case ${1:-build} in
  build)
    # The app-descriptor stamp: its build script has no env rerun guard
    # (tools/remap.sh, THE STAMP), so it is cleaned before every build.
    cargo clean --release -p esp-bootloader-esp-idf
    cargo build --release --locked
    remap_gate "$elf"
    # Assigned first: `echo "$(f)"` hides f's exit status, an assignment
    # under set -e does not (tools/remap.sh, remap_text_sha).
    elf_sha=$(sha256sum "$elf" | cut -d' ' -f1)
    text_sha=$(remap_text_sha "$elf")
    echo "elf  sha256 $elf_sha  $(remap_show "$elf")"
    echo "text sha256 $text_sha  (.text image; the comparable, README § Rebuild + run)"
    remap_canary "$here" --release --locked
    ;;
  clippy)
    cargo clippy --release --locked -- -D warnings
    ;;
  *)
    echo "usage: $0 [build|clippy]" >&2
    exit 2
    ;;
esac
