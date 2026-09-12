#!/usr/bin/env bash
# crates/neuralos-nir2json/build.sh
#
# The release artifact: the static musl binary, built with the path
# remap, gated for personal strings, probed by the trim-paths canary
# (tools/remap.sh carries the why; the firmware's build.sh is the same
# shape). The published bring-up binary carried 38 home-path strings
# (35 registry, 3 rust-src; ISA round 26); this build carries none or
# it does not exist. Output: dist/neuralos-nir2json-v<crate version>-
# x86_64-unknown-linux-musl (dist/ is gitignored; the binary lives on
# the release, its sha in the record), the ELF sha and the .text sha.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
cd "$repo"
# shellcheck source=../../tools/remap.sh
source "$repo/tools/remap.sh"
remap_env

target=x86_64-unknown-linux-musl
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$here/Cargo.toml" | head -1)
built=${CARGO_TARGET_DIR:-$repo/target}/$target/release/neuralos-nir2json   # CARGO_TARGET_DIR honored (tools/percommit.sh sets it)
out=$here/dist/neuralos-nir2json-v$version-$target

cargo build -p neuralos-nir2json --release --locked --target "$target"
remap_gate "$built"
mkdir -p "$here/dist"
# dist/ holds release artifacts, sha-pinned in the record; a build never
# overwrites a different one under the same name (the 2026-09-10 dist
# ELF was overwritten by an unrecorded command, ISA round 26 (b)).
if [ -e "$out" ] && ! cmp -s "$built" "$out"; then
  echo "build.sh: ${out#"$repo"/} exists with a different sha ($(sha256sum "$out" | cut -c1-12)…); move it aside first, it is a pinned artifact" >&2
  exit 1
fi
cp "$built" "$out"
# Assigned first: `echo "$(f)"` hides f's exit status (tools/remap.sh, remap_text_sha).
elf_sha=$(sha256sum "$out" | cut -d' ' -f1)
text_sha=$(remap_text_sha "$out")
echo "elf  sha256 $elf_sha  ${out#"$repo"/}"
echo "text sha256 $text_sha  (.text image; the comparable)"
remap_canary "$repo" -p neuralos-nir2json --release --locked --target "$target"
