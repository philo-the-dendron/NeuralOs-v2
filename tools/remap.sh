# tools/remap.sh — sourced by the artifact build scripts
# (firmware/esp32c3/build.sh, crates/neuralos-nir2json/build.sh).
# Not executable on purpose: `source tools/remap.sh`, then call the
# functions below. ISA round 26 is the brief; this header is the why.
#
# WHY: a release asset bypasses the hooks (they guard added lines and
# the tracked tree, not binaries), and two shipped binaries carried the
# builder's home path in their panic-location strings: 16 hits in the
# alpha.5 firmware ELF, 38 in the nir2json musl binary (ISA round-23
# addendum, the ruling that left them standing). This file makes the
# build the guard: rustc's --remap-path-prefix rewrites every path at
# compile time, and a gate inside the build refuses the artifact if a
# personal string survived.
#
# VEHICLE: the flags go through the environment for the one cargo call.
# Not `.cargo/config.toml` rustflags: a RUSTFLAGS variable in the
# environment replaces config rustflags wholesale, and CI sets
# RUSTFLAGS="-D warnings" for every job (PR #13 linked without linkall.x
# that way). Not build.rs: a build script may emit link args and -l/-L
# only, never a codegen flag. So: CARGO_ENCODED_RUSTFLAGS, built from
# the inherited RUSTFLAGS (or an inherited encoded one) plus the remap
# flags, so CI's -D warnings survives. Encoded (0x1f-separated) rather
# than RUSTFLAGS because RUSTFLAGS is whitespace-split and a path may
# hold a space.
#
# THE FOUR ROOTS, computed at run time, never hardcoded, because a CI
# image may keep cargo or rustup outside $HOME:
#   $HOME                          -> ~
#   ${CARGO_HOME:-$HOME/.cargo}    -> ~/.cargo        (the registry hits)
#   $(rustc --print sysroot)       -> ~/.rustup/toolchains/<name>
#                                     (rust-src hits: <sysroot>/lib/rustlib/src/rust/library/)
#   $(git rev-parse --show-toplevel) -> ~/projets/NeuralOs-v2
#                                     (the spine, a path dependency)
# The aliases are fixed strings, identical on every machine, so a
# remapped build yields the same path strings — and the same .text —
# wherever it runs. They follow the banking convention of the record
# (`/home/<user>/` -> `~/`, ISA round 24). ORDER MATTERS: $HOME first,
# the specific roots after, because rustc applies the LAST matching
# prefix (measured on 1.92.0, ISA round 26: general-then-specific maps
# under the specific alias, the reverse under the general one).
#
# THE STAMP: esp-bootloader-esp-idf 0.5.0's build.rs reads
# SOURCE_DATE_EPOCH (else the wall clock) into the app descriptor, and
# declares only rerun-if-changed=./esp_config.yml, no env rerun guard,
# so a warm target dir keeps the cached stamp whatever the variable
# says. The firmware script cleans that one package before the build;
# the epoch is the commit's (`git log -1 --format=%ct`), exported here
# in seconds as the spec says. 0.5.0 reads it as microseconds
# (`Timestamp::from_microsecond`); the firmware script converts, for
# that locked version only (its header).
#
# WHAT IS COMPARABLE: the .text sha. The ELF sha stays path-dependent
# through cargo's symbol-name hashes in .strtab (they follow the
# package's source path), and the app-descriptor stamp follows the
# commit date; neither is in .text (evidence/esp32c3-bringup/README.md
# § Rebuild + run).
#
# THE GATE (remap_gate): two greps on the artifact. The personal
# pattern from .githooks/personal-pattern, read from the file, never a
# hardcoded login, case-insensitive like the hooks; and the four roots
# as fixed strings, because the pattern's path alternative matches
# `/home/<x>/` only and a runner whose home is /root would slip a
# failed remap past it. Any hit exits 1, and so does a grep that did
# not run (exit 2: a pattern grep -E rejects, an unreadable file): a
# scan that failed is not a pass. Hits are counted per grep, and the
# distinct strings printed with the roots masked so the log stays clean.
#
# THE CANARY (remap_canary): cargo's `profile.*.trim-paths` will one
# day do all of this on stable. On 1.92.0 the probe
#   cargo build --config 'profile.release.trim-paths="all"' ...
# exits 101 in the build front end's profile validation, before any
# unit compiles (config parse and `cargo metadata` accept the key;
# ISA round 26 holds the transcript). The probe runs after the real
# build, --offline, into a throwaway target dir. Expected failure:
# pass, one line. Exit 0, or a different failure: one loud notice —
# that is the day to re-probe on the new pin, drop the remap, and see
# whether the gate passes without it. Re-probe at each toolchain bump.
# THE GATE STAYS regardless of the canary; it is what makes
# "bump-proof" a fact, not a hope.

# remap_env: export CARGO_ENCODED_RUSTFLAGS and SOURCE_DATE_EPOCH.
# Sets REMAP_HOME, REMAP_CARGO_HOME, REMAP_SYSROOT, REMAP_REPO for the
# gate. Run from inside the repo so rustup resolves the pinned toolchain.
remap_env() {
  REMAP_REPO=$(git rev-parse --show-toplevel)
  REMAP_HOME=$HOME
  REMAP_CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}
  REMAP_SYSROOT=$(rustc --print sysroot)
  export REMAP_REPO REMAP_HOME REMAP_CARGO_HOME REMAP_SYSROOT

  local -a flags=()
  if [ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]; then
    IFS=$'\x1f' read -r -a flags <<<"$CARGO_ENCODED_RUSTFLAGS"
  elif [ -n "${RUSTFLAGS:-}" ]; then
    read -r -a flags <<<"$RUSTFLAGS"
  fi
  flags+=(
    "--remap-path-prefix=$REMAP_HOME=~"
    "--remap-path-prefix=$REMAP_CARGO_HOME=~/.cargo"
    "--remap-path-prefix=$REMAP_SYSROOT=~/.rustup/toolchains/$(basename "$REMAP_SYSROOT")"
    "--remap-path-prefix=$REMAP_REPO=~/projets/NeuralOs-v2"
  )
  CARGO_ENCODED_RUSTFLAGS=$(IFS=$'\x1f'; printf '%s' "${flags[*]}")
  export CARGO_ENCODED_RUSTFLAGS
  unset RUSTFLAGS

  SOURCE_DATE_EPOCH=$(git -C "$REMAP_REPO" log -1 --format=%ct)
  export SOURCE_DATE_EPOCH
  echo "remap: 4 roots -> ~, ~/.cargo, ~/.rustup/toolchains/$(basename "$REMAP_SYSROOT"), ~/projets/NeuralOs-v2; SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH"
}

# remap_show <path>: the path as the record writes it, `$HOME` -> `~`
# (the banking convention, ISA round 24), for every line these scripts
# print: a build log pasted into evidence must not carry the home path.
remap_show() {
  printf '%s' "${1/#"$REMAP_HOME"/\~}"
}

# remap_gate <file>: exit 1 on any personal string or unremapped root,
# and on any scan that did not run.
remap_gate() {
  local file=$1 pattern by_pattern by_root rc n_pattern n_root
  if [ ! -f "$file" ]; then
    echo "gate: $(remap_show "$file") does not exist; nothing to scan is not a pass" >&2
    return 1
  fi
  pattern=$(cat "$REMAP_REPO/.githooks/personal-pattern")
  if [ -z "$pattern" ]; then
    echo "gate: .githooks/personal-pattern is missing or empty; refusing rather than scanning with an empty pattern" >&2
    return 1
  fi
  # grep exits 0 on a hit, 1 on none, 2 on an error (a pattern grep -E
  # rejects, an unreadable file). 0 and 1 are a scan; 2 is not a pass.
  rc=0
  by_pattern=$(grep -a -o -E -i -e "$pattern" "$file") || rc=$?
  if [ "$rc" -gt 1 ]; then
    echo "gate: the pattern grep failed (exit $rc) on $(remap_show "$file"); a scan that did not run is not a pass" >&2
    return 1
  fi
  rc=0
  by_root=$(grep -a -o -F -e "$REMAP_HOME" -e "$REMAP_CARGO_HOME" -e "$REMAP_SYSROOT" -e "$REMAP_REPO" "$file") || rc=$?
  if [ "$rc" -gt 1 ]; then
    echo "gate: the root grep failed (exit $rc) on $(remap_show "$file"); a scan that did not run is not a pass" >&2
    return 1
  fi
  n_pattern=0; [ -z "$by_pattern" ] || n_pattern=$(printf '%s\n' "$by_pattern" | wc -l)
  n_root=0; [ -z "$by_root" ] || n_root=$(printf '%s\n' "$by_root" | wc -l)
  if [ "$n_pattern" -gt 0 ] || [ "$n_root" -gt 0 ]; then
    echo "gate: RED — $n_pattern pattern hit(s), $n_root root hit(s) in $(remap_show "$file"); the distinct strings, roots masked:" >&2
    printf '%s\n' "$by_pattern" "$by_root" | sed '/^$/d' | sort -u \
      | sed -e "s#$REMAP_REPO#<REPO>#g" -e "s#$REMAP_SYSROOT#<SYSROOT>#g" \
            -e "s#$REMAP_CARGO_HOME#<CARGO_HOME>#g" -e "s#$REMAP_HOME#<HOME>#g" >&2
    return 1
  fi
  echo "gate: 0 hits in $(remap_show "$file") (pattern + 4 roots)"
}

# remap_text_sha <elf>: sha256 of the .text section image. Needs
# llvm-objcopy: the toolchain's (rustup component add llvm-tools) or
# one on PATH.
remap_text_sha() {
  local elf=$1 host objcopy tmp
  host=$(rustc -vV | sed -n 's/^host: //p')
  objcopy="$REMAP_SYSROOT/lib/rustlib/$host/bin/llvm-objcopy"
  if [ ! -x "$objcopy" ]; then
    objcopy=$(command -v llvm-objcopy || true)
  fi
  if [ -z "$objcopy" ]; then
    echo "text-sha: no llvm-objcopy (rustup component add llvm-tools, or install llvm)" >&2
    return 1
  fi
  # Called as `v=$(remap_text_sha …)`: errexit is off inside a command
  # substitution (bash's inherit_errexit is off by default), so each
  # failure returns by hand, or the caller prints a wrong sha. An empty
  # image (no .text) is a failure too: its sha is the empty file's.
  tmp=$(mktemp)
  if ! "$objcopy" -O binary --only-section=.text "$elf" "$tmp" || [ ! -s "$tmp" ]; then
    echo "text-sha: no .text image from $(remap_show "$elf")" >&2
    rm -f "$tmp"
    return 1
  fi
  # The sha is checked before it is returned: with the cleanup as the last
  # command the function's status was the cleanup's, and a failed checksum
  # would have returned an empty sha with status 0.
  local sha
  sha=$(sha256sum "$tmp" | cut -d' ' -f1)
  rm -f "$tmp"
  if [ -z "$sha" ]; then
    echo "text-sha: sha256sum failed on the .text image of $(remap_show "$elf")" >&2
    return 1
  fi
  printf '%s\n' "$sha"
}

# remap_canary <crate-dir> <cargo build args...>: the trim-paths probe.
# Never fails the build; prints one line on the expected failure and a
# loud notice otherwise.
remap_canary() {
  local dir=$1; shift
  local out rc
  out=$(cd "$dir" && CARGO_TARGET_DIR="$dir/target/canary" \
    cargo build --offline --config 'profile.release.trim-paths="all"' "$@" 2>&1) && rc=0 || rc=$?
  if [ "$rc" -eq 101 ] && printf '%s' "$out" | grep -q 'feature `trim-paths` is required'; then
    echo "canary: trim-paths still unstable on $(cargo --version | cut -d' ' -f2) (exit 101, as expected); the remap stays"
    return 0
  fi
  echo "canary: NOTICE ================================================" >&2
  echo "canary: the trim-paths probe did not fail the expected way (exit $rc)." >&2
  echo "canary: if it built, cargo now trims paths on stable: re-probe on this" >&2
  echo "canary: pin, drop the remap, and see whether the gate passes without it" >&2
  echo "canary: (tools/remap.sh header). The gate stays either way." >&2
  printf '%s\n' "$out" | tail -5 >&2
  echo "canary: NOTICE ================================================" >&2
  return 0
}
