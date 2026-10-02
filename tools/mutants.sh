#!/usr/bin/env bash
# tools/mutants.sh: the mutation pass. cargo-mutants 27.1.0 and the rules
# (tools/mutants/rules.py) mutate the same lines of one tree under one
# memory cap, and tools/mutants/table.py reads both as one table. ISA.md's
# rounds 53 and 54 record the passes run by hand before it, and
# tools/mutants/planted/ keeps the traps they met as cases with known
# answers.
#
#   tools/mutants.sh diff [BASE..HEAD] [--features LIST] [-- PATH...]
#   tools/mutants.sh file [--features LIST] PATH...
#   tools/mutants.sh selftest
#
# diff: the lines BASE..HEAD changes (default main..HEAD), mutated at
# HEAD, in .mutants/diff-<base>-<head>/. file: every line of each file at
# HEAD, in .mutants/file-<head>/, for a hunt. Paths are from the repo's
# root. selftest: tools/mutants/planted/ in .mutants/selftest/, every
# line, under CAP=1G with RUSTFLAGS="-D warnings", one pass for each its
# EXPECTED.tsv names (the planted feature off, then on); exit 1 on any
# difference from that file. --features goes to cargo-mutants and to the
# rules' cargo alike. The tests are the packages owning the mutated
# files, and neuralos-nir2json when neuralos-snn is one; without
# --features the library builds with std and unstable-freeze alone, so
# bridge.rs, kernel.rs and simd.rs read UNBUILT.
#
# A pass: a git archive of HEAD in tree/; one build of the test packages
# with the run's flags (rules.py build: the features line and the pass's
# own .d files, and cargo-mutants' baseline finds it built);
# cargo-mutants --in-place --in-diff --no-shuffle with this checkout's
# .cargo/mutants.toml (an older tree has none); the rules on the same
# lines; then verdicts.tsv, the counts and blind.tsv (table.py). Tree and
# target are deleted at the end; the table and the logs stay.
#
# The run is one `systemd-run --user --scope` with MemoryMax=${CAP:-6G},
# MemorySwapMax=0 and OOMPolicy=continue: no scope, no run. RUSTC_WRAPPER
# is unset inside, or rustc would run in the sccache server's cgroup and a
# server started inside would outlive the run. The CAPPED rows must equal
# the scope's memory.events oom_kill, or the run exits 1.
#
# Three limits. An item behind an off feature, inside a file the build
# reads (the STDP items in network.rs, synapse.rs and trit.rs), reads
# MISSED: run that file with --features. simd.rs's cfg(not(target_arch =
# "x86_64")) lines never build on x86. And cargo-mutants mutates test code
# under cfg(all(test, …)) (156 of simd.rs's 178 mutants), which the rules
# leave alone.
#
# Exit: 0 the pass ran and its table holds · 1 a check failed (a selftest
# difference, CAPPED against oom_kill, a verdict the table cannot read),
# cargo-mutants failed (its baseline too), or no file of the diff lies in
# a package's src/ · 2 usage, or no scope · 4 the rules' build or
# baseline failed.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
py=$here/mutants

usage() {
  sed -n 's/^#   \(tools\/mutants\.sh .*\)/usage: \1/p' "${BASH_SOURCE[0]}" >&2
  exit 2
}

mode=${1:-}
case $mode in diff | file | selftest) ;; *) usage ;; esac

# The cap: this script again, inside a scope of its own.
if [ -z "${NEURALOS_MUTANTS_SCOPE:-}" ]; then
  cap=${CAP:-6G}
  [ "$mode" = selftest ] && cap=1G
  if ! systemd-run --user --scope --quiet true; then
    echo "mutants.sh: no user scope (systemd-run --user --scope), so no run" >&2
    exit 2
  fi
  exec systemd-run --user --scope --quiet --unit="mutants-$$" \
    -p MemoryMax="$cap" -p MemorySwapMax=0 -p OOMPolicy=continue \
    env -u RUSTC_WRAPPER -u CARGO_BUILD_RUSTC_WRAPPER NEURALOS_MUTANTS_SCOPE="$cap" \
    bash "${BASH_SOURCE[0]}" "$@"
fi
shift
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
case $cg in
  *.scope) ;;
  *) echo "mutants.sh: not in a scope: $cg" >&2; exit 2 ;;
esac
if [ "$(cat "$cg/memory.max")" != "$(numfmt --from=iec "$NEURALOS_MUTANTS_SCOPE")" ] ||
  [ "$(cat "$cg/memory.swap.max")" != 0 ]; then
  echo "mutants.sh: $cg is not capped at $NEURALOS_MUTANTS_SCOPE without swap" >&2
  exit 2
fi

features="" range="" paths=() files=()
while [ $# -gt 0 ]; do
  case $1 in
    --features) features=${2:?--features needs a list}; shift 2 ;;
    --) shift; paths=("$@"); break ;;
    -*) usage ;;
    *)
      if [ "$mode" = diff ] && [ -z "$range" ]; then range=$1; else files+=("$1"); fi
      shift
      ;;
  esac
done
case $mode in
  diff) [ ${#files[@]} -eq 0 ] || usage ;;
  file) [ ${#files[@]} -gt 0 ] && [ ${#paths[@]} -eq 0 ] || usage ;;
  selftest) [ -z "$features$range" ] && [ ${#files[@]} -eq 0 ] && [ ${#paths[@]} -eq 0 ] || usage ;;
esac

# One flag set for both tools: the caller's, any --cap-lints dropped, as
# CARGO_ENCODED_RUSTFLAGS; cargo-mutants' cap_lints and the rules each add
# --cap-lints=warn after it (rustc takes the first of two).
if [ "$mode" = selftest ]; then
  unset CARGO_ENCODED_RUSTFLAGS
  export RUSTFLAGS="-D warnings"
fi
CARGO_ENCODED_RUSTFLAGS=$(python3 "$py/rules.py" flags)
export CARGO_ENCODED_RUSTFLAGS
unset RUSTFLAGS
export CARGO_TERM_COLOR=never

case $mode in
  diff)
    range=${range:-main..HEAD}
    [[ $range == *..* ]] || usage
    base=${range%%..*} head=${range#*..}
    label=diff-$(git -C "$repo" rev-parse --short "$base")-$(git -C "$repo" rev-parse --short "$head")
    ;;
  file) head=HEAD label=file-$(git -C "$repo" rev-parse --short HEAD) ;;
  selftest) label=selftest ;;
esac
out=$repo/.mutants/${label:?}
tree=$out/tree
export CARGO_TARGET_DIR=$out/target
rm -rf "$out"
mkdir -p "$tree"
trap 'rm -rf "$tree" "$CARGO_TARGET_DIR"' EXIT
status=0

# synth TREE SPEC...: a diff that adds each SPEC's lines, a file whole or
# FILE:A-B, the shape both tools read.
synth() {
  local tree=$1 spec f a b n
  shift
  for spec; do
    f=${spec%%:*}
    n=$(awk 'END { print NR }' "$tree/$f")
    a=1 b=$n
    if [ "$spec" != "$f" ]; then
      a=${spec#*:} b=${spec##*-}
      a=${a%-*}
    fi
    if [ "$a" -eq 1 ] && [ "$b" -eq "$n" ]; then
      printf 'diff --git a/%s b/%s\nnew file mode 100644\n--- /dev/null\n+++ b/%s\n@@ -0,0 +1,%d @@\n' \
        "$f" "$f" "$f" "$n"
    else
      printf 'diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -%d,0 +%d,%d @@\n' \
        "$f" "$f" "$f" "$f" $((a - 1)) "$a" $((b - a + 1))
    fi
    awk -v a="$a" -v b="$b" 'NR >= a && NR <= b { print "+" $0 }' "$tree/$f"
    if [ "$b" -eq "$n" ] && [ -n "$(tail -c1 "$tree/$f")" ]; then
      echo '\ No newline at end of file'
    fi
  done
}

# pass DIR CONFIG: both tools over the tree, on the lines of DIR/diff,
# then table.py's summary.
pass() {
  local dir=$1 config=$2 rc=0 p names sub=${1#"$out"}
  local pkgs=() pargs=() targs=() fargs=()
  [ -n "$features" ] && fargs=(--features "$features")
  names=$(python3 "$py/rules.py" packages "$tree" "$dir/diff")
  mapfile -t pkgs <<<"$names"
  if [ -z "$names" ]; then
    echo "$label: no file of the diff lies in a package's src/" >&2
    return 1
  fi
  for p in "${pkgs[@]}"; do pargs+=(-p "$p") targs+=(--test-package "$p"); done
  echo "== $label${sub:+ ${sub#/}}: tests ${pkgs[*]}${features:+, features $features}  $(date +%T)"
  python3 "$py/rules.py" build "$tree" "$dir/built.json" "${pargs[@]}" "${fargs[@]}"
  (cd "$tree" && cargo mutants --list --json --no-config --in-diff "$dir/diff" --workspace) \
    >"$dir/cm-list.json"
  echo "   cargo-mutants  $(date +%T)"
  (cd "$tree" && cargo mutants --in-place --in-diff "$dir/diff" --no-shuffle --config "$config" \
    --workspace "${targs[@]}" "${fargs[@]}" --output "$dir") >"$dir/cargo-mutants.log" 2>&1 || rc=$?
  case $rc in
    0 | 2 | 3) ;;
    *)
      echo "$label: cargo-mutants exited $rc" >&2
      tail -20 "$dir/cargo-mutants.log" >&2
      return 1
      ;;
  esac
  echo "   the rules  $(date +%T)"
  python3 "$py/rules.py" run "$tree" "$dir/diff" "$dir/rules.tsv" "${pargs[@]}" "${fargs[@]}" \
    --config "$config" >"$dir/rules.log"
  echo "   the table  $(date +%T)"
  python3 "$py/table.py" "$dir" "$tree" --config "$config" || status=1
}

case $mode in
  diff)
    mb=$(git -C "$repo" merge-base "$base" "$head")
    git -C "$repo" archive "$head" | tar -x -C "$tree"
    rs=$(git -C "$repo" diff --name-only --no-renames "$mb" "$head" -- "${paths[@]}" | grep '\.rs$' || true)
    if [ -z "$rs" ]; then
      echo "$label: no Rust file in $range"
      exit 0
    fi
    mapfile -t rs <<<"$rs"
    git -C "$repo" diff --no-color --no-ext-diff --no-renames "$mb" "$head" -- "${rs[@]}" >"$out/diff"
    pass "$out" "$repo/.cargo/mutants.toml"
    ;;
  file)
    git -C "$repo" archive "$head" | tar -x -C "$tree"
    synth "$tree" "${files[@]}" >"$out/diff"
    pass "$out" "$repo/.cargo/mutants.toml"
    ;;
  selftest)
    planted=$here/mutants/planted
    expected=$planted/EXPECTED.tsv
    tar -C "$planted" --exclude=./target -cf - . | tar -x -C "$tree"
    want=$(sed -n 's/^#: \(cargo-mutants .*\)/\1/p' "$expected")
    if [ "$want" != "$(cargo mutants --version)" ]; then
      echo "selftest: EXPECTED.tsv is for $want, this is $(cargo mutants --version)" >&2
      exit 1
    fi
    mapfile -t passes < <(grep '^#: pass ' "$expected")
    : >"$out/actual.tsv"
    for line in "${passes[@]}"; do
      read -r -a f <<<"$line"
      name=${f[2]} features="" specs=("${f[@]:3}")
      if [ "${specs[0]}" = --features ]; then
        features=${specs[1]} specs=("${specs[@]:2}")
      fi
      mkdir -p "$out/$name"
      synth "$tree" "${specs[@]}" >"$out/$name/diff"
      pass "$out/$name" "$tree/.cargo/mutants.toml"
      awk -F'\t' -v p="$name" '{ print p "\t" $1 "\t" $2 "\t" $3 "\t" $4 }' \
        "$out/$name/verdicts.tsv" >>"$out/actual.tsv"
      awk -F'\t' -v p="$name" '{ print p "\tEXCLUDED\t" $1 "\t" $2 "\t" }' \
        "$out/$name/excluded.tsv" >>"$out/actual.tsv"
      awk -F'\t' -v p="$name" '{ print p "\tBLIND\t-\t" $1 "\t" }' \
        "$out/$name/blind.tsv" >>"$out/actual.tsv"
    done
    LC_ALL=C sort -o "$out/actual.tsv" "$out/actual.tsv"
    awk -F'\t' '!/^#/ && $1 != "pass" { print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 }' "$expected" |
      LC_ALL=C sort >"$out/expected.tsv"
    if diff "$out/expected.tsv" "$out/actual.tsv" >"$out/selftest.diff"; then
      echo "selftest: GREEN, $(wc -l <"$out/actual.tsv") rows as EXPECTED.tsv says"
    else
      echo "selftest: RED, the run differs from EXPECTED.tsv (< expected, > the run):"
      cat "$out/selftest.diff"
      status=1
    fi
    ;;
esac

capped=$(find "$out" -name verdicts.tsv -exec cat {} + | grep -c '^CAPPED' || true)
oom=$(awk '$1 == "oom_kill" { print $2 }' "$cg/memory.events")
echo "$label: CAPPED $capped, the scope's oom_kill $oom, its peak $(($(cat "$cg/memory.peak") / 1048576)) MiB"
if [ "$capped" != "$oom" ]; then
  echo "$label: RED, the CAPPED rows are not the scope's kills" >&2
  status=1
fi
exit $status
