#!/usr/bin/env bash
# proofs/no-panic-step/build.sh
#
# The link-time proof that FixedNetwork::step has no panic path. Four
# links, two binaries under two profiles:
#   no-panic-step (src/main.rs) steps a black_boxed FixedNetwork<8, 6>
#     forever, and must link;
#   canary (src/bin/canary.rs) indexes a slice by an in-range,
#     black_boxed index, and must be refused, the linker naming the
#     handler's symbol;
#   release is the firmware's profile (opt 3, fat LTO, one codegen unit,
#     abort); nolto is the same without cross-crate LTO and with 16
#     codegen units, Cargo's default release shape.
# The panic handler (src/bare.rs) calls a symbol nothing defines, so a
# panic reachable from _start is an undefined reference and the link is
# red; with none, the linker's section GC drops the handler and the link
# is green. Fat LTO can carry a caller's proof across a call the source
# leaves open, so it alone would pass a tree whose source still has a
# path: the nolto leg makes the proof a property of the code. The canary
# shows, under each leg, both halves the main's green rests on: a
# reachable panic turns the link red, and black_box hides what it is
# given (its in-range index, seen, would fold the check away and link).
# Exit 1 unless both mains link and both canaries are refused by name.
#
# What the link says, and what it does not. It proves one instance,
# FixedNetwork::<8, 6>::step for riscv32imc on the pinned toolchain, in
# release codegen (overflow checks and debug assertions off, as the
# firmware builds); another shape, target or toolchain is not linked
# here. A red main names the undefined symbol and its referencer, the
# panic handler in its codegen unit, never the site. The site is found by
# a build whose handler loops (feature handler-loops), read with
# llvm-objdump; the recipe is printed on red.
#
# The crate's own format gate runs first: `cargo fmt --all` at the root
# never reaches this workspace. CARGO_TARGET_DIR is honored by the four
# builds (tools/percommit.sh points it at its scratch); the recipe builds
# into the crate's own target/, so its two commands agree wherever they
# are pasted. Needs the riscv32imc-unknown-none-elf target on the pin.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
symbol=neuralos_fixed_step_has_a_panic_path

cargo fmt -- --check

# recipe <profile> <bin>: the site-finding build, printed on a red main.
recipe() {
  cat >&2 <<EOF
  The site: at the commit that went red, from the repo root, a build whose
  handler loops, into the crate's own target/ (CARGO_TARGET_DIR cannot move
  it), then every panic call in its disassembly, under its function
  (llvm-objdump from the pin's llvm-tools):
    cd proofs/no-panic-step
    cargo build --locked --target-dir target --profile $1 --bin $2 --features handler-loops
    "\$(rustc --print sysroot)/lib/rustlib/\$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-objdump" \\
      -d -C --no-show-raw-insn target/riscv32imc-unknown-none-elf/$1/$2 \\
      | awk '/^[0-9a-f]+ <.*>:\$/ { f = \$0 } /panic/ && !/^[0-9a-f]+ </ { print f; print }'
EOF
}

green=0
for profile in release nolto; do
  for bin in no-panic-step canary; do
    rc=0
    out=$(cargo build --locked --profile "$profile" --bin "$bin" 2>&1) || rc=$?
    named=no
    if grep -q "undefined symbol: $symbol" <<<"$out"; then
      named=yes
    fi
    if [ "$bin" = no-panic-step ] && [ "$rc" -eq 0 ]; then
      echo "$profile $bin: links, no panic path reachable from the step"
      green=$((green + 1))
    elif [ "$bin" = canary ] && [ "$rc" -ne 0 ] && [ "$named" = yes ]; then
      echo "$profile $bin: refused, the linker naming $symbol, as it must"
      green=$((green + 1))
    elif [ "$bin" = no-panic-step ] && [ "$named" = yes ]; then
      echo "$profile $bin: RED, a panic path is reachable from the step:" >&2
      # lld's lines, the referencer's included; the home directory masked.
      grep -E "undefined symbol|>>> " <<<"$out" | sed "s#$HOME#~#g" >&2 || true
      recipe "$profile" "$bin"
    elif [ "$bin" = canary ] && [ "$rc" -eq 0 ]; then
      echo "$profile $bin: RED, the canary linked: a panic is no longer a red link, or black_box no longer hides its index; a green main then proves less" >&2
    else
      echo "$profile $bin: RED, the build failed, and not on $symbol (exit $rc):" >&2
      tail -20 <<<"$out" >&2
    fi
  done
done

if [ "$green" -ne 4 ]; then
  echo "no-panic-step: RED, $green of 4 verdicts as they must be" >&2
  exit 1
fi
echo "no-panic-step: green, 4 of 4: the step links under both profiles, and both canaries are refused by name"
