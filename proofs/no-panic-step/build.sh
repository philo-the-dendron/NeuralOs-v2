#!/usr/bin/env bash
# proofs/no-panic-step/build.sh
#
# The link-time proof that FixedNetwork::step has no panic path. Four
# links, two binaries under two profiles:
#   no-panic-step (src/main.rs) steps a black_boxed FixedNetwork<8, 6>
#     forever, and must link;
#   canary (src/bin/canary.rs) indexes a slice by a black_boxed index, and
#     must be refused, the linker naming the handler's symbol;
#   release is the firmware's profile (opt 3, fat LTO, one codegen unit,
#     abort); nolto is the same without LTO and with 16 codegen units,
#     Cargo's default release shape.
# The panic handler (src/bare.rs) calls a symbol nothing defines, so a
# panic reachable from _start is an undefined reference and the link is
# red; with none, the linker's section GC drops the handler and the link
# is green. Fat LTO can carry a caller's proof across a call the source
# leaves open, so it alone would pass a tree whose source still has a
# path: the nolto leg makes the proof a property of the code, and the
# canary under each leg is what makes that leg's green real. Exit 1 unless
# both mains link and both canaries are refused by name.
#
# What a red main says, and what it does not: lld names the undefined
# symbol and the object that references it, the codegen unit that holds
# the panic handler, never the site. The site is found by a build whose
# handler loops (feature handler-loops), read with llvm-objdump; the
# recipe is printed on red.
#
# The crate's own format gate runs first: `cargo fmt --all` at the root
# never reaches this workspace. CARGO_TARGET_DIR is honored
# (tools/percommit.sh points it at its scratch). Needs the
# riscv32imc-unknown-none-elf target on the pin.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
elf_dir=${CARGO_TARGET_DIR:-$here/target}/riscv32imc-unknown-none-elf
symbol=neuralos_fixed_step_has_a_panic_path

cargo fmt -- --check

# recipe <profile> <bin>: the site-finding build, printed on a red main.
recipe() {
  local elf="${elf_dir/#"$HOME"/\~}/$1/$2"
  cat >&2 <<EOF
  lld names the handler's codegen unit, never the site. The site: a build
  whose handler loops, then every panic call in its disassembly, under its function
  (from the repo root; llvm-objdump from the pin's llvm-tools):
    (cd proofs/no-panic-step && cargo build --locked --profile $1 --bin $2 --features handler-loops)
    "\$(rustc --print sysroot)/lib/rustlib/\$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-objdump" \\
      -d -C --no-show-raw-insn $elf \\
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
      grep -E "undefined symbol|>>> referenced by" <<<"$out" >&2 || true
      recipe "$profile" "$bin"
    elif [ "$bin" = canary ] && [ "$rc" -eq 0 ]; then
      echo "$profile $bin: RED, the canary linked: a panic is no longer a red link, and a green main proves nothing" >&2
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
