#!/usr/bin/env python3
"""Diff an ESP32-C3 capture against the host's trace files.

The firmware (firmware/esp32c3/src/main.rs) replays every frozen case of
crates/neuralos-snn/tests/traces/frozen.rs: each prints its trace's
header line, then its rows, and the replay ends with one line,
`# neuralos-trace end`. This script takes the frozen set from the files
themselves (every trace whose header says plasticity=off), splits the
log at the headers, drops carriage returns, and compares each case with
its file line for line: one line per case, exit 1 on any difference. A
header missing or repeated, a header outside the frozen set, or no end
line after the last case is a difference too: a capture cut off is red,
not a shorter compare. Any capture with the headers and the end line
reads the same way, the C3's or QEMU's (proofs/qemu-trace-replay/, the
same replay on riscv64gc).

`--trace FILE`, repeatable, adds a trace from outside the tree to the
set: a stranger's graph in the firmware's slot (firmware/esp32c3/
stranger.sh), replayed after the frozen cases, whose trace
`neuralos-nir2json --freeze` wrote. Its header must say plasticity=off
and must not be in the set already; a file that does not read as text
is a usage error too. Two files with one header, in the tree or given,
are a usage error, never one case silently standing for two. A given
trace prints as `<stem> (given)`, so a stem that matches a tree case
never reads as a duplicate. The last line counts every case of the set.

usage: tools/esp32c3_trace_diff.py [--trace FILE]... LOG
Exit: 0 every case identical · 1 a difference · 2 usage.
Standard library only.
"""
import sys
from pathlib import Path

TRACES = Path(__file__).resolve().parent.parent / "crates/neuralos-snn/tests/traces"
HEAD = "# neuralos-trace v1 case="
END = "# neuralos-trace end"
USAGE = "usage: tools/esp32c3_trace_diff.py [--trace FILE]... LOG"


def lines_of(text):
    """The lines of a text, carriage returns dropped, no trailing empty line."""
    lines = text.replace("\r", "").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def usage(why):
    """A usage error: the reason and the usage line, exit 2."""
    print(f"esp32c3_trace_diff: {why}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def main():
    args, given, logs = sys.argv[1:], [], []
    while args:
        arg = args.pop(0)
        if arg == "--trace":
            if not args:
                return usage("--trace needs FILE")
            given.append(Path(args.pop(0)))
        elif arg.startswith("--"):
            return usage(f"unknown flag {arg}")
        else:
            logs.append(arg)
    if len(logs) != 1:
        return usage("one LOG")
    log = lines_of(Path(logs[0]).read_bytes().decode("utf-8", errors="replace"))

    cases = {}
    for path in sorted(TRACES.glob("*.trace")):
        lines = lines_of(path.read_text())
        if lines and " plasticity=off " in lines[0]:
            if lines[0] in cases:
                return usage(f"{path.stem} and {cases[lines[0]][0]} share one header: {lines[0]}")
            cases[lines[0]] = (path.stem, lines)
    if not cases:
        print(f"no plasticity-off trace in {TRACES}", file=sys.stderr)
        return 1
    for path in given:
        try:
            lines = lines_of(path.read_text())
        except (OSError, UnicodeDecodeError) as e:
            return usage(f"--trace {path}: {e}")
        if not lines or not lines[0].startswith(HEAD) or " plasticity=off " not in lines[0]:
            return usage(f"--trace {path}: not a plasticity-off trace header")
        if lines[0] in cases:
            return usage(f"--trace {path}: its header is already in the set ({cases[lines[0]][0]})")
        cases[lines[0]] = (f"{path.stem} (given)", lines)

    heads = [i for i, line in enumerate(log) if line.startswith(HEAD)]
    ends = [i for i, line in enumerate(log) if line == END]
    red = 0
    for i in heads:
        if log[i] not in cases:
            print(f"log line {i + 1}: a header outside the frozen set: {log[i]}")
            red += 1
    for header, (name, want) in cases.items():
        at = [i for i in heads if log[i] == header]
        if len(at) != 1:
            where = "missing from the log" if not at else f"{len(at)} times in the log"
            print(f"{name}: {where}")
            red += 1
            continue
        start = at[0]
        stop = min((i for i in heads + ends if i > start), default=len(log))
        have = log[start:stop]
        if have == want:
            print(f"{name}: identical, {len(want) - 1} rows")
            continue
        n = next(
            (k for k, (a, b) in enumerate(zip(want, have)) if a != b),
            min(len(want), len(have)),
        )
        file_line = want[n] if n < len(want) else "<end>"
        log_line = have[n] if n < len(have) else "<end>"
        print(f"{name}: DIFFERS at line {n + 1}: file {file_line!r}, log {log_line!r}")
        red += 1
    if not ends or (heads and ends[-1] < max(heads)):
        print(f"no {END!r} line after the last case: the capture is cut off")
        red += 1
    print(f"{len(cases)} cases, {red} red")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
