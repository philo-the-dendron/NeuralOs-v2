#!/usr/bin/env python3
"""A pass's table: cargo-mutants' outcomes and the rules' rows read as one
set of verdicts, the counts, the features each package built with, and
the blind lines. `tools/mutants.sh` runs it once a pass.

cargo-mutants' verdicts come from `mutants.out/outcomes.json` (`summary`,
`log_path`, `phase_results[].process_status`) and from each mutant's log,
read the way the rules read their own runs. A mutant in a file no `.d`
file of the pass's own build lists is UNBUILT, whatever it read:
cargo-mutants mutates a module behind an off feature and reads it MISSED.
A SIGKILL is CAPPED: under the cap, a killed test reads caught. A catch
by a signal notes it. An excluded mutant, by the config's `exclude_re`
(the rules read the same regexes), is counted and never run:
cargo-mutants' list without the config, less what ran, must be exactly
the names those regexes match.

A blind line is an added code line, in a file a package's `src/` holds,
that no mutant reached with a verdict other than UNVIABLE or UNBUILT. A
whole-function replacement reaches nothing, and an arm deleted over
several lines reaches its first line only: a catch says a test takes the
branch, not that the lines inside it are tested. A code line lies in a fn
body, outside test code, attributes and a match arm's pattern, and holds
a call (`f(`, `.m(`, `m!(`), an operator (`?`, `!`, or a binary one with
a space each side), a literal, or a control word (`if`, `while`,
`return`, `break`, `continue`).

usage: tools/mutants/table.py PASS_DIR TREE [--config TOML]
PASS_DIR holds `diff`, `built.json` (rules.py build), `cm-list.json`
(cargo-mutants --list --json, no config), `mutants.out/` and `rules.tsv`.
The table goes to `verdicts.tsv` (verdict, tool, name, note),
`excluded.tsv`, `blind.tsv` and `summary.txt`, the summary to stdout too.
Exit: 0 · 1 a verdict the table cannot read, or an exclusion the two
tools read apart · 2 usage.
Standard library only.
"""
import json
import os
import re
import sys

sys.dont_write_bytecode = True  # no __pycache__ beside the tracked files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules  # noqa: E402

CM, RU = "cargo-mutants", "rules"
VERDICTS = ["CAUGHT", "MISSED", "CAPPED", "TIMEOUT", "UNVIABLE", "UNBUILT", "EXCLUDED", "ERROR"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NAME = re.compile(r"^(.*?):(\d+):(\d+): ")
TOKEN = re.compile(
    r"\b(?:true|false|if|while|return|break|continue)\b"
    r"|(?<![\w.])\d"
    r"|\?"
    r"|(?<![\w!=])!(?!=)"
    r"|(?<=\s)(?:==|!=|<=|>=|&&|\|\||<<=|>>=|<<|>>|[-+*/%^&|]=|[-+*/%^&|<>])(?=\s)"
    r"|[A-Za-z_]\w*\s*!?\s*[(\[{]"
    r"|(?<![=-])>\s*\("
)


def code_lines(src):
    """The line numbers of `src`'s code lines (the module docstring says
    which)."""
    bodies = [(f.open + 1, f.close) for f in src.fns]
    pats = [(a.start, a.pat_end) for _, _, arms in src.matches for a in arms]

    def counts(p):
        return (src.within(p, bodies) and not src.within(p, src.tests)
                and not src.within(p, src.attrs) and not src.within(p, pats))

    lines = set()
    for p in range(src.n):
        if src.kind[p] == "l" and (p == 0 or src.kind[p - 1] != "l") and counts(p):
            lines.add(src.line(p))
    for m in TOKEN.finditer(src.cs):
        t = m.group(0)
        if t[-1] in "([{" and t[0] != ">":  # `>(` closes a turbofish: a call
            word = rules.IDENT.match(t)
            if not t.rstrip("([{ \t\n").endswith("!") and word and word.group(0) in rules.KEYWORDS:
                continue  # `if (`, `match [`: a keyword, no call
            if t[-1] in "[{" and not t.rstrip("[{ \t\n").endswith("!"):
                continue  # an index or a block, no call
        if counts(m.start()):
            lines.add(src.line(m.start()))
    return lines


def cm_rows(pass_dir, sources):
    """cargo-mutants' mutants as (verdict, name, file, first, last, genre,
    note), and the baseline's pass count."""
    out_dir = os.path.join(pass_dir, "mutants.out")
    path = os.path.join(out_dir, "outcomes.json")
    if not os.path.exists(path):
        return []
    outcomes = json.load(open(path, encoding="utf-8"))["outcomes"]
    rows, base = [], None
    for o in outcomes:
        log = ANSI.sub("", open(os.path.join(out_dir, o["log_path"]), encoding="utf-8",
                                errors="replace").read())
        if o["scenario"] == "Baseline":
            base = rules.passed(log)
            continue
        m, s, note = o["scenario"]["Mutant"], o["summary"], ""
        killed = rules.SIGKILL in log or any(pr["process_status"] == {"Signalled": 9}
                                              for pr in o["phase_results"])
        if m["file"] not in sources:
            v = "UNBUILT"
        elif s == "Timeout":
            v = "TIMEOUT"
        elif killed:
            v, note = "CAPPED", "build" if s == "Unviable" else ""
        elif s == "Unviable":
            v = "UNVIABLE"
        elif s == "CaughtMutant":
            v = "CAUGHT"
            sig = re.search(r"\(signal: (\d+), ", log)
            if sig:
                note = f"signal {sig.group(1)}"
        elif s == "MissedMutant":
            v = "MISSED"
            n = rules.passed(log)
            if base is not None and n != base:
                note = f"{n} passed, baseline {base}"
        else:
            v, note = "ERROR", f"cargo-mutants read {s}"
        rows.append((v, m["name"], m["file"], m["span"]["start"]["line"],
                     m["span"]["end"]["line"], m["genre"], note))
    return rows


def rules_rows(pass_dir):
    """The rules' rows as (verdict, name, file, first, last, rule, note)."""
    path = os.path.join(pass_dir, "rules.tsv")
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        v, name, file, first, last, rule, _, _, note = line.rstrip("\n").split("\t")
        rows.append((v, name, file, int(first), int(last), rule, note))
    return rows


def where(name):
    m = NAME.match(name)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (name, 0, 0)


def main():
    args = sys.argv[1:]
    config = None
    if "--config" in args:
        i = args.index("--config")
        config = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        print("usage: tools/mutants/table.py PASS_DIR TREE [--config TOML]", file=sys.stderr)
        return 2
    pass_dir, tree = args
    built = json.load(open(os.path.join(pass_dir, "built.json"), encoding="utf-8"))
    sources = set(built["sources"])
    cm, ru = cm_rows(pass_dir, sources), rules_rows(pass_dir)
    bad = []

    # the exclusion, read alike by both tools
    try:
        listed = [m["name"] for m in json.load(open(os.path.join(pass_dir, "cm-list.json"),
                                                    encoding="utf-8"))]
    except (OSError, ValueError):
        listed = []
    regexes = rules.excludes(config)
    cm_excluded = [n for n in listed if any(r.search(n) for r in regexes)]
    ran = {r[1] for r in cm}
    if ran != set(listed) - set(cm_excluded):
        bad.append("cargo-mutants ran other mutants than its list less the excluded:")
        bad += [f"  ran, not expected: {n}" for n in sorted(ran - (set(listed) - set(cm_excluded)))]
        bad += [f"  expected, not run: {n}" for n in sorted(set(listed) - set(cm_excluded) - ran)]

    table = [(v, CM, name, note) for v, name, _, _, _, _, note in cm]
    table += [(v, RU, name, note) for v, name, _, _, _, _, note in ru if v != "EXCLUDED"]
    table.sort(key=lambda r: (*where(r[2]), r[1], r[2]))
    excluded = sorted([(CM, n) for n in cm_excluded] + [(RU, r[1]) for r in ru if r[0] == "EXCLUDED"],
                      key=lambda e: (*where(e[1]), e[0]))
    bad += [f"no verdict read: {t} {n} ({note})" for v, t, n, note in table if v == "ERROR"]

    # the blind lines
    diff = open(os.path.join(pass_dir, "diff"), encoding="utf-8").read()
    reached = {}
    for v, _, file, first, last, genre, _ in cm:
        if v not in ("UNVIABLE", "UNBUILT", "ERROR") and genre != "FnValue":
            last = first if genre == "MatchArm" else last
            reached.setdefault(file, set()).update(range(first, last + 1))
    for v, _, file, first, last, rule, _ in ru:
        if v not in ("UNVIABLE", "UNBUILT", "EXCLUDED", "ERROR"):
            last = first if rule == "arm" else last
            reached.setdefault(file, set()).update(range(first, last + 1))
    blind, pkgs = [], rules.metadata(tree)
    for path, (_, added) in sorted(rules.diff_lines(diff).items()):
        full = os.path.join(tree, path)
        if not path.endswith(".rs") or rules.owner(path, pkgs) is None or not os.path.exists(full):
            continue
        text = open(full, encoding="utf-8").read()
        code = code_lines(rules.Source(path, text))
        lines = text.split("\n")
        for ln in sorted(added & code - reached.get(path, set())):
            blind.append((f"{path}:{ln}", lines[ln - 1].strip()))

    with open(os.path.join(pass_dir, "verdicts.tsv"), "w", encoding="utf-8") as f:
        f.writelines("\t".join(r) + "\n" for r in table)
    with open(os.path.join(pass_dir, "excluded.tsv"), "w", encoding="utf-8") as f:
        f.writelines(f"{t}\t{n}\n" for t, n in excluded)
    with open(os.path.join(pass_dir, "blind.tsv"), "w", encoding="utf-8") as f:
        f.writelines(f"{w}\t{text}\n" for w, text in blind)

    out = [f"{os.path.basename(os.path.normpath(pass_dir))}: {len(table)} mutants run or read, "
           f"{len(excluded)} excluded"]
    for name, feats in built["features"].items():
        off = [x for x in built["declared"].get(name, []) if x not in feats]
        out.append(f"  {name} built with {', '.join(feats) or 'no feature'}; "
                   f"left off {', '.join(off) or 'none'}")
    cols = [v for v in VERDICTS if v != "ERROR" or any(r[0] == "ERROR" for r in table)]
    out.append("  " + " " * 14 + "".join(f"{v:>9}" for v in cols))
    for tool in (CM, RU):
        n = {v: sum(1 for r in table if r[1] == tool and r[0] == v) for v in cols}
        n["EXCLUDED"] = sum(1 for t, _ in excluded if t == tool)
        out.append(f"  {tool:<14}" + "".join(f"{n[v]:>9}" for v in cols))
    for v, t, name, note in table:
        if v in ("MISSED", "TIMEOUT"):
            out.append(f"  {v:<8} {t:<14} {name}" + (f" ({note})" if note else ""))
    out.append(f"  blind lines: {len(blind)}")
    out += [f"    {w}  {text}" for w, text in blind]
    out += bad
    text = "\n".join(out) + "\n"
    open(os.path.join(pass_dir, "summary.txt"), "w", encoding="utf-8").write(text)
    sys.stdout.write(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
