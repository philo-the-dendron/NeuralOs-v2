#!/usr/bin/env python3
"""The close-out check: the ISA.md close-out a range appends, read from git
objects alone, against its form and the commits it cites. The form is the
one rounds 45 to 54 hold; the 30 lines are cut 3 of AGENTS.md § Session
protocol. It was a hand step, and a record commit once lost its heading
through every green gate (ISA round 53). A cited sha must be on the head,
not merely resolve: after a rewrite, a retired sha still resolves through
the reflog or a bundle.

Red, one line a finding:
- the range appends no `## Close-out (round N — … — YYYY-MM-DD)` heading,
  or more than one;
- no blank line after the heading;
- more than 30 body lines;
- a line over 72 characters (characters, not bytes: `—` is one);
- a line opening with a list marker (`-`, `+`, `*`, `1.`, `1)`);
- an odd count of backticks on a line;
- a 7-hex token in backticks that is no commit on the range's head (`git
  merge-base --is-ancestor`). A token of 8 hex or more is a content hash,
  skipped: rounds 45 to 54 cite commits at 7 and hashes at 8.

usage: tools/closeout.py [BASE..HEAD]   the range, default main..HEAD
       tools/closeout.py --selftest     planted close-outs in a temp repo
Exit: 0 green · 1 red · 2 usage.
Standard library only.
"""
import os
import re
import subprocess
import sys
import tempfile

HEADING = re.compile(r"^## Close-out \(round \d+ — .+ — \d{4}-\d{2}-\d{2}\)$")
LIST = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s")
HEX = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,}(?![0-9A-Za-z])")
USAGE = "usage: tools/closeout.py [BASE..HEAD] | --selftest"


def git(*args, check=True):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"closeout.py: git {' '.join(args)}: {r.stderr.strip()}")
    return r


def appended(base, head):
    """The lines BASE..HEAD adds to ISA.md, read from its hunks."""
    mb = git("merge-base", base, head).stdout.strip()
    diff = git("diff", "--no-color", "--no-ext-diff", "-U0", mb, head, "--", "ISA.md").stdout
    lines, hunk = [], False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            hunk = False
        elif line.startswith("@@"):
            hunk = True
        elif hunk and line.startswith("+"):
            lines.append(line[1:])
    return lines


def cited(body):
    """The 7-hex tokens inside the backtick spans of `body`, in order."""
    seen = []
    for line in body:
        for span in line.split("`")[1::2]:
            for m in HEX.finditer(span):
                if len(m.group(0)) == 7 and m.group(0) not in seen:
                    seen.append(m.group(0))
    return seen


def findings(base, head):
    """(findings, body line count) for the close-out BASE..HEAD appends."""
    added = appended(base, head)
    heads = [line for line in added if HEADING.match(line)]
    if len(heads) != 1:
        out = [f"the range appends {len(heads)} `## Close-out (round N — … — YYYY-MM-DD)` "
               f"headings, not one"]
        out += [f"a heading out of form: {line}" for line in added
                if line.startswith("## Close-out") and not HEADING.match(line)]
        return out, 0
    text = git("show", f"{head}:ISA.md").stdout.split("\n")
    i = max(k for k, line in enumerate(text) if line == heads[0])
    j = i + 1
    while j < len(text) and not text[j].startswith("## "):
        j += 1
    out, block = [], text[i + 1:j]
    if block and block[0] == "":
        body = block[1:]
    else:
        out.append("no blank line after the heading")
        body = block
    while body and body[-1] == "":
        body.pop()
    if len(body) > 30:
        out.append(f"{len(body)} body lines, more than 30")
    for n, line in enumerate(body, 1):
        if len(line) > 72:
            out.append(f"line {n}: {len(line)} characters, more than 72")
        if LIST.match(line):
            out.append(f"line {n}: opens with a list marker")
        if line.count("`") % 2:
            out.append(f"line {n}: an odd count of backticks")
    for tok in cited(body):
        if git("rev-parse", "--verify", "--quiet", f"{tok}^{{commit}}", check=False).returncode:
            out.append(f"`{tok}` is no commit here")
        elif git("merge-base", "--is-ancestor", tok, head, check=False).returncode:
            out.append(f"`{tok}` is not on {head}")
    return out, len(body)


def check(rng):
    base, sep, head = rng.partition("..")
    if not sep or not base or not head:
        print(USAGE, file=sys.stderr)
        return 2
    out, n = findings(base, head)
    print(f"{rng}: {'RED' if out else 'GREEN'} ({n} body lines)")
    for f in out:
        print(f"    {f}")
    return 1 if out else 0


def selftest():
    """Planted close-outs in a temp repo: a good one green, one red case a
    rule, each with its finding alone."""
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="selftest", GIT_AUTHOR_EMAIL="selftest@example.invalid",
               GIT_COMMITTER_NAME="selftest", GIT_COMMITTER_EMAIL="selftest@example.invalid")
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        os.environ.update(env)
        try:
            git("init", "-q", "-b", "main")
            isa = "# ISA\n\n## Close-out (round 1 — the base — 2026-01-01)\n\nThe base.\n"
            open("ISA.md", "w", encoding="utf-8").write(isa)
            git("add", "ISA.md")
            git("commit", "-q", "-m", "base")
            base = git("rev-parse", "--short=7", "HEAD").stdout.strip()
            git("switch", "-q", "-c", "side")
            git("commit", "-q", "--allow-empty", "-m", "a commit off the head")
            side = git("rev-parse", "--short=7", "HEAD").stdout.strip()
            git("switch", "-q", "main")
            wide = "The base, `" + base + "`, is on the head; a line of 72 characters — so "
            wide += "x" * (72 - len(wide))
            assert len(wide) == 72 < len(wide.encode()), "the 72-character line holds no `—`"
            good = ["## Close-out (round 2 — a good close-out — 2026-01-02)", "",
                    f"Branch `work/good` from `main@{base}`, one commit.", wide,
                    "`9e2e6ef1` is a content hash, 8 hex, never read as a commit."]
            heading, rest = good[0], good[2:]
            cases = [
                ("good", good, []),
                ("no heading", ["", *rest],
                 ["the range appends 0 `## Close-out (round N — … — YYYY-MM-DD)` headings, not one"]),
                ("two headings", [*good, "", heading.replace("round 2", "round 3"), "", *rest],
                 ["the range appends 2 `## Close-out (round N — … — YYYY-MM-DD)` headings, not one"]),
                ("no blank line", [heading, *rest], ["no blank line after the heading"]),
                ("31 lines", [heading, "", *rest, *["One more line."] * 28],
                 ["31 body lines, more than 30"]),
                ("73 characters", [heading, "", *rest, "y" * 73], ["line 4: 73 characters, more than 72"]),
                ("a list marker", [heading, "", *rest, "- a list item"], ["line 4: opens with a list marker"]),
                ("odd backticks", [heading, "", *rest, "one ` alone"], ["line 4: an odd count of backticks"]),
                ("no commit", [heading, "", *rest, "`abc1234` cited."], ["`abc1234` is no commit here"]),
                ("off the head", [heading, "", *rest, f"`{side}` cited."], [f"`{side}` is not on case"]),
            ]
            red = 0
            for name, lines, want in cases:
                git("switch", "-q", "-c", "case", "main")
                open("ISA.md", "w", encoding="utf-8").write(isa + "\n" + "\n".join(lines) + "\n")
                git("commit", "-q", "-a", "-m", name)
                got, _ = findings("main", "case")
                ok = got == want
                red += not ok
                print(f"{'ok ' if ok else 'BAD'} {name}: {'; '.join(got) or 'green'}")
                if not ok:
                    print(f"    wanted: {'; '.join(want) or 'green'}")
                git("switch", "-q", "main")
                git("branch", "-q", "-D", "case")
            print(f"selftest: {len(cases) - red} of {len(cases)} cases as planted")
            return 1 if red else 0
        finally:
            os.chdir(here)


def main():
    args = sys.argv[1:]
    if args == ["--selftest"]:
        return selftest()
    if len(args) > 1 or (args and args[0].startswith("-")):
        print(USAGE, file=sys.stderr)
        return 2
    return check(args[0] if args else "main..HEAD")


if __name__ == "__main__":
    sys.exit(main())
