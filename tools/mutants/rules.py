#!/usr/bin/env python3
"""The mutation rules: the mutants cargo-mutants 27.1.0 does not make, on
the lines a diff changes, run the way cargo-mutants runs its own: one at
a time, in place, built then tested, each file restored by a plain write
(a new mtime, so cargo rebuilds it). ISA.md's rounds 53 and 54 record the
passes run by hand before it; `tools/mutants.sh` runs this beside
cargo-mutants under a memory cap.

The source is read by `code_mask` (code, literal or comment, a
character) and by bracket depth over it. Every rule reads code only,
never a string or a comment, never test code (the item after `#[test]`,
`#[cfg(test)]` or `#[cfg(all(test, …))]`) and never a match guard, which
cargo-mutants makes true and false:
  if-true, if-false  an `if` condition, whole, over all its lines: true,
                     then false
  arm                in a fn whose return type names `ExitCode`, a match
                     arm deleted whole: the fn decides, not the arm's
                     text. Not in a match with a `_` arm (cargo-mutants
                     deletes those arms), nor a guarded arm (its guard
                     made false is the arm deleted)
  call               a call statement (`f(…);`, `x.m(…);`, `x.m(…)?;`)
                     deleted over all its lines, and one `.m(…)` cut
                     from a chain (`let line = line.trim();` -> `let
                     line = line;`); a statement's own last call is the
                     statement's to delete, and `.abs()` the abs rule's
  num                a number: an integer n -> n + 1, a float x -> x + 1.0
                     (its suffix kept)
  max-min, min-max   `.max(` <-> `.min(`
  abs                `.abs()` deleted
The if, arm and call rules never read inside a macro's brackets.

A mutant is made when a line of its span is one the diff changes, added
or beside a deletion, as cargo-mutants' `--in-diff` reads a diff. Its
name has cargo-mutants' shape, `<file>:<line>:<col>: <what> in <fn>`, so
one `exclude_re` in the config serves both tools; an excluded mutant is
counted, never run.

Verdicts, from the build and the test binaries, never an exit code alone:
  UNBUILT   the mutated file is in no `.d` file of the run's own build,
            so rustc never read it; decided first, never built
  UNVIABLE  the build fails
  CAUGHT    a test FAILED, or a test binary died: by a signal other than
            SIGKILL, or by an exit with no result line
  CAPPED    a SIGKILL: the memory cap
  TIMEOUT   the tests ran past max(60 s, 5 x the baseline's), or the
            build past 30 min
  MISSED    every binary passed; a pass count other than the baseline's
            is printed beside it
The flags are the caller's (`CARGO_ENCODED_RUSTFLAGS`, else `RUSTFLAGS`)
with any `--cap-lints` dropped, then `--cap-lints=warn`: the list
cargo-mutants' `cap_lints` builds from the same flags. rustc takes the
first of two `--cap-lints`, so a caller's own would win. No warning makes
a mutant unviable, and the two tools share one build.

usage: tools/mutants/rules.py packages TREE DIFF
       tools/mutants/rules.py build TREE OUT.json [-p PKG]... [--features LIST]
       tools/mutants/rules.py list TREE DIFF [--config TOML]
       tools/mutants/rules.py run TREE DIFF OUT.tsv [-p PKG]... [--features LIST]
                                  [--config TOML]
       tools/mutants/rules.py flags
`packages` prints the packages that test a diff: those whose `src/` holds
its files, and neuralos-nir2json when neuralos-snn is one (the trial's
set). `build` writes the features each package built with and the tree's
files rustc read. `flags` prints the caller's flags, `--cap-lints`
dropped, as `CARGO_ENCODED_RUSTFLAGS` holds them.
Exit: 0 done · 1 a file not restored · 2 usage · 4 the baseline failed.
Standard library only.
"""
import argparse
import bisect
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass

RAW = re.compile(r'[bc]?r(#*)"')
CHAR = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^'\\\n])'")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
NUM = re.compile(
    r"(?<![\w.])(\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)"
    r"(u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|i128|isize|f32|f64)?(?![\w.])"
)
TEST_ATTR = re.compile(
    r"#\[\s*(?:cfg\s*\(\s*(?:test|all\s*\([^()]*\btest\b[^()]*\))\s*\)"
    r"|(?:\w+\s*::\s*)*test)\s*\]"
)
MACRO = re.compile(r"\b(?:macro_rules\s*!\s*[A-Za-z_]\w*|[A-Za-z_]\w*!)\s*([(\[{])")
KEYWORDS = frozenset(
    "as async await break const continue dyn else enum extern false fn for if impl in "
    "let loop match mod move mut pub ref return static struct trait true type unsafe "
    "use where while yield".split()
)
SIGKILL = "(signal: 9, SIGKILL: kill)"
BUILD_LIMIT = 1800
RULES = ["if-true", "if-false", "arm", "call", "cut", "num", "max-min", "min-max", "abs"]


def code_mask(src):
    """One kind a character: "c" code, "l" a string or char literal, "m" a
    comment. blindspot.py's mask, which kept a boolean, now tells a literal
    from a comment, and reads `br"…"` and `cr"…"` as raw strings."""
    kind = ["c"] * len(src)
    i, n = 0, len(src)

    def mark(a, b, k):
        for x in range(a, min(b, n)):
            kind[x] = k

    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            mark(i, j, "m")
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            mark(i, j, "m")
            i = j
        elif c in "rbc" and (i == 0 or not ident_char(src[i - 1])) and RAW.match(src, i):
            m = RAW.match(src, i)
            end = src.find('"' + m.group(1), m.end())
            j = n if end < 0 else end + 1 + len(m.group(1))
            mark(i, j, "l")
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j += 1
            mark(i, j, "l")
            i = j
        elif c == "'":
            m = CHAR.match(src, i)
            if m:
                mark(i, m.end(), "l")
                i = m.end()
            else:
                i += 1  # a lifetime or a label
        else:
            i += 1
    return kind


def ident_char(c):
    return c.isalnum() or c == "_"


def squash(s):
    """cargo-mutants' `squash_lines`: each newline and the indent after it
    become one space."""
    return re.sub(r"\s*\n\s*", " ", s.strip()).replace("\t", " ")


@dataclass
class Fn:
    name: str
    ret: str
    open: int
    close: int


@dataclass
class Arm:
    start: int
    pat_end: int
    guard: int  # the guard's `if`, or -1
    arrow: int
    end: int


@dataclass
class Mutant:
    path: str
    start: int
    end: int
    new: str
    rule: str
    name: str
    first: int
    last: int
    excluded: bool = False


class Source:
    """A Rust file, read by `code_mask` and bracket depth."""

    def __init__(self, path, text):
        self.path, self.text, self.n = path, text, len(text)
        self.kind = code_mask(text)
        # the code alone: literals and comments blank, newlines kept
        self.cs = "".join(c if k == "c" or c == "\n" else " " for c, k in zip(text, self.kind))
        self.starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
        self.pair = self._brackets()
        self.attrs = [(m.start(), self.pair[m.end() - 1] + 1)
                      for m in re.finditer(r"#!?\[", self.cs) if m.end() - 1 in self.pair]
        self.tests = [(a, self.item_end(b)) for a, b in self.attrs
                      if TEST_ATTR.fullmatch(self.cs, a, b)]
        self.macros = [(m.start(1), self.pair[m.start(1)] + 1)
                       for m in MACRO.finditer(self.cs) if m.start(1) in self.pair]
        self.fns, self.spaces = self._items()
        self.matches = self._matches()
        self.guard_ifs = {a.guard for _, _, arms in self.matches for a in arms if a.guard >= 0}
        self.guards = [(a.guard, a.arrow) for _, _, arms in self.matches for a in arms
                       if a.guard >= 0]
        self.stmt_dots = set()

    # -- reading ----------------------------------------------------------

    def _brackets(self):
        pair, stack = {}, []
        for i, c in enumerate(self.text):
            if self.kind[i] != "c":
                continue
            if c in "([{":
                stack.append(i)
            elif c in ")]}" and stack and "([{".index(self.text[stack[-1]]) == ")]}".index(c):
                o = stack.pop()
                pair[o], pair[i] = i, o
        return pair

    def nxt(self, i):
        """The first character at or after `i` that is no blank and no comment."""
        while i < self.n and (self.kind[i] == "m" or (self.kind[i] == "c" and self.text[i].isspace())):
            i += 1
        return i

    def prv(self, i):
        """The last character at or before `i` that is no blank and no comment."""
        while i >= 0 and (self.kind[i] == "m" or (self.kind[i] == "c" and self.text[i].isspace())):
            i -= 1
        return i

    def word(self, i, w):
        return (self.cs.startswith(w, i) and (i == 0 or not ident_char(self.cs[i - 1]))
                and not ident_char(self.cs[i + len(w):i + len(w) + 1] or " "))

    def code(self, i, chars):
        return i < self.n and self.kind[i] == "c" and self.text[i] in chars

    def line(self, p):
        return bisect.bisect_right(self.starts, p)

    def skip_angle(self, i):
        """Past the `>` closing the `<` at `i`; `->` and `=>` close nothing."""
        depth = 0
        while i < self.n:
            if self.kind[i] == "c":
                c = self.text[i]
                if c in "([" and i in self.pair:
                    i = self.pair[i] + 1
                    continue
                if c == "<":
                    depth += 1
                elif c == ">" and self.text[i - 1] not in "-=":
                    depth -= 1
                    if depth == 0:
                        return i + 1
                elif c in "{;":
                    return i
            i += 1
        return i

    def item_end(self, i):
        """Past the `{…}` body or the `;` of the item that starts at `i`."""
        while i < self.n:
            if self.kind[i] == "c":
                c = self.text[i]
                if c in "([" and i in self.pair:
                    i = self.pair[i] + 1
                    continue
                if c == "{" and i in self.pair:
                    return self.pair[i] + 1
                if c == ";":
                    return i + 1
            i += 1
        return self.n

    def block_open(self, i):
        """The first `{` at depth 0 from `i`, or -1 when a `;` or a closing
        bracket comes first."""
        while i < self.n:
            if self.kind[i] == "c":
                c = self.text[i]
                if c in "([" and i in self.pair:
                    i = self.pair[i] + 1
                    continue
                if c == "{":
                    return i if i in self.pair else -1
                if c in ";)]}" or self.cs.startswith("=>", i):
                    return -1
            i += 1
        return -1

    def _items(self):
        """Each fn with a body, and the namespaces cargo-mutants names a fn
        by: inline mods, impls, traits and fns, as (name, open, close)."""
        fns, spaces = [], []
        for m in re.finditer(r"\bfn\s+([A-Za-z_]\w*)", self.cs):
            i = self.nxt(m.end())
            if self.code(i, "<"):
                i = self.nxt(self.skip_angle(i))
            if not self.code(i, "(") or i not in self.pair:
                continue
            i = self.nxt(self.pair[i] + 1)
            ret = ""
            if self.cs.startswith("->", i):
                a = j = self.nxt(i + 2)
                while j < self.n:
                    if self.kind[j] == "c":
                        c = self.text[j]
                        if c in "([" and j in self.pair:
                            j = self.pair[j] + 1
                            continue
                        if c == "<":
                            j = self.skip_angle(j)
                            continue
                        if c in "{;" or self.word(j, "where"):
                            break
                    j += 1
                ret = " ".join(self.text[a:j].split())
                i = j
            j = self.block_open(i) if not self.word(i, "where") else self.block_open(self.item_where(i))
            if j < 0:
                continue  # a declaration, no body
            fns.append(Fn(m.group(1), ret, j, self.pair[j]))
            spaces.append((m.group(1), j, self.pair[j]))
        for m in re.finditer(r"\bimpl\b", self.cs):
            p = self.prv(m.start() - 1)
            if p >= 0 and self.text[p] not in "};{]" and not self.cs[max(0, p - 5):p + 1] == "unsafe":
                continue  # `impl Trait` in a type
            i = self.nxt(m.end())
            if self.code(i, "<"):
                i = self.nxt(self.skip_angle(i))
            a, j, cut = i, i, -1
            while j < self.n:
                if self.kind[j] == "c":
                    c = self.text[j]
                    if c in "([" and j in self.pair:
                        j = self.pair[j] + 1
                        continue
                    if c == "<":
                        j = self.skip_angle(j)
                        continue
                    if c == "{" or self.word(j, "where"):
                        break
                    if cut < 0 and self.word(j, "for"):
                        cut = j
                j += 1
            o = self.block_open(j)
            if o < 0:
                continue
            if cut >= 0:
                name = "<impl {} for {}>".format(" ".join(self.text[a:cut].split()),
                                                 " ".join(self.text[cut + 3:j].split()))
            else:
                name = " ".join(self.text[a:j].split())
            spaces.append((name, o, self.pair[o]))
        for m in re.finditer(r"\b(?:trait|mod)\s+([A-Za-z_]\w*)", self.cs):
            o = self.block_open(m.end())
            if o >= 0:
                spaces.append((m.group(1), o, self.pair[o]))
        return fns, sorted(spaces, key=lambda s: s[1])

    def item_where(self, i):
        """Past a `where` clause that starts at `i`: its body's `{`."""
        while i < self.n and not self.code(i, "{;"):
            if self.code(i, "([") and i in self.pair:
                i = self.pair[i] + 1
                continue
            if self.code(i, "<"):
                i = self.skip_angle(i)
                continue
            i += 1
        return i

    def _matches(self):
        """Each `match` as (keyword, its `{`, its arms)."""
        out = []
        for m in re.finditer(r"\bmatch\b", self.cs):
            o = self.block_open(m.end())
            if o >= 0:
                out.append((m.start(), o, self._arms(o, self.pair[o])))
        return out

    def _arms(self, o, close):
        arms, i = [], self.nxt(o + 1)
        while i < close:
            k, guard = i, -1
            while k < close:
                if self.kind[k] == "c":
                    c = self.text[k]
                    if c in "([{" and k in self.pair:
                        k = self.pair[k] + 1
                        continue
                    if self.cs.startswith("=>", k):
                        break
                    if guard < 0 and self.word(k, "if"):
                        guard = k
                k += 1
            if k >= close:
                break
            arrow = k
            b = self.nxt(arrow + 2)
            if self.code(b, "{") and b in self.pair:
                e = self.nxt(self.pair[b] + 1)
                end = e + 1 if e < close and self.code(e, ",") else self.pair[b] + 1
            else:
                k = b
                while k < close:
                    if self.kind[k] == "c":
                        c = self.text[k]
                        if c in "([{" and k in self.pair:
                            k = self.pair[k] + 1
                            continue
                        if self.cs.startswith("::<", k):
                            k = self.skip_angle(k + 2)
                            continue
                        if c == ",":
                            break
                    k += 1
                end = k + 1 if k < close else k
            pat_end = self.prv((guard if guard >= 0 else arrow) - 1) + 1
            arms.append(Arm(i, pat_end, guard, arrow, end))
            i = self.nxt(end)
        return arms

    def within(self, p, spans):
        return any(a <= p < b for a, b in spans)

    def skipped(self, p, macros=False):
        """Test code, an attribute, a match guard, and for the if, arm and
        call rules a macro's brackets."""
        return (self.within(p, self.tests) or self.within(p, self.attrs)
                or self.within(p, self.guards) or (macros and self.within(p, self.macros)))

    def innermost_fn(self, p):
        best = None
        for f in self.fns:
            if f.open < p < f.close and (best is None or f.open > best.open):
                best = f
        return best

    def fn_path(self, p):
        """cargo-mutants' function name at `p`: the namespaces around it,
        joined by `::`, when a fn holds it."""
        if self.innermost_fn(p) is None:
            return None
        return "::".join(name for name, a, b in self.spaces if a < p < b)

    # -- the rules --------------------------------------------------------

    def mutant(self, rule, a, b, new, what):
        line = self.line(a)
        col = a - self.starts[line - 1] + 1
        fn = self.fn_path(a)
        name = f"{self.path}:{line}:{col}: {what}" + (f" in {fn}" if fn else "")
        return Mutant(self.path, a, b, new, rule, name, line, self.line(max(a, b - 1)))

    def ifs(self):
        for m in re.finditer(r"\bif\b", self.cs):
            p = m.start()
            if p in self.guard_ifs or self.skipped(p, macros=True):
                continue
            a = self.nxt(m.end())
            if self.word(a, "let"):
                continue
            o = self.block_open(a)
            if o < 0:
                continue
            b = self.prv(o - 1) + 1
            cond = self.text[a:b]
            if cond.strip() in ("true", "false"):
                continue
            what = f"replace if condition {squash(cond)} with"
            yield self.mutant("if-true", a, b, "true", what + " true")
            yield self.mutant("if-false", a, b, "false", what + " false")

    def arms(self):
        for f in self.fns:
            if not re.search(r"\bExitCode\b", f.ret):
                continue
            for kw, _, arms in self.matches:
                if not f.open < kw < f.close or self.innermost_fn(kw) is not f:
                    continue
                if self.skipped(kw, macros=True):
                    continue
                if any(self.text[a.start:a.pat_end] == "_" for a in arms):
                    continue
                for a in arms:
                    if a.guard < 0:
                        pat = squash(self.text[a.start:a.pat_end])
                        yield self.mutant("arm", a.start, a.end, "", f"delete match arm {pat}")

    def statements(self):
        """Each call statement: `f(…);`, `x.m(…);` or `x.m(…)?;`, in a fn,
        a macro's never."""
        stack, encl = [], {}
        for i, c in enumerate(self.text):
            if self.kind[i] != "c":
                continue
            if c in "([{" and i in self.pair:
                stack.append(i)
            elif c in ")]}" and i in self.pair:
                if stack and stack[-1] == self.pair[i]:
                    stack.pop()
            elif c == ";":
                encl[i] = stack[-1] if stack else -1
        for semi, e in encl.items():
            if e < 0 or self.text[e] != "{" or self.innermost_fn(semi) is None:
                continue
            if self.skipped(semi, macros=True):
                continue
            k = semi - 1
            while k > e:
                if self.kind[k] == "c":
                    c = self.text[k]
                    if c in ")]" and k in self.pair:
                        k = self.pair[k] - 1
                        continue
                    if c in ";}":
                        break
                k -= 1
            a = self.nxt(k + 1)
            if a >= semi:
                continue
            last = self.call_chain(a, semi)
            if last is None:
                continue
            if last >= 0:
                self.stmt_dots.add(last)
            yield self.mutant("call", a, semi + 1, "", f"delete {squash(self.text[a:semi + 1])}")

    def call_chain(self, a, semi):
        """For a statement `a..semi` that is a call chain ending in a call
        (`?` after it allowed): the `.` of that last call, -1 when it is a
        fn's call; None for any other statement."""
        e = self.prv(semi - 1)
        while e > a and self.code(e, "?"):
            e = self.prv(e - 1)
        if not self.code(e, ")") or e not in self.pair:
            return None
        f = self.prv(self.pair[e] - 1)
        if f < a or not ident_char(self.text[f]):
            return None  # a macro (`!`), a call of a call
        s = f
        while s > a and ident_char(self.text[s - 1]):
            s -= 1
        if not self.chain(a, e + 1):
            return None
        d = self.prv(s - 1)
        return d if d >= a and self.code(d, ".") else -1

    def chain(self, a, b):
        """`a..b` is a path or a parenthesized primary, then calls, indexes,
        fields, method calls and `?` only."""
        i = a
        if self.code(i, "(") and i in self.pair:
            i = self.pair[i] + 1
        else:
            m = IDENT.match(self.cs, i)
            if not m or m.group(0) in KEYWORDS:
                return False
            i = m.end()
            while True:
                j = self.nxt(i)
                if not self.cs.startswith("::", j):
                    break
                j = self.nxt(j + 2)
                if self.code(j, "<"):
                    i = self.skip_angle(j)
                    continue
                m = IDENT.match(self.cs, j)
                if not m:
                    return False
                i = m.end()
        while True:
            i = self.nxt(i)
            if i >= b:
                return True
            if self.kind[i] != "c":
                return False
            c = self.text[i]
            if c == "?":
                i += 1
            elif c in "([" and i in self.pair:
                i = self.pair[i] + 1
            elif c == "." and not self.cs.startswith("..", i):
                m = FIELD.match(self.cs, self.nxt(i + 1))
                if not m:
                    return False
                i = m.end()
                j = self.nxt(i)
                if self.cs.startswith("::", j):
                    k = self.nxt(j + 2)
                    if not self.code(k, "<"):
                        return False
                    i = self.skip_angle(k)
            else:
                return False

    def cuts(self):
        """One `.m(…)` cut from a chain, in a fn, a macro's never."""
        for i in range(self.n):
            if not self.code(i, ".") or self.text[i + 1:i + 2] == "." or self.text[i - 1:i] == ".":
                continue
            m = IDENT.match(self.cs, self.nxt(i + 1))
            if not m or m.group(0) in KEYWORDS:
                continue
            k = self.nxt(m.end())
            if self.cs.startswith("::", k):
                j = self.nxt(k + 2)
                if not self.code(j, "<"):
                    continue
                k = self.nxt(self.skip_angle(j))
            if not self.code(k, "(") or k not in self.pair:
                continue
            close = self.pair[k]
            if i in self.stmt_dots or (m.group(0) == "abs" and self.nxt(k + 1) == close):
                continue
            if self.innermost_fn(i) is None or self.skipped(i, macros=True):
                continue
            yield self.mutant("cut", i, close + 1, "", f"delete {squash(self.text[i:close + 1])}")

    def tokens(self):
        """blindspot.py's rules, a line at a time: num, max-min, min-max,
        abs."""
        for ln in range(1, len(self.starts) + 1):
            a = self.starts[ln - 1]
            b = self.starts[ln] - 1 if ln < len(self.starts) else self.n
            code = self.cs[a:b]
            for m in NUM.finditer(code):
                p = a + m.start()
                if code[m.start() - 2:m.start()] == "0x" or self.skipped(p):
                    continue
                new = bump(m.group(1), m.group(2))
                yield self.mutant("num", p, p + len(m.group(0)), new, f"replace {m.group(0)} with {new}")
            for old, new, rule in ((".max(", ".min(", "max-min"), (".min(", ".max(", "min-max")):
                for m in re.finditer(re.escape(old), code):
                    p = a + m.start()
                    if not self.skipped(p):
                        yield self.mutant(rule, p, p + len(old), new, f"replace {old} with {new}")
            for m in re.finditer(r"\.abs\(\)", code):
                p = a + m.start()
                if not self.skipped(p):
                    yield self.mutant("abs", p, p + 6, "", "delete .abs()")

    def mutants(self, lines):
        """Every rule's mutants whose span meets `lines`, in file order."""
        made = [*self.ifs(), *self.arms(), *self.statements(), *self.cuts(), *self.tokens()]
        made = [m for m in made if any(ln in lines for ln in range(m.first, m.last + 1))]
        return sorted(made, key=lambda m: (m.start, RULES.index(m.rule), m.end))


def bump(tok, suffix):
    digits = tok.replace("_", "")
    if suffix in ("f32", "f64") or "." in digits or "e" in digits.lower():
        return repr(float(digits) + 1.0) + (suffix or "")
    return str(int(digits) + 1) + (suffix or "")


def diff_lines(diff_text):
    """File -> (changed, added), line numbers on the new side. Changed is
    what cargo-mutants' `--in-diff` reads (its in_diff.rs, affected_lines):
    each added line and, around a deletion, the line before and the line
    after it. Added is the `+` lines alone. Header lines are read only
    before a file's first hunk."""
    files, cur, ln, header, removed = {}, None, None, False, False
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            cur, ln, header = None, None, True
            continue
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if m:
            ln, header, removed = int(m.group(1)), False, False
            continue
        if header:
            if line.startswith("+++ "):
                path = line[4:].rstrip("\t").strip()
                cur = None if path == "/dev/null" else path[2:] if path.startswith("b/") else path
                if cur:
                    files.setdefault(cur, (set(), set()))
            continue
        if cur is None or ln is None or line.startswith("\\"):
            continue
        changed, added = files[cur]
        if line.startswith("-"):
            removed = True
            if ln > 1:
                changed.add(ln - 1)
            continue
        if removed:
            changed.add(ln)
            removed = False
        if line.startswith("+"):
            changed.add(ln)
            added.add(ln)
        ln += 1
    return files


def metadata(tree):
    """Each workspace package: its directory under the tree, its declared
    features, its package id."""
    out = subprocess.run(["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"],
                         cwd=tree, capture_output=True, text=True, check=True).stdout
    meta = json.loads(out)
    pkgs = {}
    for p in meta["packages"]:
        d = os.path.relpath(os.path.dirname(p["manifest_path"]), meta["workspace_root"])
        pkgs[p["name"]] = {"dir": "" if d == "." else d, "features": sorted(p["features"]),
                           "id": p["id"]}
    return pkgs


def owner(path, pkgs):
    """The package whose `src/` holds `path`, or None."""
    best, best_len = None, -1
    for name, p in pkgs.items():
        src = os.path.join(p["dir"], "src") + "/"
        if path.startswith(src) and len(src) > best_len:
            best, best_len = name, len(src)
    return best


def packages(tree, diff_text):
    """The packages that test a diff: the owners of its Rust files, and
    neuralos-nir2json when neuralos-snn is one (the trial's set)."""
    pkgs = metadata(tree)
    names = {owner(p, pkgs) for p in diff_lines(diff_text) if p.endswith(".rs")} - {None}
    if "neuralos-snn" in names and "neuralos-nir2json" in pkgs:
        names.add("neuralos-nir2json")
    return sorted(names)


def caller_flags():
    """The caller's rustc flags, any `--cap-lints` dropped."""
    enc = os.environ.get("CARGO_ENCODED_RUSTFLAGS")
    if enc is not None:
        flags = enc.split("\x1f") if enc else []
    else:
        flags = os.environ.get("RUSTFLAGS", "").split()
    out, skip = [], False
    for f in flags:
        if skip:
            skip = False
        elif f == "--cap-lints":
            skip = True
        elif not f.startswith("--cap-lints="):
            out.append(f)
    return out


def cargo_env():
    env = dict(os.environ)
    env.pop("RUSTFLAGS", None)
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(caller_flags() + ["--cap-lints=warn"])
    env["CARGO_TERM_COLOR"] = "never"
    return env


def cargo_args(pkgs, features):
    args = [a for p in pkgs for a in ("-p", p)]
    return args + (["--features", features] if features else [])


def run(cmd, cwd, timeout):
    """(exit code, stdout, stderr, seconds); the code is None past the
    timeout, the process group killed."""
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True, env=cargo_env())
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err, time.time() - t0
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return None, out, err, time.time() - t0


def build(tree, pkgs, features):
    """One build of the test packages with the run's flags, its JSON read:
    (ok, built, stderr). `built` holds the features each package built
    with, the ones it declares, and the tree's files rustc read, from the
    `.d` file beside each unit this build made: a target that held
    another feature set keeps that set's `.d` files too."""
    meta = metadata(tree)
    ids = {p["id"]: name for name, p in meta.items()}
    rc, out, err, _ = run(["cargo", "test", "--no-run", "--message-format=json",
                           *cargo_args(pkgs, features)], tree, BUILD_LIMIT)
    feats, sources, root = {}, set(), os.path.realpath(tree)
    for line in out.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("reason") != "compiler-artifact":
            continue
        name = ids.get(msg["package_id"])
        if name:
            feats.setdefault(name, set()).update(msg["features"])
        for f in msg["filenames"]:
            d, base = os.path.split(f)
            stem, ext = os.path.splitext(base)
            if ext in (".rlib", ".rmeta", ".so", ".a") and stem.startswith("lib"):
                stem = stem[3:]
            elif ext:
                continue
            dep = os.path.join(d, stem + ".d")
            if not os.path.exists(dep):
                continue
            first = open(dep, encoding="utf-8").readline()
            for p in re.split(r"(?<!\\) ", first.split(": ", 1)[-1].strip()):
                p = os.path.realpath(os.path.join(root, p.replace("\\ ", " ")))
                if p.endswith(".rs") and p.startswith(root + os.sep):
                    sources.add(os.path.relpath(p, root))
    built = {"features": {n: sorted(f) for n, f in sorted(feats.items())},
             "declared": {n: meta[n]["features"] for n in sorted(feats)},
             "sources": sorted(sources)}
    return rc == 0, built, err


def passed(out):
    return sum(int(x) for x in re.findall(r"^test result: \w+\. (\d+) passed", out, re.M))


def verdict(rc, out, err, base):
    """A test run's verdict, from its test binaries."""
    if rc is None:
        return "TIMEOUT", ""
    if SIGKILL in err or rc == -signal.SIGKILL:
        return "CAPPED", ""
    if re.search(r"^test result: FAILED", out, re.M) or re.search(r"^test .* \.\.\. FAILED$", out, re.M):
        return "CAUGHT", ""
    m = re.search(r"\(signal: (\d+), ", err)
    if m:
        return "CAUGHT", f"signal {m.group(1)}"
    m = re.search(r"process didn't exit successfully: .*\(exit status: (\d+)\)", err)
    if m:
        return "CAUGHT", f"exit status {m.group(1)}"
    if rc == 0:
        n = passed(out)
        return "MISSED", "" if n == base else f"{n} passed, baseline {base}"
    return "ERROR", f"cargo exit {rc}, no test result read"


def excludes(config):
    """The config's `exclude_re`, compiled; cargo-mutants reads the same key."""
    if not config:
        return []
    with open(config, "rb") as f:
        return [re.compile(r) for r in tomllib.load(f).get("exclude_re", [])]


def all_mutants(tree, diff_text, config):
    pkgs = metadata(tree)
    out = []
    ex = excludes(config)
    for path, (changed, _) in sorted(diff_lines(diff_text).items()):
        if not path.endswith(".rs") or owner(path, pkgs) is None:
            continue
        full = os.path.join(tree, path)
        if not os.path.exists(full):
            continue
        src = Source(path, open(full, encoding="utf-8").read())
        for m in src.mutants(changed):
            m.excluded = any(r.search(m.name) for r in ex)
            out.append(m)
    return out


def run_mutants(a):
    diff_text = open(a.diff, encoding="utf-8").read()
    ms = all_mutants(a.tree, diff_text, a.config)
    out = open(a.out, "w", encoding="utf-8")
    originals = {m.path: open(os.path.join(a.tree, m.path), encoding="utf-8").read() for m in ms}
    sha = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in originals.items()}
    t0 = time.time()
    ok, built, err = build(a.tree, a.package, a.features)
    tb = time.time() - t0
    rc, text, terr, tt = (run(["cargo", "test", *cargo_args(a.package, a.features)], a.tree, BUILD_LIMIT)
                          if ok else (None, "", "", 0.0))
    base = passed(text)
    limit = max(60.0, 5 * tt)
    out.write(f"# rules.py run: {len(ms)} mutants, tests {' '.join(a.package)}"
              f"{', features ' + a.features if a.features else ''}\n")
    out.write(f"# baseline: build {tb:.1f}s, test {tt:.1f}s, {base} passed; a test run times out"
              f" at {limit:.0f}s\n")
    out.flush()
    if not ok or rc != 0:
        out.write("# BASELINE FAILED\n" + err[-3000:] + terr[-3000:])
        return 4
    sources = set(built["sources"])
    counts = {}
    try:
        for m in ms:
            tb = tt = 0.0
            note = ""
            if m.excluded:
                v = "EXCLUDED"
            elif m.path not in sources:
                v = "UNBUILT"
            else:
                full = os.path.join(a.tree, m.path)
                s = originals[m.path]
                open(full, "w", encoding="utf-8").write(s[:m.start] + m.new + s[m.end:])
                try:
                    rc, _, berr, tb = run(["cargo", "test", "--no-run", *cargo_args(a.package, a.features)],
                                          a.tree, BUILD_LIMIT)
                    if rc is None:
                        v, note = "TIMEOUT", "build"
                    elif rc != 0:
                        v, note = ("CAPPED", "build") if SIGKILL in berr or rc == -signal.SIGKILL else ("UNVIABLE", "")
                    else:
                        rc, text, terr, tt = run(["cargo", "test", *cargo_args(a.package, a.features)],
                                                 a.tree, limit)
                        v, note = verdict(rc, text, terr, base)
                finally:
                    open(full, "w", encoding="utf-8").write(s)
            counts[v] = counts.get(v, 0) + 1
            row = f"{v}\t{m.name}\t{m.path}\t{m.first}\t{m.last}\t{m.rule}\t{tb:.1f}\t{tt:.1f}\t{note}"
            out.write(row + "\n")
            out.flush()
            print(f"{v:<9} {m.name}" + (f"  ({note})" if note else ""), flush=True)
    finally:
        for p, s in originals.items():
            open(os.path.join(a.tree, p), "w", encoding="utf-8").write(s)
    out.write("# " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f" of {len(ms)}\n")
    for p, s in originals.items():
        now = hashlib.sha256(open(os.path.join(a.tree, p), encoding="utf-8").read().encode()).hexdigest()
        if now != sha[p]:
            print(f"rules.py: {p} not restored", file=sys.stderr)
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser(prog="rules.py", add_help=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("packages")
    p.add_argument("tree")
    p.add_argument("diff")
    p = sub.add_parser("build")
    p.add_argument("tree")
    p.add_argument("out")
    p.add_argument("-p", "--package", action="append", default=[])
    p.add_argument("--features", default="")
    p = sub.add_parser("list")
    p.add_argument("tree")
    p.add_argument("diff")
    p.add_argument("--config")
    p = sub.add_parser("run")
    p.add_argument("tree")
    p.add_argument("diff")
    p.add_argument("out")
    p.add_argument("-p", "--package", action="append", default=[])
    p.add_argument("--features", default="")
    p.add_argument("--config")
    sub.add_parser("flags")
    a = ap.parse_args()
    if a.cmd == "packages":
        print("\n".join(packages(a.tree, open(a.diff, encoding="utf-8").read())))
        return 0
    if a.cmd == "flags":
        sys.stdout.write("\x1f".join(caller_flags()))
        return 0
    if a.cmd == "build":
        ok, built, err = build(a.tree, a.package, a.features)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(built, f, indent=1)
        if not ok:
            sys.stderr.write(err[-3000:])
            return 4
        return 0
    if a.cmd == "list":
        ms = all_mutants(a.tree, open(a.diff, encoding="utf-8").read(), a.config)
        for m in ms:
            print(f"{m.name}\t{m.rule}" + ("\texcluded" if m.excluded else ""))
        print(f"# {len(ms)} mutants, {sum(m.excluded for m in ms)} excluded", file=sys.stderr)
        return 0
    return run_mutants(a)


if __name__ == "__main__":
    sys.exit(main())
