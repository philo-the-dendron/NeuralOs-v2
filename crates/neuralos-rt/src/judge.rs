//! The judge-log tools — Rust port of `tools/delta.py` +
//! `tools/margin_census.py` (deleted after parity was proven; the
//! Python sessions' banked outputs remain the record of record).
//!
//! Parses `NEURALOS_DUMP step=<n> ... id:logit id:logit ...` lines
//! (comma decimals — the fork's locale) from judge `.err` logs and
//! builds the mechanical comparison tables of the adjudication
//! protocol. Output is byte-compatible with the Python originals —
//! pinned by tests against the banked logs and their recorded
//! Python outputs (`evidence/r4-closeout/`, `evidence/session-i-*`).
//!
//! Insertion order is load-bearing: the Python dict preserved
//! capture order and `max()` keeps the FIRST entry on value ties,
//! so a step is a `Vec<(id, value)>`, never a map.

use std::collections::BTreeMap;
use std::fs;

/// One step's captured top-k: `(token id, logit)`, capture order.
pub type DumpStep = Vec<(u32, f64)>;

/// All steps of one judge log, keyed by step index (sorted on
/// iteration — the Python `sorted(steps)` of record).
pub type Dump = BTreeMap<u32, DumpStep>;

/// Pre-registered knife-edge threshold θ (margin_census of record).
pub const THETA: f64 = 0.05;

/// Parse one dump line; `None` when the line is not a dump line.
///
/// Mirrors `re.match(r'NEURALOS_DUMP step=(\d+)')` (line-start
/// anchor) + `re.findall(r'(\d+):(-?\d+,\d+)')` over the whole
/// line — the `n_out=10:` field deliberately matches nothing (no
/// comma-decimal value follows its colon).
pub fn parse_dump_line(line: &str) -> Option<(u32, DumpStep)> {
    let rest = line.strip_prefix("NEURALOS_DUMP step=")?;
    let n = rest.bytes().take_while(u8::is_ascii_digit).count();
    if n == 0 {
        return None;
    }
    let step: u32 = rest[..n].parse().ok()?;
    let mut pairs = DumpStep::new();
    scan_pairs(line, &mut pairs);
    Some((step, pairs))
}

/// Hand-rolled `(\d+):(-?\d+,\d+)` scanner (no regex dependency).
///
/// Equivalent to the regex on these logs: maximal digit runs share
/// one end position, so skipping a failed run skips only positions
/// that would fail identically. Duplicate ids keep the dict
/// semantics — first position, last value.
fn scan_pairs(line: &str, out: &mut DumpStep) {
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i >= b.len() || b[i] != b':' {
            continue;
        }
        let id: u32 = match line[start..i].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let d0 = i + 1;
        let mut j = d0;
        if j < b.len() && b[j] == b'-' {
            j += 1;
        }
        let digits0 = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if digits0 == j || j >= b.len() || b[j] != b',' {
            continue;
        }
        j += 1;
        let digits1 = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == digits1 {
            continue;
        }
        let mut val = String::with_capacity(j - d0);
        for &c in &b[d0..j] {
            val.push(if c == b',' { '.' } else { c as char });
        }
        let val: f64 = match val.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        match out.iter_mut().find(|(id0, _)| *id0 == id) {
            Some(slot) => slot.1 = val,
            None => out.push((id, val)),
        }
        i = j;
    }
}

/// Read + parse a judge `.err` log. Exits loudly (status 1) if the
/// file is unreadable — same contract as the examples of record.
pub fn parse_dump_file(path: &str) -> Dump {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut dump = Dump::new();
    for line in text.lines() {
        if let Some((step, pairs)) = parse_dump_line(line) {
            dump.insert(step, pairs);
        }
    }
    dump
}

/// First-inserted argmax id (Python `max(dict, key=...)` semantics).
pub fn argmax(step: &DumpStep) -> Option<u32> {
    max_entry(step, |_| false).map(|(id, _)| id)
}

fn max_entry(step: &DumpStep, skip: impl Fn(u32) -> bool) -> Option<(u32, f64)> {
    let mut best: Option<(u32, f64)> = None;
    for &(id, v) in step {
        if skip(id) {
            continue;
        }
        match best {
            Some((_, bv)) if v <= bv => {}
            _ => best = Some((id, v)),
        }
    }
    best
}

fn value(step: &DumpStep, id: u32) -> Option<f64> {
    step.iter().find(|(i, _)| *i == id).map(|(_, v)| *v)
}

fn shared_ids(a: &DumpStep, b: &DumpStep) -> Vec<u32> {
    a.iter()
        .filter(|(i, _)| b.iter().any(|(j, _)| i == j))
        .map(|(i, _)| *i)
        .collect()
}

/// The mechanical delta table: summary line + one row per step.
/// Panics loudly when the two logs carry different step sets (the
/// Python `assert b.keys() == p.keys()` of record).
pub fn delta_table(base: &Dump, patched: &Dump) -> String {
    let keys: Vec<u32> = base.keys().copied().collect();
    assert!(
        keys == patched.keys().copied().collect::<Vec<_>>(),
        "step sets differ"
    );
    let n = keys.len();
    let mut flips = 0;
    let mut overlaps = Vec::with_capacity(n);
    let mut deltas = Vec::with_capacity(n);
    for &s in &keys {
        let (b, p) = (&base[&s], &patched[&s]);
        let (ba, pa) = (argmax(b).expect("nonempty step"), argmax(p).expect("nonempty step"));
        if ba != pa {
            flips += 1;
        }
        let shared = shared_ids(b, p);
        overlaps.push(shared.len());
        deltas.push(match shared.len() {
            0 => f64::NAN,
            _ => shared
                .iter()
                .map(|&t| (value(p, t).expect("shared") - value(b, t).expect("shared")).abs())
                .fold(f64::NEG_INFINITY, f64::max),
        });
    }
    let mean_ov = overlaps.iter().sum::<usize>() as f64 / overlaps.len() as f64;
    let max_d = deltas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_d = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let mut out = String::new();
    out.push_str(&format!(
        "steps {n} | argmax flips {flips}/{n} | overlap min {}/10 mean {mean_ov:.2} | max|d| shared {max_d:+.4} | mean|d| {mean_d:.4}\n",
        overlaps.iter().min().copied().unwrap_or(0)
    ));
    for &s in &keys {
        let (b, p) = (&base[&s], &patched[&s]);
        let (ba, pa) = (argmax(b).expect("nonempty step"), argmax(p).expect("nonempty step"));
        let flag = if ba != pa { "FLIP" } else { "same" };
        let db = match value(p, ba) {
            Some(pv) => format!("{:+.4}", (pv - value(b, ba).expect("argmax")).abs()),
            None => "  n/a ".to_string(),
        };
        out.push_str(&format!(
            "  step {s:2}: argmax {ba}->{pa} {flag} | top1logit base {:+.4} d {db}\n",
            value(b, ba).expect("argmax")
        ));
    }
    out
}

/// The margin census over `(tag, dump)` pairs: small-margin tail
/// (margin < 0.5), then the knife-edge set (margin < θ).
pub fn margin_census(inputs: &[(&str, &Dump)]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:>28} {:>4} {:>6} {:>6} {:>8}  knife?\n",
        "file", "step", "top1", "top2", "margin"
    ));
    let mut knives: Vec<(&str, u32, f64)> = Vec::new();
    for (tag, dump) in inputs {
        for (&s, step) in dump.iter() {
            let mut vals: Vec<f64> = step.iter().map(|(_, v)| *v).collect();
            if vals.len() < 2 {
                continue;
            }
            vals.sort_by(|a, b| b.partial_cmp(a).expect("finite logits"));
            let margin = vals[0] - vals[1];
            let is_knife = margin < THETA;
            if is_knife {
                knives.push((tag, s, margin));
            }
            if margin < 0.5 {
                let top1 = argmax(step).expect("nonempty step");
                let top2 = max_entry(step, |i| i == top1)
                    .map(|(id, _)| id)
                    .expect("≥2 entries");
                out.push_str(&format!(
                    "{tag:>28} {s:>4} {top1:>6} {top2:>6} {margin:>+8.4}  {}\n",
                    if is_knife { "◄ KNIFE" } else { "" }
                ));
            }
        }
    }
    out.push('\n');
    out.push_str(&format!(
        "knife-edge set (margin < {THETA}): {} entries across all files\n",
        knives.len()
    ));
    for (tag, s, m) in knives {
        out.push_str(&format!("  {tag} step {s}: {m:+.4}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim banked line: evidence/r4-closeout/p0_base_run1.err
    /// step 0 (same line as the r4-baselines banking).
    const BANKED_STEP0: &str = "NEURALOS_DUMP step=0 n_out=10: 220:15,0541 198:11,0496 271:10,2333 13:8,5976 11:8,0046 481:7,5243 760:7,3965 2303:7,3548 23:7,2531 715:7,1897";

    #[test]
    fn parse_reads_banked_line_and_ignores_n_out() {
        let (step, pairs) = parse_dump_line(BANKED_STEP0).expect("banked line parses");
        assert_eq!(step, 0);
        assert_eq!(pairs.len(), 10);
        assert_eq!(pairs[0], (220, 15.0541));
        assert_eq!(pairs[9], (715, 7.1897));
        assert!(pairs.iter().all(|(id, _)| *id != 10), "n_out=10 must not be a pair");
        assert_eq!(argmax(&pairs), Some(220));
    }

    #[test]
    fn parse_handles_negative_comma_decimals() {
        let (step, pairs) =
            parse_dump_line("NEURALOS_DUMP step=7 n_out=10: 16:-11,0496 220:0,0001").expect("parses");
        assert_eq!(step, 7);
        assert_eq!(pairs[0], (16, -11.0496));
        assert_eq!(pairs[1], (220, 0.0001));
        assert_eq!(argmax(&pairs), Some(220));
    }

    #[test]
    fn non_dump_lines_are_none() {
        assert!(parse_dump_line("0.05.804.998 I llama_completion: prompt: '1 2 3'").is_none());
        assert!(parse_dump_line("").is_none());
    }

    #[test]
    fn ties_keep_first_inserted_including_top2() {
        let step: DumpStep = vec![(5, 1.0), (3, 1.0), (9, 0.5)];
        assert_eq!(argmax(&step), Some(5));
        let top2 = max_entry(&step, |i| i == 5).map(|(id, _)| id);
        assert_eq!(top2, Some(3));
    }

    #[test]
    fn delta_table_matches_banked_python_output() {
        // The real judge pair of r4-closeout, parsed from the banked
        // logs, against the exact 13 lines tools/delta.py printed on
        // them (recorded in evidence/r4-closeout/README.md + commit).
        let root = |p: &str| {
            ["evidence/", "../../evidence/"]
                .iter()
                .map(|d| format!("{d}{p}"))
                .find(|f| std::path::Path::new(f).exists())
                .unwrap_or_else(|| panic!("banked log {p} not found"))
        };
        let base = parse_dump_file(&root("r4-closeout/p0_base_run1.err"));
        let patched = parse_dump_file(&root("r4-closeout/p0_loop_run1.err"));
        let expected = concat!(
            "steps 12 | argmax flips 0/12 | overlap min 10/10 mean 10.00 | max|d| shared +0.4207 | mean|d| 0.0779\n",
            "  step  0: argmax 220->220 same | top1logit base +15.0541 d +0.0006\n",
            "  step  1: argmax 23->23 same | top1logit base +20.4577 d +0.0281\n",
            "  step  2: argmax 220->220 same | top1logit base +15.4098 d +0.0058\n",
            "  step  3: argmax 24->24 same | top1logit base +21.1368 d +0.0665\n",
            "  step  4: argmax 220->220 same | top1logit base +15.5383 d +0.0004\n",
            "  step  5: argmax 16->16 same | top1logit base +19.8030 d +0.0726\n",
            "  step  6: argmax 15->15 same | top1logit base +23.4061 d +0.0068\n",
            "  step  7: argmax 220->220 same | top1logit base +14.2458 d +0.0309\n",
            "  step  8: argmax 16->16 same | top1logit base +18.5944 d +0.0032\n",
            "  step  9: argmax 16->16 same | top1logit base +20.5545 d +0.4207\n",
            "  step 10: argmax 220->220 same | top1logit base +15.5465 d +0.0213\n",
            "  step 11: argmax 16->16 same | top1logit base +22.2924 d +0.0112\n",
        );
        assert_eq!(delta_table(&base, &patched), expected);
    }

    #[test]
    fn delta_table_n_a_branch_and_flip_flag() {
        let mut base = Dump::new();
        base.insert(0, vec![(1, 1.0), (2, 0.5)]);
        let mut patched = Dump::new();
        patched.insert(0, vec![(3, 9.0), (2, 0.7)]);
        assert_eq!(
            delta_table(&base, &patched),
            concat!(
                "steps 1 | argmax flips 1/1 | overlap min 1/10 mean 1.00 | max|d| shared +0.2000 | mean|d| 0.2000\n",
                "  step  0: argmax 1->3 FLIP | top1logit base +1.0000 d   n/a \n",
            )
        );
    }

    #[test]
    fn delta_table_panics_on_step_set_mismatch() {
        let mut base = Dump::new();
        base.insert(0, vec![(1, 1.0)]);
        let mut patched = Dump::new();
        patched.insert(1, vec![(1, 1.0)]);
        let _ = std::panic::catch_unwind(|| delta_table(&base, &patched))
            .expect_err("must panic");
    }

    #[test]
    fn margin_census_formats_knife_and_tail_rows() {
        let mut dump = Dump::new();
        dump.insert(0, vec![(16, 10.0), (220, 9.98)]);
        dump.insert(1, vec![(15, 10.0), (16, 3.0)]);
        let out = margin_census(&[("p0", &dump)]);
        assert_eq!(
            out,
            concat!(
                "                        file step   top1   top2   margin  knife?\n",
                "                          p0    0     16    220  +0.0200  ◄ KNIFE\n",
                "\n",
                "knife-edge set (margin < 0.05): 1 entries across all files\n",
                "  p0 step 0: +0.0200\n",
            )
        );
    }

    #[test]
    fn margin_census_hides_large_margins() {
        let mut dump = Dump::new();
        dump.insert(0, vec![(16, 10.0), (220, 5.0)]);
        let out = margin_census(&[("p0", &dump)]);
        assert_eq!(
            out,
            concat!(
                "                        file step   top1   top2   margin  knife?\n",
                "\n",
                "knife-edge set (margin < 0.05): 0 entries across all files\n",
            )
        );
    }
}
