//! Step-5 CALIBRATION — the lesion/graft positive control
//! (`evidence/step5-calibration/PREREG.md` is the spec — its current draft;
//! this file is its instrument).
//!
//! The question (§1): could the frozen step-5 readout have seen a
//! STRUCTURED, function-bearing change in the surgery window, if one had
//! been there? Eight arm files go through the SAME splice machinery as
//! every step-5 arm, each against its own dose-matched null families:
//!
//! - **LESION-h**, h ∈ {0,1,2,3} — window rows `[128h, 128h+128)` × cols
//!   `[0,512)` set to 0. One head's q-projection over the window's input
//!   dims: coherent, localized loss of function; classes `(−1→0)`/`(+1→0)`.
//! - **GRAFT-k**, k ∈ {1,18,35} — window rows `[0,128)` × cols `[0,512)`
//!   replaced by the same rows/cols of `blk.k.attn_q.weight`. Trained
//!   weights in the wrong layer; any of the six off-diagonal classes.
//! - **IDENTITY** — the base slice spliced onto the base through the same
//!   pipeline. 0 changed cells, sha ≡ base: the contamination tripwire.
//!
//! Null families (§3): **SCAT** shuffles the arm's diff over the whole
//! 512×512 window (step 5's own null construction); **LOCAL** shuffles it
//! inside the arm's own 128×512 block and embeds it there, separating
//! structure from concentration. Grafts get both — LOCAL is the band §6
//! decides on, SCAT rides in the verdict string. Lesions get SCAT only: a
//! lesion zeroes every nonzero cell of its block, so a local null IS the
//! lesion (§3).
//!
//! Two modes, and the split is the point:
//!
//! - `--measure` writes nothing to `models/`. It decodes the base window
//!   and the three graft sources, prints every arm file's changed-cell
//!   count and full per-class composition, and applies the §4 reader
//!   checks — the ones that can only fire on a broken reader. Its stdout
//!   is teed to `measure.log` and sha-pinned in `SHA256SUMS`; §4 is filled
//!   from that log, never from a transcription.
//! - `--generate` writes the eight arm files, their null families and
//!   `arms.txt`, teeing to `generate.log` so a splice panic is on record.
//!   It REFUSES while §10 reads `_unstamped_` — the stamp is a mechanical
//!   precondition here, not an operator convention (§7 step 4).
//!
//! The §4 checks are deliberately NOT a dose band: under independence any
//! class mix gives 41,580–44,760 changed graft cells, so a ±5% band could
//! only fire on real correlation between blk.0 and blk.k — a true fact
//! about the model, not a reader fault (reviewer B10). The measured graft
//! count is reported against that interval as a RESULT.
//!
//! Seeds come from `evidence/step5-calibration/seeds.txt` by decade (§7:
//! SCAT 301–370, LOCAL 371–400), first five per family, the rest reserved
//! and unused (there is no escalation, §3). Never minted at run time.
//!
//! Usage:
//!   cargo run -p neuralos-rt --release --example step5_calibration -- --measure
//!   cargo run -p neuralos-rt --release --example step5_calibration -- --generate
//!
//! Splice, null and guard logic are `harness::{splice_and_verify,
//! dose_matched_null, assert_unbanked}` verbatim — the step-5 family's own
//! implementation, not a second one (no new example by cloning).

use neuralos_rt::harness::{
    assert_unbanked, decode_slice, dose_matched_null, splice_and_verify, tix, ExperimentParams,
};
use neuralos_rt::{GgufFile, GGML_TYPE_Q2_0};
use neuralos_snn::Trit;
use std::collections::BTreeMap;
use std::io::Write;

const BASE: &str = "models/Ternary-Bonsai-4B-Q2_0.gguf";
/// PREP.md pin, re-verified before `--measure` and before `--generate`
/// (§2; a mismatch is a §7 kill criterion).
const BASE_SHA: &str = "4e0bf8b737b0431552f8c2c97695ab7c0cb214c94bcdeb4f5f267e67ddf28b8b";
const EVIDENCE_DIR: &str = "evidence/step5-calibration";
const PREREG_FILE: &str = "evidence/step5-calibration/PREREG.md";
const SEEDS_FILE: &str = "evidence/step5-calibration/seeds.txt";
const ARMS_FILE: &str = "evidence/step5-calibration/arms.txt";
const MEASURE_LOG: &str = "evidence/step5-calibration/measure.log";
const GENERATE_LOG: &str = "evidence/step5-calibration/generate.log";
const SHA256SUMS: &str = "evidence/step5-calibration/SHA256SUMS";

/// The banked window census (§2/§4, evidence/r4-baselines/loop_run2.log),
/// in `harness::tix` order: −1 · 0 · +1. The four head blocks must decode
/// to exactly this sum — a check that can only fire on a broken reader.
const WINDOW_CENSUS: [u64; 3] = [83_253, 95_802, 83_089];
/// §4 independence interval for a graft's changed-cell count: reported as
/// a RESULT, never a gate. Below it is measured correlation between the
/// layers and goes to §8.
const GRAFT_INDEPENDENCE: (u64, u64) = (41_580, 44_760);
/// Five per family, no escalation (§3: both metrics are one-sided, so more
/// nulls could only demote a band).
const NULLS_PER_FAMILY: usize = 5;
/// 4B head width: 4096 output rows / 32 heads.
const HEAD_ROWS: usize = 128;
/// §7 seed decades: 10 slots of 10, first 301–310, last 391–400.
const SEED_SLOTS: usize = 10;
const SEED_BASE: u64 = 301;

/// Graft sources, in §7 seed order. A const table, not a runtime format:
/// `ExperimentParams::tensor` is `&'static str` and the three layers are
/// pre-registered constants (§7 step 2, reviewer N6).
const GRAFT_LAYERS: [(usize, &str); 3] = [
    (1, "blk.1.attn_q.weight"),
    (18, "blk.18.attn_q.weight"),
    (35, "blk.35.attn_q.weight"),
];
const LESION_HEADS: [usize; 4] = [0, 1, 2, 3];

/// Trit bucket names for composition lines (index = `harness::tix`).
const TNAME: [&str; 3] = ["-1", "0", "+1"];

/// Stdout, teed to a log on disk line by line, with the log's sha written
/// to `SHA256SUMS` when the run ends — including when it ends in a panic,
/// so a splice failure is on record (§7 step 3).
struct Tee {
    file: std::fs::File,
    path: &'static str,
}

impl Tee {
    fn new(path: &'static str) -> Self {
        std::fs::create_dir_all(EVIDENCE_DIR)
            .unwrap_or_else(|e| panic!("cannot create {EVIDENCE_DIR}: {e}"));
        let file =
            std::fs::File::create(path).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
        Self { file, path }
    }
    fn say(&mut self, line: &str) {
        println!("{line}");
        let _ = writeln!(self.file, "{line}");
        let _ = self.file.flush();
    }
}

impl Drop for Tee {
    fn drop(&mut self) {
        // Never panic here: this also runs while unwinding from a §7 kill.
        let _ = self.file.flush();
        match sha256_try(self.path) {
            Some(sha) => update_sha256sums(self.path, &sha),
            None => eprintln!("WARNING: could not sha {} for {SHA256SUMS}", self.path),
        }
    }
}

macro_rules! say {
    ($t:expr, $($arg:tt)*) => { $t.say(&format!($($arg)*)) };
}

/// sha of a file through the system tool — the `hybrid_invivo` pattern.
/// `None` when the tool fails (missing file, bad exit): the callers that
/// must be loud assert on it, the `Drop` path warns instead of panicking.
fn sha256_try(f: &str) -> Option<String> {
    let o = std::process::Command::new("sha256sum")
        .arg(f)
        .output()
        .ok()?;
    if !o.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&o.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();
    (sha.len() == 64).then_some(sha)
}

/// sha256 of an in-memory buffer through the same system tool. The decoded
/// block is serialized one byte per trit (`harness::tix`: 0 = −1, 1 = 0,
/// 2 = +1), so the digest identifies the DECODED content, independent of
/// q2_0 packing and of where in the file it came from.
fn sha256_of_bytes(data: &[u8]) -> String {
    use std::process::{Command, Stdio};
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("sha256sum runs");
    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(data)
        .expect("write to sha256sum");
    let o = child.wait_with_output().expect("sha256sum completes");
    assert!(
        o.status.success(),
        "sha256sum over a buffer failed: {}",
        o.status
    );
    let sha = String::from_utf8_lossy(&o.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();
    assert_eq!(sha.len(), 64, "sha256sum: unexpected output {sha:?}");
    sha
}

/// Trits as one byte each, in `harness::tix` order — the digest input above.
fn trit_bytes(cells: &[Trit]) -> Vec<u8> {
    cells.iter().map(|t| tix(*t) as u8).collect()
}

/// The loud form: a missing file must never read as a drifted base.
fn sha256_of(f: &str) -> String {
    sha256_try(f).unwrap_or_else(|| panic!("sha256sum {f} failed — file missing or unreadable?"))
}

/// Rewrite `SHA256SUMS` with `path`'s line replaced (or added), other lines
/// kept, sorted by name. Best-effort: it runs from `Drop`.
fn update_sha256sums(path: &str, sha: &str) {
    let name = path.rsplit('/').next().unwrap_or(path);
    let mut lines: Vec<String> = std::fs::read_to_string(SHA256SUMS)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter(|l| l.split_whitespace().nth(1) != Some(name))
        .map(str::to_string)
        .collect();
    lines.push(format!("{sha}  {name}"));
    lines.sort_by_key(|l| l.split_whitespace().nth(1).unwrap_or("").to_string());
    let mut text = lines.join("\n");
    text.push('\n');
    if let Err(e) = std::fs::write(SHA256SUMS, text) {
        eprintln!("WARNING: cannot write {SHA256SUMS}: {e}");
    }
}

/// §2: the base is re-verified against the PREP.md pin BEFORE it is
/// decoded, in both modes. A drifted base is a §7 kill.
fn verify_base_sha(t: &mut Tee) {
    let sha = sha256_of(BASE);
    assert_eq!(
        sha, BASE_SHA,
        "base {BASE} sha {sha} != §2 pin {BASE_SHA} — KILL (§7): the base drifted"
    );
    say!(t, "base    : {BASE}");
    say!(t, "          sha {sha} == §2 pin : PASS");
}

/// §7 step 4: no arm file exists before the stamp. Enforced here rather
/// than trusted to the operator — `--generate` is the irreversible half.
fn assert_stamped(t: &mut Tee) {
    let text = std::fs::read_to_string(PREREG_FILE)
        .unwrap_or_else(|e| panic!("cannot read {PREREG_FILE}: {e}"));
    let stamp = text
        .split("## 10. Stamp")
        .nth(1)
        .unwrap_or_else(|| panic!("{PREREG_FILE}: § 10 Stamp section not found"))
        .trim();
    assert!(
        !stamp.is_empty() && !stamp.contains("_unstamped_"),
        "{PREREG_FILE} §10 is UNSTAMPED — refusing to generate arm files \
         (§7 step 4: the principal's stamp precedes --generate)"
    );
    say!(t, "stamp   : {PREREG_FILE} §10 carries a stamp : PASS");
}

/// A graft source as the GGUF itself reports it: the tensor name resolved
/// from the container's own tensor list (never the const-table string),
/// its data offset, and — filled once the window is decoded — the sha of
/// its 128×512 source block. Dose cannot tell blk.17 from blk.18 (the
/// three layers land within 0.45% of each other), so provenance is
/// recorded and the three block shas are asserted pairwise distinct
/// (§4, reviewer D5).
struct GraftSrc {
    k: usize,
    /// Resolved from `info.name`, then asserted equal to the const table.
    resolved: String,
    offset: u64,
    block_sha: String,
    census: [u64; 3],
}

/// §4 GRAFT reader check, part one: every graft source carries the host
/// tensor's exact q2_0 layout, confirmed BEFORE a byte of it is read. Any
/// difference is a §7 kill. Returns each source's resolved name and data
/// offset, as the container reports them.
fn assert_graft_layouts(t: &mut Tee, p: &ExperimentParams) -> Vec<(usize, String, u64)> {
    let buf = std::fs::read(BASE).unwrap_or_else(|e| panic!("cannot read {BASE}: {e}"));
    let f = GgufFile::parse(&buf).expect("GGUF container must parse");
    let find = |name: &str| {
        f.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("tensor {name} not found in {BASE}"))
    };
    let host = find(p.tensor);
    assert_eq!(host.ty, GGML_TYPE_Q2_0, "host tensor must be q2_0");
    say!(
        t,
        "layout  : host {} — ty q2_0 · dims {:?} · {} B",
        p.tensor,
        host.dims,
        p.tensor_bytes()
    );
    let mut resolved = Vec::with_capacity(GRAFT_LAYERS.len());
    for (k, name) in GRAFT_LAYERS {
        let info = find(name);
        assert_eq!(
            info.ty, host.ty,
            "graft source {name} is ggml type {} but host {} is {} (q2_0) — \
             layouts differ, KILL (§7), REFUSING to read it",
            info.ty, p.tensor, host.ty
        );
        assert_eq!(
            info.dims, host.dims,
            "graft source {name} dims {:?} != host {} dims {:?} — \
             shapes differ, KILL (§7), REFUSING to read it",
            info.dims, p.tensor, host.dims
        );
        let data = f.tensor_data(info).expect("graft tensor window in bounds");
        assert_eq!(
            data.len(),
            p.tensor_bytes(),
            "graft source {name} byte window {} != {} (rows × row_bytes) — \
             layout is not the host's, KILL (§7), REFUSING to read it",
            data.len(),
            p.tensor_bytes()
        );
        say!(
            t,
            "          graft k={k} {} — ty q2_0 · dims {:?} · {} B · data offset {} : MATCHES HOST",
            info.name,
            info.dims,
            data.len(),
            info.offset
        );
        resolved.push((k, info.name.clone(), info.offset));
    }
    resolved
}

/// Decode the §2 window (rows 0..512 × cols 0..512) of an arbitrary layer's
/// `attn_q` through `harness::decode_slice` — the same strict decode path
/// (type / dims / byte-size asserts) the host window goes through. Only the
/// tensor name moves.
fn decode_layer_window(tensor: &'static str, p: &ExperimentParams) -> Vec<Trit> {
    let mut q = p.clone();
    q.tensor = tensor;
    decode_slice(BASE, &q)
}

/// Census of a trit run in `harness::tix` order (−1 · 0 · +1).
fn census(cells: &[Trit]) -> [u64; 3] {
    cells.iter().fold([0u64; 3], |mut c, t| {
        c[tix(*t)] += 1;
        c
    })
}

/// LESION-h: head h's rows of the window, zeroed across all 512 input cols.
fn lesion_patch(base: &[Trit], h: usize, n: usize) -> Vec<Trit> {
    let mut out = base.to_vec();
    let (lo, hi) = (h * HEAD_ROWS * n, (h + 1) * HEAD_ROWS * n);
    assert!(
        hi <= out.len(),
        "lesion head {h} outside the {n}×{n} window"
    );
    out[lo..hi].fill(Trit::Zero);
    out
}

/// GRAFT-k: the window's first head rows replaced by layer k's same rows.
fn graft_patch(base: &[Trit], src: &[Trit], n: usize) -> Vec<Trit> {
    assert_eq!(base.len(), src.len(), "graft source window size mismatch");
    let mut out = base.to_vec();
    let cells = HEAD_ROWS * n;
    out[..cells].copy_from_slice(&src[..cells]);
    out
}

/// Changed-cell count and the full per-class composition of `arm` vs `base`.
fn composition(base: &[Trit], arm: &[Trit]) -> (u64, BTreeMap<(usize, usize), u64>) {
    let mut classes: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut cells = 0u64;
    for (b, a) in base.iter().zip(arm.iter()) {
        if a != b {
            *classes.entry((tix(*b), tix(*a))).or_default() += 1;
            cells += 1;
        }
    }
    (cells, classes)
}

fn fmt_comp(classes: &BTreeMap<(usize, usize), u64>) -> String {
    if classes.is_empty() {
        return "(none)".to_string();
    }
    classes
        .iter()
        .map(|((f, to), n)| format!("({}→{} : {n})", TNAME[*f], TNAME[*to]))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Which arm this is, and the block-scoped facts §4 checks about it.
enum Kind {
    /// head index; the block's own census, for the reported zero fraction
    Lesion(usize, [u64; 3]),
    /// the graft source, as the container reports it (§4 provenance)
    Graft(GraftSrc),
    Identity,
}

/// One arm file: its name, its patched window, its kind, and the seed slot
/// of each null family it carries (§7 decade order).
struct Arm {
    name: String,
    patch: Vec<Trit>,
    kind: Kind,
    /// (family, seed slot) — LESION: SCAT only. GRAFT: SCAT + LOCAL.
    /// IDENTITY: none (a tripwire, not a banded comparison).
    families: Vec<(Family, usize)>,
    /// The family §6 decides on, marked `*` in arms.txt. LESION: SCAT.
    /// GRAFT: LOCAL, the deconfounded one (§3). IDENTITY: none. The
    /// generator knows what an arm IS, so the reader never has to guess
    /// a primary from the arm's name.
    primary: Option<Family>,
}

/// Which dose-matched null construction a family uses (§3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    /// Whole-window shuffle — step 5's own null.
    Scat,
    /// Shuffle inside the arm's own 128×512 block, embedded there.
    Local,
}

impl Family {
    fn label(self) -> &'static str {
        match self {
            Family::Scat => "scat",
            Family::Local => "local",
        }
    }
}

/// The eight arm files of §3, built in §7 seed-decade order. Built once and
/// used by both modes — `--measure` and `--generate` must never disagree
/// about what an arm IS.
fn build_arms(base: &[Trit], p: &ExperimentParams, srcs: &[(usize, String, u64)]) -> Vec<Arm> {
    let n = p.n;
    let block = HEAD_ROWS * n;
    let mut arms: Vec<Arm> = Vec::with_capacity(LESION_HEADS.len() + GRAFT_LAYERS.len() + 1);
    for (i, h) in LESION_HEADS.into_iter().enumerate() {
        arms.push(Arm {
            name: format!("lesion-h{h}"),
            patch: lesion_patch(base, h, n),
            kind: Kind::Lesion(h, census(&base[h * block..(h + 1) * block])),
            families: vec![(Family::Scat, i)],
            primary: Some(Family::Scat),
        });
    }
    for (i, (k, tensor)) in GRAFT_LAYERS.into_iter().enumerate() {
        let src = decode_layer_window(tensor, p);
        let (rk, resolved, offset) = srcs
            .get(i)
            .unwrap_or_else(|| panic!("graft provenance missing for k={k}"));
        assert_eq!(*rk, k, "graft provenance out of order: {rk} != {k}");
        assert_eq!(
            resolved, tensor,
            "graft k={k}: the container resolved {resolved:?}, the const table says {tensor:?} — \
             KILL (§7)"
        );
        arms.push(Arm {
            name: format!("graft-k{k}"),
            kind: Kind::Graft(GraftSrc {
                k,
                resolved: resolved.clone(),
                offset: *offset,
                block_sha: sha256_of_bytes(&trit_bytes(&src[..block])),
                census: census(&src[..block]),
            }),
            patch: graft_patch(base, &src, n),
            families: vec![
                (Family::Scat, LESION_HEADS.len() + i),
                (Family::Local, LESION_HEADS.len() + GRAFT_LAYERS.len() + i),
            ],
            primary: Some(Family::Local),
        });
    }
    arms.push(Arm {
        name: "identity".to_string(),
        patch: base.to_vec(),
        kind: Kind::Identity,
        families: Vec::new(),
        primary: None,
    });
    arms
}

/// §4 provenance (reviewer D5): the three graft source blocks must be three
/// DIFFERENT blocks. Their doses land within 0.45% of each other, so the
/// dose cannot tell blk.17 from blk.18 — only the decoded block's own
/// digest can.
fn assert_distinct_graft_blocks(t: &mut Tee, arms: &[Arm]) {
    let mut seen: Vec<(usize, &str)> = Vec::new();
    for arm in arms {
        if let Kind::Graft(ref src) = arm.kind {
            if let Some((other, _)) = seen.iter().find(|(_, sha)| *sha == src.block_sha) {
                panic!(
                    "graft blocks of blk.{} and blk.{} share sha {} — the same tensor was read \
                     twice, KILL (§7)",
                    other, src.k, src.block_sha
                );
            }
            seen.push((src.k, &src.block_sha));
        }
    }
    say!(
        t,
        "provenance: {} graft source blocks, shas pairwise distinct : PASS",
        seen.len()
    );
}

/// §4 placement check (reviewer D6): every changed cell of this arm lies in
/// `[lo, hi)` of the row-major window and nothing outside it moved, so the
/// arm's block IS the rows it claims. The census-sum identity cannot see
/// this: it is invariant under a permutation of the four blocks.
fn assert_placement(name: &str, base: &[Trit], patch: &[Trit], lo: usize, hi: usize) {
    assert!(
        hi <= base.len() && lo < hi,
        "{name}: block [{lo}, {hi}) outside the window"
    );
    let outside = (0..base.len())
        .filter(|i| !(lo..hi).contains(i))
        .find(|&i| patch[i] != base[i]);
    assert!(
        outside.is_none(),
        "{name}: cell {} changed outside block [{lo}, {hi}) — KILL (§7): wrong placement",
        outside.unwrap_or_default()
    );
    let n = hi - lo;
    let width = n / HEAD_ROWS; // cells per window row
    assert_eq!(
        width * HEAD_ROWS,
        n,
        "{name}: a block of {n} cells is not {HEAD_ROWS} whole rows"
    );
    assert_eq!(
        lo % width,
        0,
        "{name}: block starts mid-row ({lo} is not a multiple of the row width {width})"
    );
    let (r0, r1) = (lo / width, hi / width);
    assert_eq!(
        r1 - r0,
        HEAD_ROWS,
        "{name}: block spans rows [{r0}, {r1}) = {} rows, not {HEAD_ROWS}",
        r1 - r0
    );
}

/// The §4 reader checks for one arm — the ones that can only fire on a
/// broken reader. Every failure is a §7 kill. Returns the changed-cell
/// count; the graft interval line is a RESULT, never a gate.
fn check_arm(t: &mut Tee, arm: &Arm, base: &[Trit], block: usize) -> u64 {
    let (changed, classes) = composition(base, &arm.patch);
    say!(t, "{}: {changed} cells", arm.name);
    say!(t, "    composition {}", fmt_comp(&classes));
    match arm.kind {
        Kind::Lesion(h, block_census) => {
            let (lo, hi) = (h * block, (h + 1) * block);
            // §4 placement (reviewer D6): the census-sum identity is
            // invariant under a permutation of the four blocks, so prove
            // block h IS window rows [128h, 128h+128) — every changed cell
            // inside them, none outside.
            assert_placement(&arm.name, base, &arm.patch, lo, hi);
            let patched = &arm.patch[lo..hi];
            assert_eq!(
                patched.len(),
                block,
                "{}: block decodes to {} trits, not {block} — KILL (§7)",
                arm.name,
                patched.len()
            );
            assert!(
                patched.iter().all(|c| *c == Trit::Zero),
                "{}: patched block is not all-zero — KILL (§7)",
                arm.name
            );
            let zero_frac = block_census[1] as f64 / block as f64;
            say!(
                t,
                "    block census {block_census:?} of {block} · zero fraction {zero_frac:.4} \
                 (measurement, not a gate) · patched block all-zero : PASS"
            );
        }
        Kind::Graft(ref src) => {
            let k = src.k;
            assert_placement(&arm.name, base, &arm.patch, 0, block);
            say!(
                t,
                "    provenance: {} · data offset {} · block sha {}",
                src.resolved,
                src.offset,
                src.block_sha
            );
            let src_census = src.census;
            let sum: u64 = src_census.iter().sum();
            assert_eq!(
                sum, block as u64,
                "{}: blk.{k} block census sums to {sum}, not {block} — KILL (§7)",
                arm.name
            );
            assert!(
                changed > 0,
                "{}: blk.{k} block is identical to blk.0's — the same tensor was read twice, \
                 KILL (§7)",
                arm.name
            );
            let (lo, hi) = GRAFT_INDEPENDENCE;
            let where_ = if changed < lo {
                "BELOW the interval — measured correlation between the layers, a RESULT for §8"
            } else if changed > hi {
                "ABOVE the interval — a RESULT for §8"
            } else {
                "inside the interval"
            };
            say!(
                t,
                "    blk.{k} block census {src_census:?} sums to {block} : PASS · changed {changed} \
                 vs independence [{lo}, {hi}] : {where_}"
            );
        }
        Kind::Identity => {
            assert_eq!(
                changed, 0,
                "identity changed {changed} cells, not 0 — KILL (§7)"
            );
            say!(
                t,
                "    0 changed cells : PASS — the tripwire is transparent"
            );
        }
    }
    changed
}

/// LOCAL-family feasibility (§3): a block-local null must place the arm's
/// whole diff inside the arm's own 128×512 block, drawing each class from
/// that block's OWN src-class cells. `dose_matched_null` asserts the exact
/// dose and panics if a pool runs short — but that panic would land
/// mid-`--generate`, after gigabytes are written. Checked here too, in the
/// mode that writes nothing.
fn local_feasible(base_block: &[Trit], arm_block: &[Trit]) -> (bool, String) {
    let mut demand = [0u64; 3];
    let mut avail = [0u64; 3];
    for (b, a) in base_block.iter().zip(arm_block.iter()) {
        avail[tix(*b)] += 1;
        if a != b {
            demand[tix(*b)] += 1;
        }
    }
    let ok = (0..3).all(|c| demand[c] <= avail[c]);
    let detail = (0..3)
        .map(|c| format!("{}: {}/{}", TNAME[c], demand[c], avail[c]))
        .collect::<Vec<_>>()
        .join(" · ");
    (
        ok,
        format!(
            "local pools (demand/available) {detail} : {}",
            if ok {
                "FEASIBLE"
            } else {
                "SHORT — LOCAL family impossible"
            }
        ),
    )
}

/// §7 seed decades: slot i is 301+10i … 310+10i. Selection is BY POSITION
/// and the map is asserted — `harness::decade_for` cannot serve these (it
/// asserts r ∈ 0..=4 and the 201–250 range), so this is the generator's own
/// selector with its own range assert (reviewer N1).
fn decade(seeds: &[u64], slot: usize) -> Vec<u64> {
    assert!(
        slot < SEED_SLOTS,
        "seed slot {slot} outside the {SEED_SLOTS} pre-registered decades"
    );
    assert!(
        seeds.len() >= 10 * (slot + 1),
        "{SEEDS_FILE} carries only {} entries — slot {slot} needs {}",
        seeds.len(),
        10 * (slot + 1)
    );
    let d: Vec<u64> = seeds.iter().copied().skip(10 * slot).take(10).collect();
    let lo = SEED_BASE + 10 * slot as u64;
    let hi = lo + 9;
    assert!(
        d.iter().all(|&s| (lo..=hi).contains(&s)),
        "seed map violation: slot {slot} must use {lo}–{hi}, got {d:?}"
    );
    d
}

/// Read the pre-committed seeds. Integer lines only; comments skipped.
fn load_seeds() -> Vec<u64> {
    let text = std::fs::read_to_string(SEEDS_FILE).unwrap_or_else(|e| {
        panic!("cannot read {SEEDS_FILE}: {e} — seeds are pre-committed, never minted")
    });
    let seeds: Vec<u64> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            l.parse::<u64>()
                .unwrap_or_else(|e| panic!("{SEEDS_FILE}: non-integer seed line {l:?}: {e}"))
        })
        .collect();
    assert!(
        seeds.len() >= 10 * SEED_SLOTS,
        "{SEEDS_FILE}: need ≥{} seeds (one decade per family, §7), found {}",
        10 * SEED_SLOTS,
        seeds.len()
    );
    seeds
}

/// The §4 window check: the four head blocks decode to exactly the banked
/// census. It can only fire on a broken reader — a wrong tensor, a wrong
/// window, a wrong stride.
fn check_window(t: &mut Tee, base: &[Trit], block: usize) {
    let mut sum = [0u64; 3];
    for h in LESION_HEADS {
        let c = census(&base[h * block..(h + 1) * block]);
        let n: u64 = c.iter().sum();
        assert_eq!(
            n, block as u64,
            "head {h} block decodes to {n} trits, not {block} — KILL (§7)"
        );
        say!(
            t,
            "          head {h}: census {c:?} · zero fraction {:.4}",
            c[1] as f64 / block as f64
        );
        for (s, v) in sum.iter_mut().zip(c) {
            *s += v;
        }
    }
    assert_eq!(
        sum, WINDOW_CENSUS,
        "the four head blocks sum to {sum:?}, not the banked window census {WINDOW_CENSUS:?} — \
         KILL (§7): the reader is not reading the window of record"
    );
    say!(
        t,
        "          heads sum to {sum:?} == banked window census : PASS"
    );
}

/// `--measure` — the §4 reader checks and dose accounting. Writes nothing
/// to `models/`; tees to `measure.log`.
fn measure() {
    let mut t = Tee::new(MEASURE_LOG);
    let p = ExperimentParams::default();
    let block = HEAD_ROWS * p.n;
    say!(
        t,
        "=== step-5 calibration: --measure (PREREG §4 reader checks; writes nothing to models/) ==="
    );
    verify_base_sha(&mut t);
    let srcs = assert_graft_layouts(&mut t, &p);
    let base = decode_slice(BASE, &p);
    say!(
        t,
        "window  : {} rows 0..{} × cols 0..{} — {} cells",
        p.tensor,
        p.n,
        p.n,
        base.len()
    );
    check_window(&mut t, &base, block);
    say!(t, "");

    let arms = build_arms(&base, &p, &srcs);
    assert_distinct_graft_blocks(&mut t, &arms);
    let mut rows: Vec<(String, u64, String)> = Vec::with_capacity(arms.len());
    for arm in &arms {
        let changed = check_arm(&mut t, arm, &base, block);
        if arm.families.iter().any(|(f, _)| *f == Family::Local) {
            let (feasible, detail) = local_feasible(&base[..block], &arm.patch[..block]);
            say!(t, "    {detail}");
            assert!(
                feasible,
                "{}: the LOCAL family cannot be built inside the block — KILL before any write",
                arm.name
            );
        }
        let (_, classes) = composition(&base, &arm.patch);
        rows.push((arm.name.clone(), changed, fmt_comp(&classes)));
    }

    // The §4 table, ready to paste — filled from this log, never from a
    // recollection of it (§7 step 3).
    say!(t, "");
    say!(t, "--- PREREG §4 table ---");
    say!(t, "| Arm | measured | composition (from→to : n) |");
    say!(t, "|---|---|---|");
    for (name, changed, comp) in &rows {
        let label = if let Some(h) = name.strip_prefix("lesion-h") {
            format!("LESION-{h}")
        } else if let Some(k) = name.strip_prefix("graft-k") {
            format!("GRAFT-{k}")
        } else {
            name.to_uppercase()
        };
        say!(t, "| {label} | {changed} | {comp} |");
    }
    say!(t, "");
    say!(
        t,
        "reader checks: {} arms PASS — no §4 check fired, no §7 kill",
        rows.len()
    );
    say!(
        t,
        "measure done — nothing written under models/. Log: {MEASURE_LOG}, sha pinned in {SHA256SUMS}."
    );
}

/// `--generate` — the eight arm files, their null families, and arms.txt.
/// Refuses while §10 is unstamped; tees to `generate.log`.
fn generate() {
    let mut t = Tee::new(GENERATE_LOG);
    let p = ExperimentParams::default();
    let block = HEAD_ROWS * p.n;
    say!(
        t,
        "=== step-5 calibration: --generate (PREREG §7 step 5) ==="
    );
    verify_base_sha(&mut t);
    assert_stamped(&mut t);
    let srcs = assert_graft_layouts(&mut t, &p);
    let base = decode_slice(BASE, &p);
    check_window(&mut t, &base, block);
    let seeds = load_seeds();
    let arms = build_arms(&base, &p, &srcs);
    assert_distinct_graft_blocks(&mut t, &arms);

    let mut manifest: Vec<String> = Vec::with_capacity(arms.len());
    for arm in &arms {
        let changed = check_arm(&mut t, arm, &base, block);
        let out = format!("models/cal-{}.gguf", arm.name);
        assert_unbanked(&out);
        splice_and_verify(BASE, &out, &arm.patch, Some(&base), &p);
        say!(t, "    S2 clean → {out}");
        if matches!(arm.kind, Kind::Identity) {
            // §7 kill criterion: the tripwire's file must BE the base.
            let (sha, base_sha) = (sha256_of(&out), sha256_of(BASE));
            assert_eq!(
                sha, base_sha,
                "identity export sha {sha} != base {base_sha} — KILL (§7): \
                 the pipeline is not transparent"
            );
            say!(t, "    identity sha == base ({base_sha:.16}…) : PASS");
        }

        let mut fams: Vec<String> = Vec::new();
        for &(family, slot) in &arm.families {
            for &seed in decade(&seeds, slot).iter().take(NULLS_PER_FAMILY) {
                let null = match family {
                    Family::Scat => dose_matched_null(&base, &arm.patch, seed),
                    Family::Local => {
                        // Shuffle inside the arm's own 128×512 block, then
                        // embed at that block: same count, same composition,
                        // positions random INSIDE the footprint (§3).
                        let shuffled = dose_matched_null(&base[..block], &arm.patch[..block], seed);
                        let mut full = base.clone();
                        full[..block].copy_from_slice(&shuffled);
                        full
                    }
                };
                let nout = format!("models/cal-{}-{}-s{seed}.gguf", arm.name, family.label());
                assert_unbanked(&nout);
                splice_and_verify(BASE, &nout, &null, Some(&base), &p);
                let n = null.iter().zip(base.iter()).filter(|(a, b)| a != b).count() as u64;
                assert_eq!(
                    n,
                    changed,
                    "{} {} s{seed}: dose {n} != arm {changed}",
                    arm.name,
                    family.label()
                );
                say!(
                    t,
                    "    {} s{seed}: {n} cells (exact dose) · S2 clean → {nout}",
                    family.label()
                );
            }
            fams.push(format!(
                "{}{}",
                family.label(),
                if arm.primary == Some(family) { "*" } else { "" }
            ));
        }
        assert_eq!(
            fams.iter().filter(|f| f.ends_with('*')).count(),
            usize::from(arm.primary.is_some()),
            "{}: arms.txt must mark exactly one primary family (none for the tripwire)",
            arm.name
        );
        manifest.push(
            format!("{} {}", arm.name, fams.join(" "))
                .trim_end()
                .to_string(),
        );
    }

    let mut text = String::from(
        "# Step-5 calibration arm files — written by step5_calibration --generate.\n\
         # One line per arm: <arm-stem> [family …], in PREREG §7 decade order;\n\
         # the family §6 decides on carries a trailing *. The tripwire has none.\n\
         # Files are models/cal-<arm>.gguf and models/cal-<arm>-<family>-s<seed>.gguf.\n\
         # The aggregator reads THIS file; no arm list is ever hardcoded (§7 step 7).\n",
    );
    for line in &manifest {
        text.push_str(line);
        text.push('\n');
    }
    std::fs::write(ARMS_FILE, &text).unwrap_or_else(|e| panic!("cannot write {ARMS_FILE}: {e}"));
    say!(t, "");
    say!(t, "arms.txt: {} arm(s) → {ARMS_FILE}", manifest.len());
    say!(
        t,
        "generate done — judge chain next (§7 step 6). Log: {GENERATE_LOG}, sha in {SHA256SUMS}."
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--measure") if args.len() == 1 => measure(),
        Some("--generate") if args.len() == 1 => generate(),
        _ => {
            eprintln!("usage: step5_calibration --measure | --generate");
            std::process::exit(2);
        }
    }
}
