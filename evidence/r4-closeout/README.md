# R4 close-out — leg 3 re-pins + judge leg (2026-08-20/21)

The R4(iii) leg 3 (invivo + null_patches onto the harness) and R4(iv)
re-pin evidence, banked at commit d3311a7 + this tree. Protocol:
`evidence/r4-baselines/README.md` § Re-pin protocol.

## null_patches — PERFECT re-pin

- Rebuild: `cargo run -p neuralos-rt --release --example null_patches`
- `null_patches_r4iii.log`: H2 terminal diff 87,119 cells (pre-registered
  value exact); dose ×10 + flip ×3, every file 87,119 cells, S2 clean.
- `null_shas_{before,after}.txt`: ALL 13 regenerated `models/null-*.gguf`
  **byte-identical** to the session-I banked artifacts (sha256 before vs
  after the regeneration run — `diff` empty).

## judge p0 double-run pair — stats EXACT vs the banked pins

- Rebuild: `bash tools/build_fork.sh`, then for each side (base =
  `models/Ternary-Bonsai-4B-Q2_0.gguf`, loop =
  `models/Ternary-Bonsai-4B-Q2_0-loop.gguf`, sha 24ffe5f3… verified
  before the runs) ×2:
  `NEURALOS_DUMP=1 fork-build/llama.cpp/build/bin/llama-completion -m <model> -p '1 2 3 4 5 6 7' -n 12 --temp 0 --top-k 0 --top-p 1.0 --min-p 0.0 --seed 42 -no-cnv -c 512 -t 4`
- Determinism: base run1≡run2 byte-identical (dumps + continuation);
  loop run1≡run2 byte-identical.
- Continuation: base ≡ loop byte-identical (`1 2 3 4 5 6 7 8 9 10 11 1`).
- `judge_delta` (Rust port of the Python original; `cargo run -p
  neuralos-rt --release --example judge_delta -- p0_base_run1.err
  p0_loop_run1.err` — byte-compatible, deleted after parity):
  **12 steps | 0/12 argmax flips | overlap 10/10 | max |Δ| +0.4207 |
  mean 0.0779** — identical to the r4-baselines banked numbers
  (incl. step 9 carrying the max). Same fork binary as the banking
  (no rebuild → no FP jitter; session-f's binary differed at the 4th
  decimal, 15.0547 vs 15.0541 — cross-build variation, recorded).

## hybrid_invivo — the r4-baselines bar was STALE (root cause banked)

The README §3 bar cited session-H pins (H(i,c)=41,190, clamp 69.796%,
export 67,309 cells). That record is **unreachable on any tree at or
after the sH2 drive redesign (2026-08-19)** — which predates the
banking (2fb7c5b, 2026-08-20):

- The drive is single-truncated-pass (first min(STEPS, tokens) of the
  stream). On session H's 332-token corpus (recovered verbatim:
  `git show 3b512df:evidence/session-f-judge/README.md`, sha 2d64e907…,
  1,024 B — the on-disk README has since grown to 1,057 B) the run
  executes 332 steps; the frozen 400-step init cycle then swallows the
  entire learn tier → 0 events, NaN quarters, `adaptation alive:
  FAIL (frozen)`. Banked falsifier: `h1_invivo_r4iii.log`.
- NOT a refactor bug: Tier-1 semantics scale exactly with steps
  (totals 6,874 ×6 ≈ 41,024 vs H's 40,863; clamp fraction 69.796%
  IDENTICAL; G0 ordering preserved).
- **The correct like-for-like bar is the H2 record at default args**
  (pinned corpus, 2,000 tokens): drive stats RMS 0.0447 / k=10,060.46 /
  clamp 69.477% (568,321/818,000) / dim-199 1,786×; T1 H(i,c)=41,555 >
  H(i,z)=30,724, H(c,z)=29,013; learn 43.65 Hz, flips 1,112,771,
  Hamming 33.30%; record: `evidence/session-h2/run.log`.
- Status: **DONE — MATCH (2026-08-21).** The full H2 re-run at default
  args completed in 21,065.6 s (idle box; the banked 29,668 s ran
  under load), peak RSS 6,595 MB (banked 6,593). Banked log:
  `h2_invivo_r4iii.log`. Diff vs `evidence/session-h2/run.log` with
  wall/RSS stripped: **empty except the Tier-2 export section** —
  session H2 ran with the `export` argv flag; the re-run was plain
  mode deliberately (running export mode would overwrite
  `models/Ternary-Bonsai-4B-Q2_0-invivo*.gguf`, the session-I
  adjudication artifacts, recoverable only via another ~6–8 h run).
  The export machinery itself is pinned elsewhere: hybrid_loop's
  surgery re-pinned byte-identical (export sha 24ffe5f3…, leg 1) and
  null_patches exercises encode+write+re-read across 13 files
  (13/13 byte-identical, above). Every verdict-bearing line — drive
  stats, T1 tables, learn-tier counters (events 25,322,176 · flips
  1,112,771), Hamming 0.3330, mechanism label, verdict block —
  byte-identical. **This match closes R4.**

## r4-baselines/README.md amendment

§3's invivo row and the § Re-pin table's `hybrid_invivo` line are
amended by this directory: the H1 bar is void (stale, root cause
above); the H2 record is the re-pin target. The judge-leg row is
CLOSED by this directory (all stats exact). **The H2 re-pin MATCHED
(byte-identical modulo the by-design-not-run export tier) — R4 is
closed.**
