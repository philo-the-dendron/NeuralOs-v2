# Step-5 burn window — orchestration (PREREG §3/§8 authority)

Order matters: the NULL family of replicate r shuffles r's OWN ON
diff — the ON run must finish first. Wall-times: idle-box H2 re-pin
was 5.9 h/run (21,065.6 s); banked under load 8.24 h. Judge ≈ 23–50 s
per prompt-run (2–4 min per file).

## Chain

```bash
# 0. FREE-arm pre-judge re-verify (seconds; voids if any ck drifted):
cargo run -p neuralos-rt --release --example step5_aggregate -- --verify-free

# 1. OFF driven r0 — the toggle proof (~6–8 h; export asserts byte-≡ base):
cargo run -p neuralos-rt --release --example hybrid_invivo -- --off
#    k-check assert fires pre-burn (r0 k == 10060.46 probe expectation)

# 2. OFF identity tripwires r1/r2 (seconds each; byte-≡ base asserted):
cargo run -p neuralos-rt --release --example hybrid_invivo -- --identity 1
cargo run -p neuralos-rt --release --example hybrid_invivo -- --identity 2

# 3. ON r0 — the H2 re-pin (export MUST equal banked 71f2518a…):
cargo run -p neuralos-rt --release --example hybrid_invivo -- --window 0

# 4. nulls r0 → judge r0 (then r1, r2 the same way — ON first, nulls after):
cargo run -p neuralos-rt --release --example step5_nulls -- --replicate 0
B=evidence/step5-readout/burn
for f in models/null-r0-s*.gguf; do
  d=$(basename "$f" .gguf)                       # null-r0-s201, no .gguf
  tools/run_prompts.sh "$f" "$B/$d"              # single-run
done
tools/run_prompts.sh models/Ternary-Bonsai-4B-Q2_0-invivo-r0.gguf "$B/on-r0" --double
tools/run_prompts.sh models/Ternary-Bonsai-4B-Q2_0-invivo-off-r0.gguf "$B/off-r0" --double

# 5. ON r1 → nulls r1 → judge r1; then ON r2 → nulls r2 → judge r2
#    (same shapes as 3–4 with --window 1 / --replicate 1, …)

# 6. DOMAIN arm (report-only; norm-unit drive, window 0):
cargo run -p neuralos-rt --release --example hybrid_invivo -- --domain-corrected
tools/run_prompts.sh models/Ternary-Bonsai-4B-Q2_0-invivo-domain.gguf "$B/domain" --double

# 7. identity tripwire judge legs (double-run, contamination check):
tools/run_prompts.sh models/Ternary-Bonsai-4B-Q2_0-invivo-identity-r1.gguf "$B/identity-r1" --double
tools/run_prompts.sh models/Ternary-Bonsai-4B-Q2_0-invivo-identity-r2.gguf "$B/identity-r2" --double

# 8. FREE arm — the banked ck files, judged (single-run, family protocol):
tools/run_prompts.sh models/Ternary-Bonsai-4B-Q2_0-invivo-ck400.gguf  "$B/free-ck400"
tools/run_prompts.sh models/Ternary-Bonsai-4B-Q2_0-invivo-ck800.gguf  "$B/free-ck800"
tools/run_prompts.sh models/Ternary-Bonsai-4B-Q2_0-invivo-ck1200.gguf "$B/free-ck1200"

# 9. The verdict — mechanical, from the bands (PARTIAL ROOT guard on <3):
cargo run -p neuralos-rt --release --example step5_aggregate -- "$B"
```

## Rules of the window

- **ON-r MUST precede nulls-r** (the null family shuffles r's own diff).
- **ON r0 re-pin is load-bearing**: its export must byte-equal the
  banked H2 sha (asserted in-pipeline) — the new code path's proof it
  reproduces the frozen artifact. A miss voids the session's code, not
  the banked record.
- **Unbanked guard**: every step-5 output is arm-named; writes to any
  banked path panic before the write (harness `assert_unbanked`).
- **Void protocol §6**: exit≠0 / empty dump = void; re-run once; twice
  → excluded, denominators adjusted, noted.
- **Double-run arms**: ON/OFF/IDENTITY/DOMAIN (run1==run2 asserted by
  run_prompts.sh). NULL + FREE: single-run (family determinism).
- **Parallel track (the sequencing ruling)**: the exporter session
  (step 4) and R18 (4b) slot into the burn window — different stack,
  zero shared surface; they never touch models/, the judge, or the
  evidence dirs above.
- Seeds live ONLY in `null_seeds.txt` (201–250 by decade per ON
  window: r0–r2 201–230, r3 231–240, r4 241–250 — the escalation
  extension committed before any escalation arm ran).

## Expected totals

7 driven-class runs (ON ×5, OFF-r0, DOMAIN) · 50 nulls +
judged files × 2–4 min judge · aggregator renders the verdict table
mechanically (no hand-counted verdicts).

> Amended post-escalation (n=3 → n=5); pre-escalation expectations:
> PREREG.md §4 / §5 escalation ladder + the ISA amendment (2026-08-26).

## Power-loss / reboot recovery

Nothing in the burn can corrupt a banked artifact (unbanked guard,
in-pipeline asserts) — worst case is wall-time, never correctness.

1. **Forensics (seconds):** `ls evidence/step5-readout/burn/logs/` —
   the last leg with a `start` line and no `done` line is the
   interrupted one. HALT present = a command failed (read it); no HALT
   + dead process = crash/reboot.
2. **Paranoia pin (~1 min):** re-sha the five banked files against
   PREP.md (base, invivo, ck400/800/1200) — proves the frozen record
   untouched:
   ```bash
   sha256sum models/Ternary-Bonsai-4B-Q2_0{,-invivo,-invivo-ck400,-invivo-ck800,-invivo-ck1200}.gguf
   ```
3. **Resume from the interrupted leg only.** Deterministic runs re-pin
   byte-identical; judge legs overwrite their own outputs. Never
   re-run `all` blindly. Interrupted mid-rep with the ON export
   already complete (the final sha line in the log is the witness —
   existence on disk alone is NOT)? Run that replicate's nulls + judge
   commands manually (chain steps 4–5) and skip the 6–8 h re-run.
4. **Relaunch detached:** `tmux new -s burn 'tools/burn.sh <leg>'` —
   then continue the remaining legs in order.
