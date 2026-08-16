//! Stage 4, session 4 — THE GATE: tokenize → generate → judge.
//!
//! Greedy (deterministic) decode through the full stack on the real
//! Bonsai-1.7B-Q1_0: GGUF container → embedded Qwen2 BPE tokenizer →
//! 28-block integer forward (incremental KV cache) → tied-embedding
//! argmax. Run in RELEASE:
//!
//! ```text
//! cargo run -p neuralos-rt --release --example bonsai_generate
//! ```
//!
//! # The prompt set (chosen BEFORE any run — no prompt-fishing)
//!
//! Five STRICT prompts, each judged by decoded-TEXT prefix against the
//! pinned expected continuation (ids resolved at runtime; ids are never
//! the contract because Qwen splits digits — " 8" is two tokens):
//!
//! 1. `"1 2 3 4 5 6 7"` → `" 8"` — pure pattern continuation at single
//!    digit tokens; the cheapest thing a counting-adjacent 1-bit 1.7B
//!    should hold.
//! 2. `"10 11 12 13"` → `" 14"` — the same skill with multi-digit
//!    numbers (per-digit tokens, carry structure) — strictly harder
//!    than 1.
//! 3. `"one two three four"` → `" five"` — word counting; `Ġfive` is a
//!    single token.
//! 4. `"Monday Tuesday Wednesday"` → `" Thursday"` — weekday sequence
//!    knowledge; `ĠThursday` is a single token.
//! 5. `"The capital of France is"` → `" Paris"` — the honestly-at-risk
//!    one: factual recall, not pattern. A miss here is a real result,
//!    not a fluke.
//!
//! One CHAT demonstrator (NOT in the verdict): the embedded qwen3 chat
//! template (rendered per its own fragments — see `render_chat`), user
//! "Count from 1 to 5.", judged structurally only (non-empty, mostly
//! printable, clean stop) because a 1-bit 1.7B's chat reply legitimately
//! starts several ways. The principal pin: **YES = 5/5 strict — the
//! chat demonstrator never carries the verdict.**
//!
//! # Verdict
//!
//! `STAGE 4 GATE: YES` iff all five strict prompts match their expected
//! text prefix AND every run's residual stream stayed under the
//! norm-soundness rail. Anything else prints NO with the evidence and
//! exits 1 — a NO is a recorded result (the bridge stops with shipped
//! artifacts per gate doctrine), never a reason to shop for prompts.

use neuralos_rt::{GgufFile, MetadataValue, Qwen3, Tokenizer, RESIDUAL_SOUND_MAX};
use std::time::Instant;

const MAX_POS: usize = 512;
const GEN_CAP: usize = 12;

/// (prompt, expected continuation prefix) — the strict set.
const STRICT: &[(&str, &str)] = &[
    ("1 2 3 4 5 6 7", " 8"),
    ("10 11 12 13", " 14"),
    ("one two three four", " five"),
    ("Monday Tuesday Wednesday", " Thursday"),
    ("The capital of France is", " Paris"),
];

const CHAT_USER: &str = "Count from 1 to 5.";

/// Render the single-user-turn, no-system, no-tools,
/// `add_generation_prompt` case of the EMBEDDED qwen3 template. The
/// template (read from the file this session) has no default system
/// prompt on the non-tools path (a `{%- if messages[0].role == 'system'
/// %}` with no else) and its generation block pre-closes the think
/// section: `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Each
/// fragment used here is asserted present in the template verbatim —
/// provenance without a Jinja engine.
fn render_chat(tmpl: &str) -> String {
    let user_block = "<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n'";
    let gen_block = "'<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'";
    let system_guard = "messages[0].role == 'system'";
    assert!(tmpl.contains(user_block), "template user block fragment absent");
    assert!(tmpl.contains(gen_block), "template generation block fragment absent");
    assert!(
        tmpl.contains(system_guard),
        "template system guard absent (derivation assumption broken)"
    );
    format!(
        "<|im_start|>user\n{CHAT_USER}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

struct RunResult {
    prompt: String,
    prompt_ids: Vec<u32>,
    text: String,
    n_gen: usize,
    stopped_on_eos: bool,
    prefill_tps: f64,
    decode_tps: f64,
    residual: i32,
}

fn generate(model: &Qwen3, tok: &Tokenizer, prompt: &str) -> RunResult {
    let prompt_ids = tok.encode(prompt);
    let n_prompt = prompt_ids.len();
    let eos = tok
        .special_id("<|im_end|>")
        .expect("special <|im_end|> present (pinned by tests)");
    let mut ses = model.new_session();
    let t0 = Instant::now();
    let h = model
        .prefill(&mut ses, &prompt_ids)
        .expect("prefill on the real model");
    let prefill = t0.elapsed();
    let mut last = *h.last().expect("nonempty prompt");
    let mut gen: Vec<u32> = Vec::new();
    let mut stopped_on_eos = false;
    let t1 = Instant::now();
    loop {
        if gen.len() >= GEN_CAP {
            break;
        }
        let (id, _) = model.argmax_logit(&last).expect("argmax");
        if id == eos {
            stopped_on_eos = true;
            break;
        }
        gen.push(id);
        last = model.step(&mut ses, id).expect("decode step");
    }
    let decode = t1.elapsed();
    let text = tok.decode(&gen).expect("generated ids decode");
    let prefill_tps = n_prompt as f64 / prefill.as_secs_f64();
    let decode_tps = gen.len() as f64 / decode.as_secs_f64().max(1e-9);
    let residual = ses.max_abs_residual();
    RunResult {
        prompt: prompt.to_string(),
        prompt_ids,
        text,
        n_gen: gen.len(),
        stopped_on_eos,
        prefill_tps,
        decode_tps,
        residual,
    }
}

/// Mostly-printable structural check for the chat demonstrator:
/// non-empty after trimming specials/whitespace, ≥80% printable
/// characters, stopped cleanly (eos or cap).
fn chat_structural_pass(r: &RunResult) -> bool {
    let cleaned = r
        .text
        .replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .replace("<think>", "")
        .replace("</think>", "");
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return false;
    }
    let printable = trimmed
        .chars()
        .filter(|c| c.is_ascii_graphic() || matches!(c, ' ' | '\n' | '\t'))
        .count();
    let frac = printable as f64 / trimmed.chars().count() as f64;
    frac >= 0.8 && (r.stopped_on_eos || r.n_gen == GEN_CAP)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Bonsai-1.7B-Q1_0.gguf".into());
    let buf = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let t0 = Instant::now();
    let f = GgufFile::parse(&buf).expect("container parses");
    let tok = Tokenizer::from_gguf(&f).expect("embedded tokenizer loads");
    let model = Qwen3::load(&f, MAX_POS).expect("model loads");
    println!(
        "loaded: {} tensors, {} vocab tokens, {:.1?}",
        f.tensors.len(),
        tok.len(),
        t0.elapsed()
    );
    let eos = tok.special_id("<|im_end|>").expect("eos present");
    println!("eos <|im_end|> = {eos} (pinned: 151645)");

    let mut strict_pass = 0_usize;
    let mut rail_fail = 0_usize;
    let rail: u32 = u32::try_from(RESIDUAL_SOUND_MAX).unwrap_or(u32::MAX);

    println!("\n=== STRICT PROMPTS (verdict-bearing) ===");
    for (pi, (prompt, expected)) in STRICT.iter().enumerate() {
        let r = generate(&model, &tok, prompt);
        let pass = r.text.starts_with(expected);
        let rail_ok = r.residual.unsigned_abs() < rail;
        if pass {
            strict_pass += 1;
        }
        if !rail_ok {
            rail_fail += 1;
        }
        println!();
        println!("prompt {pi}: {:?}", r.prompt);
        println!(
            "  ids ({}): {:?}...",
            r.prompt_ids.len(),
            &r.prompt_ids[..r.prompt_ids.len().min(16)]
        );
        println!(
            "  generated {:>2} tok{} {:?}",
            r.n_gen,
            if r.stopped_on_eos { ", eos" } else { "" },
            r.text
        );
        println!("  expected prefix: {:?}", expected);
        println!(
            "  strict: {}  residual: {} (rail {}) {}",
            if pass { "PASS" } else { "FAIL" },
            r.residual,
            RESIDUAL_SOUND_MAX,
            if rail_ok { "OK" } else { "RAILED" }
        );
        println!(
            "  speed: prefill {:.2} tok/s, decode {:.3} tok/s",
            r.prefill_tps, r.decode_tps
        );
    }

    println!("\n=== CHAT DEMONSTRATOR (structural, NOT verdict-bearing) ===");
    let tmpl = match f.value("tokenizer.chat_template") {
        Some(MetadataValue::String(s)) => s.clone(),
        _ => {
            println!("CHAT: NO (no tokenizer.chat_template in file)");
            std::process::exit(1);
        }
    };
    let chat_prompt = render_chat(&tmpl);
    let r = generate(&model, &tok, &chat_prompt);
    let chat_ok = chat_structural_pass(&r);
    let chat_rail_ok = r.residual.unsigned_abs() < rail;
    if !chat_rail_ok {
        rail_fail += 1;
    }
    println!("prompt: <embedded qwen3 template> {:?}", CHAT_USER);
    println!(
        "  rendered ({} tok): {:?}...",
        r.prompt_ids.len(),
        chat_prompt
    );
    println!(
        "  generated {:>2} tok{} {:?}",
        r.n_gen,
        if r.stopped_on_eos { ", eos" } else { "" },
        r.text
    );
    println!(
        "  structural: {}  residual: {} {}",
        if chat_ok { "PASS" } else { "FAIL" },
        r.residual,
        if chat_rail_ok { "OK" } else { "RAILED" }
    );

    println!("\n=== VERDICT ===");
    println!(
        "strict: {strict_pass}/{} prompts matched expected text prefix",
        STRICT.len()
    );
    if strict_pass == STRICT.len() && rail_fail == 0 {
        println!("STAGE 4 GATE: YES — tokenize → generate → judge, end to end, on real Q1_0 weights, integer compute path");
    } else {
        let why = if strict_pass < STRICT.len() {
            format!("{} strict prompt(s) failed", STRICT.len() - strict_pass)
        } else {
            format!("{rail_fail} residual-rail violation(s)")
        };
        println!("STAGE 4 GATE: NO — {why}");
        std::process::exit(1);
    }
}
