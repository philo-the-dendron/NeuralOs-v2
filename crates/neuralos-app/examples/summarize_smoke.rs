//! summarize_smoke — Phase 2.3 ISC-17.
//!
//! Downloads `bartowski/Qwen2.5-1.5B-Instruct-GGUF` (Q4_K_M, ~1 GB) into a local
//! cache, loads it via candle's quantized qwen2, summarizes a sample arXiv
//! abstract on THIS CPU, and prints tokens/second. Proves the local summarization
//! stack runs end-to-end + benchmarks the i5 before we wire a `QwenSummarizer`
//! behind the `Summarize` trait.
//!
//! Run with: `cargo run --example summarize_smoke --features qwen --release`
//! (release is mandatory — debug candle is glacial).

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2 as qwen;

const MODEL_REPO: &str = "bartowski/Qwen2.5-1.5B-Instruct-GGUF";
const MODEL_FILE: &str = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf";
const TOKENIZER_REPO: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const EOS_TOKEN: &str = "<|im_end|>";
const MAX_TOKENS: usize = 220;

const SAMPLE_ABSTRACT: &str = "Spiking neural networks (SNNs) encode information \
in the timing of discrete events rather than continuous rates, making them a \
potentially low-power substrate for edge inference. Training them, however, \
remains difficult because the spike-generation function is non-differentiable. \
Recent surrogate-gradient methods have narrowed the gap to rate-based baselines \
on image and keyword tasks, but generalization to long temporal sequences is \
still an open problem.";

fn build_prompt(abstract_text: &str) -> String {
    format!(
        "<|im_start|>system\nYou are a concise research-paper summarizer. \
Summarize the abstract below in 3 clear sentences.<|im_end|>\n\
<|im_start|>user\n{abstract_text}<|im_end|>\n\
<|im_start|>assistant\n"
    )
}

fn cache_dir() -> std::path::PathBuf {
    let mut p = std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    p.push(".cache");
    p.push("neuralos-app");
    p
}

/// HF Hub file URL: `https://huggingface.co/{repo}/resolve/main/{file}`.
fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

/// Download `url` to `dest` via ureq (follows redirects, incl. HF's relative ones).
fn download(url: &str, dest: &std::path::Path) -> anyhow::Result<()> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    let mut file = std::fs::File::create(dest)?;
    std::io::copy(&mut resp.into_reader(), &mut file)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cache = cache_dir();
    std::fs::create_dir_all(&cache).ok();
    println!("[smoke] cache dir: {}", cache.display());

    let gguf_path = cache.join(MODEL_FILE);
    let tok_path = cache.join("tokenizer.json");
    if !gguf_path.exists() {
        println!("[smoke] downloading {MODEL_REPO}/{MODEL_FILE} (~1 GB)…");
        download(&hf_url(MODEL_REPO, MODEL_FILE), &gguf_path)?;
    }
    if !tok_path.exists() {
        println!("[smoke] downloading {TOKENIZER_REPO}/tokenizer.json…");
        download(&hf_url(TOKENIZER_REPO, "tokenizer.json"), &tok_path)?;
    }
    println!("[smoke] gguf:       {}", gguf_path.display());
    println!("[smoke] tokenizer:  {}", tok_path.display());

    let device = Device::Cpu;
    let mut file = std::fs::File::open(&gguf_path)?;
    let t0 = std::time::Instant::now();
    let content = gguf_file::Content::read(&mut file)?;
    let mut total_size = 0_usize;
    for tensor in content.tensor_infos.values() {
        total_size +=
            tensor.shape.elem_count() * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    let mut model = qwen::ModelWeights::from_gguf(content, &mut file, &device)?;
    println!(
        "[smoke] model loaded ({:.2} GB) in {:.2}s",
        total_size as f64 / 1e9,
        t0.elapsed().as_secs_f32()
    );

    let tokenizer =
        tokenizers::Tokenizer::from_file(&tok_path).map_err(anyhow::Error::msg)?;
    let eos_id = tokenizer
        .get_vocab(true)
        .get(EOS_TOKEN)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("eos token {EOS_TOKEN} not in vocab"))?;

    let prompt = build_prompt(SAMPLE_ABSTRACT);
    let enc = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    println!("[smoke] prompt tokens: {}", prompt_ids.len());

    let mut logits_processor = LogitsProcessor::from_sampling(299_792_458, Sampling::ArgMax);

    // Prompt processing — one forward over the full prompt.
    let t_prompt = std::time::Instant::now();
    let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits)?;
    println!(
        "[smoke] prompt processed in {:.2}s",
        t_prompt.elapsed().as_secs_f32()
    );

    // Generation loop — greedy (deterministic → reproducible summaries).
    let mut generated = vec![next_token];
    let t_gen = std::time::Instant::now();
    for index in 0..MAX_TOKENS {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_ids.len() + index)?.squeeze(0)?;
        next_token = logits_processor.sample(&logits)?;
        if next_token == eos_id {
            break;
        }
        generated.push(next_token);
    }
    let dt = t_gen.elapsed();

    let summary = tokenizer
        .decode(&generated, true)
        .map_err(anyhow::Error::msg)?;

    println!("\n=== SUMMARY ===\n{}\n", summary.trim());
    println!("=== STATS ===");
    println!(
        "generated: {} tokens in {:.2}s → {:.2} tok/s",
        generated.len(),
        dt.as_secs_f64(),
        generated.len() as f64 / dt.as_secs_f64()
    );
    Ok(())
}
