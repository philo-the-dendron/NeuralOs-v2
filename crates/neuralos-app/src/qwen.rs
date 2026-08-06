//! Qwen2.5-1.5B-Instruct (int4) local summarizer — the candle-backed `Summarize` impl.
//!
//! Downloads `bartowski/Qwen2.5-1.5B-Instruct-GGUF` (Q4_K_M, ~1 GB) +
//! `tokenizer.json` on first construction into a local cache, loads via candle's
//! quantized qwen2, and summarizes through the Qwen chat template. Fully
//! on-device — no cloud, ever.

use crate::{Summarize, SummarizeError};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2 as qwen;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const MODEL_REPO: &str = "bartowski/Qwen2.5-1.5B-Instruct-GGUF";
const MODEL_FILE: &str = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf";
const TOKENIZER_REPO: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const EOS_TOKEN: &str = "<|im_end|>";
const MAX_TOKENS: usize = 220;

/// Configuration for [`QwenSummarizer`].
#[derive(Debug, Clone)]
pub struct QwenConfig {
    /// Where the gguf + tokenizer are cached.
    pub cache_dir: PathBuf,
    /// Max tokens to generate per summary.
    pub max_tokens: usize,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            max_tokens: MAX_TOKENS,
        }
    }
}

fn default_cache_dir() -> PathBuf {
    let mut p = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    p.push(".cache");
    p.push("neuralos-app");
    p
}

/// HF Hub file URL: `https://huggingface.co/{repo}/resolve/main/{file}`.
fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

/// Download `url` to `dest` via ureq (follows redirects, incl. HF's relative ones).
fn download(url: &str, dest: &Path) -> Result<(), SummarizeError> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| SummarizeError::Model(format!("GET {url}: {e}")))?;
    let mut file =
        std::fs::File::create(dest).map_err(|e| SummarizeError::Model(format!("create {}: {e}", dest.display())))?;
    std::io::copy(&mut resp.into_reader(), &mut file)
        .map_err(|e| SummarizeError::Model(format!("copy: {e}")))?;
    Ok(())
}

/// Build the Qwen2.5-Instruct chat-template prompt for summarizing `text`.
fn build_prompt(text: &str) -> String {
    format!(
        "<|im_start|>system\nYou are a concise research-paper summarizer. \
Summarize the abstract below in 3 clear sentences.<|im_end|>\n\
<|im_start|>user\n{text}<|im_end|>\n\
<|im_start|>assistant\n"
    )
}

/// Local summarizer backed by Qwen2.5-1.5B-Instruct (int4).
///
/// Holds the model behind a [`Mutex`] (candle's `forward` is `&mut self`); the
/// `Summarize` trait stays `&self` so it composes like `Fetch` and a
/// `Box<dyn Summarize>` remains practical.
pub struct QwenSummarizer {
    model: Mutex<qwen::ModelWeights>,
    tokenizer: tokenizers::Tokenizer,
    eos_id: u32,
    device: Device,
    max_tokens: usize,
}

impl std::fmt::Debug for QwenSummarizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenSummarizer")
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

impl QwenSummarizer {
    /// Download (if not cached) + load the model. Heavy: first run downloads
    /// ~1 GB, then ~5 s load. Subsequent constructions reuse the cache.
    pub fn new(cfg: QwenConfig) -> Result<Self, SummarizeError> {
        let _ = std::fs::create_dir_all(&cfg.cache_dir);
        let gguf_path = cfg.cache_dir.join(MODEL_FILE);
        let tok_path = cfg.cache_dir.join("tokenizer.json");
        if !gguf_path.exists() {
            download(&hf_url(MODEL_REPO, MODEL_FILE), &gguf_path)?;
        }
        if !tok_path.exists() {
            download(&hf_url(TOKENIZER_REPO, "tokenizer.json"), &tok_path)?;
        }

        let device = Device::Cpu;
        let mut file =
            std::fs::File::open(&gguf_path).map_err(|e| SummarizeError::Model(format!("open gguf: {e}")))?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| SummarizeError::Model(format!("read gguf: {e}")))?;
        let model = qwen::ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| SummarizeError::Model(format!("from_gguf: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| SummarizeError::Tokenizer(format!("load tokenizer: {e}")))?;
        let eos_id = tokenizer
            .get_vocab(true)
            .get(EOS_TOKEN)
            .copied()
            .ok_or_else(|| SummarizeError::Tokenizer(format!("eos {EOS_TOKEN} not in vocab")))?;

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            eos_id,
            device,
            max_tokens: cfg.max_tokens,
        })
    }
}

impl Summarize for QwenSummarizer {
    fn summarize(&self, text: &str) -> Result<String, SummarizeError> {
        let prompt = build_prompt(text);
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| SummarizeError::Tokenizer(format!("encode: {e}")))?;
        let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

        let mut model = self
            .model
            .lock()
            .map_err(|e| SummarizeError::Infer(format!("lock: {e}")))?;
        let mut lp = LogitsProcessor::from_sampling(299_792_458, Sampling::ArgMax);

        // Prompt forward (single pass).
        let input = Tensor::new(prompt_ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| SummarizeError::Infer(format!("tensor: {e}")))?;
        let logits = model
            .forward(&input, 0)
            .map_err(|e| SummarizeError::Infer(format!("forward: {e}")))?;
        let logits = logits
            .squeeze(0)
            .map_err(|e| SummarizeError::Infer(format!("squeeze: {e}")))?;
        let mut next_token = lp
            .sample(&logits)
            .map_err(|e| SummarizeError::Infer(format!("sample: {e}")))?;

        // Generation loop (greedy → deterministic, reproducible summaries).
        let mut generated = vec![next_token];
        for index in 0..self.max_tokens {
            let input = Tensor::new(&[next_token], &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| SummarizeError::Infer(format!("tensor: {e}")))?;
            let logits = model
                .forward(&input, prompt_ids.len() + index)
                .map_err(|e| SummarizeError::Infer(format!("forward: {e}")))?;
            let logits = logits
                .squeeze(0)
                .map_err(|e| SummarizeError::Infer(format!("squeeze: {e}")))?;
            next_token = lp
                .sample(&logits)
                .map_err(|e| SummarizeError::Infer(format!("sample: {e}")))?;
            if next_token == self.eos_id {
                break;
            }
            generated.push(next_token);
        }

        let summary = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| SummarizeError::Tokenizer(format!("decode: {e}")))?;
        Ok(summary.trim().to_string())
    }
}
