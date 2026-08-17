//! NeuralOS ternary-LLM runtime — Stage 4 of the ternary bridge.
//!
//! The goal (see `docs/VISION.md`): load and run Bonsai-style `Q1_0`
//! ternary models in pure Rust — the Rust answer to `bitnet.cpp`,
//! sovereignty-grade local AI sharing the SNN's arithmetic natively
//! ([`neuralos_snn::bridge`] owns the block codecs; this crate owns the
//! container and, in later sessions, the model execution).
//!
//! # Session 1 scope (this crate today)
//!
//! - [`gguf`]: a buffer-based GGUF container parser — header, metadata
//!   key-values (all 13 value types), tensor infos, validated data-slice
//!   access. Layout pinned verbatim from the Prism fork's `gguf.h` +
//!   `gguf.cpp` reader (GGUF v3; see the module docs).
//! - `examples/bonsai_probe.rs`: parse a real Bonsai GGUF, verify tensor
//!   geometry, and decode a real `q1_0` block through the Stage-2 codec.
//!
//! # Posture
//!
//! `std` crate (file IO lives at the edges — examples and, later, the
//! loader); the parse core operates on caller-provided byte slices with no
//! I/O of its own, so a `no_std` edge story remains open. No `unsafe`.

pub mod gguf;
pub mod math;
pub mod model;
pub mod norm;
pub mod q1_0;
pub mod q2_0;
pub mod token;

pub use gguf::{GgufError, GgufFile, MetadataValue, TensorInfo, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0};
pub use math::{div_round_half_away, MathKit, RopeTables};
pub use model::{ForwardHealth, ModelConfig, ModelError, Qwen3, Session, RESIDUAL_SOUND_MAX};
pub use norm::{f32_bits_to_milli, isqrt, rms_norm_milli};
pub use q1_0::{matvec_scaled, q1_0_matvec, q1_0_row_to_milli, Q1_0_BLOCK, Q1_0_BLOCK_BYTES, Q10Error};
pub use q2_0::{matvec_scaled as q2_0_matvec_scaled, q2_0_matvec, q2_0_row_to_milli, Q2_0_BLOCK, Q2_0_BLOCK_BYTES, Q20Error};
pub use token::{Tokenizer, TokenizerError};
