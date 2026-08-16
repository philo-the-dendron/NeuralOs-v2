//! GPT-2 byte-level BPE tokenizer for Qwen — Stage 4, session 4.
//!
//! Loads the tokenizer embedded in the GGUF (`tokenizer.ggml.*` metadata)
//! and encodes/decodes text against it — the runtime stays
//! dependency-free: the Qwen2 pre-tokenizer is a hand-rolled scanner
//! (below), not a regex engine.
//!
//! # Provenance (pinned from reference source, fetched 2026-08-16)
//!
//! Every behavior here mirrors `PrismML-Eng/llama.cpp` — the runtime that
//! produces and consumes these GGUF files — specifically:
//!
//! - `src/unicode.cpp::unicode_regex_split_custom_qwen2` — the executed
//!   pre-tokenizer for `pre = "qwen2"` (a hand-rolled scanner in the
//!   reference too, replacing the original regex from Qwen's
//!   `tokenizer.json`). The regex it implements:
//!
//!   ```text
//!   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//!   |[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*
//!   |\s*[\r\n]+|\s+(?!\S)|\s+
//!   ```
//!
//!   Note `\p{N}` — ONE digit per piece (the `{1,3}` grouping belongs to
//!   the llama3/GPT-4 patterns, not Qwen2).
//! - `src/vocab.cpp::llm_tokenizer_bpe_session::tokenize` — BPE by
//!   priority queue ordered (rank, left-index): lowest merge rank first,
//!   leftmost occurrence on ties; stale bigrams skipped by link checks.
//! - `src/vocab.cpp::tokenizer_st_partition` — text is partitioned on
//!   special-token strings (token types CONTROL/USER_DEFINED) before BPE.
//! - The GPT-2 byte↔unicode table per OpenAI's `encoder.py` algorithm
//!   (printable bytes map to themselves; the other 68 map to U+0100+n).
//!
//! # Deviations from the reference (documented, deliberate)
//!
//! - Character classes use Rust std: `is_alphabetic` (`\p{Alphabetic}`,
//!   slightly wider than the fork's `\p{L}` tables — includes Nl and
//!   Other_Alphabetic marks), `is_numeric` (`\p{N}`, same), `is_whitespace`
//!   (`\p{White_Space}`, same). ASCII text — everything the gate
//!   exercises — is identical under both.
//! - The fork skips unassigned codepoints in its punctuation rule (their
//!   unicode tables flag only assigned cps); Rust std cannot make that
//!   distinction, so unassigned cps group as punctuation instead of
//!   falling through to the single-char rule. Piece boundaries of rare
//!   unassigned cps may differ; byte-level round-tripping is unaffected.
//! - Contraction case-folding is ASCII (`to_ascii_lowercase`), matching
//!   the effective reach of the fork's simple-lowercase table.
//!
//! # Token types (`tokenizer.ggml.token_type`, fork `gguf.h` conventions)
//!
//! 1 normal, 2 unknown, 3 control, 4 user_defined, 5 unused, 6 byte.
//! Types 3 and 4 are "special": they partition text at encode time and
//! render literally at decode time; everything else lives in the
//! byte-encoded BPE domain.

use crate::gguf::{GgufFile, MetadataValue};
use std::collections::{BinaryHeap, HashMap};

/// `tokenizer.ggml.token_type`: normal token.
pub const TOKEN_TYPE_NORMAL: i32 = 1;
/// `tokenizer.ggml.token_type`: unknown token.
pub const TOKEN_TYPE_UNKNOWN: i32 = 2;
/// `tokenizer.ggml.token_type`: control token (Qwen specials such as
/// `<|im_start|>`).
pub const TOKEN_TYPE_CONTROL: i32 = 3;
/// `tokenizer.ggml.token_type`: user-defined added token.
pub const TOKEN_TYPE_USER_DEFINED: i32 = 4;
/// `tokenizer.ggml.token_type`: unused token.
pub const TOKEN_TYPE_UNUSED: i32 = 5;
/// `tokenizer.ggml.token_type`: byte token (SPM-style byte fallback).
pub const TOKEN_TYPE_BYTE: i32 = 6;

/// Errors from the tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// `tokenizer.ggml.model` is not `gpt2` (value shown) — only the
    /// GPT-2-style BPE family is implemented.
    UnsupportedModel(String),
    /// `tokenizer.ggml.pre` is present but not `qwen2` (value shown) —
    /// only the qwen2 pre-tokenizer is implemented. An ABSENT key
    /// defaults to qwen2 (documented default, matching the fork's
    /// per-model dispatch).
    UnsupportedPre(String),
    /// A required metadata key is missing or not the expected value type.
    MissingOrBadKv(&'static str),
    /// `tokenizer.ggml.token_type` length disagrees with the token count.
    LengthMismatch {
        /// Number of token strings.
        tokens: usize,
        /// Number of token-type entries.
        types: usize,
    },
    /// A merge line does not have the `A B` shape.
    BadMergeLine(String),
    /// A merge references a token string absent from the normal vocab.
    UnknownMergeToken(String),
    /// The vocab lacks one of the 256 single byte-encoded characters
    /// (byte-level BPE is not possible without them).
    MissingByteToken(char),
    /// A token id is outside the vocabulary, or a decode-time id cannot
    /// be represented.
    IdOutOfRange(u64),
    /// A normal token contains a character outside the GPT-2 byte table
    /// (impossible for files produced by the reference tooling).
    InvalidByteChar(char),
    /// Decoded bytes are not valid UTF-8.
    InvalidUtf8,
}

impl core::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedModel(m) => {
                write!(f, "tokenizer.ggml.model = {m:?}, want \"gpt2\"")
            }
            Self::UnsupportedPre(p) => {
                write!(f, "tokenizer.ggml.pre = {p:?}, want \"qwen2\"")
            }
            Self::MissingOrBadKv(k) => write!(f, "metadata key missing or wrong type: {k}"),
            Self::LengthMismatch { tokens, types } => {
                write!(f, "token count {tokens} != token_type count {types}")
            }
            Self::BadMergeLine(l) => write!(f, "merge line not `A B`: {l:?}"),
            Self::UnknownMergeToken(t) => write!(f, "merge references unknown token {t:?}"),
            Self::MissingByteToken(c) => {
                write!(f, "byte-table char {c:?} (U+{:04X}) missing from vocab", *c as u32)
            }
            Self::IdOutOfRange(i) => write!(f, "token id {i} outside vocabulary"),
            Self::InvalidByteChar(c) => write!(f, "char {c:?} not in the GPT-2 byte table"),
            Self::InvalidUtf8 => write!(f, "decoded bytes are not valid UTF-8"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// One merge rule: pair `(left, right)` produces token `out` at `rank`.
#[derive(Debug, Clone)]
struct MergeRule {
    rank: u32,
    out: u32,
}

/// A GPT-2 byte-level BPE tokenizer loaded from GGUF metadata.
///
/// Build with [`Tokenizer::from_gguf`]; [`Tokenizer::encode`] is
/// infallible (every byte has a token — validated at load),
/// [`Tokenizer::decode`] fails loudly on out-of-range ids or invalid
/// UTF-8.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Raw token strings by id. Normal tokens are byte-encoded domain
    /// (space = `Ġ`); specials are literal.
    vocab: Vec<String>,
    /// `true` for token types CONTROL / USER_DEFINED (partition-time
    /// specials, literal decode).
    is_special: Vec<bool>,
    /// Byte-encoded-domain lookup for BPE-eligible tokens (types NORMAL
    /// and BYTE). First id wins on duplicate strings.
    id_of: HashMap<String, u32>,
    /// Merge table keyed by the operand ids.
    merges: HashMap<(u32, u32), MergeRule>,
    /// Special (text, id) pairs, longest text first — the scan order that
    /// makes overlapping specials unambiguous.
    specials: Vec<(String, u32)>,
    /// GPT-2 byte → unicode char table.
    byte_char: [char; 256],
    /// Inverse of `byte_char`.
    char_byte: HashMap<char, u8>,
}

/// Build the GPT-2 byte↔unicode table (the `encoder.py` algorithm):
/// bytes 33–126, 161–172, 174–255 map to themselves; the remaining 68
/// map to U+0100, U+0101, … in byte order.
fn build_byte_table() -> ([char; 256], HashMap<char, u8>) {
    let mut byte_char = ['\0'; 256];
    let mut n = 0_u32;
    for b in 0..=255_u32 {
        let c = if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b)
        {
            char::from_u32(b).expect("latin-1 range chars are valid")
        } else {
            let c = char::from_u32(256 + n).expect("256..323 are valid chars");
            n += 1;
            c
        };
        byte_char[b as usize] = c;
    }
    let mut char_byte = HashMap::with_capacity(256);
    for (b, &c) in byte_char.iter().enumerate() {
        char_byte.insert(c, b as u8);
    }
    (byte_char, char_byte)
}

/// Character classification helpers mirroring the fork's `unicode_cpt_flags`
/// consumers (see the module docs for the documented deviations).
#[inline]
fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

#[inline]
fn is_number(c: char) -> bool {
    c.is_numeric()
}

#[inline]
fn is_ws_letter_num(c: Option<char>) -> bool {
    c.is_some_and(|d| d.is_whitespace() || is_letter(d) || is_number(d))
}

/// The qwen2 pre-tokenizer: split `chars` into pieces per
/// `unicode_regex_split_custom_qwen2` (see module docs for the pinned
/// pattern and the rule-by-rule correspondence). Returns `(start, end)`
/// char ranges; concatenation reproduces the input exactly.
fn split_pieces(chars: &[char]) -> Vec<(usize, usize)> {
    let n = chars.len();
    let mut pieces = Vec::new();
    let mut start = 0_usize;
    let mut i = 0_usize;
    while i < n {
        let c = chars[i];
        let piece_end;

        // Rule 1 — (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
        if c == '\'' && i + 1 < n {
            let c1 = chars[i + 1].to_ascii_lowercase();
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                piece_end = i + 2;
                pieces.push((start, piece_end));
                start = piece_end;
                i = piece_end;
                continue;
            }
            if i + 2 < n {
                let c2 = chars[i + 2].to_ascii_lowercase();
                if matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                    piece_end = i + 3;
                    pieces.push((start, piece_end));
                    start = piece_end;
                    i = piece_end;
                    continue;
                }
            }
        }

        // Rule 2 — [^\r\n\p{L}\p{N}]?\p{L}+ : an optional non-letter,
        // non-number, non-newline char (a space qualifies) glued onto a
        // letter run.
        if !(c == '\r' || c == '\n' || is_number(c))
            && (is_letter(c) || chars.get(i + 1).copied().is_some_and(is_letter))
        {
            i += 1;
            while chars.get(i).copied().is_some_and(is_letter) {
                i += 1;
            }
            pieces.push((start, i));
            start = i;
            continue;
        }

        // Rule 3 — \p{N} : ONE digit per piece (Qwen2 has no digit
        // grouping and no optional leading space on numbers).
        if is_number(c) {
            i += 1;
            pieces.push((start, i));
            start = i;
            continue;
        }

        // Rule 4 — ` ?[^\s\p{L}\p{N}]+[\r\n]*` : optional space, a run of
        // "punctuation" (defined non-space/letter/number), trailing
        // newlines. The probe char is the one AFTER an optional space.
        let probe = if c == ' ' { chars.get(i + 1).copied() } else { Some(c) };
        if !is_ws_letter_num(probe) {
            i += usize::from(c == ' ');
            while i < n && !is_ws_letter_num(Some(chars[i])) {
                i += 1;
            }
            while i < n && matches!(chars[i], '\r' | '\n') {
                i += 1;
            }
            pieces.push((start, i));
            start = i;
            continue;
        }

        // Whitespace run scan shared by rules 5–7: `ws` is the maximal
        // run length at `i`; `last_nl_end` the end of the LAST \r/\n
        // inside it.
        let mut ws = 0_usize;
        let mut last_nl_end = 0_usize;
        while i + ws < n && chars[i + ws].is_whitespace() {
            if matches!(chars[i + ws], '\r' | '\n') {
                last_nl_end = i + ws + 1;
            }
            ws += 1;
        }

        // Rule 5 — \s*[\r\n]+ : through the last newline of the run.
        if last_nl_end > 0 {
            i = last_nl_end;
            pieces.push((start, i));
            start = i;
            continue;
        }

        // Rule 6 — \s+(?!\S) : a run of >1 whitespace followed by a
        // non-space yields all but the last (the fork's
        // `num_whitespaces > 1 && next != OUT_OF_RANGE`).
        if ws > 1 && i + ws < n {
            i += ws - 1;
            pieces.push((start, i));
            start = i;
            continue;
        }

        // Rule 7 — \s+ : the whole run.
        if ws > 0 {
            i += ws;
            pieces.push((start, i));
            start = i;
            continue;
        }

        // Rule 8 — fallback: one char per piece (matches the reference's
        // catch-all for codepoints its tables do not flag).
        i += 1;
        pieces.push((start, i));
        start = i;
    }
    pieces
}

impl Tokenizer {
    /// Load from a parsed GGUF file's `tokenizer.ggml.*` metadata.
    ///
    /// Requirements (loud on violation): `model` = "gpt2"; `pre` = "qwen2"
    /// when present (absent defaults to qwen2); `tokens` and
    /// `token_type` arrays of equal length (`token_type` absent → all
    /// normal, the reference default); `merges` lines shaped `A B` with
    /// `A`, `B`, and `A+B` all present as normal tokens; and all 256
    /// single byte-encoded characters in the vocab.
    ///
    /// # Errors
    ///
    /// [`TokenizerError`] as documented per variant.
    pub fn from_gguf(f: &GgufFile<'_>) -> Result<Self, TokenizerError> {
        match f.value("tokenizer.ggml.model") {
            Some(MetadataValue::String(m)) if m == "gpt2" => {}
            Some(MetadataValue::String(m)) => return Err(TokenizerError::UnsupportedModel(m.clone())),
            _ => return Err(TokenizerError::MissingOrBadKv("tokenizer.ggml.model")),
        }
        match f.value("tokenizer.ggml.pre") {
            Some(MetadataValue::String(p)) if p == "qwen2" => {}
            Some(MetadataValue::String(p)) => return Err(TokenizerError::UnsupportedPre(p.clone())),
            None => {}
            _ => return Err(TokenizerError::MissingOrBadKv("tokenizer.ggml.pre")),
        }
        let tokens: &[MetadataValue] = match f.value("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array(items)) => items,
            _ => return Err(TokenizerError::MissingOrBadKv("tokenizer.ggml.tokens")),
        };
        let vocab: Vec<String> = tokens
            .iter()
            .map(|v| match v {
                MetadataValue::String(s) => s.clone(),
                _ => String::new(),
            })
            .collect();
        if vocab.iter().any(String::is_empty) {
            return Err(TokenizerError::MissingOrBadKv("tokenizer.ggml.tokens"));
        }
        let types: Vec<i32> = match f.value("tokenizer.ggml.token_type") {
            Some(MetadataValue::Array(items)) => items
                .iter()
                .map(|v| match v {
                    MetadataValue::I32(t) => *t,
                    _ => i32::MAX, // poison: caught by the length/type check below
                })
                .collect(),
            None => vec![TOKEN_TYPE_NORMAL; vocab.len()],
            _ => return Err(TokenizerError::MissingOrBadKv("tokenizer.ggml.token_type")),
        };
        if types.len() != vocab.len() || types.contains(&i32::MAX) {
            return Err(TokenizerError::LengthMismatch {
                tokens: vocab.len(),
                types: types.len(),
            });
        }

        let (byte_char, char_byte) = build_byte_table();

        let mut id_of: HashMap<String, u32> = HashMap::with_capacity(vocab.len());
        let mut is_special = vec![false; vocab.len()];
        let mut specials = Vec::new();
        for (id, (text, &ty)) in vocab.iter().zip(types.iter()).enumerate() {
            let id = u32::try_from(id).map_err(|_| TokenizerError::IdOutOfRange(u64::MAX))?;
            match ty {
                TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED => {
                    is_special[id as usize] = true;
                    specials.push((text.clone(), id));
                }
                TOKEN_TYPE_NORMAL | TOKEN_TYPE_BYTE => {
                    id_of.entry(text.clone()).or_insert(id);
                }
                _ => {}
            }
        }
        // Longest-first scan order (the reference partitions on each
        // special in length order; equal lengths keep id order).
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        // All 256 byte characters must be BPE-eligible tokens.
        for &c in byte_char.iter() {
            let s = c.to_string();
            if !id_of.contains_key(&s) {
                return Err(TokenizerError::MissingByteToken(c));
            }
        }

        let merges: HashMap<(u32, u32), MergeRule> = match f.value("tokenizer.ggml.merges") {
            Some(MetadataValue::Array(items)) => {
                let mut m = HashMap::with_capacity(items.len());
                for (rank, v) in items.iter().enumerate() {
                    let MetadataValue::String(line) = v else {
                        return Err(TokenizerError::BadMergeLine("<non-string>".into()));
                    };
                    let Some((a, b)) = line.split_once(' ') else {
                        return Err(TokenizerError::BadMergeLine(line.clone()));
                    };
                    if a.is_empty() || b.is_empty() || b.contains(' ') {
                        return Err(TokenizerError::BadMergeLine(line.clone()));
                    }
                    let (Some(&ia), Some(&ib)) = (id_of.get(a), id_of.get(b)) else {
                        return Err(TokenizerError::UnknownMergeToken(line.clone()));
                    };
                    let merged = format!("{a}{b}");
                    let Some(&out) = id_of.get(&merged) else {
                        return Err(TokenizerError::UnknownMergeToken(merged));
                    };
                    m.insert(
                        (ia, ib),
                        MergeRule {
                            rank: rank as u32,
                            out,
                        },
                    );
                }
                m
            }
            _ => return Err(TokenizerError::MissingOrBadKv("tokenizer.ggml.merges")),
        };

        Ok(Self {
            vocab,
            is_special,
            id_of,
            merges,
            specials,
            byte_char,
            char_byte,
        })
    }

    /// Number of tokens in the loaded vocab.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// Whether the vocab is empty (never true for a successful load).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// Look up a special token's id by its literal text (e.g.
    /// `"<|im_end|>"`). The pinned constants (`eos = 151645` etc.) are
    /// facts of the real file — tests pin them; code looks them up.
    #[must_use]
    pub fn special_id(&self, text: &str) -> Option<u32> {
        self.specials
            .iter()
            .find(|(s, _)| s == text)
            .map(|(_, id)| *id)
    }

    /// Look up any token's id by text — specials literally, normal
    /// tokens after byte-encoding the query (so `" world"` finds the
    /// `Ġworld` entry). Test-support accessor: expected-id vectors are
    /// derived from the table, never hand-pinned.
    #[must_use]
    pub fn token_id(&self, text: &str) -> Option<u32> {
        let mut encoded = String::with_capacity(text.len());
        for &b in text.as_bytes() {
            encoded.push(self.byte_char[b as usize]);
        }
        self.id_of
            .get(&encoded)
            .copied()
            .or_else(|| self.special_id(text))
    }

    /// Encode text to token ids: partition on special-token strings
    /// (longest first), then qwen2 pre-tokenize + BPE each fragment.
    /// Infallible — every byte has a token (validated at load).
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut frag_start = 0_usize;
        let mut i = 0_usize;
        while i < bytes.len() {
            if let Some((s, id)) = self
                .specials
                .iter()
                .find(|(s, _)| bytes[i..].starts_with(s.as_bytes()))
            {
                if i > frag_start {
                    self.encode_fragment(&text[frag_start..i], &mut out);
                }
                out.push(*id);
                i += s.len();
                frag_start = i;
            } else {
                i += 1;
            }
        }
        if frag_start < bytes.len() {
            self.encode_fragment(&text[frag_start..], &mut out);
        }
        out
    }

    /// Pre-tokenize + BPE one special-free fragment.
    fn encode_fragment(&self, frag: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = frag.chars().collect();
        for (start, end) in split_pieces(&chars) {
            let mut byte_chars = Vec::with_capacity(end - start);
            for &c in &chars[start..end] {
                let mut buf = [0_u8; 4];
                for &b in c.encode_utf8(&mut buf).as_bytes() {
                    byte_chars.push(self.byte_char[b as usize]);
                }
            }
            self.bpe(&byte_chars, out);
        }
    }

    /// BPE over one byte-encoded piece, appending ids (the fork's
    /// priority-queue algorithm: min (rank, left-index), stale bigrams
    /// skipped by link validation).
    fn bpe(&self, piece: &[char], out: &mut Vec<u32>) {
        let n = piece.len();
        debug_assert!(n > 0, "pieces are non-empty by construction");
        // Symbol i covers piece[start..start+len]; dead symbols keep
        // len 0. ids: the vocab id of the symbol's string.
        let mut starts = Vec::with_capacity(n);
        let mut lens: Vec<usize> = Vec::with_capacity(n);
        let mut ids: Vec<u32> = Vec::with_capacity(n);
        for (i, &c) in piece.iter().enumerate() {
            let s = c.to_string();
            let id = self.id_of[&s];
            starts.push(i);
            lens.push(1);
            ids.push(id);
        }
        let mut prev: Vec<i64> = (0..n as i64 - 1).collect();
        prev.insert(0, -1);
        let mut next: Vec<i64> = (1..n as i64).collect();
        next.push(-1);

        // Heap of (rank, left); right is carried for staleness checks.
        let mut heap: BinaryHeap<std::cmp::Reverse<(u32, u64, u64)>> = BinaryHeap::new();
        let push_bigram = |l: i64, r: i64,
                               ids: &Vec<u32>,
                               merges: &HashMap<(u32, u32), MergeRule>,
                               heap: &mut BinaryHeap<std::cmp::Reverse<(u32, u64, u64)>>| {
            if l < 0 || r < 0 {
                return;
            }
            let (l, r) = (l as usize, r as usize);
            if let Some(rule) = merges.get(&(ids[l], ids[r])) {
                heap.push(std::cmp::Reverse((
                    rule.rank,
                    l as u64,
                    r as u64,
                )));
            }
        };
        for i in 1..n {
            push_bigram(i as i64 - 1, i as i64, &ids, &self.merges, &mut heap);
        }

        while let Some(std::cmp::Reverse((_, l, r))) = heap.pop() {
            let (l, r) = (l as usize, r as usize);
            // Staleness: both alive and still adjacent (a merge on either
            // side kills the link — the fork's text comparison).
            if lens[l] == 0 || lens[r] == 0 || next[l] != r as i64 {
                continue;
            }
            let Some(rule) = self.merges.get(&(ids[l], ids[r])) else {
                continue;
            };
            let out_id = rule.out;
            // Merge r into l.
            lens[l] += lens[r];
            lens[r] = 0;
            ids[l] = out_id;
            next[l] = next[r];
            if next[r] >= 0 {
                prev[next[r] as usize] = l as i64;
            }
            push_bigram(prev[l], l as i64, &ids, &self.merges, &mut heap);
            push_bigram(l as i64, next[l], &ids, &self.merges, &mut heap);
        }

        // Walk the chain from the head (symbol 0 never dies).
        let mut idx = 0_i64;
        while idx >= 0 {
            let i = idx as usize;
            debug_assert!(lens[i] > 0, "chain must not pass dead symbols");
            out.push(ids[i]);
            idx = next[i];
        }
    }

    /// Decode token ids to text. Special tokens render literally; normal
    /// tokens go byte-table → UTF-8.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::IdOutOfRange`] for ids beyond the vocab;
    /// [`TokenizerError::InvalidByteChar`] / [`TokenizerError::InvalidUtf8`]
    /// for corrupt vocabs.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let i = usize::try_from(id).map_err(|_| TokenizerError::IdOutOfRange(u64::from(id)))?;
            let Some(text) = self.vocab.get(i) else {
                return Err(TokenizerError::IdOutOfRange(u64::from(id)));
            };
            if self.is_special[i] {
                bytes.extend_from_slice(text.as_bytes());
            } else {
                for c in text.chars() {
                    let Some(&b) = self.char_byte.get(&c) else {
                        return Err(TokenizerError::InvalidByteChar(c));
                    };
                    bytes.push(b);
                }
            }
        }
        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- byte table ----

    #[test]
    fn byte_table_pins_and_bijection() {
        let (byte_char, char_byte) = build_byte_table();
        // Printable ranges map to themselves.
        assert_eq!(byte_char[b'!' as usize], '!');
        assert_eq!(byte_char[b'~' as usize], '~');
        assert_eq!(byte_char[0xFF], 'ÿ');
        assert_eq!(byte_char[0xA1], '¡');
        // The famous ones: space → Ġ (U+0120), \n → Ċ (U+010A), NUL → Ā.
        assert_eq!(byte_char[b' ' as usize], '\u{0120}');
        assert_eq!(byte_char[b'\n' as usize], '\u{010A}');
        assert_eq!(byte_char[0], '\u{0100}');
        assert_eq!(byte_char[0x7F], '\u{0121}');
        assert_eq!(byte_char[0xAD], '\u{0143}');
        // Bijective: 256 distinct chars, inverse round-trips.
        let mut seen = std::collections::HashSet::new();
        for (b, &c) in byte_char.iter().enumerate() {
            assert!(seen.insert(c), "duplicate char for byte {b}");
            assert_eq!(char_byte.get(&c), Some(&(b as u8)));
        }
        assert_eq!(seen.len(), 256);
        // Excluded-byte census: 33 + 34 + 1 = 68 remapped chars.
        let remapped = byte_char.iter().filter(|&&c| (c as u32) >= 256).count();
        assert_eq!(remapped, 68);
    }

    // ---- scanner ----

    fn pieces_of(s: &str) -> Vec<String> {
        let chars: Vec<char> = s.chars().collect();
        split_pieces(&chars)
            .into_iter()
            .map(|(a, b)| chars[a..b].iter().collect())
            .collect()
    }

    #[test]
    fn scanner_rule_vectors() {
        // Rule 2: optional leading char (space or punct) glues to letters.
        assert_eq!(pieces_of("Hello world"), ["Hello", " world"]);
        assert_eq!(pieces_of("(hi)"), ["(hi", ")"]);
        // Rule 1: contractions (case-insensitive), no leading-space rule.
        assert_eq!(pieces_of("It's I'M don't"), ["It", "'s", " I", "'M", " don", "'t"]);
        // Rule 3: ONE digit per piece; spaces before digits stand alone.
        assert_eq!(pieces_of("1 2 3"), ["1", " ", "2", " ", "3"]);
        assert_eq!(pieces_of("1234"), ["1", "2", "3", "4"]);
        // Rule 4: optional space + punct run + trailing newlines.
        assert_eq!(pieces_of("hello, world"), ["hello", ",", " world"]);
        assert_eq!(pieces_of("hello ,world"), ["hello", " ,", "world"]);
        assert_eq!(pieces_of("3.14"), ["3", ".", "1", "4"]);
        // Rule 5: whitespace through the last newline.
        assert_eq!(pieces_of("a\nb"), ["a", "\n", "b"]);
        assert_eq!(pieces_of("a \n\n b"), ["a", " \n\n", " b"]);
        assert_eq!(pieces_of("x\r\ny"), ["x", "\r\n", "y"]);
        // Rule 6: run-minus-one before a non-space; rule 7: full run at
        // EOS or single space.
        assert_eq!(pieces_of("hello  world"), ["hello", " ", " world"]);
        assert_eq!(pieces_of("hello   world"), ["hello", "  ", " world"]);
        assert_eq!(pieces_of("hello world x"), ["hello", " world", " x"]);
        assert_eq!(pieces_of("x  "), ["x", "  "]);
        assert_eq!(pieces_of("x "), ["x", " "]);
        // Tabs are whitespace, not the rule-4 optional space.
        assert_eq!(pieces_of("a\tb"), ["a", "\tb"]); // tab is a rule-2 optional char
        assert_eq!(pieces_of("a\t\tb"), ["a", "\t", "\tb"]);
        // Punct run followed by newlines keeps them.
        assert_eq!(pieces_of("wow!!!\n\nnext"), ["wow", "!!!\n\n", "next"]);
        // Unicode letters join runs; CJK too.
        assert_eq!(pieces_of("héllo wörld"), ["héllo", " wörld"]);
        assert_eq!(pieces_of("你好吗"), ["你好吗"]);
        assert_eq!(pieces_of("  你"), [" ", " 你"]);
    }

    #[test]
    fn scanner_is_lossless() {
        // Concatenation reproduces the input exactly — the property that
        // makes byte-level round-tripping possible.
        let corpus = [
            "",
            "a",
            "  double  spaces  ",
            "\n\n\n",
            " \t \r\n mixed\r\nws ",
            "It's 1234 o'clock!!! (really)",
            "日本語と English 123 mixed",
            "emoji 🎉🎉 at end",
            "'ll 'LL 're 'RE 'q '",
            "trailing space ",
            " leading space",
        ];
        for s in corpus {
            let chars: Vec<char> = s.chars().collect();
            let pieces = split_pieces(&chars);
            let joined: String = pieces
                .iter()
                .flat_map(|&(a, b)| chars[a..b].iter().copied())
                .collect();
            assert_eq!(joined, s, "lossless split failed for {s:?}");
            assert!(pieces.iter().all(|&(a, b)| b > a), "empty piece in {s:?}");
        }
    }

    // ---- synthetic tokenizer ----

    /// A minimal BPE-able tokenizer: the 256 byte tokens (ids = byte
    /// value) plus `extra` vocab entries and `merges` given as
    /// `(left_str, right_str)` pairs (rank = index).
    fn synthetic(extra: &[&str], merges: &[(&str, &str)]) -> Tokenizer {
        let (byte_char, char_byte) = build_byte_table();
        let mut vocab = Vec::new();
        let mut id_of = HashMap::new();
        for (b, &c) in byte_char.iter().enumerate() {
            let s = c.to_string();
            id_of.insert(s.clone(), b as u32);
            vocab.push(s);
        }
        let mut next_id = vocab.len() as u32;
        for e in extra {
            id_of.insert((*e).to_string(), next_id);
            vocab.push((*e).to_string());
            next_id += 1;
        }
        let mut merge_map = HashMap::new();
        for (rank, (a, b)) in merges.iter().enumerate() {
            let ia = id_of[*a];
            let ib = id_of[*b];
            let out = id_of[&format!("{a}{b}")];
            merge_map.insert((ia, ib), MergeRule { rank: rank as u32, out });
        }
        Tokenizer {
            vocab,
            is_special: vec![false; 256 + extra.len()],
            id_of,
            merges: merge_map,
            specials: Vec::new(),
            byte_char,
            char_byte,
        }
    }

    #[test]
    fn bpe_merges_by_rank_then_position() {
        // (b,c)→bc at rank 0; (a,b)→ab at rank 1: lower rank wins.
        let t = synthetic(&["ab", "bc"], &[("b", "c"), ("a", "b")]);
        let mut out = Vec::new();
        t.bpe(&['a', 'b', 'c'], &mut out);
        let bc = t.token_id("bc").unwrap();
        let a = t.token_id("a").unwrap();
        assert_eq!(out, vec![a, bc]);

        // Same-pair multiplicities merge leftmost-first, non-overlapping.
        let t = synthetic(&["aa"], &[("a", "a")]);
        let mut out = Vec::new();
        t.bpe(&['a', 'a', 'a', 'a'], &mut out);
        let aa = t.token_id("aa").unwrap();
        assert_eq!(out, vec![aa, aa]);
        // Odd count leaves one single.
        let mut out = Vec::new();
        t.bpe(&['a', 'a', 'a'], &mut out);
        assert_eq!(out, vec![aa, t.token_id("a").unwrap()]);

        // Rank beats position: (x,y) rank 0 applies before (a,a) rank 1
        // even though (a,a) is leftmost.
        let t = synthetic(&["aa", "xy"], &[("x", "y"), ("a", "a")]);
        let mut out = Vec::new();
        t.bpe(&['a', 'a', 'x', 'y'], &mut out);
        assert_eq!(
            out,
            vec![t.token_id("aa").unwrap(), t.token_id("xy").unwrap()]
        );
    }

    #[test]
    fn encode_decode_roundtrip_synthetic() {
        // Full pipeline with merges that build "ab" and "abc" — encode
        // then decode must reproduce any input byte-for-byte.
        // Vocab/merges in the byte-encoded domain: space is Ġ (U+0120).
        // Every multi-char token needs its full merge chain (lo → llo →
        // hello → Ġhello) — ranks are list indices.
        let t = synthetic(
            &["ab", "bc", "abc", "lo", "llo", "hello", "Ġhello", "he"],
            &[
                ("h", "e"),
                ("a", "b"),
                ("ab", "c"),
                ("l", "o"),
                ("l", "lo"),
                ("he", "llo"),
                ("Ġ", "hello"),
            ],
        );
        for s in [
            "",
            "abc abc",
            "hello hello",
            "  spaces\tand\nnewlines ",
            "bytes \u{0000}\u{007F}\u{00FF} weird",
            "héllo 🎉 你好",
        ] {
            let ids = t.encode(s);
            let back = t.decode(&ids).expect("decode");
            assert_eq!(back, s, "roundtrip failed for {s:?}");
        }
        // Specific composition: "ab c" → scanner pieces ["ab", " ", "c"]
        // → byte-encoded ["ab", "Ġ", "c"] → a+b merges; Ġ and c stay
        // single (piece independence — merges never cross pieces).
        let ids = t.encode("ab c");
        let expect = vec![
            t.token_id("ab").unwrap(),
            t.token_id(" ").unwrap(),
            t.token_id("c").unwrap(),
        ];
        assert_eq!(ids, expect);
        // The merges build multi-level tokens: " hello" → Ġ+he+llo →
        // Ġ+hello → Ġhello (token_id byte-encodes the query, so the raw
        // text " hello" addresses the Ġhello entry).
        assert_eq!(t.encode(" hello"), vec![t.token_id(" hello").unwrap()]);
    }

    #[test]
    fn specials_partition_and_decode() {
        let mut t = synthetic(&["ab"], &[("a", "b")]);
        let base_len = t.vocab.len() as u32;
        // Add specials: <|end|> (longer) and <|e|> — longest-first must
        // win on the overlap.
        for (i, s) in ["<|end|>", "<|e|>"].iter().enumerate() {
            t.vocab.push((*s).to_string());
            t.is_special.push(true);
            t.specials.push(((*s).to_string(), base_len + i as u32));
        }
        t.specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let end = base_len;
        let e = base_len + 1;
        assert_eq!(t.special_id("<|end|>"), Some(end));
        assert_eq!(t.special_id("<|e|>"), Some(e));

        let ids = t.encode("ab<|end|>x<|e|>");
        assert_eq!(
            ids,
            vec![
                t.token_id("ab").unwrap(),
                end,
                t.token_id("x").unwrap(),
                e
            ]
        );
        // Specials decode literally, normals through the byte table.
        assert_eq!(t.decode(&ids).unwrap(), "ab<|end|>x<|e|>");
        // A special adjacent to letters does not leak into word pieces.
        let ids = t.encode("<|e|><|e|>");
        assert_eq!(ids, vec![e, e]);
    }

    #[test]
    fn decode_rejects_bad_input() {
        let t = synthetic(&[], &[]);
        assert_eq!(
            t.decode(&[256]),
            Err(TokenizerError::IdOutOfRange(256))
        );
        assert_eq!(
            t.decode(&[u32::MAX]),
            Err(TokenizerError::IdOutOfRange(u64::from(u32::MAX)))
        );
        // Valid ids that decode to invalid UTF-8: in the synthetic vocab
        // ids equal byte values, so 0xE9 (é's first UTF-8 byte) followed
        // by 'a' (0x61) is 0xE9 0x61 — invalid.
        let e9 = 0xE9_u32;
        let a = 0x61_u32;
        assert_eq!(t.decode(&[e9, a]), Err(TokenizerError::InvalidUtf8));
        assert_eq!(t.decode(&[e9, e9]), Err(TokenizerError::InvalidUtf8));
        assert!(t.decode(&[]).unwrap().is_empty());
    }

    // ---- real-file tests (run explicitly: `cargo test -p neuralos-rt
    // -- --ignored`; needs models/Bonsai-1.7B-Q1_0.gguf) ----

    fn real_file() -> Option<Vec<u8>> {
        ["models/Bonsai-1.7B-Q1_0.gguf", "../../models/Bonsai-1.7B-Q1_0.gguf"]
            .iter()
            .find_map(|p| std::fs::read(p).ok())
    }

    fn real_tokenizer(buf: &[u8]) -> Tokenizer {
        let f = GgufFile::parse(buf).expect("container parses");
        Tokenizer::from_gguf(&f).expect("tokenizer loads")
    }

    #[test]
    #[ignore = "needs models/Bonsai-1.7B-Q1_0.gguf (gitignored, 248 MB)"]
    fn real_vocab_and_specials_pinned() {
        let Some(buf) = real_file() else {
            eprintln!("model file absent — skipping");
            return;
        };
        let t = real_tokenizer(&buf);
        // Counts pinned from the file's own arrays.
        assert_eq!(t.len(), 151_669);
        assert_eq!(t.merges.len(), 151_387);
        // The mission's pinned ids — derived from the table here, so a
        // vocab shift is loud.
        assert_eq!(t.special_id("<|endoftext|>"), Some(151_643));
        assert_eq!(t.special_id("<|im_start|>"), Some(151_644));
        assert_eq!(t.special_id("<|im_end|>"), Some(151_645));
        // Common tokens exist as whole vocab entries (scanner pieces +
        // merges should reproduce them).
        for s in ["hello", " world", " the", " Paris", " Thursday", " five", "<think>"] {
            assert!(
                t.token_id(s).is_some(),
                "expected vocab entry {s:?} missing"
            );
        }
        // Qwen's digit-splitting design, pinned: numbers never merge with
        // the preceding space — " 8" is TWO tokens (Ġ + 8), and multi-digit
        // strings stay per-digit. The gate's expected-continuation check
        // is TEXT-prefix for exactly this reason.
        assert!(t.token_id(" 8").is_none() || t.encode(" 8").len() == 2);
        assert_eq!(
            t.encode(" 8"),
            vec![t.token_id(" ").unwrap(), t.token_id("8").unwrap()]
        );
        assert!(t.encode("14").len() == 2);
    }

    #[test]
    #[ignore = "needs models/Bonsai-1.7B-Q1_0.gguf (gitignored, 248 MB)"]
    fn real_roundtrips_and_expected_ids() {
        let Some(buf) = real_file() else {
            eprintln!("model file absent — skipping");
            return;
        };
        let t = real_tokenizer(&buf);
        // Round-trip through the embedded vocab itself.
        for s in [
            "Hello, world!",
            "The capital of France is",
            "1 2 3 4 5 6 7",
            "one two three four",
            "Monday Tuesday Wednesday",
            "10 11 12 13",
            "\ttabs and\nnewlines\r\n\r\n",
            "héllo — naïve 你好 🎉",
            "<|im_start|>user\nCount from 1 to 5.<|im_end|>\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ] {
            let ids = t.encode(s);
            assert!(!ids.is_empty(), "empty encode for {s:?}");
            let back = t.decode(&ids).expect("decode");
            assert_eq!(back, s, "roundtrip failed for {s:?}");
        }
        // Expected-id vectors DERIVED from the table (never hand-pinned):
        // the gate prompts must tokenize to their natural vocab entries.
        let id = |s: &str| t.token_id(s).expect("vocab entry");
        assert_eq!(
            t.encode("Hello, world!"),
            vec![id("Hello"), id(","), id(" world"), id("!")]
        );
        // Scanner + BPE composition on the real vocab: digits are single
        // tokens, the space piece stays its own token (no Ġ-digit merge).
        assert_eq!(t.encode("1 2 3 4 5 6 7"), {
            let mut v = Vec::new();
            for piece in ["1", " ", "2", " ", "3", " ", "4", " ", "5", " ", "6", " ", "7"] {
                v.push(id(piece));
            }
            v
        });
        // A longer word exercising multiple merges, derived from the
        // table: encode("Wednesday") == id("Wednesday") if whole, else
        // assert it round-trips (above) and is multi-token but stable.
        let wed = t.encode("Wednesday");
        if let Some(w) = t.token_id("Wednesday") {
            assert_eq!(wed, vec![w]);
        } else {
            assert!(wed.len() > 1);
        }
        // The chat prompt renders and tokenizes with specials split out.
        let chat = "<|im_start|>user\nCount from 1 to 5.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let ids = t.encode(chat);
        let im_start = t.special_id("<|im_start|>").unwrap();
        let im_end = t.special_id("<|im_end|>").unwrap();
        assert_eq!(ids[0], im_start);
        assert!(ids.contains(&im_end));
        assert_eq!(*ids.last().unwrap(), id("\n\n"));
    }
}
