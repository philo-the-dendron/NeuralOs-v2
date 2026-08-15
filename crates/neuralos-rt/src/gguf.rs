//! GGUF container parser — the tensor-file format of the llama.cpp
//! ecosystem (and therefore of Prism ML's Bonsai models).
//!
//! # Layout (pinned verbatim from PrismML-Eng/llama.cpp `gguf.h` +
//! `gguf.cpp` reader, fetched 2026-08-15)
//!
//! ```text
//! magic            "GGUF" (4 bytes)
//! version          u32            // GGUF_VERSION = 3; v1 rejected, >3 rejected
//! n_tensors        u64
//! n_kv             u64
//! kv[n_kv]         { key: string, type: u32, value }
//! tensor_info[n]   { name: string, n_dims: u32, dims: u64 × n_dims,
//!                    type: u32, offset: u64 }
//! <padding to alignment>           // general.alignment KV if present (pow2), else 32
//! tensor data      // offsets in tensor_info are relative to this section
//! ```
//!
//! `string` = `u64` length + UTF-8 bytes. Value types (fork `gguf.h`
//! `gguf_type`): u8=0, i8=1, u16=2, i16=3, u32=4, i32=5, f32=6, bool=7,
//! string=8, array=9, u64=10, i64=11, f64=12. Arrays are flat-only — a
//! nested array is a reader error in the reference, and here too.
//!
//! ggml tensor types we care about (fork `ggml/include/ggml.h`):
//! `F32=0`, `F16=1`, `Q8_0=8`, `BF16=30`, `TQ1_0=34`, **`Q1_0=41`**,
//! **`Q2_0=42`**. `Q1_0` blocks are 18 bytes per 128 weights (fp16 scale +
//! 16 sign bytes); `Q2_0` is 18 bytes per 64 — see
//! `docs/TERNARY_FORMAT.md` and [`neuralos_snn::bridge`].
//!
//! # Validation policy
//!
//! The reference reader's checks are reproduced, not invented: magic,
//! version bounds, duplicate tensor names, `n_dims ≤ 4`, non-zero pow2
//! alignment, and data offsets that stay inside the buffer. Every failure
//! is a loud [`GgufError`]; nothing is best-effort.

use core::fmt;

/// ggml tensor type: Prism fork's 1-bit (`block_q1_0`, 18 B / 128 weights).
pub const GGML_TYPE_Q1_0: u32 = 41;
/// ggml tensor type: Prism fork's ternary (`block_q2_0`, 18 B / 64 weights).
pub const GGML_TYPE_Q2_0: u32 = 42;

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_DIMS: usize = 4;

/// gguf value-type discriminants (fork `gguf.h` `gguf_type`).
mod types {
    pub const UINT8: u32 = 0;
    pub const INT8: u32 = 1;
    pub const UINT16: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT32: u32 = 4;
    pub const INT32: u32 = 5;
    pub const FLOAT32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const UINT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const FLOAT64: u32 = 12;
}

/// A metadata value — one of the 13 GGUF value types.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// One tensor's header info. `dims` holds `n_dims` entries (ggml row-major
/// `ne[]` order: dim 0 is contiguous).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name (e.g. `token_embd.weight`).
    pub name: String,
    /// Shape, outermost-first as written in the file (`ne[0]` = row width).
    pub dims: Vec<u64>,
    /// ggml type discriminant (e.g. [`GGML_TYPE_Q1_0`] = 41).
    pub ty: u32,
    /// Byte offset of this tensor's data, relative to the data section.
    pub offset: u64,
}

/// A parsed GGUF file borrowing the caller's byte buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile<'a> {
    /// Container version (2 or 3).
    pub version: u32,
    /// Metadata key-value pairs, in file order.
    pub kv: Vec<(String, MetadataValue)>,
    /// Tensor infos, in file order.
    pub tensors: Vec<TensorInfo>,
    /// Data-section alignment (bytes; pow2).
    pub alignment: u64,
    /// Byte offset of the data section within the buffer.
    pub data_start: u64,
    buf: &'a [u8],
}

/// Parser errors. Loud by design; nothing best-effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufError {
    /// Buffer shorter than the structure being read (or a length field
    /// lies about what follows).
    UnexpectedEof,
    /// First four bytes are not `GGUF`.
    BadMagic,
    /// Version 0, 1, or newer than the pinned spec.
    BadVersion(u32),
    /// A count field (n_tensors / n_kv / string len / array len) is
    /// absurd relative to the buffer.
    BadCount(u64),
    /// Unknown value-type discriminant.
    BadValueType(u32),
    /// An array whose element type is itself array (the reference reader
    /// rejects this too).
    NestedArray,
    /// Duplicate tensor names.
    DuplicateTensorName,
    /// `n_dims` outside 1..=4.
    BadDimCount(u32),
    /// Alignment zero or not a power of two.
    BadAlignment(u64),
    /// A tensor's data slice would fall outside the buffer.
    DataOutOfBounds,
    /// Key is not valid UTF-8.
    BadUtf8,
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of buffer"),
            Self::BadMagic => write!(f, "magic is not GGUF"),
            Self::BadVersion(v) => write!(f, "unsupported GGUF version {v}"),
            Self::BadCount(c) => write!(f, "absurd count {c}"),
            Self::BadValueType(t) => write!(f, "unknown value type {t}"),
            Self::NestedArray => write!(f, "array of arrays (flat arrays only)"),
            Self::DuplicateTensorName => write!(f, "duplicate tensor name"),
            Self::BadDimCount(n) => write!(f, "n_dims {n} outside 1..=4"),
            Self::BadAlignment(a) => write!(f, "alignment {a} not a nonzero pow2"),
            Self::DataOutOfBounds => write!(f, "tensor data outside buffer"),
            Self::BadUtf8 => write!(f, "string is not UTF-8"),
        }
    }
}

impl std::error::Error for GgufError {}

/// Byte cursor with bounds-checked fixed-width reads (all little-endian).
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(GgufError::BadCount(u64::MAX))?;
        if end > self.buf.len() {
            return Err(GgufError::UnexpectedEof);
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        let b = self.take(8)?;
        let mut a = [0_u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(self.u64()? as i64)
    }

    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.u8()? != 0)
    }

    /// GGUF string: u64 byte length + UTF-8 bytes. Length must fit the
    /// remaining buffer (checked via `take`), and counts beyond usize are
    /// absurd.
    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > (self.buf.len() - self.pos) as u64 {
            return Err(GgufError::BadCount(len));
        }
        let bytes = self.take(len as usize)?;
        core::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| GgufError::BadUtf8)
    }
}

impl<'a> GgufFile<'a> {
    /// Parse a GGUF buffer. See the module docs for the layout contract;
    /// validation mirrors the reference reader.
    ///
    /// # Errors
    ///
    /// Every structural violation returns its [`GgufError`] — see the
    /// enum for the full list.
    pub fn parse(buf: &'a [u8]) -> Result<Self, GgufError> {
        let mut r = Reader::new(buf);

        if r.take(4)? != MAGIC {
            return Err(GgufError::BadMagic);
        }
        let version = r.u32()?;
        // Reference: v0/v1 rejected, > GGUF_VERSION (3) rejected.
        if version == 0 || version == 1 || version > 3 {
            return Err(GgufError::BadVersion(version));
        }
        let n_tensors = count(&mut r, buf)?;
        let n_kv = count(&mut r, buf)?;

        let mut kv = Vec::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let value = read_value(&mut r, ty, buf)?;
            kv.push((key, value));
        }

        // Alignment: general.alignment KV (u32) if present, else 32. Must
        // be a nonzero pow2 (reference behavior).
        let alignment = kv
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .map(|(_, v)| match v {
                MetadataValue::U32(a) => *a as u64,
                _ => DEFAULT_ALIGNMENT,
            })
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || alignment & (alignment - 1) != 0 {
            return Err(GgufError::BadAlignment(alignment));
        }

        let mut tensors: Vec<TensorInfo> = Vec::new();
        for _ in 0..n_tensors {
            let name = r.string()?;
            if tensors.iter().any(|t| t.name == name) {
                return Err(GgufError::DuplicateTensorName);
            }
            let n_dims = r.u32()?;
            if n_dims == 0 || n_dims as usize > MAX_DIMS {
                return Err(GgufError::BadDimCount(n_dims));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ty = r.u32()?;
            let offset = r.u64()?;
            tensors.push(TensorInfo { name, dims, ty, offset });
        }

        // Data section begins at the next alignment boundary after the
        // tensor-info table.
        let overhang = (r.pos as u64) % alignment;
        let pad = if overhang == 0 { 0 } else { alignment - overhang };
        let data_start = (r.pos as u64) + pad;

        let file = Self {
            version,
            kv,
            tensors,
            alignment,
            data_start,
            buf,
        };

        // Every tensor's data slice must lie inside the buffer.
        for t in &file.tensors {
            let _ = file.tensor_data(t)?;
        }
        Ok(file)
    }

    /// The full data slice of one tensor, bounds-checked against the
    /// buffer. The slice length is whatever the file's layout implies
    /// (offset of the NEXT tensor, or buffer end for the last) — GGUF
    /// stores no per-tensor byte size.
    ///
    /// # Errors
    ///
    /// [`GgufError::DataOutOfBounds`] if the slice would leave the buffer.
    pub fn tensor_data(&self, info: &TensorInfo) -> Result<&'a [u8], GgufError> {
        let start = self
            .data_start
            .checked_add(info.offset)
            .ok_or(GgufError::DataOutOfBounds)?;
        let start = usize::try_from(start).map_err(|_| GgufError::DataOutOfBounds)?;
        if start > self.buf.len() {
            return Err(GgufError::DataOutOfBounds);
        }
        // End: the next tensor's offset (tensors are laid out in order),
        // else the buffer end.
        let next_offset = self
            .tensors
            .iter()
            .filter(|t| t.offset > info.offset)
            .map(|t| t.offset)
            .min();
        let end = match next_offset {
            Some(off) => {
                let e = self
                    .data_start
                    .checked_add(off)
                    .ok_or(GgufError::DataOutOfBounds)?;
                usize::try_from(e).map_err(|_| GgufError::DataOutOfBounds)?
            }
            None => self.buf.len(),
        };
        if end > self.buf.len() || end < start {
            return Err(GgufError::DataOutOfBounds);
        }
        Ok(&self.buf[start..end])
    }

    /// Look up a tensor by name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Look up a metadata value by key.
    #[must_use]
    pub fn value(&self, key: &str) -> Option<&MetadataValue> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

fn count(r: &mut Reader<'_>, buf: &[u8]) -> Result<u64, GgufError> {
    let c = r.u64()?;
    // A count larger than the whole buffer is absurd (each KV/tensor
    // occupies at least a few bytes) — reject before allocating.
    if c > u64::try_from(buf.len()).map_err(|_| GgufError::BadCount(c))? * 2 {
        return Err(GgufError::BadCount(c));
    }
    Ok(c)
}

fn read_value(r: &mut Reader<'_>, ty: u32, buf: &[u8]) -> Result<MetadataValue, GgufError> {
    use MetadataValue as V;
    let v = match ty {
        types::UINT8 => V::U8(r.u8()?),
        types::INT8 => V::I8(r.u8()? as i8),
        types::UINT16 => V::U16(r.u16()?),
        types::INT16 => V::I16(r.u16()? as i16),
        types::UINT32 => V::U32(r.u32()?),
        types::INT32 => V::I32(r.u32()? as i32),
        types::FLOAT32 => V::F32(r.f32()?),
        types::BOOL => V::Bool(r.bool()?),
        types::STRING => V::String(r.string()?),
        types::UINT64 => V::U64(r.u64()?),
        types::INT64 => V::I64(r.i64()?),
        types::FLOAT64 => V::F64(r.f64()?),
        types::ARRAY => {
            let elem_ty = r.u32()?;
            if elem_ty == types::ARRAY {
                return Err(GgufError::NestedArray);
            }
            let n = count(r, buf)?;
            let mut items = Vec::new();
            for _ in 0..n {
                items.push(read_value(r, elem_ty, buf)?);
            }
            V::Array(items)
        }
        other => return Err(GgufError::BadValueType(other)),
    };
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GGUF builder for tests — assembles bytes exactly per the
    /// pinned layout.
    struct W(Vec<u8>);

    impl W {
        fn new() -> Self {
            let mut b = Vec::new();
            b.extend_from_slice(MAGIC);
            b.extend_from_slice(&3_u32.to_le_bytes());
            b.extend_from_slice(&0_u64.to_le_bytes()); // n_tensors (patched)
            b.extend_from_slice(&0_u64.to_le_bytes()); // n_kv (patched)
            Self(b)
        }
        fn counts(&mut self, tensors: u64, kv: u64) {
            let bb = &mut self.0;
            bb[8..16].copy_from_slice(&tensors.to_le_bytes());
            bb[16..24].copy_from_slice(&kv.to_le_bytes());
        }
        fn u32(&mut self, v: u32) {
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn u64(&mut self, v: u64) {
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn str(&mut self, s: &str) {
            self.u64(s.len() as u64);
            self.0.extend_from_slice(s.as_bytes());
        }
        fn kv_str(&mut self, k: &str, v: &str) {
            self.str(k);
            self.u32(types::STRING);
            self.str(v);
        }
        fn kv_u32(&mut self, k: &str, v: u32) {
            self.str(k);
            self.u32(types::UINT32);
            self.u32(v);
        }
        fn kv_arr_u32(&mut self, k: &str, vals: &[u32]) {
            self.str(k);
            self.u32(types::ARRAY);
            self.u32(types::UINT32);
            self.u64(vals.len() as u64);
            for v in vals {
                self.u32(*v);
            }
        }
        fn tensor(&mut self, name: &str, dims: &[u64], ty: u32, offset: u64) {
            self.str(name);
            self.u32(dims.len() as u32);
            for d in dims {
                self.u64(*d);
            }
            self.u32(ty);
            self.u64(offset);
        }
        /// Pad to alignment and append data bytes.
        fn finish(mut self, alignment: u64, data: &[u8]) -> Vec<u8> {
            let overhang = self.0.len() as u64 % alignment;
            if overhang != 0 {
                for _ in 0..(alignment - overhang) {
                    self.0.push(0);
                }
            }
            self.0.extend_from_slice(data);
            self.0
        }
    }

    fn sample_file() -> Vec<u8> {
        let mut w = W::new();
        w.kv_str("general.architecture", "qwen3");
        w.kv_u32("general.alignment", 32);
        w.kv_arr_u32("qwen3.attention.head_count", &[16]);
        // One q1_0 tensor: 128 weights = 1 block = 18 bytes at offset 0.
        w.counts(1, 3);
        w.tensor("probe.weight", &[128, 1], GGML_TYPE_Q1_0, 0);
        let mut block = Vec::new();
        block.extend_from_slice(&0x3C00_u16.to_le_bytes()); // fp16 1.0
        block.extend(std::iter::repeat_n(0xB5_u8, 16));
        w.finish(32, &block)
    }

    #[test]
    fn parses_sample_file() {
        let bytes = sample_file();
        let f = GgufFile::parse(&bytes).expect("parse");
        assert_eq!(f.version, 3);
        assert_eq!(f.alignment, 32);
        assert_eq!(f.kv.len(), 3);
        assert_eq!(
            f.value("general.architecture"),
            Some(&MetadataValue::String("qwen3".into()))
        );
        assert_eq!(
            f.value("qwen3.attention.head_count"),
            Some(&MetadataValue::Array(vec![MetadataValue::U32(16)]))
        );
        assert_eq!(f.tensors.len(), 1);
        let t = &f.tensors[0];
        assert_eq!(t.name, "probe.weight");
        assert_eq!(t.dims, vec![128, 1]);
        assert_eq!(t.ty, GGML_TYPE_Q1_0);
        let data = f.tensor_data(t).expect("slice");
        assert_eq!(data.len(), 18);
        assert_eq!(&data[..2], &0x3C00_u16.to_le_bytes());
    }

    #[test]
    fn data_start_is_aligned() {
        let bytes = sample_file();
        let f = GgufFile::parse(&bytes).expect("parse");
        assert_eq!(f.data_start % f.alignment, 0);
        // The block starts exactly at the data section.
        assert_eq!(&bytes[f.data_start as usize..f.data_start as usize + 2],
                   &0x3C00_u16.to_le_bytes());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = sample_file();
        bytes[0] = b'X';
        assert_eq!(GgufFile::parse(&bytes), Err(GgufError::BadMagic));
    }

    #[test]
    fn rejects_bad_versions() {
        for bad in [0_u32, 1, 4] {
            let mut bytes = sample_file();
            bytes[4..8].copy_from_slice(&bad.to_le_bytes());
            assert_eq!(
                GgufFile::parse(&bytes),
                Err(GgufError::BadVersion(bad)),
                "version {bad}"
            );
        }
    }

    #[test]
    fn truncated_tail_yields_short_slice_not_error() {
        // GGUF stores no per-tensor byte size, so a truncated LAST tensor
        // cannot be detected from layout alone — the slice comes back
        // short (13 of 18 bytes) and the consumer's size check catches it
        // (bonsai_probe does exactly this for q1_0). Reference-faithful:
        // llama.cpp's gguf layer doesn't validate this either.
        let bytes = sample_file();
        let cut = &bytes[..bytes.len() - 5];
        let f = GgufFile::parse(cut).expect("layout parses");
        let data = f.tensor_data(&f.tensors[0]).expect("slice");
        assert_eq!(data.len(), 13);
        // Truncation into the HEADER, by contrast, is a hard error.
        assert_eq!(
            GgufFile::parse(&bytes[..30]),
            Err(GgufError::UnexpectedEof)
        );
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        let mut w = W::new();
        w.kv_str("general.architecture", "qwen3");
        w.counts(2, 1);
        w.tensor("dup", &[4, 1], 0, 0);
        w.tensor("dup", &[4, 1], 0, 4);
        let bytes = w.finish(32, &[0; 32]);
        assert_eq!(GgufFile::parse(&bytes), Err(GgufError::DuplicateTensorName));
    }

    #[test]
    fn rejects_nested_array() {
        let mut w = W::new();
        w.str("evil");
        w.u32(types::ARRAY);
        w.u32(types::ARRAY); // element type array — the reference rejects
        w.u64(1);
        w.counts(0, 1);
        let bytes = w.finish(32, &[]);
        assert_eq!(GgufFile::parse(&bytes), Err(GgufError::NestedArray));
    }

    #[test]
    fn rejects_bad_dim_count() {
        let mut w = W::new();
        w.counts(1, 0);
        w.str("t");
        w.u32(5); // n_dims > 4
        for _ in 0..5 {
            w.u64(2);
        }
        w.u32(0);
        w.u64(0);
        let bytes = w.finish(32, &[0; 64]);
        assert_eq!(GgufFile::parse(&bytes), Err(GgufError::BadDimCount(5)));
    }

    #[test]
    fn rejects_non_pow2_alignment() {
        let mut w = W::new();
        w.kv_u32("general.alignment", 24); // not pow2
        w.counts(0, 1);
        let bytes = w.finish(32, &[]);
        assert_eq!(GgufFile::parse(&bytes), Err(GgufError::BadAlignment(24)));
    }

    #[test]
    fn rejects_absurd_counts() {
        let mut w = W::new();
        w.counts(0, u64::MAX / 2);
        w.kv_str("a", "b"); // one real kv so the loop can start
        let bytes = w.finish(32, &[]);
        assert_eq!(GgufFile::parse(&bytes).err(), Some(GgufError::BadCount(u64::MAX / 2)));
    }

    #[test]
    fn default_alignment_is_32_without_kv() {
        let mut w = W::new();
        w.counts(0, 0);
        let bytes = w.finish(16, &[0xAB]);
        let f = GgufFile::parse(&bytes).expect("parse");
        assert_eq!(f.alignment, 32);
    }
}
