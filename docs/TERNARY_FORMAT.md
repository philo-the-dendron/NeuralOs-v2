# NeuralOS Ternary Format Spec — Stage 2 of the bridge

*Status: implemented + gate-verified 2026-08-15 (`examples/ternary_format_gate.rs`,
`crates/neuralos-snn/src/bridge.rs`). Every worked example below is byte-identical
to a unit-test vector in the crate.*

This is the wire-format contract for moving ternary tensors between NeuralOS
and the two ecosystems that actually ship ternary models. All layouts were
pinned **verbatim from reference source code** (not blogs, not model cards):

| Format | Reference source (fetched 2026-08-15) | Direction |
|---|---|---|
| BitNet `i2_s` | `microsoft/BitNet` `utils/convert-hf-to-gguf-bitnet.py::quantize_to_i2_s` | export **and** import |
| Prism `q1_0` | `PrismML-Eng/llama.cpp` `ggml-common.h` `block_q1_0` + `ggml-quants.c` | **import only** |
| Prism `q2_0` | same fork, `block_q2_0` | **import only** |

## The shared code table

All 2-bit ternary encodings use the same code→value mapping (LSB-first
lanes within each byte):

| Code | Value |
|---|---|
| `00` | −1 |
| `01` | 0 |
| `10` | +1 |
| `11` | *(unreachable — see lossiness rules)* |

## Scale conventions (they differ — know which one you're holding)

| Format | Scale stored | Semantics | Container |
|---|---|---|---|
| `i2_s` | f32 bits (raw `u32`) | BitNet-Round `γ = mean|w|` — same convention as `trit::tensor_scale` | 32-byte tail per row, first 4 bytes LE |
| `q1_0` | fp16 bits (raw `u16`) | `γ = mean|w|` (BitNet-compatible) | 2 bytes LE per 128-weight block |
| `q2_0` | fp16 bits (raw `u16`) | `d = max|w|` (TWN-style) | 2 bytes LE per 64-weight block |

NeuralOS is integer-only: scales travel as raw bits (`u16`/`u32`).
`bridge::half_to_f32_bits` widens fp16→f32 bit-exactly in pure integer math;
`bridge::half_to_milli` gives the fixed-point view `round(v × 1000)`
(saturating; ±inf → `i32::MAX/MIN`, NaN → 0, all documented + tested).

## BitNet `i2_s`

**Layout.** For a row of `n` values (`n % 128 == 0` — see rules):
`n/4` packed bytes, then a 32-byte tail (first 4 bytes = LE f32 scale
bits, remaining 28 zero).

**The transposed packing.** The reference does `numpy.reshape(n, 4, 32)`
and packs lane 0 `<<6`, lane 1 `<<4`, lane 2 `<<2`, lane 3 `<<0`. In flat
index terms, element `i` lives at:

```
byte  = (i / 128) * 32 + (i % 32)
shift = 6 − 2 * ((i % 128) / 32)      # lane of the 128-block
code  = (byte >> shift) & 0b11        # {0,1,2} = {−1, 0, +1}
```

That is: the *first* 32 elements occupy the top 2 bits of bytes 0–31, the
*next* 32 the `<<4` lane of the same bytes, etc. Not sequential 4-per-byte
packing — this trips up every hand-rolled decoder.

**Worked example** (== `bridge::tests::i2_s_known_vector`). Pattern
`[+1, −1, 0, +1, 0, 0, −1, +1]` (codes `2,0,1,2,1,1,0,2`) repeated to 128
elements, scale f32 `2.0` = bits `0x40000000`. Since the pattern has period
8 and 32 % 8 == 0, all four lanes of byte `j` share a code → bytes 0–7:

```
AA 00 55 AA 55 55 00 AA     (repeated ×4 → 32 bytes)
00 00 00 40                  (tail: LE f32 2.0)
00 ×28                       (tail zero padding)
```

## Prism `q1_0`

**Layout.** Per 128-weight block, 18 bytes: LE fp16 scale, then 16 sign
bytes. Element `j`'s sign is bit `j % 8` of byte `j / 8` (LSB-first); set →
`+γ`, clear → `−γ`. The reference scale is `mean|w|` — BitNet's γ, computed
per block.

The format is **binary** `{−γ, +γ}` — no zero state. Importing into NeuralOS
is lossless (`{−1, +1} ⊂ {−1, 0, +1}`); every imported trit is `One` or
`MinusOne`.

**Worked example** (== `bridge::tests::q1_0_known_vector`). Scale fp16 `1.0`
= `0x3C00`, all 16 sign bytes `0xB5 = 0b1011_0101` → LSB-first bits
`1,0,1,0,1,1,0,1` → elements `[+1, −1, +1, −1, +1, +1, −1, +1]` ×16:

```
00 3C                         (LE fp16 1.0)
B5 B5 B5 B5 B5 B5 B5 B5
B5 B5 B5 B5 B5 B5 B5 B5
```

## Prism `q2_0`

**Layout.** Per 64-weight block, 18 bytes: LE fp16 scale (`max|w|`), then
16 bytes of LSB-first 2-bit lanes. Element `j`'s code is bits
`2·(j%4) .. 2·(j%4)+1` of byte `j/4`.

**The code-3 question.** The reference dequantizer maps `11 → +2·d`, but
the reference *quantizer* can never emit it (with `d = max|w|`,
`round(w/d) ∈ [−1, 1]` when `|w| ≤ d`). NeuralOS rejects code 3 with
`BridgeError::UnsupportedCode` — a loud error, never a silent clamp.

**Worked example** (== `bridge::tests::q2_0_known_vector`). Scale fp16
`4.0` = `0x4400`, all 16 code bytes `0xA4 = 0b10_10_01_00` → LSB-first
codes `00, 01, 10, 10` → elements `[−1, 0, +1, +1]` ×16:

```
00 44                         (LE fp16 4.0)
A4 A4 A4 A4 A4 A4 A4 A4
A4 A4 A4 A4 A4 A4 A4 A4
```

## Length rules (honesty over permissiveness)

- `i2_s` requires `n % 128 == 0` — **not** merely `% 4`. The reference
  truncates its output at `n/4` bytes; when `n % 128 != 0` the transposed
  packing places live elements in truncated bytes (silently dropped). A
  permissive codec would be silently lossy; ours refuses
  (`BadLength`).
- `q1_0` requires `n % 128 == 0`; `q2_0` requires `n % 64 == 0` (both
  asserted in the C reference).
- Real model dimensions are multiples of 256; these rules never bite in
  practice.

## Lossiness boundaries (what we refuse to pretend)

- **No `q1_0`/`q2_0` export.** Ternary → binary would silently map zeros to
  `+γ` (q1_0) or invent scale semantics (q2_0's `+2·d`). We read Prism
  formats; we do not author them.
- **Code 3 is an error**, not `+1`-with-a-wink.
- **No silent clamping anywhere** in decode paths: wrong lengths and short
  buffers return `Err`.

## Corrections to earlier docs

`docs/RESEARCH_FINDINGS.md` called the Prism ternary format "Q2_0_g128".
The fork's C source defines `QK2_0 = 64` — **group size 64** (2.25 bpw per
weight including the fp16 scale). The C code is authoritative; the "g128"
label was the model-card name, not the layout.

## The `no_std` / RISC-V posture

All codecs are buffer-based (caller-provided slices), zero-alloc,
integer-only — no float types exist in `bridge.rs` (a grep for
`f32|f64` in the module matches only comments). A RISC-V edge device can
decode Bonsai weights with this code as-is; that posture is the whole point
of the bridge (Stage 4's territory).

## What this spec does not cover

- GGUF container parsing (metadata, tensor naming, file framing) — Stage 4.
- Prism's MLX-side formats (noted, not implemented).
- NativeTernary (2.000 bpw, 2026 paper) — no shipping models; it becomes an
  import path only when something real emits it.
