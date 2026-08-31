//! SIMD-accelerated batch LIF integration (AVX2, `x86_64`).
//!
//! Ported from v0.1 `libneuralos_before_bridge_removal/src/core/simd_vectorization.rs`,
//! with two correctness bugs fixed (see `BUG_FIXES` below).
//!
//! The MODULE is gated on the `simd` feature only (`simd = ["std"]` in
//! `Cargo.toml`, `#[cfg(feature = "simd")] pub mod simd;` in `lib.rs`) — not on
//! the architecture. `cfg(target_arch = "x86_64")` gates the intrinsics import,
//! `integrate_batch_avx2` and `lif_lane` inside it, so on any other target the
//! module still compiles and `integrate_lif_batch` runs the scalar reference.
//! Checked, not assumed: `cargo check -p neuralos-snn --features simd --target
//! riscv64gc-unknown-linux-musl` is green.
//!
//! # What it does
//!
//! Vectorises the leaky-integrate step of [`crate::LIFNeuron::integrate_and_fire`]
//! across a batch of neurons held in `structure-of-arrays` (SoA) form: separate
//! slices for membrane / resting / current / resistance / threshold. One call
//! integrates N neurons; AVX2 processes 16 i16 per iteration, with a scalar
//! remainder tail.
//!
//! # The SoA seam
//!
//! The network stores neurons as `Vec<LIFNeuron>` (`array-of-structures`). SIMD
//! needs SoA. The batch entry point takes slices; an adapter that gathers
//! `&mut [LIFNeuron]` → SoA slices, runs the batch, scatters back, is a
//! follow-up concern (measured separately from the kernel speedup).
//!
//! One type seam that adapter will have to close: `resistance` is `u16`
//! (`resistance_mohm`) on [`crate::LIFNeuron`] and `i16` here, so the batch
//! accepts negative resistances no neuron can hold, and rejects the top half of
//! the neuron's range. Only `0..=i16::MAX` is representable in both.
//!
//! # Approximation vs scalar
//!
//! The AVX2 kernel replaces `/1000` with `÷1024` — the standard fixed-point
//! fast-division approximation, ~2.4% error (`1024/1000`). What makes that
//! biologically irrelevant is the grid, not the noise floor: on the default mV
//! grid a steady current below ~200 μA at rest moves the membrane by exactly
//! zero forever ([`crate::lif_neuron`] § the dead zone), so a ≤2 mV
//! approximation sits inside the grid's own blindness. The older phrasing here
//! compared a millivolt error against the ±5 μA default noise amplitude, which
//! are different units and not comparable. The scalar reference here uses exact
//! `/1000`. The correctness test asserts the two agree within ±2 mV per neuron,
//! not bit-exact.
//!
//! ## Corrected 2026-08-30: the two divisions must round the same way
//!
//! Both `÷1024` sites used to be a bare `_mm256_srai_epi32(_, 10)`, which
//! rounds toward −∞, while the scalar `/` rounds toward zero. Every negative
//! quotient therefore came out one unit more negative than the reference. In a
//! single step that hides inside the ±2 mV tolerance, so every test in this
//! module passed. Across a stepped simulation it does not: the two halves drift
//! apart until each is caught by the dead zone, and then they PARK IN DIFFERENT
//! PLACES. Measured before the fix, N = 16, `resting = −70`, `resistance = 100`,
//! `dt_over_tau = 50`, threshold unreachable, 10_000 steps: at 0 μA drive the
//! halves agreed, at +200 μA they were 1 mV apart, and at −200 μA the scalar
//! parked at −71 mV against the AVX2 half at −90 mV. A 19 mV gap, wider than
//! the 15 mV between rest and threshold — enough to decide whether a neuron
//! ever fires.
//!
//! Fixed by [`div1024_toward_zero`], which biases negatives before the shift so
//! both sites truncate toward zero like the scalar. The scale stays ÷1024; only
//! the rounding direction changed.
//!
//! What that does NOT buy is agreement at the fixed point, and the reason is
//! the scale, not the rounding. The scalar parks where
//! `|dt·(leak + P/1000)| < 1000` and the vector where
//! `|dt·(leak + P/1024)| < 1024` — two different dead-zone intervals, offset by
//! `|P/1000 − P/1024|`, which is ~2.4% of the current term and reaches 3 at the
//! edge of the equivalence domain. Worked example, 1000 μA into 100 MΩ: the
//! scalar's current term is 100 against the vector's 97, so the scalar parks
//! anywhere in `mp ∈ (10, 50)` and lands on 11, the vector in `(6.5, 47.5)` and
//! lands on 7. After the fix the worst gap is **8 mV**, both at the fixed point
//! and at any step along the way, and it does not grow with step count —
//! identical at 1_000 steps and at 1_000_000. Pinned by
//! `avx2_and_scalar_trajectories_stay_bounded`.
//!
//! Those two maxima are equal, and that is not a coincidence to gloss over: the
//! worst cases are arms where the vector half never moves at all, so its
//! largest deviation is its final one. Witness, `resting = 37`,
//! `input × resistance = 100_000`, `dt_over_tau = 5`, from −70 mV: the scalar's
//! `current_term` is 100 and it climbs to −62, while the vector's is 97 and
//! `5 × (134 + 70) = 1020 < 1024` truncates to zero every step, so it sits at
//! −70 forever. 8 mV apart on step one and on step 20_000. The extremal case
//! found by the sweep is the same shape: `resting = 50`, `current_term` 87
//! against 84, `dt_over_tau = 5`.
//!
//! Established by exhaustive sweep, not sampling: `resting` over the whole mV
//! grid × all 395 distinct `(current_term_scalar, current_term_avx2)` classes
//! for `|input × resistance| ≤ 100_000` × `dt_over_tau` in `0..=200`, each run
//! to its fixed point from −70 mV.
//!
//! (This section said 4 mV and 5 mV until 2026-08-31. Those were the maxima
//! over the nine arms the test happened to carry, not over the domain, and the
//! 4-versus-5 split was an artifact of that arm set. The arms that reach 8 are
//! now in the test.)
//!
//! **Two different bounds, do not conflate them.** The ±2 mV in
//! § Equivalence domain is a SINGLE step from the SAME membrane. The 4 and 5
//! above are trajectory differences between two states that have already
//! diverged. The single-step contract is unchanged by this fix.
//!
//! ### Recorded fork: truncate only the delta shift
//!
//! Not taken. Half the added instructions: leave `current_term` on the plain
//! floor shift and truncate only the delta. Measured by the reviewer on the real
//! kernel — B1 is fixed identically (named arms `[0, 1, 1]`), the equivalence
//! maximum is unchanged at 2, and the corner triple comes out `[2, 270, 18]`
//! against the committed `[4, 180, 0]`. **Trigger for revisiting: if the ~15%
//! vector-path cost ever matters to a consumer, this recovers about half of it,
//! at the price of 18 corner spike disagreements where the committed choice has
//! none.** The committed `(truncate, truncate)` stands because zero spike
//! disagreement at the corners is worth more here than the instructions.
//!
//! ### Recorded fork: exact ÷1000 in the vector half
//!
//! Not taken. `_mm256_mullo_epi32` by a reciprocal plus a shift would make the
//! two halves bit-equal and remove the parking offset entirely, at the cost of
//! two extra multiplies per lane per step. **Trigger for revisiting: when a
//! consumer needs bit-equal batch and scalar results across targets.** Until
//! then the ÷1024 scale stays and the offset is documented rather than removed.
//!
//! # What "matching `integrate_and_fire`" means
//!
//! [`integrate_batch_scalar`] is bit-equal to
//! [`crate::LIFNeuron::integrate_and_fire`] on the mV grid — same
//! `dt_over_tau`, same `leak + (I·R)/1000`, same `/1000`, same
//! `saturating_add` and same clamp — and produces the same spike bit. Pinned by
//! `prop_scalar_batch_is_bit_equal_to_integrate_and_fire`, which checks it for
//! any `i16` input with `dt_us ≤ tau_membrane_us`. Verified against
//! `lif_neuron.rs`, not inherited from the port.
//!
//! Four differences are NOT arithmetic, and a caller carries each one itself:
//!
//! - **Current accumulation.** `integrate_and_fire` adds synaptic current and
//!   LFSR noise and subtracts the adaptation current. The batch takes the total
//!   already summed.
//! - **The post-spike reset.** `integrate_and_fire` overwrites the membrane with
//!   `reset_potential` when it fires. The batch's membrane is the pre-reset
//!   value, and it writes no history and starts no refractory period.
//! - **The refractory period.** `integrate_and_fire` skips integration entirely
//!   while `refractory_time_us > 0`. The batch has no such state.
//! - **The leak subtraction width.** `integrate_and_fire` computes
//!   `resting_potential − membrane_potential` in `i16` and then widens; the
//!   batch widens both to `i32` first. On the mV grid the difference is
//!   unreachable, but off-grid `i16` values that overflow the subtraction would
//!   panic in one and not the other.
//!
//! # Overflow domain (the `dt_over_tau` bound)
//!
//! Every intermediate in `lif_lane` and [`integrate_batch_scalar`] is `i32`, and
//! the two halves disagree on what overflow means: `_mm256_mullo_epi32` wraps
//! silently, while the scalar `*` panics in debug and wraps in release. So the
//! kernel does not permit overflow at all — it saturates `dt_over_tau` into the
//! range where no intermediate can overflow, for **any** `i16` input.
//!
//! Derivation, worst case over the full `i16` domain:
//!
//! - `leak = resting − membrane`, widened to `i32`: `|leak| ≤ 65_535`.
//! - `input × resistance`: `|P| ≤ 32_768 × 32_768 = 1_073_741_824`, always inside
//!   `i32`. This product is safe unconditionally.
//! - `current_term`: `|P| / 1000 ≤ 1_073_741` (scalar), `|P| ÷ 1024 ≤ 1_048_576`
//!   (AVX2). The scalar is the larger, so it binds. The toward-zero bias in
//!   [`div1024_toward_zero`] only ever moves a negative value closer to zero, so
//!   it cannot widen any of these magnitudes and the bound is unaffected.
//! - `sum = leak + current_term`: `|sum| ≤ 65_535 + 1_073_741 = 1_139_276`.
//! - `sum × dt_over_tau` must fit `i32`: `|dt_over_tau| ≤ i32::MAX / 1_139_276 = 1884`.
//!
//! Hence [`DT_OVER_TAU_MAX`] `= 1884`. Both public entry points saturate to
//! `−DT_OVER_TAU_MAX..=DT_OVER_TAU_MAX`, and [`dt_over_tau`] never returns
//! anything outside it.
//!
//! **Saturate, not reject.** Rejection needs an error channel, and neither
//! [`integrate_lif_batch`] nor [`integrate_batch_scalar`] has one — adding
//! `Result` would change a hot-path signature for a condition no physical `dt/τ`
//! reaches (the standard 1 ms / 20 ms step gives `dt_over_tau = 50`). A silent
//! wrap in one half and a debug panic in the other is the worse outcome, so the
//! bound is enforced rather than reported.
//!
//! Inside the bound nothing overflows, but the two halves are *not* bit-equal at
//! the extremes — that is what the equivalence domain below is for. Measured
//! over all 3125 combinations of `{i16::MIN, -1, 0, 1, i16::MAX}` at
//! `dt_over_tau = DT_OVER_TAU_MAX`: both halves stay on the mV grid, 180 of 3125
//! membranes differ, the largest difference is 4 mV, and no spike bit differs.
//! Pinned by `overflow_corners_at_max_dt_over_tau`. (Was 375 / 3 mV / 44 spike
//! bits before the rounding fix in § Approximation. Matching the scalar's
//! rounding more than halves how often the corners disagree and removes spike
//! disagreement entirely, while adding 1 mV to the single worst case — the
//! corners sit far outside the equivalence domain, so no bound is claimed here,
//! only the exact measurement.)
//!
//! # Equivalence domain (where ±2 mV holds)
//!
//! The ±2 mV agreement below is a claim about a **narrower** domain than the
//! overflow bound, and the two must not be confused:
//!
//! - `membrane`, `resting` on the mV grid, `−100..=50`;
//! - `|input_current × resistance| ≤ 100_000`;
//! - `0 ≤ dt_over_tau ≤ 200`.
//!
//! Inside it, `|membrane_avx2 − membrane_scalar| ≤ 2` and the two disagree on a
//! spike only where the scalar membrane sits within 2 mV of that neuron's
//! threshold. The bound is exhaustive, not sampled: over that domain the
//! difference depends only on `(membrane, resting, current_term_scalar,
//! current_term_avx2)`, and enumerating every reachable combination gives a
//! maximum of exactly 2. Re-enumerated after the rounding fix in
//! § Approximation, not carried over: the maximum over the domain is still
//! exactly 2 and the edge is still the same value, but the interior improved —
//! at the default `dt_over_tau = 50` the worst case fell from 2 to 1, and at
//! `dt_over_tau = 1` it is now 0. The first `dt_over_tau` that reaches 3 at the same
//! current bound is **228** (`membrane = −100`, `resting = 50`, scalar
//! `current_term = 100` against the AVX2 `97`, giving −43 against −46); every
//! value through 227 still gives 2. The stated bound of 200 is therefore
//! conservative by 27, which is deliberate — it is a round number well inside
//! the edge rather than sitting on it. (An earlier draft of this section said
//! the first failure was at 256, which was read off a coarse power-of-two
//! sample and never the true edge.)
//!
//! At the default 1 ms / 20 ms step (`dt_over_tau = 50`) and the default
//! `resistance = 100` MΩ, the current bound admits `|input| ≤ 1000` μA, two
//! orders of magnitude above the ±5 μA default noise.
//!
//! # Grid limitation — mV only
//!
//! The batch kernel (and its scalar reference) operate on the **default mV
//! grid only**: the reference clamps to `−100..50` and ignores
//! [`crate::VoltageResolution`] scale (`integrate_batch_scalar`, below). A
//! centi-mV consumer would get silently wrong membranes. Until the kernel
//! takes a scale parameter, do not route `CentiMillivolt` state through it (no
//! in-tree consumer does — 2026-08-20 audit, re-verified 2026-08-30: the only
//! callers of `integrate_lif_batch` / `integrate_batch_scalar` anywhere in the
//! workspace are `examples/bench_simd.rs` and this module's own tests, and none
//! constructs a `CentiMillivolt` neuron).
//!
//! # `BUG_FIXES` vs v0.1
//!
//! - **Widen-both-halves.** v0.1's `integrate_neurons_avx2` (`simd_vectorization.rs:237-240`)
//!   called `_mm256_cvtepi16_epi32(_mm256_castsi256_si128(mp))`, which widens
//!   only the LOW 8 of each 16-element load, so the high 8 neurons of every
//!   chunk were never computed. What got stored in their place was a
//!   deterministic DUPLICATE of the low 8, not stale memory: the pack step is
//!   `_mm256_packs_epi32(clamped_mp, clamped_mp)` (`simd_vectorization.rs:263-267`),
//!   which packs the same eight `i32` lanes twice, and all 16 `i16` lanes are
//!   then stored. (This bullet said "stale memory was stored back" until
//!   2026-08-30; that was wrong about the mechanism, and wrong in the direction
//!   that makes the bug sound less reproducible than it is. Corrected against
//!   the archive, not recalled.) Fixed: widen both halves via
//!   `_mm256_extracti128_si256(_, 1)`, process both, repack.
//! - **Spike semantics + mask.** v0.1 used `_mm256_cmpgt_epi16` (strict `>`)
//!   and indexed byte-bits (`1 << j`) as if they were i16 lanes (two bytes per
//!   i16). Fixed: `>=` via `cmpgt | cmpeq` (matches the scalar `>=` contract),
//!   and `(mask >> (j*2)) & 1` to read the correct byte of each i16 lane.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]
// SIMD intrinsics are inherently unsafe; this module is OS-dev territory
// (workspace has `unsafe_code = "allow"`).
#![allow(clippy::missing_safety_doc)]
// Canonical SIMD idiom: `use std::arch::x86_64::*` brings in hundreds of
// intrinsics by design; listing them explicitly is noise.
#![allow(clippy::wildcard_imports)]
// Unaligned intrinsic loads (`_mm256_loadu_si256`) cast `*const i16` →
// `*const __m256i` deliberately — the `u` in `loadu` is the unaligned path, so
// the stricter alignment of the target type is semantically irrelevant.
#![allow(clippy::cast_ptr_alignment)]
// `ptr as ptr` in intrinsic arg lists is the documented call convention.
#![allow(clippy::ptr_as_ptr)]
// This module is dense with hardware acronyms (AVX2, SoA, x86_64, SSE) that
// read fine in prose; the doc-markdown lint's per-acronym backtick nag is noise here.
#![allow(clippy::doc_markdown)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD instruction set detected at runtime (x86_64 only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdSupport {
    /// No usable SIMD — the scalar fallback runs.
    None,
    /// AVX2 (256-bit, 16 × i16 per iteration). The fast path.
    Avx2,
}

/// Detect the best available SIMD instruction set at runtime.
///
/// Returns [`SimdSupport::None`] on non-x86_64 targets (compile-time) or when
/// AVX2 is absent at runtime. The dispatch in [`integrate_lif_batch`] consults this.
///
/// # Examples
/// ```
/// # use neuralos_snn::simd::detect_simd_support;
/// let s = detect_simd_support();
/// // On an AVX2 box: SimdSupport::Avx2. On older x86 / non-x86: None.
/// println!("simd: {s:?}");
/// ```
#[must_use]
pub fn detect_simd_support() -> SimdSupport {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            SimdSupport::Avx2
        } else {
            SimdSupport::None
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        SimdSupport::None
    }
}

/// Largest `|dt_over_tau|` for which no `i32` intermediate in the kernel can
/// overflow, for any `i16` input. See the module doc § Overflow domain for the
/// derivation; `i32::MAX / 1_139_276 = 1884`.
pub const DT_OVER_TAU_MAX: i32 = 1884;

/// Precompute the `dt/τ` scaling factor (passed into [`integrate_lif_batch`]).
///
/// Matches [`crate::LIFNeuron::integrate_and_fire`]: `dt_over_tau = (dt_us * 1000) / tau_us`,
/// then saturated to [`DT_OVER_TAU_MAX`]. Hoisted out so a batch sharing one `dt`
/// and `tau` computes it once.
///
/// The product and the division are computed in `u64`. The previous `as i32`
/// casts wrapped for `dt_us > i32::MAX` and for `tau_membrane_us > i32::MAX` —
/// `dt_over_tau(2_147_484, u32::MAX)` returned `-2_147_483_647`, a value that
/// overflows every downstream multiply. Both inputs are `u32`, so `u64` covers
/// the whole domain exactly and the result is always non-negative.
#[must_use]
pub fn dt_over_tau(dt_us: u32, tau_membrane_us: u32) -> i32 {
    if tau_membrane_us == 0 {
        return 0; // Guard; the network rejects tau == 0 at construction.
    }
    let raw = (u64::from(dt_us) * 1000) / u64::from(tau_membrane_us);
    i32::try_from(raw).unwrap_or(i32::MAX).min(DT_OVER_TAU_MAX)
}

/// Integrate one LIF step across a batch of N neurons (SoA slices).
///
/// Updates `membrane` in place and writes the spike mask to `spikes_out`.
/// Picks AVX2 at runtime when available, else the scalar reference. All slices
/// must be equal length (asserted, in every profile).
///
/// `input_currents` is the *total* effective current per neuron (external +
/// synaptic + noise − adaptation); the batch computes only the membrane
/// update, not current accumulation — that's the caller's job.
///
/// # Panics
///
/// Panics if the six slices are not all the same length — in **every** profile,
/// not just debug. The AVX2 kernel indexes all of them by the same chunk
/// offsets, so an unequal length is an out-of-bounds read from a safe function
/// in a published crate; enforcing the contract here is the only way the
/// `SAFETY` comment below can name an invariant that actually holds.
///
/// Asserting rather than clamping `n` to the shortest slice is deliberate. The
/// documented contract has always been "all slices equal length"; clamping
/// would silently redefine it, integrate a prefix, and hide the caller's bug in
/// a numerical kernel where a short slice is never intentional. Five length
/// comparisons per call cost nothing against N-element work.
pub fn integrate_lif_batch(
    membrane: &mut [i16],
    resting: &[i16],
    input_currents: &[i16],
    resistance: &[i16],
    threshold: &[i16],
    dt_over_tau: i32,
    spikes_out: &mut [bool],
) {
    let n = membrane.len();
    assert_eq!(resting.len(), n, "resting.len() != membrane.len()");
    assert_eq!(input_currents.len(), n, "input_currents.len() != membrane.len()");
    assert_eq!(resistance.len(), n, "resistance.len() != membrane.len()");
    assert_eq!(threshold.len(), n, "threshold.len() != membrane.len()");
    assert_eq!(spikes_out.len(), n, "spikes_out.len() != membrane.len()");

    // Saturate into the no-overflow domain (module doc § Overflow domain). Both
    // halves must see the same value, or AVX2 wraps where the scalar panics.
    let dt_over_tau = dt_over_tau.clamp(-DT_OVER_TAU_MAX, DT_OVER_TAU_MAX);

    #[cfg(target_arch = "x86_64")]
    if matches!(detect_simd_support(), SimdSupport::Avx2) {
        // SAFETY: slices are valid and equal-length — asserted above in every
        // profile, not merely debug-asserted — and the AVX2
        // kernel processes 16-element aligned chunks plus a scalar tail, so no
        // out-of-bounds access occurs. `membrane` is &mut and uniquely borrowed
        // here; the kernel writes within bounds.
        unsafe { integrate_batch_avx2(membrane, resting, input_currents, resistance, threshold, dt_over_tau, spikes_out) };
        return;
    }
    integrate_batch_scalar(membrane, resting, input_currents, resistance, threshold, dt_over_tau, spikes_out);
}

/// Scalar reference — exact v2 LIF math (÷1000). Also the remainder tail.
///
/// `dt_over_tau` is saturated to [`DT_OVER_TAU_MAX`] on entry, so no `i32`
/// intermediate here can overflow for any `i16` input (module doc § Overflow
/// domain). Inside that bound the membrane arithmetic is bit-equal to
/// [`crate::LIFNeuron::integrate_and_fire`] on the mV grid — pinned by
/// `prop_scalar_batch_is_bit_equal_to_integrate_and_fire`, with the four
/// non-arithmetic differences listed in the module doc.
pub fn integrate_batch_scalar(
    membrane: &mut [i16],
    resting: &[i16],
    input_currents: &[i16],
    resistance: &[i16],
    threshold: &[i16],
    dt_over_tau: i32,
    spikes_out: &mut [bool],
) {
    let dt_over_tau = dt_over_tau.clamp(-DT_OVER_TAU_MAX, DT_OVER_TAU_MAX);
    for i in 0..membrane.len() {
        let mp = i32::from(membrane[i]);
        let leak = i32::from(resting[i]) - mp;
        let current_term = (i32::from(input_currents[i]) * i32::from(resistance[i])) / 1000;
        let delta_v = (dt_over_tau * (leak + current_term)) / 1000;
        let new_v = mp.saturating_add(delta_v).clamp(-100, 50);
        membrane[i] = new_v as i16;
        spikes_out[i] = new_v >= i32::from(threshold[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn integrate_batch_avx2(
    membrane: &mut [i16],
    resting: &[i16],
    input_currents: &[i16],
    resistance: &[i16],
    threshold: &[i16],
    dt_over_tau: i32,
    spikes_out: &mut [bool],
) {
    const WIDTH: usize = 16; // AVX2: 256-bit / 16-bit = 16 lanes.
    // Same saturation as the scalar entry — `_mm256_mullo_epi32` wraps silently.
    let dt_over_tau = dt_over_tau.clamp(-DT_OVER_TAU_MAX, DT_OVER_TAU_MAX);
    let n = membrane.len();
    let chunks = n / WIDTH;

    let dt_v = _mm256_set1_epi32(dt_over_tau);

    for c in 0..chunks {
        let off = c * WIDTH;

        // Load 16 i16 each (unaligned — safe for any slice alignment).
        let mp = _mm256_loadu_si256(membrane.as_ptr().add(off) as *const __m256i);
        let rp = _mm256_loadu_si256(resting.as_ptr().add(off) as *const __m256i);
        let ic = _mm256_loadu_si256(input_currents.as_ptr().add(off) as *const __m256i);
        let res = _mm256_loadu_si256(resistance.as_ptr().add(off) as *const __m256i);
        let th = _mm256_loadu_si256(threshold.as_ptr().add(off) as *const __m256i);

        // BUG FIX vs v0.1: widen BOTH halves (low 8 + high 8), not just the low.
        let mp_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mp));
        let mp_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mp, 1));
        let rp_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(rp));
        let rp_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(rp, 1));
        let ic_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(ic));
        let ic_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ic, 1));
        let res_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(res));
        let res_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res, 1));

        // LIF math on each half. ÷1000 approximated as ÷1024, rounded toward zero.
        let new_lo = lif_lane(mp_lo, rp_lo, ic_lo, res_lo, dt_v);
        let new_hi = lif_lane(mp_hi, rp_hi, ic_hi, res_hi, dt_v);

        // Pack 16 × i32 → 16 × i16 (signed-saturate; values pre-clamped to [-100,50],
        // so no saturation actually triggers). packs is per-128-bit-lane, so permute
        // the 64-bit lanes to restore linear order [lo0..7, hi0..7].
        let packed = _mm256_packs_epi32(new_lo, new_hi);
        let final_mp = _mm256_permute4x64_epi64(packed, 0b1101_1000);

        _mm256_storeu_si256(membrane.as_mut_ptr().add(off) as *mut __m256i, final_mp);

        // Spike detection: final_mp >= threshold  →  (>) OR (==).
        // BUG FIX vs v0.1: was strict `>` (mismatched scalar `>=` contract).
        let gt = _mm256_cmpgt_epi16(final_mp, th);
        let eq = _mm256_cmpeq_epi16(final_mp, th);
        let ge = _mm256_or_si256(gt, eq);
        // movemask_epi8: one bit per byte → 2 bits per i16 lane. Read low byte of each pair.
        let mask = _mm256_movemask_epi8(ge) as u32;
        for j in 0..WIDTH {
            spikes_out[off + j] = (mask >> (j * 2)) & 1 != 0;
        }
    }

    // Scalar tail for the remainder.
    let tail = chunks * WIDTH;
    integrate_batch_scalar(
        &mut membrane[tail..],
        &resting[tail..],
        &input_currents[tail..],
        &resistance[tail..],
        &threshold[tail..],
        dt_over_tau,
        &mut spikes_out[tail..],
    );
}

/// `x / 1024` rounded TOWARD ZERO, per `i32` lane.
///
/// `_mm256_srai_epi32(x, 10)` rounds toward −∞; Rust's `/` rounds toward zero.
/// A bare shift therefore makes every negative quotient one unit more negative
/// than the scalar reference, and in a stepped simulation that error does not
/// cancel — it is a constant downward drift, so the two halves settle at
/// different fixed points (module doc § Approximation vs scalar).
///
/// For `x < 0` the sign broadcast `x >> 31` is all-ones, so `& 1023` adds 1023
/// before the shift and floor becomes truncation. For `x >= 0` it adds nothing.
/// The bias only ever moves a negative value toward zero, so it cannot overflow
/// `i32` anywhere inside the domain in § Overflow domain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn div1024_toward_zero(x: __m256i) -> __m256i {
    let bias = _mm256_and_si256(_mm256_srai_epi32(x, 31), _mm256_set1_epi32(1023));
    _mm256_srai_epi32(_mm256_add_epi32(x, bias), 10)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn lif_lane(
    mp: __m256i,
    rp: __m256i,
    ic: __m256i,
    res: __m256i,
    dt_over_tau: __m256i,
) -> __m256i {
    // leak = rp − mp
    let leak = _mm256_sub_epi32(rp, mp);
    // current_term = (ic * res) / 1024, rounded toward zero  [÷1024 ≈ ÷1000]
    let current_scaled = _mm256_mullo_epi32(ic, res);
    let current_term = div1024_toward_zero(current_scaled);
    // delta = ((leak + current) * dt_over_tau) / 1024, rounded toward zero
    let sum = _mm256_add_epi32(leak, current_term);
    let delta = _mm256_mullo_epi32(sum, dt_over_tau);
    let delta_scaled = div1024_toward_zero(delta);
    // new_mp = mp + delta
    let new_mp = _mm256_add_epi32(mp, delta_scaled);
    // clamp to [-100, 50] (matches MEMBRANE_MV_MIN/MAX).
    _mm256_max_epi32(_mm256_set1_epi32(-100), _mm256_min_epi32(_mm256_set1_epi32(50), new_mp))
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    #![allow(clippy::shadow_unrelated)]
    #![allow(clippy::cast_precision_loss)] // test counters are tiny; usize→f64 is lossless in practice
    use super::*;
    use crate::lif_neuron::{LIFNeuron, VoltageResolution};
    use proptest::prelude::*;

    /// Equivalence-domain bound on `|input_current × resistance|` (module doc
    /// § Equivalence domain). Deliberately not public: it constrains what a
    /// caller may pass, and the module doc states it in prose.
    const EQUIV_CURRENT_PRODUCT_MAX: i32 = 100_000;
    /// Equivalence-domain bound on `dt_over_tau` (module doc § Equivalence domain).
    const EQUIV_DT_OVER_TAU_MAX: i32 = 200;
    /// The agreement the equivalence domain buys, in mV.
    const EQUIV_TOLERANCE_MV: i32 = 2;

    /// The five SoA slices a batch needs, owned. Named because the tuple trips
    /// `clippy::type_complexity` under `--features simd`, which no workspace gate
    /// lints (the workspace build does not enable `simd`).
    type SoaBatch = (Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>);

    /// Build an SoA batch inside the equivalence domain from raw proptest draws:
    /// `resistance` is free across `i16`, and `input_current` is squeezed so the
    /// product respects `EQUIV_CURRENT_PRODUCT_MAX`.
    fn soa_in_domain(raw: &[(i16, i16, i16, i16, i16)]) -> SoaBatch {
        let mut membrane = Vec::with_capacity(raw.len());
        let mut resting = Vec::with_capacity(raw.len());
        let mut current = Vec::with_capacity(raw.len());
        let mut resistance = Vec::with_capacity(raw.len());
        let mut threshold = Vec::with_capacity(raw.len());
        for &(mp, rp, ic_raw, res, th) in raw {
            let lim = EQUIV_CURRENT_PRODUCT_MAX / i32::from(res).abs().max(1);
            let lim = lim.min(i32::from(i16::MAX)) as i16;
            membrane.push(mp);
            resting.push(rp);
            current.push(ic_raw.clamp(-lim, lim));
            resistance.push(res);
            threshold.push(th);
        }
        (membrane, resting, current, resistance, threshold)
    }

    /// AVX2 kernel output stays within the biological bounds, like the scalar.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_membrane_stays_bounded() {
        let n = 256;
        let mut mp = vec![60i16; n]; // deliberately above MAX
        let rp = vec![-70i16; n];
        let ic = vec![1000i16; n];
        let res = vec![100i16; n];
        let th = vec![-55i16; n];
        let mut spikes = vec![false; n];
        let dtot = dt_over_tau(1000, 20_000);
        integrate_lif_batch(&mut mp, &rp, &ic, &res, &th, dtot, &mut spikes);
        for &v in &mp {
            assert!((-100..=50).contains(&v), "out of bounds: {v}");
        }
    }

    /// SIMD ≈ scalar within ±2 mV (÷1024 vs ÷1000 approximation). The honesty test.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_approximates_scalar_within_tolerance() {
        if !matches!(detect_simd_support(), SimdSupport::Avx2) {
            eprintln!("(AVX2 not available — skipping equivalence test)");
            return;
        }
        // Varied inputs so both signs + magnitudes of delta are exercised.
        let n = 512;
        let mut mp_a = vec![0i16; n];
        let mut mp_b = vec![0i16; n];
        let mut rp = vec![0i16; n];
        let mut ic = vec![0i16; n];
        let res = vec![100i16; n];
        let mut th = vec![0i16; n];
        for i in 0..n {
            let v = ((i as i32 * 7) % 201 - 100) as i16; // -100..=100
            mp_a[i] = v;
            mp_b[i] = v;
            rp[i] = ((i as i32 * 3) % 201 - 100) as i16;
            ic[i] = ((i as i32 * 11) % 2001 - 1000) as i16; // -1000..=1000
            th[i] = -55 - (i as i16 % 20);
        }
        let dtot = dt_over_tau(1000, 20_000);
        let mut spikes_a = vec![false; n];
        let mut spikes_b = vec![false; n];
        integrate_batch_scalar(&mut mp_a, &rp, &ic, &res, &th, dtot, &mut spikes_a);
        // SAFETY: equal-length slices, AVX2 verified available above.
        unsafe {
            integrate_batch_avx2(&mut mp_b, &rp, &ic, &res, &th, dtot, &mut spikes_b);
        }

        let mut max_diff = 0i32;
        let mut disagree = 0usize;
        for i in 0..n {
            let d = (i32::from(mp_a[i]) - i32::from(mp_b[i])).abs();
            if d > max_diff {
                max_diff = d;
            }
            // Spike agreement is softer — at the exact threshold edge, a 1-mV
            // difference flips the bit. Count disagreements, don't fail on them.
            if spikes_a[i] != spikes_b[i] {
                disagree += 1;
            }
        }
        assert!(max_diff <= 2, "SIMD diverged from scalar by {max_diff} mV (>2)");
        // Sanity: most spikes agree (>90%). Edge-flips only near threshold.
        let disagree_ratio = disagree as f64 / n as f64;
        assert!(disagree_ratio < 0.10, "{disagree}/{n} spike disagreements (>10%)");
    }

    // ----- Overflow domain (module doc § Overflow domain) -----

    /// `DT_OVER_TAU_MAX` is exactly the bound the module doc derives — the test
    /// encodes the same arithmetic, so widening the doc without widening the
    /// code (or the reverse) turns this red.
    #[test]
    fn dt_over_tau_max_is_the_documented_bound() {
        // |leak| ≤ 65_535 and |P|/1000 ≤ 1_073_741  ⇒  |sum| ≤ 1_139_276.
        let max_leak = i64::from(i16::MAX) - i64::from(i16::MIN);
        let max_product = i64::from(i16::MIN) * i64::from(i16::MIN);
        let max_current_term = max_product / 1000;
        let max_sum = max_leak + max_current_term;
        assert_eq!(max_leak, 65_535);
        assert_eq!(max_product, 1_073_741_824);
        assert_eq!(max_sum, 1_139_276);

        let bound = i64::from(DT_OVER_TAU_MAX);
        assert!(
            bound * max_sum <= i64::from(i32::MAX),
            "DT_OVER_TAU_MAX is too large: {DT_OVER_TAU_MAX} * {max_sum} overflows i32"
        );
        let next = DT_OVER_TAU_MAX + 1;
        assert!(
            (bound + 1) * max_sum > i64::from(i32::MAX),
            "DT_OVER_TAU_MAX is not tight: {next} would also fit"
        );
    }

    /// `dt_over_tau` never returns a value outside the safe domain, and the
    /// `as i32` casts it used to do are gone.
    #[test]
    fn dt_over_tau_is_non_negative_and_saturated_over_the_whole_u32_domain() {
        // The historical regression: `dt_us as i32` and `tau as i32` both wrapped.
        // `(2_147_484 * 1000).saturating_mul` hit i32::MAX, `u32::MAX as i32` was
        // -1, and the quotient came out -2_147_483_647.
        assert_eq!(dt_over_tau(2_147_484, u32::MAX), 0);

        for &dt in &[0u32, 1, 1000, 10_000, i32::MAX as u32, 2_147_484, u32::MAX] {
            for &tau in &[1u32, 20_000, i32::MAX as u32, 2_147_483_648, u32::MAX] {
                let v = dt_over_tau(dt, tau);
                assert!(
                    (0..=DT_OVER_TAU_MAX).contains(&v),
                    "dt_over_tau({dt}, {tau}) = {v} is outside 0..={DT_OVER_TAU_MAX}"
                );
            }
        }
        assert_eq!(dt_over_tau(0, 20_000), 0, "tau == 0 guard unchanged");
        assert_eq!(dt_over_tau(1000, 0), 0, "tau == 0 guard unchanged");
        assert_eq!(dt_over_tau(1000, 20_000), 50, "the physical default is untouched");
    }

    /// Every `i16` corner, at the largest legal `dt_over_tau`, in both halves:
    /// no `i32` intermediate overflows (debug builds panic on overflow, which is
    /// the falsifier) and both halves stay on the mV grid. They do NOT agree
    /// here — the corners are far outside the equivalence domain — so the
    /// divergence is pinned exactly instead.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn overflow_corners_at_max_dt_over_tau() {
        const CORNERS: [i16; 5] = [i16::MIN, -1, 0, 1, i16::MAX];
        let (mut mp, mut rp, mut ic, mut res, mut th) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &a in &CORNERS {
            for &b in &CORNERS {
                for &c in &CORNERS {
                    for &d in &CORNERS {
                        for &e in &CORNERS {
                            mp.push(a);
                            rp.push(b);
                            ic.push(c);
                            res.push(d);
                            th.push(e);
                        }
                    }
                }
            }
        }
        assert_eq!(mp.len(), 5usize.pow(5), "3125 = chunks plus a tail");

        // Scalar first: in a debug build an overflowing `*` panics here.
        let mut mp_s = mp.clone();
        let mut sp_s = vec![false; mp.len()];
        integrate_batch_scalar(&mut mp_s, &rp, &ic, &res, &th, DT_OVER_TAU_MAX, &mut sp_s);
        for &v in &mp_s {
            assert!((-100..=50).contains(&v), "scalar left the mV grid: {v}");
        }

        if !matches!(detect_simd_support(), SimdSupport::Avx2) {
            eprintln!("(AVX2 not available — corner agreement not checked)");
            return;
        }
        let mut mp_v = mp.clone();
        let mut sp_v = vec![false; mp.len()];
        // SAFETY: equal-length slices, AVX2 verified available above.
        unsafe {
            integrate_batch_avx2(&mut mp_v, &rp, &ic, &res, &th, DT_OVER_TAU_MAX, &mut sp_v);
        }
        for &v in &mp_v {
            assert!((-100..=50).contains(&v), "AVX2 left the mV grid: {v}");
        }

        // The corners are far OUTSIDE the equivalence domain, so the halves are
        // not expected to agree here — only to stay finite and on the grid. The
        // divergence is pinned exactly, so any arithmetic change moves it.
        let max_diff = mp_s
            .iter()
            .zip(&mp_v)
            .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
            .max()
            .expect("non-empty batch");
        let membrane_diffs = mp_s.iter().zip(&mp_v).filter(|(a, b)| a != b).count();
        let spike_diffs = sp_s.iter().zip(&sp_v).filter(|(a, b)| a != b).count();
        assert_eq!(
            (max_diff, membrane_diffs, spike_diffs),
            (4, 180, 0),
            "corner divergence moved; the module doc records 4 mV / 180 / 0 at dt_over_tau = {DT_OVER_TAU_MAX}"
        );
    }

    /// The slice-length contract is enforced BEFORE the caller's buffer is
    /// touched, in every profile.
    ///
    /// It used to be five `debug_assert_eq!`s, so release builds enforced
    /// nothing. The AVX2 chunk loop indexes all six slices by the same offsets,
    /// so a short slice was read past its end and the derived values were
    /// written into the caller's `membrane` before the tail slicing panicked.
    /// Reproduced on this branch before the fix: `--release`, membrane 32,
    /// resting 17 — `membrane[17..]` came back holding values no in-bounds
    /// input could produce, then `range start index 32 out of range for slice
    /// of length 17`. An out-of-bounds read reachable from safe code in a
    /// published crate.
    ///
    /// Asserting only "it panics" is not enough, and a first draft of this test
    /// made exactly that mistake: with `debug_assert_eq!` restored it still
    /// passed in release, because the out-of-bounds run panics anyway when it
    /// reaches the tail slicing. So this checks the two things that actually
    /// separate an enforced contract from an accidental bounds-check crash:
    ///
    /// - a slice SHORTER than `membrane` must panic with `membrane` still
    ///   untouched — no partial write from out-of-bounds reads;
    /// - a slice LONGER than `membrane` must panic at all. Nothing is out of
    ///   bounds there, so the unenforced version integrated a prefix and
    ///   returned normally.
    ///
    /// `catch_unwind` rather than `#[should_panic]` so one test covers all five
    /// slices in both directions, identically under `cargo test` and
    /// `cargo test --release`.
    #[test]
    fn unequal_slice_lengths_panic_before_touching_the_caller() {
        const N: usize = 32;
        const SHORT: usize = 17; // not a multiple of the 16-lane width
        const SENTINEL: i16 = -70;
        const NAMES: [&str; 5] =
            ["resting", "input_currents", "resistance", "threshold", "spikes_out"];

        // resting = 0 against membrane = -70 gives leak = 70 and delta = +3, so
        // any element the kernel actually processes moves off SENTINEL.
        let run = |lens: [usize; 5]| -> (bool, Vec<i16>) {
            let mut membrane = vec![SENTINEL; N];
            let mut spikes = vec![false; lens[4]];
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                integrate_lif_batch(
                    &mut membrane,
                    &vec![0i16; lens[0]],
                    &vec![0i16; lens[1]],
                    &vec![100i16; lens[2]],
                    &vec![-55i16; lens[3]],
                    50,
                    &mut spikes,
                );
            }))
            .is_err();
            (panicked, membrane)
        };

        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let mut failures: Vec<String> = Vec::new();

        for (i, name) in NAMES.iter().enumerate() {
            let mut lens = [N; 5];
            lens[i] = SHORT;
            let (panicked, membrane) = run(lens);
            if !panicked {
                failures.push(format!("short `{name}`: no panic"));
            }
            if let Some(pos) = membrane.iter().position(|&v| v != SENTINEL) {
                failures.push(format!(
                    "short `{name}`: membrane[{pos}] = {} was written before the panic",
                    membrane[pos]
                ));
            }

            let mut lens = [N; 5];
            lens[i] = N * 2;
            let (panicked, membrane) = run(lens);
            if !panicked {
                failures.push(format!("long `{name}`: no panic — a prefix was integrated silently"));
            }
            if let Some(pos) = membrane.iter().position(|&v| v != SENTINEL) {
                failures.push(format!(
                    "long `{name}`: membrane[{pos}] = {} was written before the panic",
                    membrane[pos]
                ));
            }
        }
        std::panic::set_hook(previous);

        assert!(failures.is_empty(), "length contract unenforced:
  {}", failures.join("
  "));
    }

    /// The batch with `SimdSupport::None`-forcing path must still match scalar
    /// exactly when the scalar is called directly (sanity for the dispatch seam).
    #[test]
    fn scalar_matches_itself() {
        let n = 64;
        let mut a = vec![-70i16; n];
        let mut b = vec![-70i16; n];
        let rp = vec![-70i16; n];
        let ic = vec![200i16; n];
        let res = vec![100i16; n];
        let th = vec![-55i16; n];
        let dtot = dt_over_tau(1000, 20_000);
        let mut sa = vec![false; n];
        let mut sb = vec![false; n];
        integrate_batch_scalar(&mut a, &rp, &ic, &res, &th, dtot, &mut sa);
        integrate_batch_scalar(&mut b, &rp, &ic, &res, &th, dtot, &mut sb);
        assert_eq!(a, b);
        assert_eq!(sa, sb);
    }

    /// The bit-equality below, swept DETERMINISTICALLY over the regime where the
    /// result lands strictly inside the `[-100, 50]` clamp.
    ///
    /// The clamp is what makes a random-i16 sweep blind: saturate both sides and
    /// any arithmetic error is hidden behind the same bound. A property test can
    /// therefore pass on one seed and fail on another, which is not a pin. This
    /// test takes no draws, so every constant in `integrate_batch_scalar` is
    /// observable on every run — `/1000 -> /1024` turns it red.
    #[test]
    fn scalar_batch_matches_integrate_and_fire_in_the_unclamped_regime() {
        let mut compared = 0usize;
        let mut unclamped = 0usize;
        for &mp in &[-90i16, -70, -55, 0, 40] {
            for &rp in &[-100i16, -70, 0, 50] {
                for &resistance in &[1i16, 10, 100, 500, 1000] {
                    for input in (-1500i16..=1500).step_by(37) {
                        for &(dt_us, tau_us) in &[(1000u32, 20_000u32), (500, 20_000), (100, 10_000)]
                        {
                            let mut n = LIFNeuron::new(0);
                            n.voltage_resolution = VoltageResolution::Millivolt;
                            n.membrane_potential = mp;
                            n.resting_potential = rp;
                            n.threshold = i16::MAX; // unreachable: raw membrane, no reset
                            n.tau_membrane_us = tau_us;
                            n.resistance_mohm = resistance as u16;
                            n.noise_amplitude_ua = 0;
                            n.synaptic_current_ua = 0;
                            n.adaptation_current_ua = 0;
                            n.refractory_time_us = 0;
                            let fired = n.integrate_and_fire(input, dt_us, 0);
                            assert!(!fired, "i16::MAX threshold must be unreachable");

                            let dtot = dt_over_tau(dt_us, tau_us);
                            let mut membrane = vec![mp];
                            let mut spikes = vec![false];
                            integrate_batch_scalar(
                                &mut membrane,
                                &[rp],
                                &[input],
                                &[resistance],
                                &[i16::MAX],
                                dtot,
                                &mut spikes,
                            );
                            assert_eq!(
                                membrane[0], n.membrane_potential,
                                "mp={mp} rp={rp} input={input} resistance={resistance} \
                                 dt_us={dt_us} tau_us={tau_us} dt_over_tau={dtot}"
                            );
                            assert_eq!(spikes[0], fired, "spike bit differs at an unreachable threshold");
                            compared += 1;
                            if membrane[0] != -100 && membrane[0] != 50 {
                                unclamped += 1;
                            }
                        }
                    }
                }
            }
        }
        // The point of the sweep is the unclamped rows. If the grid ever drifts
        // into all-saturating territory it stops testing the arithmetic, and this
        // test would go quietly useless the way the property test nearly did.
        assert!(compared >= 5000, "sweep shrank to {compared} rows");
        assert!(
            unclamped * 4 >= compared,
            "only {unclamped}/{compared} rows landed inside the clamp — the sweep has gone blind"
        );
    }

    /// The AVX2 scalar remainder covers EVERY element past the last full 16-lane
    /// chunk, including the last one. Lengths 1, 15, 16, 17, 31 and 33 cover
    /// sub-width, exact-chunk and both sides of a chunk boundary.
    ///
    /// Inputs are chosen so the correct answer differs from the input membrane at
    /// every index and every spike bit flips to `true`, so a skipped element is
    /// visible rather than accidentally correct: `tail = chunks * WIDTH + 1`
    /// leaves index 0 untouched at n = 1 and panics on the slice for n >= 16.
    ///
    /// The two halves are deliberately NOT compared to each other here. The
    /// inputs are chosen so they still disagree by 1 mV after the rounding fix,
    /// because that disagreement is the marker: `leak = 70` with 500 μA into
    /// 100 MΩ gives the scalar `current_term = 50`, `delta = +6`, `-64`, and
    /// the vector `current_term = 48` (50_000 ÷ 1024), `delta = +5`, `-65`. The
    /// gap is the ÷1024 scale, which the rounding fix does not close and is not
    /// meant to. That split marks exactly where the chunk loop stops and the
    /// scalar tail starts, so the boundary itself is what gets asserted.
    ///
    /// The previous marker (`leak = -70`, giving `-4` against `-3`) stopped
    /// working when the shift began truncating toward zero: both halves then
    /// gave `-3` and the boundary became invisible. A test that silently stops
    /// discriminating is the failure mode this module keeps hitting, so the
    /// marker is asserted as an exact vector, not as a tolerance.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_tail_writes_every_remainder_element() {
        const DTOT: i32 = 50; // 1 ms / 20 ms
        if !matches!(detect_simd_support(), SimdSupport::Avx2) {
            eprintln!("(AVX2 not available — skipping tail test)");
            return;
        }
        for n in [0usize, 1, 15, 16, 17, 31, 32, 33, 47, 48, 257] {
            let membrane = vec![-70i16; n];
            let resting = vec![0i16; n];
            let current = vec![500i16; n];
            let resistance = vec![100i16; n];
            let threshold = vec![-100i16; n]; // every correct element spikes

            let mut mp_v = membrane.clone();
            let mut sp_v = vec![false; n];
            // SAFETY: equal-length slices, AVX2 verified available above.
            unsafe {
                integrate_batch_avx2(
                    &mut mp_v, &resting, &current, &resistance, &threshold, DTOT, &mut sp_v,
                );
            }

            let vector_lanes = (n / 16) * 16;
            let expected: Vec<i16> = (0..n)
                .map(|i| if i < vector_lanes { -65 } else { -64 })
                .collect();
            assert_eq!(
                mp_v, expected,
                "n={n}: {vector_lanes} vector lanes then a {} element tail",
                n - vector_lanes
            );
            assert!(
                sp_v.iter().all(|&b| b),
                "n={n}: a spike bit was never written — got {sp_v:?}"
            );
            if n > 0 {
                assert_ne!(mp_v[n - 1], -70, "n={n}: the LAST element was not written");
                assert!(sp_v[n - 1], "n={n}: the LAST spike bit was not written");
            }
        }
    }

    /// The two halves settle in the same place across a long run, and the gap
    /// between them does not grow with step count.
    ///
    /// This is the multi-step claim the single-step ±2 mV bound does not make.
    /// Before `div1024_toward_zero`, the vector half rounded every negative
    /// delta one unit further from zero than the scalar, which is invisible in
    /// one step and decisive over many: at −200 μA drive the scalar parked at
    /// −71 mV and the AVX2 half at −90 mV, a 19 mV gap against the 15 mV that
    /// separates rest from threshold.
    ///
    /// What remains after the fix is the ÷1024 scale, which shifts each half's
    /// dead-zone interval and so its parking spot. That residual is bounded, not
    /// accumulating: every arm below gives the same difference at 1_000 steps as
    /// at 20_000, and the three named arms were separately checked out to
    /// 1_000_000. The maxima are pinned exactly rather than as inequalities, so
    /// any change to the kernel's arithmetic moves them.
    ///
    /// The arm set is not a sample. The last two arms are the domain-wide worst
    /// cases found by exhaustively sweeping `resting` × every distinct
    /// `(current_term_scalar, current_term_avx2)` class × `dt_over_tau`, so the
    /// pinned `(8, 8)` is the true maximum over the equivalence domain and not
    /// merely the maximum over whichever arms someone thought to write down.
    /// It was `(4, 5)` before those two arms existed, which is exactly that
    /// failure — the arm set was the measurement, and it was not the domain.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_and_scalar_trajectories_stay_bounded() {
        // (resting, drive, resistance, dt_over_tau), all inside the equivalence
        // domain. The first three are the reviewer's named arms; the last two are
        // the worst fixed-point and worst trajectory cases found by sweeping it.
        const ARMS: [(i16, i16, i16, i32); 11] = [
            (-70, 0, 100, 50),
            (-70, 200, 100, 50),
            (-70, -200, 100, 50),
            (-70, 1000, 100, 50),
            (-70, -1000, 100, 50),
            (-70, 500, 100, 50),
            (-70, -500, 100, 50),
            (0, -10000, 10, 50),
            (50, 7500, 10, 200),
            // The two domain-wide worst cases, both 8 mV. The vector half never
            // moves in either: its delta truncates to zero every step while the
            // scalar climbs away from the start. Named in the module doc.
            (37, 1000, 100, 5),   // the reviewer's witness
            (50, 870, 100, 5),    // the extremal case found by the exhaustive sweep
        ];
        const N: usize = 16;

        if !matches!(detect_simd_support(), SimdSupport::Avx2) {
            eprintln!("(AVX2 not available — skipping trajectory test)");
            return;
        }

        // Returns (difference at the end, worst difference at any step).
        let run = |resting: i16, drive: i16, resistance: i16, dtot: i32, steps: usize| {
            let rp = vec![resting; N];
            let ic = vec![drive; N];
            let res = vec![resistance; N];
            let th = vec![i16::MAX; N]; // unreachable: no spike, no reset
            let mut mp_s = vec![-70i16; N];
            let mut mp_v = vec![-70i16; N];
            let mut sp_s = vec![false; N];
            let mut sp_v = vec![false; N];
            let mut worst_step = 0i32;
            for _ in 0..steps {
                integrate_batch_scalar(&mut mp_s, &rp, &ic, &res, &th, dtot, &mut sp_s);
                integrate_lif_batch(&mut mp_v, &rp, &ic, &res, &th, dtot, &mut sp_v);
                let d = (i32::from(mp_s[0]) - i32::from(mp_v[0])).abs();
                if d > worst_step {
                    worst_step = d;
                }
            }
            ((i32::from(mp_s[0]) - i32::from(mp_v[0])).abs(), worst_step)
        };

        let mut worst_end = 0i32;
        let mut worst_traj = 0i32;
        for &(resting, drive, resistance, dtot) in &ARMS {
            let (short_end, short_worst) = run(resting, drive, resistance, dtot, 1_000);
            let (long_end, long_worst) = run(resting, drive, resistance, dtot, 20_000);
            assert_eq!(
                (short_end, short_worst),
                (long_end, long_worst),
                "arm rp={resting} drive={drive} res={resistance} dt_over_tau={dtot}: the gap \
                 grew between 1_000 and 20_000 steps, so it is accumulating, not parking"
            );
            worst_end = worst_end.max(long_end);
            worst_traj = worst_traj.max(long_worst);
        }

        // The reviewer's three named arms, exactly.
        let named: Vec<i32> = [0i16, 200, -200]
            .iter()
            .map(|&drive| run(-70, drive, 100, 50, 10_000).0)
            .collect();
        assert_eq!(named, vec![0, 1, 1], "the named arms moved (they were 0, 1, 19 before the fix)");

        assert_eq!(
            (worst_end, worst_traj),
            (8, 8),
            "trajectory bounds moved; the module doc records 8 mV domain-wide, at \
             the fixed point and at any step, established by exhaustive sweep"
        );
    }

    // ----- Property tests -----

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// The batch scalar reference IS `LIFNeuron::integrate_and_fire` on the mV
        /// grid — bit-equal membranes, identical spike bit — for any `i16` input
        /// with `dt_us <= tau_membrane_us`.
        ///
        /// The module doc has claimed this since the port ("exact /1000, matching
        /// integrate_and_fire") with nothing pinning it. Two neurons are stepped
        /// from the same state: one with an unreachable threshold, whose membrane
        /// is therefore the raw integrated value the batch computes, and one with
        /// the real threshold, whose return value is the spike bit. The four
        /// differences that are NOT arithmetic are neutralised in the fixture and
        /// named in the module doc: noise, synaptic current, adaptation current,
        /// and the post-spike reset.
        #[test]
        fn prop_scalar_batch_is_bit_equal_to_integrate_and_fire(
            mp in -100i16..=50,
            rp in -100i16..=50,
            th in -100i16..=50,
            // Two regimes. The wide arm saturates the clamp on almost every draw,
            // which makes it blind on its own: `/1000 -> /1024` in
            // integrate_batch_scalar survived it, because both sides then pin to
            // the same clamp bound. The small-signal arm keeps the result inside
            // the clamp, where the arithmetic is observable. The deterministic
            // sweep below is what actually holds the pin; this arm only widens
            // the search.
            // resistance_mohm is u16 on the neuron and i16 in the SoA batch; only
            // the non-negative overlap is representable in both.
            (input, resistance) in prop_oneof![
                (any::<i16>(), 0i16..=i16::MAX),
                (-2000i16..=2000, 0i16..=1000),
            ],
            tau_us in 1u32..=1_000_000,
            dt_ratio in 0u32..=1000,
        ) {
            let dt_us = u32::try_from(u64::from(tau_us) * u64::from(dt_ratio) / 1000)
                .expect("dt_us <= tau_us <= u32::MAX");

            let fixture = |threshold: i16| {
                let mut n = LIFNeuron::new(0);
                n.voltage_resolution = VoltageResolution::Millivolt;
                n.membrane_potential = mp;
                n.resting_potential = rp;
                n.threshold = threshold;
                n.tau_membrane_us = tau_us;
                n.resistance_mohm = resistance as u16;
                // The four non-arithmetic differences, neutralised.
                n.noise_amplitude_ua = 0;
                n.synaptic_current_ua = 0;
                n.adaptation_current_ua = 0;
                n.refractory_time_us = 0;
                n
            };

            // The shared scaling factor, computed both ways.
            let dtot = dt_over_tau(dt_us, tau_us);
            let inline = ((dt_us as i32) * 1000) / tau_us as i32;
            prop_assert_eq!(
                dtot, inline,
                "dt_over_tau({}, {}) disagrees with the inline formula in integrate_and_fire",
                dt_us, tau_us
            );

            // Unreachable threshold: no spike, so the membrane is the raw value.
            let mut quiet = fixture(i16::MAX);
            let fired_quiet = quiet.integrate_and_fire(input, dt_us, 0);
            prop_assert!(!fired_quiet, "i16::MAX threshold must be unreachable");

            // Real threshold: the return value is the spike bit.
            let mut live = fixture(th);
            let fired = live.integrate_and_fire(input, dt_us, 0);

            let mut membrane = vec![mp];
            let mut spikes = vec![false];
            integrate_batch_scalar(
                &mut membrane, &[rp], &[input], &[resistance], &[th], dtot, &mut spikes,
            );

            prop_assert_eq!(
                membrane[0], quiet.membrane_potential,
                "membrane differs: batch {} vs integrate_and_fire {} \
                 (mp={} rp={} input={} resistance={} dt_over_tau={})",
                membrane[0], quiet.membrane_potential, mp, rp, input, resistance, dtot
            );
            prop_assert_eq!(
                spikes[0], fired,
                "spike bit differs at threshold {} with membrane {}",
                th, quiet.membrane_potential
            );
        }

        /// AVX2 ≡ scalar over the documented equivalence domain: membranes agree
        /// within ±2 mV, and the two disagree on a spike ONLY where the scalar
        /// membrane sits inside that same 2 mV band around the neuron's threshold.
        ///
        /// Lengths run 0..=257 so the empty batch, sub-width batches, exact
        /// 16-lane chunks and every tail remainder all shrink.
        #[test]
        #[cfg(target_arch = "x86_64")]
        fn prop_avx2_matches_scalar_in_the_equivalence_domain(
            raw in prop::collection::vec(
                (-100i16..=50, -100i16..=50, any::<i16>(), any::<i16>(), -100i16..=50),
                0..=257,
            ),
            dtot in 0i32..=EQUIV_DT_OVER_TAU_MAX,
        ) {
            if !matches!(detect_simd_support(), SimdSupport::Avx2) {
                return Ok(());
            }
            let (membrane, resting, current, resistance, threshold) = soa_in_domain(&raw);
            let n = membrane.len();

            let mut mp_s = membrane.clone();
            let mut sp_s = vec![false; n];
            integrate_batch_scalar(
                &mut mp_s, &resting, &current, &resistance, &threshold, dtot, &mut sp_s,
            );

            let mut mp_v = membrane.clone();
            let mut sp_v = vec![false; n];
            // SAFETY: equal-length slices, AVX2 verified available above.
            unsafe {
                integrate_batch_avx2(
                    &mut mp_v, &resting, &current, &resistance, &threshold, dtot, &mut sp_v,
                );
            }

            for i in 0..n {
                let diff = (i32::from(mp_s[i]) - i32::from(mp_v[i])).abs();
                prop_assert!(
                    diff <= EQUIV_TOLERANCE_MV,
                    "neuron {i}: scalar {} vs avx2 {} differ by {diff} mV (> {EQUIV_TOLERANCE_MV}); \
                     mp={} rp={} ic={} res={} dt_over_tau={dtot}",
                    mp_s[i], mp_v[i], membrane[i], resting[i], current[i], resistance[i],
                );
                if sp_s[i] != sp_v[i] {
                    let margin = (i32::from(mp_s[i]) - i32::from(threshold[i])).abs();
                    prop_assert!(
                        margin <= EQUIV_TOLERANCE_MV,
                        "neuron {i}: spike disagreement {} vs {} with the scalar membrane {} \
                         a full {margin} mV from threshold {} — outside the edge band",
                        sp_s[i], sp_v[i], mp_s[i], threshold[i],
                    );
                }
            }
        }
    }
}
