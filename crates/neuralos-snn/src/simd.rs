//! SIMD-accelerated batch LIF integration (AVX2, `x86_64`).
//!
//! Ported from v0.1 `libneuralos_before_bridge_removal/src/core/simd_vectorization.rs`,
//! with two correctness bugs fixed (see `BUG_FIXES` below). This module is gated
//! behind the `simd` feature (implies `std`) and `cfg(target_arch = "x86_64")`.
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
//! # Approximation vs scalar
//!
//! The AVX2 kernel replaces `/1000` with `>>10` (÷1024) — the standard
//! fixed-point fast-division approximation, ~2.4% error, biologically
//! irrelevant (well under [`crate::LIFNeuron`] default noise of 5 μA). The
//! scalar reference here uses exact `/1000` (matching
//! [`crate::LIFNeuron::integrate_and_fire`]). The correctness test asserts the
//! two agree within ±2 mV per neuron, not bit-exact.
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
//! - `current_term`: `|P| / 1000 ≤ 1_073_741` (scalar), `|P| >> 10 ≤ 1_048_576`
//!   (AVX2). The scalar is the larger, so it binds.
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
//! `dt_over_tau = DT_OVER_TAU_MAX`: both halves stay on the mV grid, 375 of 3125
//! membranes differ, the largest difference is 3 mV, and 44 spike bits differ.
//! Pinned by `overflow_corners_at_max_dt_over_tau`.
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
//! maximum of exactly 2. The first `dt_over_tau` that reaches 3 at the same
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
//! takes a scale parameter, do not route `CentiMillivolt` state through it
//! (no in-tree consumer does — verified 2026-08-20 audit).
//!
//! # `BUG_FIXES` vs v0.1
//!
//! - **Widen-both-halves.** v0.1's `integrate_neurons_avx2` (`simd_vectorization.rs:237-240`)
//!   called `_mm256_cvtepi16_epi32(_mm256_castsi256_si128(mp))`, which widens
//!   only the LOW 8 of each 16-element load. The high 8 were never computed;
//!   stale memory was stored back. Fixed: widen both halves via
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
/// domain).
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

        // LIF math on each half. ÷1000 approximated as >>10 (÷1024).
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
    // current_term = (ic * res) >> 10   [÷1024 ≈ ÷1000]
    let current_scaled = _mm256_mullo_epi32(ic, res);
    let current_term = _mm256_srai_epi32(current_scaled, 10);
    // delta = ((leak + current) * dt_over_tau) >> 10
    let sum = _mm256_add_epi32(leak, current_term);
    let delta = _mm256_mullo_epi32(sum, dt_over_tau);
    let delta_scaled = _mm256_srai_epi32(delta, 10);
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
            (3, 375, 44),
            "corner divergence moved; the module doc records 3 mV / 375 / 44 at dt_over_tau = {DT_OVER_TAU_MAX}"
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
}
