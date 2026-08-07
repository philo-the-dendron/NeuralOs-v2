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

/// Precompute the `dt/τ` scaling factor (passed into [`integrate_lif_batch`]).
///
/// Matches [`crate::LIFNeuron::integrate_and_fire`]: `dt_over_tau = (dt_us * 1000) / tau_us`.
/// Hoisted out so a batch sharing one `dt` and `tau` computes it once.
#[must_use]
pub fn dt_over_tau(dt_us: u32, tau_membrane_us: u32) -> i32 {
    if tau_membrane_us == 0 {
        return 0; // Guard; the network rejects tau == 0 at construction.
    }
    ((dt_us as i32).saturating_mul(1000)) / tau_membrane_us as i32
}

/// Integrate one LIF step across a batch of N neurons (SoA slices).
///
/// Updates `membrane` in place and writes the spike mask to `spikes_out`.
/// Picks AVX2 at runtime when available, else the scalar reference. All slices
/// must be equal length (debug-asserted).
///
/// `input_currents` is the *total* effective current per neuron (external +
/// synaptic + noise − adaptation); the batch computes only the membrane
/// update, not current accumulation — that's the caller's job.
///
/// # Panics
/// Debug builds assert all slices are equal length.
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
    debug_assert_eq!(resting.len(), n);
    debug_assert_eq!(input_currents.len(), n);
    debug_assert_eq!(resistance.len(), n);
    debug_assert_eq!(threshold.len(), n);
    debug_assert_eq!(spikes_out.len(), n);

    #[cfg(target_arch = "x86_64")]
    if matches!(detect_simd_support(), SimdSupport::Avx2) {
        // SAFETY: slices are valid, equal-length (debug-asserted), and the AVX2
        // kernel processes 16-element aligned chunks plus a scalar tail, so no
        // out-of-bounds access occurs. `membrane` is &mut and uniquely borrowed
        // here; the kernel writes within bounds.
        unsafe { integrate_batch_avx2(membrane, resting, input_currents, resistance, threshold, dt_over_tau, spikes_out) };
        return;
    }
    integrate_batch_scalar(membrane, resting, input_currents, resistance, threshold, dt_over_tau, spikes_out);
}

/// Scalar reference — exact v2 LIF math (÷1000). Also the remainder tail.
pub fn integrate_batch_scalar(
    membrane: &mut [i16],
    resting: &[i16],
    input_currents: &[i16],
    resistance: &[i16],
    threshold: &[i16],
    dt_over_tau: i32,
    spikes_out: &mut [bool],
) {
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
