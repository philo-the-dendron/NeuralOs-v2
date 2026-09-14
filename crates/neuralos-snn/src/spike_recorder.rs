//! Spike recorder: the bounded spike-timestamp history a caller keeps when
//! it wants one.
//!
//! [`LIFNeuron`](crate::lif_neuron::LIFNeuron) keeps no history. It stamps
//! `last_spike_time_us` and returns `true` from `integrate_and_fire` on a
//! spike; a caller that wants the spike times records them:
//!
//! ```
//! use neuralos_snn::{LIFNeuron, SpikeRecorder};
//!
//! let mut n = LIFNeuron::new(0);
//! n.threshold = -100; // the membrane floor: every integrating step fires
//! let mut rec = SpikeRecorder::new();
//! for step in 0..4_u32 {
//!     let t = step * 1_000;
//!     if n.integrate_and_fire(0, 1_000, t) {
//!         rec.record(t);
//!     }
//! }
//! // 2 ms refractory at 1 ms steps: a spike at 0, the next at 3 ms.
//! assert!(rec.iter().eq([0, 3_000]));
//! ```
//!
//! The ring was a field of every `LIFNeuron` until alpha.8: most of the
//! neuron's bytes, written on every spike, read only by the tests.
//!
//! # `no_std`
//!
//! A ring over `[u32; MAX_SPIKE_HISTORY]`: compile-time capacity, no
//! allocator, no dependency.

/// Maximum spike history length of a [`SpikeRecorder`] (compile-time, no
/// allocator).
pub const MAX_SPIKE_HISTORY: usize = 64;
// `SpikeRecorder::record` indexes with `& (MAX_SPIKE_HISTORY - 1)`, the same
// slot as `% MAX_SPIKE_HISTORY` only for a power of two.
const _: () = assert!(MAX_SPIKE_HISTORY.is_power_of_two());

/// Bounded spike-timestamp history: a ring over
/// `[u32; MAX_SPIKE_HISTORY]` with the index of the oldest entry and a
/// length. [`record`](Self::record) is O(1): once full, the oldest entry is
/// overwritten (recent spikes are what the consumers want). Iteration runs
/// oldest to newest. Replaced `heapless::Vec` (2026-09-10): its `remove(0)`
/// shifted the whole buffer on every spike past the 64th.
#[derive(Clone)]
pub struct SpikeRecorder {
    buf: [u32; MAX_SPIKE_HISTORY],
    /// Index of the oldest live entry. Meaningless while `len == 0`.
    head: usize,
    len: usize,
}

impl SpikeRecorder {
    /// An empty recorder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buf: [0; MAX_SPIKE_HISTORY],
            head: 0,
            len: 0,
        }
    }

    /// Append `t_us` as the newest entry; drop the oldest when full.
    ///
    /// Both stores index with `& (MAX_SPIKE_HISTORY - 1)`: the same slot as
    /// `%` and as the bare `head` (a power of two, asserted beside the
    /// constant; `head` stays below it), and a bound the optimizer reads from
    /// the index itself. The bare `self.buf[self.head]` was the one index
    /// whose bound did not follow from its own expression, and its bounds
    /// check changed how LLVM treated the whole neuron (ISA round 26,
    /// record-only (d); round 27, item 4).
    pub fn record(&mut self, t_us: u32) {
        debug_assert!(self.head < MAX_SPIKE_HISTORY);
        if self.len < MAX_SPIKE_HISTORY {
            self.buf[(self.head + self.len) & (MAX_SPIKE_HISTORY - 1)] = t_us;
            self.len += 1;
        } else {
            self.buf[self.head & (MAX_SPIKE_HISTORY - 1)] = t_us;
            self.head = (self.head + 1) % MAX_SPIKE_HISTORY;
        }
    }

    /// Live entries, at most [`MAX_SPIKE_HISTORY`].
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when nothing was recorded since [`new`](Self::new) or the last
    /// [`clear`](Self::clear).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Entries oldest to newest. Indexes the buffer itself: the checked
    /// `get` is test-only.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.len).map(move |i| self.buf[(self.head + i) % MAX_SPIKE_HISTORY])
    }

    /// Drop every entry.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

/// An empty recorder, as [`SpikeRecorder::new`].
impl Default for SpikeRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Only the live entries, oldest first — never the stale slots.
impl core::fmt::Debug for SpikeRecorder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// The read side the tests need, test-gated like its callers (the neuron's
/// tests and this module's; a public walk is alpha.9's question).
#[cfg(test)]
#[allow(clippy::cast_possible_truncation)] // bounds at each cast
impl SpikeRecorder {
    /// The `i`-th entry, oldest first. Panics when `i >= len()`, like a
    /// slice index would.
    pub(crate) fn get(&self, i: usize) -> u32 {
        assert!(
            i < self.len,
            "SpikeRecorder index {i} out of range {}",
            self.len
        );
        self.buf[(self.head + i) % MAX_SPIKE_HISTORY]
    }

    /// Firing rate within the window ending at `now_us`, in millihertz.
    /// The neuron's tests pass its `last_update_time_us` (the v0.1 bug fix:
    /// the window is keyed on advancing time, not on the last spike).
    pub(crate) fn firing_rate_mhz(&self, now_us: u32, window_us: u32) -> u32 {
        if self.is_empty() || window_us == 0 {
            return 0;
        }
        let window_start = now_us.saturating_sub(window_us);
        // At most MAX_SPIKE_HISTORY: fits u32.
        let count = self.iter().filter(|&t| t >= window_start).count() as u32;
        // spikes/sec × 1000 = (count × 1_000_000_000) / window_us
        count
            .saturating_mul(1_000_000_000)
            .checked_div(window_us)
            .unwrap_or(0)
    }

    /// Inter-spike-interval statistics: `(mean_us, std_dev_us)`, or
    /// `None` if `< 2` spikes. Pure integer math; std-dev via
    /// sum-of-squares.
    pub(crate) fn isi_stats_us(&self) -> Option<(u32, u32)> {
        if self.len() < 2 {
            return None;
        }
        let n = self.len();
        // u128 accumulators (alpha.6): an interval is at most u32::MAX,
        // its square below 2^64, and up to MAX_SPIKE_HISTORY - 1 of them
        // are summed. In u64 the sum of squares overflowed once three
        // intervals exceeded 2^32 / sqrt(3), about 2.48e9 µs, which the
        // non-monotonic filter below makes reachable (found 2026-09-10
        // by the differential proptest).
        let mut intervals_sum: u128 = 0;
        let mut intervals_sqsum: u128 = 0;
        let mut count: u128 = 0;
        for i in 1..n {
            let prev = self.get(i - 1);
            let curr = self.get(i);
            if curr >= prev {
                let d = u128::from(curr - prev);
                intervals_sum += d;
                intervals_sqsum += d * d;
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        // mean <= u32::MAX, so mean_sq and sq_mean are below 2^64 and
        // the variance fits u64 exactly.
        let mean = intervals_sum / count;
        // Variance = E[x²] − E[x]²
        let mean_sq = mean * mean;
        let sq_mean = intervals_sqsum / count;
        let variance = u64::try_from(sq_mean.saturating_sub(mean_sq))
            .expect("variance of u32 intervals is below 2^64");
        let std = isqrt_u64(variance);
        Some((mean as u32, std as u32))
    }
}

/// Integer square root (`no_std`-safe, no float). Only consumer is
/// the test-gated `isi_stats_us`.
#[cfg(test)]
fn isqrt_u64(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = u64::midpoint(x, n / x);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lif_neuron::LIFNeuron;
    use proptest::prelude::*;

    // ----- 2026-09-10: the ring replaced heapless::Vec. The oracle is the
    // pre-ring implementation kept verbatim: heapless::Vec with remove(0)
    // when full, and the three consumers' formulas over it. Same spike
    // times into both; iteration order and every consumer must agree. -----

    /// The pre-ring history and its consumers, as they were.
    struct HeaplessOracle {
        history: heapless::Vec<u32, MAX_SPIKE_HISTORY>,
    }

    // Verbatim, so its casts keep the allow lif_neuron.rs gave them.
    #[allow(clippy::cast_possible_truncation)]
    impl HeaplessOracle {
        fn new() -> Self {
            Self {
                history: heapless::Vec::new(),
            }
        }

        fn push(&mut self, t: u32) {
            if self.history.is_full() {
                self.history.remove(0);
            }
            let _ = self.history.push(t);
        }

        fn firing_rate_mhz(&self, last_update_time_us: u32, window_us: u32) -> u32 {
            if self.history.is_empty() || window_us == 0 {
                return 0;
            }
            let window_start = last_update_time_us.saturating_sub(window_us);
            let count = self.history.iter().filter(|&&t| t >= window_start).count() as u32;
            count
                .saturating_mul(1_000_000_000)
                .checked_div(window_us)
                .unwrap_or(0)
        }

        fn isi_stats_us(&self) -> Option<(u32, u32)> {
            if self.history.len() < 2 {
                return None;
            }
            let n = self.history.len();
            // Same u128 accumulators as the ring side (alpha.6): the
            // oracle is the pre-ring HISTORY kept verbatim; the
            // consumers' arithmetic moves on both sides together.
            let mut intervals_sum: u128 = 0;
            let mut intervals_sqsum: u128 = 0;
            let mut count: u128 = 0;
            for i in 1..n {
                let prev = self.history[i - 1];
                let curr = self.history[i];
                if curr >= prev {
                    let d = u128::from(curr - prev);
                    intervals_sum += d;
                    intervals_sqsum += d * d;
                    count += 1;
                }
            }
            if count == 0 {
                return None;
            }
            let mean = intervals_sum / count;
            let mean_sq = mean * mean;
            let sq_mean = intervals_sqsum / count;
            let variance = u64::try_from(sq_mean.saturating_sub(mean_sq))
                .expect("variance of u32 intervals is below 2^64");
            let std = isqrt_u64(variance);
            Some((mean as u32, std as u32))
        }
    }

    proptest! {
        /// Differential: the recorder against the heapless oracle on the
        /// same spike times — contents in order, then every consumer.
        /// Up to 300 spikes so the ring wraps several times; unordered
        /// times so non-monotonic sequences exercise the ISI filter.
        /// Times are arbitrary u32: the 2^27 bound this test carried on
        /// 2026-09-10 worked around the u64 overflow of `isi_stats_us`
        /// it had found, and the u128 accumulators (alpha.6) removed it.
        #[test]
        fn prop_ring_matches_the_heapless_oracle(
            times in proptest::collection::vec(any::<u32>(), 0..300),
            last_update_time_us in any::<u32>(),
            window_us in any::<u32>(),
        ) {
            let mut rec = SpikeRecorder::new();
            let mut oracle = HeaplessOracle::new();
            for &t in &times {
                rec.record(t);
                oracle.push(t);
            }
            let ring: std::vec::Vec<u32> = rec.iter().collect();
            let vec: std::vec::Vec<u32> = oracle.history.iter().copied().collect();
            prop_assert_eq!(ring, vec, "iteration order and contents");

            prop_assert_eq!(rec.len(), oracle.history.len());
            prop_assert_eq!(
                rec.firing_rate_mhz(last_update_time_us, window_us),
                oracle.firing_rate_mhz(last_update_time_us, window_us)
            );
            prop_assert_eq!(rec.isi_stats_us(), oracle.isi_stats_us());
        }

        /// The ring alone: after any number of records it holds the newest
        /// `min(records, MAX_SPIKE_HISTORY)` entries, oldest first, by
        /// iteration and by index; clear empties it and it works again.
        #[test]
        fn prop_ring_keeps_the_newest_in_order(
            times in proptest::collection::vec(any::<u32>(), 0..300),
        ) {
            let mut ring = SpikeRecorder::new();
            for &t in &times {
                ring.record(t);
            }
            let live = times.len().min(MAX_SPIKE_HISTORY);
            prop_assert_eq!(ring.len(), live);
            prop_assert_eq!(ring.is_empty(), live == 0);
            let expected = &times[times.len() - live..];
            let got: std::vec::Vec<u32> = ring.iter().collect();
            prop_assert_eq!(got.as_slice(), expected);
            for (i, &t) in expected.iter().enumerate() {
                prop_assert_eq!(ring.get(i), t);
            }

            ring.clear();
            prop_assert!(ring.is_empty());
            prop_assert_eq!(ring.iter().count(), 0);
            ring.record(7);
            prop_assert_eq!(ring.len(), 1);
            prop_assert_eq!(ring.get(0), 7);
        }

        /// A neuron's recorded history never exceeds MAX_SPIKE_HISTORY.
        #[test]
        fn prop_history_never_exceeds_capacity(id in 0u16..=10) {
            let mut n = LIFNeuron::new(id);
            let mut rec = SpikeRecorder::new();
            n.threshold = -100;
            for i in 0..1000_u32 {
                let t = i * 1000;
                if n.integrate_and_fire(1000, 1000, t) {
                    rec.record(t);
                }
            }
            prop_assert!(rec.len() <= MAX_SPIKE_HISTORY);
        }
    }

    #[test]
    fn isi_stats_none_with_fewer_than_two_spikes() {
        let rec = SpikeRecorder::new();
        assert!(rec.isi_stats_us().is_none());
    }

    #[test]
    fn isi_stats_computed_with_two_spikes() {
        let mut rec = SpikeRecorder::new();
        rec.record(1_000);
        rec.record(2_000); // ISI = 1000 μs
        let (mean, std) = rec.isi_stats_us().expect("two spikes → Some");
        assert_eq!(mean, 1000);
        assert_eq!(std, 0);
    }

    /// Three intervals of `u32::MAX` through the non-monotonic filter: in
    /// u64 the sum of their squares overflowed (three above 2^32/sqrt(3)
    /// suffice; the earlier note said "near 2^31", which is below the
    /// threshold). Both sides agree on the exact answer.
    #[test]
    fn isi_stats_survive_three_maximal_intervals() {
        let times = [0, u32::MAX, 0, u32::MAX, 0, u32::MAX];
        let mut rec = SpikeRecorder::new();
        let mut oracle = HeaplessOracle::new();
        for &t in &times {
            rec.record(t);
            oracle.push(t);
        }
        assert_eq!(rec.isi_stats_us(), Some((u32::MAX, 0)));
        assert_eq!(oracle.isi_stats_us(), Some((u32::MAX, 0)));
    }

    #[test]
    fn ring_debug_shows_live_entries_only() {
        let mut ring = SpikeRecorder::new();
        assert_eq!(std::format!("{ring:?}"), "[]");
        ring.record(1);
        ring.record(2);
        assert_eq!(std::format!("{ring:?}"), "[1, 2]");
        // Wrap: 70 records of 0..70 keep 6..70, and only those print.
        let mut ring = SpikeRecorder::new();
        for t in 0..70_u32 {
            ring.record(t);
        }
        let shown = std::format!("{ring:?}");
        assert!(shown.starts_with("[6, 7, 8,"), "{shown}");
        assert!(shown.ends_with(" 69]"), "{shown}");
        assert_eq!(ring.iter().count(), MAX_SPIKE_HISTORY);
    }

    #[test]
    #[should_panic(expected = "SpikeRecorder index 2 out of range 2")]
    fn ring_get_past_len_panics_like_a_slice() {
        let mut ring = SpikeRecorder::new();
        ring.record(10);
        ring.record(20);
        let _ = ring.get(2);
    }
}
