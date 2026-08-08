//! Framework-agnostic SNN sim-runner core for the `NeuralOS` visualizer.
//!
//! Holds a [`SpikingNeuralNetwork`] and a rolling spike raster (ring buffer).
//! UI-agnostic: the Slint bin calls [`SimRunner::tick`] each frame and renders
//! the raster bytes this returns. Pure compute — no Slint, no I/O.
//!
//! This is the keystone seam: `neuralos-app` finally runs `neuralos-snn` at
//! runtime. The library is the floor; the app is its microscope.

#![warn(missing_debug_implementations)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss // stats counters are tiny; usize/i64 → f64 is lossless in practice
)]

use neuralos_snn::{NetworkTopology, NeuronType, SpikingNeuralNetwork};

/// Raster columns = visible timesteps. Newest spike column appears on the right.
/// At ~60 ticks/s this shows ~3.3 s of recent activity.
pub const RASTER_COLS: usize = 200;

const BG: [u8; 4] = [0x12, 0x12, 0x1F, 0xFF]; // background (dark indigo)
const E_SPIKE: [u8; 4] = [0xFF, 0x99, 0x33, 0xFF]; // excitatory (warm orange)
const I_SPIKE: [u8; 4] = [0x33, 0xCC, 0xFF, 0xFF]; // inhibitory (cool cyan)

/// Weight-matrix background (slightly distinct from the raster bg so the two
/// panes read as separate views).
const WM_BG: [u8; 4] = [0x0A, 0x0A, 0x14, 0xFF];
/// Weight magnitude at which a synapse reaches full color saturation.
const W_CAP: u32 = 200;

/// Map a synapse weight to a color: excitatory (>0) → orange scale, inhibitory
/// (<0) → cyan scale, brightness ∝ |weight|. Zero → background.
#[must_use]
fn weight_color(w: i16) -> [u8; 4] {
    let w = i32::from(w);
    let mag = w.unsigned_abs().min(W_CAP); // 0..=200
                                           // (r, g, b) base for the sign; brightness scales by mag/W_CAP.
    let (base_r, base_g, base_b) = match w.cmp(&0) {
        core::cmp::Ordering::Greater => (0xFF_u32, 0x99_u32, 0x33_u32), // orange
        core::cmp::Ordering::Less => (0x33_u32, 0xCC_u32, 0xFF_u32),    // cyan
        core::cmp::Ordering::Equal => return WM_BG,
    };
    [
        (base_r * mag / W_CAP) as u8,
        (base_g * mag / W_CAP) as u8,
        (base_b * mag / W_CAP) as u8,
        0xFF,
    ]
}

/// SNN sim-runner + rolling spike raster. The visualizer's model.
#[derive(Debug)]
pub struct SimRunner {
    net: SpikingNeuralNetwork,
    /// Per-neuron spike color (E or I), precomputed once at construction.
    neuron_color: Vec<[u8; 4]>,
    /// Reused input-current buffer (one μA value per neuron), avoiding per-tick alloc.
    inputs: Vec<i16>,
    /// Ring buffer: `RASTER_COLS` columns × `rows` rows, RGBA bytes, row-major.
    /// Column `head` is the oldest (next to be overwritten); `(head-1)` is newest.
    ring: Vec<u8>,
    head: usize,
    rows: usize,
    /// Display buffer (ring reordered to linear, newest-right). Reused each frame.
    display: Vec<u8>,
    /// N×N synaptic weight matrix, RGBA bytes, row-major (row=pre, col=post).
    /// Rebuilt each frame from the network's synapses — STDP changes show live.
    weight_matrix: Vec<u8>,
    /// Snapshot of synapse weights right after topology build, for "% changed".
    /// Same ordering as `net.synapses()` (STDP never adds/removes synapses).
    initial_weights: Vec<i16>,
}

impl SimRunner {
    /// Build a balanced E/I (80/20) network of `neuron_count` neurons, wire its
    /// topology, and init the raster to the background color.
    ///
    /// # Errors
    /// Propagates [`neuralos_snn::Error`] if the network cannot be constructed
    /// or the topology fails to build (e.g. `neuron_count == 0`).
    pub fn new(neuron_count: u16, time_step_us: u32) -> neuralos_snn::Result<Self> {
        let mut net =
            SpikingNeuralNetwork::new(neuron_count, time_step_us, NetworkTopology::default())?;
        net.build_topology()?;
        let rows = usize::from(neuron_count);

        let neuron_color = net
            .neurons()
            .iter()
            .map(|n| match n.neuron_type {
                NeuronType::Excitatory => E_SPIKE,
                NeuronType::Inhibitory => I_SPIKE,
            })
            .collect();

        let cell_count = RASTER_COLS * rows;
        let mut ring = vec![0u8; cell_count * 4];
        for cell in ring.chunks_exact_mut(4) {
            cell.copy_from_slice(&BG);
        }

        // Snapshot initial weights for the "% synapses changed" stat.
        let initial_weights: Vec<i16> = net.synapses().iter().map(|s| s.weight).collect();
        let wm_cells = rows * rows;
        let mut weight_matrix = vec![0u8; wm_cells * 4];
        for cell in weight_matrix.chunks_exact_mut(4) {
            cell.copy_from_slice(&WM_BG);
        }

        let mut runner = Self {
            net,
            neuron_color,
            inputs: vec![0; rows],
            ring,
            head: 0,
            rows,
            display: vec![0u8; cell_count * 4],
            weight_matrix,
            initial_weights,
        };

        // Sustained-firing default: plasticity OFF so STDP can't quench the
        // network (it depresses E weights without bound → silence). The user
        // toggles learning ON to watch STDP work — including watching it quiet
        // the network, which is honest neuroscience.
        runner.set_learning(false);

        // Pre-fill the raster with ~150 warmup ticks so the window opens already
        // dense with activity, not a black pane that slowly fills from the right.
        for _ in 0..150 {
            runner.tick(600);
        }

        Ok(runner)
    }

    /// Toggle STDP learning. Off (default) = fixed weights, sustained firing.
    /// On = weights drift under STDP; watch the heatmap shift and the network
    /// eventually quiet as inhibition wins.
    pub fn set_learning(&mut self, on: bool) {
        self.net.set_plasticity_enabled(on);
    }

    /// Is STDP learning currently enabled?
    #[must_use]
    pub fn learning(&self) -> bool {
        self.net.plasticity_enabled()
    }

    /// Advance the sim one step with a uniform input drive (μA). Writes the
    /// resulting spike column into the raster ring at `head` and advances head.
    ///
    /// `step()` only errors on index/param violations that cannot occur in
    /// steady-state stepping (correctly-sized inputs, valid indices); any such
    /// error is swallowed as a no-spike frame rather than panicking the UI.
    pub fn tick(&mut self, input_drive_ua: i16) {
        for slot in &mut self.inputs {
            *slot = input_drive_ua;
        }
        let spikes = self.net.step(&self.inputs).unwrap_or_default();

        // Overwrite the oldest column with the new frame: background first,
        // then light up any neurons that spiked this tick.
        for r in 0..self.rows {
            let off = (r * RASTER_COLS + self.head) * 4;
            self.ring[off..off + 4].copy_from_slice(&BG);
        }
        for s in &spikes {
            let r = usize::from(s.neuron_id);
            if r < self.rows {
                let off = (r * RASTER_COLS + self.head) * 4;
                self.ring[off..off + 4].copy_from_slice(&self.neuron_color[r]);
            }
        }
        self.head = (self.head + 1) % RASTER_COLS;
    }

    /// Reorder the ring into a linear display buffer (oldest left, newest right)
    /// and return it as raw RGBA bytes with dims `(RASTER_COLS, rows)`.
    /// Reuses the internal display buffer to avoid per-frame allocation.
    #[must_use]
    pub fn raster_display(&mut self) -> (usize, usize, &[u8]) {
        let row_bytes = RASTER_COLS * 4;
        let head_bytes = self.head * 4;
        for r in 0..self.rows {
            let off = r * row_bytes;
            self.display[off..off + row_bytes].copy_from_slice(&self.ring[off..off + row_bytes]);
            // ring row column order is [0..COLS]; display wants [head..COLS, 0..head]
            // (oldest at head → newest at head-1), which is a left-rotation by `head`.
            self.display[off..off + row_bytes].rotate_left(head_bytes);
        }
        (RASTER_COLS, self.rows, &self.display)
    }

    /// Rebuild the N×N synaptic weight matrix from current synapse weights and
    /// return it as raw RGBA bytes with dims `(rows, rows)`. Row = presynaptic,
    /// col = postsynaptic. STDP weight drift shows up as live color shifts.
    /// Reuses the internal buffer to avoid per-frame allocation.
    #[must_use]
    pub fn weight_matrix_display(&mut self) -> (usize, usize, &[u8]) {
        let n = self.rows;
        for cell in self.weight_matrix.chunks_exact_mut(4) {
            cell.copy_from_slice(&WM_BG);
        }
        for s in self.net.synapses() {
            let pre = usize::from(s.pre_neuron_id);
            let post = usize::from(s.post_neuron_id);
            if pre < n && post < n {
                let off = (pre * n + post) * 4;
                self.weight_matrix[off..off + 4].copy_from_slice(&weight_color(s.weight));
            }
        }
        (n, n, &self.weight_matrix)
    }

    /// One-line human-readable stats string for the UI status bar.
    #[must_use]
    pub fn stats_text(&self) -> String {
        let s = self.net.stats();
        let sim_ms = f64::from(self.net.current_time_us()) / 1000.0;

        // STDP learning readouts: mean weight + % of synapses changed since start.
        let syns = self.net.synapses();
        let (mean_w, pct_changed) = if syns.is_empty() {
            (0.0_f64, 0.0_f64)
        } else {
            let sum: i64 = syns.iter().map(|s| i64::from(s.weight)).sum();
            let mean = sum as f64 / syns.len() as f64;
            let changed = syns
                .iter()
                .zip(self.initial_weights.iter())
                .filter(|(s, init)| s.weight != **init)
                .count();
            (mean, changed as f64 * 100.0 / syns.len() as f64)
        };

        let stats = format!(
            "{:.1} Hz  ·  {} spikes  ·  {} plasticity events  ·  {:.0} ms sim  ·  \
             mean w {:.0}  ·  {:.0}% synapses changed  ·  {} neurons / {} synapses",
            s.firing_rate_hz,
            s.total_spikes,
            s.plasticity_events,
            sim_ms,
            mean_w,
            pct_changed,
            self.net.neuron_count(),
            self.net.synapse_count(),
        );
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_constructs_and_steps() {
        let mut runner = SimRunner::new(80, 1000).expect("80 neurons valid");
        let (w, h, _) = runner.raster_display();
        assert_eq!(w, RASTER_COLS);
        assert_eq!(h, 80);
        // Strong drive over many ticks must produce some lit pixels.
        for _ in 0..200 {
            runner.tick(800);
        }
        let (_, _, bytes) = runner.raster_display();
        let lit = bytes.chunks_exact(4).any(|px| px != BG);
        assert!(lit, "sustained strong drive should light some raster cells");
    }

    #[test]
    fn display_is_raster_dims() {
        let mut runner = SimRunner::new(64, 1000).unwrap();
        let (w, h, bytes) = runner.raster_display();
        assert_eq!(w * h * 4, bytes.len());
    }

    #[test]
    fn weight_matrix_has_both_signs() {
        // The balanced topology wires both excitatory (w>0, orange) and
        // inhibitory (w<0, cyan) synapses — the heatmap must reflect both.
        let mut r = SimRunner::new(128, 1000).unwrap();
        let (w, h, bytes) = r.weight_matrix_display();
        assert_eq!((w, h), (128, 128));
        let mut warm = 0_usize; // R > B (excitatory lineage)
        let mut cool = 0_usize; // B > R (inhibitory lineage)
        for px in bytes.chunks_exact(4) {
            if px[0] > px[2] {
                warm += 1;
            } else if px[2] > px[0] {
                cool += 1;
            }
        }
        assert!(warm > 0, "no excitatory cells in the weight matrix");
        assert!(cool > 0, "no inhibitory cells in the weight matrix");
    }
}
