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
    clippy::cast_sign_loss
)]

use neuralos_snn::{NetworkTopology, NeuronType, SpikingNeuralNetwork};

/// Raster columns = visible timesteps. Newest spike column appears on the right.
/// At ~60 ticks/s this shows ~3.3 s of recent activity.
pub const RASTER_COLS: usize = 200;

const BG: [u8; 4] = [0x12, 0x12, 0x1F, 0xFF]; // background (dark indigo)
const E_SPIKE: [u8; 4] = [0xFF, 0x99, 0x33, 0xFF]; // excitatory (warm orange)
const I_SPIKE: [u8; 4] = [0x33, 0xCC, 0xFF, 0xFF]; // inhibitory (cool cyan)

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
}

impl SimRunner {
    /// Build a balanced E/I (80/20) network of `neuron_count` neurons, wire its
    /// topology, and init the raster to the background color.
    ///
    /// # Errors
    /// Propagates [`neuralos_snn::Error`] if the network cannot be constructed
    /// or the topology fails to build (e.g. `neuron_count == 0`).
    pub fn new(neuron_count: u16, time_step_us: u32) -> neuralos_snn::Result<Self> {
        let mut net = SpikingNeuralNetwork::new(neuron_count, time_step_us, NetworkTopology::default())?;
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

        Ok(Self {
            net,
            neuron_color,
            inputs: vec![0; rows],
            ring,
            head: 0,
            rows,
            display: vec![0u8; cell_count * 4],
        })
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

    /// One-line human-readable stats string for the UI status bar.
    #[must_use]
    pub fn stats_text(&self) -> String {
        let s = self.net.stats();
        let sim_ms = f64::from(self.net.current_time_us()) / 1000.0;
        format!(
            "{:.1} Hz  ·  {} spikes  ·  {} plasticity events  ·  {:.0} ms sim  ·  {} neurons / {} synapses",
            s.firing_rate_hz, s.total_spikes, s.plasticity_events, sim_ms,
            self.net.neuron_count(), self.net.synapse_count(),
        )
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
}
