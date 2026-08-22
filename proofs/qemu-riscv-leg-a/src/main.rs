#![no_std]
#![no_main]

use core::arch::global_asm;
use core::fmt::Write as _;
use core::panic::PanicInfo;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicU32, Ordering};

use neuralos_snn::bridge::{
    decode_i2_s, decode_q1_0, decode_q2_0, encode_i2_s, encode_q2_0, half_to_f32_bits,
    half_to_milli, q2_0_encoded_len, repack_i2s_to_kernel, wire_gamma_to_substrate, I2_S_BLOCK,
    I2_S_TAIL_BYTES, Q1_0_BLOCK, Q2_0_BLOCK,
};
use neuralos_snn::kernel::{absmax_normalize_q15, pack_trits, ternary_matvec, unpack_trit};
use neuralos_snn::lif_neuron::{LIFNeuron, NeuronType, VoltageResolution};
use neuralos_snn::nir::{quantize_lif, quantize_linear, NirError, NirImportOptions, NirLinear};
use neuralos_snn::synapse::{Synapse, SynapseType, STDPRule, SCALE};
use neuralos_snn::trit::{project_to_ternary, stochastic_ternary_flip, tensor_scale, Trit};
use neuralos_snn::Error;

const UART_BASE: usize = 0x1000_0000;
const SIFIVE_TEST: usize = 0x0010_0000;

global_asm!(
    ".section .text._start, \"ax\"",
    ".global _start",
    "_start:",
    "csrw mie, zero",
    "csrw mip, zero",
    "fence iorw, iorw",
    "li t0, 0x2000",
    "csrs mstatus, t0",
    ".option push",
    ".option norelax",
    "la gp, __global_pointer$",
    ".option pop",
    "la sp, _stack_top",
    "la t0, _bss_start",
    "la t1, _bss_end",
    "1:",
    "bgeu t0, t1, 2f",
    "sd zero, 0(t0)",
    "addi t0, t0, 8",
    "j 1b",
    "2:",
    "call rust_main",
    "3:",
    "wfi",
    "j 3b",
);

fn uart_putc(c: u8) {
    unsafe {
        while read_volatile((UART_BASE + 5) as *const u8) & 0x20 == 0 {}
        write_volatile(UART_BASE as *mut u8, c);
    }
}

fn poweroff(code: u32) -> ! {
    unsafe { write_volatile(SIFIVE_TEST as *mut u32, code) };
    loop {
        core::hint::spin_loop();
    }
}

fn pass() -> ! {
    poweroff(0x5555)
}

fn fail() -> ! {
    poweroff(0x3333 | (1 << 16))
}

struct Uart;

impl core::fmt::Write for Uart {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            uart_putc(b);
        }
        Ok(())
    }
}

static CHECKS: AtomicU32 = AtomicU32::new(0);
static FAILED: AtomicU32 = AtomicU32::new(0);

fn next_check_id() -> u32 {
    CHECKS.fetch_add(1, Ordering::Relaxed) + 1
}

fn ck_s<T: PartialEq + core::fmt::Debug>(name: &str, suffix: &str, got: T, want: T) {
    let id = next_check_id();
    if got == want {
        let _ = writeln!(Uart, "ok [{id:3}] {name} {suffix}");
    } else {
        FAILED.fetch_add(1, Ordering::Relaxed);
        let _ = writeln!(
            Uart,
            "FAIL [{id:3}] {name} {suffix}  got={got:?} want={want:?}"
        );
    }
}

fn ck<T: PartialEq + core::fmt::Debug>(name: &str, got: T, want: T) {
    ck_s(name, "", got, want);
}

fn ck_bool_s(name: &str, suffix: &str, ok: bool) {
    let id = next_check_id();
    if ok {
        let _ = writeln!(Uart, "ok [{id:3}] {name} {suffix}");
    } else {
        FAILED.fetch_add(1, Ordering::Relaxed);
        let _ = writeln!(Uart, "FAIL [{id:3}] {name} {suffix}");
    }
}

fn ck_bool(name: &str, ok: bool) {
    ck_bool_s(name, "", ok);
}

fn ck_f64_eps(name: &str, got: f64, want: f64, eps: f64) {
    let id = next_check_id();
    if (got - want).abs() < eps {
        let _ = writeln!(Uart, "ok [{id:3}] {name}");
    } else {
        FAILED.fetch_add(1, Ordering::Relaxed);
        let _ = writeln!(Uart, "FAIL [{id:3}] {name}  got={got:?} want={want:?}");
    }
}

fn ck_slice<T: PartialEq + core::fmt::Debug>(name: &str, got: &[T], want: &[T]) {
    ck(name, got, want);
}

fn quiet_neuron(id: u16, r: VoltageResolution) -> LIFNeuron {
    let mut n = LIFNeuron::new_with_type_resolution(id, NeuronType::Excitatory, r);
    n.noise_amplitude_ua = 0;
    n
}

fn lin_sentinel() -> NirLinear {
    NirLinear {
        rows: 0,
        cols: 0,
        weight_offset: 0,
        scale: 0.0,
        absmax: 0.0,
        max_abs_err: f64::NAN,
        zero_tensor: false,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_main() -> ! {
    let _ = writeln!(
        Uart,
        "neuralos-snn Leg A: bare-metal riscv64gc-unknown-none-elf, QEMU virt, no allocator"
    );
    let _ = writeln!(Uart, "cited asserts: values mirrored from crates/neuralos-snn unit tests");
    lif_checks();
    synapse_checks();
    bridge_checks();
    trit_checks();
    kernel_checks();
    nir_checks();
    spike_raster();
    let checks = CHECKS.load(Ordering::Relaxed);
    let failed = FAILED.load(Ordering::Relaxed);
    let _ = writeln!(Uart, "---");
    let _ = writeln!(Uart, "checks: {checks}  failed: {failed}");
    if failed == 0 {
        let _ = writeln!(Uart, "LEG A PASS");
        pass();
    } else {
        let _ = writeln!(Uart, "LEG A FAIL");
        fail();
    }
}

const TRACE_TAGS: [&str; 6] = [
    "step1=-67", "step2=-65", "step3=-63", "step4=-61", "step5=-59", "step6=-57",
];

fn lif_checks() {
    let name = "lif::millivolt_trace_is_pinned_to_the_historical_arithmetic";
    let mut n = quiet_neuron(11, VoltageResolution::Millivolt);
    for (k, &want) in [-67, -65, -63, -61, -59, -57].iter().enumerate() {
        let spiked = n.integrate_and_fire(600, 1000, 0);
        ck_bool_s(name, TRACE_TAGS[k], !spiked);
        cks_or(name, TRACE_TAGS[k], n.membrane_potential, want);
    }
    let spiked = n.integrate_and_fire(600, 1000, 0);
    ck_bool("lif::millivolt_trace step7 spikes", spiked);
    ck(
        "lif::millivolt_trace step7 reset",
        n.membrane_potential,
        -80,
    );

    let mut mv = quiet_neuron(12, VoltageResolution::Millivolt);
    mv.integrate_and_fire(12, 1000, 0);
    ck(
        "lif::the_dead_zone_12ua_pair mV",
        mv.membrane_potential,
        -70,
    );
    let mut cmv = quiet_neuron(13, VoltageResolution::CentiMillivolt);
    cmv.integrate_and_fire(12, 1000, 0);
    ck(
        "lif::the_dead_zone_12ua_pair centi",
        cmv.membrane_potential,
        -7_000 + 6,
    );

    let mut n = quiet_neuron(23, VoltageResolution::Millivolt);
    n.membrane_potential = -90;
    let _ = n.integrate_and_fire(0, 20_000, 20_000);
    ck(
        "lif::leak_convergence from below",
        n.membrane_potential,
        n.resting_potential,
    );
    n.membrane_potential = -20;
    let _ = n.integrate_and_fire(0, 20_000, 40_000);
    ck(
        "lif::leak_convergence from above",
        n.membrane_potential,
        n.resting_potential,
    );
    let _ = n.integrate_and_fire(0, 20_000, 60_000);
    ck(
        "lif::leak_convergence stays at rest",
        n.membrane_potential,
        n.resting_potential,
    );

    let mut n = quiet_neuron(21, VoltageResolution::Millivolt);
    n.adaptation_current_ua = 3;
    n.decay_adaptation_current();
    ck(
        "lif::adaptation_decay 3->2",
        n.adaptation_current_ua,
        2,
    );
    n.decay_adaptation_current();
    ck("lif::adaptation_decay 2->1", n.adaptation_current_ua, 1);
    n.decay_adaptation_current();
    ck("lif::adaptation_decay 1->0", n.adaptation_current_ua, 0);
    n.decay_adaptation_current();
    ck(
        "lif::adaptation_decay floor 0",
        n.adaptation_current_ua,
        0,
    );

    let mut n = quiet_neuron(22, VoltageResolution::Millivolt);
    n.threshold = -100;
    let spiked = n.integrate_and_fire(1000, 1000, 0);
    ck_bool("lif::spike_adds_two_quanta spiked", spiked);
    ck(
        "lif::spike_adds_two_quanta +2",
        n.adaptation_current_ua,
        2,
    );

    let n = quiet_neuron(16, VoltageResolution::CentiMillivolt);
    ck("lif::centi_mode membrane", n.membrane_potential, -7_000);
    ck("lif::centi_mode resting", n.resting_potential, -7_000);
    ck("lif::centi_mode threshold", n.threshold, -5_500);
    ck("lif::centi_mode reset", n.reset_potential, -8_000);
    let mut m = quiet_neuron(17, VoltageResolution::CentiMillivolt);
    m.membrane_potential = -9_999;
    m.integrate_and_fire(-30_000, 1000, 0);
    ck_bool(
        "lif::centi_mode clamp at scaled floor",
        m.membrane_potential >= -10_000,
    );

    let mut n = quiet_neuron(18, VoltageResolution::Millivolt);
    for t in 0..50 {
        n.integrate_and_fire(300, 1000, t);
    }
    ck(
        "lif::rectification sticks at -59",
        n.membrane_potential,
        -59,
    );
    n.integrate_and_fire(300, 1000, 100);
    ck(
        "lif::rectification stays stuck",
        n.membrane_potential,
        -59,
    );
    n.add_synaptic_current(12);
    n.integrate_and_fire(300, 1000, 101);
    ck(
        "lif::rectification +12ua ratchets",
        n.membrane_potential,
        -58,
    );
    n.add_synaptic_current(-12);
    n.integrate_and_fire(300, 1000, 102);
    ck(
        "lif::rectification -12ua absorbed",
        n.membrane_potential,
        -58,
    );
}

fn cks_or<T: PartialEq + core::fmt::Debug>(name: &str, suffix: &str, got: T, want: T) {
    ck_s(name, suffix, got, want);
}

fn synapse_checks() {
    let s = Synapse::new(1, 2, 100).unwrap_or_else(|_| Synapse::new(1, 2, 0).unwrap());
    ck("synapse::default_params pre", s.pre_neuron_id, 1);
    ck("synapse::default_params post", s.post_neuron_id, 2);
    ck("synapse::default_params weight", s.weight, 100);
    ck(
        "synapse::default_params type",
        s.synapse_type,
        SynapseType::Excitatory,
    );
    ck("synapse::default_params tau_rise", s.tau_rise_us, 500);
    ck("synapse::default_params tau_decay", s.tau_decay_us, 5_000);
    ck("synapse::default_params max", s.max_weight, 2000);
    ck("synapse::default_params min", s.min_weight, 0);

    let s = Synapse::new(1, 2, -100).unwrap_or_else(|_| Synapse::new(1, 2, 0).unwrap());
    ck(
        "synapse::negative_weight type",
        s.synapse_type,
        SynapseType::Inhibitory,
    );
    ck("synapse::negative_weight tau_rise", s.tau_rise_us, 300);
    ck(
        "synapse::negative_weight tau_decay",
        s.tau_decay_us,
        10_000,
    );
    ck("synapse::negative_weight max", s.max_weight, 0);
    ck("synapse::negative_weight min", s.min_weight, -2000);

    ck_bool(
        "synapse::self_connection_rejected",
        matches!(Synapse::new(5, 5, 100), Err(Error::InvalidParameter)),
    );

    let rule = STDPRule::new();
    let delta = rule.calculate_weight_change(0);
    let expected =
        (i32::from(rule.a_plus) * SCALE * i32::from(rule.learning_rate)) / (SCALE * SCALE);
    ck("synapse::stdp_zero_at_zero_dt", delta, expected as i16);

    ck(
        "synapse::stdp_zero_outside_window dt=-200000",
        rule.calculate_weight_change(-200_000),
        0,
    );
    ck(
        "synapse::stdp_zero_outside_window dt=+200000",
        rule.calculate_weight_change(200_000),
        0,
    );

    let mut s = Synapse::new(1, 2, 100).unwrap_or_else(|_| Synapse::new(1, 2, 0).unwrap());
    s.update_weight(10_000);
    ck(
        "synapse::weight_clamped_at_bounds max",
        s.weight,
        s.max_weight,
    );
    s.update_weight(-10_000);
    ck(
        "synapse::weight_clamped_at_bounds min",
        s.weight,
        s.min_weight,
    );
}

const REAL_Q2_0_FIRST_BLOCK: [u8; 34] = [
    0xC8, 0x24, 0x14, 0x44, 0x45, 0x1A, 0x18, 0x68, 0x68, 0x61, 0x8A, 0xA8, 0x91, 0x66, 0x45,
    0x42, 0x91, 0x80, 0x11, 0x62, 0x18, 0x11, 0x29, 0x48, 0x61, 0x00, 0x1A, 0x94, 0x81, 0x24,
    0x54, 0x0A, 0x86, 0x84,
];

fn bridge_checks() {
    let pat = [
        Trit::One,
        Trit::MinusOne,
        Trit::Zero,
        Trit::One,
        Trit::Zero,
        Trit::Zero,
        Trit::MinusOne,
        Trit::One,
    ];
    let mut trits = [Trit::Zero; I2_S_BLOCK];
    for (i, slot) in trits.iter_mut().enumerate() {
        *slot = pat[i % 8];
    }
    let scale_bits: u32 = 0x4000_0000;
    let mut buf = [0xFF_u8; 128];
    let written = encode_i2_s(&trits, scale_bits, &mut buf).unwrap_or(0);
    ck(
        "bridge::i2_s_known_vector written",
        written,
        32 + I2_S_TAIL_BYTES,
    );
    let mut expected = [0_u8; 64];
    for k in 0..4 {
        expected[k * 8..k * 8 + 8]
            .copy_from_slice(&[0xAA, 0x00, 0x55, 0xAA, 0x55, 0x55, 0x00, 0xAA]);
    }
    expected[32..36].copy_from_slice(&scale_bits.to_le_bytes());
    ck_slice(
        "bridge::i2_s_known_vector bytes",
        &buf[..written],
        &expected,
    );
    let mut back = [Trit::Zero; I2_S_BLOCK];
    let got_scale = decode_i2_s(&buf, &mut back).unwrap_or(0);
    ck(
        "bridge::i2_s_known_vector decode scale",
        got_scale,
        scale_bits,
    );
    ck_slice(
        "bridge::i2_s_known_vector decode trits",
        &back,
        &trits,
    );

    let pat = [
        Trit::One,
        Trit::MinusOne,
        Trit::One,
        Trit::MinusOne,
        Trit::One,
        Trit::One,
        Trit::MinusOne,
        Trit::One,
    ];
    let mut bytes = [0_u8; 18];
    bytes[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    for slot in &mut bytes[2..] {
        *slot = 0xB5;
    }
    let mut trits = [Trit::Zero; Q1_0_BLOCK];
    let mut scales = [0_u16; 1];
    ck_bool(
        "bridge::q1_0_known_vector decodes",
        decode_q1_0(&bytes, &mut trits, &mut scales).is_ok(),
    );
    ck("bridge::q1_0_known_vector scale", scales[0], 0x3C00);
    let mut want = [Trit::Zero; Q1_0_BLOCK];
    for (i, slot) in want.iter_mut().enumerate() {
        *slot = pat[i % 8];
    }
    ck_slice("bridge::q1_0_known_vector trits", &trits, &want);

    let pat = [Trit::MinusOne, Trit::Zero, Trit::One, Trit::One];
    let mut bytes = [0_u8; 34];
    bytes[0..2].copy_from_slice(&0x4400_u16.to_le_bytes());
    for slot in &mut bytes[2..] {
        *slot = 0xA4;
    }
    let mut trits = [Trit::Zero; Q2_0_BLOCK];
    let mut scales = [0_u16; 1];
    ck_bool(
        "bridge::q2_0_known_vector decodes",
        decode_q2_0(&bytes, &mut trits, &mut scales).is_ok(),
    );
    ck("bridge::q2_0_known_vector scale", scales[0], 0x4400);
    let mut want = [Trit::Zero; Q2_0_BLOCK];
    for (i, slot) in want.iter_mut().enumerate() {
        *slot = pat[i % 4];
    }
    ck_slice("bridge::q2_0_known_vector trits", &trits, &want);

    ck("bridge::q2_0_block_geometry Q2_0_BLOCK", Q2_0_BLOCK, 128);
    ck(
        "bridge::q2_0_block_geometry len(128)",
        q2_0_encoded_len(128),
        34,
    );
    ck(
        "bridge::q2_0_block_geometry len(2560)",
        q2_0_encoded_len(2560),
        680,
    );
    ck(
        "bridge::q2_0_block_geometry len(384)",
        q2_0_encoded_len(384),
        102,
    );

    let mut trits = [Trit::Zero; Q2_0_BLOCK];
    let mut scales = [0_u16; 1];
    ck_bool(
        "bridge::q2_0_real_artifact decodes",
        decode_q2_0(&REAL_Q2_0_FIRST_BLOCK, &mut trits, &mut scales).is_ok(),
    );
    ck("bridge::q2_0_real_artifact scale", scales[0], 0x24C8);
    let census = trits.iter().fold((0_i32, 0_i32, 0_i32), |(p, z, m), t| match t {
        Trit::One => (p + 1, z, m),
        Trit::Zero => (p, z + 1, m),
        Trit::MinusOne => (p, z, m + 1),
    });
    ck("bridge::q2_0_real_artifact census", census, (37, 43, 48));
    let mut back = [0_u8; 34];
    ck_bool(
        "bridge::q2_0_real_artifact encodes",
        encode_q2_0(&trits, &scales, &mut back).is_ok(),
    );
    ck_slice(
        "bridge::q2_0_real_artifact byte identity",
        &back,
        &REAL_Q2_0_FIRST_BLOCK,
    );

    let half_cases: [(u16, u32, i32, &str); 12] = [
        (0x0000, 0x0000_0000, 0, "+0"),
        (0x8000, 0x8000_0000, 0, "-0"),
        (0x3C00, 0x3F80_0000, 1000, "1.0"),
        (0x3800, 0x3F00_0000, 500, "0.5"),
        (0xC000, 0xC000_0000, -2000, "-2.0"),
        (0x4400, 0x4080_0000, 4000, "4.0"),
        (0x7BFF, 0x477F_E000, 65_504_000, "max-finite"),
        (0x03FF, 0x387F_C000, 0, "max-subnormal"),
        (0x0400, 0x3880_0000, 0, "min-normal"),
        (0x7C00, 0x7F80_0000, i32::MAX, "+inf"),
        (0xFC00, 0xFF80_0000, i32::MIN, "-inf"),
        (0x7E00, 0x7FC0_0000, 0, "NaN"),
    ];
    for &(h, f32_bits, milli, tag) in &half_cases {
        ck_s(
            "bridge::half_known_vectors f32",
            tag,
            half_to_f32_bits(h),
            f32_bits,
        );
        ck_s(
            "bridge::half_known_vectors milli",
            tag,
            half_to_milli(h),
            milli,
        );
    }

    ck("bridge::wire_gamma 24", wire_gamma_to_substrate(24), 24);
    ck("bridge::wire_gamma 0", wire_gamma_to_substrate(0), 0);
    ck("bridge::wire_gamma 125", wire_gamma_to_substrate(125), 125);
    ck(
        "bridge::wire_gamma fp16-max",
        wire_gamma_to_substrate(65_504_000),
        i16::MAX,
    );
    ck(
        "bridge::wire_gamma -40000",
        wire_gamma_to_substrate(-40_000),
        i16::MIN,
    );
    ck(
        "bridge::wire_gamma -40000000",
        wire_gamma_to_substrate(-40_000_000),
        i16::MIN,
    );
    ck("bridge::wire_gamma -5", wire_gamma_to_substrate(-5), -5);
    ck("bridge::scale_constant_is_pinned", SCALE, 1000);

    let pat = [
        Trit::One,
        Trit::MinusOne,
        Trit::Zero,
        Trit::One,
        Trit::Zero,
        Trit::Zero,
        Trit::One,
        Trit::MinusOne,
    ];
    let mut trits = [Trit::Zero; I2_S_BLOCK];
    for (i, slot) in trits.iter_mut().enumerate() {
        *slot = pat[i % 8];
    }
    let mut wire = [0_u8; 64];
    ck_bool(
        "bridge::repack encodes",
        encode_i2_s(&trits, 0x4000_0000, &mut wire).is_ok(),
    );
    let mut kernel = [0_u8; 32];
    ck_bool(
        "bridge::repack repacks",
        repack_i2s_to_kernel(&wire, I2_S_BLOCK, &mut kernel).is_ok(),
    );
    let mut all_ok = true;
    for (i, &t) in trits.iter().enumerate() {
        if unpack_trit(&kernel, i) != Ok(t) {
            all_ok = false;
        }
    }
    ck_bool("bridge::repack_known_vector order", all_ok);
}

fn trit_checks() {
    let g = 125_i16;
    ck(
        "trit::from_weight_boundaries 62",
        Trit::from_weight(62, g),
        Trit::Zero,
    );
    ck(
        "trit::from_weight_boundaries 63",
        Trit::from_weight(63, g),
        Trit::One,
    );
    ck(
        "trit::from_weight_boundaries -62",
        Trit::from_weight(-62, g),
        Trit::Zero,
    );
    ck(
        "trit::from_weight_boundaries -63",
        Trit::from_weight(-63, g),
        Trit::MinusOne,
    );
    ck(
        "trit::from_weight_boundaries 125",
        Trit::from_weight(125, g),
        Trit::One,
    );
    ck(
        "trit::from_weight_boundaries -125",
        Trit::from_weight(-125, g),
        Trit::MinusOne,
    );
    ck(
        "trit::from_weight_boundaries 0",
        Trit::from_weight(0, g),
        Trit::Zero,
    );

    ck("trit::odd_gamma 2@5", Trit::from_weight(2, 5), Trit::Zero);
    ck("trit::odd_gamma 3@5", Trit::from_weight(3, 5), Trit::One);

    ck("trit::tensor_scale [10,15]", tensor_scale(&[10, 15]), 13);
    ck("trit::tensor_scale [10,14]", tensor_scale(&[10, 14]), 12);

    ck("trit::project 130", project_to_ternary(130, 125), 125);
    ck("trit::project 60", project_to_ternary(60, 125), 0);
    ck("trit::project -200", project_to_ternary(-200, 125), -125);

    let g = 125_i16;
    ck(
        "trit::flip draw0 -g ltp",
        stochastic_ternary_flip(-g, g, 1, 0),
        0,
    );
    ck(
        "trit::flip draw0 0 ltp",
        stochastic_ternary_flip(0, g, 1, 0),
        g,
    );
    ck(
        "trit::flip draw0 +g ltp saturate",
        stochastic_ternary_flip(g, g, 1, 0),
        g,
    );
    ck(
        "trit::flip draw0 +g ltd",
        stochastic_ternary_flip(g, g, -1, 0),
        0,
    );
    ck(
        "trit::flip draw0 0 ltd",
        stochastic_ternary_flip(0, g, -1, 0),
        -g,
    );
    ck(
        "trit::flip draw0 -g ltd saturate",
        stochastic_ternary_flip(-g, g, -1, 0),
        -g,
    );

    let mut all_ok = true;
    for &w in &[g, 0, -g] {
        if stochastic_ternary_flip(w, g, 5, 65535) != w
            || stochastic_ternary_flip(w, g, -5, 65535) != w
        {
            all_ok = false;
        }
    }
    ck_bool("trit::flip draw_max_never_flips", all_ok);
}

fn kernel_checks() {
    let trits = [Trit::One, Trit::MinusOne, Trit::Zero, Trit::Zero];
    let mut out = [0_u8; 1];
    ck(
        "kernel::pack_known_vector len",
        pack_trits(&trits, &mut out),
        Ok(1),
    );
    ck("kernel::pack_known_vector byte", out[0], 0x52);
    let mut all_ok = true;
    for (i, &t) in trits.iter().enumerate() {
        if unpack_trit(&out, i) != Ok(t) {
            all_ok = false;
        }
    }
    ck_bool("kernel::pack_known_vector round-trip", all_ok);

    let vals = [10_i16, 5, 0, -10];
    let mut out = [0_i16; 4];
    let scale = absmax_normalize_q15(&vals, &mut out);
    ck("kernel::absmax_known_vector scale", scale, 10);
    ck_slice(
        "kernel::absmax_known_vector out",
        &out,
        &[32_767, 16_384, 0, -32_767],
    );

    let vals = [i16::MIN, 0, i16::MAX];
    let mut out = [0_i16; 3];
    let scale = absmax_normalize_q15(&vals, &mut out);
    ck("kernel::absmax_i16_min scale", scale, 32_768);
    ck_slice(
        "kernel::absmax_i16_min out",
        &out,
        &[-32_767, 0, 32_766],
    );

    let mut out = [7_i16; 3];
    let scale = absmax_normalize_q15(&[0, 0, 0], &mut out);
    ck("kernel::absmax_zero_vector scale", scale, 0);
    ck_slice("kernel::absmax_zero_vector out", &out, &[0, 0, 0]);

    let row0 = [Trit::One, Trit::Zero, Trit::MinusOne, Trit::Zero];
    let row1 = [Trit::MinusOne; 4];
    let mut packed = [0_u8; 2];
    let _ = pack_trits(&row0, &mut packed[..1]);
    let _ = pack_trits(&row1, &mut packed[1..]);
    let a = [1000_i16, 5000, 200, 7];
    let mut out = [0_i32; 2];
    ck_bool(
        "kernel::matvec_known_vector runs",
        ternary_matvec(&packed, &a, 2, &mut out).is_ok(),
    );
    ck_slice(
        "kernel::matvec_known_vector out",
        &out,
        &[800, -6207],
    );
}

fn nir_checks() {
    let default = NirImportOptions::default();
    ck(
        "nir::lif_hard_failures tau=0",
        quantize_lif(0.0, 1e8, -0.07, -0.055, -0.08, false, default),
        Err(NirError::BadNumber("tau")),
    );
    ck(
        "nir::lif_hard_failures tau<0",
        quantize_lif(-0.02, 1e8, -0.07, -0.055, -0.08, false, default),
        Err(NirError::BadNumber("tau")),
    );
    ck(
        "nir::lif_hard_failures r=0",
        quantize_lif(0.02, 0.0, -0.07, -0.055, -0.08, false, default),
        Err(NirError::BadNumber("r")),
    );
    ck(
        "nir::lif_hard_failures threshold-zero",
        quantize_lif(0.02, 1e8, -0.07, -0.0004, -0.08, false, default),
        Err(NirError::ThresholdZero),
    );
    ck(
        "nir::lif_hard_failures tau<dt",
        quantize_lif(
            0.02,
            1e8,
            -0.07,
            -0.055,
            -0.08,
            false,
            NirImportOptions::new(30_000, VoltageResolution::Millivolt),
        ),
        Err(NirError::TauBelowDt),
    );
    ck(
        "nir::lif_hard_failures threshold out of range",
        quantize_lif(0.02, 1e8, -0.07, 0.06, -0.08, false, default),
        Err(NirError::PotentialOutOfRange("v_threshold")),
    );
    let lif = quantize_lif(
        0.02,
        1e8,
        -0.07,
        -0.0555,
        -0.08,
        false,
        NirImportOptions::new(1_000, VoltageResolution::CentiMillivolt),
    )
    .unwrap_or_default();
    ck(
        "nir::lif_hard_failures centi threshold_q",
        lif.threshold_q,
        -5550,
    );

    ck(
        "nir::round_half_away r one-ulp-below-half",
        quantize_lif(
            0.02,
            499_999.999_999_999_94,
            -0.07,
            -0.055,
            -0.08,
            false,
            default,
        ),
        Err(NirError::BadNumber("r")),
    );

    let vals = [0.5, -1.0, 0.25];
    let mut arena = [0_i16; 8];
    let lin = quantize_linear(&vals, 1, 3, &mut arena, 0).unwrap_or_else(|_| lin_sentinel());
    ck(
        "nir::quantize_linear dyadic shape",
        (lin.rows, lin.cols, lin.weight_offset),
        (1, 3, 0),
    );
    ck_f64_eps(
        "nir::quantize_linear dyadic scale",
        lin.scale,
        1.0 / 32_767.0,
        1e-18,
    );
    ck_slice(
        "nir::quantize_linear dyadic arena",
        &arena[..3],
        &[16384, -32767, 8192],
    );
    ck_bool(
        "nir::quantize_linear dyadic bounded loss",
        lin.max_abs_err > 0.0 && lin.max_abs_err <= lin.scale / 2.0,
    );
    let lin2 = quantize_linear(&vals, 1, 3, &mut arena, 3).unwrap_or_else(|_| lin_sentinel());
    ck("nir::quantize_linear offset 3", lin2.weight_offset, 3);
    ck_slice(
        "nir::quantize_linear offset arena",
        &arena[3..6],
        &[16384, -32767, 8192],
    );
    ck_bool(
        "nir::quantize_linear same-tensor same-scale",
        lin2.scale == lin.scale,
    );
    let lin3 =
        quantize_linear(&[32767.0, -16384.0, 0.0], 1, 3, &mut arena, 0).unwrap_or_else(|_| lin_sentinel());
    ck("nir::quantize_linear exact scale", lin3.scale, 1.0);
    ck("nir::quantize_linear exact err", lin3.max_abs_err, 0.0);
    ck_slice(
        "nir::quantize_linear exact arena",
        &arena[..3],
        &[32767, -16384, 0],
    );

    let mut arena = [7_i16; 4];
    let lin = quantize_linear(&[0.0; 4], 2, 2, &mut arena, 0).unwrap_or_else(|_| lin_sentinel());
    ck_bool("nir::quantize_linear zero_tensor", lin.zero_tensor);
    ck("nir::quantize_linear zero scale", lin.scale, 1.0);
    ck("nir::quantize_linear zero err", lin.max_abs_err, 0.0);
    ck_slice(
        "nir::quantize_linear zero arena",
        &arena[..4],
        &[0, 0, 0, 0],
    );
    let lin2 = quantize_linear(&[3.0, -3.0, 1.0], 1, 3, &mut arena, 0).unwrap_or_else(|_| lin_sentinel());
    ck_bool(
        "nir::quantize_linear full_scale not zero",
        !lin2.zero_tensor,
    );
    ck("nir::quantize_linear full_scale absmax", lin2.absmax, 3.0);
    ck_slice(
        "nir::quantize_linear full_scale arena",
        &arena[..3],
        &[32767, -32767, 10922],
    );

    for &(v, tag) in [
        (5e-324, "5e-324"),
        (1e-320, "1e-320"),
        (2.47e-321, "2.47e-321"),
    ]
    .iter()
    {
        let mut arena = [0_i16; 4];
        ck_s(
            "nir::denormal_absmax_is_loud",
            tag,
            quantize_linear(&[v], 1, 1, &mut arena, 0),
            Err(NirError::BadNumber("weight")),
        );
    }

    let mut arena = [0_i16; 8];
    for &(v, tag) in [
        (f64::NAN, "NaN"),
        (f64::INFINITY, "+inf"),
        (f64::NEG_INFINITY, "-inf"),
    ]
    .iter()
    {
        ck_s(
            "nir::rejects_loudly non-finite",
            tag,
            quantize_linear(&[1.0, v], 1, 2, &mut arena, 0),
            Err(NirError::BadNumber("weight")),
        );
    }
    ck(
        "nir::rejects_loudly len mismatch",
        quantize_linear(&[1.0], 1, 2, &mut arena, 0),
        Err(NirError::BadShape("weight")),
    );
    ck(
        "nir::rejects_loudly zero rows",
        quantize_linear(&[], 0, 3, &mut arena, 0),
        Err(NirError::BadShape("weight")),
    );
    ck(
        "nir::rejects_loudly zero cols",
        quantize_linear(&[1.0], 1, 0, &mut arena, 0),
        Err(NirError::BadShape("weight")),
    );
    ck(
        "nir::rejects_loudly arena overflow",
        quantize_linear(&[1.0; 6], 2, 3, &mut arena, 4),
        Err(NirError::BufferOverflow),
    );
    ck(
        "nir::rejects_loudly offset overflow",
        quantize_linear(&[1.0; 6], 2, 3, &mut arena, usize::MAX - 2),
        Err(NirError::BufferOverflow),
    );
}

fn spike_raster() {
    let _ = writeln!(
        Uart,
        "raster: 4 LIFNeurons x 40 steps x 1ms (neuron-level; network.rs is std-gated — the network wire is Leg B)"
    );
    let grids = [
        VoltageResolution::Millivolt,
        VoltageResolution::Millivolt,
        VoltageResolution::CentiMillivolt,
        VoltageResolution::CentiMillivolt,
    ];
    let currents = [600_i16, 300, 300, 12];
    let mut neurons = [None, None, None, None];
    for i in 0..4 {
        neurons[i] = Some(quiet_neuron(100 + i as u16, grids[i]));
    }
    let mut rows = [[b'.'; 40]; 4];
    for step in 0..40_usize {
        let t = (step as u32) * 1000;
        for i in 0..4 {
            if let Some(n) = neurons[i].as_mut() {
                if n.integrate_and_fire(currents[i], 1000, t) {
                    rows[i][step] = b'|';
                }
            }
        }
    }
    let labels = [
        "n0 mV  +600uA",
        "n1 mV  +300uA",
        "n2 cmV +300uA (same drive as n1, finer ruler)",
        "n3 cmV  +12uA",
    ];
    for (i, row) in rows.iter().enumerate() {
        let mut line = [0_u8; 40];
        line.copy_from_slice(row);
        let text = core::str::from_utf8(&line).unwrap_or("?");
        let _ = writeln!(Uart, "{} {text}", labels[i]);
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let _ = writeln!(Uart, "PANIC: {info}");
    let _ = writeln!(Uart, "LEG A FAIL (panic)");
    fail()
}
