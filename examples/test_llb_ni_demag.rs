//! Phase P3b smoke test — LLB integrator under a Ni-style ultrafast pulse.
//!
//! NOT a Beaurepaire calibration (that's a P3c / P5 target). This run only
//! demonstrates that with `enable_llb = true` and a pulse-driven M3TM:
//!   1. |m| drops below 1 (demagnetization happens).
//!   2. |m| recovers toward m_e(T_ambient) as T_e cools (M3TM → LLB loop).
//!   3. No NaN / no blow-up over 10 ps.
//!
//! 1×1×1 single-cell configuration with a 20-nm Ni-like layer.

use magnonic_clock_sim::config::{Geometry, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{parse_pulse_spec, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

fn main() {
    let mut stack = Stack::monolayer(BulkMaterial::permalloy_bulk(), 20e-9, [0.0, 0.0, 1.0]);
    let preset = material_thermal::ni_m3tm();
    // Align LLG α with the thermal preset α_0 so transverse damping is
    // consistent between paths.
    stack.layers[0].material.alpha_bulk = preset.alpha_0;

    let mut cfg = SimConfig {
        stack,
        substrate: Substrate::vacuum(),
        geometry: Geometry { nx: 1, ny: 1, cell_size: 20e-9 },
        dt: 1.0e-15,
        b_ext: [0.0, 0.0, 0.0],
        stab_coeff: 0.0,
        j_current: [0.0, 0.0, 0.0],
        photonic: Default::default(),
        readback_interval: 100,
        total_steps: 10_000, // 10 ps @ dt=1fs
        probe_idx: None,
        probe_layer: None,
    };
    let pulse = parse_pulse_spec("t=300fs,fwhm=100fs,peak=0.0,dir=z,fluence=1.0,R=0.0")
        .expect("pulse");
    cfg.photonic.pulses.push(pulse);
    cfg.photonic.thermal = Some(ThermalConfig {
        t_ambient: 300.0,
        per_layer: vec![preset],
        thermal_dt_cap: 1.0e-15,
        thermal_window: (0.5e-12, 10e-12),
        enable_llb: true,
    });

    let mut solver = GpuSolver::new(&cfg).expect("GPU init");
    solver.reset_uniform_z();

    let mut m_floor = 1.0_f32;
    let mut t_e_peak = 0.0_f32;
    let mut last_m = 1.0_f32;
    let mut any_nan = false;

    for step in 0..cfg.total_steps {
        solver.step_n(1);
        let obs = solver.observables();
        if obs.min_norm.is_nan() || obs.min_norm.is_infinite() {
            any_nan = true;
            break;
        }
        m_floor = m_floor.min(obs.min_norm);
        t_e_peak = t_e_peak.max(obs.max_t_e);
        last_m = obs.min_norm;

        if step == 200 || step == 400 || step == 1000 || step == 5000 || step == 9999 {
            println!(
                "step {:5} t={:.2} ps: T_e_max={:7.1}K  |m|_min={:.4}  m_reduced_min={:.4}",
                step,
                obs.time_ps,
                obs.max_t_e,
                obs.min_norm,
                obs.min_m_reduced,
            );
        }
    }

    println!(
        "\nSUMMARY: T_e peak = {:.1} K, |m| floor = {:.4}, |m| @ 10 ps = {:.4}",
        t_e_peak, m_floor, last_m
    );

    assert!(!any_nan, "NaN detected in LLB integrator");
    assert!(t_e_peak > 400.0, "T_e never exceeded 400 K — pulse not driving M3TM");
    assert!(m_floor < 0.95, "LLB did not demagnetize at all (floor = {})", m_floor);
    assert!(last_m > 0.5, "LLB did not recover — final |m| = {}", last_m);
    println!("PASS: LLB demag + recovery, no NaN.");
}
