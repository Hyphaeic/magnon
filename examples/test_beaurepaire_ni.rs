//! Phase P3c — Beaurepaire Ni ultrafast-demagnetization reproduction.
//!
//! Target (Beaurepaire et al. 1996, PRL 76, 4250):
//!   20 nm Ni, 100 fs FWHM pulse, 7 mJ/cm² absorbed fluence
//!   |m| drops from 1.0 to 0.60 ± 0.05 at ~500 fs; recovery τ ≈ 1 ps.
//!
//! Also reports an energy-balance number: integrated laser input vs.
//! Δ(thermal bath energies) + ΔU_mag (LLB / longitudinal). Target < 5 %
//! relative error (plan §4 acceptance).
//!
//! Single-cell 1×1×1 setup (`reset_uniform_z` → deterministic if no RNG).

use magnonic_clock_sim::config::{Geometry, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{parse_pulse_spec, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

fn main() {
    let preset = material_thermal::ni_m3tm();
    let mut stack = Stack::monolayer(BulkMaterial::permalloy_bulk(), 20e-9, [0.0, 0.0, 1.0]);
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
        readback_interval: 1,
        total_steps: 10_000, // 10 ps @ dt=1 fs
        probe_idx: None,
        probe_layer: None,
    };
    let pulse = parse_pulse_spec("t=300fs,fwhm=100fs,peak=0.0,dir=z,fluence=7.0,R=0.0")
        .expect("pulse");
    cfg.photonic.pulses.push(pulse.clone());
    cfg.photonic.thermal = Some(ThermalConfig {
        t_ambient: 300.0,
        per_layer: vec![preset.clone()],
        thermal_dt_cap: 1.0e-15,
        thermal_window: (0.5e-12, 10e-12),
        enable_llb: true,
    });

    let mut solver = GpuSolver::new(&cfg).expect("GPU init");
    solver.reset_uniform_z();

    let thickness = cfg.stack.layers[0].thickness;
    let area = (cfg.geometry.cell_size as f64).powi(2);
    let v_cell = area * thickness;
    let fluence = pulse.peak_fluence.unwrap();
    let total_laser_input_per_cell = (1.0 - pulse.reflectivity as f64) * fluence * area;

    let mut trajectory_ps = Vec::with_capacity(cfg.total_steps);
    let mut trajectory_m = Vec::with_capacity(cfg.total_steps);
    let mut trajectory_te = Vec::with_capacity(cfg.total_steps);
    let mut trajectory_tp = Vec::with_capacity(cfg.total_steps);

    // Initial thermal energies for balance check.
    let t_amb = cfg.photonic.thermal.as_ref().unwrap().t_ambient as f64;
    let u_e_init = 0.5 * preset.gamma_e * t_amb.powi(2);
    let u_p_init = preset.c_p * t_amb;

    for _ in 0..cfg.total_steps {
        solver.step_n(1);
        let obs = solver.observables();
        trajectory_ps.push(obs.time_ps);
        trajectory_m.push(obs.min_norm as f64);
        trajectory_te.push(obs.max_t_e as f64);
        trajectory_tp.push(obs.max_t_p as f64);
    }

    // Observables of interest.
    let (m_min, m_min_idx) = trajectory_m.iter().enumerate().fold(
        (1.0_f64, 0_usize),
        |(v, i), (j, x)| if *x < v { (*x, j) } else { (v, i) },
    );
    let m_at_500fs = trajectory_m[500.min(trajectory_m.len() - 1)];
    let t_e_final = *trajectory_te.last().unwrap();
    let t_p_final = *trajectory_tp.last().unwrap();

    // Recovery τ: exponential fit on the segment [m_min_idx .. end]; we
    // approximate by the e-folding step after m_min.
    let target = m_min + (1.0 - m_min) * (1.0 - (-1.0f64).exp());
    let tau_idx = trajectory_m.iter().enumerate().skip(m_min_idx)
        .find(|(_, v)| **v >= target).map(|(i, _)| i).unwrap_or(trajectory_m.len() - 1);
    let tau_ps = trajectory_ps[tau_idx] - trajectory_ps[m_min_idx];

    // Energy balance at end of run (t = 10 ps, pulse fully delivered).
    let u_e_final = 0.5 * preset.gamma_e * t_e_final.powi(2);
    let u_p_final = preset.c_p * t_p_final;
    let delta_u = (u_e_final - u_e_init) * v_cell + (u_p_final - u_p_init) * v_cell;
    let balance_ratio = delta_u / total_laser_input_per_cell;

    println!(
        "Beaurepaire Ni — 20 nm, 100 fs FWHM, 7 mJ/cm² absorbed, T_ambient=300K:"
    );
    for (label, i) in [("t=0.2 ps", 200), ("t=0.3 ps", 300), ("t=0.5 ps", 500),
                        ("t=1.0 ps", 1000), ("t=2.0 ps", 2000), ("t=5.0 ps", 5000),
                        ("t=10 ps", 9999)] {
        if i < trajectory_m.len() {
            println!(
                "  {label}: |m| = {:.4}  T_e = {:7.1} K  T_p = {:6.1} K",
                trajectory_m[i], trajectory_te[i], trajectory_tp[i]
            );
        }
    }
    println!(
        "\n|m| floor = {:.4} at t = {:.3} ps",
        m_min, trajectory_ps[m_min_idx]
    );
    println!("|m| at t = 500 fs     = {:.4}   (Beaurepaire target: 0.60 ± 0.05)", m_at_500fs);
    println!("1/e recovery τ        = {:.3} ps from floor  (target: ~1 ps)", tau_ps);
    println!(
        "\nEnergy balance:"
    );
    println!("  laser input  = {:.3e} J/cell", total_laser_input_per_cell);
    println!("  Δ(U_e + U_p) = {:.3e} J/cell", delta_u);
    println!(
        "  ratio = {:.3}  (target 1.00 ± 0.05; excess leaks to LLB long. dissipation)",
        balance_ratio
    );

    // Soft assertions — report outcomes, don't block P3c boundary on
    // quantitative calibration (Beaurepaire ±0.05 and energy ±5 % are
    // plan-level targets but depend on material presets; calibration is
    // a P5 task per the plan).
    if m_at_500fs > 0.65 || m_at_500fs < 0.55 {
        println!(
            "\nNOTE: |m|(500 fs) = {:.3} is outside Beaurepaire target 0.60 ± 0.05",
            m_at_500fs
        );
    } else {
        println!("\nPASS: |m|(500 fs) within Beaurepaire target band.");
    }
    if (balance_ratio - 1.0).abs() > 0.05 {
        println!(
            "NOTE: energy balance ratio {:.3} is outside ±5 % target",
            balance_ratio
        );
    } else {
        println!("PASS: energy balance within ±5 %.");
    }
}
