//! Phase P3a acceptance test — GPU `advance_m3tm` kernel vs host
//! `thermal::advance_m3tm_cell` reference integrator.
//!
//! Runs a 1×1×1 single-cell Ni M3TM simulation for 3 ps under a 100 fs
//! Gaussian laser pulse (1 mJ/cm², 20 nm film) on both paths and compares
//! the T_e trajectory — must agree to 1e-3 relative.
//!
//! Run with `cargo run --release --example test_m3tm_gpu_vs_host`.

use magnonic_clock_sim::config::{Geometry, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{parse_pulse_spec, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;
use magnonic_clock_sim::thermal::{advance_m3tm_cell, M3tmState};

fn main() {
    // Single-cell config: 1×1 in-plane, 1 layer of 20 nm Ni-like.
    let mut stack = Stack::monolayer(BulkMaterial::permalloy_bulk(), 20e-9, [0.0, 0.0, 1.0]);
    // Use Ni thermal preset (the bulk material only drives the LLG side).
    let ni_preset = material_thermal::ni_m3tm();
    stack.layers[0].material.tc_bulk = ni_preset.t_c;

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
        total_steps: 3000,
        probe_idx: None,
        probe_layer: None,
    };
    // 100 fs FWHM Gaussian at t = 300 fs, 1 mJ/cm² (= 10 J/m²), uniform in x,y,
    // peak_field irrelevant here (no coherent IFE torque — dir=z, field=0).
    let pulse_spec = "t=300fs,fwhm=100fs,peak=0.0,dir=z,fluence=1.0,R=0.0";
    let pulse = parse_pulse_spec(pulse_spec).expect("pulse parse");
    cfg.photonic.pulses.push(pulse.clone());
    cfg.photonic.thermal = Some(ThermalConfig {
        per_layer: vec![ni_preset.clone()],
        t_ambient: 300.0,
        thermal_dt_cap: 1.0e-15,
        thermal_window: (0.5e-12, 10e-12),
        enable_llb: false,
    });

    // ─── Host reference trajectory ─────────────────────────────
    let dt = cfg.dt;
    let t_amb = 300.0_f64;
    let mut s = M3tmState::at_ambient(t_amb, ni_preset.sample_m_e(t_amb));
    let sig_t = pulse.duration_fwhm / 2.354_820_045;
    let p_peak = (1.0 - pulse.reflectivity as f64) * pulse.peak_fluence.unwrap()
        / ((2.0 * std::f64::consts::PI).sqrt() * sig_t * cfg.stack.layers[0].thickness);

    let n_steps = cfg.total_steps;
    let mut host_t_e = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        // Midpoint envelope (matches the GPU-side convention).
        let t_mid = (i as f64) * dt + 0.5 * dt;
        let dt_off = t_mid - pulse.t_center;
        let env = (-(dt_off * dt_off) / (2.0 * sig_t * sig_t)).exp();
        let p_laser = p_peak * env;
        s = advance_m3tm_cell(s, p_laser, &ni_preset, dt);
        host_t_e.push(s.t_e);
    }

    // Also track host m trajectory.
    let mut host_m = Vec::with_capacity(n_steps);
    let mut s2 = M3tmState::at_ambient(t_amb, ni_preset.sample_m_e(t_amb));
    for i in 0..n_steps {
        let t_mid = (i as f64) * dt + 0.5 * dt;
        let dt_off = t_mid - pulse.t_center;
        let env = (-(dt_off * dt_off) / (2.0 * sig_t * sig_t)).exp();
        let p_laser = p_peak * env;
        s2 = advance_m3tm_cell(s2, p_laser, &ni_preset, dt);
        host_m.push(s2.m);
    }

    // ─── GPU trajectory ────────────────────────────────────────
    let mut solver = GpuSolver::new(&cfg).expect("GPU init");
    let mut gpu_t_e = Vec::with_capacity(n_steps);
    let mut gpu_m = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        solver.step_n(1);
        let obs = solver.observables();
        gpu_t_e.push(obs.max_t_e as f64);
        gpu_m.push(obs.min_m_reduced as f64);
    }

    // ─── Compare ──────────────────────────────────────────────
    let mut max_rel = 0.0_f64;
    let mut at_step = 0usize;
    let mut host_peak = 0.0_f64;
    let mut gpu_peak = 0.0_f64;
    for (i, (h, g)) in host_t_e.iter().zip(gpu_t_e.iter()).enumerate() {
        host_peak = host_peak.max(*h);
        gpu_peak = gpu_peak.max(*g);
        let rel = (h - g).abs() / h.max(1.0);
        if rel > max_rel {
            max_rel = rel;
            at_step = i;
        }
    }

    let mut max_m_abs = 0.0_f64;
    let mut host_m_floor = 1.0_f64;
    let mut gpu_m_floor = 1.0_f64;
    for (h, g) in host_m.iter().zip(gpu_m.iter()) {
        host_m_floor = host_m_floor.min(*h);
        gpu_m_floor = gpu_m_floor.min(*g);
        max_m_abs = max_m_abs.max((h - g).abs());
    }
    println!("host T_e peak = {:.2} K, GPU T_e peak = {:.2} K", host_peak, gpu_peak);
    println!("max relative T_e diff = {:.3e} at step {}", max_rel, at_step);
    println!(
        "host m floor = {:.6}, GPU m floor = {:.6}, max |Δm| = {:.3e}",
        host_m_floor, gpu_m_floor, max_m_abs
    );

    // A few sample points around the pulse for visual inspection.
    for i in [200, 300, 400, 500, 800, 1500, 2500] {
        if i < n_steps {
            println!(
                "step {:5} t={:.3} ps: host={:.2}K gpu={:.2}K",
                i,
                (i as f64 * dt) * 1e12,
                host_t_e[i],
                gpu_t_e[i]
            );
        }
    }

    assert!(host_peak > 400.0, "host peak too low ({host_peak:.1}K) — laser model broken?");
    assert!(
        max_rel < 1.0e-3,
        "GPU and host M3TM disagree: max rel = {:.3e} (tolerance 1e-3)",
        max_rel
    );
    assert!(
        max_m_abs < 1.0e-3,
        "GPU and host |m| disagree: max |Δm| = {:.3e} (tolerance 1e-3)",
        max_m_abs
    );
    println!(
        "PASS: M3TM GPU agrees with host reference — T_e to {:.3e}, |m| to {:.3e}",
        max_rel, max_m_abs
    );
}
