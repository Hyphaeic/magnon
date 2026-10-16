//! F4 gate — GPU Atxitia-LLB kernel vs the host single-cell reference.
//!
//! Runs the Zhou operating point (1×1 cell, uniform pulse, λ > 0) on both
//! `GpuSolver` and `llb_host::step` and compares the |m|(t) and T_e(t)
//! trajectories step by step. Both integrate the same equations in the same
//! order; the residual is f32-vs-f64 round-off plus table interpolation.
//!
//! Pass criterion: max |Δ|m|| < 1e-3 and max |ΔT_e| / T_e < 1e-3.

use magnonic_clock_sim::config::{Geometry, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::llb_host::{self, LlbCellParams, LlbCellState};
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{parse_pulse_spec, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

const T_AMBIENT_K: f32 = 210.0;
const B_EXT_T: f64 = 1.0;
const THICKNESS_M: f64 = 5.0e-9;
const DT: f64 = 5.0e-15;
const STEPS: usize = 3_000; // 15 ps

fn main() {
    env_logger::init();
    let mut preset = material_thermal::fgt_zhou_calibrated();
    preset.lambda_bath = 0.1;
    preset.g_sub_phonon = 6.0e16;
    let material = BulkMaterial::fgt_bulk();
    let mut stack = Stack::monolayer(material.clone(), THICKNESS_M, [0.0, 0.0, 1.0]);
    stack.layers[0].material.alpha_bulk = preset.alpha_0;

    let mut cfg = SimConfig {
        stack,
        substrate: Substrate::vacuum(),
        geometry: Geometry {
            nx: 1,
            ny: 1,
            cell_size: 20e-9,
        },
        dt: DT,
        b_ext: [0.0, 0.0, B_EXT_T],
        stab_coeff: 0.0,
        j_current: [0.0, 0.0, 0.0],
        photonic: Default::default(),
        readback_interval: 1,
        total_steps: STEPS,
        probe_idx: None,
        probe_layer: None,
    };
    let pulse = parse_pulse_spec("t=4ps,fwhm=150fs,peak=0.0,dir=z,fluence=0.24,R=0.5").unwrap();
    cfg.photonic.pulses.push(pulse.clone());
    cfg.photonic.thermal = Some(ThermalConfig {
        t_ambient: T_AMBIENT_K,
        per_layer: vec![preset.clone()],
        thermal_dt_cap: DT,
        thermal_window: (1e-12, 40e-12),
        enable_llb: true,
    });

    // Host reference.
    let atten = (1.0 - (-THICKNESS_M / preset.optical_skin_depth_m).exp()) / THICKNESS_M;
    let host_p = LlbCellParams {
        gamma: material.gamma,
        anis_pf: 2.0 * material.k_u_bulk / material.ms_bulk,
        u_axis: [0.0, 0.0, 1.0],
        b_ext: cfg.b_ext,
        thermal: preset.clone(),
        t_ambient: T_AMBIENT_K as f64,
        optical_atten: atten,
    };
    let mut hs = LlbCellState::equilibrium_z(&host_p);
    let m0 = hs.m[2] as f32;

    let mut solver = GpuSolver::new(&cfg).expect("GPU init");
    solver.reset_uniform_z_at(m0);

    let sigma = pulse.duration_fwhm / 2.354_820_045;
    let tau = (2.0 * std::f64::consts::PI).sqrt();
    let mut max_dm = 0.0f64;
    let mut max_dt_rel = 0.0f64;
    for i in 0..STEPS {
        let t_mid = (i as f64 + 0.5) * DT;
        let flux = (1.0 - pulse.reflectivity as f64) * pulse.peak_fluence.unwrap() / (tau * sigma)
            * pulse.envelope_at(t_mid);
        hs = llb_host::step(&host_p, hs, flux, DT);
        solver.step_n(1);
        let obs = solver.observables();
        let te = solver.readback_temp_e()[0] as f64;
        let hm = (hs.m[0].powi(2) + hs.m[1].powi(2) + hs.m[2].powi(2)).sqrt();
        let dm = (hm - obs.min_norm as f64).abs();
        let dte = (hs.t_e - te).abs() / hs.t_e;
        max_dm = max_dm.max(dm);
        max_dt_rel = max_dt_rel.max(dte);
        if i % 500 == 0 || i == STEPS - 1 {
            println!(
                "step {i:5} t={:6.2} ps  host |m|={hm:.5} gpu |m|={:.5}  host T_e={:.1} gpu T_e={te:.1}",
                obs.time_ps, obs.min_norm, hs.t_e
            );
        }
    }
    println!("max |Δ|m|| = {max_dm:.3e}, max |ΔT_e|/T_e = {max_dt_rel:.3e}");
    if max_dm < 1e-3 && max_dt_rel < 1e-3 {
        println!("PASS: Atxitia LLB GPU kernel agrees with host reference.");
    } else {
        println!("FAIL: GPU/host mismatch exceeds tolerance.");
        std::process::exit(1);
    }
}
