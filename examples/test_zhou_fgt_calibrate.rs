//! Zhou 2025 FGT re-calibration — Phase F4 (Atxitia longitudinal LLB).
//!
//! Fits the `fgt_zhou_calibrated` preset against the aggregate ultrafast-
//! demagnetisation morphology of Zhou et al. *Natl. Sci. Rev.* 12, nwaf185
//! (2025) at the literal operating point T = T_c = 210 K, B_z = 1 T,
//! F = 0.24 mJ/cm², 150 fs FWHM, 5 nm flake:
//!   (a) demag fraction      = (|m|_0 − |m|_floor)/|m|_0          target 0.79
//!   (b) recovery at 22 ps   = (|m|_22 − |m|_floor)/(|m|_0 − |m|_floor)  target 0.55
//!   L = (a − 0.79)² + 0.5·(b − 0.55)²
//!
//! What changed in F4 relative to the F1–F3 fit:
//!   * The longitudinal torque is the Garanin/Atxitia form driven by the
//!     MFA χ_∥(T, B) table. The sub-femtosecond `tau_long`/`tau_fast`
//!     knobs are gone; the only dynamic-physics knob is the bath coupling λ.
//!   * K_u follows Callen-Callen (κ = 3), α_⊥ is finite above T_c.
//!   * The run starts at thermal equilibrium |m| = m_e(T_c, 1 T) instead of
//!     |m| = 1, so `m_initial` is the true pre-pulse state.
//!
//! Grid: λ × R × g_sub_phonon. The *diagnostic* question is whether the
//! best-fit g_sub lands near the Kapitza-conductance estimate for a 5 nm
//! vdW flake (G ≈ 50–100 MW m⁻² K⁻¹ ⇒ g_sub = G/t ≈ 1–2·10¹⁶ W m⁻³ K⁻¹)
//! and R near a measured value, rather than wherever the loss minimum
//! happens to sit.
//!
//! Usage:
//!   cargo run --release --example test_zhou_fgt_calibrate [-- --cpu]
//!       [--lambda 0.03,0.1,0.3] [--r 0.3,0.5] [--gsub 2e16,6e16] [--fluence 0.12]
//! `--cpu` uses the host single-cell reference (`llb_host`) — identical
//! equations, no GPU needed; `test_llb_atxitia_gpu_vs_host` checks the two
//! agree.

use magnonic_clock_sim::config::{Geometry, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::llb_host::{self, LlbCellParams, LlbCellState};
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{parse_pulse_spec, LayerThermalParams, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

const T_AMBIENT_K: f32 = 210.0;
const T_C_K: f64 = 210.0;
const B_EXT_T: f64 = 1.0;
const FLUENCE_MJ_CM2_DEFAULT: f64 = 0.24;
static FLUENCE_MJ_CM2: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
fn fluence() -> f64 {
    *FLUENCE_MJ_CM2.get().unwrap_or(&FLUENCE_MJ_CM2_DEFAULT)
}
const PULSE_FWHM_FS: f64 = 150.0;
const FLAKE_THICKNESS_NM: f64 = 5.0;
const SKIN_DEPTH_M: f64 = 18.0e-9;
const DT: f64 = 5.0e-15;
const TOTAL_STEPS: usize = 12_000; // 60 ps
const PULSE_T0_PS: f64 = 4.0;

const TARGET_DEMAG_FRAC: f64 = 0.79;
const TARGET_RECOVERY_FRAC: f64 = 0.55;

const LAMBDA_GRID: &[f64] = &[0.03, 0.1, 0.3, 1.0];
const R_GRID: &[f32] = &[0.30, 0.50, 0.70];
const G_SUB_GRID: &[f64] = &[2.0e16, 6.0e16, 2.0e17, 6.0e17];

#[derive(Clone, Debug)]
struct TrialMetrics {
    lambda: f64,
    r: f32,
    g_sub: f64,
    demag_frac: f64,
    recovery_frac: f64,
    m_initial: f64,
    m_floor: f64,
    m_22ps: f64,
    t_min_ps: f64,
    t_e_peak: f64,
    loss: f64,
}

fn preset_for(lambda: f64, g_sub: f64) -> LayerThermalParams {
    let mut preset = material_thermal::fgt_ni_surrogate();
    preset.t_c = T_C_K;
    preset.lambda_bath = lambda;
    preset.g_sub_phonon = g_sub;
    preset.optical_skin_depth_m = SKIN_DEPTH_M;
    preset.k_u_exponent = 3.0;
    let (m_e_table, chi_table) = material_thermal::brillouin_tables_spin_half_2d(
        preset.t_c,
        preset.llb_table_n,
        preset.llb_table_n_b,
        preset.b_max_t,
    );
    preset.m_e_table = m_e_table;
    preset.chi_par_table = chi_table;
    preset
}

fn pulse_spec(r: f32) -> String {
    format!(
        "t={PULSE_T0_PS}ps,fwhm={:.0}fs,peak=0.0,dir=z,fluence={:.4},R={:.3}",
        PULSE_FWHM_FS,
        fluence(),
        r
    )
}

/// Returns (time_ps, |m|, T_e) per step.
fn trajectory_gpu(preset: &LayerThermalParams, r: f32) -> Vec<(f64, f64, f64)> {
    let mut stack = Stack::monolayer(
        BulkMaterial::fgt_bulk(),
        FLAKE_THICKNESS_NM * 1e-9,
        [0.0, 0.0, 1.0],
    );
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
        total_steps: TOTAL_STEPS,
        probe_idx: None,
        probe_layer: None,
    };
    cfg.photonic
        .pulses
        .push(parse_pulse_spec(&pulse_spec(r)).unwrap());
    cfg.photonic.thermal = Some(ThermalConfig {
        t_ambient: T_AMBIENT_K,
        per_layer: vec![preset.clone()],
        thermal_dt_cap: DT,
        thermal_window: (1e-12, 40e-12),
        enable_llb: true,
    });
    let m0 = preset.sample_m_e_2d(T_AMBIENT_K as f64, B_EXT_T) as f32;
    let mut solver = GpuSolver::new(&cfg).expect("GPU init");
    solver.reset_uniform_z_at(m0);
    let mut out = Vec::with_capacity(TOTAL_STEPS);
    for _ in 0..TOTAL_STEPS {
        solver.step_n(1);
        let obs = solver.observables();
        let te = solver.readback_temp_e()[0] as f64;
        out.push((obs.time_ps, obs.min_norm as f64, te));
    }
    out
}

fn trajectory_cpu(preset: &LayerThermalParams, r: f32) -> Vec<(f64, f64, f64)> {
    let material = BulkMaterial::fgt_bulk();
    let pulse = parse_pulse_spec(&pulse_spec(r)).unwrap();
    let t = FLAKE_THICKNESS_NM * 1e-9;
    let p = LlbCellParams {
        gamma: material.gamma,
        anis_pf: 2.0 * material.k_u_bulk / material.ms_bulk,
        u_axis: [0.0, 0.0, 1.0],
        b_ext: [0.0, 0.0, B_EXT_T],
        thermal: preset.clone(),
        t_ambient: T_AMBIENT_K as f64,
        optical_atten: (1.0 - (-t / SKIN_DEPTH_M).exp()) / t,
    };
    let sigma = pulse.duration_fwhm / 2.354_820_045;
    let tau = (2.0 * std::f64::consts::PI).sqrt();
    let mut s = LlbCellState::equilibrium_z(&p);
    let mut out = Vec::with_capacity(TOTAL_STEPS);
    for i in 0..TOTAL_STEPS {
        let t_mid = (i as f64 + 0.5) * DT;
        let flux = (1.0 - pulse.reflectivity as f64) * pulse.peak_fluence.unwrap() / (tau * sigma)
            * pulse.envelope_at(t_mid);
        s = llb_host::step(&p, s, flux, DT);
        let m = (s.m[0].powi(2) + s.m[1].powi(2) + s.m[2].powi(2)).sqrt();
        out.push(((i + 1) as f64 * DT * 1e12, m, s.t_e));
    }
    out
}

fn run_trial(lambda: f64, r: f32, g_sub: f64, cpu: bool) -> TrialMetrics {
    let preset = preset_for(lambda, g_sub);
    let traj = if cpu {
        trajectory_cpu(&preset, r)
    } else {
        trajectory_gpu(&preset, r)
    };

    // Reference just before the pulse peak.
    let i_ref = traj
        .iter()
        .position(|&(t, _, _)| t > PULSE_T0_PS - 0.5)
        .unwrap_or(0);
    let m_initial = traj[i_ref].1.max(1e-6);
    let (m_floor, i_min) =
        traj.iter()
            .enumerate()
            .skip(i_ref)
            .fold(
                (m_initial, i_ref),
                |(v, i), (j, &(_, x, _))| {
                    if x < v {
                        (x, j)
                    } else {
                        (v, i)
                    }
                },
            );
    let i_22 = traj
        .iter()
        .position(|&(t, _, _)| t > PULSE_T0_PS + 22.0)
        .unwrap_or(traj.len() - 1);
    let m_22ps = traj[i_22].1;
    let t_e_peak = traj.iter().map(|x| x.2).fold(0.0, f64::max);

    let demag_frac = 1.0 - m_floor / m_initial;
    let recovery_frac = if (m_initial - m_floor).abs() > 1e-9 {
        (m_22ps - m_floor) / (m_initial - m_floor)
    } else {
        0.0
    };
    let loss = (demag_frac - TARGET_DEMAG_FRAC).powi(2)
        + 0.5 * (recovery_frac - TARGET_RECOVERY_FRAC).powi(2);

    TrialMetrics {
        lambda,
        r,
        g_sub,
        demag_frac,
        recovery_frac,
        m_initial,
        m_floor,
        m_22ps,
        t_min_ps: traj[i_min].0,
        t_e_peak,
        loss,
    }
}

fn parse_list<T: std::str::FromStr>(s: &str) -> Vec<T> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let mut lambda_grid: Vec<f64> = LAMBDA_GRID.to_vec();
    let mut r_grid: Vec<f32> = R_GRID.to_vec();
    let mut g_sub_grid: Vec<f64> = G_SUB_GRID.to_vec();
    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--lambda" => lambda_grid = parse_list(&args[i + 1]),
            "--r" => r_grid = parse_list(&args[i + 1]),
            "--gsub" => g_sub_grid = parse_list(&args[i + 1]),
            "--fluence" => {
                let f: f64 = args[i + 1].parse().expect("--fluence mJ/cm²");
                let _ = FLUENCE_MJ_CM2.set(f);
            }
            _ => {
                i += 1;
                continue;
            }
        }
        i += 2;
    }

    let n_trials = lambda_grid.len() * r_grid.len() * g_sub_grid.len();
    println!(
        "=== FGT Zhou-morphology calibration (F4 Atxitia LLB), grid search ({n_trials} trials, {}) ===",
        if cpu { "host reference" } else { "GPU" }
    );
    println!(
        "Operating point: T = {} K, B_z = {} T, F = {} mJ/cm², FWHM = {} fs, t = {} nm FGT, δ = {} nm, κ = 3",
        T_AMBIENT_K, B_EXT_T, fluence(), PULSE_FWHM_FS, FLAKE_THICKNESS_NM, SKIN_DEPTH_M * 1e9,
    );
    println!(
        "Targets:  demag_frac = {:.2}   recovery_frac(22 ps) = {:.2}",
        TARGET_DEMAG_FRAC, TARGET_RECOVERY_FRAC,
    );
    println!();

    let mut results: Vec<TrialMetrics> = Vec::with_capacity(n_trials);
    let t0 = std::time::Instant::now();
    let mut count = 0usize;
    for &lambda in &lambda_grid {
        for &r in &r_grid {
            for &g_sub in &g_sub_grid {
                count += 1;
                let m = run_trial(lambda, r, g_sub, cpu);
                println!(
                    "[{count:3}/{n_trials}] λ={:.3} R={:.2} g_sub={:.1e} → m0={:.3} floor={:.3}@{:.1}ps m22={:.3} Te_pk={:.0}K demag={:.3} rec={:.3} L={:.4}",
                    m.lambda, m.r, m.g_sub, m.m_initial, m.m_floor, m.t_min_ps, m.m_22ps,
                    m.t_e_peak, m.demag_frac, m.recovery_frac, m.loss,
                );
                results.push(m);
            }
        }
    }
    println!("\nGrid done in {:.1} s", t0.elapsed().as_secs_f64());

    results.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap());
    println!("\nTop 5 by loss:");
    println!(
        "  {:>6} {:>5} {:>9} {:>7} {:>7} {:>8}",
        "λ", "R", "g_sub", "demag", "rec", "loss"
    );
    for m in results.iter().take(5) {
        println!(
            "  {:>6.3} {:>5.2} {:>9.1e} {:>7.3} {:>7.3} {:>8.4}",
            m.lambda, m.r, m.g_sub, m.demag_frac, m.recovery_frac, m.loss
        );
    }
    let best = &results[0];
    println!("\nBest fit:");
    println!(
        "lambda_bath = {:.3}  R = {:.2}  g_sub_phonon = {:.3e} W/(m³·K)",
        best.lambda, best.r, best.g_sub
    );
    println!(
        "Observed: demag = {:.1} % (target 79%),  recovery@22ps = {:.1} % (target 55%),  loss = {:.4}",
        best.demag_frac * 100.0,
        best.recovery_frac * 100.0,
        best.loss
    );
    println!("\nPaste into material_thermal::fgt_zhou_calibrated():");
    println!("    p.lambda_bath = {:.3};", best.lambda);
    println!("    p.g_sub_phonon = {:.3e};", best.g_sub);
    println!("    // pulse reflectivity R = {:.2}", best.r);
}
