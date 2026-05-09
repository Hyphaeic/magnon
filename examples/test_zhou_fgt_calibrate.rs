//! Phase P5 — FGT parameter calibration against Zhou 2025 morphology.
//!
//! **Assumptions made explicit (see docs/zhou_fgt_calibration.md §Limits):**
//!
//! 1. Operating point shifted to T = 150 K (Zhou measured at T = T_c = 210 K).
//!    Reason: our m_e(T) tables are the zero-field Brillouin / MFA solution.
//!    At T = T_c the zero-field m_e = 0, so the pre-pulse state is ill-defined;
//!    Zhou's 1 T field enhances m_e at T_c to a finite value but our tables
//!    don't carry that field dependence. At T = 150 K (≈ 0.71·T_c) m_e = 0.85
//!    and the field-induced correction is small (<5 %).
//!
//! 2. We fit the *morphology* of Zhou's curve, not the exact experimental
//!    trace. Matched features:
//!       (a) demagnetization fraction = (|m|_initial − |m|_floor) / |m|_initial
//!           target: 0.79 (Zhou's headline number at T_c with B = 1 T)
//!       (b) recovery fraction at 22 ps after the pulse =
//!           (|m|_22ps − |m|_floor) / (|m|_initial − |m|_floor)
//!           target: 0.55 (partial recovery in the 22 ps window, consistent
//!           with Zhou's slow-stage τ_2 ≈ 22 ps)
//!
//! 3. Our LLB ships with a phenomenological longitudinal relaxation —
//!    single-timescale by construction. Zhou reports a two-stage decay
//!    (τ_1 = 0.4 ps, τ_2 = 5.1–22.2 ps). A clean two-stage signature is NOT
//!    reproducible with the current model regardless of parameters; we only
//!    fit aggregate demag + recovery.
//!
//! 4. Pump intensity in the absence of optical-skin-depth physics: we assume
//!    uniform absorption across the 5 nm flake (`absorbed power density =
//!    (1 − R) · F / thickness / (√(2π)·σ_t)`). Real experiments have an
//!    exponential absorption profile; for a 5 nm flake with 400-nm optical
//!    skin depth ≈ 15 nm this approximation is ~0.7× error in the deposited
//!    energy — absorbed fluence is bounded, so we don't compensate in the
//!    fit.
//!
//! 5. **`a_sf` is inert under the LLB back-coupling path (P3c).** When LLB
//!    owns |m| dynamics, the Koopmans dm/dt term is replaced by the LLB
//!    longitudinal torque, so `a_sf` drops out of the observables. The
//!    effective calibration knob for the demag *depth* is `tau_long_base`
//!    (the LLB longitudinal-relaxation time constant). `a_sf` reappears as
//!    relevant only if LLB is disabled (which removes the whole P3b upgrade)
//!    or in a future full-Atxitia LLB form where χ_∥ carries a_sf
//!    implicitly. Calibration grid therefore uses `tau_long_base` instead.
//!
//! Calibration grid:
//!   tau_long_base ∈ {0.1, 0.3, 1.0, 3.0, 10.0} fs
//!   reflectivity  ∈ {0.10, 0.30, 0.50, 0.70, 0.85}
//!   g_sub_phonon  ∈ {5e16, 1e17, 2e17, 5e17, 1e18}
//! 125 trials — ≈2 min wall-clock on an RTX 4060.
//!
//! Output: prints top-5 (by L) and best fit, formatted for direct paste into
//! a `fgt_calibrated()` preset. Loss function:
//!   L = (demag_frac − 0.79)² + 0.5·(recovery_frac − 0.55)²

use magnonic_clock_sim::config::{Geometry, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{parse_pulse_spec, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

// ─── Fixed operating point (Zhou 2025 literal, post-F1) ────
// Pre-F1 the calibration shifted to T = 150 K because zero-field tables
// made T = T_c degenerate. After F1 (field-dependent m_e tables) the
// equilibrium m_e(T_c, 1 T) is well-defined, so we now run at the
// experiment's literal operating point.
const T_AMBIENT_K: f32 = 210.0;
const T_C_K: f64 = 210.0;
const B_EXT_T: f64 = 1.0;
const FLUENCE_MJ_CM2: f64 = 0.24;
const PULSE_FWHM_FS: f64 = 150.0;
const FLAKE_THICKNESS_NM: f64 = 5.0;
// F3: optical skin depth fixed at the FGT-at-400nm physics estimate. Kept
// out of the grid because it has a direct measurement; calibration handles
// stay on the dynamic-physics side.
const SKIN_DEPTH_M: f64 = 18.0e-9;

// Fit targets (Zhou 2025 aggregate morphology)
const TARGET_DEMAG_FRAC: f64 = 0.79;
const TARGET_RECOVERY_FRAC: f64 = 0.55;

// Grid axes (post-F1+F2+F3):
// tau_long_base — slow LLB longitudinal stage (was the only LLB knob pre-F2).
// tau_fast_base — fast LLB stage (new in F2; m_target chases m_e on this).
// R              — pulse reflectivity (compensates absorption uncertainty).
// g_sub_phonon  — phonon→substrate sink (controls recovery τ_2).
const TAU_LONG_GRID: &[f64] = &[1.0e-15, 3.0e-15, 10.0e-15];
const TAU_FAST_GRID: &[f64] = &[0.1e-15, 0.3e-15, 1.0e-15];
const R_GRID: &[f32] = &[0.30, 0.50, 0.70];
const G_SUB_GRID: &[f64] = &[1.0e17, 3.0e17, 1.0e18];

#[derive(Clone, Debug)]
struct TrialMetrics {
    tau_long: f64,
    tau_fast: f64,
    r: f32,
    g_sub: f64,
    demag_frac: f64,
    recovery_frac: f64,
    m_initial: f64,
    m_floor: f64,
    m_22ps: f64,
    t_min_ps: f64,
    loss: f64,
}

fn run_trial(tau_long: f64, tau_fast: f64, r: f32, g_sub: f64) -> TrialMetrics {
    let mut preset = material_thermal::fgt_ni_surrogate();
    preset.t_c = T_C_K;
    preset.tau_long_base = tau_long;
    preset.tau_fast_base = tau_fast;
    preset.g_sub_phonon = g_sub;
    preset.optical_skin_depth_m = SKIN_DEPTH_M;
    let (m_e_table, chi_table) =
        material_thermal::brillouin_tables_spin_half_2d(
            preset.t_c, preset.llb_table_n, preset.llb_table_n_b, preset.b_max_t,
        );
    preset.m_e_table = m_e_table;
    preset.chi_par_table = chi_table;

    let mut stack = Stack::monolayer(BulkMaterial::fgt_bulk(), FLAKE_THICKNESS_NM * 1e-9, [0.0, 0.0, 1.0]);
    stack.layers[0].material.alpha_bulk = preset.alpha_0;

    let mut cfg = SimConfig {
        stack,
        substrate: Substrate::vacuum(),
        geometry: Geometry { nx: 1, ny: 1, cell_size: 20e-9 },
        dt: 5.0e-15,
        b_ext: [0.0, 0.0, B_EXT_T],
        stab_coeff: 0.0,
        j_current: [0.0, 0.0, 0.0],
        photonic: Default::default(),
        readback_interval: 1,
        total_steps: 12_000, // 60 ps total
        probe_idx: None,
        probe_layer: None,
    };
    // Pulse peaks at 4 ps so LLB has time to relax to m_e(T_ambient) first.
    let pulse_spec = format!(
        "t=4ps,fwhm={:.0}fs,peak=0.0,dir=z,fluence={:.4},R={:.3}",
        PULSE_FWHM_FS, FLUENCE_MJ_CM2, r
    );
    cfg.photonic.pulses.push(parse_pulse_spec(&pulse_spec).unwrap());
    cfg.photonic.thermal = Some(ThermalConfig {
        t_ambient: T_AMBIENT_K,
        per_layer: vec![preset.clone()],
        thermal_dt_cap: 5e-15,
        thermal_window: (1e-12, 40e-12),
        enable_llb: true,
    });

    let mut solver = GpuSolver::new(&cfg).expect("GPU init");
    solver.reset_uniform_z();

    let mut traj_ps = Vec::with_capacity(cfg.total_steps);
    let mut traj_m = Vec::with_capacity(cfg.total_steps);
    for _ in 0..cfg.total_steps {
        solver.step_n(1);
        let obs = solver.observables();
        traj_ps.push(obs.time_ps);
        traj_m.push(obs.min_norm as f64);
    }

    // Reference: just before the pulse peak (t = 3.5 ps). LLB has had 3.5 ps
    // to relax from the |m| = 1 initial to ≈ m_e(150 K) = 0.85.
    let i_ref = traj_ps.iter().position(|&t| t > 3.5).unwrap_or(0);
    let m_initial = traj_m[i_ref].max(1e-6);
    let (m_floor, i_min) = traj_m.iter().enumerate().skip(i_ref).fold(
        (m_initial, i_ref),
        |(v, i), (j, x)| if *x < v { (*x, j) } else { (v, i) },
    );
    let i_22 = traj_ps
        .iter()
        .position(|&t| t > 4.0 + 22.0)
        .unwrap_or(traj_ps.len() - 1);
    let m_22ps = traj_m[i_22];

    let demag_frac = 1.0 - m_floor / m_initial;
    let recovery_frac = if (m_initial - m_floor).abs() > 1e-9 {
        (m_22ps - m_floor) / (m_initial - m_floor)
    } else {
        0.0
    };

    let loss = (demag_frac - TARGET_DEMAG_FRAC).powi(2)
        + 0.5 * (recovery_frac - TARGET_RECOVERY_FRAC).powi(2);

    TrialMetrics {
        tau_long,
        tau_fast,
        r,
        g_sub,
        demag_frac,
        recovery_frac,
        m_initial,
        m_floor,
        m_22ps,
        t_min_ps: traj_ps[i_min],
        loss,
    }
}

fn main() {
    let n_trials = TAU_LONG_GRID.len() * TAU_FAST_GRID.len() * R_GRID.len() * G_SUB_GRID.len();
    println!("=== FGT Zhou-morphology calibration (F1+F2+F3), grid search ({} trials) ===", n_trials);
    println!(
        "Operating point: T = {} K, B_z = {} T, F = {} mJ/cm², FWHM = {} fs, t = {} nm FGT, δ = {} nm",
        T_AMBIENT_K, B_EXT_T, FLUENCE_MJ_CM2, PULSE_FWHM_FS, FLAKE_THICKNESS_NM, SKIN_DEPTH_M * 1e9,
    );
    println!(
        "Targets:  demag_frac = {:.2}   recovery_frac(22 ps) = {:.2}",
        TARGET_DEMAG_FRAC, TARGET_RECOVERY_FRAC,
    );
    println!();

    let mut results: Vec<TrialMetrics> = Vec::with_capacity(n_trials);
    let t0 = std::time::Instant::now();
    let mut count = 0usize;
    for &tau_long in TAU_LONG_GRID {
        for &tau_fast in TAU_FAST_GRID {
            for &r in R_GRID {
                for &g_sub in G_SUB_GRID {
                    count += 1;
                    let m = run_trial(tau_long, tau_fast, r, g_sub);
                    eprintln!(
                        "[{count:3}/{n_trials}] τ_L={:.1}fs τ_F={:.1}fs R={:.2} g_sub={:.1e} → demag={:.2} rec={:.2} L={:.4}",
                        m.tau_long * 1e15, m.tau_fast * 1e15, m.r, m.g_sub, m.demag_frac, m.recovery_frac, m.loss,
                    );
                    results.push(m);
                }
            }
        }
    }
    results.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap());
    let elapsed = t0.elapsed().as_secs_f64();

    println!();
    println!("Grid search done: {n_trials} trials in {elapsed:.1}s");
    println!();
    println!("Top 5 by loss (lower = closer to Zhou morphology):");
    println!(
        "  {:>7} {:>7} {:>5} {:>10} | {:>6} {:>6} | {:>6} {:>6} {:>6} {:>7} | {:>6}",
        "τ_L[fs]", "τ_F[fs]", "R", "g_sub", "demag", "rec22", "m_ini", "m_min", "m_22", "t_min", "loss"
    );
    for m in results.iter().take(5) {
        println!(
            "  {:>7.2} {:>7.2} {:>5.2} {:>10.2e} | {:>6.3} {:>6.3} | {:>6.3} {:>6.3} {:>6.3} {:>5.2}ps | {:>6.4}",
            m.tau_long * 1e15, m.tau_fast * 1e15, m.r, m.g_sub,
            m.demag_frac, m.recovery_frac,
            m.m_initial, m.m_floor, m.m_22ps, m.t_min_ps,
            m.loss,
        );
    }
    let best = &results[0];
    println!();
    println!("=== BEST FIT ===");
    println!(
        "tau_long_base = {:.3e} s  tau_fast_base = {:.3e} s  R = {:.2}  g_sub = {:.3e} W/(m³·K)",
        best.tau_long, best.tau_fast, best.r, best.g_sub,
    );
    println!(
        "Observed: demag = {:.1} % (target 79%),  recovery@22ps = {:.1} % (target 55%)",
        best.demag_frac * 100.0, best.recovery_frac * 100.0,
    );

    println!();
    println!("--- Ready-to-paste preset update (src/material_thermal.rs) ---");
    println!("// FGT — Zhou 2025 calibrated (T = T_c = {} K, B = 1 T, F = 0.24 mJ/cm² morphology fit, post-F1/F2/F3)", T_AMBIENT_K);
    println!("tau_long_base:    {:.3e},", best.tau_long);
    println!("tau_fast_base:    {:.3e},", best.tau_fast);
    println!("g_sub_phonon:     {:.3e},", best.g_sub);
    println!("optical_skin_depth_m: {:.1e},", SKIN_DEPTH_M);
    println!("(reflectivity {:.2} should be applied on the pulse, not the preset)", best.r);
}
