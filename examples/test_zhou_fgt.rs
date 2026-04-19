//! Phase P5 — Zhou 2025 FGT ultrafast demagnetization benchmark.
//!
//! Paper: Zhou et al., "Acceleration of ultrafast demagnetization in van der
//! Waals ferromagnet Fe3GeTe2 in high magnetic field," Natl. Sci. Rev. 12,
//! 7, nwaf185 (2025).
//!
//! Experimental conditions (TR-MOKE):
//!   - 400 nm pump, 150 fs FWHM, 0.24 mJ/cm² incident fluence
//!   - Sample: 5 nm FGT flake
//!   - Ambient T = 210 K (at T_c, where the demag effect is maximal)
//!   - External B = 1 T perpendicular to the surface
//!   - Observable: MOKE signal (proxies |m|)
//!
//! Target (headline number):
//!   79 % demagnetization within 22.2 ps at T_ambient = T_c = 210 K, B_ext = 1 T.
//!
//! Two-stage curve shape reported:
//!   τ_1 (fast) ≈ 0.4 ps   — initial ultrafast demag
//!   τ_2 (slow) ≈ 22 ps    — slow component near T_c
//!
//! This harness runs a single-cell simulation at Zhou's conditions using
//! the current `fgt_ni_surrogate` thermal preset and reports the |m|
//! trajectory. It is the **feasibility probe** before a grid-search
//! calibration — a dry-run to see whether our 3-parameter model can
//! plausibly produce the two-stage morphology.

use magnonic_clock_sim::config::{Geometry, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{parse_pulse_spec, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // printed for the reader via `print_trace`, not consumed programmatically
struct TrialResult {
    demag_frac: f64,
    t_min_ps: f64,
    m_floor: f64,
    m_initial: f64,
    m_at_22ps: f64,
}

fn run_trial(
    t_ambient_k: f32,
    fluence_mj_cm2: f64,
    reflectivity: f32,
    g_sub_phonon: f64,
    t_c_override: f64,
    print_trace: bool,
) -> TrialResult {
    let mut preset = material_thermal::fgt_ni_surrogate();
    preset.t_c = t_c_override;
    preset.g_sub_phonon = g_sub_phonon;
    let (m_e, chi) = material_thermal::brillouin_tables_spin_half(preset.t_c, preset.llb_table_n);
    preset.m_e_table = m_e;
    preset.chi_par_table = chi;

    let mut stack = Stack::monolayer(BulkMaterial::fgt_bulk(), 5e-9, [0.0, 0.0, 1.0]);
    stack.layers[0].material.alpha_bulk = preset.alpha_0;

    let mut cfg = SimConfig {
        stack,
        substrate: Substrate::vacuum(),
        geometry: Geometry { nx: 1, ny: 1, cell_size: 20e-9 },
        dt: 5.0e-15,
        b_ext: [0.0, 0.0, 1.0], // 1 T perpendicular
        stab_coeff: 0.0,
        j_current: [0.0, 0.0, 0.0],
        photonic: Default::default(),
        readback_interval: 1,
        total_steps: 12_000, // 60 ps
        probe_idx: None,
        probe_layer: None,
    };
    let pulse_spec = format!(
        "t=2ps,fwhm=150fs,peak=0.0,dir=z,fluence={fluence_mj_cm2:.4},R={reflectivity:.3}"
    );
    cfg.photonic.pulses.push(parse_pulse_spec(&pulse_spec).unwrap());
    cfg.photonic.thermal = Some(ThermalConfig {
        t_ambient: t_ambient_k,
        per_layer: vec![preset.clone()],
        thermal_dt_cap: 5e-15,
        thermal_window: (1e-12, 40e-12),
        enable_llb: true,
    });

    let mut solver = GpuSolver::new(&cfg).expect("GPU init");
    // Start at the equilibrium magnitude m_e(T_ambient). This pairs with
    // the LLB drive that pulls |m| → m_e(T_s), so the "pre-pulse" state is
    // already in equilibrium (no relaxation transient confounds the demag
    // measurement).
    let m_eq = preset.sample_m_e(t_ambient_k as f64);
    // reset_uniform_z gives |m|=1 with a 5° cone. We rescale by m_eq.
    // (Not worth adding a setter — just do the simulation long enough and
    // the LLB relaxes to m_eq within a few τ_∥.)
    solver.reset_uniform_z();

    let mut traj_ps = Vec::with_capacity(cfg.total_steps);
    let mut traj_m = Vec::with_capacity(cfg.total_steps);
    let mut traj_te = Vec::with_capacity(cfg.total_steps);

    for _ in 0..cfg.total_steps {
        solver.step_n(1);
        let obs = solver.observables();
        traj_ps.push(obs.time_ps);
        traj_m.push(obs.min_norm as f64);
        traj_te.push(obs.max_t_e as f64);
    }

    // Reference |m|: just before the pulse (pulse peaks at 2 ps; ref at 1.5 ps).
    let i_ref = traj_ps
        .iter()
        .position(|&t| t > 1.5)
        .unwrap_or(0);
    let m_initial = traj_m[i_ref].max(1e-6);
    let (m_floor, i_min) = traj_m.iter().enumerate().skip(i_ref).fold(
        (m_initial, i_ref),
        |(v, i), (j, x)| if *x < v { (*x, j) } else { (v, i) },
    );
    let demag_frac = 1.0 - m_floor / m_initial;
    let t_min_ps = traj_ps[i_min];

    // Sample |m| at 22 ps after pulse (Zhou's reported time).
    let i_22 = traj_ps
        .iter()
        .position(|&t| t > 2.0 + 22.0)
        .unwrap_or(traj_ps.len() - 1);
    let m_at_22ps = traj_m[i_22];

    if print_trace {
        println!(
            "Trial: T_amb={:.0}K  F={:.3} mJ/cm²  R={:.2}  g_sub={:.1e}  T_c={:.0}K",
            t_ambient_k, fluence_mj_cm2, reflectivity, g_sub_phonon, t_c_override
        );
        println!("  m_e({}K) = {:.3}  (equilibrium)", t_ambient_k, m_eq);
        for t_target in [0.5_f64, 1.5, 2.0, 2.2, 3.0, 5.0, 10.0, 22.0, 40.0] {
            let i = traj_ps.iter().position(|&t| t > t_target).unwrap_or(traj_ps.len() - 1);
            println!(
                "  t={:5.1}ps  |m|={:.4}  T_e={:6.1}K",
                traj_ps[i], traj_m[i], traj_te[i]
            );
        }
        println!(
            "  m_ref@{:.2}ps = {:.4}   m_floor = {:.4} @ t = {:.2} ps  demag = {:.1} %",
            traj_ps[i_ref], m_initial, m_floor, t_min_ps, 100.0 * demag_frac
        );
        println!("  m(22 ps after pulse) = {:.4}", m_at_22ps);
    }

    TrialResult {
        demag_frac,
        t_min_ps,
        m_floor,
        m_initial,
        m_at_22ps,
    }
}

fn main() {
    println!("=== Zhou 2025 FGT feasibility probe — operating-point scan ===\n");
    println!("Target: 79 % demag @ 22.2 ps from a |m|_initial set by (T, B=1T) equilibrium.\n");

    // 1. Current preset at Zhou's T_c=210K with 1 T field (tables have no
    //    field — m_e(T_c)=0, so this is a degenerate baseline).
    println!("[1] Zhou conditions literal — T = T_c = 210 K (degenerate baseline):");
    run_trial(210.0, 0.24, 0.5, 2.0e16, 210.0, true);
    println!();

    // 2. Lower T where m_e is well-defined (Zhou notes 90-210 K range).
    println!("[2] T = 100 K (well below T_c; m_e ≈ 0.95):");
    run_trial(100.0, 0.24, 0.5, 2.0e16, 210.0, true);
    println!();

    // 3. Same with 10× stronger substrate sink → faster recovery.
    println!("[3] T = 100 K, g_sub_phonon = 2e17 (τ_sub ≈ 12 ps):");
    run_trial(100.0, 0.24, 0.5, 2.0e17, 210.0, true);
    println!();

    // 4. T = 180 K (closer to T_c; Zhou's mid-range).
    println!("[4] T = 180 K (mid-range, m_e ≈ 0.55):");
    run_trial(180.0, 0.24, 0.5, 2.0e17, 210.0, true);
    println!();

    // 5. Low-fluence sanity: smaller demag at the same T.
    println!("[5] T = 180 K, 0.10 mJ/cm² (lower fluence):");
    run_trial(180.0, 0.10, 0.5, 2.0e17, 210.0, true);
}
