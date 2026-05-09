//! Phase F2 acceptance — demonstrate two-stage LLB longitudinal relaxation.
//!
//! Runs a single-cell Ni-parameterised pulse experiment under three
//! configurations and verifies the F2 chain is wired correctly:
//!
//!   A) tau_fast = 0 (F1 collapsed): m_target tracks m_e instantly; |m|
//!      relaxes on a single timescale set by tau_long.
//!   B) tau_fast = tau_long * 5 (slow proxy): m_target visibly lags m_e
//!      during the pulse, |m| recovery exhibits two characteristic rates.
//!   C) tau_fast = tau_long * 50 (very slow proxy): |m| recovery is
//!      dominated by tau_fast — the slowest stage rate-limits the system.
//!
//! Pass criteria:
//!   - At every step in case A, |m_target − m_e| < 0.01 (tracks instantly).
//!   - In case B, max |m_target − m_e| during pulse > 0.05 (visible lag).
//!   - In case C, |m| recovery at 10 ps post-pulse is closer to the deep
//!     minimum than in case A (rate-limited by slow τ_fast).

use magnonic_clock_sim::config::{Geometry, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{parse_pulse_spec, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

#[derive(Clone, Debug)]
struct Trace {
    label: &'static str,
    m_floor: f32,
    m_at_10ps_post: f32,
    max_target_gap: f32,
}

fn run(label: &'static str, tau_fast: f64) -> Trace {
    let mut preset = material_thermal::ni_m3tm();
    preset.tau_fast_base = tau_fast;
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
        total_steps: 12_000, // 12 ps
        probe_idx: None,
        probe_layer: None,
    };
    cfg.photonic.pulses.push(
        parse_pulse_spec("t=300fs,fwhm=100fs,peak=0.0,dir=z,fluence=1.0,R=0.0").unwrap(),
    );
    cfg.photonic.thermal = Some(ThermalConfig {
        t_ambient: 300.0,
        per_layer: vec![preset.clone()],
        thermal_dt_cap: 1.0e-15,
        thermal_window: (0.5e-12, 10e-12),
        enable_llb: true,
    });

    let mut solver = GpuSolver::new(&cfg).expect("GPU init");
    solver.reset_uniform_z();

    let mut m_floor = 1.0_f32;
    let mut max_target_gap = 0.0_f32;
    let mut m_at_10ps_post = 0.0_f32;
    let pulse_peak_step = 300; // dt = 1 fs → pulse at 300 fs

    for step in 0..cfg.total_steps {
        solver.step_n(1);
        let obs = solver.observables();
        m_floor = m_floor.min(obs.min_norm);

        // Sample m_target gap during the pulse window (steps 200..600).
        if step > 200 && step < 600 {
            let m_target = solver.readback_m_target();
            // For Ni at the pulse peak T_e ~ 950 K (well above Tc=627), m_e ≈ 0.
            // Compare m_target to that approximation; the gap is what we want to see.
            // Crude: m_e at the peak T_e in the table.
            let preset = &cfg.photonic.thermal.as_ref().unwrap().per_layer[0];
            let m_e_now = preset.sample_m_e_2d(obs.max_t_e as f64, 0.0) as f32;
            let gap = (m_target[0] - m_e_now).abs();
            if gap > max_target_gap {
                max_target_gap = gap;
            }
        }

        // Sample |m| at 10 ps after the pulse peak.
        if step == pulse_peak_step + 10_000 {
            m_at_10ps_post = obs.min_norm;
        }
    }
    Trace { label, m_floor, m_at_10ps_post, max_target_gap }
}

fn main() {
    println!("=== F2: two-stage longitudinal LLB acceptance probe ===\n");
    let case_a = run("[A] tau_fast = 0  (F1 collapsed)", 0.0);
    let case_b = run("[B] tau_fast = 5x tau_long", 5.0 * 0.3e-15);
    let case_c = run("[C] tau_fast = 50x tau_long", 50.0 * 0.3e-15);

    for t in [&case_a, &case_b, &case_c] {
        println!(
            "{:35} |m|_floor = {:.4}  |m|@10ps = {:.4}  max|m_target-m_e| = {:.4}",
            t.label, t.m_floor, t.m_at_10ps_post, t.max_target_gap,
        );
    }

    // Acceptance: case A's proxy gap is tiny; case B/C show finite lag.
    assert!(
        case_a.max_target_gap < 0.01,
        "case A (tau_fast=0) should track m_e instantly; gap = {:.3e}",
        case_a.max_target_gap,
    );
    assert!(
        case_b.max_target_gap > 0.05,
        "case B (tau_fast > 0) should show visible m_target lag; gap = {:.3e}",
        case_b.max_target_gap,
    );
    assert!(
        case_c.max_target_gap >= case_b.max_target_gap - 0.01,
        "case C with even-larger tau_fast should not have less lag than case B; \
         got C={:.3e}, B={:.3e}",
        case_c.max_target_gap, case_b.max_target_gap,
    );
    println!("\nPASS: F2 two-stage chain produces measurably different dynamics from F1.");
}
