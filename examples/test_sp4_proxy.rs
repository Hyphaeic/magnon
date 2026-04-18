//! Phase P3c — µMAG SP4 proxy: deterministic LLB=LLG regression gate.
//!
//! The literal µMAG Standard Problem 4 requires a demagnetization field —
//! this simulator has no demag (explicitly out-of-scope per `docs/plan.md`
//! §5.7). As a substitute LLG regression gate we use a **deterministic
//! skyrmion-seed initial condition** driven by a small transverse bias
//! field, run the LLG path and the LLB-at-T=0 path, and compare their
//! in-plane dynamics (avg_mx, avg_my, avg_mz) point-by-point. At T=0 K
//! the LLB longitudinal rate is exactly zero so the two paths must
//! produce identical trajectories to float precision. ADR-004 documents
//! this SP4 substitution.
//!
//! The skyrmion seed (`reset_skyrmion_seed`) uses no RNG; the test is
//! fully deterministic on both paths.

use magnonic_clock_sim::config::SimConfig;
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::ThermalConfig;

fn run_sp4_proxy(enable_llb: bool) -> Vec<[f32; 3]> {
    let mut cfg = SimConfig::fgt_default(32, 32);
    cfg.dt = 1.0e-14;
    cfg.total_steps = 2000;
    // Small transverse bias — drives precession without spin-flipping.
    cfg.b_ext = [0.01, 0.0, 0.0];

    // Engage the LLB path with T_ambient = 0 K and a_sf = 0 so neither
    // longitudinal relaxation nor M3TM source contribute. Preset α_0
    // matches bulk material α for apples-to-apples transverse damping.
    let mut preset = material_thermal::fgt_ni_surrogate();
    preset.a_sf = 0.0;
    preset.alpha_0 = cfg.stack.layers[0].material.alpha_bulk;
    cfg.photonic.thermal = Some(ThermalConfig {
        t_ambient: 0.0,
        per_layer: vec![preset],
        thermal_dt_cap: 1.0e-15,
        thermal_window: (0.0, 0.0),
        enable_llb,
    });

    let mut solver = GpuSolver::new(&cfg).expect("GPU init");
    solver.reset_skyrmion_seed(12.0); // deterministic seed (no RNG)

    let mut trace = Vec::with_capacity(cfg.total_steps / 10);
    for batch in 0..(cfg.total_steps / 10) {
        solver.step_n(10);
        let obs = solver.observables();
        trace.push([obs.avg_mx, obs.avg_my, obs.avg_mz]);
        let _ = batch;
    }
    trace
}

fn main() {
    let llg = run_sp4_proxy(false);
    let llb = run_sp4_proxy(true);
    assert_eq!(llg.len(), llb.len());

    let mut max_abs_mx = 0.0_f32;
    let mut max_abs_my = 0.0_f32;
    let mut max_abs_mz = 0.0_f32;
    let mut at_step = 0usize;
    for (i, (a, b)) in llg.iter().zip(llb.iter()).enumerate() {
        let dx = (a[0] - b[0]).abs();
        let dy = (a[1] - b[1]).abs();
        let dz = (a[2] - b[2]).abs();
        if dx > max_abs_mx { max_abs_mx = dx; at_step = i; }
        max_abs_my = max_abs_my.max(dy);
        max_abs_mz = max_abs_mz.max(dz);
    }
    let llg_end = llg.last().unwrap();
    let llb_end = llb.last().unwrap();
    println!(
        "LLG final (avg_mx, avg_my, avg_mz) = ({:.6}, {:.6}, {:.6})",
        llg_end[0], llg_end[1], llg_end[2]
    );
    println!(
        "LLB final (avg_mx, avg_my, avg_mz) = ({:.6}, {:.6}, {:.6})",
        llb_end[0], llb_end[1], llb_end[2]
    );
    println!(
        "max |ΔM| components: mx={:.3e} my={:.3e} mz={:.3e} (worst at sample {})",
        max_abs_mx, max_abs_my, max_abs_mz, at_step
    );

    // Tight gate: 1e-3 absolute per component (plan's P3b spec).
    // With deterministic skyrmion init and matched α, we expect <1e-4.
    for (name, val) in [("mx", max_abs_mx), ("my", max_abs_my), ("mz", max_abs_mz)] {
        assert!(
            val < 1.0e-3,
            "SP4-proxy: LLG vs LLB {name} diverge ({val:.3e} > 1e-3)"
        );
    }
    println!("PASS: SP4-proxy LLB=LLG @ T=0 gate — all components within 1e-3.");
}
