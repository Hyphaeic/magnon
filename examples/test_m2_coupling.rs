//! Phase M2 validation: interlayer exchange coupling.
//!
//! Three tests:
//!   A. Nz=1 regression — single layer must behave the same as pre-M2.
//!   B. Nz=2 with J_inter = 0 — decoupled (baseline).
//!   C. Nz=2 with J_inter > 0 — strong coupling locks layers in phase.
//!
//! We probe this by applying an asymmetric pulse to layer 0 only (actually,
//! applying a tilt to one layer via random perturbation), then letting the
//! system evolve. With coupling, layer 1 follows layer 0. Without coupling,
//! they evolve independently.

use magnonic_clock_sim::config::{Geometry, Layer, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::substrate::Substrate;

fn run_case(
    label: &str,
    stack: Stack,
    steps: usize,
    interval: usize,
) {
    let config = SimConfig {
        stack,
        substrate: Substrate::vacuum(),
        geometry: Geometry::thin_2d(32, 32),
        dt: 1.0e-14,
        b_ext: [0.0, 0.0, 0.0],
        stab_coeff: 1.0e11,
        j_current: [0.0, 0.0, 0.0],
        readback_interval: interval,
        total_steps: steps,
        probe_idx: None,
        probe_layer: None,
    };
    println!("\n══════════════════════════════════════════");
    println!("CASE: {label}");
    println!("══════════════════════════════════════════");
    config.print_summary();
    let mut solver = GpuSolver::new(&config).expect("init");
    println!("step,time_ps,avg_mx,avg_my,avg_mz,min_norm,max_norm,probe_mz");
    let obs = solver.observables();
    println!("{obs}");
    let batches = steps / interval;
    for _ in 0..batches {
        solver.step_n(interval);
        println!("{}", solver.observables());
    }
}

fn main() {
    env_logger::init();
    let fgt = BulkMaterial::fgt_bulk();

    // Case A: Nz=1 (regression)
    let stack_a = Stack::monolayer(fgt.clone(), 0.7e-9, [0.0, 0.0, 1.0]);
    run_case("A — Nz=1 regression (fgt-bulk monolayer)", stack_a, 3000, 500);

    // Case B: Nz=2, J_inter = 0 (decoupled)
    let stack_b = Stack {
        layers: vec![
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [0.0, 0.0, 1.0] },
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [0.0, 0.0, 1.0] },
        ],
        interlayer_a: vec![0.0],            // ZERO
        layer_spacing: vec![0.5e-9],
    };
    run_case("B — Nz=2 decoupled (J_inter = 0)", stack_b, 3000, 500);

    // Case C: Nz=2, moderate ferromagnetic J_inter.
    // Target prefactor ≈ half of B_anis (7.77 T → want ~4 T).
    // With Ms=3.76e5, dz=0.7nm: A_inter = 4·Ms·dz²/2 ≈ 4e-13 J/m gives ~4.3 T.
    let stack_c = Stack {
        layers: vec![
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [0.0, 0.0, 1.0] },
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [0.0, 0.0, 1.0] },
        ],
        interlayer_a: vec![4.0e-13],        // moderate ferromagnetic
        layer_spacing: vec![0.7e-9],
    };
    run_case("C — Nz=2 ferromagnetic coupled (J_inter = +4e-13 J/m)", stack_c, 3000, 500);

    // Case D: Nz=2, antiferromagnetic coupling.
    // Same magnitude but negative. Ground state should have alternating layers.
    let stack_d = Stack {
        layers: vec![
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [0.0, 0.0, 1.0] },
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [0.0, 0.0, 1.0] },
        ],
        interlayer_a: vec![-4.0e-13],       // antiferromagnetic
        layer_spacing: vec![0.7e-9],
    };
    run_case("D — Nz=2 antiferromagnetic coupled (J_inter = -4e-13 J/m)", stack_d, 3000, 500);

    // Case E: CROSSED anisotropy axes. Layer 0 prefers +z, Layer 1 prefers +x.
    // Without coupling: each layer relaxes to its own easy axis → average
    // of (+z) + (+x) normalized → avg_mz ≈ 0.5, avg_mx ≈ 0.5.
    // With strong coupling: layers forced into compromise → common direction,
    // avg_mz and avg_mx both shift together toward one axis or intermediate.
    let stack_e0 = Stack {
        layers: vec![
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [0.0, 0.0, 1.0] }, // +z easy
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [1.0, 0.0, 0.0] }, // +x easy
        ],
        interlayer_a: vec![0.0],
        layer_spacing: vec![0.7e-9],
    };
    run_case("E0 — Crossed easy axes, DECOUPLED (J_inter = 0)", stack_e0, 5000, 500);

    let stack_e1 = Stack {
        layers: vec![
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [0.0, 0.0, 1.0] },
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [1.0, 0.0, 0.0] },
        ],
        interlayer_a: vec![4.0e-13],
        layer_spacing: vec![0.7e-9],
    };
    run_case("E1 — Crossed easy axes, FM COUPLED (J_inter = +4e-13 J/m)", stack_e1, 5000, 500);

    println!("\n══════════════════════════════════════════");
    println!("M2 validation complete.");
    println!("══════════════════════════════════════════");
}
