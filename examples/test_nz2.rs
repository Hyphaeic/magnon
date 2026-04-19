use magnonic_clock_sim::config::{Geometry, Layer, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::substrate::Substrate;

fn main() {
    env_logger::init();

    let fgt = BulkMaterial::fgt_bulk();
    let stack = Stack {
        layers: vec![
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [0.0, 0.0, 1.0] },
            Layer { material: fgt.clone(), thickness: 0.7e-9, u_axis: [0.0, 0.0, 1.0] },
        ],
        interlayer_a: vec![0.0],        // ZERO interlayer coupling
        layer_spacing: vec![0.5e-9],
    };

    let config = SimConfig {
        stack,
        substrate: Substrate::vacuum(),
        geometry: Geometry::thin_2d(32, 32),
        dt: 1.0e-14,
        b_ext: [0.0, 0.0, 0.0],
        stab_coeff: 1.0e11,
        j_current: [0.0, 0.0, 0.0],
        photonic: Default::default(),
        readback_interval: 500,
        total_steps: 3000,
        probe_idx: None,
        probe_layer: None,
    };

    config.print_summary();

    let mut solver = GpuSolver::new(&config).expect("init");
    println!("step,time_ps,avg_mx,avg_my,avg_mz,min_norm,max_norm,probe_mz");
    let obs = solver.observables();
    println!("{obs}");
    for _ in 0..(config.total_steps / config.readback_interval) {
        solver.step_n(config.readback_interval);
        println!("{}", solver.observables());
    }
}
