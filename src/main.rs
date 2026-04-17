use magnonic_clock_sim::config::SimConfig;
use magnonic_clock_sim::gpu::GpuSolver;

fn main() {
    env_logger::init();

    let mut config = SimConfig::fgt_default(256, 256);

    // Parse basic CLI args
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--nx" => { config.nx = args[i + 1].parse().unwrap(); i += 2; }
            "--ny" => { config.ny = args[i + 1].parse().unwrap(); i += 2; }
            "--steps" => { config.total_steps = args[i + 1].parse().unwrap(); i += 2; }
            "--interval" => { config.readback_interval = args[i + 1].parse().unwrap(); i += 2; }
            "--alpha" => {
                config.material.alpha = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--a-ex" => {
                config.material.a_ex = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--bz" => {
                config.b_ext[2] = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--dt" => { config.dt = args[i + 1].parse().unwrap(); i += 2; }
            "--help" | "-h" => {
                eprintln!("magnonic-sim [options]");
                eprintln!("  --nx N         Grid width  (default 256)");
                eprintln!("  --ny N         Grid height (default 256)");
                eprintln!("  --steps N      Total steps (default 10000)");
                eprintln!("  --interval N   Readback interval (default 100)");
                eprintln!("  --alpha F      Gilbert damping (default 0.01)");
                eprintln!("  --a-ex F       Exchange stiffness J/m (default 1.4e-12)");
                eprintln!("  --bz F         External field z-component [T]");
                eprintln!("  --dt F         Timestep [s] (default 1e-13)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    config.print_summary();

    let mut solver = GpuSolver::new(&config).expect("Failed to initialize GPU solver");

    // CSV header
    println!("step,time_ps,avg_mx,avg_my,avg_mz,min_norm,max_norm,probe_mz");

    // Initial state
    let obs = solver.observables();
    println!("{obs}");

    let batches = config.total_steps / config.readback_interval;
    let t0 = std::time::Instant::now();

    for b in 0..batches {
        solver.step_n(config.readback_interval);
        let obs = solver.observables();
        println!("{obs}");

        // Progress update every 10 batches
        if (b + 1) % 10 == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let steps_done = (b + 1) * config.readback_interval;
            let steps_per_sec = steps_done as f64 / elapsed;
            eprintln!(
                "[{}/{}] {:.0} steps/s | avg_mz={:.4} | |m| in [{:.6}, {:.6}]",
                steps_done, config.total_steps, steps_per_sec,
                obs.avg_mz, obs.min_norm, obs.max_norm,
            );
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "Done: {} steps in {:.2}s ({:.0} steps/s)",
        config.total_steps,
        elapsed,
        config.total_steps as f64 / elapsed,
    );
}
