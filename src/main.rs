use magnonic_clock_sim::config::SimConfig;
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::substrate::Substrate;

fn main() {
    env_logger::init();

    let mut config = SimConfig::fgt_default(256, 256);

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--nx" => { config.geometry.nx = args[i + 1].parse().unwrap(); i += 2; }
            "--ny" => { config.geometry.ny = args[i + 1].parse().unwrap(); i += 2; }
            "--thickness" => {
                // Argument in nm
                let nm: f64 = args[i + 1].parse().unwrap();
                config.geometry.thickness = nm * 1e-9;
                i += 2;
            }
            "--cell-size" => {
                let nm: f64 = args[i + 1].parse().unwrap();
                config.geometry.cell_size = nm * 1e-9;
                i += 2;
            }
            "--material" => {
                let name = &args[i + 1];
                match BulkMaterial::lookup(name) {
                    Some(m) => { config.bulk = m; }
                    None => {
                        eprintln!("Unknown material: {name}. Use --list-materials to see options.");
                        std::process::exit(1);
                    }
                }
                i += 2;
            }
            "--substrate" => {
                let name = &args[i + 1];
                match Substrate::lookup(name) {
                    Some(s) => { config.substrate = s; }
                    None => {
                        eprintln!("Unknown substrate: {name}. Use --list-substrates to see options.");
                        std::process::exit(1);
                    }
                }
                i += 2;
            }
            "--list-materials" => {
                println!("Available bulk materials:");
                for m in BulkMaterial::list_all() {
                    print!("{m}");
                    println!();
                }
                std::process::exit(0);
            }
            "--list-substrates" => {
                println!("Available substrates:");
                for s in Substrate::list_all() {
                    print!("{s}");
                    println!();
                }
                std::process::exit(0);
            }
            "--steps" => { config.total_steps = args[i + 1].parse().unwrap(); i += 2; }
            "--interval" => { config.readback_interval = args[i + 1].parse().unwrap(); i += 2; }
            "--alpha" => {
                // Overrides bulk damping (post-lookup)
                config.bulk.alpha_bulk = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--a-ex" => {
                config.bulk.a_ex_bulk = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--k-u" => {
                config.bulk.k_u_bulk = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ms" => {
                config.bulk.ms_bulk = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--init" => {
                // set later via solver.reset_*(); for now just stash
                std::env::set_var("MAGNONIC_INIT", &args[i + 1]);
                i += 2;
            }
            "--skyrmion-radius" => {
                std::env::set_var("MAGNONIC_SKYRMION_R_NM", &args[i + 1]);
                i += 2;
            }
            "--d-dmi" => {
                // Override substrate DMI (J/m²) — useful for sweeps
                let d: f64 = args[i + 1].parse().unwrap();
                config.substrate.d_dmi_surface = d;
                i += 2;
            }
            "--bx" => { config.b_ext[0] = args[i + 1].parse().unwrap(); i += 2; }
            "--by" => { config.b_ext[1] = args[i + 1].parse().unwrap(); i += 2; }
            "--bz" => { config.b_ext[2] = args[i + 1].parse().unwrap(); i += 2; }
            "--jx" => { config.j_current[0] = args[i + 1].parse().unwrap(); i += 2; }
            "--jy" => { config.j_current[1] = args[i + 1].parse().unwrap(); i += 2; }
            "--jz" => { config.j_current[2] = args[i + 1].parse().unwrap(); i += 2; }
            "--theta-sh" => {
                // Override substrate spin Hall angle
                config.substrate.spin_hall_angle = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--dt" => { config.dt = args[i + 1].parse().unwrap(); i += 2; }
            "--help" | "-h" => {
                eprintln!("magnonic-sim [options]");
                eprintln!();
                eprintln!("Design choices:");
                eprintln!("  --material NAME     Bulk material (default fgt-effective)");
                eprintln!("  --substrate NAME    Substrate (default vacuum)");
                eprintln!("  --thickness NM      Film thickness in nm (default 0.7)");
                eprintln!("  --cell-size NM      In-plane cell size in nm (default 2.5)");
                eprintln!("  --nx N / --ny N     Grid dimensions (default 256)");
                eprintln!();
                eprintln!("Conditions:");
                eprintln!("  --bx F / --by F / --bz F   External field components [T]");
                eprintln!("  --dt F              Timestep [s] (default 1e-13)");
                eprintln!();
                eprintln!("Material overrides (post-lookup):");
                eprintln!("  --alpha F           Bulk Gilbert damping");
                eprintln!("  --a-ex F            Exchange stiffness [J/m]");
                eprintln!("  --k-u F             Uniaxial anisotropy [J/m³]");
                eprintln!("  --ms F              Saturation magnetization [A/m]");
                eprintln!();
                eprintln!("Run control:");
                eprintln!("  --steps N           Total steps (default 10000)");
                eprintln!("  --interval N        Readback interval (default 100)");
                eprintln!();
                eprintln!("Catalog:");
                eprintln!("  --list-materials    Print all available bulk materials");
                eprintln!("  --list-substrates   Print all available substrates");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    config.print_summary();
    config.effective().print_sot(config.j_current, config.geometry.thickness);

    let mut solver = GpuSolver::new(&config).expect("Failed to initialize GPU solver");

    // Handle --init flag
    if let Ok(init_mode) = std::env::var("MAGNONIC_INIT") {
        match init_mode.as_str() {
            "random" => { solver.reset_random(); eprintln!("INIT: random unit vectors"); }
            "stripe" => { solver.reset_stripe_domains(8); eprintln!("INIT: stripe domains"); }
            "uniform" => { solver.reset_uniform_z(); eprintln!("INIT: uniform +z + 5° cone"); }
            "skyrmion" => {
                let default_r = (config.geometry.nx.min(config.geometry.ny) as f64)
                    * config.geometry.cell_size * 1e9 / 6.0;
                let r_nm: f64 = std::env::var("MAGNONIC_SKYRMION_R_NM")
                    .ok().and_then(|s| s.parse().ok()).unwrap_or(default_r);
                solver.reset_skyrmion_seed(r_nm);
                eprintln!("INIT: Néel skyrmion seed, R = {r_nm:.1} nm");
            }
            other => { eprintln!("Unknown --init mode: {other}"); std::process::exit(1); }
        }
    }

    println!("step,time_ps,avg_mx,avg_my,avg_mz,min_norm,max_norm,probe_mz");

    let obs = solver.observables();
    println!("{obs}");

    let batches = config.total_steps / config.readback_interval;
    let t0 = std::time::Instant::now();

    for b in 0..batches {
        solver.step_n(config.readback_interval);
        let obs = solver.observables();
        println!("{obs}");

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
