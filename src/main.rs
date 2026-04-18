use magnonic_clock_sim::config::{Layer, SimConfig};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{parse_pulse_spec, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

/// Parse a `--stack` spec like "fgt-bulk:0.7,yig:2.0,fgt-bulk:0.7" into Layers.
fn parse_stack(spec: &str) -> Result<Vec<Layer>, String> {
    spec.split(',')
        .map(|entry| {
            let parts: Vec<&str> = entry.trim().split(':').collect();
            if parts.len() != 2 {
                return Err(format!(
                    "Bad layer spec '{entry}' — expected MATERIAL:THICKNESS_NM"
                ));
            }
            let name = parts[0].trim();
            let thickness_nm: f64 = parts[1].trim().parse()
                .map_err(|e| format!("Bad thickness in '{entry}': {e}"))?;
            let material = BulkMaterial::lookup(name)
                .ok_or_else(|| format!("Unknown material '{name}' — see --list-materials"))?;
            Ok(Layer {
                material,
                thickness: thickness_nm * 1e-9,
                u_axis: [0.0, 0.0, 1.0],
            })
        })
        .collect()
}

/// Parse a comma-separated list of f64 values.
fn parse_f64_list(s: &str) -> Result<Vec<f64>, String> {
    s.split(',')
        .map(|x| x.trim().parse::<f64>().map_err(|e| format!("'{x}': {e}")))
        .collect()
}

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
                // Sets thickness of the FIRST layer (backward compat).
                // For multilayer, use --stack (M3).
                let nm: f64 = args[i + 1].parse().unwrap();
                config.stack.layers[0].thickness = nm * 1e-9;
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
                    Some(m) => { config.stack.layers[0].material = m; }
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
            "--stack" => {
                // Multi-layer stack spec: "MAT1:T_nm,MAT2:T_nm,..."
                match parse_stack(&args[i + 1]) {
                    Ok(layers) => {
                        let n_interfaces = layers.len().saturating_sub(1);
                        config.stack.layers = layers;
                        // Resize coupling arrays to match (fill with zeros / default spacing)
                        config.stack.interlayer_a.resize(n_interfaces, 0.0);
                        config.stack.layer_spacing.resize(n_interfaces, 0.7e-9);
                    }
                    Err(e) => {
                        eprintln!("Error parsing --stack: {e}");
                        std::process::exit(1);
                    }
                }
                i += 2;
            }
            "--interlayer-a" => {
                // Comma-separated A_inter values [J/m], one per interface.
                match parse_f64_list(&args[i + 1]) {
                    Ok(vals) => {
                        let expected = config.stack.layers.len().saturating_sub(1);
                        if vals.len() != expected {
                            eprintln!(
                                "--interlayer-a has {} values, expected {} (N_layers - 1)",
                                vals.len(), expected
                            );
                            std::process::exit(1);
                        }
                        config.stack.interlayer_a = vals;
                    }
                    Err(e) => { eprintln!("--interlayer-a parse error: {e}"); std::process::exit(1); }
                }
                i += 2;
            }
            "--layer-spacing" => {
                // Comma-separated spacings [nm], one per interface.
                match parse_f64_list(&args[i + 1]) {
                    Ok(vals_nm) => {
                        let expected = config.stack.layers.len().saturating_sub(1);
                        if vals_nm.len() != expected {
                            eprintln!(
                                "--layer-spacing has {} values, expected {} (N_layers - 1)",
                                vals_nm.len(), expected
                            );
                            std::process::exit(1);
                        }
                        config.stack.layer_spacing = vals_nm.iter().map(|nm| nm * 1e-9).collect();
                    }
                    Err(e) => { eprintln!("--layer-spacing parse error: {e}"); std::process::exit(1); }
                }
                i += 2;
            }
            "--probe-layer" => {
                config.probe_layer = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--pulse" => {
                match parse_pulse_spec(&args[i + 1]) {
                    Ok(pulse) => { config.photonic.pulses.push(pulse); }
                    Err(e) => {
                        eprintln!("--pulse parse error: {e}");
                        std::process::exit(1);
                    }
                }
                i += 2;
            }
            "--enable-thermal" => {
                // Start with a minimal default ThermalConfig; per-layer
                // presets must be supplied via --thermal-params-for.
                if config.photonic.thermal.is_none() {
                    config.photonic.thermal = Some(ThermalConfig::default());
                }
                i += 1;
            }
            "--enable-llb" => {
                let t = config.photonic.thermal.get_or_insert_with(ThermalConfig::default);
                t.enable_llb = true;
                i += 1;
            }
            "--t-ambient" => {
                let t = config.photonic.thermal.get_or_insert_with(ThermalConfig::default);
                t.t_ambient = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--thermal-dt-cap" => {
                let t = config.photonic.thermal.get_or_insert_with(ThermalConfig::default);
                t.thermal_dt_cap = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--thermal-params-for" => {
                // Apply preset to every layer in the stack.
                let key = args[i + 1].to_string();
                let preset = match material_thermal::for_key(&key) {
                    Some(p) => p,
                    None => {
                        eprintln!("Unknown --thermal-params-for key: {key}");
                        std::process::exit(1);
                    }
                };
                let t = config.photonic.thermal.get_or_insert_with(ThermalConfig::default);
                let n_layers = config.stack.layers.len();
                t.per_layer = vec![preset; n_layers];
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
                // Overrides alpha of the first layer's bulk material
                config.stack.layers[0].material.alpha_bulk = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--a-ex" => {
                config.stack.layers[0].material.a_ex_bulk = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--k-u" => {
                config.stack.layers[0].material.k_u_bulk = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ms" => {
                config.stack.layers[0].material.ms_bulk = args[i + 1].parse().unwrap();
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
                eprintln!("Monolayer design choices:");
                eprintln!("  --material NAME     Bulk material (default fgt-bulk)");
                eprintln!("  --substrate NAME    Substrate (default vacuum)");
                eprintln!("  --thickness NM      First-layer thickness in nm (default 0.7)");
                eprintln!("  --cell-size NM      In-plane cell size in nm (default 2.5)");
                eprintln!("  --nx N / --ny N     Grid dimensions (default 256)");
                eprintln!();
                eprintln!("Multilayer stack (M3):");
                eprintln!("  --stack \"MAT1:T_nm,MAT2:T_nm,...\"");
                eprintln!("                      Specify a heterostructure. Overrides --material and --thickness.");
                eprintln!("                      Example: \"yig:2.0,fgt-bulk:0.7\"");
                eprintln!("  --interlayer-a F1,F2,...     Per-interface exchange A [J/m], N_layers-1 values");
                eprintln!("  --layer-spacing NM1,NM2,...  Per-interface z-spacing [nm], N_layers-1 values (default 0.7)");
                eprintln!("  --probe-layer N     Layer index to probe (default 0 = bottom)");
                eprintln!();
                eprintln!("Conditions:");
                eprintln!("  --bx F / --by F / --bz F   External field components [T]");
                eprintln!("  --jx F / --jy F / --jz F   Charge current [A/m²] (SOT drive of layer 0)");
                eprintln!("  --dt F              Timestep [s] (default 1e-14)");
                eprintln!("  --init MODE         uniform | random | stripe | skyrmion | alternating");
                eprintln!();
                eprintln!("Photonic drive (Phase P1-P2 — IFE laser pulses):");
                eprintln!("  --pulse \"t=T,fwhm=W,peak=B,dir=D[,x=X,y=Y,sigma=S]\"");
                eprintln!("                      Laser pulse: t = center time (ps default, or fs/ns/s),");
                eprintln!("                      fwhm = FWHM duration (fs default), peak = IFE field [T],");
                eprintln!("                      dir = x/y/z/-x/-y/-z or 'a,b,c' 3-vec.");
                eprintln!("                      Optional spatial (P2): x, y = spot center [nm default,");
                eprintln!("                      accepts um/mm/m suffix], sigma = 1-σ beam radius.");
                eprintln!("                      Omit spatial keys or set sigma=0 for uniform illumination.");
                eprintln!("                      Max 4 pulses. Example (focused):");
                eprintln!("                      \"t=1ps,fwhm=100fs,peak=0.5T,dir=z,x=40nm,y=40nm,sigma=25nm\"");
                eprintln!();
                eprintln!("Material overrides (apply to first layer post-lookup):");
                eprintln!("  --alpha F           Bulk Gilbert damping");
                eprintln!("  --a-ex F            Exchange stiffness [J/m]");
                eprintln!("  --k-u F             Uniaxial anisotropy [J/m³]");
                eprintln!("  --ms F              Saturation magnetization [A/m]");
                eprintln!("  --d-dmi F           Substrate DMI override [J/m²]");
                eprintln!("  --theta-sh F        Substrate spin Hall angle");
                eprintln!();
                eprintln!("Run control:");
                eprintln!("  --steps N           Total steps (default 10000)");
                eprintln!("  --interval N        Readback interval (default 100)");
                eprintln!();
                eprintln!("Catalog:");
                eprintln!("  --list-materials    Print all available bulk materials");
                eprintln!("  --list-substrates   Print all available substrates");
                eprintln!();
                eprintln!("Example — YIG/FGT ferromagnetically coupled bilayer:");
                eprintln!("  magnonic-sim --stack \"yig:2.0,fgt-bulk:0.7\" \\");
                eprintln!("               --interlayer-a 4e-13 --layer-spacing 1.35 \\");
                eprintln!("               --substrate pt --dt 1e-14 --steps 10000");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    config.print_summary();
    config.effective().print_sot(config.j_current);

    let mut solver = GpuSolver::new(&config).expect("Failed to initialize GPU solver");

    // Handle --init flag
    if let Ok(init_mode) = std::env::var("MAGNONIC_INIT") {
        match init_mode.as_str() {
            "random" => { solver.reset_random(); eprintln!("INIT: random unit vectors"); }
            "stripe" => { solver.reset_stripe_domains(8); eprintln!("INIT: stripe domains"); }
            "uniform" => { solver.reset_uniform_z(); eprintln!("INIT: uniform +z + 5° cone"); }
            "alternating" => {
                solver.reset_uniform_z_alternating();
                eprintln!("INIT: alternating +z/-z per layer (synthetic AFM)");
            }
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

    println!("step,time_ps,avg_mx,avg_my,avg_mz,min_norm,max_norm,probe_mz,max_t_e,max_t_p,min_m_reduced");

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
