//! Parametric sweep harness — iterates over (material × substrate × thickness × conditions),
//! runs a standard pulse-and-measure protocol per design point, emits CSV for phase-diagram analysis.
//!
//! Measurement protocol per point:
//!   1. Relax initial state for N_relax steps (let system find equilibrium).
//!   2. Apply a transverse B_x pulse for N_pulse steps to excite precession.
//!   3. Sample probe_mz every M steps for (M × N_sample) steps of free decay.
//!   4. FFT the samples to extract peak frequency, envelope-fit for decay τ.
//!
//! Output rows: physical design choices + derived effective params + clock metrics.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use magnonic_clock_sim::config::{Geometry, Layer, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::metrics::{analyze_time_series, ClockMetrics};
use magnonic_clock_sim::photonic::{LaserPulse, PhotonicConfig, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

struct SweepArgs {
    materials: Vec<String>,
    substrates: Vec<String>,
    thicknesses_nm: Vec<f64>,
    bz_values: Vec<f64>,
    jx_values: Vec<f64>,

    /// If Some, overrides materials × thicknesses and uses the given stack
    /// (spec format: "MAT1:T_nm,MAT2:T_nm,...") for every design point.
    stack_spec: Option<String>,
    /// Comma-separated interlayer A [J/m], one per interface.
    stack_interlayer_a: Option<Vec<f64>>,
    /// Comma-separated layer spacing [nm], one per interface (default 0.7 per interface).
    stack_spacing_nm: Option<Vec<f64>>,

    nx: u32,
    ny: u32,
    cell_size_nm: f64,
    dt_ps: f64,

    // Protocol
    relax_steps: usize,
    pulse_strength_t: f32,
    pulse_duration_steps: usize,
    sample_interval_steps: usize,
    num_samples: usize,

    output_path: String,

    // ─── Phase P4 — pump-probe sequencer ──────────────────
    /// When true, the sweep iterates over pump-probe delays rather than
    /// material × substrate × thickness × bz × jx. Axes other than the
    /// delay collapse to their first value each.
    pump_probe_mode: bool,
    /// Pump center time [s] (pulses emit their temporal Gaussian around this).
    pump_t_center: f64,
    /// Pump FWHM duration [s].
    pump_fwhm: f64,
    /// Pump peak IFE-equivalent B field [T].
    pump_peak_t: f32,
    /// Optional pump fluence [J/m²] — engages M3TM source term on pump.
    pump_fluence_j_m2: Option<f64>,
    /// Pump reflectivity (0..1).
    pump_reflectivity: f32,
    /// Probe peak IFE-equivalent B field [T].
    probe_peak_t: f32,
    /// Probe FWHM duration [s].
    probe_fwhm: f64,
    /// Optional probe fluence [J/m²].
    probe_fluence_j_m2: Option<f64>,
    /// Probe reflectivity (0..1).
    probe_reflectivity: f32,
    /// Pump-probe delay axis: (start_ps, end_ps, steps).
    pump_probe_delay_range: Option<(f64, f64, usize)>,
    /// Thermal preset key applied to every layer when `enable_thermal` is true.
    thermal_preset_key: Option<String>,
    /// When true, a ThermalConfig is attached to every design point.
    enable_thermal: bool,
    /// When true, `ThermalConfig.enable_llb = true` (implies `enable_thermal`).
    enable_llb: bool,
    /// Ambient / starting temperature [K] for the thermal baths.
    thermal_t_ambient: f32,
}

impl Default for SweepArgs {
    fn default() -> Self {
        Self {
            materials: vec!["fgt-bulk".to_string(), "fgt-effective".to_string()],
            substrates: vec!["vacuum".to_string(), "pt".to_string(), "wte2".to_string(), "yig".to_string()],
            thicknesses_nm: vec![0.7, 2.1],
            bz_values: vec![0.0],
            jx_values: vec![0.0],
            nx: 32,
            ny: 32,
            cell_size_nm: 2.5,
            dt_ps: 0.1,
            relax_steps: 5000,
            pulse_strength_t: 1.0,
            pulse_duration_steps: 100,
            sample_interval_steps: 20,
            num_samples: 1024,
            output_path: "sweep.csv".to_string(),
            stack_spec: None,
            stack_interlayer_a: None,
            stack_spacing_nm: None,
            // P4 pump-probe defaults
            pump_probe_mode: false,
            pump_t_center: 200e-12,
            pump_fwhm: 100e-15,
            pump_peak_t: 0.5,
            pump_fluence_j_m2: None,
            pump_reflectivity: 0.0,
            probe_peak_t: 0.1,
            probe_fwhm: 100e-15,
            probe_fluence_j_m2: None,
            probe_reflectivity: 0.0,
            pump_probe_delay_range: None,
            thermal_preset_key: None,
            enable_thermal: false,
            enable_llb: false,
            thermal_t_ambient: 300.0,
        }
    }
}

/// Parse a stack spec "MAT1:T_nm,MAT2:T_nm,..." into Vec<Layer>.
fn parse_stack_spec(spec: &str) -> Result<Vec<Layer>, String> {
    spec.split(',').map(|entry| {
        let parts: Vec<&str> = entry.trim().split(':').collect();
        if parts.len() != 2 {
            return Err(format!("Bad layer spec '{entry}'"));
        }
        let name = parts[0].trim();
        let thickness_nm: f64 = parts[1].trim().parse().map_err(|e| format!("{e}"))?;
        let material = BulkMaterial::lookup(name)
            .ok_or_else(|| format!("Unknown material '{name}'"))?;
        Ok(Layer { material, thickness: thickness_nm * 1e-9, u_axis: [0.0, 0.0, 1.0] })
    }).collect()
}

fn parse_list_f64(s: &str) -> Vec<f64> {
    s.split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

fn parse_list_str(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

fn parse_args() -> SweepArgs {
    let mut args = SweepArgs::default();
    let raw: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--materials" => { args.materials = parse_list_str(&raw[i + 1]); i += 2; }
            "--substrates" => { args.substrates = parse_list_str(&raw[i + 1]); i += 2; }
            "--thicknesses" => { args.thicknesses_nm = parse_list_f64(&raw[i + 1]); i += 2; }
            "--bz-values" => { args.bz_values = parse_list_f64(&raw[i + 1]); i += 2; }
            "--jx-values" => { args.jx_values = parse_list_f64(&raw[i + 1]); i += 2; }
            "--nx" => { args.nx = raw[i + 1].parse().unwrap(); i += 2; }
            "--ny" => { args.ny = raw[i + 1].parse().unwrap(); i += 2; }
            "--cell-size" => { args.cell_size_nm = raw[i + 1].parse().unwrap(); i += 2; }
            "--dt-ps" => { args.dt_ps = raw[i + 1].parse().unwrap(); i += 2; }
            "--relax-steps" => { args.relax_steps = raw[i + 1].parse().unwrap(); i += 2; }
            "--pulse" => { args.pulse_strength_t = raw[i + 1].parse().unwrap(); i += 2; }
            "--pulse-steps" => { args.pulse_duration_steps = raw[i + 1].parse().unwrap(); i += 2; }
            "--sample-interval" => { args.sample_interval_steps = raw[i + 1].parse().unwrap(); i += 2; }
            "--num-samples" => { args.num_samples = raw[i + 1].parse().unwrap(); i += 2; }
            "--output" | "-o" => { args.output_path = raw[i + 1].clone(); i += 2; }
            "--stack-spec" => {
                args.stack_spec = Some(raw[i + 1].clone());
                i += 2;
            }
            "--stack-interlayer-a" => {
                args.stack_interlayer_a = Some(parse_list_f64(&raw[i + 1]));
                i += 2;
            }
            "--stack-spacing" => {
                args.stack_spacing_nm = Some(parse_list_f64(&raw[i + 1]));
                i += 2;
            }
            "--pump-probe-mode" => { args.pump_probe_mode = true; i += 1; }
            "--pump-t-center" => {
                args.pump_t_center = raw[i + 1].parse::<f64>().unwrap() * 1e-12;
                i += 2;
            }
            "--pump-fwhm" => {
                args.pump_fwhm = raw[i + 1].parse::<f64>().unwrap() * 1e-15;
                i += 2;
            }
            "--pump-peak" => { args.pump_peak_t = raw[i + 1].parse().unwrap(); i += 2; }
            "--pump-fluence" => {
                args.pump_fluence_j_m2 = Some(raw[i + 1].parse::<f64>().unwrap() * 10.0);
                i += 2;
            }
            "--pump-reflectivity" => {
                args.pump_reflectivity = raw[i + 1].parse().unwrap();
                i += 2;
            }
            "--probe-peak" => { args.probe_peak_t = raw[i + 1].parse().unwrap(); i += 2; }
            "--probe-fwhm" => {
                args.probe_fwhm = raw[i + 1].parse::<f64>().unwrap() * 1e-15;
                i += 2;
            }
            "--probe-fluence" => {
                args.probe_fluence_j_m2 = Some(raw[i + 1].parse::<f64>().unwrap() * 10.0);
                i += 2;
            }
            "--probe-reflectivity" => {
                args.probe_reflectivity = raw[i + 1].parse().unwrap();
                i += 2;
            }
            "--pump-probe-delay-range" => {
                let start: f64 = raw[i + 1].parse().unwrap();
                let end: f64 = raw[i + 2].parse().unwrap();
                let steps: usize = raw[i + 3].parse().unwrap();
                if steps < 1 {
                    eprintln!("--pump-probe-delay-range STEPS must be ≥ 1");
                    std::process::exit(1);
                }
                args.pump_probe_delay_range = Some((start, end, steps));
                i += 4;
            }
            "--thermal-preset" => {
                args.thermal_preset_key = Some(raw[i + 1].clone());
                args.enable_thermal = true;
                i += 2;
            }
            "--enable-thermal" => { args.enable_thermal = true; i += 1; }
            "--enable-llb" => { args.enable_llb = true; args.enable_thermal = true; i += 1; }
            "--t-ambient" => {
                args.thermal_t_ambient = raw[i + 1].parse().unwrap();
                i += 2;
            }
            "--help" | "-h" => { print_help(); std::process::exit(0); }
            other => {
                eprintln!("Unknown arg: {other}");
                print_help();
                std::process::exit(1);
            }
        }
    }
    args
}

fn print_help() {
    eprintln!("magnonic-sweep — parametric sweep harness");
    eprintln!();
    eprintln!("Design axes (comma-separated lists):");
    eprintln!("  --materials M1,M2,...       Bulk material names (default: fgt-bulk,fgt-effective)");
    eprintln!("  --substrates S1,S2,...      Substrate names (default: vacuum,pt,wte2,yig)");
    eprintln!("  --thicknesses T1,T2,...     Film thicknesses [nm] (default: 0.7,2.1)");
    eprintln!("  --bz-values B1,B2,...       External field z [T] (default: 0.0)");
    eprintln!("  --jx-values J1,J2,...       Current density along x [A/m²] (default: 0.0)");
    eprintln!();
    eprintln!("Multilayer stack (M4 — overrides --materials/--thicknesses):");
    eprintln!("  --stack-spec \"MAT1:T_nm,MAT2:T_nm,...\"  Fixed stack used for every design point");
    eprintln!("  --stack-interlayer-a A1,A2,...           Interlayer exchange [J/m], N_layers-1 values");
    eprintln!("  --stack-spacing D1,D2,...                Per-interface z-spacing [nm], N_layers-1 values");
    eprintln!();
    eprintln!("Geometry:");
    eprintln!("  --nx / --ny N               Grid (default 32)");
    eprintln!("  --cell-size F               Cell size [nm] (default 2.5)");
    eprintln!("  --dt-ps F                   Timestep [ps] (default 0.1)");
    eprintln!();
    eprintln!("Measurement protocol:");
    eprintln!("  --relax-steps N             Equilibration steps (default 5000)");
    eprintln!("  --pulse F                   Transverse Bx pulse amplitude [T] (default 1.0)");
    eprintln!("  --pulse-steps N             Pulse duration [steps] (default 100)");
    eprintln!("  --sample-interval N         Steps between probe samples (default 20)");
    eprintln!("  --num-samples N             Samples recorded after pulse (default 1024)");
    eprintln!();
    eprintln!("Output:");
    eprintln!("  --output PATH / -o PATH     CSV output path (default sweep.csv)");
    eprintln!();
    eprintln!("Pump-probe sequencer (P4):");
    eprintln!("  --pump-probe-mode                       enable pump-probe protocol");
    eprintln!("  --pump-t-center PS                      pump center time [ps] (default 200)");
    eprintln!("  --pump-fwhm FS / --probe-fwhm FS        temporal FWHM [fs] (default 100)");
    eprintln!("  --pump-peak T / --probe-peak T          IFE peak field [T] (default 0.5 / 0.1)");
    eprintln!("  --pump-fluence MJ / --probe-fluence MJ  absorbed fluence [mJ/cm²] — engages M3TM");
    eprintln!("  --pump-reflectivity F / --probe-reflectivity F   reflectivity 0..1 (default 0)");
    eprintln!("  --pump-probe-delay-range START END N    delay axis [ps]");
    eprintln!();
    eprintln!("Thermal (adds M3TM source + optional LLB integrator):");
    eprintln!("  --enable-thermal              attach a thermal config (M3TM observables populated)");
    eprintln!("  --enable-llb                  engage LLB integrator (implies --enable-thermal)");
    eprintln!("  --thermal-preset KEY          ni | py | fgt | yig | cofeb (default fgt)");
    eprintln!("  --t-ambient K                 ambient bath temperature [K] (default 300)");
}

fn run_design_point(config: &SimConfig, args: &SweepArgs) -> Result<ClockMetrics, String> {
    let mut solver = GpuSolver::new(config)?;
    let bz_f32 = config.b_ext[2] as f32;

    // 1. Relax
    solver.step_n(args.relax_steps);

    // 2. Pulse: apply transverse Bx, run for N_pulse steps, then turn off
    solver.set_b_ext(args.pulse_strength_t, 0.0, bz_f32);
    solver.step_n(args.pulse_duration_steps);
    solver.set_b_ext(0.0, 0.0, bz_f32);

    // 3. Record avg_mx during free decay — transverse component swings fully
    // during precession (unlike mz which stays near ±1 for shallow cones).
    let mut samples = Vec::with_capacity(args.num_samples);
    for _ in 0..args.num_samples {
        solver.step_n(args.sample_interval_steps);
        samples.push(solver.observables().avg_mx);
    }

    // 4. Analyze
    let dt_sample_s = config.dt * args.sample_interval_steps as f64;
    Ok(analyze_time_series(&samples, dt_sample_s))
}

/// Outcome of a single pump-probe design point (P4).
#[derive(Clone, Debug)]
struct PumpProbeOutcome {
    clock: ClockMetrics,
    pulse_count: usize,
    first_pulse_t_ps: f64,
    total_fluence_mj_cm2: f64,
    max_t_e_k: f32,
    min_m_reduced: f32,
}

fn build_pump_probe_pulses(args: &SweepArgs, delay_ps: f64) -> Vec<LaserPulse> {
    let mut pulses = Vec::with_capacity(2);
    let pump = LaserPulse {
        t_center: args.pump_t_center,
        duration_fwhm: args.pump_fwhm,
        peak_field: args.pump_peak_t,
        direction: [0.0, 0.0, 1.0],
        spot_center: [0.0, 0.0],
        spot_sigma: 0.0,
        peak_fluence: args.pump_fluence_j_m2,
        reflectivity: args.pump_reflectivity,
    };
    let probe = LaserPulse {
        t_center: args.pump_t_center + delay_ps * 1e-12,
        duration_fwhm: args.probe_fwhm,
        peak_field: args.probe_peak_t,
        direction: [0.0, 0.0, 1.0],
        spot_center: [0.0, 0.0],
        spot_sigma: 0.0,
        peak_fluence: args.probe_fluence_j_m2,
        reflectivity: args.probe_reflectivity,
    };
    pulses.push(pump);
    pulses.push(probe);
    pulses
}

fn total_fluence_mj_cm2(pulses: &[LaserPulse]) -> f64 {
    pulses
        .iter()
        .filter_map(|p| p.peak_fluence.map(|f| (1.0 - p.reflectivity as f64) * f * 0.1))
        .sum()
}

fn config_nz(stack: &Stack) -> usize {
    stack.layers.len()
}

fn pump_probe_delays(args: &SweepArgs) -> Vec<f64> {
    let (start, end, n) = args.pump_probe_delay_range.unwrap_or((0.1, 10.0, 20));
    if n == 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (end - start) * (i as f64) / ((n - 1) as f64))
        .collect()
}

fn build_thermal_config(args: &SweepArgs, nz: usize) -> Option<ThermalConfig> {
    if !args.enable_thermal {
        return None;
    }
    let key = args.thermal_preset_key.as_deref().unwrap_or("fgt-ni-surrogate");
    let preset = match material_thermal::for_key(key) {
        Some(p) => p,
        None => {
            eprintln!("Unknown --thermal-preset: {key}");
            std::process::exit(1);
        }
    };
    Some(ThermalConfig {
        t_ambient: args.thermal_t_ambient,
        per_layer: vec![preset; nz],
        thermal_dt_cap: 1.0e-15,
        thermal_window: (0.5e-12, 10e-12),
        enable_llb: args.enable_llb,
    })
}

/// Pump-probe design-point runner. Emits IFE pulses at (t_pump, t_pump+delay)
/// and samples probe_mz *after* the probe pulse during a free-decay window.
fn run_pump_probe_point(
    config: &SimConfig,
    args: &SweepArgs,
    delay_ps: f64,
) -> Result<PumpProbeOutcome, String> {
    let mut solver = GpuSolver::new(config)?;
    let bz_f32 = config.b_ext[2] as f32;

    // 1. Relax under quiescent state.
    solver.step_n(args.relax_steps);
    solver.set_b_ext(0.0, 0.0, bz_f32);

    // 2. Advance through the pump + (pump+delay) window. The photonic
    // pulses fire automatically via step_n's per-step amplitude rewrite
    // when config.photonic.pulses is non-empty.
    //
    // We need to cross both pulse peaks. Use a fixed "pulse+settle" window
    // of (delay_ps + 20 ps) before starting the free-decay sample window,
    // relative to t_pump.
    let pulse_window_s = delay_ps * 1e-12 + 20e-12;
    let n_window = (pulse_window_s / config.dt).ceil() as usize;
    solver.step_n(n_window);

    // 3. Record avg_mx during free decay — identical protocol to the
    // transverse-Bx sweep so ClockMetrics comparison stays valid.
    let mut samples = Vec::with_capacity(args.num_samples);
    let mut max_t_e_k = 0.0_f32;
    let mut min_m_reduced = 1.0_f32;
    for _ in 0..args.num_samples {
        solver.step_n(args.sample_interval_steps);
        let obs = solver.observables();
        samples.push(obs.avg_mx);
        max_t_e_k = max_t_e_k.max(obs.max_t_e);
        min_m_reduced = min_m_reduced.min(obs.min_m_reduced);
    }

    let dt_sample_s = config.dt * args.sample_interval_steps as f64;
    let clock = analyze_time_series(&samples, dt_sample_s);
    Ok(PumpProbeOutcome {
        clock,
        pulse_count: config.photonic.pulses.len(),
        first_pulse_t_ps: config
            .photonic
            .pulses
            .first()
            .map(|p| p.t_center * 1e12)
            .unwrap_or(0.0),
        total_fluence_mj_cm2: total_fluence_mj_cm2(&config.photonic.pulses),
        max_t_e_k,
        min_m_reduced,
    })
}

fn build_stack_from_args(
    args: &SweepArgs,
    fallback_material: &BulkMaterial,
    fallback_thickness_nm: f64,
) -> Result<(Stack, String, u32), String> {
    // If the user supplied an explicit stack spec, use that.
    if let Some(spec) = &args.stack_spec {
        let layers = parse_stack_spec(spec)?;
        let n = layers.len();
        let n_int = n.saturating_sub(1);
        let interlayer_a = args
            .stack_interlayer_a
            .clone()
            .unwrap_or_else(|| vec![0.0; n_int]);
        if interlayer_a.len() != n_int {
            return Err(format!(
                "stack has {n} layers but --stack-interlayer-a has {} values (need {n_int})",
                interlayer_a.len()
            ));
        }
        let layer_spacing: Vec<f64> = args
            .stack_spacing_nm
            .clone()
            .unwrap_or_else(|| vec![0.7; n_int])
            .iter()
            .map(|nm| nm * 1e-9)
            .collect();
        if layer_spacing.len() != n_int {
            return Err(format!(
                "stack has {n} layers but --stack-spacing has {} values (need {n_int})",
                layer_spacing.len()
            ));
        }
        Ok((
            Stack { layers, interlayer_a, layer_spacing },
            spec.clone(),
            n as u32,
        ))
    } else {
        // Monolayer from the (material × thickness) iteration
        Ok((
            Stack::monolayer(
                fallback_material.clone(),
                fallback_thickness_nm * 1e-9,
                [0.0, 0.0, 1.0],
            ),
            format!("{}:{}", fallback_material.name, fallback_thickness_nm),
            1u32,
        ))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = parse_args();

    // If a stack spec is provided, material/thickness axes collapse to a single
    // dummy iteration (the stack overrides them).
    let eff_materials: Vec<String> = if args.stack_spec.is_some() {
        vec!["(stack)".to_string()]
    } else {
        args.materials.clone()
    };
    let eff_thicknesses: Vec<f64> = if args.stack_spec.is_some() {
        vec![0.0]
    } else {
        args.thicknesses_nm.clone()
    };

    let total = eff_materials.len()
        * args.substrates.len()
        * eff_thicknesses.len()
        * args.bz_values.len()
        * args.jx_values.len();
    eprintln!("Sweep: {} design points", total);
    if let Some(s) = &args.stack_spec {
        eprintln!("  Stack:       {s}");
        if let Some(a) = &args.stack_interlayer_a {
            eprintln!("  interlayer_a: {:?} J/m", a);
        }
        if let Some(d) = &args.stack_spacing_nm {
            eprintln!("  layer_spacing: {:?} nm", d);
        }
    } else {
        eprintln!("  Materials:   {:?}", args.materials);
        eprintln!("  Thicknesses: {:?} nm", args.thicknesses_nm);
    }
    eprintln!("  Substrates:  {:?}", args.substrates);
    eprintln!("  Bz values:   {:?} T", args.bz_values);
    eprintln!("  Jx values:   {:?} A/m²", args.jx_values);
    eprintln!();

    let file = File::create(&args.output_path)?;
    let mut out = BufWriter::new(file);

    // Extended CSV schema. Pump-probe mode (P4) adds five extra columns;
    // the transverse-Bx mode leaves them empty / zero.
    writeln!(
        out,
        "material,substrate,thickness_nm,n_layers,stack_desc,ms_eff,a_eff,k_u_eff,alpha_eff,d_dmi_eff,bz_T,jx_A_per_m2,\
         freq_GHz,freq_width_GHz,q_factor,decay_time_ns,final_value,late_amplitude,dt_sample_ps,\
         delay_ps,pulse_count,first_pulse_t_ps,total_fluence_mj_cm2,max_te_k,min_m_reduced"
    )?;

    let t_start = Instant::now();
    let mut idx = 0;

    // Fallback material (used when --stack-spec is absent)
    let fallback_fgt = BulkMaterial::fgt_bulk();

    for m_name in &eff_materials {
        let bulk = if args.stack_spec.is_some() {
            fallback_fgt.clone()
        } else {
            let Some(b) = BulkMaterial::lookup(m_name) else {
                eprintln!("!! Unknown material {m_name}, skipping");
                continue;
            };
            b
        };
        for s_name in &args.substrates {
            let Some(substrate) = Substrate::lookup(s_name) else {
                eprintln!("!! Unknown substrate {s_name}, skipping");
                continue;
            };
            for &t_nm in &eff_thicknesses {
                for &bz in &args.bz_values {
                    for &jx in &args.jx_values {
                        idx += 1;

                        let (stack, stack_desc, n_layers) =
                            match build_stack_from_args(&args, &bulk, t_nm) {
                                Ok(s) => s,
                                Err(e) => {
                                    eprintln!("!! Stack build error: {e}");
                                    continue;
                                }
                            };

                        let nz = config_nz(&stack);
                        let mut config = SimConfig {
                            stack,
                            substrate: substrate.clone(),
                            geometry: Geometry {
                                cell_size: args.cell_size_nm * 1e-9,
                                nx: args.nx,
                                ny: args.ny,
                            },
                            dt: args.dt_ps * 1e-12,
                            b_ext: [0.0, 0.0, bz],
                            stab_coeff: 1.0e11,
                            j_current: [jx, 0.0, 0.0],
                            photonic: PhotonicConfig::default(),
                            readback_interval: 100,
                            total_steps: 0,
                            probe_idx: None,
                            probe_layer: None,
                        };
                        config.photonic.thermal = build_thermal_config(&args, nz);

                        let eff = config.effective();
                        let layer0 = &eff.layers[0];

                        // Column: material name — for stack runs, use layer 0 material name
                        let m_col = if args.stack_spec.is_some() {
                            config.stack.layers[0].material.name.to_string()
                        } else {
                            m_name.clone()
                        };
                        let t_col = config.stack.layers[0].thickness * 1e9;

                        if args.pump_probe_mode {
                            // Iterate the pump-probe delay axis. Other axes
                            // still loop (material × substrate × bz × jx) —
                            // each (design × delay) gets one CSV row.
                            let delays_ps = pump_probe_delays(&args);
                            for delay_ps in delays_ps.iter().copied() {
                                let mut cfg = config.clone();
                                cfg.photonic.pulses = build_pump_probe_pulses(&args, delay_ps);
                                let t0 = Instant::now();
                                let result = run_pump_probe_point(&cfg, &args, delay_ps);
                                let dt_pt = t0.elapsed().as_secs_f64();
                                match result {
                                    Ok(outcome) => {
                                        let m = &outcome.clock;
                                        writeln!(
                                            out,
                                            "{m_col},{s_name},{t_col:.3},{n_layers},\"{stack_desc}\",\
                                             {:.4e},{:.4e},{:.4e},{:.5},{:.4e},{bz},{jx:.3e},\
                                             {:.4},{:.4},{:.3e},{:.4e},{:.6},{:.4e},{:.3},\
                                             {delay_ps:.4},{},{:.4},{:.4},{:.2},{:.6}",
                                            layer0.ms, layer0.a_ex, layer0.k_u, layer0.alpha, layer0.d_dmi,
                                            m.freq_ghz, m.freq_width_ghz, m.q_factor,
                                            m.decay_time_ns, m.final_value, m.late_amplitude,
                                            m.dt_sample_ps,
                                            outcome.pulse_count,
                                            outcome.first_pulse_t_ps,
                                            outcome.total_fluence_mj_cm2,
                                            outcome.max_t_e_k,
                                            outcome.min_m_reduced,
                                        )?;
                                        out.flush()?;
                                        eprintln!(
                                            "[{idx}/{total}] {stack_desc}+{s_name} Δ={delay_ps:.2}ps \
                                             → f={:.2}GHz Q={:.1} |m|_min={:.3} ({:.1}s)",
                                            m.freq_ghz, m.q_factor, outcome.min_m_reduced, dt_pt,
                                        );
                                    }
                                    Err(e) => {
                                        eprintln!("[{idx}/{total}] {stack_desc}+{s_name} Δ={delay_ps:.2}ps FAILED: {e}");
                                    }
                                }
                            }
                        } else {
                            let t0 = Instant::now();
                            let result = run_design_point(&config, &args);
                            let dt_pt = t0.elapsed().as_secs_f64();

                            match result {
                                Ok(m) => {
                                    writeln!(
                                        out,
                                        "{m_col},{s_name},{t_col:.3},{n_layers},\"{stack_desc}\",\
                                         {:.4e},{:.4e},{:.4e},{:.5},{:.4e},{bz},{jx:.3e},\
                                         {:.4},{:.4},{:.3e},{:.4e},{:.6},{:.4e},{:.3},\
                                         ,0,0,0,0,0",
                                        layer0.ms, layer0.a_ex, layer0.k_u, layer0.alpha, layer0.d_dmi,
                                        m.freq_ghz, m.freq_width_ghz, m.q_factor,
                                        m.decay_time_ns, m.final_value, m.late_amplitude,
                                        m.dt_sample_ps,
                                    )?;
                                    out.flush()?;
                                    eprintln!(
                                        "[{idx}/{total}] {stack_desc}+{s_name} Bz={bz:.2} Jx={jx:.1e} \
                                         → f={:.2}GHz Q={:.1} τ={:.2}ns  ({:.1}s)",
                                        m.freq_ghz, m.q_factor, m.decay_time_ns, dt_pt,
                                    );
                                }
                                Err(e) => {
                                    eprintln!("[{idx}/{total}] {stack_desc}+{s_name} FAILED: {e}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let total_elapsed = t_start.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("Sweep complete: {idx} points in {total_elapsed:.1}s ({:.2}s/point avg)",
              total_elapsed / (idx as f64).max(1.0));
    eprintln!("CSV: {}", args.output_path);

    Ok(())
}
