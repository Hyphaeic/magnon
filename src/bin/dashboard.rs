use std::sync::Arc;
use std::time::Instant;

use minifb::{Key, KeyRepeat, Window, WindowOptions};
use rustfft::{num_complex::Complex, Fft, FftPlanner};

use magnonic_clock_sim::config::{Layer, SimConfig};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{LaserPulse, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

/// Thermal-preset cycling order (matches `material_thermal::for_key`).
const THERMAL_PRESETS: &[&str] = &["fgt", "fgt-zhou", "ni", "py", "cofeb", "yig"];

#[path = "dashboard_text.rs"]
mod text;

fn parse_stack(spec: &str) -> Result<Vec<Layer>, String> {
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

fn parse_f64_list(s: &str) -> Result<Vec<f64>, String> {
    s.split(',').map(|x| x.trim().parse().map_err(|e| format!("{e}"))).collect()
}

const PLOT_HEIGHT: usize = 200;
const LABEL_HEIGHT: usize = 14;
/// Minimum window width — leaves room for a status panel when the grid is small.
const MIN_WIN_W: usize = 640;

// ── Heatmap fields ────────────────────────────────────────────────
//
// Phase D1 addition: the heatmap can visualize more than just m_z. Users
// cycle through the available fields with the `V` key; thermal-only fields
// are skipped when `thermal` is disabled.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeatmapField {
    Mz,
    Mx,
    My,
    MMag,       // |m|
    TeK,        // electron temperature [K]
    TpK,        // phonon temperature [K]
    MReduced,   // M3TM's m_reduced (mirrors |mag| under LLB back-coupling)
    KspacePower, // |FFT(mx + i·my)|² over the displayed layer, fftshifted (k=0 centre)
}

impl HeatmapField {
    fn label(self) -> &'static str {
        match self {
            Self::Mz => "Mz",
            Self::Mx => "Mx",
            Self::My => "My",
            Self::MMag => "|m|",
            Self::TeK => "Te",
            Self::TpK => "Tp",
            Self::MReduced => "m_red",
            Self::KspacePower => "|M(k)|^2",
        }
    }
    fn units(self) -> &'static str {
        match self {
            Self::TeK | Self::TpK => "K",
            Self::KspacePower => "dB",
            _ => "",
        }
    }
    fn requires_thermal(self) -> bool {
        matches!(self, Self::TeK | Self::TpK | Self::MReduced)
    }
    fn signed(self) -> bool {
        matches!(self, Self::Mz | Self::Mx | Self::My)
    }
    fn is_kspace(self) -> bool {
        matches!(self, Self::KspacePower)
    }
}

// ── 2D FFT helper ─────────────────────────────────────────────────
//
// Separable forward 2D FFT on an nx × ny grid. Caches two 1D plans
// (length nx and length ny) across calls so the plan setup cost is paid
// once; inner buffers are passed in to avoid per-frame allocation.

struct KspaceEngine {
    nx: usize,
    ny: usize,
    fft_x: Arc<dyn Fft<f32>>,
    fft_y: Arc<dyn Fft<f32>>,
    grid: Vec<Complex<f32>>,
    col: Vec<Complex<f32>>,
    /// |ψ(k)|² at each (kx, ky) bin, fftshifted so (nx/2, ny/2) is k=0.
    power_shifted: Vec<f32>,
    /// Cached (min, max) of log-scale display values (dB).
    last_db_range: (f32, f32),
}

impl KspaceEngine {
    fn new(nx: usize, ny: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft_x = planner.plan_fft_forward(nx);
        let fft_y = planner.plan_fft_forward(ny);
        Self {
            nx,
            ny,
            fft_x,
            fft_y,
            grid: vec![Complex { re: 0.0, im: 0.0 }; nx * ny],
            col: vec![Complex { re: 0.0, im: 0.0 }; ny],
            power_shifted: vec![0.0; nx * ny],
            last_db_range: (-60.0, 0.0),
        }
    }

    /// Recompute |FFT(mx + i·my)|² for one layer. `mag` is the full
    /// magnetization buffer (nx·ny·nz cells × 4 floats). After this returns,
    /// `power_shifted[iy*nx + ix]` gives the fftshifted magnitude-squared
    /// at bin (ix, iy), with k=0 at screen centre (ix=nx/2, iy=ny/2).
    ///
    /// Log-scale (dB relative to peak) is computed here too; `last_db_range`
    /// holds (floor, 0 dB).
    fn compute(&mut self, mag: &[f32], layer: usize) {
        let nx = self.nx;
        let ny = self.ny;
        let offset = layer * nx * ny;
        // Populate ψ(x, y) = mx + i·my. Row-major (iy outer, ix inner) so
        // that one "row" in `grid` is ix ∈ [0, nx).
        for iy in 0..ny {
            for ix in 0..nx {
                let cell = offset + ix * ny + iy;
                let mx = mag[cell * 4];
                let my = mag[cell * 4 + 1];
                self.grid[iy * nx + ix] = Complex { re: mx, im: my };
            }
        }
        // Row FFTs (length nx).
        for iy in 0..ny {
            let row_start = iy * nx;
            self.fft_x.process(&mut self.grid[row_start..row_start + nx]);
        }
        // Column FFTs (length ny) via transpose-and-back.
        for ix in 0..nx {
            for iy in 0..ny {
                self.col[iy] = self.grid[iy * nx + ix];
            }
            self.fft_y.process(&mut self.col);
            for iy in 0..ny {
                self.grid[iy * nx + ix] = self.col[iy];
            }
        }
        // fftshift + |ψ|² into power_shifted.
        let hx = nx / 2;
        let hy = ny / 2;
        let mut peak = 0.0_f32;
        for iy in 0..ny {
            for ix in 0..nx {
                let src_ix = (ix + hx) % nx;
                let src_iy = (iy + hy) % ny;
                let c = self.grid[src_iy * nx + src_ix];
                let p = c.re * c.re + c.im * c.im;
                self.power_shifted[iy * nx + ix] = p;
                if p > peak {
                    peak = p;
                }
            }
        }
        // Convert to dB relative to peak. Floor at -80 dB so the log never
        // goes to -inf on pure-zero bins.
        let log_peak = (peak.max(1e-30)).ln() * (10.0 / std::f32::consts::LN_10);
        let mut mn = f32::INFINITY;
        for p in self.power_shifted.iter_mut() {
            let db = (p.max(1e-30)).ln() * (10.0 / std::f32::consts::LN_10) - log_peak;
            let db = db.max(-80.0);
            *p = db;
            if db < mn {
                mn = db;
            }
        }
        // Display range: [-60 dB, 0 dB] default, but stretch to actual floor
        // if the scene has less dynamic range (uniform state → shallow).
        let display_floor = mn.max(-60.0);
        self.last_db_range = (display_floor, 0.0);
    }

    /// Read fftshifted display value at (ix, iy) in screen coordinates.
    fn sample(&self, ix: usize, iy: usize) -> f32 {
        self.power_shifted[iy * self.nx + ix]
    }
}

fn main() {
    env_logger::init();

    let mut config = SimConfig::fgt_default(256, 256);

    // Thermal CLI state — built incrementally while parsing args.
    let mut enable_thermal = false;
    let mut enable_llb = false;
    let mut thermal_preset_key: Option<String> = None;
    let mut t_ambient: f32 = 300.0;

    // Pump-pulse defaults for the interactive `L` key (D2). These can be
    // overridden from the CLI; at runtime the user can still push the
    // classic transverse-Bx pulse with `P` at a separate strength.
    let mut pump_fwhm_fs: f64 = 100.0;
    let mut pump_peak_t: f32 = 0.5;
    let mut pump_dir: [f32; 3] = [0.0, 0.0, 1.0];
    let mut pump_fluence_j_m2: Option<f64> = None;
    let mut pump_reflectivity: f32 = 0.0;
    let mut pump_delay_ps: f64 = 1.0; // L fires a pulse at t_now + pump_delay_ps

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--nx" => { config.geometry.nx = args[i + 1].parse().unwrap(); i += 2; }
            "--ny" => { config.geometry.ny = args[i + 1].parse().unwrap(); i += 2; }
            "--thickness" => {
                let nm: f64 = args[i + 1].parse().unwrap();
                config.stack.layers[0].thickness = nm * 1e-9;
                i += 2;
            }
            "--material" => {
                if let Some(m) = BulkMaterial::lookup(&args[i + 1]) {
                    config.stack.layers[0].material = m;
                } else {
                    eprintln!("Unknown material: {}", args[i + 1]);
                    std::process::exit(1);
                }
                i += 2;
            }
            "--substrate" => {
                if let Some(s) = Substrate::lookup(&args[i + 1]) {
                    config.substrate = s;
                } else {
                    eprintln!("Unknown substrate: {}", args[i + 1]);
                    std::process::exit(1);
                }
                i += 2;
            }
            "--stack" => {
                match parse_stack(&args[i + 1]) {
                    Ok(layers) => {
                        let n_interfaces = layers.len().saturating_sub(1);
                        config.stack.layers = layers;
                        config.stack.interlayer_a.resize(n_interfaces, 0.0);
                        config.stack.layer_spacing.resize(n_interfaces, 0.7e-9);
                    }
                    Err(e) => { eprintln!("--stack error: {e}"); std::process::exit(1); }
                }
                i += 2;
            }
            "--interlayer-a" => {
                match parse_f64_list(&args[i + 1]) {
                    Ok(v) => { config.stack.interlayer_a = v; }
                    Err(e) => { eprintln!("--interlayer-a error: {e}"); std::process::exit(1); }
                }
                i += 2;
            }
            "--layer-spacing" => {
                match parse_f64_list(&args[i + 1]) {
                    Ok(v) => { config.stack.layer_spacing = v.iter().map(|nm| nm * 1e-9).collect(); }
                    Err(e) => { eprintln!("--layer-spacing error: {e}"); std::process::exit(1); }
                }
                i += 2;
            }
            "--probe-layer" => {
                config.probe_layer = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--alpha" => { config.stack.layers[0].material.alpha_bulk = args[i + 1].parse().unwrap(); i += 2; }
            "--bz" => { config.b_ext[2] = args[i + 1].parse().unwrap(); i += 2; }
            "--dt" => { config.dt = args[i + 1].parse().unwrap(); i += 2; }
            "--enable-thermal" => { enable_thermal = true; i += 1; }
            "--enable-llb" => { enable_thermal = true; enable_llb = true; i += 1; }
            "--thermal-preset" | "--thermal-params-for" => {
                thermal_preset_key = Some(args[i + 1].clone());
                enable_thermal = true;
                i += 2;
            }
            "--t-ambient" => { t_ambient = args[i + 1].parse().unwrap(); i += 2; }
            "--pump-fwhm" => { pump_fwhm_fs = args[i + 1].parse().unwrap(); i += 2; }
            "--pump-peak" => { pump_peak_t = args[i + 1].parse().unwrap(); i += 2; }
            "--pump-dir" => {
                pump_dir = match args[i + 1].as_str() {
                    "z" | "+z" => [0.0, 0.0, 1.0],
                    "-z" => [0.0, 0.0, -1.0],
                    "x" | "+x" => [1.0, 0.0, 0.0],
                    "-x" => [-1.0, 0.0, 0.0],
                    "y" | "+y" => [0.0, 1.0, 0.0],
                    "-y" => [0.0, -1.0, 0.0],
                    other => {
                        eprintln!("Unknown --pump-dir: {other}");
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            "--pump-fluence" => {
                // mJ/cm² default → J/m²
                let mj_cm2: f64 = args[i + 1].parse().unwrap();
                pump_fluence_j_m2 = Some(mj_cm2 * 10.0);
                i += 2;
            }
            "--pump-reflectivity" => { pump_reflectivity = args[i + 1].parse().unwrap(); i += 2; }
            "--pump-delay" => { pump_delay_ps = args[i + 1].parse().unwrap(); i += 2; }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => { i += 1; }
        }
    }

    // Attach thermal config if requested.
    if enable_thermal {
        let key = thermal_preset_key.as_deref().unwrap_or("fgt");
        let preset = material_thermal::for_key(key).unwrap_or_else(|| {
            eprintln!("Unknown --thermal-preset: {key}");
            std::process::exit(1);
        });
        let per_layer = vec![preset; config.stack.layers.len()];
        config.photonic.thermal = Some(ThermalConfig {
            t_ambient,
            per_layer,
            thermal_dt_cap: 1.0e-15,
            thermal_window: (0.5e-12, 10.0e-12),
            enable_llb,
        });
    }

    config.print_summary();

    let nx = config.geometry.nx as usize;
    let ny = config.geometry.ny as usize;
    let win_w = nx.max(MIN_WIN_W);
    let win_h = ny.max(240) + PLOT_HEIGHT + LABEL_HEIGHT;
    // Effective heatmap rows: always ny (may be smaller than plot_start_y).
    let plot_start_y = ny.max(240) + LABEL_HEIGHT;
    let side_panel_x = nx + 8; // 8-px gap between heatmap and status panel
    let heatmap_h = ny.max(240);

    let mut window = Window::new(
        "Magnonic Clock Simulator",
        win_w,
        win_h,
        WindowOptions {
            resize: false,
            scale: minifb::Scale::X2,
            ..Default::default()
        },
    )
    .expect("Failed to create window");

    window.set_target_fps(60);

    let mut solver = GpuSolver::new(&config).expect("Failed to init GPU solver");
    let mut framebuf = vec![0x1a1a1au32; win_w * win_h];
    let mut running = false;
    let mut steps_per_frame: usize = 100;
    let mut display_layer: u32 = 0;
    let nz = config.nz();

    // Live parameters
    let mut alpha = config.effective().layers[0].alpha as f32;
    let mut b_ext_x = 0.0f32;
    let mut b_ext_z = config.b_ext[2] as f32;

    // Transverse pulse state
    let mut pulse_strength = 1.0f32; // Tesla
    let mut pulse_frames_left = 0u32;
    let pulse_duration = 5u32;

    // History (plot strip)
    let mut mz_hist: Vec<f32> = Vec::new();
    let mut mx_hist: Vec<f32> = Vec::new();
    let mut my_hist: Vec<f32> = Vec::new();
    let mut probe_hist: Vec<f32> = Vec::new();

    let mut steps_per_sec = 0.0f64;

    // Initial readback
    let mut mag_data = solver.readback_mag();
    let mut temp_e_data: Vec<f32> = Vec::new();
    let mut temp_p_data: Vec<f32> = Vec::new();
    let mut m_red_data: Vec<f32> = Vec::new();
    if enable_thermal {
        temp_e_data = solver.readback_temp_e();
        m_red_data = solver.readback_m_reduced();
    }

    // Field cycling (D1 + C)
    let all_fields: &[HeatmapField] = &[
        HeatmapField::Mz,
        HeatmapField::Mx,
        HeatmapField::My,
        HeatmapField::MMag,
        HeatmapField::KspacePower,
        HeatmapField::TeK,
        HeatmapField::TpK,
        HeatmapField::MReduced,
    ];
    let mut field_idx = 0usize;

    // Lazy k-space engine — allocated on first use. A common path
    // (view ≠ KspacePower) pays zero allocation / FFT cost.
    let mut kspace: Option<KspaceEngine> = None;

    print_controls();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t0 = Instant::now();

        if window.is_key_pressed(Key::Space, KeyRepeat::No) { running = !running; }
        if window.is_key_pressed(Key::Up, KeyRepeat::Yes) {
            steps_per_frame = (steps_per_frame * 2).min(10_000);
            eprintln!("steps/frame = {steps_per_frame}");
        }
        if window.is_key_pressed(Key::Down, KeyRepeat::Yes) {
            steps_per_frame = (steps_per_frame / 2).max(1);
            eprintln!("steps/frame = {steps_per_frame}");
        }
        if window.is_key_pressed(Key::A, KeyRepeat::Yes) {
            alpha = (alpha * 1.5).min(1.0);
            solver.set_alpha(alpha);
            eprintln!("alpha = {alpha:.5}");
        }
        if window.is_key_pressed(Key::Z, KeyRepeat::Yes) {
            alpha = (alpha / 1.5).max(1e-5);
            solver.set_alpha(alpha);
            eprintln!("alpha = {alpha:.6}");
        }
        if window.is_key_pressed(Key::B, KeyRepeat::Yes) {
            b_ext_z += 0.1;
            solver.set_b_ext(b_ext_x, 0.0, b_ext_z);
            eprintln!("Bz = {b_ext_z:.2} T");
        }
        if window.is_key_pressed(Key::N, KeyRepeat::Yes) {
            b_ext_z -= 0.1;
            solver.set_b_ext(b_ext_x, 0.0, b_ext_z);
            eprintln!("Bz = {b_ext_z:.2} T");
        }
        if window.is_key_pressed(Key::P, KeyRepeat::No) {
            pulse_frames_left = pulse_duration;
            b_ext_x = pulse_strength;
            solver.set_b_ext(b_ext_x, 0.0, b_ext_z);
            eprintln!("PULSE: Bx = {pulse_strength:.1} T for {pulse_duration} frames");
            if !running { running = true; }
        }
        if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
            pulse_strength = (pulse_strength + 0.5).min(10.0);
            eprintln!("pulse strength = {pulse_strength:.1} T");
        }
        if window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
            pulse_strength = (pulse_strength - 0.5).max(0.1);
            eprintln!("pulse strength = {pulse_strength:.1} T");
        }
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            solver.reset_random();
            mag_data = solver.readback_mag();
            eprintln!("RESET: random magnetization");
        }
        if window.is_key_pressed(Key::D, KeyRepeat::No) {
            solver.reset_stripe_domains(8);
            mag_data = solver.readback_mag();
            eprintln!("RESET: stripe domains (width=8)");
        }
        if window.is_key_pressed(Key::U, KeyRepeat::No) {
            solver.reset_uniform_z();
            mag_data = solver.readback_mag();
            eprintln!("RESET: uniform +z");
        }
        if window.is_key_pressed(Key::S, KeyRepeat::No) {
            let nx_f = config.geometry.nx as f64;
            let ny_f = config.geometry.ny as f64;
            let dx_nm = config.geometry.cell_size * 1e9;
            let r_nm = dx_nm * nx_f.min(ny_f) / 6.0;
            solver.reset_skyrmion_seed(r_nm);
            mag_data = solver.readback_mag();
            eprintln!("RESET: Néel skyrmion seed, R ≈ {r_nm:.1} nm");
        }
        if window.is_key_pressed(Key::K, KeyRepeat::No) {
            solver.reset_uniform_z_alternating();
            mag_data = solver.readback_mag();
            eprintln!("RESET: alternating +z/-z per layer (synthetic AFM)");
        }
        if window.is_key_pressed(Key::C, KeyRepeat::No) {
            mz_hist.clear();
            mx_hist.clear();
            my_hist.clear();
            probe_hist.clear();
            eprintln!("CLEAR: plot history");
        }
        // ── D2 runtime thermal/photonic controls ──────────────────────

        // L — fire an IFE laser pulse at the configured pump params.
        if window.is_key_pressed(Key::L, KeyRepeat::No) {
            let t_now = solver.t_sim();
            let fwhm = pump_fwhm_fs * 1e-15;
            let mut pulse = LaserPulse::new(t_now + pump_delay_ps * 1e-12, fwhm, pump_peak_t, pump_dir);
            pulse.peak_fluence = pump_fluence_j_m2;
            pulse.reflectivity = pump_reflectivity;
            solver.add_pulse(pulse);
            let fluence_tag = match pump_fluence_j_m2 {
                Some(f) => format!(" + {:.3} mJ/cm² (M3TM)", f * 0.1),
                None => String::new(),
            };
            eprintln!(
                "LASER: t={:.2}+{:.2}ps fwhm={:.0}fs peak={:.2}T dir=({:.1},{:.1},{:.1}){fluence_tag}",
                t_now * 1e12, pump_delay_ps, pump_fwhm_fs, pump_peak_t,
                pump_dir[0], pump_dir[1], pump_dir[2],
            );
            if !running { running = true; }
        }

        // T — toggle thermal on/off.
        if window.is_key_pressed(Key::T, KeyRepeat::No) {
            if enable_thermal {
                solver.set_thermal(None);
                enable_thermal = false;
                // Leave enable_llb flag alone — re-engages on next T.
                temp_e_data.clear();
                temp_p_data.clear();
                m_red_data.clear();
                eprintln!("THERMAL: off");
            } else {
                let key = thermal_preset_key.as_deref().unwrap_or("fgt");
                if let Some(preset) = material_thermal::for_key(key) {
                    let cfg = ThermalConfig {
                        t_ambient,
                        per_layer: vec![preset; config.stack.layers.len()],
                        thermal_dt_cap: 1.0e-15,
                        thermal_window: (0.5e-12, 10.0e-12),
                        enable_llb,
                    };
                    solver.set_thermal(Some(cfg));
                    enable_thermal = true;
                    temp_e_data = solver.readback_temp_e();
                    m_red_data = solver.readback_m_reduced();
                    eprintln!("THERMAL: on  preset={key}  T_amb={t_ambient:.0}K  LLB={enable_llb}");
                } else {
                    eprintln!("THERMAL: unknown preset {key}");
                }
            }
        }

        // G — toggle LLB within thermal (no-op if thermal is off).
        if window.is_key_pressed(Key::G, KeyRepeat::No) {
            enable_llb = !enable_llb;
            if enable_thermal {
                let key = thermal_preset_key.as_deref().unwrap_or("fgt");
                if let Some(preset) = material_thermal::for_key(key) {
                    let cfg = ThermalConfig {
                        t_ambient,
                        per_layer: vec![preset; config.stack.layers.len()],
                        thermal_dt_cap: 1.0e-15,
                        thermal_window: (0.5e-12, 10.0e-12),
                        enable_llb,
                    };
                    solver.set_thermal(Some(cfg));
                    eprintln!("INTEGRATOR: {}", if enable_llb { "LLB" } else { "LLG (+ M3TM observability)" });
                }
            } else {
                eprintln!("INTEGRATOR: enable thermal (T) first to engage LLB");
            }
        }

        // [ / ] — T_ambient ∓ 10 K. Only affects the thermal bath; no effect when thermal off.
        if window.is_key_pressed(Key::LeftBracket, KeyRepeat::Yes) {
            t_ambient = (t_ambient - 10.0).max(0.0);
            if enable_thermal { solver.set_thermal_ambient(t_ambient); }
            eprintln!("T_ambient = {t_ambient:.0} K");
        }
        if window.is_key_pressed(Key::RightBracket, KeyRepeat::Yes) {
            t_ambient = (t_ambient + 10.0).min(5000.0);
            if enable_thermal { solver.set_thermal_ambient(t_ambient); }
            eprintln!("T_ambient = {t_ambient:.0} K");
        }

        // I — cycle thermal preset. Requires thermal to be on; engages with the new preset.
        if window.is_key_pressed(Key::I, KeyRepeat::No) {
            let current = thermal_preset_key.as_deref().unwrap_or("fgt");
            let idx = THERMAL_PRESETS.iter().position(|&k| k == current).unwrap_or(0);
            let next = THERMAL_PRESETS[(idx + 1) % THERMAL_PRESETS.len()];
            thermal_preset_key = Some(next.to_string());
            if enable_thermal {
                if let Some(preset) = material_thermal::for_key(next) {
                    let cfg = ThermalConfig {
                        t_ambient,
                        per_layer: vec![preset; config.stack.layers.len()],
                        thermal_dt_cap: 1.0e-15,
                        thermal_window: (0.5e-12, 10.0e-12),
                        enable_llb,
                    };
                    solver.set_thermal(Some(cfg));
                    temp_e_data = solver.readback_temp_e();
                    m_red_data = solver.readback_m_reduced();
                }
            }
            eprintln!("THERMAL PRESET: {next}");
        }

        // X — clear all queued pulses (cancels mid-pulse).
        if window.is_key_pressed(Key::X, KeyRepeat::No) {
            solver.clear_pulses();
            eprintln!("PULSES: cleared");
        }

        // V — cycle heatmap field. Skips fields unavailable in the current mode.
        if window.is_key_pressed(Key::V, KeyRepeat::No) {
            for _ in 0..all_fields.len() {
                field_idx = (field_idx + 1) % all_fields.len();
                if !all_fields[field_idx].requires_thermal() || enable_thermal {
                    break;
                }
            }
            // If we just switched to KspacePower while paused, force one
            // FFT pass so the user sees a populated spectrum rather than a
            // blank panel until Space is pressed.
            if all_fields[field_idx] == HeatmapField::KspacePower && !running {
                let eng = kspace.get_or_insert_with(|| KspaceEngine::new(nx, ny));
                eng.compute(&mag_data, display_layer as usize);
            }
            eprintln!("VIEW: {}", all_fields[field_idx].label());
        }
        for (key, layer) in [
            (Key::Key1, 0u32), (Key::Key2, 1u32),
            (Key::Key3, 2u32), (Key::Key4, 3u32),
        ] {
            if window.is_key_pressed(key, KeyRepeat::No) && layer < nz {
                display_layer = layer;
                eprintln!("DISPLAY: layer {display_layer} / {nz}");
            }
        }

        // Pulse countdown
        if pulse_frames_left > 0 {
            pulse_frames_left -= 1;
            if pulse_frames_left == 0 {
                b_ext_x = 0.0;
                solver.set_b_ext(0.0, 0.0, b_ext_z);
                eprintln!("PULSE OFF: Bx = 0");
            }
        }

        // Simulate
        if running {
            solver.step_n(steps_per_frame);
            mag_data = solver.readback_mag();
            if enable_thermal {
                temp_e_data = solver.readback_temp_e();
                m_red_data = solver.readback_m_reduced();
                // Only read temp_p when currently visualising it — three
                // full-cell readbacks per frame is fine but four costs
                // ~0.5 ms at 256² on the RTX 4060.
                if all_fields[field_idx] == HeatmapField::TpK {
                    temp_p_data = solver.readback_temp_p();
                }
            }
            // k-space view: 2D FFT of (mx + i·my) on the displayed layer.
            // Compute per-frame when the view is active, skip otherwise.
            if all_fields[field_idx] == HeatmapField::KspacePower {
                let eng = kspace.get_or_insert_with(|| KspaceEngine::new(nx, ny));
                eng.compute(&mag_data, display_layer as usize);
            }
            let obs = solver.observables();

            push_trim(&mut mz_hist, obs.avg_mz, win_w);
            push_trim(&mut mx_hist, obs.avg_mx, win_w);
            push_trim(&mut my_hist, obs.avg_my, win_w);
            push_trim(&mut probe_hist, obs.probe_mz, win_w);

            let dt = t0.elapsed().as_secs_f64();
            if dt > 0.0 { steps_per_sec = steps_per_frame as f64 / dt; }

            let pulse_tag = if pulse_frames_left > 0 {
                format!(" | PULSE Bx={b_ext_x:.1}T [{pulse_frames_left}]")
            } else { String::new() };
            let layer_tag = if nz > 1 {
                format!(" | Layer {}/{}", display_layer, nz)
            } else { String::new() };
            window.set_title(&format!(
                "Step {} | {:.1}ps | Mz={:.4} | probe={:.4} | α={:.4} | Bz={:.1}T{}{} | {:.0}st/s | {}/fr",
                obs.step, obs.time_ps, obs.avg_mz, obs.probe_mz,
                alpha, b_ext_z, pulse_tag, layer_tag, steps_per_sec, steps_per_frame,
            ));
        }
        let obs = solver.observables();

        // ── Render ─────────────────────────────────────────────
        // Background (heatmap left column) is overwritten by draw_heatmap;
        // the rest is painted by the panel / label / plot draws.
        let field = all_fields[field_idx];
        let (field_min, field_max) = field_range(
            field, &mag_data, &temp_e_data, &temp_p_data, &m_red_data,
            kspace.as_ref(),
            nx, ny, display_layer as usize, t_ambient,
        );
        draw_heatmap(
            &mut framebuf, win_w, win_h, field,
            &mag_data, &temp_e_data, &temp_p_data, &m_red_data,
            kspace.as_ref(),
            nx, ny, display_layer as usize, field_min, field_max,
        );
        draw_status_panel(
            &mut framebuf, win_w, win_h,
            side_panel_x, 0, win_w.saturating_sub(side_panel_x), heatmap_h,
            &obs, field, field_min, field_max,
            enable_thermal, enable_llb,
            alpha, b_ext_x, b_ext_z,
            pulse_frames_left, pulse_strength,
            display_layer, nz,
            steps_per_frame, steps_per_sec,
            t_ambient,
        );
        draw_label_bar(&mut framebuf, win_w, win_h, plot_start_y - LABEL_HEIGHT, LABEL_HEIGHT, pulse_frames_left > 0);
        draw_plot_strip(
            &mut framebuf, win_w, win_h, plot_start_y, PLOT_HEIGHT,
            &mz_hist, &mx_hist, &my_hist, &probe_hist,
        );

        window.update_with_buffer(&framebuf, win_w, win_h).unwrap();
    }
}

// ── Field access + range ───────────────────────────────────────────

fn field_sample(
    field: HeatmapField,
    mag: &[f32], te: &[f32], tp: &[f32], mr: &[f32],
    cell: usize,
) -> f32 {
    match field {
        HeatmapField::Mz => mag.get(cell * 4 + 2).copied().unwrap_or(0.0),
        HeatmapField::Mx => mag.get(cell * 4).copied().unwrap_or(0.0),
        HeatmapField::My => mag.get(cell * 4 + 1).copied().unwrap_or(0.0),
        HeatmapField::MMag => {
            let x = mag.get(cell * 4).copied().unwrap_or(0.0);
            let y = mag.get(cell * 4 + 1).copied().unwrap_or(0.0);
            let z = mag.get(cell * 4 + 2).copied().unwrap_or(0.0);
            (x * x + y * y + z * z).sqrt()
        }
        HeatmapField::TeK => te.get(cell).copied().unwrap_or(0.0),
        HeatmapField::TpK => tp.get(cell).copied().unwrap_or(0.0),
        HeatmapField::MReduced => mr.get(cell).copied().unwrap_or(0.0),
        // Reached only if called by mistake — draw_heatmap short-circuits
        // via `is_kspace()` before calling field_sample.
        HeatmapField::KspacePower => 0.0,
    }
}

fn field_range(
    field: HeatmapField,
    mag: &[f32], te: &[f32], tp: &[f32], mr: &[f32],
    kspace: Option<&KspaceEngine>,
    nx: usize, ny: usize, display_layer: usize, t_ambient: f32,
) -> (f32, f32) {
    if field.is_kspace() {
        return kspace.map(|k| k.last_db_range).unwrap_or((-60.0, 0.0));
    }
    if field.signed() {
        // Symmetric auto-scale around 0 so the diverging colormap's centre
        // (white) stays aligned with m=0. Without this, a nearly-uniform +z
        // state with 5° init cone (|Mx|, |My| ≈ 0.08) maps to the middle
        // of [-1, +1] and all non-Mz views collapse to pale-grey noise
        // indistinguishable from each other.
        let layer_offset = display_layer * nx * ny;
        let mut amax = 0.0_f32;
        for iy in 0..ny {
            for ix in 0..nx {
                let cell = layer_offset + ix * ny + iy;
                let v = field_sample(field, mag, te, tp, mr, cell).abs();
                if v > amax { amax = v; }
            }
        }
        // Floor to 0.05 so a pure-equilibrium state doesn't amplify numerical
        // noise to full-saturation. Ceiling is 1.0 because unit magnetisation
        // is the physical upper bound.
        let amax = amax.max(0.05).min(1.0);
        return (-amax, amax);
    }
    match field {
        HeatmapField::MMag | HeatmapField::MReduced => (0.0, 1.0),
        HeatmapField::TeK | HeatmapField::TpK => {
            // Auto-scale per frame, clamped to a sensible floor/ceiling so
            // the colormap doesn't flicker at equilibrium.
            let layer_offset = display_layer * nx * ny;
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for iy in 0..ny {
                for ix in 0..nx {
                    let cell = layer_offset + ix * ny + iy;
                    let v = field_sample(field, mag, te, tp, mr, cell);
                    if v < mn { mn = v; }
                    if v > mx { mx = v; }
                }
            }
            if !mn.is_finite() || !mx.is_finite() {
                return (t_ambient, t_ambient + 100.0);
            }
            let floor = t_ambient.min(mn);
            let ceil = (t_ambient + 100.0).max(mx);
            if ceil - floor < 5.0 { (floor, floor + 5.0) } else { (floor, ceil) }
        }
        _ => (-1.0, 1.0),
    }
}

fn push_trim(v: &mut Vec<f32>, val: f32, max: usize) {
    v.push(val);
    if v.len() > max { v.remove(0); }
}

// ── Rendering ──────────────────────────────────────────────────────

fn draw_heatmap(
    buf: &mut [u32], stride: usize, total_h: usize,
    field: HeatmapField,
    mag: &[f32], te: &[f32], tp: &[f32], mr: &[f32],
    kspace: Option<&KspaceEngine>,
    nx: usize, ny: usize, display_layer: usize,
    field_min: f32, field_max: f32,
) {
    let layer_offset = display_layer * nx * ny;
    let signed = field.signed();
    let is_kspace = field.is_kspace();
    for iy in 0..ny {
        for ix in 0..nx {
            let v = if is_kspace {
                kspace.map(|k| k.sample(ix, iy)).unwrap_or(field_min)
            } else {
                let cell = layer_offset + ix * ny + iy;
                field_sample(field, mag, te, tp, mr, cell)
            };
            let color = if signed {
                diverging_rgb(v, field_min, field_max)
            } else {
                // k-space uses a hot-style ramp too (high power = bright).
                let hot = matches!(field, HeatmapField::TeK | HeatmapField::TpK | HeatmapField::KspacePower);
                sequential_rgb(v, field_min, field_max, hot)
            };
            if iy < total_h && ix < stride {
                buf[iy * stride + ix] = color;
            }
        }
    }
    // Fill gap column between heatmap and status panel with dark gray.
    let panel_start = nx + 8;
    if nx < stride {
        for y in 0..ny.min(total_h) {
            for x in nx..panel_start.min(stride) {
                buf[y * stride + x] = 0x0f0f0f;
            }
        }
    }
}

fn draw_label_bar(buf: &mut [u32], stride: usize, total_h: usize, y_offset: usize, height: usize, pulsing: bool) {
    let bg = if pulsing { 0x442200 } else { 0x111111 };
    text::fill_rect(buf, stride, total_h, 0, y_offset, stride, height, bg);
    let y_mid = y_offset + height / 2;
    let colors = [
        (0xDD3333, "Mz", 20),
        (0x3366DD, "Mx", 100),
        (0x33BB55, "My", 180),
        (0xCC9933, "probe", 260),
    ];
    for (color, label, x_start) in colors {
        for dx in 0..8 {
            for dy in 0..3 {
                let x = x_start + dx;
                let y = y_mid - 1 + dy;
                if x < stride && y < y_offset + height {
                    buf[y * stride + x] = color;
                }
            }
        }
        text::draw_text(buf, stride, total_h, x_start + 12, y_mid - 3, label, 0xBBBBBB);
    }
}

fn draw_plot_strip(
    buf: &mut [u32], stride: usize, total_h: usize, y_offset: usize, height: usize,
    mz: &[f32], mx: &[f32], my: &[f32], probe: &[f32],
) {
    let bg = 0x1a1a1a;
    text::fill_rect(buf, stride, total_h, 0, y_offset, stride, height, bg);

    let zero_y = y_offset + height / 2;
    for x in 0..stride {
        buf[zero_y * stride + x] = 0x555555;
    }
    let q1_y = y_offset + height / 4;
    let q3_y = y_offset + 3 * height / 4;
    for x in (0..stride).step_by(4) {
        buf[q1_y * stride + x] = 0x333333;
        buf[q3_y * stride + x] = 0x333333;
    }
    for x in (0..stride).step_by(8) {
        buf[y_offset * stride + x] = 0x333333;
        buf[(y_offset + height - 1) * stride + x] = 0x333333;
    }
    text::draw_text(buf, stride, total_h, 4, y_offset + 2, "+1", 0x666666);
    text::draw_text(buf, stride, total_h, 4, zero_y - 3, "0", 0x666666);
    text::draw_text(buf, stride, total_h, 4, y_offset + height - 10, "-1", 0x666666);

    let plot_line = |buf: &mut [u32], data: &[f32], color: u32, thick: bool| {
        let n = data.len();
        let x_start = stride.saturating_sub(n);
        for (i, &val) in data.iter().enumerate() {
            let x = x_start + i;
            if x >= stride { continue; }
            let t = ((val + 1.0) * 0.5).clamp(0.0, 1.0);
            let y = y_offset + height - 1 - ((t * (height - 1) as f32) as usize).min(height - 1);
            buf[y * stride + x] = color;
            if thick {
                if y > y_offset { buf[(y - 1) * stride + x] = color; }
                if y + 1 < y_offset + height { buf[(y + 1) * stride + x] = color; }
            }
        }
    };
    plot_line(buf, mx, 0x3366DD, false);
    plot_line(buf, my, 0x33BB55, false);
    plot_line(buf, probe, 0xCC9933, false);
    plot_line(buf, mz, 0xDD3333, true);
}

// ── Status panel (D1) ──────────────────────────────────────────

fn draw_status_panel(
    buf: &mut [u32], stride: usize, total_h: usize,
    x0: usize, y0: usize, w: usize, h: usize,
    obs: &magnonic_clock_sim::gpu::Observables,
    field: HeatmapField, field_min: f32, field_max: f32,
    thermal_on: bool, llb_on: bool,
    alpha: f32, b_ext_x: f32, b_ext_z: f32,
    pulse_frames_left: u32, pulse_strength: f32,
    display_layer: u32, nz: u32,
    steps_per_frame: usize, steps_per_sec: f64,
    t_ambient: f32,
) {
    if w == 0 { return; }
    text::fill_rect(buf, stride, total_h, x0, y0, w, h, 0x0f0f0f);
    // 1-px right border for visual separation from the left edge of the window.
    for y in y0..(y0 + h).min(total_h) {
        if x0 < stride { buf[y * stride + x0] = 0x333333; }
    }

    let label_color: u32 = 0x888888;
    let value_color: u32 = 0xDDDDDD;
    let ok_color: u32 = 0x66CC88;
    let warn_color: u32 = 0xCC8844;
    let mode_color: u32 = if llb_on { 0x88AACC } else if thermal_on { ok_color } else { 0x666666 };

    let mut y = y0 + 4;
    let x_label = x0 + 6;
    let x_val = x0 + 60;

    // Header — integrator mode + thermal state.
    let integrator = if llb_on { "LLB" } else { "LLG" };
    let thermal_tag = if !thermal_on {
        "thermal: off".to_string()
    } else if llb_on {
        format!("LLB+M3TM @ {:.0}K", t_ambient)
    } else {
        format!("LLG+M3TM @ {:.0}K", t_ambient)
    };
    text::draw_text(buf, stride, total_h, x_label, y, "[", 0x666666);
    text::draw_text(buf, stride, total_h, x_label + 6, y, integrator, mode_color);
    text::draw_text(buf, stride, total_h, x_label + 6 + integrator.len() * text::CELL_W, y, "]", 0x666666);
    text::draw_text(buf, stride, total_h, x_label + 6 + (integrator.len() + 2) * text::CELL_W, y, &thermal_tag, label_color);
    y += text::CELL_H + 2;

    // Separator line
    for dx in 0..w.saturating_sub(12) {
        if x_label + dx < stride && y < total_h { buf[y * stride + x_label + dx] = 0x222222; }
    }
    y += 3;

    // Time + throughput
    text::draw_text(buf, stride, total_h, x_label, y, "t", label_color);
    text::draw_text(buf, stride, total_h, x_val, y, &format!("{:.2} ps", obs.time_ps), value_color);
    y += text::CELL_H;
    text::draw_text(buf, stride, total_h, x_label, y, "step", label_color);
    text::draw_text(buf, stride, total_h, x_val, y, &format!("{}", obs.step), value_color);
    y += text::CELL_H;
    text::draw_text(buf, stride, total_h, x_label, y, "rate", label_color);
    text::draw_text(buf, stride, total_h, x_val, y, &format!("{:.0}/s  {}/fr", steps_per_sec, steps_per_frame), value_color);
    y += text::CELL_H + 3;

    // Current field
    text::draw_text(buf, stride, total_h, x_label, y, "view", label_color);
    let field_desc = match field.units() {
        "" => format!("{}", field.label()),
        u => format!("{} [{}]", field.label(), u),
    };
    text::draw_text(buf, stride, total_h, x_val, y, &field_desc, 0xFFEE99);
    y += text::CELL_H;
    text::draw_text(buf, stride, total_h, x_label, y, "range", label_color);
    text::draw_text(buf, stride, total_h, x_val, y, &format!("{:.3} .. {:.3}", field_min, field_max), value_color);
    y += text::CELL_H + 3;

    // Magnetization summary
    text::draw_text(buf, stride, total_h, x_label, y, "avg Mz", label_color);
    text::draw_text(buf, stride, total_h, x_val, y, &format!("{:+.4}", obs.avg_mz), value_color);
    y += text::CELL_H;
    text::draw_text(buf, stride, total_h, x_label, y, "|m|", label_color);
    text::draw_text(buf, stride, total_h, x_val, y, &format!("{:.4}..{:.4}", obs.min_norm, obs.max_norm), value_color);
    y += text::CELL_H;
    text::draw_text(buf, stride, total_h, x_label, y, "probe", label_color);
    text::draw_text(buf, stride, total_h, x_val, y, &format!("{:+.4}", obs.probe_mz), value_color);
    y += text::CELL_H + 3;

    // Thermal readouts
    if thermal_on {
        text::draw_text(buf, stride, total_h, x_label, y, "Te max", label_color);
        text::draw_text(buf, stride, total_h, x_val, y, &format!("{:.0} K", obs.max_t_e), value_color);
        y += text::CELL_H;
        text::draw_text(buf, stride, total_h, x_label, y, "Tp max", label_color);
        text::draw_text(buf, stride, total_h, x_val, y, &format!("{:.0} K", obs.max_t_p), value_color);
        y += text::CELL_H;
        text::draw_text(buf, stride, total_h, x_label, y, "|m|red", label_color);
        text::draw_text(buf, stride, total_h, x_val, y, &format!("{:.4}", obs.min_m_reduced), value_color);
        y += text::CELL_H + 3;
    }

    // Controls
    text::draw_text(buf, stride, total_h, x_label, y, "a", label_color);
    text::draw_text(buf, stride, total_h, x_val, y, &format!("{:.5}", alpha), value_color);
    y += text::CELL_H;
    text::draw_text(buf, stride, total_h, x_label, y, "Bz", label_color);
    text::draw_text(buf, stride, total_h, x_val, y, &format!("{:+.2} T", b_ext_z), value_color);
    y += text::CELL_H;
    if b_ext_x.abs() > 1e-6 || pulse_frames_left > 0 {
        text::draw_text(buf, stride, total_h, x_label, y, "Bx", label_color);
        text::draw_text(buf, stride, total_h, x_val, y, &format!("{:+.2} T [{}]", b_ext_x, pulse_frames_left), warn_color);
        y += text::CELL_H;
    }
    text::draw_text(buf, stride, total_h, x_label, y, "pulse", label_color);
    text::draw_text(buf, stride, total_h, x_val, y, &format!("{:.1} T (P to fire)", pulse_strength), value_color);
    y += text::CELL_H + 3;

    // Stack
    if nz > 1 {
        text::draw_text(buf, stride, total_h, x_label, y, "layer", label_color);
        text::draw_text(buf, stride, total_h, x_val, y, &format!("{} / {}", display_layer, nz), value_color);
    }

    // Footer — abbreviated controls.
    let footer_y = y0 + h.saturating_sub(4 * text::CELL_H + 4);
    text::draw_text(buf, stride, total_h, x_label, footer_y, "V:view  L:laser  P:Bx", 0x666666);
    text::draw_text(buf, stride, total_h, x_label, footer_y + text::CELL_H, "T:thermal  G:LLB  I:preset", 0x666666);
    text::draw_text(buf, stride, total_h, x_label, footer_y + 2 * text::CELL_H, "[/]:T_amb  X:clear pulses", 0x666666);
    text::draw_text(buf, stride, total_h, x_label, footer_y + 3 * text::CELL_H, "R/D/U/S/K reset  C:plot", 0x666666);
}

// ── Colormaps ──────────────────────────────────────────────────────

/// Diverging blue-white-red map for signed data. `vmin` and `vmax` must be
/// supplied symmetric around 0 (e.g., ±amax); white corresponds to v = 0.
fn diverging_rgb(v: f32, vmin: f32, vmax: f32) -> u32 {
    let range = (vmax - vmin).max(1e-6);
    let t = ((v - vmin) / range).clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        ((s * 255.0) as u32, (s * 255.0) as u32, 255u32)
    } else {
        let s = (t - 0.5) * 2.0;
        (255u32, ((1.0 - s) * 255.0) as u32, ((1.0 - s) * 255.0) as u32)
    };
    (r << 16) | (g << 8) | b
}

/// Sequential colormap. `hot` = true uses a black→red→yellow→white ramp for
/// temperatures; false uses a viridis-like blue→green→yellow ramp for
/// magnitude-style data.
fn sequential_rgb(v: f32, vmin: f32, vmax: f32, hot: bool) -> u32 {
    let range = (vmax - vmin).max(1e-6);
    let t = ((v - vmin) / range).clamp(0.0, 1.0);
    if hot {
        // 0..0.33 black→red, 0.33..0.66 red→yellow, 0.66..1.0 yellow→white
        let (r, g, b) = if t < 1.0 / 3.0 {
            let s = t * 3.0;
            ((s * 255.0) as u32, 0u32, 0u32)
        } else if t < 2.0 / 3.0 {
            let s = (t - 1.0 / 3.0) * 3.0;
            (255u32, (s * 255.0) as u32, 0u32)
        } else {
            let s = (t - 2.0 / 3.0) * 3.0;
            (255u32, 255u32, (s * 255.0) as u32)
        };
        (r << 16) | (g << 8) | b
    } else {
        // Cheap viridis-ish: blue→teal→green→yellow
        let r = ((1.0 - (2.0 * t - 1.0).abs()).max(0.0) * 200.0 + t * 55.0) as u32;
        let g = (t.clamp(0.0, 1.0) * 255.0) as u32;
        let b = ((1.0 - t) * 200.0) as u32;
        (r.min(255) << 16) | (g.min(255) << 8) | b.min(255)
    }
}

// ── Help ───────────────────────────────────────────────────────

fn print_controls() {
    eprintln!("╔═════════════════════════════════════════════════════════╗");
    eprintln!("║  Controls:                                              ║");
    eprintln!("║  Space        Play / Pause                              ║");
    eprintln!("║  V            Cycle heatmap field                       ║");
    eprintln!("║                                                         ║");
    eprintln!("║  ── Pulses ──────                                       ║");
    eprintln!("║  P            Fire transverse Bx pulse                  ║");
    eprintln!("║  L            Fire IFE laser pulse (pump params)        ║");
    eprintln!("║  X            Clear all queued pulses                   ║");
    eprintln!("║  Left/Right   Pulse strength ±0.5 T (Bx)                ║");
    eprintln!("║                                                         ║");
    eprintln!("║  ── Thermal / integrator ──                             ║");
    eprintln!("║  T            Toggle thermal M3TM                       ║");
    eprintln!("║  G            Toggle LLB (within thermal)               ║");
    eprintln!("║  I            Cycle thermal preset                      ║");
    eprintln!("║  [ / ]        T_ambient ∓ 10 K                          ║");
    eprintln!("║                                                         ║");
    eprintln!("║  ── Sweeps / magnet controls ──                         ║");
    eprintln!("║  A / Z        α ×1.5 / ÷1.5                             ║");
    eprintln!("║  B / N        Bz ±0.1 T                                 ║");
    eprintln!("║  Up / Down    Steps per frame ×2 / ÷2                   ║");
    eprintln!("║                                                         ║");
    eprintln!("║  ── Init / display ──                                   ║");
    eprintln!("║  R  D  U  S  K  Reset: rand/stripe/unif/skyrmion/±z AFM ║");
    eprintln!("║  1/2/3/4      Cycle displayed layer (Nz>1)              ║");
    eprintln!("║  C            Clear plot history                        ║");
    eprintln!("║  Escape       Quit                                      ║");
    eprintln!("╚═════════════════════════════════════════════════════════╝");
}

fn print_help() {
    eprintln!("magnonic-dashboard — interactive simulator");
    eprintln!();
    eprintln!("Stack / geometry:");
    eprintln!("  --material NAME / --substrate NAME / --thickness NM");
    eprintln!("  --stack \"MAT1:T_nm,MAT2:T_nm,...\" --interlayer-a F,F,... --layer-spacing NM,NM,...");
    eprintln!("  --nx N  --ny N");
    eprintln!();
    eprintln!("Dynamics:");
    eprintln!("  --alpha F / --bz T / --dt S");
    eprintln!();
    eprintln!("Thermal / LLB (Phase P3+):");
    eprintln!("  --enable-thermal           attach a ThermalConfig (M3TM observables populated)");
    eprintln!("  --enable-llb               engage LLB integrator (implies --enable-thermal)");
    eprintln!("  --thermal-preset KEY       ni | py | fgt | fgt-zhou | yig | cofeb (default fgt)");
    eprintln!("  --t-ambient K              ambient bath temperature (default 300)");
    eprintln!();
    eprintln!("Pump pulse for the interactive `L` key (D2):");
    eprintln!("  --pump-fwhm FS             temporal FWHM [fs]   (default 100)");
    eprintln!("  --pump-peak T              IFE peak field [T]    (default 0.5)");
    eprintln!("  --pump-dir x|y|z|-x|-y|-z  polarisation          (default z)");
    eprintln!("  --pump-fluence MJ_CM2      absorbed fluence — engages M3TM source");
    eprintln!("  --pump-reflectivity F      reflectivity 0..1     (default 0)");
    eprintln!("  --pump-delay PS            fire at t_now + delay (default 1.0)");
    eprintln!();
    eprintln!("Example: YIG/FGT bilayer dashboard with thermal dynamics:");
    eprintln!("  magnonic-dashboard --stack yig:2,fgt-bulk:0.7 --interlayer-a 4e-13 \\");
    eprintln!("                     --enable-llb --thermal-preset fgt-zhou --t-ambient 150");
}
