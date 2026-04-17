use std::time::Instant;

use minifb::{Key, KeyRepeat, Window, WindowOptions};

use magnonic_clock_sim::config::SimConfig;
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::substrate::Substrate;

const PLOT_HEIGHT: usize = 200;
const LABEL_HEIGHT: usize = 14;

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
                let nm: f64 = args[i + 1].parse().unwrap();
                config.geometry.thickness = nm * 1e-9;
                i += 2;
            }
            "--material" => {
                if let Some(m) = BulkMaterial::lookup(&args[i + 1]) {
                    config.bulk = m;
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
            "--alpha" => { config.bulk.alpha_bulk = args[i + 1].parse().unwrap(); i += 2; }
            "--bz" => { config.b_ext[2] = args[i + 1].parse().unwrap(); i += 2; }
            "--dt" => { config.dt = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }
    config.print_summary();

    let nx = config.geometry.nx as usize;
    let ny = config.geometry.ny as usize;
    let win_w = nx.max(512);
    let win_h = ny + PLOT_HEIGHT + LABEL_HEIGHT;

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

    // Live parameters
    let mut alpha = config.effective().alpha as f32;
    let mut b_ext_x = 0.0f32;
    let mut b_ext_z = config.b_ext[2] as f32;

    // Transverse pulse state
    let mut pulse_strength = 1.0f32; // Tesla
    let mut pulse_frames_left = 0u32;
    let pulse_duration = 5u32; // frames (at 100 steps/frame = 500 steps = 50 ps)

    // History for plot strip
    let mut mz_hist: Vec<f32> = Vec::new();
    let mut mx_hist: Vec<f32> = Vec::new();
    let mut my_hist: Vec<f32> = Vec::new();
    let mut probe_hist: Vec<f32> = Vec::new();

    let mut steps_per_sec = 0.0f64;

    // Initial readback
    let mut mag_data = solver.readback_mag();

    eprintln!("╔══════════════════════════════════════════╗");
    eprintln!("║  Controls:                               ║");
    eprintln!("║  Space     Play / Pause                  ║");
    eprintln!("║  P         Fire transverse Bx pulse      ║");
    eprintln!("║  Left/Right  Pulse strength ±0.5 T       ║");
    eprintln!("║  A / Z     Increase / decrease alpha      ║");
    eprintln!("║  B / N     Increase / decrease Bz ±0.1 T ║");
    eprintln!("║  Up / Down Steps per frame ×2 / ÷2       ║");
    eprintln!("║  R         Reset: random magnetization    ║");
    eprintln!("║  D         Reset: stripe domains          ║");
    eprintln!("║  U         Reset: uniform +z              ║");
    eprintln!("║  C         Clear plot history             ║");
    eprintln!("║  Escape    Quit                           ║");
    eprintln!("╚══════════════════════════════════════════╝");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t0 = Instant::now();

        // ── Input ──────────────────────────────────────────────
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            running = !running;
        }
        if window.is_key_pressed(Key::Up, KeyRepeat::Yes) {
            steps_per_frame = (steps_per_frame * 2).min(10_000);
            eprintln!("steps/frame = {steps_per_frame}");
        }
        if window.is_key_pressed(Key::Down, KeyRepeat::Yes) {
            steps_per_frame = (steps_per_frame / 2).max(1);
            eprintln!("steps/frame = {steps_per_frame}");
        }

        // Damping
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

        // External field z
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

        // Transverse pulse
        if window.is_key_pressed(Key::P, KeyRepeat::No) {
            pulse_frames_left = pulse_duration;
            b_ext_x = pulse_strength;
            solver.set_b_ext(b_ext_x, 0.0, b_ext_z);
            eprintln!("PULSE: Bx = {pulse_strength:.1} T for {pulse_duration} frames");
            if !running { running = true; } // auto-start on pulse
        }
        if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
            pulse_strength = (pulse_strength + 0.5).min(10.0);
            eprintln!("pulse strength = {pulse_strength:.1} T");
        }
        if window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
            pulse_strength = (pulse_strength - 0.5).max(0.1);
            eprintln!("pulse strength = {pulse_strength:.1} T");
        }

        // Reset states
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
        if window.is_key_pressed(Key::C, KeyRepeat::No) {
            mz_hist.clear();
            mx_hist.clear();
            my_hist.clear();
            probe_hist.clear();
            eprintln!("CLEAR: plot history");
        }

        // ── Pulse countdown ────────────────────────────────────
        if pulse_frames_left > 0 {
            pulse_frames_left -= 1;
            if pulse_frames_left == 0 {
                b_ext_x = 0.0;
                solver.set_b_ext(0.0, 0.0, b_ext_z);
                eprintln!("PULSE OFF: Bx = 0");
            }
        }

        // ── Simulate ───────────────────────────────────────────
        if running {
            solver.step_n(steps_per_frame);
            mag_data = solver.readback_mag();
            let obs = solver.observables();

            // Push history
            push_trim(&mut mz_hist, obs.avg_mz, win_w);
            push_trim(&mut mx_hist, obs.avg_mx, win_w);
            push_trim(&mut my_hist, obs.avg_my, win_w);
            push_trim(&mut probe_hist, obs.probe_mz, win_w);

            let dt = t0.elapsed().as_secs_f64();
            if dt > 0.0 {
                steps_per_sec = steps_per_frame as f64 / dt;
            }

            let pulse_tag = if pulse_frames_left > 0 {
                format!(" | PULSE Bx={b_ext_x:.1}T [{pulse_frames_left}]")
            } else {
                String::new()
            };

            window.set_title(&format!(
                "Step {} | {:.1}ps | Mz={:.4} | probe={:.4} | α={:.4} | Bz={:.1}T{} | {:.0}st/s | {}/fr",
                obs.step, obs.time_ps, obs.avg_mz, obs.probe_mz,
                alpha, b_ext_z, pulse_tag, steps_per_sec, steps_per_frame,
            ));
        }

        // ── Render ─────────────────────────────────────────────
        draw_heatmap(&mut framebuf, &mag_data, nx, ny, win_w);
        draw_label_bar(&mut framebuf, win_w, ny, LABEL_HEIGHT, pulse_frames_left > 0);
        draw_plot_strip(
            &mut framebuf, win_w, ny + LABEL_HEIGHT, PLOT_HEIGHT,
            &mz_hist, &mx_hist, &my_hist, &probe_hist,
        );

        window.update_with_buffer(&framebuf, win_w, win_h).unwrap();
    }
}

fn push_trim(v: &mut Vec<f32>, val: f32, max: usize) {
    v.push(val);
    if v.len() > max { v.remove(0); }
}

// ─── Rendering ─────────────────────────────────────────────────

fn draw_heatmap(buf: &mut [u32], mag: &[f32], nx: usize, ny: usize, stride: usize) {
    for iy in 0..ny {
        for ix in 0..nx {
            let cell = ix * ny + iy;
            let mz = mag[cell * 4 + 2];
            buf[iy * stride + ix] = mz_to_rgb(mz);
        }
    }
    for y in 0..ny {
        for x in nx..stride {
            buf[y * stride + x] = 0x1a1a1a;
        }
    }
}

fn draw_label_bar(buf: &mut [u32], width: usize, y_offset: usize, height: usize, pulsing: bool) {
    let bg = if pulsing { 0x442200 } else { 0x111111 };
    for y in y_offset..y_offset + height {
        for x in 0..width {
            buf[y * width + x] = bg;
        }
    }
    // Color legend dots
    let y_mid = y_offset + height / 2;
    let colors = [
        (0xDD3333, 20),  // red: avg Mz
        (0x3366DD, 100), // blue: avg Mx
        (0x33BB55, 180), // green: avg My
        (0xCC9933, 260), // amber: probe Mz
    ];
    for (color, x_start) in colors {
        for dx in 0..8 {
            for dy in 0..3 {
                let x = x_start + dx;
                let y = y_mid - 1 + dy;
                if x < width && y < y_offset + height {
                    buf[y * width + x] = color;
                }
            }
        }
    }
}

fn draw_plot_strip(
    buf: &mut [u32],
    width: usize,
    y_offset: usize,
    height: usize,
    mz: &[f32],
    mx: &[f32],
    my: &[f32],
    probe: &[f32],
) {
    let bg = 0x1a1a1a;
    for y in y_offset..y_offset + height {
        for x in 0..width {
            buf[y * width + x] = bg;
        }
    }

    // Zero line (solid)
    let zero_y = y_offset + height / 2;
    for x in 0..width {
        buf[zero_y * width + x] = 0x555555;
    }

    // ±0.5 gridlines (dotted)
    let q1_y = y_offset + height / 4;
    let q3_y = y_offset + 3 * height / 4;
    for x in (0..width).step_by(4) {
        buf[q1_y * width + x] = 0x333333;
        buf[q3_y * width + x] = 0x333333;
    }

    // +1 / -1 border lines (thin dots)
    for x in (0..width).step_by(8) {
        buf[y_offset * width + x] = 0x333333;
        buf[(y_offset + height - 1) * width + x] = 0x333333;
    }

    let plot_line = |buf: &mut [u32], data: &[f32], color: u32, thick: bool| {
        let n = data.len();
        let x_start = width.saturating_sub(n);
        for (i, &val) in data.iter().enumerate() {
            let x = x_start + i;
            if x >= width { continue; }
            let t = ((val + 1.0) * 0.5).clamp(0.0, 1.0);
            let y = y_offset + height - 1 - ((t * (height - 1) as f32) as usize).min(height - 1);
            buf[y * width + x] = color;
            if thick {
                if y > y_offset { buf[(y - 1) * width + x] = color; }
                if y + 1 < y_offset + height { buf[(y + 1) * width + x] = color; }
            }
        }
    };

    plot_line(buf, mx, 0x3366DD, false);    // blue: avg Mx
    plot_line(buf, my, 0x33BB55, false);    // green: avg My
    plot_line(buf, probe, 0xCC9933, false); // amber: probe Mz
    plot_line(buf, mz, 0xDD3333, true);     // red: avg Mz (thick, on top)
}

fn mz_to_rgb(mz: f32) -> u32 {
    let t = ((mz + 1.0) * 0.5).clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        ((s * 255.0) as u32, (s * 255.0) as u32, 255u32)
    } else {
        let s = (t - 0.5) * 2.0;
        (255u32, ((1.0 - s) * 255.0) as u32, ((1.0 - s) * 255.0) as u32)
    };
    (r << 16) | (g << 8) | b
}
