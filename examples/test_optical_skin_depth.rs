//! Phase F3 acceptance — Beer-Lambert per-layer absorption profile.
//!
//! Two configurations on the same single-cell-per-layer 2-layer stack:
//!
//!   A) Uniform mode (skin_depth = 0): both layers see the same
//!      "P_in / t_i" volumetric source. Pre-F3 behaviour.
//!   B) Beer-Lambert (skin_depth = 14 nm): light is incident from the top
//!      (layer 1). After traversing 20 nm of attenuating material it
//!      reaches the bottom layer (layer 0) much weaker. The bottom-layer
//!      T_e rise is therefore strictly smaller than the top-layer rise,
//!      and smaller than in case A where uniform mode gave both layers
//!      the full surface power.
//!
//! Pass criteria:
//!   - Case A: T_e peaks of both layers are equal within 1 K (uniform).
//!   - Case B: layer-0 T_e peak < layer-1 T_e peak by > 50 K
//!     (cumulative attenuation through 20 nm at δ = 14 nm:
//!      e^(-20/14) ≈ 0.24, so bottom layer sees ~24 % of surface power).
//!   - Case B's bottom-layer T_e peak is strictly less than case A's.

use magnonic_clock_sim::config::{Geometry, Layer, SimConfig, Stack};
use magnonic_clock_sim::gpu::GpuSolver;
use magnonic_clock_sim::material::BulkMaterial;
use magnonic_clock_sim::material_thermal;
use magnonic_clock_sim::photonic::{parse_pulse_spec, ThermalConfig};
use magnonic_clock_sim::substrate::Substrate;

#[derive(Clone, Debug)]
struct PerLayerPeak {
    label: &'static str,
    layer_top_te_peak: f32,
    layer_bot_te_peak: f32,
}

fn run(label: &'static str, skin_depth_m: f64) -> PerLayerPeak {
    let mut preset = material_thermal::ni_m3tm();
    preset.optical_skin_depth_m = skin_depth_m;

    // Two-layer stack: layer 0 = bottom (20 nm Ni-like), layer 1 = top (20 nm Ni-like).
    // Light enters the top of layer 1 (incidence on the top surface of the stack).
    let bulk = BulkMaterial::permalloy_bulk();
    let mut bulk_with_alpha = bulk;
    bulk_with_alpha.alpha_bulk = preset.alpha_0;
    let layers = vec![
        Layer { material: bulk_with_alpha.clone(), thickness: 20e-9, u_axis: [0.0, 0.0, 1.0] },
        Layer { material: bulk_with_alpha,         thickness: 20e-9, u_axis: [0.0, 0.0, 1.0] },
    ];
    let stack = Stack {
        layers,
        interlayer_a: vec![0.0],
        layer_spacing: vec![0.7e-9],
    };

    let mut cfg = SimConfig {
        stack,
        substrate: Substrate::vacuum(),
        // 1×1 in-plane × 2 layers ⇒ 2 cells, one per layer.
        geometry: Geometry { nx: 1, ny: 1, cell_size: 20e-9 },
        dt: 1.0e-15,
        b_ext: [0.0, 0.0, 0.0],
        stab_coeff: 0.0,
        j_current: [0.0, 0.0, 0.0],
        photonic: Default::default(),
        readback_interval: 1,
        total_steps: 3_000, // 3 ps
        probe_idx: None,
        probe_layer: None,
    };
    cfg.photonic.pulses.push(
        parse_pulse_spec("t=300fs,fwhm=100fs,peak=0.0,dir=z,fluence=1.0,R=0.0").unwrap(),
    );
    cfg.photonic.thermal = Some(ThermalConfig {
        t_ambient: 300.0,
        per_layer: vec![preset.clone(); 2],
        thermal_dt_cap: 1.0e-15,
        thermal_window: (0.5e-12, 10.0e-12),
        enable_llb: false,
    });

    let mut solver = GpuSolver::new(&cfg).expect("GPU init");
    solver.reset_uniform_z();

    let mut top_peak = 0.0_f32;
    let mut bot_peak = 0.0_f32;
    for _ in 0..cfg.total_steps {
        solver.step_n(1);
        let te = solver.readback_temp_e();
        // Cell layout: layer-flat. cell[0] = layer 0 (bottom), cell[1] = layer 1 (top).
        let bot = te[0];
        let top = te[1];
        if bot > bot_peak { bot_peak = bot; }
        if top > top_peak { top_peak = top; }
    }
    PerLayerPeak {
        label,
        layer_top_te_peak: top_peak,
        layer_bot_te_peak: bot_peak,
    }
}

fn main() {
    println!("=== F3: optical skin-depth absorption profile probe ===\n");
    let case_a = run("[A] uniform mode (skin_depth = 0)", 0.0);
    let case_b = run("[B] Beer-Lambert (skin_depth = 14 nm)", 14.0e-9);

    for r in [&case_a, &case_b] {
        println!(
            "{:42} top T_e peak = {:7.2} K  bot T_e peak = {:7.2} K  (Δ = {:+.2} K)",
            r.label,
            r.layer_top_te_peak,
            r.layer_bot_te_peak,
            r.layer_top_te_peak - r.layer_bot_te_peak,
        );
    }

    // Acceptance: case A = uniform → top and bottom layers receive identical
    // P_vol → identical T_e peaks within float precision.
    let a_diff = (case_a.layer_top_te_peak - case_a.layer_bot_te_peak).abs();
    assert!(
        a_diff < 1.0,
        "case A (uniform mode) should give identical top/bot peaks; |Δ| = {a_diff:.3}",
    );

    // Case B: top layer sees full surface flux, bottom layer sees attenuated.
    // For δ = 14 nm and t = 20 nm, transmittance ≈ 0.24 → expect bottom peak
    // visibly below top peak.
    let b_diff = case_b.layer_top_te_peak - case_b.layer_bot_te_peak;
    assert!(
        b_diff > 50.0,
        "case B (Beer-Lambert) should show large top-vs-bottom split; got Δ = {b_diff:.2} K",
    );

    // Bottom layer in B should heat less than in A (attenuation actually attenuates).
    assert!(
        case_b.layer_bot_te_peak < case_a.layer_bot_te_peak - 50.0,
        "case B bottom peak ({:.2}) should be markedly below case A bottom peak ({:.2})",
        case_b.layer_bot_te_peak,
        case_a.layer_bot_te_peak,
    );

    println!(
        "\nPASS: F3 Beer-Lambert produces top-vs-bottom split (Δ = {:.0} K) and the\n\
         bottom layer in Beer-Lambert mode receives less energy than uniform mode.",
        b_diff,
    );
}
