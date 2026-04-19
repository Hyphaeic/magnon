use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::config::{SimConfig, Stack};

/// Pack per-layer LLB tables into flat `MAX_LAYERS × LLB_TABLE_N` buffers.
/// Missing or disabled layers receive m_e = 1.0, χ_∥ = 0.0 — i.e. "frozen
/// at T = 0", a safe identity when the M3TM kernel is not dispatched.
fn pack_llb_tables(cfg: &SimConfig) -> (Vec<f32>, Vec<f32>) {
    let total = MAX_LAYERS * LLB_TABLE_N;
    let mut m_e = vec![1.0f32; total];
    let mut chi = vec![0.0f32; total];
    let thermal = match &cfg.photonic.thermal {
        Some(t) => t,
        None => return (m_e, chi),
    };
    for (iz, p) in thermal.per_layer.iter().take(MAX_LAYERS).enumerate() {
        let n = p.llb_table_n.min(LLB_TABLE_N);
        let base = iz * LLB_TABLE_N;
        for i in 0..n {
            m_e[base + i] = p.m_e_table.get(i).copied().unwrap_or(0.0);
            chi[base + i] = p.chi_par_table.get(i).copied().unwrap_or(0.0);
        }
    }
    (m_e, chi)
}

// ─── GPU-side parameter struct (must match WGSL Params exactly) ────
//
// Layout strategy (multilayer, M1):
//   - Globals first (grid, dt, γ, stab_coeff, B_ext)
//   - Per-layer scalar params packed into vec4 (4 layers per vec4 slot)
//   - Per-layer vec3 params stored as arrays of vec4 (16-byte stride)
// MAX_LAYERS = 4 sets the array sizes. Layers beyond the stack's layer
// count are zeroed and ignored by the shader (which iterates iz < nz).

const MAX_LAYERS: usize = Stack::MAX_LAYERS;

/// Rows in the LLB (m_e, χ_∥) lookup tables. Uniform T grid on [0, 1.5·T_c].
/// Matches the default in `material_thermal::brillouin_tables_spin_half`.
pub const LLB_TABLE_N: usize = 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    // 0-15: grid
    nx: u32,
    ny: u32,
    nz: u32,
    cell_count: u32,

    // 16-31: dynamics
    dt: f32,
    gamma: f32,
    stab_coeff: f32,
    // cell_size in meters (in-plane dx = dy). Needed by the shader for
    // computing cell positions during the per-pulse Gaussian-weight sum.
    cell_size_m: f32,

    // 32-47: B_ext (global Zeeman, applies to all layers)
    b_ext: [f32; 4],

    // 48-63: packed per-layer α                           (vec4 — 4 layers)
    layer_alphas: [f32; 4],
    // 64-79: packed per-layer in-plane exchange prefactor (vec4)
    layer_exchange_pfs: [f32; 4],
    // 80-95: packed per-layer anisotropy prefactor
    layer_anisotropy_pfs: [f32; 4],
    // 96-111: packed per-layer DMI prefactor
    layer_dmi_pfs: [f32; 4],
    // 112-127: packed per-layer SOT damping-like
    layer_sot_tau_dls: [f32; 4],
    // 128-143: packed per-layer SOT field-like
    layer_sot_tau_fls: [f32; 4],
    // 144-159: packed per-layer thicknesses [m]
    layer_thicknesses: [f32; 4],
    // 160-175: packed per-layer interlayer-exchange prefactor for interface BELOW
    //          (zero for layer 0). See effective.rs:interlayer_prefactor_below.
    layer_ilx_below: [f32; 4],
    // 176-191: packed per-layer interlayer-exchange prefactor for interface ABOVE
    //          (zero for layer Nz-1).
    layer_ilx_above: [f32; 4],

    // 192-255: per-layer anisotropy axes (4 × vec4, xyz+pad)
    layer_u_axes: [[f32; 4]; MAX_LAYERS],
    // 256-319: per-layer B_bias (4 × vec4)
    layer_b_biases: [[f32; 4]; MAX_LAYERS],
    // 320-383: per-layer SOT σ direction (4 × vec4)
    layer_sigmas: [[f32; 4]; MAX_LAYERS],

    // ─── Phase P2 — per-pulse photonic uniforms ─────────────────────────
    // Up to MAX_PULSES = 4 active pulses. Superseded the P1 single-b_laser
    // field: now the shader computes per-cell B_laser as the Gaussian-
    // weighted sum over all active pulses. Pulses with spot_sigma = 0 are
    // treated as spatially uniform (P1-equivalent).

    // 384-399: active pulse count + P3c flag + padding.
    // `enable_llb_flag` (offset 388) engages the LLB back-coupling in the
    // M3TM kernel — when 1, advance_m3tm reads |m| from `mag` and mirrors
    // `m_reduced` to it; when 0, m_reduced is the M3TM's independent track
    // (P3a behaviour).
    pulse_count: u32,
    enable_llb_flag: u32,
    _pad_pc1: u32,
    _pad_pc2: u32,

    // 400-415: per-pulse current-step amplitude [T], packed 4 pulses per vec4.
    // Host rewrites per step from Gaussian temporal envelope.
    pulse_amplitudes: [f32; 4],

    // 416-479: per-pulse spot data (4 × vec4). Each entry is (x, y, σ_r, _pad)
    // in meters. Spot is where the beam is centered in the grid; σ_r is the
    // 1-σ radius. σ_r = 0 ⇒ spatially uniform pulse.
    pulse_spot_centers: [[f32; 4]; 4],

    // 480-543: per-pulse propagation direction (4 × vec4). Unit vectors;
    // normalized on ingest. k̂ direction of IFE-induced B_eff.
    pulse_directions: [[f32; 4]; 4],

    // ─── Phase P3 — per-layer thermal (M3TM + LLB) scalars ───────────────
    // Appended at the end to preserve the P1-P2 layout (pulse offsets stay
    // at 400+). See ADR-003 for the rationale vs. the plan's "shift pulse
    // offset" variant.
    //
    // When `photonic.thermal.is_none()` these slots are zero — the M3TM
    // kernel is not dispatched, so the values are never read.

    // 544-559: γ_e per layer [J/(m³·K²)]
    layer_thermal_gamma_e: [f32; 4],
    // 560-575: C_p per layer [J/(m³·K)]
    layer_thermal_c_p: [f32; 4],
    // 576-591: g_ep per layer [W/(m³·K)]
    layer_thermal_g_ep: [f32; 4],
    // 592-607: precomputed Koopmans R prefactor per layer [1/s]
    layer_thermal_a_sf_r: [f32; 4],
    // 608-623: Curie temperature per layer [K]
    layer_thermal_t_c: [f32; 4],
    // 624-639: low-T Gilbert damping α_0 per layer
    layer_thermal_alpha_0: [f32; 4],
    // 640-655: P3b — per-layer longitudinal-relaxation base time [s].
    //   τ_∥(T) = tau_long_base / α_∥(T); α_∥(T) = α_0·(2T/(3·T_c)).
    //   At T = 0 → α_∥ = 0 → τ_∥ = ∞ (LLG reduction).
    layer_thermal_tau_long: [f32; 4],
    // 656-671: per-layer phonon-substrate coupling [W/(m³·K)] (substrate sink).
    //   dT_p/dt |_sub = −g_sub_phonon · (T_p − t_ambient) / c_p
    layer_thermal_g_sub_p: [f32; 4],
    // 672-687: thermal globals — (t_ambient, pad, pad, pad).
    //   t_ambient is shared across layers (bath reference for substrate
    //   sink); other slots reserved for future globals.
    thermal_globals: [f32; 4],
}

const OFFSET_PULSE_AMPLITUDES: u64 = 400;
const OFFSET_GPUPARAMS_END: u64 = 688;
const _: () = assert!(OFFSET_GPUPARAMS_END as usize == std::mem::size_of::<GpuParams>());

impl GpuParams {
    fn from_config(cfg: &SimConfig) -> Self {
        let eff = cfg.effective();
        let nz = cfg.nz() as usize;
        assert!(nz <= MAX_LAYERS, "Stack has {} layers; MAX_LAYERS = {}", nz, MAX_LAYERS);

        let mut layer_alphas = [0.0f32; 4];
        let mut layer_exchange_pfs = [0.0f32; 4];
        let mut layer_anisotropy_pfs = [0.0f32; 4];
        let mut layer_dmi_pfs = [0.0f32; 4];
        let mut layer_sot_tau_dls = [0.0f32; 4];
        let mut layer_sot_tau_fls = [0.0f32; 4];
        let mut layer_thicknesses = [0.0f32; 4];
        let mut layer_u_axes = [[0.0f32; 4]; MAX_LAYERS];
        let mut layer_b_biases = [[0.0f32; 4]; MAX_LAYERS];
        let mut layer_sigmas = [[0.0f32; 4]; MAX_LAYERS];

        for (i, l) in eff.layers.iter().enumerate() {
            layer_alphas[i] = l.alpha as f32;
            layer_exchange_pfs[i] =
                eff.exchange_prefactor(i, cfg.geometry.cell_size) as f32;
            layer_anisotropy_pfs[i] = eff.anisotropy_prefactor(i) as f32;
            layer_dmi_pfs[i] = eff.dmi_prefactor(i, cfg.geometry.cell_size) as f32;

            let (tdl, tfl, sigma) = eff.sot_coefficients(i, cfg.j_current);
            layer_sot_tau_dls[i] = tdl as f32;
            layer_sot_tau_fls[i] = tfl as f32;
            layer_sigmas[i] = [sigma[0] as f32, sigma[1] as f32, sigma[2] as f32, 0.0];

            layer_thicknesses[i] = l.thickness as f32;

            // Normalize u_axis at upload
            let [ux, uy, uz] = l.u_axis;
            let norm = (ux * ux + uy * uy + uz * uz).sqrt();
            let (ax, ay, az) = if norm > 1e-12 {
                (ux / norm, uy / norm, uz / norm)
            } else {
                (0.0, 0.0, 1.0)
            };
            layer_u_axes[i] = [ax as f32, ay as f32, az as f32, 0.0];

            layer_b_biases[i] = [
                l.b_bias[0] as f32, l.b_bias[1] as f32, l.b_bias[2] as f32, 0.0,
            ];
        }

        // Per-layer interlayer-exchange prefactors (M2 — active)
        let mut layer_ilx_below = [0.0f32; 4];
        let mut layer_ilx_above = [0.0f32; 4];
        for k in 0..nz {
            layer_ilx_below[k] = eff.interlayer_prefactor_below(k) as f32;
            layer_ilx_above[k] = eff.interlayer_prefactor_above(k) as f32;
        }

        // B_ext is global
        let gamma = eff.layers.first().map(|l| l.gamma).unwrap_or(1.7595e11);

        Self {
            nx: cfg.geometry.nx,
            ny: cfg.geometry.ny,
            nz: cfg.nz(),
            cell_count: cfg.cell_count(),
            dt: cfg.dt as f32,
            gamma: gamma as f32,
            stab_coeff: cfg.stab_coeff as f32,
            cell_size_m: cfg.geometry.cell_size as f32,
            b_ext: [cfg.b_ext[0] as f32, cfg.b_ext[1] as f32, cfg.b_ext[2] as f32, 0.0],
            layer_alphas,
            layer_exchange_pfs,
            layer_anisotropy_pfs,
            layer_dmi_pfs,
            layer_sot_tau_dls,
            layer_sot_tau_fls,
            layer_thicknesses,
            layer_ilx_below,
            layer_ilx_above,
            layer_u_axes,
            layer_b_biases,
            layer_sigmas,
            // Photonic P2 — populate per-pulse uniforms
            pulse_count: {
                let n = cfg.photonic.pulses.len();
                if n > 4 {
                    eprintln!(
                        "WARNING: {n} pulses specified but MAX_PULSES = 4; ignoring extras",
                    );
                }
                n.min(4) as u32
            },
            enable_llb_flag: cfg
                .photonic
                .thermal
                .as_ref()
                .map_or(0, |t| if t.enable_llb { 1 } else { 0 }),
            _pad_pc1: 0,
            _pad_pc2: 0,
            // Amplitudes start at 0; host writes them per step from envelope
            pulse_amplitudes: [0.0; 4],
            pulse_spot_centers: {
                let mut centers = [[0.0f32; 4]; 4];
                for (i, p) in cfg.photonic.pulses.iter().take(4).enumerate() {
                    centers[i] = [
                        p.spot_center[0] as f32,
                        p.spot_center[1] as f32,
                        p.spot_sigma as f32,
                        0.0,
                    ];
                }
                centers
            },
            pulse_directions: {
                let mut dirs = [[0.0f32; 4]; 4];
                for (i, p) in cfg.photonic.pulses.iter().take(4).enumerate() {
                    dirs[i] = [p.direction[0], p.direction[1], p.direction[2], 0.0];
                }
                dirs
            },
            // P3 thermal scalars (zero when thermal disabled)
            layer_thermal_gamma_e: Self::thermal_vec4(cfg, |p| p.gamma_e as f32),
            layer_thermal_c_p: Self::thermal_vec4(cfg, |p| p.c_p as f32),
            layer_thermal_g_ep: Self::thermal_vec4(cfg, |p| p.g_ep as f32),
            layer_thermal_a_sf_r: Self::thermal_vec4(cfg, |p| p.r_koopmans_prefactor() as f32),
            layer_thermal_t_c: Self::thermal_vec4(cfg, |p| p.t_c as f32),
            layer_thermal_alpha_0: Self::thermal_vec4(cfg, |p| p.alpha_0 as f32),
            layer_thermal_tau_long: Self::thermal_vec4(cfg, |p| p.tau_long_base as f32),
            layer_thermal_g_sub_p: Self::thermal_vec4(cfg, |p| p.g_sub_phonon as f32),
            thermal_globals: {
                let t_amb = cfg
                    .photonic
                    .thermal
                    .as_ref()
                    .map(|t| t.t_ambient)
                    .unwrap_or(300.0);
                [t_amb, 0.0, 0.0, 0.0]
            },
        }
    }

    fn thermal_vec4<F>(cfg: &SimConfig, f: F) -> [f32; 4]
    where
        F: Fn(&crate::photonic::LayerThermalParams) -> f32,
    {
        let mut out = [0.0f32; 4];
        if let Some(t) = &cfg.photonic.thermal {
            for (i, p) in t.per_layer.iter().take(MAX_LAYERS).enumerate() {
                out[i] = f(p);
            }
        }
        out
    }
}

// ─── Observables ───────────────────────────────────────────────

pub struct Observables {
    pub step: usize,
    pub time_ps: f64,
    /// Averages over ALL layers and all cells
    pub avg_mx: f32,
    pub avg_my: f32,
    pub avg_mz: f32,
    pub min_norm: f32,
    pub max_norm: f32,
    /// Mz at the configured probe cell (single cell on a single layer)
    pub probe_mz: f32,

    // ─── Phase P3 thermal observables ───────────────
    // Populated when `photonic.thermal.is_some()`; all zero otherwise.
    pub max_t_e: f32,
    pub max_t_p: f32,
    pub min_m_reduced: f32,
}

impl std::fmt::Display for Observables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},{:.4},{:.6},{:.6},{:.6},{:.8},{:.8},{:.6},{:.2},{:.2},{:.6}",
            self.step, self.time_ps, self.avg_mx, self.avg_my, self.avg_mz,
            self.min_norm, self.max_norm, self.probe_mz,
            self.max_t_e, self.max_t_p, self.min_m_reduced,
        )
    }
}

// ─── GPU solver ────────────────────────────────────────────────

pub struct GpuSolver {
    config: SimConfig,

    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    params_buf: wgpu::Buffer,
    mag_buf: wgpu::Buffer,

    // Buffers below are bound via `bind_group` but never touched through
    // the solver handle after creation — they exist to keep wgpu's buffer
    // handles alive for the bind group. The compiler can't see that and
    // would warn; the `_` prefix silences that cleanly.
    _mag_pred_buf: wgpu::Buffer,
    _h_eff_buf: wgpu::Buffer,
    _torque_0_buf: wgpu::Buffer,
    _torque_1_buf: wgpu::Buffer,

    // Phase P3 thermal buffers — always allocated to keep the BGL stable,
    // but only dispatched to when `config.photonic.thermal.is_some()`.
    temp_e_buf: wgpu::Buffer,
    temp_p_buf: wgpu::Buffer,
    m_reduced_buf: wgpu::Buffer,
    // Table buffers are bound through `bind_group`; no direct writes after
    // creation. Hence the `_` prefix. `chi_par_table` is currently unused
    // by the shader but retained for a future full-Atxitia LLB upgrade —
    // see docs/zhou_fgt_calibration.md §6.
    _m_e_table_buf: wgpu::Buffer,
    _chi_par_table_buf: wgpu::Buffer,

    mag_staging: wgpu::Buffer,
    temp_e_staging: wgpu::Buffer,
    m_reduced_staging: wgpu::Buffer,

    ft_phase0_pipeline: wgpu::ComputePipeline,
    ft_phase1_pipeline: wgpu::ComputePipeline,
    predict_pipeline: wgpu::ComputePipeline,
    correct_pipeline: wgpu::ComputePipeline,
    m3tm_pipeline: wgpu::ComputePipeline,
    // P3b — LLB path pipelines. Dispatched instead of LLG heun_* when
    // `photonic.thermal.as_ref().map_or(false, |t| t.enable_llb)`.
    ft_phase0_llb_pipeline: wgpu::ComputePipeline,
    ft_phase1_llb_pipeline: wgpu::ComputePipeline,
    llb_predict_pipeline: wgpu::ComputePipeline,
    llb_correct_pipeline: wgpu::ComputePipeline,

    bind_group: wgpu::BindGroup,

    cell_count: u32,
    workgroups: u32,
    step: usize,
    /// Running wall-clock simulation time [s]; advances by `dt` per step.
    t_sim: f64,
}

impl GpuSolver {
    /// Create solver with its own GPU device (headless mode).
    pub fn new(config: &SimConfig) -> Result<Self, String> {
        config.stack.validate()?;

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("No GPU adapter: {e}"))?;

        eprintln!("GPU adapter: {}", adapter.get_info().name);

        // P3 needs 10 storage buffers (5 LLG + 3 thermal r/w + 2 LLB tables);
        // default limit is 8, so request 16 — every discrete GPU we target
        // reports ≥64 per the wgpu compat table.
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(16);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("magnonic-llg"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            },
        ))
        .map_err(|e| format!("Failed to create device: {e}"))?;

        Self::from_device_queue(Arc::new(device), Arc::new(queue), config)
    }

    pub fn from_device_queue(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: &SimConfig,
    ) -> Result<Self, String> {
        config.stack.validate()?;

        let cell_count = config.cell_count();
        let buf_bytes = (cell_count as u64) * 4 * 4; // 4 f32 per cell

        let mag_data = Self::init_magnetization(config);

        let gpu_params = GpuParams::from_config(config);

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mag_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mag"),
            contents: bytemuck::cast_slice(&mag_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let zeros = vec![0.0f32; (cell_count * 4) as usize];

        let mag_pred_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mag_pred"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let h_eff_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("h_eff"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let torque_0_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("torque_0"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let torque_1_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("torque_1"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mag_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mag_staging"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ─── Phase P3 thermal buffers ─────────────────────────────────
        let scalar_bytes = (cell_count as u64) * 4; // 1 f32 per cell
        let table_bytes = (MAX_LAYERS * LLB_TABLE_N * 4) as u64;

        let thermal = config.photonic.thermal.as_ref();
        let t_ambient = thermal.map(|t| t.t_ambient).unwrap_or(300.0);

        let init_temp = vec![t_ambient; cell_count as usize];
        let temp_e_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("temp_e"),
            contents: bytemuck::cast_slice(&init_temp),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let temp_p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("temp_p"),
            contents: bytemuck::cast_slice(&init_temp),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Initialize m_reduced per-cell at m_e(T_ambient) of that cell's layer.
        // When thermal is None, every cell holds 1.0 as a harmless default.
        let m_red_init: Vec<f32> = {
            let mut v = vec![1.0f32; cell_count as usize];
            if let Some(tc) = thermal {
                let in_plane = (config.geometry.nx * config.geometry.ny) as usize;
                for cell in 0..cell_count as usize {
                    let iz = cell / in_plane;
                    if let Some(p) = tc.per_layer.get(iz) {
                        v[cell] = p.sample_m_e(t_ambient as f64) as f32;
                    }
                }
            }
            v
        };
        let m_reduced_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("m_reduced"),
            contents: bytemuck::cast_slice(&m_red_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Pack m_e / χ_∥ tables: row-major [layer][row].
        let (m_e_data, chi_par_data) = pack_llb_tables(config);
        let m_e_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("m_e_table"),
            contents: bytemuck::cast_slice(&m_e_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let chi_par_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chi_par_table"),
            contents: bytemuck::cast_slice(&chi_par_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let _ = (scalar_bytes, table_bytes); // kept for doc; sizes implicit in init data

        let temp_e_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temp_e_staging"),
            size: (cell_count as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let m_reduced_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("m_reduced_staging"),
            size: (cell_count as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("llg"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/llg.wgsl").into()),
        });

        // 11 bindings: 0 uniform + 1..5 LLG storage + 6..8 thermal storage
        // (r/w) + 9..10 LLB lookup tables (read-only).
        let bgl_entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("llg_bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(7, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(8, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(9, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(10, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("llg_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let ft_phase0_pipeline = make_pipeline("field_torque_phase0");
        let ft_phase1_pipeline = make_pipeline("field_torque_phase1");
        let predict_pipeline = make_pipeline("heun_predict");
        let correct_pipeline = make_pipeline("heun_correct");
        let m3tm_pipeline = make_pipeline("advance_m3tm");
        let ft_phase0_llb_pipeline = make_pipeline("field_torque_phase0_llb");
        let ft_phase1_llb_pipeline = make_pipeline("field_torque_phase1_llb");
        let llb_predict_pipeline = make_pipeline("llb_predict");
        let llb_correct_pipeline = make_pipeline("llb_correct");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("llg_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: mag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: mag_pred_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: h_eff_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: torque_0_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: torque_1_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: temp_e_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: temp_p_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: m_reduced_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: m_e_table_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: chi_par_table_buf.as_entire_binding() },
            ],
        });

        let workgroups = (cell_count + 63) / 64;

        Ok(Self {
            config: config.clone(),
            device,
            queue,
            params_buf,
            mag_buf,
            _mag_pred_buf: mag_pred_buf,
            _h_eff_buf: h_eff_buf,
            _torque_0_buf: torque_0_buf,
            _torque_1_buf: torque_1_buf,
            temp_e_buf,
            temp_p_buf,
            m_reduced_buf,
            _m_e_table_buf: m_e_table_buf,
            _chi_par_table_buf: chi_par_table_buf,
            mag_staging,
            temp_e_staging,
            m_reduced_staging,
            ft_phase0_pipeline,
            ft_phase1_pipeline,
            predict_pipeline,
            correct_pipeline,
            m3tm_pipeline,
            ft_phase0_llb_pipeline,
            ft_phase1_llb_pipeline,
            llb_predict_pipeline,
            llb_correct_pipeline,
            bind_group,
            cell_count,
            workgroups,
            step: 0,
            t_sim: 0.0,
        })
    }

    /// Initialize magnetization: uniform +z with 5° random perturbation.
    /// Applies the same profile to every layer.
    fn init_magnetization(config: &SimConfig) -> Vec<f32> {
        use rand::Rng;
        let n = config.cell_count() as usize;
        let mut data = vec![0.0f32; n * 4];
        let mut rng = rand::thread_rng();
        for i in 0..n {
            let theta: f32 = rng.gen_range(0.0..0.087);
            let phi: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let st = theta.sin();
            data[i * 4] = st * phi.cos();
            data[i * 4 + 1] = st * phi.sin();
            data[i * 4 + 2] = theta.cos();
        }
        data
    }

    /// Run N Heun steps.
    ///
    /// Two internal paths:
    /// - **Fast batched path** (no photonic pulses): all N steps submitted in a
    ///   single command buffer — ~13-15k steps/s on an RTX 4060.
    /// - **Per-step path** (photonic.pulses non-empty): b_laser uniform is
    ///   rewritten before each step with the Gaussian-envelope value at the
    ///   step midpoint t_n + dt/2. Each step is its own submission. ~5× slower
    ///   but necessary for time-dependent forcing.
    ///
    /// The midpoint-time approximation is O(dt²) in the pulse envelope — the
    /// same order as Heun's truncation error — so it introduces no additional
    /// accuracy loss for realistic pulses (fs-scale envelope, fs-scale dt).
    pub fn step_n(&mut self, n: usize) {
        let wg = self.workgroups;
        let has_pulses = !self.config.photonic.pulses.is_empty();
        let has_thermal = self.config.photonic.thermal.is_some();

        if !has_pulses && !has_thermal {
            // Fast batched path — unchanged from pre-P1. This is the
            // bit-identical LLG regression path: no pulse amps written,
            // no M3TM dispatch, no dt cap.
            let mut encoder = self.device.create_command_encoder(&Default::default());
            for _ in 0..n {
                for pipeline in [
                    &self.ft_phase0_pipeline,
                    &self.predict_pipeline,
                    &self.ft_phase1_pipeline,
                    &self.correct_pipeline,
                ] {
                    let mut p = encoder.begin_compute_pass(&Default::default());
                    p.set_pipeline(pipeline);
                    p.set_bind_group(0, &self.bind_group, &[]);
                    p.dispatch_workgroups(wg, 1, 1);
                }
            }
            self.queue.submit(Some(encoder.finish()));
            self.step += n;
            self.t_sim += (n as f64) * self.config.dt;
            return;
        }

        // Per-step path — per-pulse amplitudes must be rewritten before each
        // Heun step regardless of whether M3TM is also active (they share the
        // same `pulse_amplitudes` uniform).
        //
        // When a pulse also carries `peak_fluence`, the host also re-writes
        // `pulse_directions[p].w` to the instantaneous absorbed volumetric
        // power density [W/m³] at the step midpoint — the M3TM kernel reads
        // this channel via `laser_power_density_at`. ADR-003 documents this
        // reuse of the direction-vec4 w-component.
        let n_pulses = self.config.photonic.pulses.len().min(4);
        let layer0_thickness = self.config.stack.layers.first().map(|l| l.thickness).unwrap_or(1e-9);
        for _ in 0..n {
            let dt = self.config.dt;
            let t_mid = self.t_sim + 0.5 * dt;
            let mut amps = [0.0f32; 4];
            for (i, pulse) in self.config.photonic.pulses.iter().take(n_pulses).enumerate() {
                let env = pulse.envelope_at(t_mid);
                amps[i] = (pulse.peak_field as f64 * env) as f32;
            }
            self.queue.write_buffer(
                &self.params_buf,
                OFFSET_PULSE_AMPLITUDES,
                bytemuck::cast_slice(&amps),
            );

            if has_thermal {
                // Rewrite pulse_directions[p].w per step with instantaneous
                // absorbed power density. For pulses without `peak_fluence`
                // we still overwrite with 0.0 so the M3TM kernel sees no
                // source from that pulse.
                let sigma_t = |fwhm: f64| fwhm / 2.354_820_045;
                let tau = (2.0 * std::f64::consts::PI).sqrt();
                for (i, pulse) in self.config.photonic.pulses.iter().take(n_pulses).enumerate() {
                    let env = pulse.envelope_at(t_mid);
                    let p_w = match pulse.peak_fluence {
                        Some(f) => {
                            let sig = sigma_t(pulse.duration_fwhm).max(1e-18);
                            (1.0 - pulse.reflectivity as f64) * f / (tau * sig * layer0_thickness) * env
                        }
                        None => 0.0,
                    } as f32;
                    // pulse_directions layout: 4 vec4s at offset 480; each vec4 is
                    // (x, y, z, w_power). Overwrite only the w-component each step
                    // to avoid touching the unit direction vector.
                    let offset_w = 480u64 + (i as u64) * 16 + 12;
                    self.queue.write_buffer(&self.params_buf, offset_w, bytemuck::bytes_of(&p_w));
                }
            }

            let mut encoder = self.device.create_command_encoder(&Default::default());

            // M3TM update first (reads pulse_amplitudes and spot centers;
            // writes temp_e, temp_p, m_reduced). In P3a the LLB path is not
            // yet active — m_reduced evolves but LLG torque kernels don't
            // read it, so cell |m| stays normalized (regression preserved).
            if has_thermal {
                let mut p = encoder.begin_compute_pass(&Default::default());
                p.set_pipeline(&self.m3tm_pipeline);
                p.set_bind_group(0, &self.bind_group, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }

            let enable_llb = self.config
                .photonic
                .thermal
                .as_ref()
                .map_or(false, |t| t.enable_llb);
            let pipelines: [&wgpu::ComputePipeline; 4] = if enable_llb {
                [
                    &self.ft_phase0_llb_pipeline,
                    &self.llb_predict_pipeline,
                    &self.ft_phase1_llb_pipeline,
                    &self.llb_correct_pipeline,
                ]
            } else {
                [
                    &self.ft_phase0_pipeline,
                    &self.predict_pipeline,
                    &self.ft_phase1_pipeline,
                    &self.correct_pipeline,
                ]
            };
            for pipeline in pipelines {
                let mut p = encoder.begin_compute_pass(&Default::default());
                p.set_pipeline(pipeline);
                p.set_bind_group(0, &self.bind_group, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));

            self.step += 1;
            self.t_sim += dt;
        }
    }

    /// Readback `temp_e` (K, per cell). Allocates one MAP_READ staging copy.
    pub fn readback_temp_e(&self) -> Vec<f32> {
        self.readback_f32_scalar(&self.temp_e_buf, &self.temp_e_staging)
    }

    /// Readback `m_reduced` (dimensionless, per cell).
    pub fn readback_m_reduced(&self) -> Vec<f32> {
        self.readback_f32_scalar(&self.m_reduced_buf, &self.m_reduced_staging)
    }

    /// Readback `temp_p` (K, per cell). Reuses the temp_e staging buffer
    /// because scalar readbacks are serialised through the GPU queue.
    pub fn readback_temp_p(&self) -> Vec<f32> {
        self.readback_f32_scalar(&self.temp_p_buf, &self.temp_e_staging)
    }

    fn readback_f32_scalar(&self, src: &wgpu::Buffer, staging: &wgpu::Buffer) -> Vec<f32> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src, 0, staging, 0, staging.size());
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }

    pub fn readback_mag(&self) -> Vec<f32> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.mag_buf, 0,
            &self.mag_staging, 0,
            self.mag_staging.size(),
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.mag_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.mag_staging.unmap();
        out
    }

    /// Compute observables over the entire stack (all cells, all layers).
    pub fn observables(&self) -> Observables {
        let mag = self.readback_mag();
        let n = self.cell_count as usize;

        let mut sum_mx = 0.0f64;
        let mut sum_my = 0.0f64;
        let mut sum_mz = 0.0f64;
        let mut min_norm: f32 = f32::MAX;
        let mut max_norm: f32 = 0.0;

        for i in 0..n {
            let mx = mag[i * 4] as f64;
            let my = mag[i * 4 + 1] as f64;
            let mz = mag[i * 4 + 2] as f64;
            sum_mx += mx;
            sum_my += my;
            sum_mz += mz;
            let norm = (mx * mx + my * my + mz * mz).sqrt() as f32;
            min_norm = min_norm.min(norm);
            max_norm = max_norm.max(norm);
        }

        let inv = 1.0 / n as f64;

        // Probe: cell (probe_idx) at layer (probe_layer, default 0)
        let in_plane = self.config.geometry.nx * self.config.geometry.ny;
        let probe_layer = self.config.probe_layer.unwrap_or(0).min(self.config.nz() - 1);
        let probe_in_plane = self.config.probe_idx.unwrap_or(in_plane / 2).min(in_plane - 1);
        let probe_cell = probe_layer * in_plane + probe_in_plane;
        let probe_mz = if (probe_cell as usize) < n { mag[probe_cell as usize * 4 + 2] } else { 0.0 };

        // Thermal observables (only meaningful when thermal is active).
        let (max_t_e, max_t_p, min_m_reduced) = if self.config.photonic.thermal.is_some() {
            let t_e = self.readback_temp_e();
            let t_p = self.readback_f32_scalar(&self.temp_p_buf, &self.temp_e_staging);
            let m_r = self.readback_m_reduced();
            let max_te = t_e.iter().copied().fold(f32::MIN, f32::max);
            let max_tp = t_p.iter().copied().fold(f32::MIN, f32::max);
            let min_mr = m_r.iter().copied().fold(f32::MAX, f32::min);
            (max_te, max_tp, min_mr)
        } else {
            (0.0, 0.0, 0.0)
        };

        Observables {
            step: self.step,
            time_ps: self.t_sim * 1e12,
            avg_mx: (sum_mx * inv) as f32,
            avg_my: (sum_my * inv) as f32,
            avg_mz: (sum_mz * inv) as f32,
            min_norm,
            max_norm,
            probe_mz,
            max_t_e,
            max_t_p,
            min_m_reduced,
        }
    }

    /// Fully re-upload GpuParams (useful after runtime changes to stack/substrate).
    pub fn upload_params(&self) {
        let params = GpuParams::from_config(&self.config);
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }

    /// Current simulation time [s]. Matches `observables().time_ps / 1e12`.
    pub fn t_sim(&self) -> f64 {
        self.t_sim
    }

    /// Append a pulse to the photonic config. If already at MAX_PULSES (= 4)
    /// the oldest pulse is evicted. Re-uploads params so the GPU sees the
    /// new pulse on the next `step_n`. Used by the dashboard for runtime
    /// pulse firing.
    pub fn add_pulse(&mut self, pulse: crate::photonic::LaserPulse) {
        if self.config.photonic.pulses.len() >= 4 {
            self.config.photonic.pulses.remove(0);
        }
        self.config.photonic.pulses.push(pulse);
        self.upload_params();
    }

    /// Clear all queued pulses (including the current one, even mid-envelope).
    pub fn clear_pulses(&mut self) {
        self.config.photonic.pulses.clear();
        self.upload_params();
    }

    /// Toggle thermal dynamics on/off at runtime. When enabling, the per-
    /// cell temperature and m_reduced buffers are reset to the config's
    /// `t_ambient` / m_e(t_ambient) values; pre-existing buffer contents
    /// are discarded. When disabling, those buffers are left alone (the
    /// LLG path ignores them).
    pub fn set_thermal(&mut self, thermal: Option<crate::photonic::ThermalConfig>) {
        let engaging = self.config.photonic.thermal.is_none() && thermal.is_some();
        self.config.photonic.thermal = thermal;
        self.upload_params();
        if engaging {
            // Reset thermal buffers so the newly-engaged M3TM doesn't see
            // stale data from a prior thermal run.
            self.reset_thermal_state();
            // Also re-upload the LLB tables (they might have changed with a
            // new preset).
            let (m_e_data, chi_par_data) = pack_llb_tables(&self.config);
            self.queue.write_buffer(&self._m_e_table_buf, 0, bytemuck::cast_slice(&m_e_data));
            self.queue.write_buffer(&self._chi_par_table_buf, 0, bytemuck::cast_slice(&chi_par_data));
        }
    }

    /// Reset temp_e, temp_p, m_reduced to the current thermal config's
    /// ambient / equilibrium values. Called from `set_thermal` and available
    /// externally as well (e.g., for re-initialising after a sequence).
    pub fn reset_thermal_state(&mut self) {
        let t_amb = self
            .config
            .photonic
            .thermal
            .as_ref()
            .map(|t| t.t_ambient)
            .unwrap_or(300.0);
        let init_t = vec![t_amb; self.cell_count as usize];
        self.queue.write_buffer(&self.temp_e_buf, 0, bytemuck::cast_slice(&init_t));
        self.queue.write_buffer(&self.temp_p_buf, 0, bytemuck::cast_slice(&init_t));

        let in_plane = (self.config.geometry.nx * self.config.geometry.ny) as usize;
        let mut m_red = vec![1.0f32; self.cell_count as usize];
        if let Some(tc) = &self.config.photonic.thermal {
            for cell in 0..self.cell_count as usize {
                let iz = cell / in_plane;
                if let Some(p) = tc.per_layer.get(iz) {
                    m_red[cell] = p.sample_m_e(t_amb as f64) as f32;
                }
            }
        }
        self.queue.write_buffer(&self.m_reduced_buf, 0, bytemuck::cast_slice(&m_red));
    }

    /// Adjust the shared thermal `t_ambient` without changing presets.
    /// Cheap — writes only the `thermal_globals` vec4 (offset 672).
    pub fn set_thermal_ambient(&mut self, t_ambient: f32) {
        if let Some(tc) = self.config.photonic.thermal.as_mut() {
            tc.t_ambient = t_ambient;
        }
        let data = [t_ambient, 0.0f32, 0.0, 0.0];
        self.queue.write_buffer(&self.params_buf, 672, bytemuck::cast_slice(&data));
    }

    /// Update external field at runtime. Offset 32 in the new layout.
    pub fn set_b_ext(&mut self, bx: f32, by: f32, bz: f32) {
        self.config.b_ext = [bx as f64, by as f64, bz as f64];
        let data = [bx, by, bz, 0.0f32];
        self.queue.write_buffer(&self.params_buf, 32, bytemuck::cast_slice(&data));
    }

    /// Override α of all layers simultaneously at runtime.
    pub fn set_alpha(&mut self, alpha: f32) {
        let data = [alpha; 4];
        self.queue.write_buffer(&self.params_buf, 48, bytemuck::cast_slice(&data));
    }

    /// Override exchange prefactor of all layers simultaneously at runtime.
    pub fn set_exchange_pf(&self, pf: f32) {
        let data = [pf; 4];
        self.queue.write_buffer(&self.params_buf, 64, bytemuck::cast_slice(&data));
    }

    /// Override per-layer DMI prefactor for all layers at runtime.
    pub fn set_dmi_pf(&self, pf: f32) {
        let data = [pf; 4];
        self.queue.write_buffer(&self.params_buf, 96, bytemuck::cast_slice(&data));
    }

    /// Update interlayer-exchange prefactors at runtime (Phase M2).
    pub fn set_interlayer_pfs(&self, below: [f32; 4], above: [f32; 4]) {
        self.queue.write_buffer(&self.params_buf, 160, bytemuck::cast_slice(&below));
        self.queue.write_buffer(&self.params_buf, 176, bytemuck::cast_slice(&above));
    }

    /// Override a single pulse's amplitude at runtime (Phase P2).
    /// Useful for programmatic pulse control or debugging.
    /// Note: `step_n` overwrites this each step when pulses are active.
    pub fn set_pulse_amplitude(&self, pulse_idx: usize, amplitude_t: f32) {
        if pulse_idx >= 4 { return; }
        let offset = OFFSET_PULSE_AMPLITUDES + (pulse_idx * 4) as u64;
        self.queue.write_buffer(&self.params_buf, offset, bytemuck::bytes_of(&amplitude_t));
    }

    /// Update exchange bias for layer 0 at runtime.
    pub fn set_b_bias(&self, bx: f32, by: f32, bz: f32) {
        // layer_b_biases starts at offset 256, layer 0 occupies first 16 bytes
        let data = [bx, by, bz, 0.0f32];
        self.queue.write_buffer(&self.params_buf, 256, bytemuck::cast_slice(&data));
    }

    /// Update SOT torques for layer 0 at runtime.
    pub fn set_sot(&self, tau_dl: f32, tau_fl: f32) {
        // layer_sot_tau_dls at 112, layer_sot_tau_fls at 128
        let tdl = [tau_dl; 4];
        let tfl = [tau_fl; 4];
        self.queue.write_buffer(&self.params_buf, 112, bytemuck::cast_slice(&tdl));
        self.queue.write_buffer(&self.params_buf, 128, bytemuck::cast_slice(&tfl));
    }

    /// Update σ direction for layer 0 at runtime.
    pub fn set_sigma(&self, sx: f32, sy: f32, sz: f32) {
        // layer_sigmas starts at offset 320, layer 0 first 16 bytes
        let data = [sx, sy, sz, 0.0f32];
        self.queue.write_buffer(&self.params_buf, 320, bytemuck::cast_slice(&data));
    }

    pub fn step_count(&self) -> usize { self.step }
    pub fn config(&self) -> &SimConfig { &self.config }

    /// Re-upload magnetization: uniform +z with small perturbation (all layers).
    pub fn reset_uniform_z(&mut self) {
        let data = Self::init_magnetization(&self.config);
        self.queue.write_buffer(&self.mag_buf, 0, bytemuck::cast_slice(&data));
        self.step = 0;
        self.t_sim = 0.0;
    }

    /// Re-upload magnetization: alternating +z / -z through the stack layers.
    /// Layer 0 → +z, Layer 1 → -z, Layer 2 → +z, etc. (with 5° random cone).
    /// Natural initial condition for synthetic-AFM heterostructures.
    pub fn reset_uniform_z_alternating(&mut self) {
        use rand::Rng;
        let n = self.cell_count as usize;
        let in_plane = (self.config.geometry.nx * self.config.geometry.ny) as usize;
        let mut data = vec![0.0f32; n * 4];
        let mut rng = rand::thread_rng();
        for cell in 0..n {
            let iz = cell / in_plane;
            let sign: f32 = if iz % 2 == 0 { 1.0 } else { -1.0 };
            let theta: f32 = rng.gen_range(0.0..0.087);
            let phi: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let st = theta.sin();
            data[cell * 4] = st * phi.cos();
            data[cell * 4 + 1] = st * phi.sin();
            data[cell * 4 + 2] = sign * theta.cos();
        }
        self.queue.write_buffer(&self.mag_buf, 0, bytemuck::cast_slice(&data));
        self.step = 0;
        self.t_sim = 0.0;
    }

    /// Re-upload magnetization: fully random unit vectors (all layers).
    pub fn reset_random(&mut self) {
        use rand::Rng;
        let n = self.cell_count as usize;
        let mut data = vec![0.0f32; n * 4];
        let mut rng = rand::thread_rng();
        for i in 0..n {
            loop {
                let x: f32 = rng.gen_range(-1.0..1.0);
                let y: f32 = rng.gen_range(-1.0..1.0);
                let s = x * x + y * y;
                if s < 1.0 {
                    let f = (1.0 - s).sqrt();
                    data[i * 4] = 2.0 * x * f;
                    data[i * 4 + 1] = 2.0 * y * f;
                    data[i * 4 + 2] = 1.0 - 2.0 * s;
                    break;
                }
            }
        }
        self.queue.write_buffer(&self.mag_buf, 0, bytemuck::cast_slice(&data));
        self.step = 0;
        self.t_sim = 0.0;
    }

    /// Re-upload magnetization: stripe domains (alternating +z / -z) on each layer.
    pub fn reset_stripe_domains(&mut self, stripe_width: u32) {
        use rand::Rng;
        let n = self.cell_count as usize;
        let nx = self.config.geometry.nx;
        let ny = self.config.geometry.ny;
        let in_plane = nx * ny;
        let mut data = vec![0.0f32; n * 4];
        let mut rng = rand::thread_rng();
        for cell in 0..n {
            let cell_u = cell as u32;
            let in_plane_idx = cell_u % in_plane;
            let ix = in_plane_idx / ny;
            let mz_sign: f32 = if (ix / stripe_width) % 2 == 0 { 1.0 } else { -1.0 };
            let theta: f32 = rng.gen_range(0.0..0.087);
            let phi: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let st = theta.sin();
            data[cell * 4] = st * phi.cos();
            data[cell * 4 + 1] = st * phi.sin();
            data[cell * 4 + 2] = mz_sign * theta.cos();
        }
        self.queue.write_buffer(&self.mag_buf, 0, bytemuck::cast_slice(&data));
        self.step = 0;
        self.t_sim = 0.0;
    }

    /// Re-upload magnetization: Néel skyrmion seed on every layer.
    pub fn reset_skyrmion_seed(&mut self, radius_nm: f64) {
        let n = self.cell_count as usize;
        let nx = self.config.geometry.nx;
        let ny = self.config.geometry.ny;
        let in_plane = nx * ny;
        let dx = self.config.geometry.cell_size;
        let r0 = radius_nm * 1e-9;
        let cx = nx as f64 * dx * 0.5;
        let cy = ny as f64 * dx * 0.5;
        let mut data = vec![0.0f32; n * 4];
        for cell in 0..n {
            let cell_u = cell as u32;
            let in_plane_idx = cell_u % in_plane;
            let ix = in_plane_idx / ny;
            let iy = in_plane_idx % ny;
            let x = (ix as f64 + 0.5) * dx - cx;
            let y = (iy as f64 + 0.5) * dx - cy;
            let r = (x * x + y * y).sqrt();
            let theta = if r < 1e-20 {
                std::f64::consts::PI
            } else {
                2.0 * (r0 / r).atan()
            };
            let phi = y.atan2(x);
            let mz = theta.cos();
            let ms = theta.sin();
            let mx = ms * phi.cos();
            let my = ms * phi.sin();
            data[cell * 4] = mx as f32;
            data[cell * 4 + 1] = my as f32;
            data[cell * 4 + 2] = mz as f32;
        }
        self.queue.write_buffer(&self.mag_buf, 0, bytemuck::cast_slice(&data));
        self.step = 0;
        self.t_sim = 0.0;
    }
}
