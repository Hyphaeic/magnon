use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::config::{SimConfig, Stack};

// ─── GPU-side parameter struct (must match WGSL Params exactly) ────
//
// Layout strategy (multilayer, M1):
//   - Globals first (grid, dt, γ, stab_coeff, B_ext)
//   - Per-layer scalar params packed into vec4 (4 layers per vec4 slot)
//   - Per-layer vec3 params stored as arrays of vec4 (16-byte stride)
// MAX_LAYERS = 4 sets the array sizes. Layers beyond the stack's layer
// count are zeroed and ignored by the shader (which iterates iz < nz).

const MAX_LAYERS: usize = Stack::MAX_LAYERS;

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
    _pad_dyn: f32,

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
}

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
            _pad_dyn: 0.0,
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
        }
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
}

impl std::fmt::Display for Observables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},{:.4},{:.6},{:.6},{:.6},{:.8},{:.8},{:.6}",
            self.step, self.time_ps, self.avg_mx, self.avg_my, self.avg_mz,
            self.min_norm, self.max_norm, self.probe_mz,
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
    mag_pred_buf: wgpu::Buffer,
    h_eff_buf: wgpu::Buffer,
    torque_0_buf: wgpu::Buffer,
    torque_1_buf: wgpu::Buffer,

    mag_staging: wgpu::Buffer,

    ft_phase0_pipeline: wgpu::ComputePipeline,
    ft_phase1_pipeline: wgpu::ComputePipeline,
    predict_pipeline: wgpu::ComputePipeline,
    correct_pipeline: wgpu::ComputePipeline,

    bind_group: wgpu::BindGroup,

    cell_count: u32,
    workgroups: u32,
    step: usize,
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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("magnonic-llg"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("llg"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/llg.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("llg_bgl"),
            entries: &(0..6u32)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if i == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
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
            ],
        });

        let workgroups = (cell_count + 63) / 64;

        Ok(Self {
            config: config.clone(),
            device,
            queue,
            params_buf,
            mag_buf,
            mag_pred_buf,
            h_eff_buf,
            torque_0_buf,
            torque_1_buf,
            mag_staging,
            ft_phase0_pipeline,
            ft_phase1_pipeline,
            predict_pipeline,
            correct_pipeline,
            bind_group,
            cell_count,
            workgroups,
            step: 0,
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

    /// Run N Heun steps in a single GPU submission.
    pub fn step_n(&mut self, n: usize) {
        let wg = self.workgroups;
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

        Observables {
            step: self.step,
            time_ps: self.step as f64 * self.config.dt * 1e12,
            avg_mx: (sum_mx * inv) as f32,
            avg_my: (sum_my * inv) as f32,
            avg_mz: (sum_mz * inv) as f32,
            min_norm,
            max_norm,
            probe_mz,
        }
    }

    /// Fully re-upload GpuParams (useful after runtime changes to stack/substrate).
    pub fn upload_params(&self) {
        let params = GpuParams::from_config(&self.config);
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
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
    }
}
