use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::config::SimConfig;

// ─── GPU-side parameter struct (must match WGSL Params exactly) ────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    nx: u32,
    ny: u32,
    cell_count: u32,
    _pad_grid: u32,

    dt: f32,
    gamma: f32,
    alpha: f32,
    stab_coeff: f32,

    exchange_pf: f32,
    anisotropy_pf: f32,
    _pad_mat0: f32,
    _pad_mat1: f32,

    u_axis_x: f32,
    u_axis_y: f32,
    u_axis_z: f32,
    _pad_axis: f32,

    b_ext_x: f32,
    b_ext_y: f32,
    b_ext_z: f32,
    _pad_ext: f32,
}

impl GpuParams {
    fn from_config(cfg: &SimConfig) -> Self {
        Self {
            nx: cfg.nx,
            ny: cfg.ny,
            cell_count: cfg.nx * cfg.ny,
            _pad_grid: 0,
            dt: cfg.dt as f32,
            gamma: cfg.material.gamma as f32,
            alpha: cfg.material.alpha as f32,
            stab_coeff: cfg.stab_coeff as f32,
            exchange_pf: cfg.exchange_prefactor() as f32,
            anisotropy_pf: cfg.anisotropy_prefactor() as f32,
            _pad_mat0: 0.0,
            _pad_mat1: 0.0,
            u_axis_x: cfg.u_axis[0] as f32,
            u_axis_y: cfg.u_axis[1] as f32,
            u_axis_z: cfg.u_axis[2] as f32,
            _pad_axis: 0.0,
            b_ext_x: cfg.b_ext[0] as f32,
            b_ext_y: cfg.b_ext[1] as f32,
            b_ext_z: cfg.b_ext[2] as f32,
            _pad_ext: 0.0,
        }
    }
}

// ─── Observables ───────────────────────────────────────────────

pub struct Observables {
    pub step: usize,
    pub time_ps: f64,
    pub avg_mx: f32,
    pub avg_my: f32,
    pub avg_mz: f32,
    pub min_norm: f32,
    pub max_norm: f32,
    /// Mz at virtual probe cell (MTJ readout)
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

    /// Create solver using an existing GPU device (GUI / shared mode).
    pub fn from_device_queue(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: &SimConfig,
    ) -> Result<Self, String> {
        let cell_count = config.nx * config.ny;
        let buf_bytes = (cell_count as u64) * 4 * 4; // 4 f32 per cell

        // ── Initial magnetization ──────────────────────────────
        let mag_data = Self::init_magnetization(config);

        // ── Buffers ────────────────────────────────────────────
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

        // ── Shader + pipelines ─────────────────────────────────
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

    /// Initialize magnetization: uniform +z with small random perturbation.
    fn init_magnetization(config: &SimConfig) -> Vec<f32> {
        use rand::Rng;
        let n = (config.nx * config.ny) as usize;
        let mut data = vec![0.0f32; n * 4];
        let mut rng = rand::thread_rng();

        for i in 0..n {
            // Small tilt from +z (5° cone)
            let theta: f32 = rng.gen_range(0.0..0.087); // ~5°
            let phi: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let st = theta.sin();
            data[i * 4] = st * phi.cos();
            data[i * 4 + 1] = st * phi.sin();
            data[i * 4 + 2] = theta.cos();
            // data[i*4 + 3] = 0.0 (padding, already zero)
        }
        data
    }

    /// Run N Heun steps in a single GPU submission.
    pub fn step_n(&mut self, n: usize) {
        let wg = self.workgroups;
        let mut encoder = self.device.create_command_encoder(&Default::default());

        for _ in 0..n {
            // Phase 0: field + torque from mag → torque_0
            {
                let mut p = encoder.begin_compute_pass(&Default::default());
                p.set_pipeline(&self.ft_phase0_pipeline);
                p.set_bind_group(0, &self.bind_group, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }
            // Predict: m* = normalize(m + dt·T₀)
            {
                let mut p = encoder.begin_compute_pass(&Default::default());
                p.set_pipeline(&self.predict_pipeline);
                p.set_bind_group(0, &self.bind_group, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }
            // Phase 1: field + torque from mag_pred → torque_1
            {
                let mut p = encoder.begin_compute_pass(&Default::default());
                p.set_pipeline(&self.ft_phase1_pipeline);
                p.set_bind_group(0, &self.bind_group, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }
            // Correct: m = normalize(m + dt/2·(T₀+T₁))
            {
                let mut p = encoder.begin_compute_pass(&Default::default());
                p.set_pipeline(&self.correct_pipeline);
                p.set_bind_group(0, &self.bind_group, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        self.step += n;
    }

    /// Read full magnetization back to CPU.
    pub fn readback_mag(&self) -> Vec<f32> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.mag_buf, 0, &self.mag_staging, 0, self.mag_staging.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = self.mag_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.mag_staging.unmap();
        out
    }

    /// Compute observables from readback data.
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

        // Virtual probe: center cell or configured index
        let probe_idx = self.config.probe_idx.unwrap_or(self.cell_count / 2) as usize;
        let probe_mz = if probe_idx < n { mag[probe_idx * 4 + 2] } else { 0.0 };

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

    /// Update external field at runtime.
    pub fn set_b_ext(&self, bx: f32, by: f32, bz: f32) {
        // b_ext starts at byte offset 64 in GpuParams (4th vec4)
        let data = [bx, by, bz, 0.0f32];
        self.queue.write_buffer(&self.params_buf, 64, bytemuck::cast_slice(&data));
    }

    /// Update damping parameter at runtime.
    pub fn set_alpha(&self, alpha: f32) {
        // alpha is at byte offset 24 (6th f32)
        self.queue.write_buffer(&self.params_buf, 24, bytemuck::bytes_of(&alpha));
    }

    /// Update exchange prefactor at runtime (for parameter sweeps).
    pub fn set_exchange_pf(&self, pf: f32) {
        // exchange_pf is at byte offset 32 (8th f32)
        self.queue.write_buffer(&self.params_buf, 32, bytemuck::bytes_of(&pf));
    }

    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Re-upload magnetization: uniform +z with small perturbation.
    pub fn reset_uniform_z(&mut self) {
        let data = Self::init_magnetization(&self.config);
        self.queue.write_buffer(&self.mag_buf, 0, bytemuck::cast_slice(&data));
        self.step = 0;
    }

    /// Re-upload magnetization: fully random unit vectors.
    pub fn reset_random(&mut self) {
        use rand::Rng;
        let n = self.cell_count as usize;
        let mut data = vec![0.0f32; n * 4];
        let mut rng = rand::thread_rng();
        for i in 0..n {
            // Uniform random point on sphere (Marsaglia method)
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

    /// Re-upload magnetization: stripe domains (alternating +z / -z).
    pub fn reset_stripe_domains(&mut self, stripe_width: u32) {
        let n = self.cell_count as usize;
        let ny = self.config.ny;
        let mut data = vec![0.0f32; n * 4];
        for i in 0..n {
            let ix = (i as u32) / ny;
            let mz: f32 = if (ix / stripe_width) % 2 == 0 { 1.0 } else { -1.0 };
            data[i * 4 + 2] = mz;
        }
        self.queue.write_buffer(&self.mag_buf, 0, bytemuck::cast_slice(&data));
        self.step = 0;
    }

    pub fn config(&self) -> &SimConfig {
        &self.config
    }
}
