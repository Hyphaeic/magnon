// Magnonic Clock Simulator — LLG Compute Kernels
// Heun's method with pitchfork stabilization
//
// Buffer layout: 4 × f32 per cell (mx, my, mz, pad)
// Grid: row-major 2D, idx = ix * ny + iy

struct Params {
    nx: u32,
    ny: u32,
    cell_count: u32,
    _pad_grid: u32,

    dt: f32,             // timestep [s]
    gamma: f32,          // gyromagnetic ratio [rad/(s·T)]
    alpha: f32,          // Gilbert damping
    stab_coeff: f32,     // pitchfork stabilization [1/s]

    exchange_pf: f32,    // 2A/(Ms·dx²) [T]
    anisotropy_pf: f32,  // 2K_u/Ms [T]
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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> mag: array<f32>;
@group(0) @binding(2) var<storage, read_write> mag_pred: array<f32>;
@group(0) @binding(3) var<storage, read_write> h_eff: array<f32>;
@group(0) @binding(4) var<storage, read_write> torque_0: array<f32>;
@group(0) @binding(5) var<storage, read_write> torque_1: array<f32>;

// ─── Helpers ───────────────────────────────────────────────────

fn neighbor_idx(ix: i32, iy: i32) -> u32 {
    let cx = clamp(ix, 0i, i32(params.nx) - 1i);
    let cy = clamp(iy, 0i, i32(params.ny) - 1i);
    return u32(cx) * params.ny + u32(cy);
}

fn read_mag(idx: u32) -> vec3<f32> {
    let b = idx * 4u;
    return vec3<f32>(mag[b], mag[b + 1u], mag[b + 2u]);
}

fn read_mag_at(ix: i32, iy: i32) -> vec3<f32> {
    return read_mag(neighbor_idx(ix, iy));
}

fn read_pred(idx: u32) -> vec3<f32> {
    let b = idx * 4u;
    return vec3<f32>(mag_pred[b], mag_pred[b + 1u], mag_pred[b + 2u]);
}

fn read_pred_at(ix: i32, iy: i32) -> vec3<f32> {
    return read_pred(neighbor_idx(ix, iy));
}

// Exchange field from mag buffer (4-neighbor Laplacian, Neumann BC)
fn exchange_from_mag(idx: u32) -> vec3<f32> {
    let ix = i32(idx / params.ny);
    let iy = i32(idx % params.ny);
    let m_c = read_mag(idx);
    let lap = read_mag_at(ix + 1i, iy) + read_mag_at(ix - 1i, iy)
            + read_mag_at(ix, iy + 1i) + read_mag_at(ix, iy - 1i)
            - 4.0 * m_c;
    return params.exchange_pf * lap;
}

// Exchange field from mag_pred buffer
fn exchange_from_pred(idx: u32) -> vec3<f32> {
    let ix = i32(idx / params.ny);
    let iy = i32(idx % params.ny);
    let m_c = read_pred(idx);
    let lap = read_pred_at(ix + 1i, iy) + read_pred_at(ix - 1i, iy)
            + read_pred_at(ix, iy + 1i) + read_pred_at(ix, iy - 1i)
            - 4.0 * m_c;
    return params.exchange_pf * lap;
}

// Uniaxial anisotropy field
fn anisotropy_field(m: vec3<f32>) -> vec3<f32> {
    let u = vec3<f32>(params.u_axis_x, params.u_axis_y, params.u_axis_z);
    return params.anisotropy_pf * dot(m, u) * u;
}

// Uniform external (Zeeman) field
fn zeeman_field() -> vec3<f32> {
    return vec3<f32>(params.b_ext_x, params.b_ext_y, params.b_ext_z);
}

// LLG torque in Landau-Lifshitz form + pitchfork stabilization:
//   dm/dt = -γ'(m × B) - γ'α(m × (m × B)) + A·m·(1 - |m|²)
fn llg_torque(m: vec3<f32>, b_eff: vec3<f32>) -> vec3<f32> {
    let a2 = params.alpha * params.alpha;
    let gp = params.gamma / (1.0 + a2);

    let mxb = cross(m, b_eff);
    let mxmxb = cross(m, mxb);

    var dmdt = -gp * (mxb + params.alpha * mxmxb);

    // Pitchfork stabilization: asymptotically drives |m| → 1
    let norm_sq = dot(m, m);
    dmdt += params.stab_coeff * m * (1.0 - norm_sq);

    return dmdt;
}

// ─── Entry points ──────────────────────────────────────────────

// Phase 0: compute effective field from mag, LLG torque → torque_0
@compute @workgroup_size(64)
fn field_torque_phase0(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }

    let m = read_mag(idx);
    let b_eff = exchange_from_mag(idx) + anisotropy_field(m) + zeeman_field();
    let t = llg_torque(m, b_eff);

    let base = idx * 4u;
    h_eff[base] = b_eff.x; h_eff[base + 1u] = b_eff.y; h_eff[base + 2u] = b_eff.z;
    torque_0[base] = t.x; torque_0[base + 1u] = t.y; torque_0[base + 2u] = t.z;
}

// Phase 1: compute effective field from mag_pred, LLG torque → torque_1
@compute @workgroup_size(64)
fn field_torque_phase1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }

    let m = read_pred(idx);
    let b_eff = exchange_from_pred(idx) + anisotropy_field(m) + zeeman_field();
    let t = llg_torque(m, b_eff);

    let base = idx * 4u;
    h_eff[base] = b_eff.x; h_eff[base + 1u] = b_eff.y; h_eff[base + 2u] = b_eff.z;
    torque_1[base] = t.x; torque_1[base + 1u] = t.y; torque_1[base + 2u] = t.z;
}

// Heun predictor: m* = normalize(m + dt · T₀)
@compute @workgroup_size(64)
fn heun_predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }

    let b = idx * 4u;
    let m = vec3<f32>(mag[b], mag[b + 1u], mag[b + 2u]);
    let t0 = vec3<f32>(torque_0[b], torque_0[b + 1u], torque_0[b + 2u]);

    let m_star = normalize(m + params.dt * t0);

    mag_pred[b] = m_star.x;
    mag_pred[b + 1u] = m_star.y;
    mag_pred[b + 2u] = m_star.z;
}

// Heun corrector: m_{n+1} = normalize(m + dt/2 · (T₀ + T₁))
@compute @workgroup_size(64)
fn heun_correct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }

    let b = idx * 4u;
    let m = vec3<f32>(mag[b], mag[b + 1u], mag[b + 2u]);
    let t0 = vec3<f32>(torque_0[b], torque_0[b + 1u], torque_0[b + 2u]);
    let t1 = vec3<f32>(torque_1[b], torque_1[b + 1u], torque_1[b + 2u]);

    let m_new = normalize(m + 0.5 * params.dt * (t0 + t1));

    mag[b] = m_new.x;
    mag[b + 1u] = m_new.y;
    mag[b + 2u] = m_new.z;
}
