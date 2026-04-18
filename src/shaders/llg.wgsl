// Magnonic Clock Simulator — LLG Compute Kernels (Multilayer, Phase M1)
//
// Integrator: PROJECTED HEUN (explicit trapezoidal + post-stage renormalize).
// See notes in earlier 2D version; the scheme is unchanged.
//
// Geometry:
//   3D grid Nx × Ny × Nz, flat index:
//     idx = iz * (Nx * Ny) + ix * Ny + iy
//   Layers are ordered bottom → top; layer 0 carries substrate contributions.
//
// Phase M2 status: interlayer exchange is ACTIVE. Each cell at (ix, iy, iz)
// couples to the same (ix, iy) cell in the layers immediately above and
// below, weighted by per-layer prefactors `layer_ilx_above[iz]` and
// `layer_ilx_below[iz]`. At the top/bottom of the stack the missing
// neighbor is absent (clamping Neumann BC — contribution is zero).
//
// Per-layer params are indexed by iz. Max 4 layers (MAX_LAYERS = 4).
//
// h_eff buffer semantics: STAGE SCRATCH (unchanged).

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    cell_count: u32,

    dt: f32,
    gamma: f32,
    stab_coeff: f32,
    _pad_dyn: f32,

    b_ext: vec4<f32>,

    // Packed per-layer scalars (vec4 = 4 layers)
    layer_alphas: vec4<f32>,
    layer_exchange_pfs: vec4<f32>,
    layer_anisotropy_pfs: vec4<f32>,
    layer_dmi_pfs: vec4<f32>,
    layer_sot_tau_dls: vec4<f32>,
    layer_sot_tau_fls: vec4<f32>,
    layer_thicknesses: vec4<f32>,

    // Per-layer interlayer-exchange prefactors (Phase M2 — active)
    //   layer_ilx_below[k]: prefactor for layer k reading interface BELOW it
    //   layer_ilx_above[k]: prefactor for layer k reading interface ABOVE it
    // Asymmetric across heterostructure interfaces (Ms_k ≠ Ms_{k-1}).
    layer_ilx_below: vec4<f32>,
    layer_ilx_above: vec4<f32>,

    // Per-layer vec3-in-vec4 fields (array stride 16 bytes)
    layer_u_axes: array<vec4<f32>, 4>,
    layer_b_biases: array<vec4<f32>, 4>,
    layer_sigmas: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> mag: array<f32>;
@group(0) @binding(2) var<storage, read_write> mag_pred: array<f32>;
@group(0) @binding(3) var<storage, read_write> h_eff: array<f32>;
@group(0) @binding(4) var<storage, read_write> torque_0: array<f32>;
@group(0) @binding(5) var<storage, read_write> torque_1: array<f32>;

// ─── Index helpers ─────────────────────────────────────────────

fn in_plane_count() -> u32 {
    return params.nx * params.ny;
}

fn layer_of(idx: u32) -> u32 {
    return idx / in_plane_count();
}

/// Flat 3D index for an in-plane neighbor on the same layer, with Neumann BC.
fn neighbor_idx(iz: u32, ix: i32, iy: i32) -> u32 {
    let cx = clamp(ix, 0i, i32(params.nx) - 1i);
    let cy = clamp(iy, 0i, i32(params.ny) - 1i);
    return iz * in_plane_count() + u32(cx) * params.ny + u32(cy);
}

fn read_mag(idx: u32) -> vec3<f32> {
    let b = idx * 4u;
    return vec3<f32>(mag[b], mag[b + 1u], mag[b + 2u]);
}

fn read_mag_at(iz: u32, ix: i32, iy: i32) -> vec3<f32> {
    return read_mag(neighbor_idx(iz, ix, iy));
}

fn read_pred(idx: u32) -> vec3<f32> {
    let b = idx * 4u;
    return vec3<f32>(mag_pred[b], mag_pred[b + 1u], mag_pred[b + 2u]);
}

fn read_pred_at(iz: u32, ix: i32, iy: i32) -> vec3<f32> {
    return read_pred(neighbor_idx(iz, ix, iy));
}

// ─── Per-layer field helpers ───────────────────────────────────

fn exchange_from_mag(idx: u32) -> vec3<f32> {
    let iz = layer_of(idx);
    let in_plane_idx = idx % in_plane_count();
    let ix = i32(in_plane_idx / params.ny);
    let iy = i32(in_plane_idx % params.ny);
    let m_c = read_mag(idx);
    let lap = read_mag_at(iz, ix + 1i, iy) + read_mag_at(iz, ix - 1i, iy)
            + read_mag_at(iz, ix, iy + 1i) + read_mag_at(iz, ix, iy - 1i)
            - 4.0 * m_c;
    return params.layer_exchange_pfs[iz] * lap;
}

fn exchange_from_pred(idx: u32) -> vec3<f32> {
    let iz = layer_of(idx);
    let in_plane_idx = idx % in_plane_count();
    let ix = i32(in_plane_idx / params.ny);
    let iy = i32(in_plane_idx % params.ny);
    let m_c = read_pred(idx);
    let lap = read_pred_at(iz, ix + 1i, iy) + read_pred_at(iz, ix - 1i, iy)
            + read_pred_at(iz, ix, iy + 1i) + read_pred_at(iz, ix, iy - 1i)
            - 4.0 * m_c;
    return params.layer_exchange_pfs[iz] * lap;
}

/// Interlayer (z-direction) exchange field — M2 active.
///
/// For layer k at in-plane (ix, iy), the two adjacent-layer neighbors at the
/// SAME (ix, iy) on layers k±1 contribute:
///   B = ilx_below[k]·(m_{k-1} - m_k) + ilx_above[k]·(m_{k+1} - m_k)
///
/// At the stack boundaries (k=0 or k=Nz-1) the missing neighbor is absent;
/// the corresponding prefactor is zero by construction (from Rust side).
fn exchange_interlayer_from_mag(idx: u32) -> vec3<f32> {
    let iz = layer_of(idx);
    let in_plane_idx = idx % in_plane_count();
    let m_c = read_mag(idx);
    var result = vec3<f32>(0.0, 0.0, 0.0);
    if iz > 0u {
        let below_idx = (iz - 1u) * in_plane_count() + in_plane_idx;
        let m_below = read_mag(below_idx);
        result += params.layer_ilx_below[iz] * (m_below - m_c);
    }
    if iz + 1u < params.nz {
        let above_idx = (iz + 1u) * in_plane_count() + in_plane_idx;
        let m_above = read_mag(above_idx);
        result += params.layer_ilx_above[iz] * (m_above - m_c);
    }
    return result;
}

fn exchange_interlayer_from_pred(idx: u32) -> vec3<f32> {
    let iz = layer_of(idx);
    let in_plane_idx = idx % in_plane_count();
    let m_c = read_pred(idx);
    var result = vec3<f32>(0.0, 0.0, 0.0);
    if iz > 0u {
        let below_idx = (iz - 1u) * in_plane_count() + in_plane_idx;
        let m_below = read_pred(below_idx);
        result += params.layer_ilx_below[iz] * (m_below - m_c);
    }
    if iz + 1u < params.nz {
        let above_idx = (iz + 1u) * in_plane_count() + in_plane_idx;
        let m_above = read_pred(above_idx);
        result += params.layer_ilx_above[iz] * (m_above - m_c);
    }
    return result;
}

fn anisotropy_field(m: vec3<f32>, iz: u32) -> vec3<f32> {
    let u = params.layer_u_axes[iz].xyz;
    return params.layer_anisotropy_pfs[iz] * dot(m, u) * u;
}

fn zeeman_field() -> vec3<f32> {
    return params.b_ext.xyz;
}

fn exchange_bias_field(iz: u32) -> vec3<f32> {
    return params.layer_b_biases[iz].xyz;
}

fn dmi_from_mag(idx: u32) -> vec3<f32> {
    let iz = layer_of(idx);
    let in_plane_idx = idx % in_plane_count();
    let ix = i32(in_plane_idx / params.ny);
    let iy = i32(in_plane_idx % params.ny);
    let m_px = read_mag_at(iz, ix + 1i, iy);
    let m_mx = read_mag_at(iz, ix - 1i, iy);
    let m_py = read_mag_at(iz, ix, iy + 1i);
    let m_my = read_mag_at(iz, ix, iy - 1i);
    let dmz_dx = m_px.z - m_mx.z;
    let dmz_dy = m_py.z - m_my.z;
    let dmx_dx = m_px.x - m_mx.x;
    let dmy_dy = m_py.y - m_my.y;
    return params.layer_dmi_pfs[iz] * vec3<f32>(-dmz_dx, -dmz_dy, (dmx_dx + dmy_dy));
}

fn dmi_from_pred(idx: u32) -> vec3<f32> {
    let iz = layer_of(idx);
    let in_plane_idx = idx % in_plane_count();
    let ix = i32(in_plane_idx / params.ny);
    let iy = i32(in_plane_idx % params.ny);
    let m_px = read_pred_at(iz, ix + 1i, iy);
    let m_mx = read_pred_at(iz, ix - 1i, iy);
    let m_py = read_pred_at(iz, ix, iy + 1i);
    let m_my = read_pred_at(iz, ix, iy - 1i);
    let dmz_dx = m_px.z - m_mx.z;
    let dmz_dy = m_py.z - m_my.z;
    let dmx_dx = m_px.x - m_mx.x;
    let dmy_dy = m_py.y - m_my.y;
    return params.layer_dmi_pfs[iz] * vec3<f32>(-dmz_dx, -dmz_dy, (dmx_dx + dmy_dy));
}

/// LLG torque + Slonczewski SOT (both per-layer).
fn llg_torque(m: vec3<f32>, b_eff: vec3<f32>, iz: u32) -> vec3<f32> {
    let alpha = params.layer_alphas[iz];
    let a2 = alpha * alpha;
    let gp = params.gamma / (1.0 + a2);

    let mxb = cross(m, b_eff);
    let mxmxb = cross(m, mxb);

    var dmdt = -gp * (mxb + alpha * mxmxb);

    // SOT (only nonzero for layer 0 by current convention)
    let sigma = params.layer_sigmas[iz].xyz;
    let mxs = cross(m, sigma);
    let mxmxs = cross(m, mxs);
    dmdt += -params.gamma * (params.layer_sot_tau_dls[iz] * mxmxs
                            + params.layer_sot_tau_fls[iz] * mxs);

    return dmdt;
}

// ─── Entry points ──────────────────────────────────────────────

@compute @workgroup_size(64)
fn field_torque_phase0(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }
    let iz = layer_of(idx);

    let m = read_mag(idx);
    let b_eff = exchange_from_mag(idx) + exchange_interlayer_from_mag(idx)
              + anisotropy_field(m, iz) + zeeman_field()
              + exchange_bias_field(iz) + dmi_from_mag(idx);
    let t = llg_torque(m, b_eff, iz);

    let base = idx * 4u;
    h_eff[base] = b_eff.x; h_eff[base + 1u] = b_eff.y; h_eff[base + 2u] = b_eff.z;
    torque_0[base] = t.x; torque_0[base + 1u] = t.y; torque_0[base + 2u] = t.z;
}

@compute @workgroup_size(64)
fn field_torque_phase1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }
    let iz = layer_of(idx);

    let m = read_pred(idx);
    let b_eff = exchange_from_pred(idx) + exchange_interlayer_from_pred(idx)
              + anisotropy_field(m, iz) + zeeman_field()
              + exchange_bias_field(iz) + dmi_from_pred(idx);
    let t = llg_torque(m, b_eff, iz);

    let base = idx * 4u;
    h_eff[base] = b_eff.x; h_eff[base + 1u] = b_eff.y; h_eff[base + 2u] = b_eff.z;
    torque_1[base] = t.x; torque_1[base + 1u] = t.y; torque_1[base + 2u] = t.z;
}

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
