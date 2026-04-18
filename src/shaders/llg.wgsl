// Magnonic Clock Simulator — LLG Compute Kernels
//
// Integrator: PROJECTED HEUN (explicit trapezoidal + post-stage renormalize).
//   Not plain textbook Heun — we renormalize |m|=1 after both the predictor
//   and corrector. This is standard practice in micromagnetics (MuMax3,
//   OOMMF) because explicit discretizations of LLG do not preserve |m|=1
//   in discrete time, and projection is a simple, stable remedy.
//
//   An alternative (the 2025 Peiris stabilization term) is retained as the
//   dormant `stab_coeff` uniform for future experimentation with a
//   smooth-correction approach. In the current code path it has no effect.
//
// Field convention: B_eff in Tesla (not H in A/m). See config.rs for
// the unit convention block and conversion formulas.
//
// Assumed geometry: square 2D cells with isotropic in-plane spacing dx = dy.
// Exchange stencil uses one prefactor for both directions; rectangular
// cells would require direction-dependent prefactors.
//
// Buffer layout: 4 × f32 per cell (mx, my, mz, pad)
// Grid: row-major 2D, idx = ix * ny + iy
//
// h_eff buffer semantics: STAGE SCRATCH, NOT OBSERVABLE STATE.
//   Phase 0 writes B_eff(mag); phase 1 overwrites with B_eff(mag_pred).
//   After heun_correct, h_eff reflects the predicted (mid-step) field, NOT
//   the field of the final corrected magnetization. Do not read h_eff as
//   the instantaneous field of the current state; add a final field
//   recompute pass if you need that diagnostic.

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

    // Exchange bias from AFM substrate (Phase 2)
    b_bias_x: f32,
    b_bias_y: f32,
    b_bias_z: f32,
    _pad_bias: f32,

    // Interfacial (Rashba) DMI prefactor (Phase 3): D / (Ms·dx) [T per unit neighbor difference]
    dmi_pf: f32,
    _pad_dmi0: f32,
    _pad_dmi1: f32,
    _pad_dmi2: f32,

    // Slonczewski SOT (Phase 4): current-driven torque from heavy-metal substrate
    sot_tau_dl: f32,      // damping-like torque effective field [T]
    sot_tau_fl: f32,      // field-like torque effective field [T]
    _pad_sot0: f32,
    _pad_sot1: f32,
    sigma_x: f32,         // spin polarization direction (unit vector)
    sigma_y: f32,
    sigma_z: f32,
    _pad_sigma: f32,
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

// Substrate-pinned exchange bias field (Phase 2).
// Contributed by antiferromagnetic substrates (IrMn, CoO) that pin the
// ferromagnet via interfacial exchange coupling. Acts as an additional
// uniform Zeeman-like term along a fixed direction set by the substrate's
// field-cooling history.
fn exchange_bias_field() -> vec3<f32> {
    return vec3<f32>(params.b_bias_x, params.b_bias_y, params.b_bias_z);
}

// Interfacial (Rashba) DMI field (Phase 3) — from mag buffer.
// Energy (Thiaville 2012 convention, standard in micromagnetics literature):
//   e_DMI = D · [m_x·∂m_z/∂x + m_y·∂m_z/∂y - m_z·∂m_x/∂x - m_z·∂m_y/∂y]
// Derived effective field (δE/δm → -B/Ms):
//   B_DMI_x = -(2D/Ms)·∂m_z/∂x
//   B_DMI_y = -(2D/Ms)·∂m_z/∂y
//   B_DMI_z = +(2D/Ms)·(∂m_x/∂x + ∂m_y/∂y)
// With this convention, positive D stabilizes OUTWARD-radial Néel skyrmions
// (standard Pt/Co heavy-metal heterostructure chirality).
// Central differences: ∂m/∂x ≈ (m[i+1] - m[i-1]) / (2·dx).
// dmi_pf = D / (Ms·dx) absorbs the /(2·dx) from the stencil and the 2 from
// the variational derivative — stencil returns raw (m[i+1] - m[i-1]).
// Boundaries use the same clamping-Neumann treatment as exchange.
fn dmi_from_mag(idx: u32) -> vec3<f32> {
    let ix = i32(idx / params.ny);
    let iy = i32(idx % params.ny);
    let m_px = read_mag_at(ix + 1i, iy);
    let m_mx = read_mag_at(ix - 1i, iy);
    let m_py = read_mag_at(ix, iy + 1i);
    let m_my = read_mag_at(ix, iy - 1i);
    let dmz_dx = m_px.z - m_mx.z;
    let dmz_dy = m_py.z - m_my.z;
    let dmx_dx = m_px.x - m_mx.x;
    let dmy_dy = m_py.y - m_my.y;
    return params.dmi_pf * vec3<f32>(-dmz_dx, -dmz_dy, (dmx_dx + dmy_dy));
}

fn dmi_from_pred(idx: u32) -> vec3<f32> {
    let ix = i32(idx / params.ny);
    let iy = i32(idx % params.ny);
    let m_px = read_pred_at(ix + 1i, iy);
    let m_mx = read_pred_at(ix - 1i, iy);
    let m_py = read_pred_at(ix, iy + 1i);
    let m_my = read_pred_at(ix, iy - 1i);
    let dmz_dx = m_px.z - m_mx.z;
    let dmz_dy = m_py.z - m_my.z;
    let dmx_dx = m_px.x - m_mx.x;
    let dmy_dy = m_py.y - m_my.y;
    return params.dmi_pf * vec3<f32>(-dmz_dx, -dmz_dy, (dmx_dx + dmy_dy));
}

// LLG torque in Landau-Lifshitz form with Slonczewski SOT (Phase 4):
//   dm/dt = -γ'[(m × B) + α(m × (m × B))]
//         - γ [τ_DL·m × (m × σ) + τ_FL·(m × σ)]
// where γ' = γ/(1+α²). First line is the standard LLG precession + Gilbert
// damping. Second line is spin-orbit torque: σ is the spin polarization
// direction set by the charge current in the substrate (σ = ẑ × ĵ_c);
// τ_DL is the damping-like effective field, τ_FL the field-like.
//
// SOT vanishes when τ_DL and τ_FL are zero (no current), reducing to pure LLG.
// |m| is preserved by post-step normalize in heun_predict/heun_correct.
fn llg_torque(m: vec3<f32>, b_eff: vec3<f32>) -> vec3<f32> {
    let a2 = params.alpha * params.alpha;
    let gp = params.gamma / (1.0 + a2);

    let mxb = cross(m, b_eff);
    let mxmxb = cross(m, mxb);

    var dmdt = -gp * (mxb + params.alpha * mxmxb);

    // SOT contribution (Slonczewski form)
    let sigma = vec3<f32>(params.sigma_x, params.sigma_y, params.sigma_z);
    let mxs = cross(m, sigma);
    let mxmxs = cross(m, mxs);
    dmdt += -params.gamma * (params.sot_tau_dl * mxmxs + params.sot_tau_fl * mxs);

    return dmdt;
}

// ─── Entry points ──────────────────────────────────────────────

// Phase 0: compute effective field from mag, LLG torque → torque_0
@compute @workgroup_size(64)
fn field_torque_phase0(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }

    let m = read_mag(idx);
    let b_eff = exchange_from_mag(idx) + anisotropy_field(m)
              + zeeman_field() + exchange_bias_field() + dmi_from_mag(idx);
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
    let b_eff = exchange_from_pred(idx) + anisotropy_field(m)
              + zeeman_field() + exchange_bias_field() + dmi_from_pred(idx);
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
