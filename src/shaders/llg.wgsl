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
    cell_size_m: f32,  // in-plane cell size [m]; used in Phase P2 for pulse spatial profile

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

    // Phase P2 — per-pulse photonic uniforms. Supersedes P1's single b_laser.
    // Up to 4 active pulses, each with independent amplitude (time-dependent,
    // host-updated per step), spot center + σ_r (static), and direction.
    pulse_count: u32,
    enable_llb_flag: u32,
    _pad_pc1: u32,
    _pad_pc2: u32,
    pulse_amplitudes: vec4<f32>,                      // [T], packed 4 pulses
    pulse_spot_centers: array<vec4<f32>, 4>,           // (x, y, σ_r, _pad) meters
    pulse_directions: array<vec4<f32>, 4>,             // unit vec3 + pad

    // Phase P3 — per-layer thermal (M3TM + LLB) scalars. Appended at the
    // end so the P1-P2 pulse offsets do not shift. See ADR-003.
    layer_thermal_gamma_e: vec4<f32>,   // J/(m³·K²)
    layer_thermal_c_p: vec4<f32>,       // J/(m³·K)
    layer_thermal_g_ep: vec4<f32>,      // W/(m³·K)
    layer_thermal_a_sf_r: vec4<f32>,    // precomputed R Koopmans prefactor [1/s]
    layer_thermal_t_c: vec4<f32>,       // K
    layer_thermal_alpha_0: vec4<f32>,   // dimensionless
    layer_thermal_tau_long: vec4<f32>,  // seconds (P3b longitudinal relaxation base)
    layer_thermal_g_sub_p: vec4<f32>,   // W/(m³·K) — phonon → substrate sink
    thermal_globals: vec4<f32>,         // (t_ambient, pad, pad, pad)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> mag: array<f32>;
@group(0) @binding(2) var<storage, read_write> mag_pred: array<f32>;
@group(0) @binding(3) var<storage, read_write> h_eff: array<f32>;
@group(0) @binding(4) var<storage, read_write> torque_0: array<f32>;
@group(0) @binding(5) var<storage, read_write> torque_1: array<f32>;

// Phase P3 — thermal storage buffers. Always allocated so the bind-group
// layout is stable across runs; only written by `advance_m3tm`.
@group(0) @binding(6) var<storage, read_write> temp_e: array<f32>;
@group(0) @binding(7) var<storage, read_write> temp_p: array<f32>;
@group(0) @binding(8) var<storage, read_write> m_reduced: array<f32>;
@group(0) @binding(9) var<storage, read> m_e_table: array<f32>;
@group(0) @binding(10) var<storage, read> chi_par_table: array<f32>;

const LLB_TABLE_N: u32 = 256u;

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

// Phase P2 — photonic drive at a specific in-plane cell (ix, iy).
// Sums Gaussian-weighted contributions from all active pulses.
// Each pulse contributes: amp · weight(r) · direction
//   where weight(r) = exp(-r²/(2σ²))  for σ > 0,  or 1.0 for σ = 0 (uniform).
// When params.pulse_count == 0 (no photonic drive) the loop returns vec3(0).
fn laser_field_at(idx: u32) -> vec3<f32> {
    let in_plane_idx = idx % in_plane_count();
    let ix = in_plane_idx / params.ny;
    let iy = in_plane_idx % params.ny;
    let x = (f32(ix) + 0.5) * params.cell_size_m;
    let y = (f32(iy) + 0.5) * params.cell_size_m;

    var total = vec3<f32>(0.0, 0.0, 0.0);
    for (var p: u32 = 0u; p < params.pulse_count; p = p + 1u) {
        let center = params.pulse_spot_centers[p];   // (x, y, σ, _)
        let dir = params.pulse_directions[p].xyz;
        let amp = params.pulse_amplitudes[p];

        var weight: f32 = 1.0;
        let sigma = center.z;
        if sigma > 0.0 {
            let dx = x - center.x;
            let dy = y - center.y;
            let r2 = dx * dx + dy * dy;
            weight = exp(-r2 / (2.0 * sigma * sigma));
        }

        total = total + (amp * weight * dir);
    }
    return total;
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
              + anisotropy_field(m, iz) + zeeman_field() + laser_field_at(idx)
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
              + anisotropy_field(m, iz) + zeeman_field() + laser_field_at(idx)
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

// ─── Phase P3 — M3TM kernel ────────────────────────────────────
//
// Per-cell Heun-step of the Koopmans 2010 three-temperature model:
//     C_e(T_e) dT_e/dt = −g_ep·(T_e − T_p) + P_laser(r, t)
//     C_p       dT_p/dt =  g_ep·(T_e − T_p)
//              dm/dt  = R · m · (T_p/T_c) · (1 − m · coth(m·T_c/T_e))
//
// with C_e = γ_e · T_e. Laser absorbed power density:
//     P_laser(r, t) = (1 − R) · F / (Δt_eff · thickness) · envelope(r, t)
// where envelope is the SAME spatial × amplitude Gaussian used by
// `laser_field_at`. Because the pulse amplitude uniform is re-written per
// host step with the temporal envelope at the step midpoint, this kernel
// receives `pulse_amplitudes[p]` as the instantaneous peak B field — to
// recover intensity we need the companion fluence term.
//
// We pass absorbed-peak power density in via a dedicated channel: the
// fourth component of each `pulse_directions[p]` (previously zero padding).
// Host sets it to `(1 − R)·F / (√(2π)·σ_t · t_layer)` when fluence is
// specified; zero otherwise. See `GpuParams::compute_pulse_power_density`.

fn coth_safe(x: f32) -> f32 {
    let ax = abs(x);
    if ax > 20.0 {
        return sign(x);
    }
    return 1.0 / tanh(x);
}

/// Absorbed volumetric power density at cell idx at the current step midpoint
/// [W/m³]. Returns 0 if no pulse carries fluence (peak_fluence = None on host).
fn laser_power_density_at(idx: u32) -> f32 {
    let in_plane_idx = idx % in_plane_count();
    let ix = in_plane_idx / params.ny;
    let iy = in_plane_idx % params.ny;
    let x = (f32(ix) + 0.5) * params.cell_size_m;
    let y = (f32(iy) + 0.5) * params.cell_size_m;

    // Temporal envelope amplitude (linear in pulse_amplitudes[p] / peak_B).
    // To recover the dimensionless envelope we divide the current amp by
    // the nominal peak_field uniform — but we don't store that separately.
    // Instead, host writes envelope directly into `pulse_directions[p].w`
    // each step: w = (1 − R)·P_peak·envelope_at(t_mid), i.e. instantaneous
    // peak absorbed volumetric power density [W/m³] BEFORE the spatial
    // Gaussian.
    var total: f32 = 0.0;
    for (var p: u32 = 0u; p < params.pulse_count; p = p + 1u) {
        let center = params.pulse_spot_centers[p];
        let sigma = center.z;
        let p_inst = params.pulse_directions[p].w;
        if p_inst == 0.0 { continue; }

        var weight: f32 = 1.0;
        if sigma > 0.0 {
            let dx = x - center.x;
            let dy = y - center.y;
            let r2 = dx * dx + dy * dy;
            weight = exp(-r2 / (2.0 * sigma * sigma));
        }
        total = total + p_inst * weight;
    }
    return total;
}

/// Koopmans dm/dt term. Evaluates the ferromagnet-saturation-aware form:
///   R · m · (T_p / T_c) · (1 − m · coth(m · T_c / T_e))
/// with a small-argument expansion near m·Tc/Te ≈ 0 to avoid the 1/x pole
/// in coth.
fn koopmans_dmdt(m: f32, t_e: f32, t_p: f32, r_pref: f32, t_c: f32) -> f32 {
    if t_c < 1.0 || t_e < 1.0 || r_pref == 0.0 {
        return 0.0;
    }
    let x = m * t_c / t_e;
    var m_coth: f32;
    if abs(x) < 1e-3 {
        // m · coth(m·Tc/Te) ≈ Te/Tc + m²·Tc/(3·Te)
        m_coth = t_e / t_c + m * m * t_c / (3.0 * t_e);
    } else {
        m_coth = m * coth_safe(x);
    }
    return r_pref * m * (t_p / t_c) * (1.0 - m_coth);
}

struct M3tmState { t_e: f32, t_p: f32, m: f32 }

fn m3tm_derivs(s: M3tmState, p_laser: f32, iz: u32) -> vec3<f32> {
    let gamma_e = params.layer_thermal_gamma_e[iz];
    let c_p = params.layer_thermal_c_p[iz];
    let g_ep = params.layer_thermal_g_ep[iz];
    let r_pref = params.layer_thermal_a_sf_r[iz];
    let t_c = params.layer_thermal_t_c[iz];
    let g_sub = params.layer_thermal_g_sub_p[iz];
    let t_ambient = params.thermal_globals.x;

    let c_e = max(gamma_e * s.t_e, 1.0);
    let dt_e = (p_laser - g_ep * (s.t_e - s.t_p)) / c_e;
    // Substrate sink acts on the phonon bath: heat flows to the substrate
    // reservoir proportional to (T_p − T_ambient). Energy released by the
    // lattice is not tracked within the simulation (the substrate is an
    // external heat bath), so the energy-balance gate must now include it.
    let dt_p = (g_ep * (s.t_e - s.t_p) - g_sub * (s.t_p - t_ambient)) / max(c_p, 1.0);
    let dm = koopmans_dmdt(s.m, s.t_e, s.t_p, r_pref, t_c);
    return vec3<f32>(dt_e, dt_p, dm);
}

// ─── Phase P3b — LLB integrator helpers + kernels ───────────────
//
// LLB equation (simplified form used here):
//     dm/dt = −γ/(1 + α_⊥²) · (m × B_eff + α_⊥ · m̂ × (m̂ × B_eff))      (transverse)
//           + rate_∥(T_s) · (m_e(T_s) − |m|) · m̂                         (longitudinal)
//
// with
//     α_⊥(T) = α_0 · (1 − T/(3·T_c))        [low-T limit: α_0]
//     α_∥(T) = α_0 · (2·T/(3·T_c))          [zero at T = 0]
//     rate_∥(T) = α_∥(T) / tau_long_base    [1/s]
//
// The longitudinal part is a phenomenological exponential relaxation to
// m_e(T_s) — stable for any dt, reduces to LLG at T = 0 (α_∥ = 0), and
// captures ultrafast demag at high T. See ADR-003 / plan-photonic.md
// Implementation notes for P3b for the rationale vs. full Atxitia form.

fn sample_m_e(iz: u32, t: f32) -> f32 {
    let t_c = params.layer_thermal_t_c[iz];
    if t_c < 1.0 { return 1.0; }
    let t_max = 1.5 * t_c;
    let n_f = f32(LLB_TABLE_N);
    let u = clamp(t / t_max, 0.0, 0.999999) * (n_f - 1.0);
    let i0 = u32(floor(u));
    let i1 = min(i0 + 1u, LLB_TABLE_N - 1u);
    let frac = u - f32(i0);
    let base = iz * LLB_TABLE_N;
    return m_e_table[base + i0] * (1.0 - frac) + m_e_table[base + i1] * frac;
}

fn alpha_perp(iz: u32, t_s: f32) -> f32 {
    let t_c = params.layer_thermal_t_c[iz];
    if t_c < 1.0 { return params.layer_alphas[iz]; }
    return params.layer_thermal_alpha_0[iz] * (1.0 - t_s / (3.0 * t_c));
}

fn alpha_par(iz: u32, t_s: f32) -> f32 {
    let t_c = params.layer_thermal_t_c[iz];
    if t_c < 1.0 { return 0.0; }
    return params.layer_thermal_alpha_0[iz] * (2.0 * t_s / (3.0 * t_c));
}

/// LLB torque in "Euler form": returns dm/dt [1/s] for a vector m that is
/// NOT constrained to |m|=1. Used by llb_predict / llb_correct.
fn llb_torque(m: vec3<f32>, b_eff: vec3<f32>, iz: u32, t_s: f32) -> vec3<f32> {
    let m_mag = max(length(m), 1e-6);
    let m_hat = m / m_mag;

    // ─ Transverse (LLG) part with temperature-dependent α_⊥ ─
    let a_perp = max(alpha_perp(iz, t_s), 0.0);
    let gp = params.gamma / (1.0 + a_perp * a_perp);
    let mxb = cross(m, b_eff);
    let mxmxb = cross(m_hat, cross(m_hat, b_eff)) * m_mag;  // keep |m| prefactor
    var dmdt = -gp * (mxb + a_perp * mxmxb);

    // ─ Longitudinal relaxation (phenomenological): drives |m| → m_e(T_s) ─
    let a_par = alpha_par(iz, t_s);
    let tau_base = params.layer_thermal_tau_long[iz];
    if tau_base > 0.0 && a_par > 1e-9 {
        let rate = a_par / tau_base;         // 1/s
        let m_eq = sample_m_e(iz, t_s);
        dmdt = dmdt + rate * (m_eq - m_mag) * m_hat;
    }

    // SOT (only nonzero for layer 0) — preserved from LLG path.
    let sigma = params.layer_sigmas[iz].xyz;
    let mxs = cross(m, sigma);
    let mxmxs = cross(m_hat, cross(m_hat, sigma)) * m_mag;
    dmdt = dmdt + -params.gamma * (params.layer_sot_tau_dls[iz] * mxmxs
                                  + params.layer_sot_tau_fls[iz] * mxs);
    return dmdt;
}

/// Field-torque phase0 replacement for LLB: reads the same B_eff that the
/// LLG `field_torque_phase0` computes, but stores `llb_torque(m, B_eff)`
/// rather than `llg_torque(m, B_eff)` in torque_0. Keeps b_eff in h_eff so
/// diagnostics and any shared downstream kernels still work.
@compute @workgroup_size(64)
fn field_torque_phase0_llb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }
    let iz = layer_of(idx);
    let m = read_mag(idx);
    let b_eff = exchange_from_mag(idx) + exchange_interlayer_from_mag(idx)
              + anisotropy_field(m, iz) + zeeman_field() + laser_field_at(idx)
              + exchange_bias_field(iz) + dmi_from_mag(idx);
    let t_s = temp_e[idx];
    let t = llb_torque(m, b_eff, iz, t_s);
    let base = idx * 4u;
    h_eff[base] = b_eff.x; h_eff[base + 1u] = b_eff.y; h_eff[base + 2u] = b_eff.z;
    torque_0[base] = t.x; torque_0[base + 1u] = t.y; torque_0[base + 2u] = t.z;
}

@compute @workgroup_size(64)
fn field_torque_phase1_llb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }
    let iz = layer_of(idx);
    let m = read_pred(idx);
    let b_eff = exchange_from_pred(idx) + exchange_interlayer_from_pred(idx)
              + anisotropy_field(m, iz) + zeeman_field() + laser_field_at(idx)
              + exchange_bias_field(iz) + dmi_from_pred(idx);
    let t_s = temp_e[idx];
    let t = llb_torque(m, b_eff, iz, t_s);
    let base = idx * 4u;
    h_eff[base] = b_eff.x; h_eff[base + 1u] = b_eff.y; h_eff[base + 2u] = b_eff.z;
    torque_1[base] = t.x; torque_1[base + 1u] = t.y; torque_1[base + 2u] = t.z;
}

@compute @workgroup_size(64)
fn llb_predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }
    let b = idx * 4u;
    let m = vec3<f32>(mag[b], mag[b + 1u], mag[b + 2u]);
    let t0 = vec3<f32>(torque_0[b], torque_0[b + 1u], torque_0[b + 2u]);
    // NO normalize — LLB tracks |m|.
    var m_star = m + params.dt * t0;
    let len = length(m_star);
    if len < 1e-6 {
        // Floor clamp: preserve direction, clip magnitude.
        m_star = (m / max(length(m), 1e-12)) * 1e-6;
    }
    mag_pred[b] = m_star.x;
    mag_pred[b + 1u] = m_star.y;
    mag_pred[b + 2u] = m_star.z;
}

@compute @workgroup_size(64)
fn llb_correct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }
    let b = idx * 4u;
    let m = vec3<f32>(mag[b], mag[b + 1u], mag[b + 2u]);
    let t0 = vec3<f32>(torque_0[b], torque_0[b + 1u], torque_0[b + 2u]);
    let t1 = vec3<f32>(torque_1[b], torque_1[b + 1u], torque_1[b + 2u]);
    var m_new = m + 0.5 * params.dt * (t0 + t1);
    let len = length(m_new);
    if len < 1e-6 {
        m_new = (m / max(length(m), 1e-12)) * 1e-6;
    }
    mag[b] = m_new.x;
    mag[b + 1u] = m_new.y;
    mag[b + 2u] = m_new.z;
}

@compute @workgroup_size(64)
fn advance_m3tm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }
    let iz = layer_of(idx);
    let dt = params.dt;

    // P3c: when LLB owns |m|, feed it into Koopmans R from the mag buffer.
    // Otherwise (P3a/P3b path with enable_llb=0), M3TM keeps its own |m|
    // track in m_reduced.
    let b = idx * 4u;
    let mag_mag = length(vec3<f32>(mag[b], mag[b + 1u], mag[b + 2u]));
    let m_input = select(m_reduced[idx], mag_mag, params.enable_llb_flag != 0u);

    var s: M3tmState;
    s.t_e = temp_e[idx];
    s.t_p = temp_p[idx];
    s.m   = m_input;

    let p_laser = laser_power_density_at(idx);

    // Heun predictor
    let d1 = m3tm_derivs(s, p_laser, iz);
    var s_pred: M3tmState;
    s_pred.t_e = max(s.t_e + dt * d1.x, 1.0);
    s_pred.t_p = max(s.t_p + dt * d1.y, 1.0);
    s_pred.m   = clamp(s.m + dt * d1.z, -1.0, 1.0);

    // Heun corrector
    let d2 = m3tm_derivs(s_pred, p_laser, iz);
    let new_te = max(s.t_e + 0.5 * dt * (d1.x + d2.x), 1.0);
    let new_tp = max(s.t_p + 0.5 * dt * (d1.y + d2.y), 1.0);
    let new_m  = clamp(s.m + 0.5 * dt * (d1.z + d2.z), -1.0, 1.0);

    temp_e[idx] = new_te;
    temp_p[idx] = new_tp;
    // With LLB active, mirror m_reduced to |mag| for observability; LLB
    // owns the source of truth. Otherwise M3TM writes its integrated |m|.
    m_reduced[idx] = select(new_m, mag_mag, params.enable_llb_flag != 0u);
}
