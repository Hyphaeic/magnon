//! Phase F4 — Host-side reference single-cell LLB + M3TM integrator.
//!
//! Mirrors `llg.wgsl` for ONE thermally-isolated cell with no in-plane or
//! interlayer neighbours (exchange and DMI vanish for a uniform state), so it
//! is exact for the uniform-illumination 1×1 configurations used by the
//! Zhou calibration and the pump-probe gates. It serves as:
//!   1. the oracle for `examples/test_llb_atxitia_gpu_vs_host.rs`;
//!   2. a CPU path for `examples/test_zhou_fgt_calibrate.rs --cpu`;
//!   3. the readable statement of the F4 longitudinal torque.
//!
//! Per sub-step, in the same order as `GpuSolver::step_n`:
//!   1. `advance_m3tm`   — Heun on (T_e, T_p); implicit-Euler on `m_target`.
//!   2. LLB Heun         — torque evaluated with the post-step T_e.
//!
//! Longitudinal torque (λ > 0, Garanin 1997 / Atxitia 2011):
//!     d|m|/dt = γ · α_∥(T) · H_∥ · |m|
//!     H_∥     = (1 − |m|²/m_e²) / (2·χ̃_∥)                 m_e > 0
//!     H_∥     = −(1 + 3·T_c·|m|²/(5·(T − T_c))) / χ̃_∥      m_e ≈ 0
//!     α_∥(T)  = 2·λ·T/(3·T_c),   χ̃_∥ = ∂m_e/∂B  [1/T]
//! Legacy (λ ≤ 0): d|m|/dt = α_∥(T)/τ_long · (m_target − |m|).

use crate::material_thermal::ZEEMAN_K_PER_T;
use crate::photonic::LayerThermalParams;

/// Static per-cell inputs.
#[derive(Clone, Debug)]
pub struct LlbCellParams {
    /// Gyromagnetic ratio [rad/(s·T)].
    pub gamma: f64,
    /// Anisotropy prefactor 2·K_u/Ms [T].
    pub anis_pf: f64,
    /// Unit easy axis.
    pub u_axis: [f64; 3],
    /// Applied field [T].
    pub b_ext: [f64; 3],
    /// Thermal / LLB preset for this layer.
    pub thermal: LayerThermalParams,
    /// Ambient bath temperature [K].
    pub t_ambient: f64,
    /// Optical attenuation factor [1/m] — incident flux × this = W/m³.
    pub optical_atten: f64,
}

/// Dynamic per-cell state.
#[derive(Clone, Copy, Debug)]
pub struct LlbCellState {
    pub m: [f64; 3],
    pub t_e: f64,
    pub t_p: f64,
    pub m_target: f64,
}

impl LlbCellState {
    /// Thermal equilibrium at `t_ambient` with |m| = m_e(T, |B_z|) along +z.
    pub fn equilibrium_z(p: &LlbCellParams) -> Self {
        let m_e = p.thermal.sample_m_e_2d(p.t_ambient, p.b_ext[2].abs());
        Self {
            m: [0.0, 0.0, m_e],
            t_e: p.t_ambient,
            t_p: p.t_ambient,
            m_target: m_e,
        }
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn llb_scale(m_e: f64, exponent: f64) -> f64 {
    let p = exponent - 2.0;
    if p.abs() < 1e-6 {
        1.0
    } else {
        m_e.max(1e-4).powf(p)
    }
}

/// Effective field for the single cell: scaled anisotropy + Zeeman.
fn b_eff(p: &LlbCellParams, m: [f64; 3], m_e: f64) -> [f64; 3] {
    let s_k = llb_scale(m_e, p.thermal.k_u_exponent);
    let mu = dot(m, p.u_axis);
    add(scale(p.u_axis, s_k * p.anis_pf * mu), p.b_ext)
}

/// dm/dt for the LLB vector m (not unit-norm). Mirrors `llg.wgsl::llb_torque`.
pub fn llb_torque(p: &LlbCellParams, m: [f64; 3], t_s: f64, m_target: f64, dt: f64) -> [f64; 3] {
    let th = &p.thermal;
    let m_mag = norm(m).max(1e-6);
    let m_hat = scale(m, 1.0 / m_mag);
    let b_long = p.b_ext[2].abs();
    let m_e = th.sample_m_e_2d(t_s, b_long);
    let b = b_eff(p, m, m_e);
    let a_perp = th.alpha_perp(t_s).max(0.0);
    let gamma = p.gamma;
    let mxb = cross(m, b);

    if th.uses_atxitia() {
        let mxmxb = cross(m_hat, cross(m_hat, b));
        let mut dmdt = scale(add(mxb, scale(mxmxb, a_perp)), -gamma);
        let chi = (th.sample_chi_par_2d(t_s, b_long) * ZEEMAN_K_PER_T).max(1e-12);
        let a_par = th.alpha_par_lambda(t_s);
        if a_par > 1e-12 {
            let (h_long, m_eq) = if m_e > 1e-3 {
                ((1.0 - m_mag * m_mag / (m_e * m_e)) / (2.0 * chi), m_e)
            } else {
                let dt_c = (t_s - th.t_c).max(1e-3 * th.t_c);
                (-(1.0 + 0.6 * th.t_c * m_mag * m_mag / dt_c) / chi, 0.0)
            };
            let mut long_rate = gamma * a_par * h_long;
            let dist = m_mag - m_eq;
            let step_mag = (long_rate * m_mag).abs() * dt;
            if step_mag > 0.5 * dist.abs() {
                long_rate *= 0.5 * dist.abs() / step_mag;
            }
            dmdt = add(dmdt, scale(m_hat, long_rate * m_mag));
        }
        dmdt
    } else {
        let gp = gamma / (1.0 + a_perp * a_perp);
        let mxmxb = scale(cross(m_hat, cross(m_hat, b)), m_mag);
        let mut dmdt = scale(add(mxb, scale(mxmxb, a_perp)), -gp);
        let a_par = if th.t_c < 1.0 {
            0.0
        } else {
            th.alpha_0 * 2.0 * t_s / (3.0 * th.t_c)
        };
        if th.tau_long_base > 0.0 && a_par > 1e-9 {
            let rate = a_par / th.tau_long_base;
            dmdt = add(dmdt, scale(m_hat, rate * (m_target - m_mag)));
        }
        dmdt
    }
}

/// One full sub-step: M3TM baths, m_target proxy, then LLB Heun.
/// `p_flux` is the instantaneous incident surface flux [W/m²] at the
/// step midpoint (0 when no fluence is attached to the pulse).
pub fn step(p: &LlbCellParams, s: LlbCellState, p_flux: f64, dt: f64) -> LlbCellState {
    use crate::thermal::{advance_m3tm_cell, M3tmState};
    let th = &p.thermal;

    // 1. M3TM on (T_e, T_p). Koopmans |m| track is a mirror of |m| under LLB.
    let m_mag = norm(s.m);
    let bath = advance_m3tm_cell(
        M3tmState {
            t_e: s.t_e,
            t_p: s.t_p,
            m: m_mag,
        },
        p_flux * p.optical_atten,
        th,
        p.t_ambient,
        dt,
    );
    let t_new = bath.t_e;

    // m_target proxy (legacy F2 chain; unused by the Atxitia torque).
    let b_long = p.b_ext[2].abs();
    let m_e_now = th.sample_m_e_2d(t_new, b_long);
    let a_par_now = if th.t_c < 1.0 {
        0.0
    } else {
        th.alpha_0 * 2.0 * t_new / (3.0 * th.t_c)
    };
    let m_target = if th.tau_fast_base > 0.0 && a_par_now > 1e-9 {
        let alpha_dt = a_par_now / th.tau_fast_base * dt;
        ((s.m_target + alpha_dt * m_e_now) / (1.0 + alpha_dt)).clamp(-1.0, 1.0)
    } else {
        m_e_now
    };

    // 2. LLB Heun with the post-step T_e (same as the GPU ordering).
    let floor = |v: [f64; 3], prev: [f64; 3]| {
        if norm(v) < 1e-6 {
            scale(prev, 1e-6 / norm(prev).max(1e-12))
        } else {
            v
        }
    };
    let t0 = llb_torque(p, s.m, t_new, m_target, dt);
    let m_star = floor(add(s.m, scale(t0, dt)), s.m);
    let t1 = llb_torque(p, m_star, t_new, m_target, dt);
    let m_new = floor(add(s.m, scale(add(t0, t1), 0.5 * dt)), s.m);

    LlbCellState {
        m: m_new,
        t_e: bath.t_e,
        t_p: bath.t_p,
        m_target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material_thermal;

    fn fgt_cell(lambda: f64, t_ambient: f64, b_z: f64) -> LlbCellParams {
        let mut th = material_thermal::fgt_zhou_calibrated();
        th.lambda_bath = lambda;
        LlbCellParams {
            gamma: 1.7595e11,
            anis_pf: 2.0 * 1.46e6 / 3.76e5,
            u_axis: [0.0, 0.0, 1.0],
            b_ext: [0.0, 0.0, b_z],
            thermal: th,
            t_ambient,
            optical_atten: 0.0,
        }
    }

    #[test]
    fn atxitia_equilibrium_is_a_fixed_point() {
        let p = fgt_cell(0.1, 150.0, 1.0);
        let s0 = LlbCellState::equilibrium_z(&p);
        let mut s = s0;
        for _ in 0..2000 {
            s = step(&p, s, 0.0, 1e-15);
        }
        assert!(
            (norm(s.m) - norm(s0.m)).abs() < 1e-4,
            "|m| drifted from m_e: {} -> {}",
            norm(s0.m),
            norm(s.m)
        );
    }

    #[test]
    fn atxitia_relaxes_toward_m_e_from_above_and_below() {
        let p = fgt_cell(0.1, 150.0, 1.0);
        let m_e = p.thermal.sample_m_e_2d(150.0, 1.0);
        for start in [0.5 * m_e, 1.0] {
            let mut s = LlbCellState::equilibrium_z(&p);
            s.m = [0.0, 0.0, start];
            let d0 = (start - m_e).abs();
            for _ in 0..20_000 {
                s = step(&p, s, 0.0, 1e-15);
            }
            let d1 = (norm(s.m) - m_e).abs();
            assert!(d1 < 0.5 * d0, "no relaxation from {start}: {d0} -> {d1}");
        }
    }

    #[test]
    fn atxitia_longitudinal_vanishes_at_zero_kelvin() {
        let p = fgt_cell(0.1, 0.0, 0.0);
        let m = [0.0, 0.1, 0.9];
        let t = llb_torque(&p, m, 0.0, 1.0, 1e-15);
        // Longitudinal projection of the torque must be ~0 (α_∥(0) = 0).
        let m_hat = scale(m, 1.0 / norm(m));
        assert!(dot(t, m_hat).abs() < 1e-3 * norm(t).max(1.0));
    }

    #[test]
    fn alpha_perp_is_continuous_and_positive_above_tc() {
        let th = material_thermal::fgt_zhou_calibrated();
        let tc = th.t_c;
        let below = th.alpha_perp(tc - 1e-6);
        let above = th.alpha_perp(tc + 1e-6);
        assert!((below - above).abs() < 1e-6 * th.alpha_0);
        assert!(th.alpha_perp(3.0 * tc) > 0.0);
    }

    #[test]
    fn stiffness_guard_keeps_low_t_stable() {
        // At 0.05·Tc the MFA χ̃ is ~1e-30: unguarded the longitudinal rate
        // would be ~1e40 /s. The guard must keep |m| finite and pinned.
        let p = fgt_cell(0.3, 0.05 * 210.0, 0.0);
        let mut s = LlbCellState::equilibrium_z(&p);
        s.m = [0.0, 0.0, 0.98];
        for i in 0..500 {
            s = step(&p, s, 0.0, 5e-15);
            assert!(
                norm(s.m).is_finite() && norm(s.m) <= 1.0 + 1e-6,
                "step {i}: {:?}",
                s.m
            );
        }
        assert!((norm(s.m) - 1.0).abs() < 1e-3);
    }
}
