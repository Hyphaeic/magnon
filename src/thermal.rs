//! Phase P3a — Host-side reference single-cell M3TM integrator.
//!
//! Serves two purposes:
//!   1. Unit-test oracle for the GPU `advance_m3tm` kernel (1e-3 relative match).
//!   2. Readable source-of-truth for the Koopmans 2010 equations.
//!
//! Physics (Koopmans 2010 eq. 3-4, extended with a substrate-sink term on T_p):
//!     C_e(T_e) dT_e/dt = −g_ep·(T_e − T_p) + P_laser − R·m·(T_p/T_c)·(1 − m·coth(m·T_c/T_e))·E_mag
//!     C_p       dT_p/dt =  g_ep·(T_e − T_p) − g_sub·(T_p − T_ambient)
//!              dm/dt  =  R·m·(T_p/T_c)·(1 − m·coth(m·T_c/T_e))
//!
//! with C_e(T_e) = γ_e · T_e. We drop the magnetic back-reaction energy term
//! on C_e (standard Koopmans simplification; the electronic reservoir is much
//! larger than the magnetic-energy budget over 100 fs).
//!
//! Integrator: Heun (explicit trapezoidal, 2nd order). Stable for dt ≤ 1 fs.

use crate::photonic::LayerThermalParams;

/// Per-cell thermal / magnetic state tracked by M3TM.
#[derive(Clone, Copy, Debug)]
pub struct M3tmState {
    /// Electron bath temperature [K].
    pub t_e: f64,
    /// Phonon bath temperature [K].
    pub t_p: f64,
    /// Reduced cell magnetization |m| = |M|/Ms(0). Dimensionless 0..1.
    pub m: f64,
}

impl M3tmState {
    pub fn at_ambient(t_ambient: f64, m_initial: f64) -> Self {
        Self { t_e: t_ambient, t_p: t_ambient, m: m_initial }
    }
}

/// One Heun step of M3TM for a single cell.
///
/// `p_laser` is the absorbed volumetric power density [W/m³] at this cell at
/// the step's midpoint. `dt` is the thermal timestep [s] — typically ≤ 1 fs
/// during a pulse.
///
/// Returns the new state after `dt`.
pub fn advance_m3tm_cell(
    s: M3tmState,
    p_laser: f64,
    params: &LayerThermalParams,
    t_ambient: f64,
    dt: f64,
) -> M3tmState {
    let r = params.r_koopmans_prefactor();
    let k = M3tmDerivs { r, params, t_ambient };

    // Heun predictor: Euler step.
    let d1 = k.derivs(s, p_laser);
    let s_pred = M3tmState {
        t_e: (s.t_e + dt * d1.dt_e).max(1.0),
        t_p: (s.t_p + dt * d1.dt_p).max(1.0),
        m: (s.m + dt * d1.dm).clamp(-1.0, 1.0),
    };

    // Heun corrector: trapezoidal update.
    let d2 = k.derivs(s_pred, p_laser);
    M3tmState {
        t_e: (s.t_e + 0.5 * dt * (d1.dt_e + d2.dt_e)).max(1.0),
        t_p: (s.t_p + 0.5 * dt * (d1.dt_p + d2.dt_p)).max(1.0),
        m: (s.m + 0.5 * dt * (d1.dm + d2.dm)).clamp(-1.0, 1.0),
    }
}

struct M3tmDerivs<'a> {
    r: f64,
    params: &'a LayerThermalParams,
    t_ambient: f64,
}

#[derive(Clone, Copy, Debug)]
struct Rates {
    dt_e: f64,
    dt_p: f64,
    dm: f64,
}

impl<'a> M3tmDerivs<'a> {
    fn derivs(&self, s: M3tmState, p_laser: f64) -> Rates {
        let p = self.params;
        // C_e(T_e) = γ_e · T_e; guard against T_e = 0.
        let c_e = (p.gamma_e * s.t_e).max(1.0);
        let dt_e = (p_laser - p.g_ep * (s.t_e - s.t_p)) / c_e;
        let dt_p = (p.g_ep * (s.t_e - s.t_p) - p.g_sub_phonon * (s.t_p - self.t_ambient))
            / p.c_p.max(1.0);

        let dm = if self.r.abs() < 1e-20 || s.t_e < 1.0 || p.t_c < 1e-12 {
            0.0
        } else {
            // R·m·(T_p/T_c)·(1 − m·coth(m·T_c/T_e))
            let x = s.m * p.t_c / s.t_e;
            let coth = if x.abs() < 1e-4 {
                // Taylor-expand coth near 0 to avoid 1/x singularity:
                //   coth(x) = 1/x + x/3 − x³/45 + ...
                //   m · coth(m·Tc/Te) ≈ Te/Tc + (m²·Tc)/(3·Te) + …
                s.t_e / p.t_c + s.m * s.m * p.t_c / (3.0 * s.t_e)
            } else {
                s.m * coth_safe(x)
            };
            self.r * s.m * (s.t_p / p.t_c) * (1.0 - coth)
        };
        Rates { dt_e, dt_p, dm }
    }
}

#[inline]
fn coth_safe(x: f64) -> f64 {
    // coth(x) = (e^x + e^-x) / (e^x − e^-x). Use tanh form for stability.
    if x.abs() > 20.0 {
        x.signum()
    } else {
        1.0 / x.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material_thermal;

    #[test]
    fn no_laser_equilibrium_is_stable() {
        // Starting at (T_ambient, m_e(T_ambient)) — the system's true
        // equilibrium — the M3TM ODE has no drive (P_laser=0) and dm/dt=0.
        // Coupled (T_e, T_p, m) should not drift over 1000 fs.
        let p = material_thermal::ni_m3tm();
        let t_amb = 300.0_f64;
        let m_eq = p.sample_m_e(t_amb);
        let mut s = M3tmState::at_ambient(t_amb, m_eq);
        for _ in 0..1000 {
            s = advance_m3tm_cell(s, 0.0, &p, t_amb, 1e-15);
        }
        assert!((s.t_e - t_amb).abs() < 1e-3, "T_e drifted: {}", s.t_e);
        assert!((s.t_p - t_amb).abs() < 1e-3, "T_p drifted: {}", s.t_p);
        assert!((s.m - m_eq).abs() < 1e-3, "m drifted: {} from {}", s.m, m_eq);
    }

    #[test]
    fn ni_ultrafast_demag_7mj_cm2() {
        // Beaurepaire-style check (smoke-only; the real acceptance gate sits
        // against the GPU in integration tests). Target: |m| drops to ≈ 0.60
        // at ~500 fs for 7 mJ/cm² on 20 nm Ni.
        let p = material_thermal::ni_m3tm();
        let fluence = 70.0_f64;          // J/m² = 7 mJ/cm²
        let fwhm = 100e-15_f64;          // 100 fs
        let thickness = 20e-9_f64;       // 20 nm
        let sigma = fwhm / 2.354_820_045;
        // Peak power density: F / thickness / (sigma·sqrt(2π))
        let p_peak = fluence / thickness / (sigma * (2.0 * std::f64::consts::PI).sqrt());

        let mut s = M3tmState::at_ambient(300.0, 1.0);
        let dt = 1e-15;
        let t_peak = 300e-15;
        let mut min_m = 1.0_f64;
        for i in 0..3000 {
            let t = i as f64 * dt;
            let env = (-((t - t_peak).powi(2)) / (2.0 * sigma * sigma)).exp();
            let p_laser = p_peak * env;
            s = advance_m3tm_cell(s, p_laser, &p, 300.0, dt);
            min_m = min_m.min(s.m);
        }
        // Sanity envelope: |m| should drop non-trivially (> 20%) but not go
        // negative. Exact 40% target is an integration-level gate.
        assert!(min_m < 0.9, "|m| floor = {min_m:.3}, expected < 0.9");
        assert!(min_m > 0.0, "|m| went negative ({min_m:.3})");
    }
}
