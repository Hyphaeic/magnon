//! Phase P3 — Per-material M3TM + LLB preset library.
//!
//! Every `LayerThermalParams` preset carries:
//!   - Microscopic M3TM parameters (γ_e, c_p, g_ep, a_sf, μ_at, V_at, θ_D)
//!   - LLB parameters (Tc, α_0)
//!   - Precomputed m_e(T) and χ_∥(T) tables (Brillouin / MFA, spin-1/2 by default)
//!
//! Tables are sampled on a uniform grid `T_i = i · (1.5·Tc) / (N − 1)` for
//! `i = 0 .. N − 1`. Default N = 256.
//!
//! Provenance per preset:
//!   - `ni_m3tm`          Koopmans 2010, Nat. Mater. 9, 259 — verified parameter set
//!   - `py_m3tm`          Battiato 2010, PRL 105, 027203 — Permalloy
//!   - `fgt_ni_surrogate` FGT thermal-bath parameters with Ni a_sf as
//!                        starting-point surrogate (Tc=220 K from Leon-Brito 2016).
//!                        Flagged uncalibrated; P5 will fit against Zhou 2025.
//!   - `yig_inert`        YIG with a_sf ≈ 0 — effectively no M3TM.
//!   - `cofeb_m3tm`       CoFeB per Sato 2018 PRB 97, 014433 (β = 1.73 Bloch).

use crate::photonic::LayerThermalParams;

const DEFAULT_TABLE_N: usize = 256;
/// Number of bins on the longitudinal-field axis of the 2D LLB tables (Phase F1).
pub const DEFAULT_TABLE_N_B: usize = 32;
/// Default upper bound of the B-axis [Tesla] (Phase F1). Covers the saturation
/// regime for typical ferromagnets; finer resolution available by overriding.
pub const DEFAULT_B_MAX_T: f64 = 10.0;

/// Public accessor: try to look up a preset by short key.
pub fn for_key(key: &str) -> Option<LayerThermalParams> {
    match key.to_ascii_lowercase().as_str() {
        "ni" | "ni-m3tm" => Some(ni_m3tm()),
        "py" | "permalloy" | "permalloy-m3tm" => Some(py_m3tm()),
        "fgt" | "fgt-ni-surrogate" | "fgt-surrogate" => Some(fgt_ni_surrogate()),
        "fgt-zhou" | "fgt-calibrated" => Some(fgt_zhou_calibrated()),
        "yig" | "yig-inert" => Some(yig_inert()),
        "cofeb" | "cofeb-m3tm" => Some(cofeb_m3tm()),
        _ => None,
    }
}

/// Ni — Koopmans 2010 verified. a_sf = 0.185, Tc = 627 K.
pub fn ni_m3tm() -> LayerThermalParams {
    let t_c = 627.0;
    let (m_e_table, chi_par_table) =
        brillouin_tables_spin_half_2d(t_c, DEFAULT_TABLE_N, DEFAULT_TABLE_N_B, DEFAULT_B_MAX_T);
    LayerThermalParams {
        gamma_e: 1.065e3,   // J/(m³·K²); Lin/Zhigilei 2008
        c_p: 3.0e6,         // J/(m³·K); Dulong-Petit near ambient for Ni
        g_ep: 8.0e17,       // W/(m³·K); Koopmans 2010
        g_sub_phonon: 3.0e16,  // τ_sub ≈ 100 ps for 20 nm Ni on sapphire
        a_sf: 0.185,
        mu_atom_bohr: 0.6,
        v_atom: 1.094e-29,  // m³; fcc Ni
        theta_d: 477.0,
        t_c,
        alpha_0: 0.04,      // mid-range Ni Gilbert damping
        llb_table_n: DEFAULT_TABLE_N,
        llb_table_n_b: DEFAULT_TABLE_N_B,
        b_max_t: DEFAULT_B_MAX_T,
        m_e_table,
        chi_par_table,
        tau_long_base: 0.3e-15,
        tau_fast_base: 0.0,  // F2: 0 ⇒ collapse to F1 single-stage; recalibrate per-material for two-stage decay.
        optical_skin_depth_m: 0.0,  // F3: 0 ⇒ uniform-absorption (pre-F3); reference value at 400 nm: 14 nm.
        notes: "Ni M3TM, Koopmans 2010 verified; gamma_e/C_p from Lin-Zhigilei 2008.",
    }
}

/// Permalloy (Ni₈₀Fe₂₀) — Battiato 2010, soft magnet.
pub fn py_m3tm() -> LayerThermalParams {
    let t_c = 870.0;
    let (m_e_table, chi_par_table) =
        brillouin_tables_spin_half_2d(t_c, DEFAULT_TABLE_N, DEFAULT_TABLE_N_B, DEFAULT_B_MAX_T);
    LayerThermalParams {
        gamma_e: 7.1e2,
        c_p: 3.5e6,
        g_ep: 1.0e18,
        g_sub_phonon: 3.0e16,
        a_sf: 0.08,         // Battiato estimate for Py
        mu_atom_bohr: 1.0,
        v_atom: 1.18e-29,
        theta_d: 470.0,
        t_c,
        alpha_0: 0.008,
        llb_table_n: DEFAULT_TABLE_N,
        llb_table_n_b: DEFAULT_TABLE_N_B,
        b_max_t: DEFAULT_B_MAX_T,
        m_e_table,
        chi_par_table,
        tau_long_base: 0.3e-15,
        tau_fast_base: 0.0,  // F2: 0 ⇒ collapse to F1 single-stage; recalibrate per-material for two-stage decay.
        optical_skin_depth_m: 0.0,  // F3 default; reference value at 400 nm: 16 nm.
        notes: "Permalloy Ni80Fe20 M3TM, Battiato 2010 a_sf.",
    }
}

/// FGT — Ni surrogate (a_sf = 0.185) with FGT-specific T_c = 220 K.
/// FLAGGED: thermal-bath coefficients are uncalibrated; P5 fits against Zhou 2025.
pub fn fgt_ni_surrogate() -> LayerThermalParams {
    let t_c = 220.0;
    let (m_e_table, chi_par_table) =
        brillouin_tables_spin_half_2d(t_c, DEFAULT_TABLE_N, DEFAULT_TABLE_N_B, DEFAULT_B_MAX_T);
    LayerThermalParams {
        gamma_e: 1.0e3,     // vdW metal, rough estimate
        c_p: 2.5e6,
        g_ep: 5.0e17,
        g_sub_phonon: 2.0e16,  // weaker sink: vdW contact, worse thermal conduction
        a_sf: 0.185,        // Ni surrogate — UNCALIBRATED for FGT
        mu_atom_bohr: 1.6,  // Leon-Brito 2016 bulk Fe moment ≈ 1.6 μ_B
        v_atom: 3.5e-29,    // FGT unit cell / atoms
        theta_d: 220.0,
        t_c,
        alpha_0: 0.001,
        llb_table_n: DEFAULT_TABLE_N,
        llb_table_n_b: DEFAULT_TABLE_N_B,
        b_max_t: DEFAULT_B_MAX_T,
        m_e_table,
        chi_par_table,
        tau_long_base: 0.3e-15,
        tau_fast_base: 0.0,  // F2: 0 ⇒ collapse to F1 single-stage; recalibrate per-material for two-stage decay.
        optical_skin_depth_m: 0.0,  // F3 default; reference value at 400 nm: 18 nm.
        notes: "FGT thermal — Ni a_sf surrogate, UNCALIBRATED. Fit vs Zhou 2025 in P5.",
    }
}

/// YIG — inert. a_sf ≈ 0 makes the Koopmans R prefactor zero, so no M3TM action.
pub fn yig_inert() -> LayerThermalParams {
    let t_c = 560.0;
    let (m_e_table, chi_par_table) =
        brillouin_tables_spin_half_2d(t_c, DEFAULT_TABLE_N, DEFAULT_TABLE_N_B, DEFAULT_B_MAX_T);
    LayerThermalParams {
        gamma_e: 0.0,
        c_p: 3.5e6,
        g_ep: 0.0,
        g_sub_phonon: 0.0,     // inert — no thermal coupling to worry about
        a_sf: 0.0,          // insulator — no Elliott-Yafet
        mu_atom_bohr: 5.0,
        v_atom: 2.0e-28,    // garnet unit cell
        theta_d: 500.0,
        t_c,
        alpha_0: 3.0e-5,
        llb_table_n: DEFAULT_TABLE_N,
        llb_table_n_b: DEFAULT_TABLE_N_B,
        b_max_t: DEFAULT_B_MAX_T,
        m_e_table,
        chi_par_table,
        tau_long_base: 0.3e-15,
        tau_fast_base: 0.0,  // F2: 0 ⇒ collapse to F1 single-stage; recalibrate per-material for two-stage decay.
        optical_skin_depth_m: 0.0,  // F3: YIG is largely transparent at visible; uniform mode is harmless because a_sf ≈ 0.
        notes: "YIG inert — insulator, M3TM source effectively zero.",
    }
}

/// FGT — **Zhou 2025 morphology-calibrated** (not a literal reproduction).
///
/// Derived from `fgt_ni_surrogate` with two parameters tuned against Zhou et
/// al. *Natl. Sci. Rev.* 12, nwaf185 (2025) *aggregate morphology* at
/// T = 150 K, B_z = 1 T, F = 0.24 mJ/cm², 150 fs pulse on a 5 nm flake.
/// See `docs/zhou_fgt_calibration.md` and ADR-005 for the full set of
/// assumptions — including the temperature shift from Zhou's T = T_c = 210 K
/// to T = 150 K (our zero-field tables make T = T_c degenerate).
///
/// Fit numbers (best-fit grid point from `examples/test_zhou_fgt_calibrate.rs`):
///   demag fraction = 86.5 % (Zhou target 79 %)
///   recovery fraction at 22 ps = 60.1 % (Zhou target ≈ 55 %)
///
/// Use this preset only with reflectivity = 0.50 applied on the pulse spec.
pub fn fgt_zhou_calibrated() -> LayerThermalParams {
    let mut p = fgt_ni_surrogate();
    p.t_c = 210.0;
    let (m_e, chi) =
        brillouin_tables_spin_half_2d(p.t_c, p.llb_table_n, p.llb_table_n_b, p.b_max_t);
    p.m_e_table = m_e;
    p.chi_par_table = chi;
    p.tau_long_base = 3.0e-15;
    p.g_sub_phonon = 2.0e17;
    p.notes = "FGT Zhou 2025 morphology fit at T=150K B=1T (see docs/zhou_fgt_calibration.md + ADR-005). NOT a literal T_c reproduction.";
    p
}

/// CoFeB — Sato 2018 PRB 97, 014433 (β=1.73).
pub fn cofeb_m3tm() -> LayerThermalParams {
    let t_c = 1100.0;
    let (m_e_table, chi_par_table) =
        brillouin_tables_spin_half_2d(t_c, DEFAULT_TABLE_N, DEFAULT_TABLE_N_B, DEFAULT_B_MAX_T);
    LayerThermalParams {
        gamma_e: 7.0e2,
        c_p: 3.2e6,
        g_ep: 1.1e18,
        g_sub_phonon: 3.0e16,
        a_sf: 0.15,
        mu_atom_bohr: 1.7,
        v_atom: 1.2e-29,
        theta_d: 460.0,
        t_c,
        alpha_0: 0.006,
        llb_table_n: DEFAULT_TABLE_N,
        llb_table_n_b: DEFAULT_TABLE_N_B,
        b_max_t: DEFAULT_B_MAX_T,
        m_e_table,
        chi_par_table,
        tau_long_base: 0.3e-15,
        tau_fast_base: 0.0,  // F2: 0 ⇒ collapse to F1 single-stage; recalibrate per-material for two-stage decay.
        optical_skin_depth_m: 0.0,  // F3 default; reference value at 400 nm: 17 nm.
        notes: "CoFeB M3TM (Sato 2018 Bloch exponent β=1.73, a_sf fit).",
    }
}

// ─── Brillouin / MFA table generation (spin-1/2) ─────────────────
//
// Mean-field equations (Weiss molecular-field theory, spin-1/2) **with an
// applied longitudinal field** (Phase F1):
//     m_e(T, B) = tanh( (m_e · T_c + h_B) / T )
// where the reduced Zeeman temperature is
//     h_B = g · μ_B · B / k_B
// (with g = 2 by convention). At B = 0 this reduces to the original
//     m_e(T, 0) = tanh(m_e · T_c / T)
// solved by `solve_m_e_spin_half`.
//
// Above T_c at B = 0 we have m_e = 0; with B > 0 the susceptibility tail
// gives a small but finite m_e set by the same self-consistent solve.
//
// χ_∥ form (low-field, near-equilibrium):
//     χ_∥(T, 0) = (1/T_c) · (1 − m_e²) / ( T/T_c − (1 − m_e²) )   for T < T_c
//     χ_∥(T, 0) = (1/T_c) / (T/T_c − 1)                            for T > T_c
// At finite B we evaluate χ_∥ at the field-shifted m_e, which captures the
// dominant correction without needing a second numerical derivative.
//
// At T = 0 we hard-set m_e = 1 and χ_∥ = 0 for any B.

/// Backward-compatibility 1D builder. Returns the B = 0 slice of the new
/// 2D tables. Length n.
pub fn brillouin_tables_spin_half(t_c: f64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let (m_e_2d, chi_2d) = brillouin_tables_spin_half_2d(t_c, n, 1, DEFAULT_B_MAX_T);
    (m_e_2d, chi_2d)
}

/// Build 2D (m_e, χ_∥) tables on a uniform (T, B) grid.
/// Layout: row-major `[i_T * n_b + i_B]`, length n_t · n_b.
pub fn brillouin_tables_spin_half_2d(
    t_c: f64,
    n_t: usize,
    n_b: usize,
    b_max_t: f64,
) -> (Vec<f32>, Vec<f32>) {
    // g·μ_B / k_B in K/T  ≈ 2 · 9.274e-24 / 1.381e-23 ≈ 1.343 K/T.
    const ZEEMAN_K_PER_T: f64 = 1.343;
    let n_b = n_b.max(1);
    let mut m_e_table = Vec::with_capacity(n_t * n_b);
    let mut chi_par_table = Vec::with_capacity(n_t * n_b);
    let t_max = 1.5 * t_c;
    for i_t in 0..n_t {
        let t = if n_t > 1 {
            (i_t as f64) / ((n_t - 1) as f64) * t_max
        } else {
            0.0
        };
        for i_b in 0..n_b {
            let b = if n_b > 1 {
                (i_b as f64) / ((n_b - 1) as f64) * b_max_t
            } else {
                0.0
            };
            let h_b_kelvin = ZEEMAN_K_PER_T * b;
            let (m_e, chi) = if t < 1e-6 {
                (1.0_f64, 0.0_f64)
            } else {
                let h_reduced = h_b_kelvin / t_c; // h in units of T_c
                let m_e = solve_m_e_spin_half_with_field(t / t_c, h_reduced);
                // χ_∥ form. At B = 0 this gives the original 1D expression;
                // the field-shifted m_e propagates into both numer and denom.
                let tr = t / t_c;
                let numer = 1.0 - m_e * m_e;
                let denom = (tr - numer).max(1e-12);
                let chi = (1.0 / t_c) * numer / denom;
                (m_e, chi)
            };
            m_e_table.push(m_e as f32);
            chi_par_table.push(chi as f32);
        }
    }
    (m_e_table, chi_par_table)
}


/// Self-consistent solve of `m = tanh((m + h)·Tc/T)` with a reduced applied
/// field `h = g·μ_B·B / (k_B·T_c)`. h = 0 reduces to the zero-field case.
///
/// At B > 0 there is always a unique solution in (0, 1) for any T > 0
/// (the field tilts the free-energy minimum toward + so the symmetric
/// disordered solution at T > T_c becomes a small but nonzero positive m).
fn solve_m_e_spin_half_with_field(t_over_tc: f64, h: f64) -> f64 {
    if t_over_tc <= 0.0 {
        return 1.0;
    }
    if h.abs() < 1e-12 {
        if t_over_tc >= 1.0 {
            return 0.0;
        }
        // Spontaneous-magnetization branch (zero field, T < Tc).
        let mut m = (1.0 - t_over_tc).sqrt().max(1e-4).min(1.0);
        for _ in 0..200 {
            let m_new = (m / t_over_tc).tanh();
            if (m_new - m).abs() < 1e-10 {
                return m_new;
            }
            m = m_new;
        }
        return m;
    }
    // Field-aligned branch: pick initial guess that's already field-tilted.
    // For T >> Tc and small h: m ≈ h / t_over_tc (Curie-Weiss high-T limit).
    // For T < Tc: m ≈ sqrt(1 - t_over_tc) + h-tilt.
    let mut m = if t_over_tc >= 1.0 {
        (h / t_over_tc).clamp(-0.999, 0.999)
    } else {
        ((1.0 - t_over_tc).sqrt() + h).clamp(1e-4, 0.999)
    };
    for _ in 0..200 {
        let m_new = ((m + h) / t_over_tc).tanh();
        if (m_new - m).abs() < 1e-10 {
            return m_new;
        }
        m = m_new;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_e_bounds_hold() {
        let p = ni_m3tm();
        // (T=0, B=0) corner.
        assert!((p.sample_m_e_2d(0.0, 0.0) - 1.0).abs() < 1e-4, "m_e(0,0) should be ≈ 1");
        // (T=1.5·Tc, B=0) corner — disordered paramagnet.
        assert!(p.sample_m_e_2d(1.5 * p.t_c, 0.0).abs() < 1e-4, "m_e(1.5·Tc, 0) should be ≈ 0");
        // (T=Tc, B=0) — exactly at critical point, m_e = 0.
        assert!(p.sample_m_e_2d(p.t_c, 0.0).abs() < 1e-3, "m_e(Tc, 0) should be ≈ 0");
        // (T=Tc, B=b_max) — strong field lifts m_e well above zero.
        let m_at_tc_high_b = p.sample_m_e_2d(p.t_c, p.b_max_t);
        assert!(m_at_tc_high_b > 0.05, "m_e(Tc, b_max) should be > 0.05; got {m_at_tc_high_b}");
        // (T=0.5·Tc, B=0) — well-magnetised; about 0.95 for spin-1/2 MFA.
        let m_lowt = p.sample_m_e_2d(0.5 * p.t_c, 0.0);
        assert!(m_lowt > 0.85 && m_lowt < 1.0, "m_e(0.5Tc, 0) outside [0.85, 1.0]; got {m_lowt}");
    }

    #[test]
    fn m_e_field_dependence_monotone() {
        // At any fixed T, m_e(T, B) should be non-decreasing in B (alignment
        // with the field). Spot-check at a few temperatures.
        let p = ni_m3tm();
        for &t_frac in &[0.3, 0.7, 1.0, 1.3] {
            let t = t_frac * p.t_c;
            let m0 = p.sample_m_e_2d(t, 0.0);
            let m_mid = p.sample_m_e_2d(t, p.b_max_t * 0.5);
            let m_max = p.sample_m_e_2d(t, p.b_max_t);
            assert!(m_mid >= m0 - 1e-6, "m_e not monotone in B at T={t}");
            assert!(m_max >= m_mid - 1e-6, "m_e not monotone in B at T={t}");
        }
    }

    #[test]
    fn chi_par_positive_and_peaks_near_tc() {
        let p = ni_m3tm();
        // Sample at B=0 (the standard chi_par definition).
        for &t_frac in &[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] {
            let chi = p.sample_chi_par_2d(t_frac * p.t_c, 0.0);
            assert!(chi >= 0.0, "chi_par negative at T/Tc = {t_frac}: {chi}");
        }
        let chi_low = p.sample_chi_par_2d(0.0, 0.0);
        let chi_near_tc = p.sample_chi_par_2d(p.t_c * 0.95, 0.0);
        assert!(chi_near_tc > chi_low, "chi_par should peak near Tc");
    }

    #[test]
    fn r_prefactor_positive_for_ni() {
        let r = ni_m3tm().r_koopmans_prefactor();
        // Ni: ≈ 2-3e12 /s; check within 2 orders of magnitude.
        assert!(r > 1e10 && r < 1e14, "Ni R = {r:.3e} not in expected band");
    }

    #[test]
    fn r_prefactor_zero_for_yig() {
        assert!(yig_inert().r_koopmans_prefactor().abs() < 1e-12);
    }
}
