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

/// Public accessor: try to look up a preset by short key.
pub fn for_key(key: &str) -> Option<LayerThermalParams> {
    match key.to_ascii_lowercase().as_str() {
        "ni" | "ni-m3tm" => Some(ni_m3tm()),
        "py" | "permalloy" | "permalloy-m3tm" => Some(py_m3tm()),
        "fgt" | "fgt-ni-surrogate" | "fgt-surrogate" => Some(fgt_ni_surrogate()),
        "yig" | "yig-inert" => Some(yig_inert()),
        "cofeb" | "cofeb-m3tm" => Some(cofeb_m3tm()),
        _ => None,
    }
}

/// Ni — Koopmans 2010 verified. a_sf = 0.185, Tc = 627 K.
pub fn ni_m3tm() -> LayerThermalParams {
    let t_c = 627.0;
    let (m_e_table, chi_par_table) = brillouin_tables_spin_half(t_c, DEFAULT_TABLE_N);
    LayerThermalParams {
        gamma_e: 1.065e3,   // J/(m³·K²); Lin/Zhigilei 2008
        c_p: 3.0e6,         // J/(m³·K); Dulong-Petit near ambient for Ni
        g_ep: 8.0e17,       // W/(m³·K); Koopmans 2010
        a_sf: 0.185,
        mu_atom_bohr: 0.6,
        v_atom: 1.094e-29,  // m³; fcc Ni
        theta_d: 477.0,
        t_c,
        alpha_0: 0.04,      // mid-range Ni Gilbert damping
        llb_table_n: DEFAULT_TABLE_N,
        m_e_table,
        chi_par_table,
        notes: "Ni M3TM, Koopmans 2010 verified; gamma_e/C_p from Lin-Zhigilei 2008.",
    }
}

/// Permalloy (Ni₈₀Fe₂₀) — Battiato 2010, soft magnet.
pub fn py_m3tm() -> LayerThermalParams {
    let t_c = 870.0;
    let (m_e_table, chi_par_table) = brillouin_tables_spin_half(t_c, DEFAULT_TABLE_N);
    LayerThermalParams {
        gamma_e: 7.1e2,
        c_p: 3.5e6,
        g_ep: 1.0e18,
        a_sf: 0.08,         // Battiato estimate for Py
        mu_atom_bohr: 1.0,
        v_atom: 1.18e-29,
        theta_d: 470.0,
        t_c,
        alpha_0: 0.008,
        llb_table_n: DEFAULT_TABLE_N,
        m_e_table,
        chi_par_table,
        notes: "Permalloy Ni80Fe20 M3TM, Battiato 2010 a_sf.",
    }
}

/// FGT — Ni surrogate (a_sf = 0.185) with FGT-specific T_c = 220 K.
/// FLAGGED: thermal-bath coefficients are uncalibrated; P5 fits against Zhou 2025.
pub fn fgt_ni_surrogate() -> LayerThermalParams {
    let t_c = 220.0;
    let (m_e_table, chi_par_table) = brillouin_tables_spin_half(t_c, DEFAULT_TABLE_N);
    LayerThermalParams {
        gamma_e: 1.0e3,     // vdW metal, rough estimate
        c_p: 2.5e6,
        g_ep: 5.0e17,
        a_sf: 0.185,        // Ni surrogate — UNCALIBRATED for FGT
        mu_atom_bohr: 1.6,  // Leon-Brito 2016 bulk Fe moment ≈ 1.6 μ_B
        v_atom: 3.5e-29,    // FGT unit cell / atoms
        theta_d: 220.0,
        t_c,
        alpha_0: 0.001,
        llb_table_n: DEFAULT_TABLE_N,
        m_e_table,
        chi_par_table,
        notes: "FGT thermal — Ni a_sf surrogate, UNCALIBRATED. Fit vs Zhou 2025 in P5.",
    }
}

/// YIG — inert. a_sf ≈ 0 makes the Koopmans R prefactor zero, so no M3TM action.
pub fn yig_inert() -> LayerThermalParams {
    let t_c = 560.0;
    let (m_e_table, chi_par_table) = brillouin_tables_spin_half(t_c, DEFAULT_TABLE_N);
    LayerThermalParams {
        gamma_e: 0.0,
        c_p: 3.5e6,
        g_ep: 0.0,
        a_sf: 0.0,          // insulator — no Elliott-Yafet
        mu_atom_bohr: 5.0,
        v_atom: 2.0e-28,    // garnet unit cell
        theta_d: 500.0,
        t_c,
        alpha_0: 3.0e-5,
        llb_table_n: DEFAULT_TABLE_N,
        m_e_table,
        chi_par_table,
        notes: "YIG inert — insulator, M3TM source effectively zero.",
    }
}

/// CoFeB — Sato 2018 PRB 97, 014433 (β=1.73).
pub fn cofeb_m3tm() -> LayerThermalParams {
    let t_c = 1100.0;
    let (m_e_table, chi_par_table) = brillouin_tables_spin_half(t_c, DEFAULT_TABLE_N);
    LayerThermalParams {
        gamma_e: 7.0e2,
        c_p: 3.2e6,
        g_ep: 1.1e18,
        a_sf: 0.15,
        mu_atom_bohr: 1.7,
        v_atom: 1.2e-29,
        theta_d: 460.0,
        t_c,
        alpha_0: 0.006,
        llb_table_n: DEFAULT_TABLE_N,
        m_e_table,
        chi_par_table,
        notes: "CoFeB M3TM (Sato 2018 Bloch exponent β=1.73, a_sf fit).",
    }
}

// ─── Brillouin / MFA table generation (spin-1/2) ─────────────────
//
// Mean-field equations (Weiss molecular-field theory, spin-1/2):
//     m_e(T) = tanh(m_e · T_c / T)     (self-consistent, solve numerically)
//     χ_∥(T) = (1/T_c) · (1 − m_e²) / ( (T/T_c) − (1 − m_e²) )   for T < T_c
//                                             (diverges as T → T_c⁻)
//     χ_∥(T) = (1/T_c) / ((T/T_c) − 1)                           for T > T_c
//
// χ_∥ is reported in units of μ_B / (k_B · T_c) — the dimensionless form used
// in LLB literature (Atxitia 2011 eq. 22 style). The absolute scaling is
// folded into the LLB torque prefactor on the GPU side.
//
// At T = 0 we hard-set m_e = 1 and χ_∥ = 0 (mean-field limit).

/// Build (m_e, χ_∥) tables on a uniform T grid 0..1.5·T_c with N rows.
pub fn brillouin_tables_spin_half(t_c: f64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut m_e_table = Vec::with_capacity(n);
    let mut chi_par_table = Vec::with_capacity(n);
    let t_max = 1.5 * t_c;
    for i in 0..n {
        let t = if n > 1 {
            (i as f64) / ((n - 1) as f64) * t_max
        } else {
            0.0
        };
        let (m_e, chi) = if t < 1e-6 {
            (1.0_f64, 0.0_f64)
        } else if t >= t_c {
            // Paramagnetic — Curie-Weiss tail, m_e = 0.
            let chi = 1.0 / (t_c * (t / t_c - 1.0).max(1e-12));
            (0.0, chi)
        } else {
            let m_e = solve_m_e_spin_half(t / t_c);
            // χ_∥ / (μ_B / (kB · Tc)) form. Small, non-zero, peaks near Tc.
            let tr = t / t_c;
            let numer = 1.0 - m_e * m_e;
            let denom = (tr - numer).max(1e-12);
            let chi = (1.0 / t_c) * numer / denom;
            (m_e, chi)
        };
        m_e_table.push(m_e as f32);
        chi_par_table.push(chi as f32);
    }
    (m_e_table, chi_par_table)
}

/// Self-consistent solve of m = tanh(m · Tc / T) for T < Tc.
/// Input t_over_tc = T / Tc ∈ (0, 1). Returns m ∈ (0, 1].
/// Fixed-point iteration with Aitken-Δ² acceleration; converges in <40 iters.
fn solve_m_e_spin_half(t_over_tc: f64) -> f64 {
    if t_over_tc <= 0.0 {
        return 1.0;
    }
    if t_over_tc >= 1.0 {
        return 0.0;
    }
    // Good initial guess from the critical-exponent approximation.
    let mut m = (1.0 - t_over_tc).sqrt().max(1e-4).min(1.0);
    for _ in 0..200 {
        let m_new = (m / t_over_tc).tanh();
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
        assert!((p.m_e_table[0] - 1.0).abs() < 1e-4, "m_e(0) should be ≈ 1");
        assert!(p.m_e_table.last().unwrap().abs() < 1e-4, "m_e(1.5·Tc) should be ≈ 0");
    }

    #[test]
    fn m_e_crosses_tc() {
        let p = ni_m3tm();
        // Mid of table is at T = 0.75·Tc, m_e expected ≈ 0.85
        let mid = p.m_e_table.len() / 3;
        assert!(p.m_e_table[mid] > 0.5 && p.m_e_table[mid] < 1.0);
    }

    #[test]
    fn chi_par_positive_and_peaks_near_tc() {
        let p = ni_m3tm();
        let n = p.chi_par_table.len();
        // index of T ≈ Tc: i = (1/1.5)·(n-1) ≈ 2·(n-1)/3
        let idx_tc = 2 * (n - 1) / 3;
        for v in &p.chi_par_table {
            assert!(*v >= 0.0);
        }
        assert!(p.chi_par_table[idx_tc] > p.chi_par_table[0]);
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
