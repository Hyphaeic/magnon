//! Effective parameters — the flat values the GPU actually sees.
//!
//! Composed from `(BulkMaterial, Substrate, Geometry)` via the thickness-
//! scaling decomposition rules documented in docs/plan.md §2.
//!
//! The GpuParams struct in `crate::gpu` converts these into the shader
//! uniform layout. Phase 1 uses only the LLG-core fields (ms, a_ex, k_u,
//! alpha, gamma). Fields `d_dmi`, `b_bias`, `spin_hall_angle`, and
//! `tau_fl_ratio` are computed and stored for use by later phases; the
//! shader ignores them until Phase 2+.

use crate::config::Geometry;
use crate::material::BulkMaterial;
use crate::substrate::Substrate;

#[derive(Clone, Debug)]
pub struct EffectiveParams {
    /// Saturation magnetization [A/m]
    pub ms: f64,
    /// Exchange stiffness [J/m]
    pub a_ex: f64,
    /// Effective uniaxial anisotropy [J/m³]
    pub k_u: f64,
    /// Effective Gilbert damping
    pub alpha: f64,
    /// Gyromagnetic ratio [rad/(s·T)]
    pub gamma: f64,
    /// Effective DMI as 2D surface density [J/m²] — Phase 3+
    pub d_dmi: f64,
    /// Exchange bias field [T] — Phase 2+
    pub b_bias: [f64; 3],
    /// Spin Hall angle — Phase 4+
    pub spin_hall_angle: f64,
    /// Field-like torque ratio — Phase 4+
    pub tau_fl_ratio: f64,
}

impl EffectiveParams {
    /// Build effective parameters from (bulk material, substrate, geometry).
    ///
    /// Decomposition rules:
    ///   Ms_eff    = Ms_bulk + ΔMs_proximity               (additive)
    ///   A_eff     = A_bulk                                 (bulk-only in leading order)
    ///   K_u_eff   = K_u_bulk + K_u_surface / t             (bulk volume + interface surface/thickness)
    ///   α_eff     = α_bulk + α_pumping                     (additive)
    ///   γ_eff     = γ                                      (material-only)
    ///   D_DMI_eff = D_DMI_surface                           (used directly — MuMax3 Dind convention, J/m²)
    ///   B_bias    = substrate.b_exchange_bias              (substrate-only)
    ///
    /// Note on D_DMI: the experimental value D (J/m²) reported in literature
    /// IS already the effective 3D micromagnetic strength for the measured
    /// sample — it is not a microscopic surface density requiring thickness
    /// scaling. Different measured thicknesses give different D values.
    /// We follow the MuMax3 `Dind` convention: use D directly.
    pub fn from_parts(bulk: &BulkMaterial, substrate: &Substrate, geometry: &Geometry) -> Self {
        let t = geometry.thickness;
        Self {
            ms: bulk.ms_bulk + substrate.delta_ms_proximity,
            a_ex: bulk.a_ex_bulk,
            k_u: bulk.k_u_bulk + substrate.k_u_surface / t,
            alpha: bulk.alpha_bulk + substrate.alpha_pumping,
            gamma: bulk.gamma,
            d_dmi: substrate.d_dmi_surface,
            b_bias: substrate.b_exchange_bias,
            spin_hall_angle: substrate.spin_hall_angle,
            tau_fl_ratio: substrate.tau_fl_ratio,
        }
    }

    /// B_exch prefactor for the Laplacian stencil: 2·A / (Ms · dx²) [Tesla]
    pub fn exchange_prefactor(&self, dx: f64) -> f64 {
        2.0 * self.a_ex / (self.ms * dx * dx)
    }

    /// B_anis prefactor for uniaxial anisotropy: 2·K_u / Ms [Tesla]
    pub fn anisotropy_prefactor(&self) -> f64 {
        2.0 * self.k_u / self.ms
    }

    /// B_DMI prefactor for the interfacial-DMI central-difference stencil:
    ///   B_DMI_x = dmi_pf · (m_z[i+1] - m_z[i-1])     [T]
    /// Derivation:
    ///   B_DMI_x = (2D/Ms) · ∂m_z/∂x
    ///   ∂m_z/∂x ≈ (m_z[i+1] - m_z[i-1]) / (2·dx)
    ///   ⇒ dmi_pf = D / (Ms · dx)
    pub fn dmi_prefactor(&self, dx: f64) -> f64 {
        if self.ms.abs() < 1e-20 { return 0.0; }
        self.d_dmi / (self.ms * dx)
    }

    /// Slonczewski SOT coefficients from charge current and substrate properties.
    ///
    /// Returns (τ_DL [T], τ_FL [T], σ [unit vector]):
    ///   τ_DL = (ℏ·θ_SH·|J_c|) / (2·e·Ms·t)   damping-like effective field
    ///   τ_FL = τ_DL · tau_fl_ratio             field-like effective field
    ///   σ    = ẑ × ĵ_c (unit)                  spin polarization direction
    ///
    /// Returns (0, 0, zero) if current or spin Hall angle vanishes, or if
    /// the current is purely along z (no in-plane spin polarization).
    pub fn sot_coefficients(&self, j_current: [f64; 3], thickness: f64) -> (f64, f64, [f64; 3]) {
        const HBAR: f64 = 1.054_571_817e-34;
        const E_CHARGE: f64 = 1.602_176_634e-19;
        let jmag = (j_current[0].powi(2) + j_current[1].powi(2) + j_current[2].powi(2)).sqrt();
        if jmag < 1e-20 || self.spin_hall_angle.abs() < 1e-20 || self.ms.abs() < 1e-20 {
            return (0.0, 0.0, [0.0, 0.0, 0.0]);
        }
        let tau_dl = (HBAR * self.spin_hall_angle * jmag) / (2.0 * E_CHARGE * self.ms * thickness);
        let tau_fl = tau_dl * self.tau_fl_ratio;
        // σ = ẑ × ĵ_c, normalized. If ĵ_c ∥ ẑ, the cross product vanishes.
        let sx: f64 = -j_current[1];
        let sy: f64 = j_current[0];
        let sz: f64 = 0.0;
        let smag = (sx.powi(2) + sy.powi(2) + sz.powi(2)).sqrt();
        let sigma = if smag > 1e-12 {
            [sx / smag, sy / smag, sz / smag]
        } else {
            [0.0, 0.0, 0.0]
        };
        (tau_dl, tau_fl, sigma)
    }

    /// Print the decomposition as "effective = bulk + interface/t" for each term.
    pub fn print_decomposition(
        &self,
        bulk: &BulkMaterial,
        substrate: &Substrate,
        geometry: &Geometry,
    ) {
        let t = geometry.thickness;
        eprintln!("Effective params (from {} bulk + {} substrate @ t={:.2}nm):",
                  bulk.name, substrate.name, t * 1e9);
        eprintln!("  Ms_eff    = {:.3e} = bulk {:.3e} + prox {:.3e} [A/m]",
                  self.ms, bulk.ms_bulk, substrate.delta_ms_proximity);
        eprintln!("  A_eff     = {:.3e} [J/m]",  self.a_ex);
        eprintln!("  K_u_eff   = {:.3e} = bulk {:.3e} + surf/t {:.3e} [J/m³]",
                  self.k_u, bulk.k_u_bulk, substrate.k_u_surface / t);
        eprintln!("  α_eff     = {:.5} = bulk {:.5} + pump {:.5}",
                  self.alpha, bulk.alpha_bulk, substrate.alpha_pumping);
        let b_bias_norm = (self.b_bias[0].powi(2) + self.b_bias[1].powi(2) + self.b_bias[2].powi(2)).sqrt();
        if b_bias_norm != 0.0 {
            eprintln!("  B_bias    = ({:.4}, {:.4}, {:.4}) T [active]",
                      self.b_bias[0], self.b_bias[1], self.b_bias[2]);
        }
        if self.d_dmi != 0.0 {
            eprintln!("  D_DMI_eff = {:.3e} J/m² [active]", self.d_dmi);
        }
        if self.spin_hall_angle != 0.0 {
            eprintln!("  θ_SH      = {:.4}, τ_FL/τ_DL = {:.3} [active]",
                      self.spin_hall_angle, self.tau_fl_ratio);
        }
    }

    /// Print the SOT drive as "effective fields + σ direction" when current applied.
    pub fn print_sot(&self, j_current: [f64; 3], thickness: f64) {
        let (tdl, tfl, sigma) = self.sot_coefficients(j_current, thickness);
        let jmag = (j_current[0].powi(2) + j_current[1].powi(2) + j_current[2].powi(2)).sqrt();
        if jmag == 0.0 && tdl == 0.0 { return; }
        eprintln!("SOT drive:");
        eprintln!("  J_c     = ({:.2e}, {:.2e}, {:.2e}) A/m², |J| = {:.2e}",
                  j_current[0], j_current[1], j_current[2], jmag);
        eprintln!("  σ       = ({:.3}, {:.3}, {:.3})", sigma[0], sigma[1], sigma[2]);
        eprintln!("  τ_DL    = {:.4e} T", tdl);
        eprintln!("  τ_FL    = {:.4e} T", tfl);
    }
}
