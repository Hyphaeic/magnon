//! Bulk material parameters — substrate-independent, intrinsic properties.
//!
//! Each `BulkMaterial` represents measured or theoretically computed values
//! from bulk single-crystal experiments where available, or from established
//! simulation conventions. Interface and substrate effects are NOT baked in
//! here — those live in `crate::substrate::Substrate`.
//!
//! See docs/plan.md §2 for the decomposition rules that combine bulk +
//! substrate + geometry into effective parameters.

#[derive(Clone, Debug)]
pub struct BulkMaterial {
    pub name: &'static str,
    /// Saturation magnetization [A/m]
    pub ms_bulk: f64,
    /// Exchange stiffness [J/m]
    pub a_ex_bulk: f64,
    /// Bulk uniaxial anisotropy constant [J/m³]
    pub k_u_bulk: f64,
    /// Bulk Gilbert damping (dimensionless)
    pub alpha_bulk: f64,
    /// Gyromagnetic ratio [rad/(s·T)]
    pub gamma: f64,
    /// Curie temperature [K]
    pub tc_bulk: f64,
    /// Citation / provenance
    pub notes: &'static str,
}

impl BulkMaterial {
    /// Fe₃GeTe₂ bulk — Leon-Brito et al. (2016) J. Appl. Phys. 120, 083903.
    /// Bulk single-crystal measurements at 5 K. These are the honest
    /// intrinsic values for the material, separate from substrate effects.
    pub fn fgt_bulk() -> Self {
        Self {
            name: "fgt-bulk",
            ms_bulk: 3.76e5,
            a_ex_bulk: 9.5e-12,
            k_u_bulk: 1.46e6,
            alpha_bulk: 0.001,
            gamma: 1.7595e11,
            tc_bulk: 220.0,
            notes: "Leon-Brito 2016 bulk single-crystal. alpha is theoretical lower bound from Fang 2022 (no direct FMR measurement in FGT).",
        }
    }

    /// Fe₃GeTe₂ effective values — Garland et al. (2026) Adv. Funct. Mater.
    /// These are the values that were in the original simulator config;
    /// retained as a separate preset for baseline regression comparison.
    /// NOT intrinsic — they fold substrate effects into an effective model.
    pub fn fgt_effective() -> Self {
        Self {
            name: "fgt-effective",
            ms_bulk: 3.76e5,
            a_ex_bulk: 1.4e-12,
            k_u_bulk: 4.0e5,
            alpha_bulk: 0.01,
            gamma: 1.7595e11,
            tc_bulk: 220.0,
            notes: "Garland 2026 simulation values. Effective, not intrinsic. Used as baseline for regression; differs from fgt_bulk by ~7x in A_ex and ~3x in K_u.",
        }
    }

    /// Fe₃GaTe₂ — above-room-temperature vdW ferromagnet (Tc ≈ 370 K).
    /// Parameter uncertainty ~50%; characterization evolving.
    pub fn fga_te2_bulk() -> Self {
        Self {
            name: "fga-te2-bulk",
            ms_bulk: 5.5e5,
            a_ex_bulk: 2.0e-12,
            k_u_bulk: 3.0e5,
            alpha_bulk: 0.005,
            gamma: 1.7595e11,
            tc_bulk: 370.0,
            notes: "Fe3GaTe2 — room-temperature vdW ferromagnet. Parameter uncertainty large; values from recent simulation literature.",
        }
    }

    /// CrI₃ monolayer — Ising-like 2D ferromagnet, strong PMA.
    pub fn cri3_bulk() -> Self {
        Self {
            name: "cri3-bulk",
            ms_bulk: 1.4e5,
            a_ex_bulk: 0.75e-12,
            k_u_bulk: 3.0e5,
            alpha_bulk: 0.001,
            gamma: 1.7595e11,
            tc_bulk: 45.0,
            notes: "CrI3 monolayer. Strong Ising character, low damping. Tc limits operation to cryogenic.",
        }
    }

    /// CoFeB — PMA workhorse for MTJs and MRAM.
    pub fn cofeb_bulk() -> Self {
        Self {
            name: "cofeb-bulk",
            ms_bulk: 1.2e6,
            a_ex_bulk: 1.5e-11,
            k_u_bulk: 4.5e5,
            alpha_bulk: 0.006,
            gamma: 1.7595e11,
            tc_bulk: 1100.0,
            notes: "CoFeB thin film. K_u strongly depends on annealing and MgO interface. Well-characterized.",
        }
    }

    /// YIG (yttrium iron garnet, Y₃Fe₅O₁₂) — ultra-low-damping ferrimagnetic insulator.
    /// Used here as a bulk layer material for hybrid heterostructures (e.g., YIG/FGT).
    /// Cubic anisotropy K₁ ≈ -610 J/m³ is negligible compared to exchange; set K_u = 0.
    pub fn yig_bulk() -> Self {
        Self {
            name: "yig-bulk",
            ms_bulk: 1.4e5,
            a_ex_bulk: 3.65e-12,
            k_u_bulk: 0.0,
            alpha_bulk: 3.0e-5,
            gamma: 1.7595e11,
            tc_bulk: 560.0,
            notes: "YIG (Y3Fe5O12). Ferrimagnet, ultra-low intrinsic damping. K1 (cubic, ~-610 J/m³) not represented in uniaxial model.",
        }
    }

    /// Permalloy (Ni₈₀Fe₂₀) — soft magnetic, zero anisotropy, magnonics reference.
    pub fn permalloy_bulk() -> Self {
        Self {
            name: "permalloy-bulk",
            ms_bulk: 8.0e5,
            a_ex_bulk: 1.3e-11,
            k_u_bulk: 0.0,
            alpha_bulk: 0.008,
            gamma: 1.7595e11,
            tc_bulk: 870.0,
            notes: "Ni80Fe20. Isotropic, soft. Standard magnonics/spintronics material.",
        }
    }

    /// Look up a material by command-line name (accepts common aliases).
    /// `--material fgt` now maps to the honest Leon-Brito bulk values.
    /// For the Garland 2026 effective preset, use `--material fgt-effective`.
    pub fn lookup(name: &str) -> Option<Self> {
        let n = name.to_lowercase();
        match n.as_str() {
            "fgt" | "fgt-bulk" => Some(Self::fgt_bulk()),
            "fgt-effective" | "fgt-garland" => Some(Self::fgt_effective()),
            "fga-te2" | "fga-te2-bulk" | "fe3gate2" => Some(Self::fga_te2_bulk()),
            "cri3" | "cri3-bulk" => Some(Self::cri3_bulk()),
            "cofeb" | "cofeb-bulk" => Some(Self::cofeb_bulk()),
            "yig" | "yig-bulk" => Some(Self::yig_bulk()),
            "permalloy" | "permalloy-bulk" | "py" => Some(Self::permalloy_bulk()),
            _ => None,
        }
    }

    /// Library of all available bulk materials.
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::fgt_bulk(),
            Self::fgt_effective(),
            Self::fga_te2_bulk(),
            Self::cri3_bulk(),
            Self::cofeb_bulk(),
            Self::yig_bulk(),
            Self::permalloy_bulk(),
        ]
    }
}

impl std::fmt::Display for BulkMaterial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BulkMaterial[{}]:", self.name)?;
        writeln!(f, "  Ms    = {:.3e} A/m",       self.ms_bulk)?;
        writeln!(f, "  A_ex  = {:.3e} J/m",       self.a_ex_bulk)?;
        writeln!(f, "  K_u   = {:.3e} J/m³",      self.k_u_bulk)?;
        writeln!(f, "  alpha = {:.5}",            self.alpha_bulk)?;
        writeln!(f, "  gamma = {:.3e} rad/(s·T)", self.gamma)?;
        writeln!(f, "  Tc    = {:.0} K",          self.tc_bulk)?;
        writeln!(f, "  notes: {}",                 self.notes)
    }
}
