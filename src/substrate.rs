//! Substrate parameters — contributions from the substrate below the
//! ferromagnetic film. Interface quantities are SURFACE densities [J/m²];
//! they scale inversely with film thickness to produce effective volume
//! parameters. See docs/plan.md §2 for the combination rules.
//!
//! All fields are zero for `Substrate::vacuum()` — that preset reproduces
//! the pre-Phase-1 "isolated film" behavior.

#[derive(Clone, Debug)]
pub struct Substrate {
    pub name: &'static str,
    /// Interface anisotropy density [J/m²] — contributes to K_u_eff after /t
    pub k_u_surface: f64,
    /// Interfacial DMI density [J/m²] — used in Phase 3
    pub d_dmi_surface: f64,
    /// Spin-pumping contribution to Gilbert damping (additive)
    pub alpha_pumping: f64,
    /// Proximity-induced change in saturation magnetization [A/m]
    pub delta_ms_proximity: f64,
    /// Exchange bias field from AFM substrate [T] — used in Phase 2
    pub b_exchange_bias: [f64; 3],
    /// Spin Hall angle (dimensionless) — used in Phase 4
    pub spin_hall_angle: f64,
    /// Field-like to damping-like torque ratio — used in Phase 4
    pub tau_fl_ratio: f64,
    /// Citation / provenance
    pub notes: &'static str,
}

impl Substrate {
    /// No substrate — film in vacuum. Baseline that reproduces pre-Phase-1 behavior.
    pub fn vacuum() -> Self {
        Self {
            name: "vacuum",
            k_u_surface: 0.0,
            d_dmi_surface: 0.0,
            alpha_pumping: 0.0,
            delta_ms_proximity: 0.0,
            b_exchange_bias: [0.0, 0.0, 0.0],
            spin_hall_angle: 0.0,
            tau_fl_ratio: 0.0,
            notes: "No substrate. Isolated film. Baseline preset.",
        }
    }

    /// SiO₂ amorphous substrate — weak, disordered interface.
    pub fn sio2() -> Self {
        Self {
            name: "sio2",
            k_u_surface: 1.0e-5,
            d_dmi_surface: 0.0,
            alpha_pumping: 0.001,
            delta_ms_proximity: 0.0,
            b_exchange_bias: [0.0, 0.0, 0.0],
            spin_hall_angle: 0.0,
            tau_fl_ratio: 0.0,
            notes: "Amorphous SiO2. Weak, disordered interface. Common substrate for exfoliated 2D materials.",
        }
    }

    /// Hexagonal boron nitride — atomically flat vdW interface, minimal coupling.
    pub fn hbn() -> Self {
        Self {
            name: "hbn",
            k_u_surface: 2.0e-5,
            d_dmi_surface: 0.0,
            alpha_pumping: 0.0001,
            delta_ms_proximity: 0.0,
            b_exchange_bias: [0.0, 0.0, 0.0],
            spin_hall_angle: 0.0,
            tau_fl_ratio: 0.0,
            notes: "Hexagonal boron nitride. Clean vdW interface. Minimal magnetic coupling; ideal for isolating intrinsic physics.",
        }
    }

    /// Pt heavy-metal — strong interfacial DMI, large spin pumping, SOT via SHE.
    pub fn pt_heavy_metal() -> Self {
        Self {
            name: "pt",
            k_u_surface: 1.3e-3,
            d_dmi_surface: 1.0e-3,
            alpha_pumping: 0.006,
            delta_ms_proximity: 5.0e3,
            b_exchange_bias: [0.0, 0.0, 0.0],
            spin_hall_angle: 0.07,
            tau_fl_ratio: 0.15,
            notes: "Platinum heavy metal. Large interfacial DMI (~1 mJ/m²), strong spin pumping, canonical SOT source.",
        }
    }

    /// WTe₂ topological substrate — chiral interfacial DMI (Wu 2020).
    pub fn wte2() -> Self {
        Self {
            name: "wte2",
            k_u_surface: 0.0,
            d_dmi_surface: 1.0e-3,
            alpha_pumping: 0.002,
            delta_ms_proximity: 0.0,
            b_exchange_bias: [0.0, 0.0, 0.0],
            spin_hall_angle: 0.05,
            tau_fl_ratio: 0.3,
            notes: "WTe2. Topological Weyl semimetal. Strong chiral DMI with FGT (Wu 2020 Nature Comms 11, 3860).",
        }
    }

    /// YIG insulator — ultra-low-damping magnon transport medium.
    pub fn yig_insulator() -> Self {
        Self {
            name: "yig",
            k_u_surface: 0.0,
            d_dmi_surface: 0.0,
            alpha_pumping: 0.0005,
            delta_ms_proximity: 0.0,
            b_exchange_bias: [0.0, 0.0, 0.0],
            spin_hall_angle: 0.0,
            tau_fl_ratio: 0.0,
            notes: "Yttrium iron garnet. Insulator, ultra-low intrinsic damping. Magnon transport layer for hybrid FM/magnon systems.",
        }
    }

    /// IrMn antiferromagnet — exchange bias pinning (+x direction convention).
    pub fn irmn_afm() -> Self {
        Self {
            name: "irmn",
            k_u_surface: 5.0e-4,
            d_dmi_surface: 0.0,
            alpha_pumping: 0.005,
            delta_ms_proximity: 0.0,
            b_exchange_bias: [0.02, 0.0, 0.0],
            spin_hall_angle: 0.03,
            tau_fl_ratio: 0.2,
            notes: "IrMn antiferromagnet. Exchange bias ~20 mT along set direction (here +x). Used for MTJ reference layers.",
        }
    }

    /// Look up a substrate by command-line name.
    pub fn lookup(name: &str) -> Option<Self> {
        let n = name.to_lowercase();
        match n.as_str() {
            "vacuum" | "none" => Some(Self::vacuum()),
            "sio2" | "silicon-oxide" => Some(Self::sio2()),
            "hbn" | "boron-nitride" => Some(Self::hbn()),
            "pt" | "platinum" | "pt-heavy-metal" => Some(Self::pt_heavy_metal()),
            "wte2" => Some(Self::wte2()),
            "yig" | "yig-insulator" => Some(Self::yig_insulator()),
            "irmn" | "irmn-afm" | "iridium-manganese" => Some(Self::irmn_afm()),
            _ => None,
        }
    }

    /// Library of all available substrates.
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::vacuum(),
            Self::sio2(),
            Self::hbn(),
            Self::pt_heavy_metal(),
            Self::wte2(),
            Self::yig_insulator(),
            Self::irmn_afm(),
        ]
    }
}

impl std::fmt::Display for Substrate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Substrate[{}]:", self.name)?;
        writeln!(f, "  K_u_surface   = {:.3e} J/m²", self.k_u_surface)?;
        writeln!(f, "  D_DMI_surface = {:.3e} J/m²", self.d_dmi_surface)?;
        writeln!(f, "  alpha_pumping = {:.5}", self.alpha_pumping)?;
        writeln!(f, "  ΔMs_proximity = {:.3e} A/m", self.delta_ms_proximity)?;
        writeln!(f, "  B_bias        = ({:.4}, {:.4}, {:.4}) T",
                 self.b_exchange_bias[0], self.b_exchange_bias[1], self.b_exchange_bias[2])?;
        writeln!(f, "  θ_SH          = {:.4}", self.spin_hall_angle)?;
        writeln!(f, "  τ_FL/τ_DL     = {:.3}", self.tau_fl_ratio)?;
        writeln!(f, "  notes: {}", self.notes)
    }
}
