/// Material parameters for micromagnetic simulation (SI units).
#[derive(Clone)]
pub struct MaterialParams {
    /// Saturation magnetization [A/m]
    pub ms: f64,
    /// Exchange stiffness [J/m]
    pub a_ex: f64,
    /// Uniaxial anisotropy constant [J/m³]
    pub k_u: f64,
    /// Gilbert damping [dimensionless]
    pub alpha: f64,
    /// Gyromagnetic ratio [rad/(s·T)]
    pub gamma: f64,
}

impl MaterialParams {
    /// Fe₃GeTe₂ default parameters (Garland 2026 simulation set).
    pub fn fgt_default() -> Self {
        Self {
            ms: 3.76e5,
            a_ex: 1.4e-12,
            k_u: 4.0e5,
            alpha: 0.01,
            gamma: 1.7595e11,
        }
    }

    /// YIG baseline for comparison (well-characterized).
    pub fn yig() -> Self {
        Self {
            ms: 1.4e5,
            a_ex: 3.65e-12,
            k_u: -610.0, // cubic, weak
            alpha: 3.0e-5,
            gamma: 1.7595e11,
        }
    }
}

/// Full simulation configuration.
#[derive(Clone)]
pub struct SimConfig {
    pub nx: u32,
    pub ny: u32,
    /// Cell size [m]
    pub dx: f64,
    /// Timestep [s]
    pub dt: f64,
    pub material: MaterialParams,
    /// External magnetic field [T]
    pub b_ext: [f64; 3],
    /// Uniaxial anisotropy axis (unit vector)
    pub u_axis: [f64; 3],
    /// Pitchfork stabilization coefficient [1/s]
    pub stab_coeff: f64,
    /// Steps between observable readback
    pub readback_interval: usize,
    /// Total simulation steps
    pub total_steps: usize,
    /// Virtual probe cell index (MTJ readout point)
    pub probe_idx: Option<u32>,
}

impl SimConfig {
    /// Default FGT config for a given grid size.
    pub fn fgt_default(nx: u32, ny: u32) -> Self {
        let material = MaterialParams::fgt_default();
        Self {
            nx,
            ny,
            dx: 2.5e-9,      // exchange length of FGT
            dt: 1.0e-13,      // 0.1 ps — safe for explicit Heun
            material,
            b_ext: [0.0, 0.0, 0.0],
            u_axis: [0.0, 0.0, 1.0], // PMA: c-axis
            stab_coeff: 1.0e11,       // ~ gamma scale
            readback_interval: 100,
            total_steps: 10_000,
            probe_idx: None,
        }
    }

    /// Exchange field prefactor: 2A / (Ms · dx²) [Tesla]
    pub fn exchange_prefactor(&self) -> f64 {
        2.0 * self.material.a_ex / (self.material.ms * self.dx * self.dx)
    }

    /// Anisotropy field prefactor: 2K_u / Ms [Tesla]
    pub fn anisotropy_prefactor(&self) -> f64 {
        2.0 * self.material.k_u / self.material.ms
    }

    pub fn print_summary(&self) {
        let ex_pf = self.exchange_prefactor();
        let an_pf = self.anisotropy_prefactor();
        eprintln!("=== Magnonic Clock Simulator ===");
        eprintln!("Grid: {}x{} ({} cells)", self.nx, self.ny, self.nx * self.ny);
        eprintln!("Cell size: {:.1} nm", self.dx * 1e9);
        eprintln!("Domain: {:.1} nm × {:.1} nm",
            self.nx as f64 * self.dx * 1e9,
            self.ny as f64 * self.dx * 1e9);
        eprintln!("Timestep: {:.2} ps", self.dt * 1e12);
        eprintln!("Material: Ms={:.2e} A/m, A={:.2e} J/m, Ku={:.2e} J/m³, α={:.4}",
            self.material.ms, self.material.a_ex, self.material.k_u, self.material.alpha);
        eprintln!("Prefactors: B_exch={:.3} T, B_anis={:.3} T", ex_pf, an_pf);
        eprintln!("B_ext: ({:.3}, {:.3}, {:.3}) T", self.b_ext[0], self.b_ext[1], self.b_ext[2]);
        eprintln!("Stabilization: {:.2e} /s", self.stab_coeff);
        eprintln!("================================");
    }
}
