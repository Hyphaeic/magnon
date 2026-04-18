// ─── Unit convention ────────────────────────────────────────────────
//
// This simulator evolves the normalized magnetization `m = M/Ms` under
// an effective field expressed in TESLA (B convention), NOT in A/m (H
// convention). All field terms — exchange, anisotropy, Zeeman — are
// computed and summed as B_eff in Tesla, then fed to the LLG torque with
// γ in rad/(s·T).
//
// Standard references and tools (OOMMF, MuMax3, Ubermag) commonly list
// effective fields in H form (A/m). To convert:
//
//     B_eff [T] = μ₀ · H_eff [A/m]     (μ₀ = 4π × 10⁻⁷ T·m/A)
//
// Material parameters (A, Ms, K_u) are the underlying physics constants
// in SI units, so there is no ambiguity when providing those directly.
// The prefactor formulas below use the B-form derivations:
//
//     B_exch = (2A / Ms) · ∇²m         [T]   (not 2A/(μ₀·Ms))
//     B_anis = (2K_u / Ms) · (m·û)·û    [T]   (not 2K_u/(μ₀·Ms))
//     B_zee  = B_ext                   [T]   (user supplies Tesla)
//
// If you have an "H_exch coefficient" from an OOMMF-style document,
// multiply by μ₀ before using it here.
// ────────────────────────────────────────────────────────────────────

use crate::effective::EffectiveParams;
use crate::material::BulkMaterial;
use crate::substrate::Substrate;

/// Simulation geometry: film thickness, cell size, grid dimensions.
///
/// `thickness` is critical — it scales the surface-density substrate
/// contributions (K_u_surface, D_DMI_surface) to volume-density effective
/// parameters. A monolayer (~0.7 nm) amplifies interface effects ~10×
/// compared to a 10-layer flake (~7 nm).
#[derive(Clone, Debug)]
pub struct Geometry {
    /// Film thickness [m]
    pub thickness: f64,
    /// In-plane cell size [m] — assumed isotropic (dx = dy)
    pub cell_size: f64,
    pub nx: u32,
    pub ny: u32,
}

impl Geometry {
    /// Thin 2D monolayer-scale geometry, default cell size ~ FGT exchange length.
    pub fn thin_2d(nx: u32, ny: u32) -> Self {
        Self {
            thickness: 0.7e-9,
            cell_size: 2.5e-9,
            nx,
            ny,
        }
    }
}

/// Full simulation configuration.
#[derive(Clone)]
pub struct SimConfig {
    /// Bulk material (intrinsic, substrate-independent properties)
    pub bulk: BulkMaterial,
    /// Substrate (contributes interface + coupling effects)
    pub substrate: Substrate,
    /// Geometry (thickness, cell size, grid dimensions)
    pub geometry: Geometry,

    /// Timestep [s]
    pub dt: f64,
    /// External magnetic field [T]
    pub b_ext: [f64; 3],
    /// Uniaxial anisotropy axis (normalized before GPU upload)
    pub u_axis: [f64; 3],
    /// Pitchfork stabilization coefficient [1/s] — dormant (see llg.wgsl)
    pub stab_coeff: f64,
    /// Charge current density [A/m²], flowing in the substrate plane.
    /// σ = ẑ × ĵ_c sets the spin polarization direction for SOT (Phase 4).
    pub j_current: [f64; 3],

    pub readback_interval: usize,
    pub total_steps: usize,
    pub probe_idx: Option<u32>,
}

impl SimConfig {
    /// Default FGT: bulk single-crystal parameters (Leon-Brito 2016) on
    /// a vacuum substrate — a physically honest "isolated FGT monolayer"
    /// baseline. Previously this used Garland 2026 effective values, which
    /// silently baked in substrate effects; see docs/plan.md §0.
    /// For the legacy Garland-effective regime, use BulkMaterial::fgt_effective().
    pub fn fgt_default(nx: u32, ny: u32) -> Self {
        Self {
            bulk: BulkMaterial::fgt_bulk(),
            substrate: Substrate::vacuum(),
            geometry: Geometry::thin_2d(nx, ny),
            dt: 1.0e-13,
            b_ext: [0.0, 0.0, 0.0],
            u_axis: [0.0, 0.0, 1.0],
            stab_coeff: 1.0e11,
            j_current: [0.0, 0.0, 0.0],
            readback_interval: 100,
            total_steps: 10_000,
            probe_idx: None,
        }
    }

    /// Compute the effective parameters that the GPU sees.
    pub fn effective(&self) -> EffectiveParams {
        EffectiveParams::from_parts(&self.bulk, &self.substrate, &self.geometry)
    }

    pub fn nx(&self) -> u32 { self.geometry.nx }
    pub fn ny(&self) -> u32 { self.geometry.ny }
    pub fn cell_size(&self) -> f64 { self.geometry.cell_size }

    /// Convenience: exchange-field prefactor for the Laplacian stencil.
    pub fn exchange_prefactor(&self) -> f64 {
        self.effective().exchange_prefactor(self.geometry.cell_size)
    }

    /// Convenience: anisotropy-field prefactor.
    pub fn anisotropy_prefactor(&self) -> f64 {
        self.effective().anisotropy_prefactor()
    }

    pub fn print_summary(&self) {
        let eff = self.effective();
        eprintln!("=== Magnonic Clock Simulator ===");
        eprintln!("Material : {}", self.bulk.name);
        eprintln!("Substrate: {}", self.substrate.name);
        eprintln!("Geometry : {}×{}, cell {:.2} nm, thickness {:.2} nm",
                  self.geometry.nx, self.geometry.ny,
                  self.geometry.cell_size * 1e9,
                  self.geometry.thickness * 1e9);
        eprintln!("Domain   : {:.1} × {:.1} nm",
                  self.geometry.nx as f64 * self.geometry.cell_size * 1e9,
                  self.geometry.ny as f64 * self.geometry.cell_size * 1e9);
        eprintln!("Timestep : {:.2} ps", self.dt * 1e12);
        eff.print_decomposition(&self.bulk, &self.substrate, &self.geometry);
        eprintln!("Prefactors: B_exch={:.3} T, B_anis={:.3} T",
                  eff.exchange_prefactor(self.geometry.cell_size),
                  eff.anisotropy_prefactor());
        eprintln!("B_ext    : ({:.3}, {:.3}, {:.3}) T",
                  self.b_ext[0], self.b_ext[1], self.b_ext[2]);
        eprintln!("================================");
    }
}
