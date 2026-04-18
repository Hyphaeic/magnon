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
// The prefactor formulas below use the B-form derivations:
//
//     B_exch = (2A / Ms) · ∇²m         [T]
//     B_anis = (2K_u / Ms) · (m·û)·û    [T]
//     B_zee  = B_ext                   [T]
// ────────────────────────────────────────────────────────────────────

use crate::effective::EffectiveParams3D;
use crate::material::BulkMaterial;
use crate::photonic::PhotonicConfig;
use crate::substrate::Substrate;

/// A single layer in a magnetic stack.
///
/// Each layer carries its own bulk material, thickness, and anisotropy axis.
/// Heterostructures (e.g., YIG/FGT) are represented as a Stack of distinct
/// Layers.
#[derive(Clone, Debug)]
pub struct Layer {
    pub material: BulkMaterial,
    /// Layer thickness [m]
    pub thickness: f64,
    /// Anisotropy axis (unit vector, normalized on GPU upload)
    pub u_axis: [f64; 3],
}

/// Ordered stack of magnetic layers, bottom → top.
///
/// `layers[0]` is the bottom layer, in contact with the substrate.
/// `interlayer_a[i]` is the exchange-coupling constant [J/m] at the
/// interface between layers i and i+1. `layer_spacing[i]` is the effective
/// z-distance between the centers of layers i and i+1 [m].
///
/// In Phase M1 the interlayer fields are stored but not yet used by the
/// shader — each layer evolves independently. Phase M2 activates them.
#[derive(Clone, Debug)]
pub struct Stack {
    pub layers: Vec<Layer>,
    pub interlayer_a: Vec<f64>,
    pub layer_spacing: Vec<f64>,
}

impl Stack {
    /// Maximum number of layers supported by the GPU shader (array size cap).
    pub const MAX_LAYERS: usize = 4;

    /// Single-layer stack — reproduces pre-multilayer behavior.
    pub fn monolayer(material: BulkMaterial, thickness: f64, u_axis: [f64; 3]) -> Self {
        Self {
            layers: vec![Layer { material, thickness, u_axis }],
            interlayer_a: Vec::new(),
            layer_spacing: Vec::new(),
        }
    }

    pub fn nz(&self) -> u32 {
        self.layers.len() as u32
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.layers.is_empty() {
            return Err("Stack has no layers".to_string());
        }
        if self.layers.len() > Self::MAX_LAYERS {
            return Err(format!(
                "Stack has {} layers; MAX_LAYERS = {}",
                self.layers.len(), Self::MAX_LAYERS
            ));
        }
        let expected_interfaces = self.layers.len().saturating_sub(1);
        if self.interlayer_a.len() != expected_interfaces {
            return Err(format!(
                "interlayer_a has length {}, expected {}",
                self.interlayer_a.len(), expected_interfaces
            ));
        }
        if self.layer_spacing.len() != expected_interfaces {
            return Err(format!(
                "layer_spacing has length {}, expected {}",
                self.layer_spacing.len(), expected_interfaces
            ));
        }
        Ok(())
    }
}

/// In-plane geometry and grid resolution. Layer thicknesses and Nz live on
/// the Stack; this struct only carries what all layers share.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub nx: u32,
    pub ny: u32,
    /// In-plane cell size [m] — assumed isotropic (dx = dy)
    pub cell_size: f64,
}

impl Geometry {
    /// Default in-plane geometry matching the pre-multilayer default
    /// (cell size ≈ FGT exchange length).
    pub fn thin_2d(nx: u32, ny: u32) -> Self {
        Self { nx, ny, cell_size: 2.5e-9 }
    }
}

/// Full simulation configuration.
#[derive(Clone)]
pub struct SimConfig {
    /// Magnetic stack (≥ 1 layer)
    pub stack: Stack,
    /// Substrate below layer 0 (top of stack is always free / vacuum)
    pub substrate: Substrate,
    pub geometry: Geometry,

    /// Timestep [s]
    pub dt: f64,
    /// External magnetic field [T] (global, applies to all layers)
    pub b_ext: [f64; 3],
    /// Pitchfork stabilization coefficient [1/s] — dormant (see llg.wgsl)
    pub stab_coeff: f64,
    /// Charge current density [A/m²] through the substrate (SOT drive of layer 0)
    pub j_current: [f64; 3],

    /// Photonic excitation — list of laser pulses. Empty = no optical drive (baseline).
    pub photonic: PhotonicConfig,

    pub readback_interval: usize,
    pub total_steps: usize,
    /// In-plane probe index (default: center of grid)
    pub probe_idx: Option<u32>,
    /// Which layer to probe. Default: 0 (bottom)
    pub probe_layer: Option<u32>,
}

impl SimConfig {
    /// Default FGT single-layer (bulk) on vacuum — preserves pre-M1 behavior.
    pub fn fgt_default(nx: u32, ny: u32) -> Self {
        Self {
            stack: Stack::monolayer(
                BulkMaterial::fgt_bulk(),
                0.7e-9,
                [0.0, 0.0, 1.0],
            ),
            substrate: Substrate::vacuum(),
            geometry: Geometry::thin_2d(nx, ny),
            // dt must satisfy dt·γ·B_max << 1 for Heun stability.
            // fgt-bulk has B_anis ≈ 7.77 T → dt·γ·B ≈ 0.14 at dt=1e-13 (unstable).
            // At dt=1e-14 we have ≈ 0.014, safely in the stable regime.
            dt: 1.0e-14,
            b_ext: [0.0, 0.0, 0.0],
            stab_coeff: 1.0e11,
            j_current: [0.0, 0.0, 0.0],
            photonic: PhotonicConfig::default(),
            readback_interval: 100,
            total_steps: 10_000,
            probe_idx: None,
            probe_layer: None,
        }
    }

    pub fn effective(&self) -> EffectiveParams3D {
        EffectiveParams3D::from_parts(&self.stack, &self.substrate, &self.geometry)
    }

    pub fn nx(&self) -> u32 { self.geometry.nx }
    pub fn ny(&self) -> u32 { self.geometry.ny }
    pub fn nz(&self) -> u32 { self.stack.nz() }
    pub fn cell_size(&self) -> f64 { self.geometry.cell_size }
    pub fn cell_count(&self) -> u32 { self.nx() * self.ny() * self.nz() }

    pub fn print_summary(&self) {
        let eff = self.effective();
        eprintln!("=== Magnonic Clock Simulator ===");
        eprintln!("Stack: {} layer(s)", self.stack.layers.len());
        for (i, layer) in self.stack.layers.iter().enumerate() {
            eprintln!("  [{}] {} @ {:.2} nm, u_axis=({:.2},{:.2},{:.2})",
                      i, layer.material.name, layer.thickness * 1e9,
                      layer.u_axis[0], layer.u_axis[1], layer.u_axis[2]);
        }
        eprintln!("Substrate (bottom interface): {}", self.substrate.name);
        eprintln!("Geometry : {}×{}×{}, cell {:.2} nm",
                  self.nx(), self.ny(), self.nz(), self.cell_size() * 1e9);
        eprintln!("Domain   : {:.1} × {:.1} nm",
                  self.nx() as f64 * self.cell_size() * 1e9,
                  self.ny() as f64 * self.cell_size() * 1e9);
        eprintln!("Timestep : {:.2} ps", self.dt * 1e12);
        eff.print_decomposition(&self.stack, &self.substrate, &self.geometry);
        eprintln!("B_ext    : ({:.3}, {:.3}, {:.3}) T",
                  self.b_ext[0], self.b_ext[1], self.b_ext[2]);
        self.photonic.print_summary();
        eprintln!("================================");
    }
}
