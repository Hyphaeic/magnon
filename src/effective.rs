//! Per-layer effective parameters — the flat values the GPU actually sees.
//!
//! Composed from `(Stack, Substrate, Geometry)` via:
//!   - bulk material → each layer
//!   - substrate → applied to layer 0 only (bottom interface; top is free)
//!   - geometry → in-plane cell size dx for exchange prefactor
//!   - layer thickness → scales K_u_surface/t and SOT 1/(Ms·t)
//!   - interlayer exchange → stored per interface (Nz - 1 entries); used by the
//!     shader starting in Phase M2 (reserved in M1).

use crate::config::{Geometry, Stack};
use crate::substrate::Substrate;

/// Effective parameters for one layer in the stack.
#[derive(Clone, Debug)]
pub struct LayerEffective {
    /// Material name (for display / debugging)
    pub name: &'static str,
    pub ms: f64,
    pub a_ex: f64,
    pub k_u: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub d_dmi: f64,
    /// Unit anisotropy axis (normalization enforced when uploaded to GPU)
    pub u_axis: [f64; 3],
    /// Exchange bias [T] — only nonzero for layer 0 (bottom)
    pub b_bias: [f64; 3],
    /// SOT spin Hall angle — only nonzero for layer 0
    pub spin_hall_angle: f64,
    pub tau_fl_ratio: f64,
    /// Layer thickness [m], for SOT 1/(Ms·t) scaling
    pub thickness: f64,
}

/// Full effective-parameters record: per-layer fields + interlayer coupling.
#[derive(Clone, Debug)]
pub struct EffectiveParams3D {
    pub layers: Vec<LayerEffective>,
    /// Interlayer exchange A [J/m] at each interface (len = layers.len() - 1).
    /// Reserved in M1; activated in M2.
    pub interlayer_a: Vec<f64>,
    /// Effective spacing [m] between adjacent layer centers (len = layers.len() - 1).
    pub layer_spacing: Vec<f64>,
}

impl EffectiveParams3D {
    /// Build from a Stack + Substrate + Geometry.
    /// Substrate contributions are applied to layer 0 only.
    pub fn from_parts(stack: &Stack, substrate: &Substrate, _geometry: &Geometry) -> Self {
        let vacuum_sub = Substrate::vacuum();
        let layers = stack
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                let sub = if i == 0 { substrate } else { &vacuum_sub };
                let t = layer.thickness;
                LayerEffective {
                    name: layer.material.name,
                    ms: layer.material.ms_bulk + sub.delta_ms_proximity,
                    a_ex: layer.material.a_ex_bulk,
                    k_u: layer.material.k_u_bulk + sub.k_u_surface / t,
                    alpha: layer.material.alpha_bulk + sub.alpha_pumping,
                    gamma: layer.material.gamma,
                    d_dmi: sub.d_dmi_surface,
                    u_axis: layer.u_axis,
                    b_bias: sub.b_exchange_bias,
                    spin_hall_angle: sub.spin_hall_angle,
                    tau_fl_ratio: sub.tau_fl_ratio,
                    thickness: layer.thickness,
                }
            })
            .collect();

        Self {
            layers,
            interlayer_a: stack.interlayer_a.clone(),
            layer_spacing: stack.layer_spacing.clone(),
        }
    }

    /// In-plane exchange prefactor: 2·A / (Ms · dx²) [Tesla]
    pub fn exchange_prefactor(&self, layer_idx: usize, dx: f64) -> f64 {
        let l = &self.layers[layer_idx];
        if l.ms.abs() < 1e-20 { return 0.0; }
        2.0 * l.a_ex / (l.ms * dx * dx)
    }

    /// Anisotropy prefactor: 2·K_u / Ms [Tesla]
    pub fn anisotropy_prefactor(&self, layer_idx: usize) -> f64 {
        let l = &self.layers[layer_idx];
        if l.ms.abs() < 1e-20 { return 0.0; }
        2.0 * l.k_u / l.ms
    }

    /// DMI prefactor: D / (Ms · dx) [Tesla per unit neighbor difference]
    pub fn dmi_prefactor(&self, layer_idx: usize, dx: f64) -> f64 {
        let l = &self.layers[layer_idx];
        if l.ms.abs() < 1e-20 { return 0.0; }
        l.d_dmi / (l.ms * dx)
    }

    /// Slonczewski SOT coefficients for a given layer under an applied current.
    /// Non-bottom layers return zero unless the substrate has been broadened
    /// to inject spin current — not in M1.
    pub fn sot_coefficients(
        &self,
        layer_idx: usize,
        j_current: [f64; 3],
    ) -> (f64, f64, [f64; 3]) {
        const HBAR: f64 = 1.054_571_817e-34;
        const E_CHARGE: f64 = 1.602_176_634e-19;
        let l = &self.layers[layer_idx];
        let jmag =
            (j_current[0].powi(2) + j_current[1].powi(2) + j_current[2].powi(2)).sqrt();
        if jmag < 1e-20 || l.spin_hall_angle.abs() < 1e-20 || l.ms.abs() < 1e-20 {
            return (0.0, 0.0, [0.0, 0.0, 0.0]);
        }
        let tau_dl = (HBAR * l.spin_hall_angle * jmag)
            / (2.0 * E_CHARGE * l.ms * l.thickness);
        let tau_fl = tau_dl * l.tau_fl_ratio;
        let sx: f64 = -j_current[1];
        let sy: f64 = j_current[0];
        let smag = (sx.powi(2) + sy.powi(2)).sqrt();
        let sigma = if smag > 1e-12 {
            [sx / smag, sy / smag, 0.0]
        } else {
            [0.0, 0.0, 0.0]
        };
        (tau_dl, tau_fl, sigma)
    }

    /// Interlayer exchange prefactor for layer `k` reading the interface BELOW it
    /// (between layers k-1 and k):
    ///     2·A_inter[k-1] / (Ms_k · dz[k-1]²)
    ///
    /// For asymmetric heterostructures (Ms_{k-1} ≠ Ms_k), each layer sees its OWN
    /// Ms in the prefactor — the field contribution on layer k is
    /// B = -(1/Ms_k) · δE/δm_k.
    pub fn interlayer_prefactor_below(&self, layer_idx: usize) -> f64 {
        if layer_idx == 0 { return 0.0; }
        let interface_idx = layer_idx - 1;
        if interface_idx >= self.interlayer_a.len() { return 0.0; }
        let a_inter = self.interlayer_a[interface_idx];
        let dz = self.layer_spacing[interface_idx];
        let ms = self.layers[layer_idx].ms;
        if ms.abs() < 1e-20 || dz.abs() < 1e-20 { return 0.0; }
        2.0 * a_inter / (ms * dz * dz)
    }

    /// Interlayer exchange prefactor for layer `k` reading the interface ABOVE it
    /// (between layers k and k+1):
    ///     2·A_inter[k] / (Ms_k · dz[k]²)
    pub fn interlayer_prefactor_above(&self, layer_idx: usize) -> f64 {
        if layer_idx >= self.interlayer_a.len() { return 0.0; }
        let a_inter = self.interlayer_a[layer_idx];
        let dz = self.layer_spacing[layer_idx];
        let ms = self.layers[layer_idx].ms;
        if ms.abs() < 1e-20 || dz.abs() < 1e-20 { return 0.0; }
        2.0 * a_inter / (ms * dz * dz)
    }

    pub fn print_decomposition(&self, stack: &Stack, substrate: &Substrate, _geometry: &Geometry) {
        eprintln!("Effective params per layer ({} substrate at bottom):", substrate.name);
        for (i, l) in self.layers.iter().enumerate() {
            let bulk = &stack.layers[i].material;
            let is_bottom = i == 0;
            eprintln!("  Layer {} [{}] t={:.2}nm:", i, l.name, l.thickness * 1e9);
            eprintln!("    Ms    = {:.3e} A/m", l.ms);
            eprintln!("    A_ex  = {:.3e} J/m", l.a_ex);
            eprintln!("    K_u   = {:.3e} J/m³ (bulk {:.3e}{})",
                      l.k_u, bulk.k_u_bulk,
                      if is_bottom && substrate.k_u_surface != 0.0 {
                          format!(" + surf/t {:.3e}", substrate.k_u_surface / l.thickness)
                      } else {
                          String::new()
                      });
            eprintln!("    α     = {:.5}{}",
                      l.alpha,
                      if is_bottom && substrate.alpha_pumping != 0.0 {
                          format!(" (bulk {:.5} + pump {:.5})", bulk.alpha_bulk, substrate.alpha_pumping)
                      } else {
                          String::new()
                      });
            if l.d_dmi != 0.0 {
                eprintln!("    D_DMI = {:.3e} J/m² [active]", l.d_dmi);
            }
            let b_bias_norm = (l.b_bias[0].powi(2) + l.b_bias[1].powi(2) + l.b_bias[2].powi(2)).sqrt();
            if b_bias_norm != 0.0 {
                eprintln!("    B_bias= ({:.4}, {:.4}, {:.4}) T",
                          l.b_bias[0], l.b_bias[1], l.b_bias[2]);
            }
            if l.spin_hall_angle != 0.0 {
                eprintln!("    θ_SH  = {:.4}, τ_FL/τ_DL = {:.3}",
                          l.spin_hall_angle, l.tau_fl_ratio);
            }
        }
        if !self.interlayer_a.is_empty() {
            eprintln!("Interlayer coupling [active]:");
            for i in 0..self.interlayer_a.len() {
                let pf_k = self.interlayer_prefactor_above(i);
                let pf_k1 = self.interlayer_prefactor_below(i + 1);
                eprintln!("    interface {} ↔ {}: A = {:.3e} J/m, dz = {:.2} nm",
                          i, i + 1, self.interlayer_a[i], self.layer_spacing[i] * 1e9);
                eprintln!("       prefactor seen by layer {}: {:.3} T", i, pf_k);
                eprintln!("       prefactor seen by layer {}: {:.3} T", i + 1, pf_k1);
            }
        }
    }

    /// Print SOT drive summary (bottom layer) when current applied.
    pub fn print_sot(&self, j_current: [f64; 3]) {
        let (tdl, tfl, sigma) = self.sot_coefficients(0, j_current);
        let jmag = (j_current[0].powi(2) + j_current[1].powi(2) + j_current[2].powi(2)).sqrt();
        if jmag == 0.0 && tdl == 0.0 { return; }
        eprintln!("SOT drive (layer 0):");
        eprintln!("  J_c   = ({:.2e}, {:.2e}, {:.2e}) A/m², |J| = {:.2e}",
                  j_current[0], j_current[1], j_current[2], jmag);
        eprintln!("  σ     = ({:.3}, {:.3}, {:.3})", sigma[0], sigma[1], sigma[2]);
        eprintln!("  τ_DL  = {:.4e} T", tdl);
        eprintln!("  τ_FL  = {:.4e} T", tfl);
    }
}
