//! Photonic excitation — laser pulses as time-dependent effective B fields.
//!
//! Phase P1 scope: spatially-uniform Inverse-Faraday-Effect (IFE) pulses.
//! Each LaserPulse has a Gaussian temporal envelope; the total B_laser at any
//! simulation time is the sum of all active pulses' contributions.
//!
//! The field is passed to the GPU via a `b_laser: vec4<f32>` uniform that the
//! host rewrites before each Heun step (see `GpuSolver::step_n`). Spatial
//! variation (Gaussian beam profile, multi-spot pumping) comes in Phase P2.
//!
//! Physics convention:
//!   peak_field [T] is the peak IFE-equivalent B, directly as a Tesla value.
//!   Users must calibrate against experimental fluence separately — V, the
//!   Verdet-like coupling constant for FGT, is not in the literature (see
//!   docs/plan-photonic.md Appendix A).

#[derive(Clone, Debug)]
pub struct LaserPulse {
    /// Time of peak intensity [seconds, absolute sim time]
    pub t_center: f64,
    /// FWHM of temporal Gaussian envelope [seconds]
    pub duration_fwhm: f64,
    /// Peak IFE-equivalent field magnitude [Tesla]
    pub peak_field: f32,
    /// Direction of the induced B_IFE. Stored as unit vector (normalized on ingest).
    pub direction: [f32; 3],

    // ─── Phase P2 fields (stored for forward compat; unused in P1) ───
    /// In-plane (x, y) center of the focal spot [meters, from grid origin].
    /// Used in Phase P2 for Gaussian beam profile.
    pub spot_center: [f64; 2],
    /// 1-σ beam radius [meters]. If 0.0 or non-finite, treated as uniform (P1 behaviour).
    pub spot_sigma: f64,

    // ─── Phase P3 fields — thermal (M3TM) source term ───
    /// Absorbed-laser peak fluence [J/m²] (value at the peak of the Gaussian
    /// envelope, before reflectivity subtraction). Some(_) enables the M3TM
    /// source term for this pulse; None means coherent-only (P1-P2 behaviour).
    pub peak_fluence: Option<f64>,
    /// Surface reflectivity, 0..1. Only consulted when `peak_fluence.is_some()`.
    pub reflectivity: f32,
}

impl LaserPulse {
    /// Build a pulse with explicit validation / direction normalization.
    pub fn new(t_center: f64, duration_fwhm: f64, peak_field: f32, direction: [f32; 3]) -> Self {
        let [dx, dy, dz] = direction;
        let norm = (dx * dx + dy * dy + dz * dz).sqrt();
        let dir = if norm > 1e-12 {
            [dx / norm, dy / norm, dz / norm]
        } else {
            [0.0, 0.0, 1.0] // fallback: +z (circular polarization at normal incidence)
        };
        Self {
            t_center,
            duration_fwhm,
            peak_field,
            direction: dir,
            spot_center: [0.0, 0.0],
            spot_sigma: 0.0,
            peak_fluence: None,
            reflectivity: 0.0,
        }
    }

    /// Scalar temporal envelope at time `t` [s]. Peak = 1 at t_center.
    /// FWHM → σ conversion: σ = FWHM / (2·sqrt(2·ln(2))) ≈ FWHM / 2.3548.
    #[inline]
    pub fn envelope_at(&self, t: f64) -> f64 {
        let sigma = self.duration_fwhm / 2.354_820_045;
        if sigma < 1e-30 { return 0.0; }
        let dt = t - self.t_center;
        (-(dt * dt) / (2.0 * sigma * sigma)).exp()
    }
}

/// Collection of laser pulses. Sum contribution = time-dependent global B_laser(t).
#[derive(Clone, Debug, Default)]
pub struct PhotonicConfig {
    pub pulses: Vec<LaserPulse>,
    /// P3+: microscopic three-temperature + LLB parameters. `None` = LLG + IFE
    /// only (P1-P2 behaviour). `Some(_)` engages the M3TM source term; the
    /// magnetic dynamics path is selected by `thermal.enable_llb`.
    pub thermal: Option<ThermalConfig>,
}

/// Phase P3 — Microscopic three-temperature + LLB configuration.
///
/// Owns per-layer thermal-bath parameters plus the timestep / integrator
/// policy. When `enable_llb == false`, M3TM is advanced for observability
/// only; magnetic dynamics still run the projected-Heun LLG path and the
/// cell |m| stays 1. `enable_llb == true` engages the longitudinal-relaxation
/// torque in the LLB integrator (P3b).
#[derive(Clone, Debug)]
pub struct ThermalConfig {
    /// Ambient / starting temperature for all three baths [K].
    pub t_ambient: f32,
    /// Per-layer M3TM + LLB parameters (one entry per layer in the stack).
    pub per_layer: Vec<LayerThermalParams>,
    /// Timestep cap while inside a pulse's thermal window [s].
    /// Typical 1e-15 s (1 fs). Enforced by host-side sub-looping.
    pub thermal_dt_cap: f64,
    /// (before, after) window around each pulse peak during which
    /// `thermal_dt_cap` is enforced [s]. Default (0.5 ps, 10 ps).
    pub thermal_window: (f64, f64),
    /// Engage LLB longitudinal-relaxation path (P3b). Default false (P3a).
    pub enable_llb: bool,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            t_ambient: 300.0,
            per_layer: Vec::new(),
            thermal_dt_cap: 1.0e-15,
            thermal_window: (0.5e-12, 10.0e-12),
            enable_llb: false,
        }
    }
}

/// Per-layer thermal-bath + LLB-table parameters.
///
/// M3TM: Koopmans 2010 (Nat. Mater. 9, 259). R is precomputed on the host
/// from the microscopic parameters here via `r_koopmans_prefactor()`.
///
/// LLB tables (Phase F1, 2D extension): `m_e_table` and `chi_par_table` are
/// stored as flat row-major `[i_T][i_B]` arrays. `i_T = 0..llb_table_n − 1`
/// indexes the temperature axis on a uniform grid `T ∈ [0, 1.5·t_c]`;
/// `i_B = 0..llb_table_n_b − 1` indexes the longitudinal-field axis on a
/// uniform grid `B ∈ [0, b_max]`. The B = 0 column reproduces the previous
/// 1D table exactly. Built offline from a finite-field Brillouin / MFA
/// solver (see `material_thermal::brillouin_tables_spin_half_2d`).
#[derive(Clone, Debug)]
pub struct LayerThermalParams {
    // ─── M3TM microscopic parameters ────────────────────────────────
    /// Electron heat-capacity coefficient: C_e(T) = γ_e · T [J/(m³·K²)].
    pub gamma_e: f64,
    /// Phonon heat capacity [J/(m³·K)] (Debye / Dulong-Petit near ambient).
    pub c_p: f64,
    /// Electron-phonon coupling [W/(m³·K)].
    pub g_ep: f64,
    /// Phonon-substrate coupling [W/(m³·K)] — volumetric heat-sink rate into
    /// the substrate bath. Governs the phonon cooling term
    ///   dT_p/dt |_sub = −g_sub_phonon·(T_p − T_ambient) / c_p
    /// which closes the energy balance for magnetic thin films on heat-
    /// conducting substrates (sapphire, Si, etc.). Set to 0 for an isolated
    /// film. Representative scale: `c_p / τ_sub` with τ_sub ~ 100 ps for
    /// 20 nm metals on sapphire → ≈ 3·10¹⁶ W/(m³·K).
    pub g_sub_phonon: f64,
    /// Koopmans Elliott-Yafet scaling (dimensionless). ≈0.185 for Ni.
    pub a_sf: f64,
    /// Atomic moment in units of μ_B (dimensionless). ≈0.6 for Ni.
    pub mu_atom_bohr: f64,
    /// Atomic volume [m³] = 1 / (atomic number density).
    pub v_atom: f64,
    /// Debye temperature [K]. E_D = k_B · θ_D.
    pub theta_d: f64,

    // ─── LLB parameters ─────────────────────────────────────────────
    /// Curie temperature [K]. Same as bulk Tc in most presets.
    pub t_c: f64,
    /// Low-temperature Gilbert damping α₀ (matches LLG α when thermal off).
    pub alpha_0: f64,
    /// Number of rows on the temperature axis.
    pub llb_table_n: usize,
    /// Number of columns on the longitudinal-field axis (Phase F1).
    /// The B-axis spans `[0, b_max_t]`. Default 32.
    pub llb_table_n_b: usize,
    /// Upper bound of the B-axis [Tesla] (Phase F1). Default 10 T — covers
    /// the saturation regime for typical ferromagnets.
    pub b_max_t: f64,
    /// m_e(T_i, B_j) flat row-major [i_T * n_b + i_B], dimensionless.
    pub m_e_table: Vec<f32>,
    /// χ_∥(T_i, B_j) flat row-major same layout. [dimensionless; μ_B / (k_B·T_c)-normalized].
    pub chi_par_table: Vec<f32>,
    /// LLB longitudinal-relaxation time constant [s]. Effective τ_∥ at temperature T
    /// is `tau_long_base / α_∥(T)`, where `α_∥(T) = α_0 · 2T/(3 · T_c)`.
    /// P3b phenomenological form (exponential relaxation to `m_e(T_s)`).
    /// Default 0.3 fs for ferromagnets (gives ≈10 fs τ_∥ near T_c for α_0 = 0.04).
    pub tau_long_base: f64,
    /// Short human-readable provenance string.
    pub notes: &'static str,
}

impl LayerThermalParams {
    /// Boltzmann constant [J/K].
    pub const K_B: f64 = 1.380_649e-23;

    /// Koopmans R prefactor [1/s]:
    ///     R = 8 · a_sf · g_ep · k_B · T_c² · V_at / (μ_at_μB · (k_B·θ_D)²)
    /// with μ_at in units of μ_B (dimensionless). This is `R` in eq. (4) of
    /// Koopmans 2010; R · m · (T_p/T_c) · (1 − m·coth(m·T_c/T_e)) gives dm/dt.
    pub fn r_koopmans_prefactor(&self) -> f64 {
        if self.mu_atom_bohr.abs() < 1e-20 || self.theta_d.abs() < 1e-20 {
            return 0.0;
        }
        let e_d = Self::K_B * self.theta_d;
        8.0 * self.a_sf * self.g_ep * Self::K_B * self.t_c.powi(2) * self.v_atom
            / (self.mu_atom_bohr * e_d.powi(2))
    }

    /// Lookup m_e(T) at zero applied field. Convenience for legacy call
    /// sites and tests; equivalent to `sample_m_e_2d(t, 0.0)`.
    pub fn sample_m_e(&self, t: f64) -> f64 {
        self.sample_m_e_2d(t, 0.0)
    }

    /// Lookup χ_∥(T) at zero applied field.
    pub fn sample_chi_par(&self, t: f64) -> f64 {
        self.sample_chi_par_2d(t, 0.0)
    }

    /// Bilinear lookup of `m_e(T, B)` from the 2D table. `b` is in Tesla.
    /// Out-of-range arguments are clamped to nearest edge.
    pub fn sample_m_e_2d(&self, t: f64, b: f64) -> f64 {
        self.sample_table_2d(&self.m_e_table, t, b)
    }

    /// Bilinear lookup of `χ_∥(T, B)`.
    pub fn sample_chi_par_2d(&self, t: f64, b: f64) -> f64 {
        self.sample_table_2d(&self.chi_par_table, t, b)
    }

    fn sample_table_2d(&self, table: &[f32], t: f64, b: f64) -> f64 {
        let n_t = self.llb_table_n;
        let n_b = self.llb_table_n_b.max(1);
        if table.is_empty() || n_t == 0 {
            return 0.0;
        }
        let t_max = 1.5 * self.t_c;
        let b_max = self.b_max_t.max(1e-12);
        // T axis fractional index.
        let ut = if t <= 0.0 {
            0.0
        } else if t >= t_max {
            (n_t - 1) as f64
        } else {
            t / t_max * (n_t - 1) as f64
        };
        // B axis fractional index.
        let ub = if b <= 0.0 {
            0.0
        } else if b >= b_max {
            (n_b - 1) as f64
        } else {
            b.abs() / b_max * (n_b - 1) as f64
        };
        let it0 = ut.floor() as usize;
        let it1 = (it0 + 1).min(n_t - 1);
        let ib0 = ub.floor() as usize;
        let ib1 = (ib0 + 1).min(n_b - 1);
        let ft = ut - it0 as f64;
        let fb = ub - ib0 as f64;
        let v00 = table[it0 * n_b + ib0] as f64;
        let v01 = table[it0 * n_b + ib1] as f64;
        let v10 = table[it1 * n_b + ib0] as f64;
        let v11 = table[it1 * n_b + ib1] as f64;
        // Bilinear: first interp on B at each T row, then on T.
        let v0 = v00 * (1.0 - fb) + v01 * fb;
        let v1 = v10 * (1.0 - fb) + v11 * fb;
        v0 * (1.0 - ft) + v1 * ft
    }
}

impl ThermalConfig {
    pub fn print_summary(&self) {
        eprintln!(
            "Thermal  : T_ambient = {:.1} K, dt_cap = {:.2} fs, window = ({:.2}, {:.2}) ps, enable_llb = {}",
            self.t_ambient,
            self.thermal_dt_cap * 1e15,
            self.thermal_window.0 * 1e12,
            self.thermal_window.1 * 1e12,
            self.enable_llb,
        );
        for (i, p) in self.per_layer.iter().enumerate() {
            let r = p.r_koopmans_prefactor();
            eprintln!(
                "  layer[{i}] a_sf={:.3} g_ep={:.2e} Tc={:.0}K θ_D={:.0}K μ_at={:.2}μ_B α_0={:.4} R={:.3e}/s — {}",
                p.a_sf, p.g_ep, p.t_c, p.theta_d, p.mu_atom_bohr, p.alpha_0, r, p.notes,
            );
        }
    }
}

impl PhotonicConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.pulses.is_empty()
    }

    /// Total B_laser at sim time `t` [s] — vector sum over all pulses.
    /// In P1 this is spatially uniform (same B at every cell).
    pub fn field_at_time(&self, t: f64) -> [f32; 3] {
        let mut total = [0.0f32; 3];
        for p in &self.pulses {
            let amp = (p.peak_field as f64 * p.envelope_at(t)) as f32;
            total[0] += amp * p.direction[0];
            total[1] += amp * p.direction[1];
            total[2] += amp * p.direction[2];
        }
        total
    }

    /// Estimated latest-time any pulse is still non-negligible (> 1% of peak),
    /// useful for choosing simulation duration. Returns 0 if no pulses.
    pub fn latest_pulse_decay(&self) -> f64 {
        // ln(100) ≈ 4.6; 1% cutoff at |dt|/σ = sqrt(2·ln(100)) ≈ 3.03
        const CUTOFF: f64 = 3.03;
        self.pulses
            .iter()
            .map(|p| {
                let sigma = p.duration_fwhm / 2.354_820_045;
                p.t_center + CUTOFF * sigma
            })
            .fold(0.0f64, f64::max)
    }

    pub fn print_summary(&self) {
        if !self.pulses.is_empty() {
            eprintln!("Photonic drive: {} laser pulse(s)", self.pulses.len());
            for (i, p) in self.pulses.iter().enumerate() {
                let spatial = if p.spot_sigma > 0.0 {
                    format!(", spot = ({:.1}, {:.1}) nm, σ = {:.1} nm",
                            p.spot_center[0] * 1e9, p.spot_center[1] * 1e9, p.spot_sigma * 1e9)
                } else {
                    " [spatially uniform]".to_string()
                };
                let fluence = match p.peak_fluence {
                    Some(f) => format!(", F = {:.3} mJ/cm² R = {:.2}", f * 0.1, p.reflectivity),
                    None => String::new(),
                };
                eprintln!(
                    "  [{i}] t_center = {:.2} ps, FWHM = {:.1} fs, peak = {:.3} T, dir = ({:.2}, {:.2}, {:.2}){spatial}{fluence}",
                    p.t_center * 1e12,
                    p.duration_fwhm * 1e15,
                    p.peak_field,
                    p.direction[0], p.direction[1], p.direction[2],
                );
            }
        }
        if let Some(t) = &self.thermal {
            t.print_summary();
        }
    }
}

// ─── CLI parsing ─────────────────────────────────────────────────

/// Parse a pulse spec string like "t=100ps,fwhm=100fs,peak=0.5T,dir=z".
///
/// Accepted keys:
///   t        — pulse center time, with ps/fs/ns suffix (default ps)
///   fwhm     — temporal FWHM, with ps/fs/ns suffix (default fs)
///   peak     — peak B field in Tesla (T suffix optional)
///   dir      — "x", "y", "z", "-x", "-y", "-z", or "a,b,c" for arbitrary 3-vec
///              (wrapped in brackets or quotes if comma-separated in a shell)
///
/// Examples:
///   "t=100ps,fwhm=100fs,peak=0.5T,dir=z"    — 100-fs pulse at 100 ps, 0.5 T along +z
///   "t=0,fwhm=10fs,peak=1.0,dir=-z"         — σ⁻ helicity 10-fs pulse at t=0
pub fn parse_pulse_spec(spec: &str) -> Result<LaserPulse, String> {
    let mut t_center: Option<f64> = None;
    let mut fwhm: Option<f64> = None;
    let mut peak: Option<f32> = None;
    let mut dir: Option<[f32; 3]> = None;
    // P2 spatial keys
    let mut spot_x: f64 = 0.0;
    let mut spot_y: f64 = 0.0;
    let mut spot_sigma: f64 = 0.0;
    // P3 thermal keys (optional)
    let mut fluence_j_m2: Option<f64> = None;
    let mut reflectivity: f32 = 0.0;

    // Split on ',' but respect arbitrary 'dir=a,b,c' by a small trick:
    // parse keys one at a time, detecting if we're inside a 'dir=' value.
    let mut it = spec.split(',').peekable();
    while let Some(part) = it.next() {
        let p = part.trim();
        if p.is_empty() { continue; }
        let eq = p.find('=').ok_or_else(|| format!("Expected key=value, got '{p}'"))?;
        let key = &p[..eq];
        let mut value = p[eq + 1..].to_string();

        // Special handling for `dir=x,y,z` — value may extend across two commas
        if key == "dir" && !["x", "y", "z", "-x", "-y", "-z", "+x", "+y", "+z"]
            .contains(&value.as_str()) && !value.contains(' ')
        {
            // Might be "dir=a" with b,c in later segments
            // Already split by ',', so peek ahead for numeric-only next segments
            while let Some(next) = it.peek() {
                let n = next.trim();
                if n.chars().next().map_or(false, |c| c.is_ascii_digit() || c == '-' || c == '+' || c == '.') {
                    value.push(',');
                    value.push_str(n);
                    it.next();
                } else {
                    break;
                }
            }
        }

        match key {
            "t" | "t_center" => { t_center = Some(parse_time(&value)?); }
            "fwhm" | "duration" => { fwhm = Some(parse_time_fs_default(&value)?); }
            "peak" | "peak_field" | "amplitude" => {
                let trimmed = value.trim_end_matches('T').trim_end_matches('t');
                peak = Some(trimmed.parse().map_err(|e| format!("peak: {e}"))?);
            }
            "dir" | "direction" => { dir = Some(parse_direction(&value)?); }
            "x" | "spot_x" => { spot_x = parse_length_nm_default(&value)?; }
            "y" | "spot_y" => { spot_y = parse_length_nm_default(&value)?; }
            "sigma" | "spot" | "spot_sigma" | "waist" => {
                spot_sigma = parse_length_nm_default(&value)?;
            }
            "fluence" | "F" => {
                // Default unit: mJ/cm². Accept J/m² explicitly via suffix "J/m2" or "Jm2".
                fluence_j_m2 = Some(parse_fluence_mj_cm2_default(&value)?);
            }
            "R" | "reflectivity" => {
                reflectivity = value.parse().map_err(|e| format!("reflectivity: {e}"))?;
            }
            other => return Err(format!("Unknown pulse key: '{other}'")),
        }
    }

    let t_center = t_center.ok_or_else(|| "Missing 't='".to_string())?;
    let fwhm = fwhm.ok_or_else(|| "Missing 'fwhm='".to_string())?;
    let peak = peak.ok_or_else(|| "Missing 'peak='".to_string())?;
    let dir = dir.unwrap_or([0.0, 0.0, 1.0]);

    let mut pulse = LaserPulse::new(t_center, fwhm, peak, dir);
    pulse.spot_center = [spot_x, spot_y];
    pulse.spot_sigma = spot_sigma;
    pulse.peak_fluence = fluence_j_m2;
    pulse.reflectivity = reflectivity;
    Ok(pulse)
}

/// Parse a fluence value. Default unit: mJ/cm². Accepts "Jm2"/"J/m2" for SI.
/// Returns fluence in J/m².
fn parse_fluence_mj_cm2_default(s: &str) -> Result<f64, String> {
    let s = s.trim();
    if let Some(r) = s.strip_suffix("J/m2").or_else(|| s.strip_suffix("Jm2")) {
        let v: f64 = r.trim().parse().map_err(|e| format!("fluence '{s}': {e}"))?;
        Ok(v)
    } else {
        let r = s.strip_suffix("mJ/cm2").unwrap_or(s);
        let v: f64 = r.trim().parse().map_err(|e| format!("fluence '{s}': {e}"))?;
        // 1 mJ/cm² = 10 J/m²
        Ok(v * 10.0)
    }
}

/// Parse a length with nm/um/mm/m suffix. Default unit: nm.
fn parse_length_nm_default(s: &str) -> Result<f64, String> {
    let s = s.trim();
    let (num_str, scale) = if let Some(r) = s.strip_suffix("nm") {
        (r, 1e-9)
    } else if let Some(r) = s.strip_suffix("um") {
        (r, 1e-6)
    } else if let Some(r) = s.strip_suffix("µm") {
        (r, 1e-6)
    } else if let Some(r) = s.strip_suffix("mm") {
        (r, 1e-3)
    } else if let Some(r) = s.strip_suffix("m") {
        (r, 1.0)
    } else {
        (s, 1e-9) // default nm
    };
    let v: f64 = num_str.trim().parse().map_err(|e| format!("length '{s}': {e}"))?;
    Ok(v * scale)
}

/// Parse a time with ps/fs/ns suffix. Default unit: ps.
fn parse_time(s: &str) -> Result<f64, String> {
    let s = s.trim();
    let (num_str, scale) = if let Some(r) = s.strip_suffix("ns") {
        (r, 1e-9)
    } else if let Some(r) = s.strip_suffix("ps") {
        (r, 1e-12)
    } else if let Some(r) = s.strip_suffix("fs") {
        (r, 1e-15)
    } else if let Some(r) = s.strip_suffix("s") {
        (r, 1.0)
    } else {
        (s, 1e-12) // default ps
    };
    let v: f64 = num_str.trim().parse().map_err(|e| format!("time '{s}': {e}"))?;
    Ok(v * scale)
}

/// Same but default unit = fs (for FWHM / duration).
fn parse_time_fs_default(s: &str) -> Result<f64, String> {
    let s = s.trim();
    let (num_str, scale) = if let Some(r) = s.strip_suffix("ns") {
        (r, 1e-9)
    } else if let Some(r) = s.strip_suffix("ps") {
        (r, 1e-12)
    } else if let Some(r) = s.strip_suffix("fs") {
        (r, 1e-15)
    } else if let Some(r) = s.strip_suffix("s") {
        (r, 1.0)
    } else {
        (s, 1e-15) // default fs
    };
    let v: f64 = num_str.trim().parse().map_err(|e| format!("duration '{s}': {e}"))?;
    Ok(v * scale)
}

fn parse_direction(s: &str) -> Result<[f32; 3], String> {
    match s.trim().to_lowercase().as_str() {
        "x" | "+x" => Ok([1.0, 0.0, 0.0]),
        "-x" => Ok([-1.0, 0.0, 0.0]),
        "y" | "+y" => Ok([0.0, 1.0, 0.0]),
        "-y" => Ok([0.0, -1.0, 0.0]),
        "z" | "+z" => Ok([0.0, 0.0, 1.0]),
        "-z" => Ok([0.0, 0.0, -1.0]),
        other => {
            // Parse "a,b,c"
            let parts: Vec<&str> = other.split(',').map(str::trim).collect();
            if parts.len() != 3 {
                return Err(format!("direction '{other}': expected 'x'/'y'/'z' or 'a,b,c'"));
            }
            let x: f32 = parts[0].parse().map_err(|e| format!("dir x: {e}"))?;
            let y: f32 = parts[1].parse().map_err(|e| format!("dir y: {e}"))?;
            let z: f32 = parts[2].parse().map_err(|e| format!("dir z: {e}"))?;
            Ok([x, y, z])
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_peaks_at_one() {
        let p = LaserPulse::new(100e-12, 100e-15, 1.0, [0.0, 0.0, 1.0]);
        assert!((p.envelope_at(100e-12) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn envelope_half_at_fwhm() {
        let p = LaserPulse::new(0.0, 100e-15, 1.0, [0.0, 0.0, 1.0]);
        let v = p.envelope_at(50e-15);
        assert!((v - 0.5).abs() < 1e-9, "at FWHM/2 should be 0.5, got {v}");
    }

    #[test]
    fn field_at_time_scales_with_direction() {
        let cfg = PhotonicConfig {
            pulses: vec![LaserPulse::new(0.0, 100e-15, 2.0, [0.0, 0.0, 1.0])],
            ..Default::default()
        };
        let b = cfg.field_at_time(0.0);
        assert!((b[0]).abs() < 1e-6);
        assert!((b[1]).abs() < 1e-6);
        assert!((b[2] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn parse_full_spec() {
        let p = parse_pulse_spec("t=100ps,fwhm=100fs,peak=0.5T,dir=z").unwrap();
        assert!((p.t_center - 100e-12).abs() < 1e-15);
        assert!((p.duration_fwhm - 100e-15).abs() < 1e-18);
        assert!((p.peak_field - 0.5).abs() < 1e-6);
        assert_eq!(p.direction, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn parse_negative_direction() {
        let p = parse_pulse_spec("t=0,fwhm=50fs,peak=0.3,dir=-z").unwrap();
        assert_eq!(p.direction, [0.0, 0.0, -1.0]);
    }
}
