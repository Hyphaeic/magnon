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
        if self.pulses.is_empty() { return; }
        eprintln!("Photonic drive: {} laser pulse(s)", self.pulses.len());
        for (i, p) in self.pulses.iter().enumerate() {
            let spatial = if p.spot_sigma > 0.0 {
                format!(", spot = ({:.1}, {:.1}) nm, σ = {:.1} nm",
                        p.spot_center[0] * 1e9, p.spot_center[1] * 1e9, p.spot_sigma * 1e9)
            } else {
                " [spatially uniform]".to_string()
            };
            eprintln!(
                "  [{i}] t_center = {:.2} ps, FWHM = {:.1} fs, peak = {:.3} T, dir = ({:.2}, {:.2}, {:.2}){spatial}",
                p.t_center * 1e12,
                p.duration_fwhm * 1e15,
                p.peak_field,
                p.direction[0], p.direction[1], p.direction[2],
            );
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
    Ok(pulse)
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
