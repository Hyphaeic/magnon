//! Clock-viability metrics extracted from a probe time series.
//!
//! The standard measurement protocol (see sweep.rs):
//!   1. Relax the system to equilibrium
//!   2. Apply a transverse pulse
//!   3. Record probe_mz during the free decay
//!   4. This module analyzes the recorded samples to extract:
//!      - Dominant precession frequency (FFT peak)
//!      - Amplitude decay time (envelope fit)
//!      - Q-factor = ω · τ_decay / 2
//!      - Final equilibrium value

use rustfft::{num_complex::Complex, FftPlanner};

#[derive(Debug, Clone, Copy)]
pub struct ClockMetrics {
    /// Dominant precession frequency [GHz], 0 if no clear peak
    pub freq_ghz: f64,
    /// Spectral width of the peak (FWHM) [GHz]
    pub freq_width_ghz: f64,
    /// Amplitude decay time [ns], infinity if no decay detected
    pub decay_time_ns: f64,
    /// Q-factor = ω · τ_decay / 2, infinity if no decay
    pub q_factor: f64,
    /// Mean of last 10% of samples (steady-state proxy)
    pub final_value: f64,
    /// Peak-to-peak amplitude in the second half of the record
    pub late_amplitude: f64,
    /// Sampling dt (for reference)
    pub dt_sample_ps: f64,
}

/// Analyze a probe time series to extract clock-viability metrics.
///
/// `samples` — probe observable (e.g. probe_mz or avg_mx) at uniform sampling intervals.
/// `dt_sample_s` — time between consecutive samples [seconds].
pub fn analyze_time_series(samples: &[f32], dt_sample_s: f64) -> ClockMetrics {
    if samples.len() < 4 {
        return empty_metrics(dt_sample_s);
    }

    // Remove DC bias
    let n = samples.len();
    let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let centered: Vec<f32> = samples.iter().map(|&x| (x as f64 - mean) as f32).collect();

    // ── FFT (pad/truncate to largest power of 2) ────────────────
    let mut fft_len = 1usize;
    while fft_len * 2 <= n { fft_len *= 2; }
    let (freq_hz, freq_width_hz) = fft_peak(&centered[..fft_len], dt_sample_s);

    // ── Envelope decay ──────────────────────────────────────────
    // Split into 4 windows, take peak |x| in each. If peaks decrease
    // monotonically, fit exp(-t/τ) to the log of the peaks via LSQ.
    let decay_time_s = estimate_decay_time(&centered, dt_sample_s);

    // ── Summary stats ───────────────────────────────────────────
    let tail_start = n * 9 / 10;
    let final_value = samples[tail_start..]
        .iter()
        .map(|&x| x as f64)
        .sum::<f64>()
        / (n - tail_start) as f64;

    let late_half = &samples[n / 2..];
    let late_max = late_half.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let late_min = late_half.iter().copied().fold(f32::INFINITY, f32::min);
    let late_amplitude = (late_max - late_min) as f64;

    let q_factor = if decay_time_s.is_finite() && freq_hz > 0.0 {
        std::f64::consts::PI * freq_hz * decay_time_s
    } else {
        f64::INFINITY
    };

    ClockMetrics {
        freq_ghz: freq_hz / 1e9,
        freq_width_ghz: freq_width_hz / 1e9,
        decay_time_ns: decay_time_s * 1e9,
        q_factor,
        final_value,
        late_amplitude,
        dt_sample_ps: dt_sample_s * 1e12,
    }
}

fn empty_metrics(dt_sample_s: f64) -> ClockMetrics {
    ClockMetrics {
        freq_ghz: 0.0,
        freq_width_ghz: 0.0,
        decay_time_ns: 0.0,
        q_factor: 0.0,
        final_value: 0.0,
        late_amplitude: 0.0,
        dt_sample_ps: dt_sample_s * 1e12,
    }
}

/// FFT peak frequency and approximate FWHM (in Hz).
fn fft_peak(samples: &[f32], dt_s: f64) -> (f64, f64) {
    let n = samples.len();
    if n < 4 { return (0.0, 0.0); }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<Complex<f32>> = samples.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buf);

    // Find peak in positive-frequency half (skip DC at k=0)
    let nyq = n / 2;
    let mags: Vec<f32> = (1..nyq).map(|k| buf[k].norm()).collect();
    if mags.is_empty() { return (0.0, 0.0); }
    let (peak_idx, &peak_mag) = mags
        .iter()
        .enumerate()
        .fold((0usize, &0.0f32), |acc, (i, m)| if m > acc.1 { (i, m) } else { acc });
    let peak_k = peak_idx + 1; // shift back since we skipped k=0
    let freq_hz = peak_k as f64 / (n as f64 * dt_s);

    // FWHM: find half-max points bracketing the peak
    let half_max = peak_mag * 0.5;
    let mut left = peak_idx;
    while left > 0 && mags[left] > half_max { left -= 1; }
    let mut right = peak_idx;
    while right + 1 < mags.len() && mags[right] > half_max { right += 1; }
    let width_k = (right - left) as f64;
    let width_hz = width_k / (n as f64 * dt_s);

    (freq_hz, width_hz)
}

/// Estimate exponential decay time τ via linear fit on log-envelope.
///
/// Splits the signal into K windows, takes max |x| in each, fits log(peak) = -t/τ + const.
/// Returns τ in seconds, or infinity if no decay is detected.
fn estimate_decay_time(samples: &[f32], dt_sample_s: f64) -> f64 {
    const K: usize = 8;
    let n = samples.len();
    if n < K * 2 { return f64::INFINITY; }

    let win = n / K;
    let mut times = Vec::with_capacity(K);
    let mut log_peaks = Vec::with_capacity(K);
    for i in 0..K {
        let start = i * win;
        let end = ((i + 1) * win).min(n);
        let peak = samples[start..end]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        if peak > 1e-6 {
            let t_center = (start + end) as f64 * 0.5 * dt_sample_s;
            times.push(t_center);
            log_peaks.push((peak as f64).ln());
        }
    }
    if times.len() < 3 { return f64::INFINITY; }

    // Linear LSQ: y = a·t + b, τ = -1/a
    let k = times.len() as f64;
    let t_mean: f64 = times.iter().sum::<f64>() / k;
    let y_mean: f64 = log_peaks.iter().sum::<f64>() / k;
    let num: f64 = times.iter().zip(log_peaks.iter())
        .map(|(t, y)| (t - t_mean) * (y - y_mean))
        .sum();
    let den: f64 = times.iter().map(|t| (t - t_mean).powi(2)).sum();
    if den < 1e-30 { return f64::INFINITY; }
    let slope = num / den;
    if slope >= -1e-9 { return f64::INFINITY; } // no (or positive) decay
    -1.0 / slope
}
