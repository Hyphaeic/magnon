# Photonic Driver Extension — Implementation Plan

**Document type:** Implementation plan (follows the pattern of `docs/plan.md` and `docs/plan-multilayer.md`)
**Author:** system-architect (from conversation with volition-billy)
**Date:** 2026-04-17
**Status:** Draft — awaiting approval to execute
**Parent project:** `project-magnonic-clock-sim`
**Related docs:**
- `docs/plan.md` — substrate extension (Phases 1-5 complete)
- `docs/plan-multilayer.md` — multilayer extension (Phases M1-M5 complete)
- `docs/viability-report.md` — §3.3 motivates optical readout; §5 lists FGT optical constants as poorly-characterized

---

## 0. Metadata & Lookback

### Motivation

The simulator has a partial phototransduction output side (`probe_mz` observable matches what a TR-MOKE experiment measures at a single point). It has **no input side** — the existing "transverse pulse" is an abstract transverse B field with no direct experimental correspondence. Real clock devices will be **driven by laser pulses**, so the simulator needs to model photon-magnetization interaction to answer:

- "Given a 100 fs, σ⁺ circularly polarized laser pulse at fluence F on the top YIG/FGT design from M5, what Q-factor does TR-MOKE actually measure?"
- "What fluence threshold triggers all-optical switching in this heterostructure?"
- "How does repetition rate interact with the electron-phonon cooling time — where's the thermal bottleneck?"

None of these questions are answerable today because the sim can't ingest a laser pulse as an input.

### Scope

This plan adds **photonic driving** to the simulator — coherent and thermal mechanisms by which a laser pulse drives magnetization. Readout (MOKE / BLS) is already covered by the existing `probe_mz` observable.

### Lookback keywords
`inverse faraday effect` · `IFE coupling constant V` · `all-optical switching` ·
`ultrafast demagnetization` · `three-temperature model` · `3TM electron-phonon-spin` ·
`pump-probe sequence` · `gaussian beam profile` · `femtosecond laser pulse` ·
`optical magnetic field` · `TR-MOKE signature simulation` · `photon impulse`

### What exists now (baseline)
- 2D LLG solver with full substrate + multilayer physics (plan.md Phases 1-5, plan-multilayer.md Phases M1-M5)
- `project-magnonic-clock-sim` binaries: `magnonic-sim`, `magnonic-dashboard`, `magnonic-sweep`
- Pulse excitation is currently modeled as `set_b_ext(Bx, 0, Bz)` for N steps — a square-wave transverse field, NOT a laser pulse
- Two experiments validated: monolayer clock viability and YIG/FGT hybrid advantage

### The gap this plan closes

| Capability | Now | After this plan |
|------------|-----|------------------|
| Coherent optical excitation (IFE) | ❌ | ✅ (P1-P2) |
| Gaussian beam spatial profile | ❌ | ✅ (P2) |
| Femtosecond temporal envelope | ❌ (square pulse only) | ✅ (P1+) |
| Multiple pulses (pump-probe) | ❌ | ✅ (P4) |
| Ultrafast demagnetization (3TM) | ❌ | ✅ (P3) |
| Thermally-mediated M dynamics | ❌ | ✅ (P3) |
| All-optical switching threshold studies | ❌ | ✅ (P3+) |

---

## 1. Requirements

### Functional

1. A `LaserPulse` type specifies: temporal peak, duration, equivalent peak B-field, direction, and (P2+) spatial profile. Simulations accept a `Vec<LaserPulse>` to support pump-probe sequences.
2. The pulse's effective field modifies `B_eff` at each cell at each time step, summed alongside exchange / anisotropy / Zeeman / bias / DMI.
3. When `pulses` is empty, dynamics are identical to Phase M5 (no regression).
4. A user can specify pulses via CLI flag (e.g., `--pulse "t=100ps,duration=100fs,peak=0.5T"`) in the same style as the existing `--bx` / `--jx` flags.
5. (P3) Ultrafast heating dynamics are modeled through a three-temperature system; Ms(T_s) and α(T_s) become time-varying when 3TM is active.
6. The sweep harness can iterate over pulse parameters (duration, peak field, polarization) as additional axes.

### Non-functional

1. **Performance:** a single-pulse simulation must not slow down the per-step cost by more than 10% vs the no-pulse baseline.
2. **Memory:** P1-P2 add no new storage buffers (pulse params fit in the uniform). P3 adds three per-cell buffers (T_e, T_p, T_s), tripling solver memory for the thermal path.
3. **Backward compat:** `magnonic-sim`, `magnonic-dashboard`, `magnonic-sweep` must work unchanged when no pulse is specified.
4. **Clarity:** each pulse contribution to B_eff must be inspectable from `print_summary`.

### Out of scope for this plan

- **Radiation pressure / momentum transfer** — the mechanical force from photon momentum is ~12 orders of magnitude weaker than IFE for magnetic effects; negligible.
- **Plasma wave dynamics** — hot-electron plasma effects beyond the 3TM are experimentally relevant for the first ~50 fs but involve physics beyond LLG (non-equilibrium electron distribution, band-structure-resolved dynamics).
- **Coherent-photon quantum coupling** — single-photon-level interactions with the magnet (cavity QED regime). Classical field treatment is sufficient for classical clock physics.
- **Photothermal coupling to acoustic phonons** (magnetoelastic effects beyond uniform temperature). Deferred.
- **MOKE / BLS detection simulation** — the probe_mz observable already provides the TR-MOKE-equivalent signal. Dedicated optical detection models are a separate project.

---

## 2. Architecture

### Physics model — three mechanisms

The plan addresses three distinct physical processes, chosen because they cleanly slot into the existing LLG framework:

#### (a) Inverse Faraday Effect (IFE) — coherent

A circularly polarized pulse with complex electric-field envelope E(t) generates an effective magnetic field:
```
B_IFE(r, t) = V · σ · I(r, t) · k̂
```
where:
- `V` is the Verdet-like IFE coupling constant [m/(V²·T)]
- `σ = ±1` is the helicity (right/left circular)
- `I(r, t)` is the instantaneous intensity [W/m²]
- `k̂` is the propagation direction

For a temporally Gaussian, spatially Gaussian pulse:
```
I(r, t) = I_peak · exp(-((t - t₀)/Δt)²) · exp(-((x-x₀)² + (y-y₀)²)/(2·σ_r²))
```

For our simulator's inputs, instead of asking users to compute V · σ · I_peak and plug in the resulting B-field, we expose `peak_field_T` directly as a Tesla value. Users calibrate against experimental fluence separately (see Appendix).

#### (b) Ultrafast heating / three-temperature model (3TM)

For intense pulses (>~0.1 mJ/cm²), the electrons absorb photon energy fast enough to drive the electron temperature T_e up by hundreds of Kelvin on ~100 fs timescales. This heat equilibrates with the phonon system and with the spin system:

```
C_e(T_e)·dT_e/dt = −g_ep·(T_e − T_p) − g_es·(T_e − T_s) + P_laser(t)
C_p·dT_p/dt       = g_ep·(T_e − T_p)
C_s·dT_s/dt       = g_es·(T_e − T_s)
```

where:
- `C_e, C_p, C_s` are heat capacities per unit volume
- `g_ep, g_es` are coupling rates [J/(m³·K·s)]
- `P_laser(t) = (1−R)·F/Δt · envelope(t)` is absorbed laser power density
- `R` is sample reflectivity, `F` is fluence, `Δt` pulse duration

The spin temperature modifies the magnetization amplitude (the layer's "Ms" effectively drops during heating):
```
Ms_eff(T_s) = Ms_bulk · (1 − (T_s/T_c)^β)     (Bloch-style law, β ≈ 1.5-2.5)
```

And optionally damping:
```
α(T_s) ≈ α_0 · (T_s / T_equilibrium)^γ         (weaker, material-dependent)
```

#### (c) Photo-spin-transfer (deferred to P5 or later)

Polarized-light-induced spin transfer is physically distinct from IFE but often experimentally entangled. For simplicity this plan models it via an effective IFE contribution with material-specific V. Dedicated modeling deferred.

### Data model

```rust
// src/photonic.rs (new module)

/// A single laser pulse — Gaussian temporal and (in P2+) spatial profile.
#[derive(Clone, Debug)]
pub struct LaserPulse {
    /// Time of peak intensity [seconds]
    pub t_center: f64,
    /// FWHM of temporal Gaussian envelope [seconds]
    pub duration_fwhm: f64,
    /// Peak IFE-equivalent field amplitude [T]
    pub peak_field: f32,
    /// Direction of the induced B_IFE (usually [0, 0, 1] for normal incidence)
    pub direction: [f32; 3],
    /// P2+: center of Gaussian focal spot in (x, y) [meters from grid origin]
    pub spot_center: [f64; 2],
    /// P2+: 1-σ spot radius [meters]. If infinite or zero, uniform illumination.
    pub spot_sigma: f64,
}

/// Full photonic configuration.
#[derive(Clone, Debug, Default)]
pub struct PhotonicConfig {
    /// List of pulses — supports pump-probe sequences
    pub pulses: Vec<LaserPulse>,
    /// P3+: three-temperature model parameters
    pub three_temp: Option<ThreeTempConfig>,
}

/// P3+: thermal-optical coupling parameters
#[derive(Clone, Debug)]
pub struct ThreeTempConfig {
    pub c_e: f64,         // electron heat capacity [J/(m³·K²)] (linear-in-T)
    pub c_p: f64,         // phonon heat capacity [J/(m³·K)]
    pub c_s: f64,         // spin heat capacity [J/(m³·K)]
    pub g_ep: f64,        // e-p coupling [W/(m³·K)]
    pub g_es: f64,        // e-s coupling [W/(m³·K)]
    pub reflectivity: f32,
    pub tc: f64,          // Curie temperature [K], for Ms(T_s)
    pub bloch_exponent: f64, // β in Ms(T) ∝ (1-(T/Tc)^β)
}
```

### GPU side

**P1 (uniform time-dependent):**
- Host computes B_laser(t) each step by summing over pulse envelopes at current sim time
- Writes to a new `b_laser: vec4<f32>` field in GpuParams (16 bytes, at the end of the struct)
- Shader sums `params.b_laser.xyz` into B_eff

**P2 (spatial profile):**
- Pulse parameters arrive as uniform arrays (up to MAX_PULSES = 4):
  - `pulse_amplitudes: vec4<f32>` (one amplitude per pulse, packed)
  - `pulse_spot_centers: array<vec4<f32>, 4>` (x, y, σ_r, unused)
  - `pulse_directions: array<vec4<f32>, 4>`
- Shader computes, per cell: `B_laser(cell) = Σ_p direction_p · amplitude_p(t) · exp(-r_cell_to_center²/(2σ_p²))`
- Host recomputes `pulse_amplitudes` each step from the temporal envelope

**P3 (thermal):**
- New per-cell buffers: `temp_electron`, `temp_phonon`, `temp_spin` (each `Nx·Ny·Nz·1` floats)
- New compute kernel `advance_three_temp` runs before the field/torque kernel each step
- LLG torque kernel reads `temp_spin[idx]` and computes `Ms_eff(T_s)`, `α_eff(T_s)` on-the-fly
- Adds one new set of buffers + one new pipeline; roughly doubles per-step compute cost when active

### CLI

New flags (follow the `--bx` / `--jx` style):

```
# Single pulse — shorthand
--pulse "t=100ps,fwhm=100fs,peak=0.5T,dir=z"

# Multiple pulses — repeat the flag
--pulse "t=100ps,fwhm=100fs,peak=0.5T,dir=z" \
--pulse "t=200ps,fwhm=100fs,peak=0.5T,dir=z"

# Spatial (P2)
--pulse "t=100ps,fwhm=100fs,peak=0.5T,dir=z,x=0,y=0,sigma=500nm"

# 3TM (P3)
--enable-3tm --tc 220 --c_e 2e2 --c_p 3e6 --g_ep 5e17 --fluence 0.5
```

---

## 3. Implementation Phases

### Phase P1 — Uniform IFE pulse driver (2-3 days)

**Goal:** Support time-dependent, spatially-uniform IFE excitation. A pulse looks like a transient B_ext whose magnitude follows a Gaussian envelope.

**Steps:**
1. Create `src/photonic.rs` with `LaserPulse`, `PhotonicConfig`
2. Add `pub photonic: PhotonicConfig` field to `SimConfig`
3. Add method `PhotonicConfig::field_at_time(t) -> [f32; 3]` that sums Gaussian contributions
4. Extend `GpuParams` with 16-byte `b_laser: [f32; 4]` at end of struct
5. Match in WGSL `Params` struct
6. Sum `params.b_laser.xyz` into `b_eff` in both phase kernels
7. In `GpuSolver::step_n`, before each step, compute `B_laser(t = step·dt)` and write to uniform
8. CLI parser in `main.rs`: `--pulse "key=val,key=val"`
9. Print-summary shows the pulse list

**Acceptance criteria:**
- With no `--pulse`, dynamics identical to Phase M5
- With a single pulse at FMR-matching frequency/duration, resonant excitation visible in avg_mx
- Pulse envelope matches expected Gaussian shape (verifiable via probe time series)

### Phase P2 — Spatial Gaussian beam profile (3-5 days)

**Goal:** Laser pulse has a finite focal spot, not uniform illumination.

**Steps:**
1. Add spot fields to `LaserPulse` (already present in data model; activate them)
2. Extend `GpuParams` with per-pulse uniforms:
   - `pulse_amplitudes: [f32; 4]` — current time's amplitude (one per pulse, re-computed each step)
   - `pulse_spot_centers: [[f32; 4]; 4]` — (x, y, σ_r, _pad)
   - `pulse_directions: [[f32; 4]; 4]`
   - `pulse_count: u32` — how many pulses are active
3. In WGSL, add `fn laser_field_at(ix, iy)` that iterates pulses 0..pulse_count and sums Gaussian-weighted contributions
4. Replace the uniform `b_laser` path with the per-cell computation
5. Host recomputes `pulse_amplitudes` at each step from temporal envelope

**Acceptance criteria:**
- A pulse with spot_sigma = 100 nm on a 500-nm-wide grid produces avg_mx response confined near the spot center
- Excitation amplitude falls off as exp(-r²/2σ²) as expected
- Two spatially-separated pulses at different times produce independent local responses

### Phase P3 — Three-temperature thermal model (2 weeks)

**Goal:** Support ultrafast-demagnetization physics. Intense pulses heat electrons → spins → cause transient M collapse.

**Steps:**
1. Add `ThreeTempConfig` to photonic module
2. Add three new per-cell storage buffers to `GpuSolver`: `temp_electron`, `temp_phonon`, `temp_spin`
3. Initialize all three to ambient temperature (default 300 K)
4. New WGSL kernel `advance_three_temp`:
   - Reads laser power density P_laser(r, t) (same computation as pulse field)
   - Reads current T_e, T_p, T_s
   - Advances 3TM equations using Heun's method (re-use the integrator pattern)
   - Writes new T_e, T_p, T_s
5. Modify the existing field/torque kernel to:
   - Read T_s from the temp_spin buffer
   - Compute `Ms_eff(T_s)` on-the-fly (Bloch law from ThreeTempConfig)
   - Use this in the anisotropy and exchange prefactors (recomputed from EffectiveParams3D at each step? or cell-by-cell correction?)
6. CLI: `--enable-3tm --tc 220 --fluence 0.5 --c_e ... --c_p ... --c_s ...`
7. Output: per-step `max(T_e)`, `max(T_s)`, `min(Ms/Ms_0)` to CSV

**Design consideration:** Rather than recomputing all prefactors from Ms(T_s), we can store a per-cell "temperature factor" `f_T = Ms_eff(T_s) / Ms(0)` and multiply existing prefactors by it at runtime. Simpler but less flexible.

**Acceptance criteria:**
- Without pulse, T_e = T_p = T_s = 300 K remain constant, M unchanged (no regression)
- With sub-ps pulse of 1 mJ/cm², peak T_e > 1000 K, then relaxes; T_s peaks ~1 ps later, M recovers on ~10 ps timescale
- All-optical-switching threshold study: scan fluence, identify value at which M flips sign after pulse
- |m|=1 preservation in LLG remains valid (now normalized within each cell's local Ms_eff)

### Phase P4 — Pump-probe sequencer (3 days)

**Goal:** Orchestrated multi-pulse protocols. Pump-probe is the canonical ultrafast experiment.

**Steps:**
1. Ensure `PhotonicConfig.pulses: Vec<LaserPulse>` supports multiple
2. CLI: repeated `--pulse` args build the list
3. Add protocol templates to the sweep harness:
   - `--pump-probe-mode` — automatically generates (pump, probe) pulse pairs
   - `--pump-probe-delay-range START END STEPS` — iterates probe delays
4. Sweep CSV: add columns `pulse_count`, `first_pulse_t_ps`, `total_fluence`

**Acceptance criteria:**
- Pump-probe sweep produces Q-factor vs pump-probe delay curves
- Two-pulse coherent control experiments replicable (add/subtract pulses to enhance/suppress specific modes)
- TR-MOKE trace reconstruction: probe at fixed delay, pump varies in intensity

### Phase P5 — First optical experiment (1-2 weeks)

**Goal:** Exercise the photonic capability via a new HiR experiment entity.

**Steps:**
1. Create `experiment-optical-clock-excitation` in `branches/hir/experiments/`
2. Charter: hypothesis that the YIG/FGT top design from M5 achieves Q ≈ experiment-measurable-equivalent with fs laser excitation
3. Registry entity, workspace scope extension, run.sh
4. Notebook: pump-probe Q-factor measurement, fluence threshold study, thermal-vs-coherent regime identification
5. Promote winning figure
6. ADR-003 if a design lesson emerges (likely: "IFE V constant for FGT is unknown, require experimental calibration")

**Acceptance criteria:**
- Q-factor measured via realistic fs pulse excitation
- Thermal vs coherent regime boundary identified (fluence)
- Experimental calibration of V-parameter proposed (even if not measured)

---

## 4. Validation Strategy

### Per-phase regression tests (cumulative)

| Test | Phase | Verifies |
|---|---|---|
| No-pulse sim identical to M5 default | P1 | No regression |
| Gaussian envelope recovers correctly in probe_mz | P1 | Temporal envelope correctness |
| Peak field matches specified peak_field_T | P1 | Amplitude calibration |
| Spatial Gaussian falls off per r² from spot center | P2 | Profile correctness |
| Three-temperature system preserves total energy | P3 | Energy conservation |
| Ultrafast demag: M drops then recovers as expected | P3 | Thermal dynamics |
| Pump-probe fluence threshold reproducible | P4 | Reproducibility |
| Analytic cross-check: single-spin response to on-resonance pulse matches perturbation theory | P1+ | Core LLG + pulse physics |

### Canonical Benchmark (from literature)

**Beaurepaire 1996** observed a 40% Ms drop at ~500 fs in Ni at 7 mJ/cm² fluence. After P3, the simulator should reproduce this (order of magnitude, given FGT material differences).

---

## 5. References

### Plan docs
- `docs/plan.md` — substrate extension (Phases 1-5 complete)
- `docs/plan-multilayer.md` — multilayer extension (Phases M1-M5 complete)
- `docs/viability-report.md` — §3.3 mentioned optical readout as motivation; §5 listed optical parameters as uncharacterized for FGT

### IFE / All-optical literature
- Pitaevskii (1961), "Electric forces in a transparent dispersive medium," Sov. Phys. JETP 12, 1008 — original IFE theory
- van der Ziel, Pershan, Malmstrom (1965), "Optically-induced magnetization resulting from the inverse Faraday effect," Phys. Rev. Lett. 15 — experimental demonstration in diamagnetic glass
- Stanciu et al. (2007), "All-optical magnetic recording with circularly polarized light," Phys. Rev. Lett. 99 — single-pulse AO switching in GdFeCo
- Kimel et al. (2005), "Ultrafast non-thermal control of magnetization by instantaneous photomagnetic pulses," Nature 435 — IFE dynamics in DyFeO₃
- Mangin et al. (2014), "Engineered materials for all-optical helicity-dependent magnetic switching," Nat. Mater. 13 — materials engineering for AOS

### Ultrafast demag / 3TM literature
- Beaurepaire et al. (1996), "Ultrafast spin dynamics in ferromagnetic nickel," Phys. Rev. Lett. 76, 4250 — original 500-fs demag observation
- Koopmans et al. (2010), "Explaining the paradoxical diversity of ultrafast laser-induced demagnetization," Nat. Mater. 9 — microscopic three-temperature extension (M3TM)

### Reference implementations
- ubermag / magnum — have phenomenological 3TM coupling to micromagnetics
- MuMax3 — does not natively support IFE or 3TM; typically added via external scripts

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| IFE coupling constant V for FGT unknown | High | Medium | Expose peak_field_T directly; document the experimental calibration requirement |
| Shader performance drops > 10% from spatial profile computation | Medium | Low | Benchmark P2; if bad, fall back to per-cell pre-computed buffer |
| 3TM adds numerical instability | Medium | Medium | Start with simple Heun on 3TM + LLG independently; cross-validate with single-cell ODE solver |
| Ms(T_s) changes violate \|m\|=1 preservation | Medium | Medium | LLG keeps \|m\|=1 always; the Ms scaling only affects anisotropy/exchange prefactors. Verify rigorously |
| Temperature gradients across cells cause numerical artifacts at interfaces | Low | Low | 3TM has no spatial diffusion term in this model (we treat each cell independently). If needed later, add spatial heat diffusion in Phase P6 |
| Pulse duration < dt | Low | High (instability) | Add warning if pulse_fwhm < 5·dt; suggest smaller dt or coarser pulse |
| Very short pulses miss temporal resolution | Medium | Medium | Document that fs-scale simulations require dt ≤ 1e-15 s |

---

## 7. Execution Protocol

Same pattern as plan.md and plan-multilayer.md:

- Work one phase at a time. Commit at each boundary.
- After each phase, run the full regression suite.
- Document design decisions inline. ADRs for non-obvious trade-offs (especially around V-constant calibration and the Ms(T_s) scaling choice).
- Update this plan as execution proceeds. Add "Implementation notes" subsections.
- **Phase P3 is the riskiest** — both in code complexity (3TM + coupling) and physics (experimental validation difficulty). Consider running P4 before P3 if a pump-probe demonstration is the pressing need.

### Suggested execution order

If the goal is **fastest path to realistic optical excitation:**
P1 → P2 → P4 → (P5 start, coherent-regime focus) → P3 (thermal validation)

If the goal is **all-optical switching study:**
P1 → P3 → P2 → P4 → P5

Both paths reach the same destination; order reflects which science question bites first.

---

## 8. Appendix

### A. IFE coupling constant V — calibration strategy

The relation between laser fluence and peak_field_T requires knowing V. For FGT this isn't in the literature. Three ways to handle this:

1. **Parametric sweep (recommended for first experiments):** treat peak_field_T as a sweep axis. Ask "what B_IFE values produce interesting dynamics?" rather than "what fluence." Report results as functions of peak_field_T.

2. **Literature surrogate:** adopt V_typical ≈ 10⁻²⁰ s²/(kg·m) from ferromagnetic metal averages. Document as a working assumption. Publish results with explicit disclaimer.

3. **Experimental collaboration:** partner with a lab doing TR-MOKE on FGT. Measure V directly. Not blockable from sim progress but necessary before quantitative predictions.

### B. Pulse-to-simulation mapping

Typical experimental pulse: 100 fs FWHM, 500 nm wavelength, 1 mJ/cm² fluence, 5 μm spot, helicity σ = +1. Translates to:

```
peak_intensity = fluence / duration_fwhm × sqrt(4·ln(2)/π) ≈ 9.4 GW/cm²
peak_field_T ≈ V · σ · peak_intensity
           ≈ 10⁻²⁰ · 1 · 9.4×10¹³ ≈ 1 T     (order of magnitude, V-limited)
```

Tunable parameters:
- Fluence: 0.1 → 10 mJ/cm² (coherent → thermal regime transition)
- Duration: 10 fs → 10 ps (sub-exchange → approaching magnonic timescales)
- Spot: 100 nm → 100 μm (single-mode → global illumination)

### C. Memory / computational estimate

Per cell, at various phases:

| Phase | Extra GPU memory per cell | Extra shader ops per cell per step |
|-------|---------------------------|------------------------------------|
| M5 baseline | 0 | ~100 |
| P1 (uniform IFE) | 0 | ~5 (sum to B_eff) |
| P2 (spatial IFE) | 0 | ~10 per pulse (Gaussian eval) |
| P3 (3TM thermal) | 12 bytes (3 × f32) | ~50 (3TM advance + Ms_eff lookup) |

At Nx=Ny=256, Nz=4, Nb_cells = 262144:
- P3 adds 3 MB per temp buffer × 3 buffers = 9 MB. Negligible.
- P3 shader cost: ~50% longer per step. Acceptable for thermal studies, can be toggled off for pure-coherent runs.

### D. Lookback keyword index

- IFE coupling constant → §2(a), Appendix A
- Three-temperature model → §2(b), Phase P3
- Bloch-law Ms(T) → Phase P3
- Pump-probe sequence → Phase P4
- Ultrafast demagnetization benchmark → §4 Beaurepaire 1996
- All-optical switching → Phase P3 acceptance, Phase P5
- TR-MOKE probe observable → §0 Motivation, existing probe_mz
- Gaussian beam profile → Phase P2
- Pulse CLI format → §2 CLI

---

**End of plan document.**

When resuming:
- Read `docs/plan.md` and `docs/plan-multilayer.md` first for project conventions
- Execute Phase P1 on approval; validation is the short-pulse response of a uniform film

Ready to execute Phase P1 on your go-ahead. Estimated session length for P1: 3-4 hours focused work plus validation.
