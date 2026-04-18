# Photonic Driver Extension — Implementation Plan

**Document type:** Implementation plan (follows the pattern of `docs/plan.md` and `docs/plan-multilayer.md`)
**Author:** system-architect (from conversation with volition-billy)
**Date:** 2026-04-17 — revised 2026-04-18 for Option B (M3TM + LLB)
**Status:** Phases P1-P2 complete. Phase P3 restructured as M3TM + LLB (Option B). Approved to execute.
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
- "At what fluence does the FGT layer cross into the non-equilibrium regime where |M| collapses (ultrafast demag), and does that collapse reset the clock phase or just damp it?"

None of these questions are answerable today because the sim can't ingest a laser pulse as an input.

### Why the Phase P3 rewrite

The original plan (dated 2026-04-17) proposed Phase P3 as a naive 3TM solver coupled to `Ms_eff(T_s) = Ms_bulk · (1 − (T_s/T_c)^β)` with |m|=1 preserved. Literature review (Atxitia/Chubykalo-Fesenko 2011; Pankratova 2022; Lepadatu 2020/Boris) shows this approach is **formally inconsistent near T_C**:

1. Naive scaling applies an equilibrium thermal suppression formula to a non-equilibrium transient, mispredicting recovery time and slope.
2. LLG with hard |m|=1 cannot represent longitudinal |M|-collapse — precisely the phenomenon ultrafast demagnetization experiments measure.
3. Koopmans-type M3TM (single parameter `a_sf` with microscopic Elliott-Yafet meaning) replaces three loosely-calibrated coupling coefficients with one well-constrained number.
4. The Landau-Lifshitz-Bloch (LLB) equation is the minimum change required to represent |M|(T) dynamics consistently. LLB reduces to LLG at low T. Boris (Lepadatu 2020) is the only extant GPU-based micromagnetic code with LLB support.

Option B (M3TM + LLB) adds ~3 weeks versus Option A (naive 3TM) but produces publication-quality ultrafast physics and opens the niche of reproducing Zhou et al. *Natl. Sci. Rev.* 12 (2025) FGT ultrafast MOKE — no published simulator has reproduced this result. See §E of the appendix for the full Option-A-vs-B trade-off argument.

### Scope

This plan adds **photonic driving** to the simulator — coherent and thermal mechanisms by which a laser pulse drives magnetization. Readout (MOKE / BLS) is already covered by the existing `probe_mz` observable.

### Lookback keywords
`inverse faraday effect` · `IFE coupling constant V` · `all-optical switching` ·
`ultrafast demagnetization` · `M3TM Koopmans` · `Landau-Lifshitz-Bloch` ·
`longitudinal susceptibility` · `adiabatic vs non-adiabatic LLB` ·
`pump-probe sequence` · `gaussian beam profile` · `femtosecond laser pulse` ·
`optical magnetic field` · `TR-MOKE signature simulation` · `photon impulse` ·
`Boris Lepadatu` · `FGT ultrafast MOKE Zhou 2025` · `Elliott-Yafet spin-flip` ·
`µMAG standard problem 4`

### What exists now (baseline, after P2)
- 2D/3D multilayer LLG solver with full substrate physics (plan.md Phases 1-5, plan-multilayer.md Phases M1-M5)
- Coherent IFE driver: `LaserPulse` struct with Gaussian temporal and spatial profile, up to 4 simultaneous pulses (P1-P2 complete)
- Per-cell Gaussian-weighted summation on GPU via `laser_field_at(idx)` in `shaders/llg.wgsl`
- `project-magnonic-clock-sim` binaries: `magnonic-sim`, `magnonic-dashboard`, `magnonic-sweep`
- Two experiments validated: monolayer clock viability and YIG/FGT hybrid advantage

### The gap this plan closes

| Capability | Now | After this plan |
|------------|-----|------------------|
| Coherent optical excitation (IFE) | ✅ (P1-P2) | ✅ |
| Gaussian beam spatial profile | ✅ (P2) | ✅ |
| Femtosecond temporal envelope | ✅ (P1) | ✅ |
| Multiple pulses (pump-probe) | ✅ data model; host-orchestration in P4 | ✅ (P4) |
| Electron-phonon-spin thermal dynamics | ❌ | ✅ M3TM (P3a) |
| Longitudinal \|M\| dynamics (LLB) | ❌ | ✅ (P3b) |
| Ultrafast demag near T_C | ❌ (naive scaling inconsistent) | ✅ (P3c) |
| Reproduce Beaurepaire Ni benchmark | ❌ | ✅ (P3 acceptance) |
| Reproduce Zhou 2025 FGT benchmark | ❌ | ✅ (P5 target) |
| All-optical switching threshold studies | ❌ | ✅ (P3c+) |

---

## 1. Requirements

### Functional

1. A `LaserPulse` type specifies: temporal peak, duration, equivalent peak B-field, direction, and spatial profile. Simulations accept a `Vec<LaserPulse>` to support pump-probe sequences. **(✅ P1-P2)**
2. The pulse's effective field modifies `B_eff` at each cell at each time step, summed alongside exchange / anisotropy / Zeeman / bias / DMI. **(✅ P1-P2)**
3. When `pulses` is empty AND `three_temp` is `None`, dynamics are identical to Phase M5 (no regression). **(✅ after P3)**
4. A user can specify pulses via CLI flag (e.g., `--pulse "t=100ps,duration=100fs,peak=0.5T"`). **(✅ P1)**
5. **(revised)** When `three_temp` is `Some(_)`, the simulator advances a per-cell M3TM (electron + phonon + spin temperatures) alongside a Landau-Lifshitz-Bloch integrator. |m| is **not** preserved; the cell magnitude tracks the local spin temperature via the longitudinal susceptibility χ_∥(T_s). LLG remains the default path when M3TM is disabled.
6. The sweep harness can iterate over pulse parameters (duration, peak field, polarization) AND M3TM parameters (fluence, reflectivity, a_sf) as additional axes.
7. **(new)** The simulator can reproduce, within published tolerances, three canonical benchmarks: µMAG Standard Problem 4 (LLG-level), Beaurepaire 1996 Ni 40% demag at 7 mJ/cm² (M3TM-level), and Zhou 2025 FGT 79% demag in 22 ps (M3TM + LLB for FGT).
8. **(new)** M3TM source term draws absorbed energy density from the same pulse envelope that drives B_IFE. A single pulse therefore has both coherent (IFE) and thermal (M3TM) action, weighted by material parameters (R, V, a_sf).

### Non-functional

1. **Performance:** a single-pulse simulation without M3TM must not slow down the per-step cost by more than 10% vs the no-pulse baseline. **With M3TM + LLB active the per-step cost budget is 3× the LLG baseline** (one new 3TM kernel, one extra torque-component, smaller dt during thermal-active windows).
2. **Memory:** P1-P2 add no new storage buffers (pulse params fit in the uniform). **P3 adds four per-cell storage buffers: T_e, T_p, T_s, and the cell \|m\| (the LLB integrator no longer normalizes).** At Nx=Ny=256, Nz=4 (≈262 k cells), this is ≈4 MB added — negligible.
3. **Backward compat:** `magnonic-sim`, `magnonic-dashboard`, `magnonic-sweep` must work unchanged when no pulse is specified AND `--enable-llb` is off. LLG remains the default integrator.
4. **Clarity:** each pulse contribution to B_eff must be inspectable from `print_summary`, and 3TM/LLB parameter provenance must be documented in the summary with citations.
5. **(new)** Thermal timestep. When M3TM is active the host auto-halves `dt` to ≤ 1 fs during the pulse + 5 ps tail, then relaxes back to the LLG default once T_s returns within 5 K of ambient. The stability bound is `dt · γ · B_eff_max < 0.03` and `dt · α_∥(T) / τ_∥(T) < 0.1` — LLB's longitudinal-relaxation term is the new tight constraint.
6. **(new)** χ_∥(T), α_∥(T), and m_e(T) (equilibrium reduced magnetization) are loaded as lookup tables on the GPU — one per layer — computed offline from material Brillouin or mean-field models and checked into `docs/llb_tables/`.

### Out of scope for this plan

- **Radiation pressure / momentum transfer** — ~12 orders of magnitude weaker than IFE; negligible.
- **Plasma wave dynamics / band-structure-resolved electron transport** — first ~50 fs physics below the M3TM abstraction. Out of scope; the M3TM treats T_e as well-defined from t = 0.
- **Coherent-photon quantum coupling** — cavity QED regime. Classical field treatment is sufficient.
- **Photothermal coupling to acoustic phonons (magnetoelastic effects)** — deferred to a later extension.
- **MOKE / BLS detection simulation** — the probe_mz observable already provides the TR-MOKE-equivalent signal.
- **Spatial heat diffusion between cells** — M3TM treats each cell as thermally isolated (no in-plane ∇²T_e term). This is the standard micromagnetic-coupled 3TM simplification (Lepadatu 2020; Atxitia 2011). A lateral-diffusion extension is listed as Phase P6+ (out of scope here).
- **Stochastic thermal torques** (thermal fluctuations in the LLB sense — Langevin noise). LLB can optionally include a stochastic term; we defer this to a later phase and run LLB in deterministic mode first.

---

## 2. Architecture

### Physics model — three mechanisms

The plan addresses three distinct physical processes, chosen because they cleanly slot into the existing LLG framework:

#### (a) Inverse Faraday Effect (IFE) — coherent  **[IMPLEMENTED: P1-P2]**

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

For our simulator's inputs, instead of asking users to compute V · σ · I_peak and plug in the resulting B-field, we expose `peak_field_T` directly as a Tesla value. Users calibrate against experimental fluence separately (see Appendix A).

#### (b) Microscopic three-temperature model (M3TM, Koopmans 2010)  **[REPLACES the original naive 3TM proposal]**

The full phenomenological 3TM (Beaurepaire 1996) has three couplings (g_ep, g_es, g_sp) that are only loosely constrained by experiment. Koopmans *et al.* (2010) Nat. Mater. 9, 259 (M3TM) replaces the spin-bath with a microscopic Elliott-Yafet spin-flip rate parameterized by a single scaling constant `a_sf`:

```
C_e(T_e)·dT_e/dt = −g_ep·(T_e − T_p) + P_laser(t) − a_sf·R_Koopmans(T_e, T_p, m)
C_p         ·dT_p/dt = g_ep·(T_e − T_p)
                dm/dt = R_Koopmans(T_e, T_p, m) / m_e_0
```

where `m = |M|/Ms(0)` (reduced magnetization), `m_e_0 = m_e(T=0)`, and:
```
R_Koopmans(T_e, T_p, m) = R · m · (T_p / T_C) · (1 − m · coth(m · T_C / T_e))
```
with `R = 8·a_sf·g_ep·k_B·T_C² · V_at / (μ_at · E_D²)`; `V_at`, `μ_at`, `E_D` are atomic volume, atomic moment, and Debye energy of the bulk material.

Rationale: the M3TM equation for `dm/dt` is what drives the LLB longitudinal relaxation below (τ_∥ is related to `R`). Using M3TM gives us one microscopic parameter (a_sf) per material, calibratable against ultrafast-demag experiments. For Ni, a_sf ≈ 0.185 (Koopmans 2010). For FGT, a_sf is unmeasured — we begin with the Ni surrogate and calibrate against Zhou 2025.

Absorbed laser power:
```
P_laser(r, t) = (1−R)·F / Δt_eff · envelope(r, t)
```
with `R` = reflectivity, `F` = fluence, `envelope(r, t)` the same Gaussian × Gaussian used for IFE (so IFE and M3TM share one pulse object).

#### (c) Landau-Lifshitz-Bloch integrator  **[REPLACES projected Heun + Ms_eff scaling]**

The LLB equation (Garanin 1997; Atxitia & Chubykalo-Fesenko 2011) is:

```
dm/dt = γ · (m × B_eff)
      − γ · α_⊥(T) / m²   · m × (m × B_eff)
      + γ · α_∥(T) / m²   · m · (m · B_eff + (1/χ_∥(T)) · m · (1 − m/m_e(T)))
```

In non-equilibrium form (what we implement), the third term becomes an explicit **longitudinal relaxation** that drives |m| toward the equilibrium reduced magnetization `m_e(T_s)`:

```
dm/dt|_long = −γ · α_∥(T_s) / χ_∥(T_s) · ((m · m_e(T_s) − m²) / m_e(T_s)²) · m̂
```

Key properties:

- **LLB does NOT enforce |m|=1.** `mag` buffer now stores actual reduced magnetization (cell |m| varies between 0 and m_e(T_s)).
- **At low T (T_s ≪ T_C):** α_∥ → 0, χ_∥ → 0⁻ limit, m_e → 1, so LLB reduces to LLG. Backward compatibility is mathematical, not merely structural.
- **Above T_C:** m_e → 0 and the longitudinal term drives |m| → 0 — ultrafast demag.
- **Input tables (loaded from `docs/llb_tables/`):** per layer, per temperature, tables of `m_e(T)`, `χ_∥(T)`, `α_⊥(T) = α_0 · (1 − T/(3·T_C))`, `α_∥(T) = α_0 · (2·T/(3·T_C))` (Atxitia standard MFA-derived forms).
- **Integrator:** Heun's method remains appropriate (explicit trapezoidal). We drop the post-step `normalize`. We add a small floor on |m| (ε = 1e-6) to guard against division by zero in the torque term when |m| transiently underflows.

#### (d) Coupling M3TM ↔ LLB

Two views of the reduced magnetization in this formulation:

- `m_cell = |mag|` (from the LLB state buffer, per cell, evolves under torque)
- `m_thermal = m(T_s)` (from the M3TM ODE, per cell, drives `m_e(T_s)` and χ_∥(T_s))

**Consistency requirement**: at steady state (far from a pulse) `|m_cell| → m_e(T_s)`. The M3TM `dm/dt` is actually the *source* for the LLB longitudinal relaxation via `m_e(T)`. So the coupling is one-way: **M3TM produces (T_s, m_e, χ_∥) each step → LLB reads these → LLB updates m_cell → M3TM reads back T_s via the equilibrium path**. We do *not* double-count `dm/dt` from M3TM; the M3TM equation is reformulated to advance only T_s and P_laser energy, while LLB does the magnetic dynamics. (This is the Lepadatu 2020 / Atxitia 2011 formulation.)

### Data model

```rust
// src/photonic.rs

/// A single laser pulse — Gaussian temporal and spatial profile. (Phases P1-P2, implemented.)
#[derive(Clone, Debug)]
pub struct LaserPulse {
    pub t_center: f64,
    pub duration_fwhm: f64,
    pub peak_field: f32,         // Tesla — IFE-equivalent
    pub direction: [f32; 3],
    pub spot_center: [f64; 2],
    pub spot_sigma: f64,
    /// (P3) Absorbed laser fluence at the peak of the Gaussian envelope [J/m²].
    /// When Some(_), drives the M3TM source term in addition to the B_IFE term.
    /// Independent so that V-calibration can be uncertain while fluence is measured.
    pub peak_fluence: Option<f64>,
    /// (P3) Surface reflectivity (0..1). Default 0.0. Used only when peak_fluence is Some.
    pub reflectivity: f32,
}

#[derive(Clone, Debug, Default)]
pub struct PhotonicConfig {
    pub pulses: Vec<LaserPulse>,
    /// P3+: M3TM + LLB configuration. When None, simulator uses LLG + IFE only.
    pub thermal: Option<ThermalConfig>,
}

/// P3+: microscopic three-temperature model + LLB parameters.
#[derive(Clone, Debug)]
pub struct ThermalConfig {
    /// Ambient / starting temperature for all three baths [K]
    pub t_ambient: f32,
    /// Per-layer M3TM parameters (one entry per layer in the stack).
    pub per_layer: Vec<LayerThermalParams>,
    /// Timestep policy during thermal-active windows.
    pub thermal_dt_cap: f64,   // default 1e-15 s
    /// Window [before, after] around each pulse_peak during which thermal_dt_cap applies [s].
    pub thermal_window: (f64, f64),   // default (0.5e-12, 10e-12)
    /// Enable LLB integrator. When false, use LLG + advance M3TM only for logging (no back-coupling).
    pub enable_llb: bool,
}

#[derive(Clone, Debug)]
pub struct LayerThermalParams {
    // ─── M3TM parameters ────────────────────────────────────────────
    /// Electron heat capacity coefficient [J/(m³·K²)]; C_e(T) = γ_e · T.
    pub gamma_e: f64,
    /// Phonon heat capacity [J/(m³·K)]. Approx Debye model near ambient.
    pub c_p: f64,
    /// Electron-phonon coupling [W/(m³·K)]. Literature values for Ni ≈ 8e17.
    pub g_ep: f64,
    /// Koopmans a_sf (Elliott-Yafet scaling). Dimensionless, ≈ 0.185 for Ni. Unknown for FGT.
    pub a_sf: f64,
    /// Atomic moment [Bohr magnetons]. For R = 8·a_sf·g_ep·kB·Tc²·V_at/(μ_at·E_D²).
    pub mu_atom_bohr: f64,
    /// Atomic volume [m³]. 1/(atomic density).
    pub v_atom: f64,
    /// Debye temperature [K]. E_D = k_B · θ_D.
    pub theta_d: f64,
    // ─── LLB parameters ─────────────────────────────────────────────
    /// Curie temperature [K] — should match bulk Tc but kept separate for robustness.
    pub t_c: f64,
    /// Low-temperature Gilbert damping α_0 (same as LLG α when thermal off).
    pub alpha_0: f64,
    /// Number of table rows, equally spaced in T from 0 to 1.5·T_c.
    pub llb_table_n: usize,
    /// Tables [0..n]: m_e(T), χ_∥(T). α_∥ / α_⊥ are computed from α_0 + T/T_c at runtime.
    pub m_e_table: Vec<f32>,
    pub chi_par_table: Vec<f32>,
}
```

### GPU side

**P1-P2 (already in place):**
- `GpuParams` (544 B) includes per-pulse arrays: `pulse_count`, `pulse_amplitudes`, `pulse_spot_centers`, `pulse_directions`.
- Host rewrites `pulse_amplitudes` each step (at uniform offset 400) before dispatch.
- Shader `laser_field_at(idx)` sums Gaussian-weighted pulse contributions.

**P3a — M3TM state + kernel:**
- Three new storage buffers: `temp_e`, `temp_p`, `temp_s` (each `cell_count × f32`). `temp_s` is actually the *reduced magnetization magnitude* `m(T_s)` used by LLB; the "spin temperature" concept is absorbed into the M3TM `dm/dt` directly (Koopmans formulation).
- Two new uniform sub-structs `LayerThermalGpu` packed into `GpuParams` (one per layer, up to MAX_LAYERS=4): `gamma_e`, `c_p`, `g_ep`, `a_sf_R_prefactor` (precomputed on host), `t_c`, `alpha_0`.
- **New table buffers**: `m_e_tables` and `chi_par_tables`, each `MAX_LAYERS × LLB_TABLE_N × f32`. Uploaded at `upload_params`, not recomputed per-step.
- New compute kernel `advance_m3tm`: for each cell, reads current (T_e, T_p, |m|), computes R_Koopmans and laser deposition, writes new (T_e, T_p).
- Execution order per step becomes: `advance_m3tm` → `field_torque_phase0` → `llb_predict` → `field_torque_phase1` → `llb_correct`.

**P3b — LLB integrator in shader:**
- New kernels `llb_predict` and `llb_correct` replace `heun_predict` / `heun_correct` when `enable_llb == true`.
- Torque computation changes: the transverse damping constant `α` becomes `α_⊥(T_s)`, and a longitudinal torque term is added.
- `mag` buffer no longer normalized; per-cell `|m|` is now state.
- Floor guard: if `|m| < 1e-6`, longitudinal torque is clamped to avoid division by zero. (Documented in shader header.)
- Temperature → table lookup done in shader via `textureSampleLevel`-equivalent (we'll use plain array indexing with bilinear interp on `T_s / T_c`).

**P3c — coupling + integration:**
- Ensure LLG path still works when `enable_llb == false`. Simplest: keep the two shader entry points live (both `heun_*` and `llb_*`), dispatch based on a host-side flag.
- Acceptance: a no-pulse, thermal-off run matches Phase M5 bitwise (not just closely).

### CLI

New flags (augment P1-P2):

```
# Coherent-only (unchanged from P2)
--pulse "t=100ps,fwhm=100fs,peak=0.5T,dir=z,x=0,y=0,sigma=500nm"

# Add M3TM + LLB (P3)
--enable-thermal
--enable-llb                        # implies --enable-thermal
--pulse "...,fluence=1.5,R=0.35"   # mJ/cm² and dimensionless reflectivity
--t-ambient 300
--thermal-params-for fgt-bulk       # load a preset LayerThermalParams record from material library
--thermal-dt-cap 1e-15
--thermal-window 0.5ps,10ps         # relative to each pulse peak
```

Presets: per-material `LayerThermalParams::for_material(bulk_name)` returns sane defaults with a clear "calibration needed" note for FGT.

---

## 3. Implementation Phases

**Current status:** P1 (uniform IFE) complete 2026-04-17. P2 (spatial Gaussian) complete 2026-04-17. Phases P3a, P3b, P3c, P4, P5 below.

### Phase P1 — Uniform IFE pulse driver  ✅ COMPLETE

Host-computed time-dependent `b_laser` uniform, summed into `B_eff` in both phase kernels. See `src/photonic.rs`, `src/shaders/llg.wgsl`, and 5 unit tests.

### Phase P2 — Spatial Gaussian beam profile  ✅ COMPLETE

Per-cell `laser_field_at(idx)` with up to 4 simultaneous pulses. Pulse amplitudes rewritten each step at uniform offset 400. See implementation notes in commit `fc9f0fd`.

### Phase P3a — M3TM solver (microscopic three-temperature) — 1 week

**Goal:** Advance per-cell (T_e, T_p) under a laser source term, with M3TM-style `dm/dt` computed but not yet coupled back to the LLG/LLB torque. This phase is self-validating against an analytic single-cell ODE.

**Steps:**
1. Add `ThermalConfig`, `LayerThermalParams` to `src/photonic.rs`.
2. Add `src/thermal.rs` with the host-side reference ODE integrator (Heun) for single-cell M3TM — used for unit tests.
3. Add `src/material_thermal.rs` with presets for Ni (Koopmans 2010 verified), Permalloy (Battiato 2010), FGT (Ni surrogate with TODO: calibrate vs Zhou 2025), YIG (effectively inert, a_sf ≈ 0).
4. Extend `GpuParams` with per-layer M3TM scalars (gamma_e, c_p, g_ep, a_sf_R_prefactor, t_c). Precompute `R` on host.
5. Allocate three new per-cell storage buffers: `temp_e_buf`, `temp_p_buf`, `m_reduced_buf`. Initialize from `t_ambient` and `m_e(t_ambient)`.
6. Create new WGSL file `src/shaders/m3tm.wgsl` with compute kernel `advance_m3tm`. Reads pulse_amplitudes + laser envelope (same Gaussian as IFE) to get P_laser(r, t), advances (T_e, T_p, |m|) one Heun step.
7. Add readout path: `max(T_e)`, `max(T_p)`, `min(|m|/m_e(T_ambient))`, all per-step, to CSV.
8. Acceptance: single-cell M3TM ODE (Rust host code) matches the GPU kernel output to 1e-3 relative for Ni at 1 mJ/cm², 100 fs FWHM.

**Acceptance criteria:**
- Ni at 1 mJ/cm² fluence: T_e_peak = 1500±100 K at ~200 fs, matches Koopmans 2010 Fig. 1.
- Ni at 7 mJ/cm², 100 fs: |m| drops from 1.0 to 0.60±0.02 at ~500 fs (Beaurepaire 1996).
- No-pulse run: T_e = T_p = 300 K, |m| = 1.0, constant (bit-identical across steps except for float-roundoff).
- LLG dynamics unchanged: |m| at cell level remains 1 because LLB is not yet active.

### Phase P3b — LLB integrator — 1 week

**Goal:** Replace projected Heun with LLB in the shader, conditioned on `--enable-llb`. Enforce backward compatibility with `--enable-llb` off.

**Steps:**
1. Generate LLB tables offline: `scripts/gen_llb_tables.py` reads material parameters, computes m_e(T) from the mean-field (Brillouin) equation, computes χ_∥(T) from ∂m_e/∂H near H=0. Output `docs/llb_tables/{material}.json` with N=256 rows.
2. Add `m_e_table_buf` and `chi_par_table_buf` as storage buffers (read-only), size `MAX_LAYERS × N × f32`. Loaded at `upload_params`, static thereafter.
3. Modify `src/shaders/llg.wgsl`: add new entry points `llb_predict`, `llb_correct`. The field-torque kernels stay the same (b_eff computation is independent of |m| except for the anisotropy prefactor which we'll discuss below).
4. Inside `llb_predict` / `llb_correct`:
   - Read `m_c = mag[idx]` (vector, no normalization).
   - Read `T_s = |m_reduced[idx]|` interpreted as reduced magnetization proxy. Via a simple consistency-restoring map `T_s ≈ T_e` at ambient, or direct table lookup.
   - Interpolate `m_e`, `chi_par`, `alpha_perp`, `alpha_par`.
   - Compute `dmdt_llb` = transverse LLG + longitudinal relaxation term.
   - Heun step without post-normalize. Clamp |m| ∈ [1e-6, 2·m_e] to prevent numerical blow-up.
5. Anisotropy prefactor handling: `B_anis = (2·K_u(T)/Ms(T)) · (m·û)·û`. Near T_C, Ms(T) → 0 but we have m = M/Ms(0), so the prefactor is stable. We keep `layer_anisotropy_pfs` as computed from Ms(0) (unchanged from P2).
6. Exchange prefactor handling: same argument — we work in reduced-magnetization units, so `2·A/(Ms(0)·dx²)` is stable at all T. (A(T) softening near T_C is a 2nd-order effect we defer.)
7. GPU-host feedback: after each step, read `m_reduced` → advance M3TM → next step.
8. Adaptive dt: during thermal window, cap dt at `thermal_dt_cap`. Implement via scheduling at the host (call `step_n(k)` in smaller slices with smaller uniform dt when inside the window).

**Acceptance criteria:**
- `--enable-llb` OFF, no thermal: identical probe_mz trace to M5 default within float precision.
- `--enable-llb` ON, T_s = 300 K constant: LLB reproduces LLG dynamics to ~1e-3 relative over 10 ps (α_∥ ≈ 0 at low T).
- `--enable-llb` ON, Ni at 7 mJ/cm²: |m| drops to 0.60 then recovers to 0.95 within 10 ps (matches Beaurepaire 1996 trace shape).
- No NaN from the longitudinal torque floor guard in any test case.

### Phase P3c — Coupling & validation — 5 days

**Goal:** Close the loop between M3TM and LLB; validate full system against canonical benchmarks.

**Steps:**
1. Remove the "M3TM advances |m| independently" stub from P3a: LLB now owns |m| dynamics. M3TM advances only (T_e, T_p) and reads |m_cell| = |mag[idx]| for the Koopmans R term.
2. Verify energy conservation: total laser input = sum of ΔU_e + ΔU_p + ΔU_magnetic over the simulation window, within 5% (the balance leaks into the longitudinal LLB dissipation — allowed).
3. Run Beaurepaire Ni reproduction: 100 fs, 7 mJ/cm², σ⁺, 20 nm film → 40±5% demag at 500 fs, recovery τ ≈ 1 ps.
4. Run SP4 regression: no-pulse, no-thermal, µMAG standard problem 4 passes with LLB integrator active but α_∥ ≈ 0. Trajectory matches LLG within 0.5%.
5. Add `--benchmark {beaurepaire-ni,mumag-sp4}` CLI helper to auto-configure these runs.
6. Write `docs/llb_validation.md` with traces of the above two benchmarks.

**Acceptance criteria:**
- Beaurepaire Ni benchmark passes within tolerance.
- SP4 passes with LLB integrator active (demonstrates LLB → LLG reduction).
- Energy conservation within 5%.
- Validation doc committed.

### Phase P4 — Pump-probe sequencer — 3 days

*(Unchanged from the original plan, but now works over both coherent IFE and thermal M3TM pulses.)*

**Goal:** Orchestrated multi-pulse protocols. Pump-probe is the canonical ultrafast experiment.

**Steps:**
1. Ensure `PhotonicConfig.pulses: Vec<LaserPulse>` supports multiple (already true from P2).
2. CLI: repeated `--pulse` args build the list (already works).
3. Add protocol templates to the sweep harness:
   - `--pump-probe-mode` — automatically generates (pump, probe) pulse pairs
   - `--pump-probe-delay-range START END STEPS` — iterates probe delays
4. Sweep CSV: add columns `pulse_count`, `first_pulse_t_ps`, `total_fluence_mj_cm2`, `max_te_k`, `min_m_reduced`.

**Acceptance criteria:**
- Pump-probe sweep produces Q-factor vs pump-probe delay curves.
- Two-pulse coherent control experiments replicable.
- TR-MOKE trace reconstruction: probe at fixed delay, pump varies in intensity.

### Phase P5 — First optical experiment: FGT ultrafast MOKE reproduction — 2-3 weeks

**Goal:** Exercise the photonic + thermal capability by reproducing Zhou et al. *Natl. Sci. Rev.* 12 (2025) — 79% demag of FGT in 22 ps under 400 nm, 50 fs pulse. **No published simulator has reproduced this result**; this is a publishable benchmark niche.

**Steps:**
1. Create `experiment-fgt-ultrafast-mokereproduction` in `branches/hir/experiments/`.
2. Charter: reproduce Zhou 2025 trace within 20% error on max-demag amplitude and recovery τ.
3. Registry entity, workspace scope extension, run.sh.
4. Initial parameter set: FGT bulk material from `material.rs`, a_sf = Ni surrogate (0.185) as starting point, γ_e/c_p/g_ep from Lin/Zhigilei general-metal fits.
5. Calibration loop: two-parameter fit (a_sf, reflectivity) to match Zhou 2025 max-demag and recovery τ.
6. Figures: (a) Zhou Fig. 3 overlay with simulation, (b) parameter sensitivity heatmap, (c) spatial map of |m|(t) under focused Gaussian spot.
7. Promote winning figure; ADR-003 documenting the calibrated FGT M3TM parameter set.

**Acceptance criteria:**
- Zhou 2025 max-demag within 20% (target: 79±5%).
- Zhou 2025 recovery τ within 30% (target: 22±5 ps).
- Calibrated (a_sf_FGT, R_FGT) recorded in ADR-003.
- Published notebook + figures promoted.

### Phase P6+ — Beyond this plan (listed for roadmap continuity)

- Spatial heat diffusion (in-plane ∇²T_e term).
- Stochastic LLB (Langevin thermal torques for magnonic noise modeling).
- Two-temperature acoustic phonon coupling (magnetoelasticity under laser pulse).
- All-optical switching regime study (GdFeCo surrogate), deferred until FGT benchmark nails coefficients.

---

## 3.5 Code Change Surface — What Option B Touches

This section enumerates the concrete refactors required by Option B (M3TM + LLB). Compared to the naive-3TM option, the structural impact is concentrated in two files (`shaders/llg.wgsl` and `gpu.rs`) plus three new module files. No existing function is deleted; LLG remains reachable as the default path.

### `src/photonic.rs`  (extend — already exists from P1-P2)

- Add fields to `LaserPulse`: `peak_fluence: Option<f64>`, `reflectivity: f32`.
- Add `ThermalConfig`, `LayerThermalParams` structs (see §2 Data model).
- Add `PhotonicConfig.thermal: Option<ThermalConfig>`.
- Add `field_at_time` variant `power_density_at(t) -> [f32; N_pulses]` returning per-pulse instantaneous P_laser amplitude for M3TM source term.
- Print-summary update to list thermal parameters and M3TM provenance.
- **No breaking changes**: existing P1-P2 callers (which pass `PhotonicConfig::default()`) continue to work with `thermal = None` ⇒ LLG path.

### `src/material.rs`  (extend)

- Each `BulkMaterial` gains a weak reference to its M3TM preset by name. This is informational only; the actual `LayerThermalParams` lives in a new module.
- Add field `pub m3tm_preset_key: &'static str` (e.g., `"ni-m3tm"` or `"fgt-ni-surrogate"`). Documents which preset is recommended.

### `src/material_thermal.rs`  (new — ~200 lines)

- `impl LayerThermalParams`:
  - `fn for_key(key: &str) -> Option<Self>`
  - Presets: `ni_m3tm()` (Koopmans 2010 verified), `py_m3tm()` (Battiato 2010), `fgt_ni_surrogate()` (Ni a_sf + FGT Tc=220K as starting-point calibration), `yig_inert()` (a_sf ≈ 0), `cofeb_m3tm()` (Sato 2018 β=1.73).
  - Each preset carries provenance in `notes: &'static str`.

### `src/effective.rs`  (minor extension)

- No changes to existing methods.
- Add `temperature_dependent_anisotropy_prefactor(layer, T_s) -> f64` — returns `2·K_u(T)/Ms(0)`, reserving for a future K_u(T) softening. For P3 we hold K_u constant (2nd-order effect).
- Add docstring clarifying that all prefactors are in "M(0) units" — i.e., LLB's reduced magnetization convention.

### `src/config.rs`  (minor)

- `SimConfig` already has `pub photonic: PhotonicConfig` (present since P1). No new field; thermal is reached through `photonic.thermal`.
- Add `SimConfig::validate_thermal_consistency()` — checks that thermal_dt_cap ≤ current dt when thermal is enabled, that per-layer thermal params match stack length, that FGT thermal preset warns "uncalibrated, Ni surrogate".
- Print-summary dispatches to `ThermalConfig::print_summary` when enabled.

### `src/thermal.rs`  (new — ~150 lines)

- Host-side reference single-cell M3TM integrator for unit tests.
- `pub fn advance_m3tm_cell(state, pulse_power_density, thermal_params, dt) -> new_state` — a vanilla Rust port of the Koopmans equations.
- Used in 6 unit tests: one for each preset + a 2-cell comparison against GPU.

### `src/gpu.rs`  (major — ~300 lines added)

The largest code-change surface. Extensions:

1. **`GpuParams` struct extension** — current size 544 B; LLB/M3TM adds **per-layer thermal scalars** (`gamma_e`, `c_p`, `g_ep`, `a_sf_R_prefactor`, `t_c`, `alpha_0` — 6 floats × 4 layers = 96 B). New `GpuParams` size: 640 B. Pulse offset (currently 400) shifts to **576 B** for `pulse_count`, **592 B** for `pulse_amplitudes`, **608 B** for `pulse_spot_centers`, etc. The per-step uniform write `queue.write_buffer(..., 400, ...)` becomes `..., 592, ...`.
2. **Three new per-cell storage buffers** (STORAGE, COPY_SRC, COPY_DST):
   - `temp_e_buf: cell_count × f32`
   - `temp_p_buf: cell_count × f32`
   - `m_reduced_buf: cell_count × f32` (this is the magnitude-only companion to `mag`; LLB torque reads it for the longitudinal term)
3. **Two new read-only storage buffers** (for LLB lookup tables):
   - `m_e_table_buf: MAX_LAYERS × LLB_TABLE_N × f32`
   - `chi_par_table_buf: MAX_LAYERS × LLB_TABLE_N × f32`
4. **Bind-group-layout extension**: current 6 bindings (params + 5 storage) grows to 11. The bind group is recreated on `GpuSolver::new` — no runtime resize needed.
5. **New compute pipelines**:
   - `m3tm_pipeline` — entry `advance_m3tm` from new WGSL `m3tm.wgsl`.
   - `llb_predict_pipeline`, `llb_correct_pipeline` — entries `llb_predict`/`llb_correct` from an extended `llg.wgsl`.
6. **Dual-integrator dispatch in `step_n`**: current code runs `[ft_phase0, predict, ft_phase1, correct]`. New code:
   - When `thermal.enable_llb`: run `[advance_m3tm, ft_phase0, llb_predict, ft_phase1, llb_correct]` (5 passes).
   - Else (no thermal or thermal-log-only): unchanged 4-pass LLG loop.
   - Dispatch selection is a host-side boolean read from `SimConfig.photonic.thermal`.
7. **Adaptive dt** inside `step_n`: when the solver's current `t_sim` lies within a thermal window of any pulse, split the requested `n` into sub-loops with temporary `dt = thermal_dt_cap`. `queue.write_buffer(..., 16, &[dt_new])` each time dt changes.
8. **Observables extension**: add `max_t_e`, `max_t_p`, `min_m_reduced` fields to `Observables`. Readback uses `m_reduced_buf`.
9. **`upload_params` extension**: upload LLB tables once; write `LayerThermalGpu` sub-structs alongside existing layer scalars.

### `src/shaders/llg.wgsl`  (major extension)

Current file is 338 lines. LLB support adds:

1. **`Params` struct** — new fields at the end (to avoid shifting P1-P2 offsets):
   - `layer_thermal_gamma_e: vec4<f32>`
   - `layer_thermal_c_p: vec4<f32>`
   - `layer_thermal_g_ep: vec4<f32>`
   - `layer_thermal_a_sf_r: vec4<f32>`  (precomputed R from host)
   - `layer_thermal_t_c: vec4<f32>`
   - `layer_thermal_alpha_0: vec4<f32>`
2. **New bindings (indices 6-10)**:
   - `@group(0) @binding(6) var<storage, read_write> temp_e: array<f32>`
   - `@group(0) @binding(7) var<storage, read_write> temp_p: array<f32>`
   - `@group(0) @binding(8) var<storage, read_write> m_reduced: array<f32>`
   - `@group(0) @binding(9) var<storage, read> m_e_table: array<f32>`
   - `@group(0) @binding(10) var<storage, read> chi_par_table: array<f32>`
3. **New functions**:
   - `sample_m_e(iz, T) -> f32` — bilinear lookup
   - `sample_chi_par(iz, T) -> f32`
   - `alpha_perp(iz, T_over_tc) -> f32 = alpha_0 · (1 - T/(3·T_c))`
   - `alpha_par(iz, T_over_tc) -> f32 = alpha_0 · (2·T/(3·T_c))`
   - `llb_torque(m, b_eff, iz, T_s) -> vec3<f32>` — includes longitudinal relaxation term
4. **New entry points**:
   - `llb_predict` — as `heun_predict` but torque uses `llb_torque` and no `normalize()` at end
   - `llb_correct` — as `heun_correct`, no `normalize()`, with floor clamp on |m|
5. **Preserved**: `heun_predict`, `heun_correct`, `field_torque_phase0`, `field_torque_phase1` all stay; LLG path is unchanged when the host dispatches the old 4-pipeline sequence.

### `src/shaders/m3tm.wgsl`  (new — ~100 lines)

New compute kernel file. Entry point `advance_m3tm` iterates per-cell:
- Reads `temp_e[idx]`, `temp_p[idx]`, `m_reduced[idx]`.
- Reads per-pulse instantaneous amplitude from `params.pulse_amplitudes`, computes P_laser contribution via same Gaussian spatial weight as `laser_field_at`.
- Evaluates Koopmans R: `R_koop = R_prefactor · m · (T_p/T_c) · (1 - m · coth(m·T_c/T_e))`.
- Advances `T_e`, `T_p` with Heun (stored internally; 2 mini-substeps per call).
- Writes updated state.

### `src/bin/sweep.rs`  (minor)

- Already has `photonic: Default::default()` default init.
- Add `--thermal` / `--enable-llb` CLI passthrough.
- Add CSV columns `max_te_k`, `min_m_reduced`.

### `src/main.rs`  (minor extension)

- Parse new CLI flags `--enable-thermal`, `--enable-llb`, `--thermal-params-for`, `--t-ambient`, `--thermal-dt-cap`, `--thermal-window`, `--benchmark`.
- Update print-summary to list thermal config.
- Add `--benchmark beaurepaire-ni` and `--benchmark mumag-sp4` canonical-config shortcuts.

### `scripts/gen_llb_tables.py`  (new — ~80 lines)

Offline Brillouin-MFA computation of `m_e(T)` and `χ_∥(T)` tables. Writes JSON → consumed by `material_thermal.rs` at build-time via `include_str!` or at runtime via file load.

### `docs/llb_tables/`  (new directory)

Checked-in precomputed tables for each supported material: `ni.json`, `fgt-ni-surrogate.json`, `yig.json`, `permalloy.json`, `cofeb.json`. One-time generation; re-run only if material Tc or spin-quantum-number changes.

### `docs/llb_validation.md`  (new, written during P3c)

Validation traces: Beaurepaire Ni benchmark figure + SP4 trajectory + FGT (from P5) overlay.

### Summary of breaking vs non-breaking changes

- **Breaking**: the `pulse_amplitudes` write-offset in `gpu.rs::step_n` moves from 400 → 592. Anyone running forked code that hard-coded offset 400 would break.
- **Breaking**: `Observables` gains new fields; callers using positional destructuring would break. Existing callers use named access and are unaffected.
- **Non-breaking**: LLG kernels preserved. `--enable-llb` off = LLG = M5 bit-identical path.
- **Non-breaking**: `PhotonicConfig::default() = { pulses: [], thermal: None }` — all existing init sites continue to work.

---

## 4. Validation Strategy

### Per-phase regression tests (cumulative)

| Test | Phase | Verifies |
|---|---|---|
| No-pulse sim identical to M5 default | P1 ✅ | No regression |
| Gaussian envelope recovers correctly in probe_mz | P1 ✅ | Temporal envelope correctness |
| Peak field matches specified peak_field_T | P1 ✅ | Amplitude calibration |
| Spatial Gaussian falls off per r² from spot center | P2 ✅ | Profile correctness |
| Single-cell M3TM ODE matches GPU kernel to 1e-3 | P3a | M3TM implementation correctness |
| µMAG Standard Problem 4 passes with LLG | P3b | LLB backward compat to LLG |
| µMAG Standard Problem 4 passes with LLB enabled at 0 K | P3b | LLB → LLG reduction at low T |
| Ni at 7 mJ/cm², 100 fs: 40% demag at 500 fs | P3c | Beaurepaire 1996 reproduction |
| Recovery τ ≈ 1 ps for Ni | P3c | M3TM recovery dynamics |
| Energy conservation within 5% | P3c | Physical consistency |
| No NaN after 10⁶ steps under any parameter combination | P3c | Numerical stability |
| Pump-probe fluence threshold reproducible | P4 | Pump-probe orchestration |
| FGT Zhou 2025: 79% demag at 22 ps | P5 | FGT-specific physics |

### Canonical benchmarks

**1. µMAG Standard Problem 4** (LLG correctness gate): 500×125×3 nm permalloy film, external field switching. Published trajectory from OOMMF/MuMax. Simulator should reproduce to ~0.5%. Serves as "did we break LLG?" regression.

**2. Beaurepaire 1996 Ni ultrafast demag**: 20 nm Ni film, 60 fs pulse, 7 mJ/cm², σ⁺, 400 nm. Observed: 40% drop in Ms at 500 fs, recovery to 95% by 4 ps. After P3c, simulator should reproduce within 5% on amplitude and 30% on τ.

**3. Zhou et al. *Natl. Sci. Rev.* 12 (2025) FGT ultrafast MOKE**: 20 nm FGT flake, 50 fs pulse, 400 nm. Observed: 79% drop in MOKE signal at 22 ps. After P5, simulator should reproduce within 20% on amplitude and 30% on τ.

The choice of three benchmarks (pure LLG, classic M3TM, FGT-specific) tests three distinct code paths end-to-end.

---

## 5. References

### Plan docs
- `docs/plan.md` — substrate extension (Phases 1-5 complete)
- `docs/plan-multilayer.md` — multilayer extension (Phases M1-M5 complete)
- `docs/viability-report.md` — §3.3 mentioned optical readout as motivation; §5 listed optical parameters as uncharacterized for FGT

### IFE / all-optical literature
- Pitaevskii (1961), "Electric forces in a transparent dispersive medium," *Sov. Phys. JETP* **12**, 1008 — original IFE theory
- van der Ziel, Pershan, Malmstrom (1965), *Phys. Rev. Lett.* **15** — IFE in diamagnetic glass
- Stanciu et al. (2007), *Phys. Rev. Lett.* **99** — single-pulse AO switching in GdFeCo
- Kimel et al. (2005), *Nature* **435** — IFE dynamics in DyFeO₃
- Mangin et al. (2014), *Nat. Mater.* **13** — materials engineering for AOS

### M3TM / three-temperature literature
- Beaurepaire et al. (1996), *Phys. Rev. Lett.* **76**, 4250 — original 500-fs demag observation in Ni
- Koopmans et al. (2010), *Nat. Mater.* **9**, 259 — microscopic three-temperature model (M3TM) with Elliott-Yafet parameter `a_sf`
- Battiato, Carva, Oppeneer (2010), *Phys. Rev. Lett.* **105**, 027203 — superdiffusive spin transport (alternative framework; we do not adopt, but cited for completeness)
- Pankratova et al. (2022), *Phys. Rev. B* **106**, 134401 — heat-conserving 3TM (HC3TM); clarifies energy-balance subtleties we track
- Lin, Zhigilei, Celli (2008), *Phys. Rev. B* **77**, 075133 — general-metal γ_e, g_ep parameter compilation

### LLB / finite-temperature micromagnetics
- Garanin (1997), *Phys. Rev. B* **55**, 3050 — original LLB derivation
- Atxitia, Chubykalo-Fesenko, Chantrell (2011), *Phys. Rev. B* **84**, 144414 — LLB formulation used here; α_∥/α_⊥ temperature dependence
- Evans et al. (2012), *Phys. Rev. B* **85**, 014433 — VAMPIRE atomistic framework that inspired parameter tables
- Lepadatu (2020), *J. Appl. Phys.* **128**, 243902 — Boris, GPU-based LLB micromagnetic code. **Closest architectural analog to our extension.**
- Leliaert, Mulkers et al. (2017), *AIP Advances* **7**, 125010 — adaptive timestep strategies for micromagnetics (informs our thermal-window dt-cap)

### Material parameters (M3TM / LLB)
- Sato et al. (2018), *Phys. Rev. B* **97**, 014433 — CoFeB Bloch exponent β = 1.73, a_sf calibration
- Zahn et al. (2022), *Phys. Rev. B* **105**, 014426 — FGT magnetic moment, Tc dependence
- Lichtenberg, Zhou et al. (2024-2025), various — FGT thin-film characterization
- **Zhou et al. *Natl. Sci. Rev.* 12 (2025) nwae453** — FGT ultrafast MOKE, 79% demag at 22 ps. P5 reproduction target.

### µMAG benchmarks
- OOMMF µMAG Standard Problem 4 — https://www.ctcms.nist.gov/~rdm/mumag.org.html

### Reference implementations
- **Boris** (Lepadatu 2020) — GPU micromagnetic with LLB. Closest analog to what we're building.
- ubermag / magnum — have phenomenological 3TM coupling (not LLB-native).
- MuMax³ — no native LLB; external scripts sometimes approximate it.
- VAMPIRE — atomistic; generates the Brillouin tables we consume.

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| IFE coupling constant V for FGT unknown | High | Medium | Expose peak_field_T directly; document in ADR-003 |
| Shader performance drops >3× with LLB active | Medium | Medium | Benchmark P3b; if >5×, precompute α_⊥/α_∥ per cell in the M3TM kernel instead of per-step in torque kernel |
| **LLB longitudinal relaxation numerically stiff near T_C** | **High** | **High** | **Thermal-window dt cap (1 fs); Heun predictor-corrector; |m| floor clamp. If still unstable, fall back to semi-implicit update for longitudinal term only** |
| **M3TM + LLB energy-accounting drifts > 10%** | Medium | Medium | HC3TM reformulation (Pankratova 2022) as fallback; validation gate in P3c |
| **FGT a_sf unmeasured — Ni surrogate may be wrong order of magnitude** | High | High for P5 | Two-parameter fit (a_sf, R_FGT) against Zhou 2025 in P5; document calibration in ADR-003; mark FGT thermal preset "uncalibrated" |
| **LLB drops |m|=1 invariant; bugs previously masked by normalize now surface** | High | Medium | Bitwise LLG regression gate (|m|=1 checked post-hoc); run before & after every P3 commit; µMAG SP4 is a failsafe |
| **Table-lookup quantization introduces low-frequency artifacts** | Low | Low | N=256 rows × bilinear interp gives ~0.4%/K resolution in m_e — below LLB's own modeling error |
| Pulse duration < dt | Low | High (instability) | Warning if pulse_fwhm < 5·thermal_dt_cap; thermal-dt-cap defaults to 1 fs |
| **Host-GPU roundtrip for adaptive dt adds latency** | Low | Low | dt changes happen only at window boundaries (2-3 times per pulse), not per-step |
| Rewriting shader entry points breaks existing dashboard binding setup | Medium | Medium | Keep LLG entry points live; dashboard uses same bind group |

Bolded rows are new relative to the original plan — Option B's structural risks.

---

## 7. Execution Protocol

Same pattern as plan.md and plan-multilayer.md:

- Work one phase at a time. Commit at each boundary.
- After each phase, run the full regression suite.
- Document design decisions inline. ADRs for non-obvious trade-offs.
- Update this plan as execution proceeds. Add "Implementation notes" subsections.
- **Phase P3 is split into P3a, P3b, P3c — commit at each sub-boundary.** Each sub-phase has independent acceptance criteria so we can pause between them if a blocker emerges.
- **Before starting P3a**: tag current HEAD as `pre-thermal-baseline` so we can diff LLG behavior if LLB regresses.
- **Mandatory regression gate**: every P3 commit runs the µMAG SP4 benchmark. A failing SP4 blocks merge.

### Suggested execution order

P1 ✅ → P2 ✅ → **P3a (M3TM, non-coupled) → P3b (LLB, thermal-off baseline) → P3c (coupled, benchmarks) → P4 (pump-probe) → P5 (FGT reproduction)**.

Option B estimated duration: **~4 weeks for P3 + 3 days P4 + 2-3 weeks P5 = ~7-8 weeks total**. This is ~3× the naive-3TM option but produces publication-quality output and the open FGT-benchmark niche.

---

## 8. Appendix

### A. IFE coupling constant V — calibration strategy

*(Unchanged from original plan.)* The relation between laser fluence and peak_field_T requires knowing V. For FGT this isn't in the literature. Three approaches: (1) parametric sweep, (2) literature surrogate V_typical ≈ 10⁻²⁰ s²/(kg·m), (3) experimental collaboration. We currently use (1) + (2). P5 provides an indirect calibration via fitting Zhou 2025.

### B. Pulse-to-simulation mapping

*(Unchanged from original plan.)* Typical experimental pulse: 100 fs FWHM, 500 nm wavelength, 1 mJ/cm² fluence, 5 μm spot, σ = +1. peak_field_T ≈ 1 T order-of-magnitude.

### C. Memory / computational estimate — revised for Option B

Per cell, at various phases:

| Phase | Extra GPU memory per cell | Extra shader ops per cell per step |
|-------|---------------------------|------------------------------------|
| M5 baseline | 0 | ~100 |
| P1 (uniform IFE) | 0 | ~5 (sum to B_eff) ✅ |
| P2 (spatial IFE) | 0 | ~10 per pulse (Gaussian eval) ✅ |
| **P3a (M3TM)** | **8 bytes (T_e, T_p)** | **~40 (Koopmans R + Heun)** |
| **P3b (LLB)** | **4 bytes (m_reduced)** | **~30 (longitudinal term + table lookup)** |
| **P3c total** | **12 bytes/cell + MAX_LAYERS · LLB_TABLE_N · 2·4 bytes tables** | **~70 added to LLG baseline** |

At Nx=Ny=256, Nz=4, 262,144 cells:
- Per-cell thermal state: 3 MB.
- LLB tables: 4 layers × 256 rows × 2 tables × 4 bytes = 8 KB. Negligible.
- Total: ≈3 MB added. Still negligible on any discrete GPU.

Shader cost during thermal-active window: roughly **3× LLG baseline** (5 passes vs 4, each slightly heavier). Outside the thermal window (99%+ of steady-state time) the cost is ~LLG + ~5% (dormant LLB longitudinal term).

### D. Lookback keyword index

- IFE coupling constant → §2(a), Appendix A
- M3TM / Koopmans → §2(b), Phase P3a
- Landau-Lifshitz-Bloch → §2(c), Phase P3b
- χ_∥(T) tables → Phase P3b, `docs/llb_tables/`
- Pump-probe sequence → Phase P4
- Beaurepaire benchmark → §4, Phase P3c
- µMAG SP4 benchmark → §4, Phase P3b
- Zhou 2025 FGT benchmark → §4, Phase P5
- All-optical switching → Phase P3c, Phase P5
- TR-MOKE probe observable → §0 Motivation, existing probe_mz
- Gaussian beam profile → Phase P2
- Pulse CLI format → §2 CLI
- Adaptive dt / thermal window → §1 non-functional #5, Phase P3b step 8
- Boris (Lepadatu 2020) reference implementation → §5, §2(c)

### E. Why Option B (LLB) over Option A (naive Ms scaling)

During the research review preceding this revision, three parallel lit searches surfaced converging evidence:

**1. Naive Ms(T_s) scaling is formally inconsistent near T_C.** The expression `Ms_eff(T) = Ms · (1 − (T/T_C)^β)` is an *equilibrium* thermodynamic relation. Applying it to a non-equilibrium transient (spin temperature rising faster than phonons can equilibrate) gives demag amplitudes and time constants that diverge from experiment by factors of 2-5 near T_C (Atxitia 2011 §III.B). Far from T_C (T_s < T_C/2) the naive scaling is fine — but that's precisely the regime where demag is small and uninteresting.

**2. LLG with hard |m|=1 cannot represent longitudinal |M| collapse.** This is the literal phenomenon ultrafast demag experiments measure. Without LLB, we can't reproduce Beaurepaire, can't reproduce Zhou, can't study switching thresholds. The simulator's output would be visibly inconsistent with any published trace.

**3. Boris (Lepadatu 2020) demonstrates LLB on GPU is tractable.** Boris is the closest architectural analog to our extension and has published LLB results. The shader-kernel pattern we need is not novel research — it's an adaptation of Boris's approach to our wgpu stack.

**4. FGT ultrafast MOKE is an open benchmark niche.** A literature scan found no published micromagnetic simulator that has reproduced Zhou et al. *Natl. Sci. Rev.* 12 (2025). Being the first to reproduce it — and publishing the calibrated FGT M3TM parameter set — is a concrete, publishable deliverable for P5.

**5. Cost difference is ~3 weeks, not ~3 months.** The LLB-specific code is concentrated in one shader and one kernel. Tables are precomputed offline. The adaptive-dt scheduler adds complexity but no deep physics.

Against all five points, the 3-week premium for Option B buys publication-quality output and a benchmark niche. Option A (naive scaling) produces numbers the simulator can't defend. This is the rationale for the revision.

---

**End of plan document.**

When resuming:
- Read `docs/plan.md`, `docs/plan-multilayer.md`, and this file (with emphasis on §3.5 Code Change Surface) for project conventions and the full Option B change set.
- Execute Phase P3a on approval; validation is the single-cell M3TM ODE match against the GPU kernel.

Ready to execute Phase P3a on your go-ahead. Estimated session length for P3a: 2-3 days focused work plus validation.
