# LLB + M3TM validation — Phase P3c

**Author:** P3 execution agent (from conversation with `volition-billy`)
**Date:** 2026-04-18
**Status:** Captured at the P3c boundary. Further calibration (Beaurepaire amplitude target, full Atxitia longitudinal torque, heat-diffusion coupling) is deferred per the plan's P5 / P6+ scope.

Simulator: `magnonic-clock-sim` @ commit tagged `p3c` (parent of
`pre-thermal-baseline` tag).

---

## 1. Summary of regression gates (all passing at head, post-F1/F2/F3)

| Gate | Measurement | Result | Tolerance | Reference |
|---|---|---|---|---|
| LLG bit-identical when thermal disabled | `min_norm / max_norm` columns over 2000 steps | **byte-for-byte match** | exact | baseline CSV, `examples/test_m3tm_gpu_vs_host.rs` |
| GPU M3TM vs host reference, 1 mJ/cm² Ni, 3 ps | max relative on T_e / max absolute on \|m\| | **1.04·10⁻⁶ / 9.51·10⁻⁷** | 1·10⁻³ | `examples/test_m3tm_gpu_vs_host.rs` |
| LLB → LLG reduction at T=0 K (deterministic skyrmion) | max abs. on (avg_mx, avg_my, avg_mz) over 2000 steps | **1.5·10⁻⁴ / 1.3·10⁻⁴ / 4.5·10⁻⁶** | 1·10⁻³ | `examples/test_sp4_proxy.rs` |
| LLB demag + recovery, 1 mJ/cm² Ni, 10 ps | \|m\|_floor, \|m\|(t=10 ps) | **0.000, 0.832** | ≠1 with no NaN | `examples/test_llb_ni_demag.rs` |
| Energy balance, Beaurepaire config | (Δ U_e + Δ U_p) / E_laser | **1.000** | 1.00 ± 0.05 | `examples/test_beaurepaire_ni.rs` |
| Beaurepaire amplitude, 7 mJ/cm² Ni | \|m\|(500 fs) | **0.000** | 0.60 ± 0.05 (plan target) | see §3.2 — model-limited |
| **F1**: m_e field-dependence monotone in B at 4 temperatures | spot-check on `sample_m_e_2d` | **PASS** | non-decreasing in B | `material_thermal::tests::m_e_field_dependence_monotone` |
| **F2**: two-stage chain reproduces F1 when tau_fast=0 | max \|m_target − m_e\| | **0.0000** | < 0.01 | `examples/test_llb_two_stage.rs` (case A) |
| **F2**: tau_fast > 0 produces measurable lag | max \|m_target − m_e\| at 5×τ_long | **0.7747** | > 0.05 | `examples/test_llb_two_stage.rs` (case B) |
| **F3**: uniform mode → identical top/bot peaks | \|ΔT_e_peak\| (2-layer 20 nm × 20 nm Ni) | **0.00 K** | < 1 K | `examples/test_optical_skin_depth.rs` (case A) |
| **F3**: Beer-Lambert produces top-vs-bot split | ΔT_e_peak at δ=14 nm on 20 nm × 20 nm | **360 K** | > 50 K | `examples/test_optical_skin_depth.rs` (case B) |
| **Zhou recalibration (T=Tc=210K, B=1T)** | demag fraction / recovery@22ps | **0.776 / 0.552** | targets 0.79 / 0.55 (loss 0.0002) | `examples/test_zhou_fgt_calibrate.rs`, ADR-006 |
| 12 lib unit tests | green | **12/12 PASS** | all | `cargo test --release --lib` |

The Beaurepaire 7 mJ/cm² amplitude miss is a known model-scope limit — see §3.2.

## 2. Substitutions and deferrals

**µMAG Standard Problem 4.** The literal SP4 requires demagnetization-field physics which this simulator explicitly lacks (plan.md §5.7 carves demag out of scope). ADR-004 records the substitution: we replace SP4 with a deterministic LLG-regression gate (`--benchmark mumag-sp4-proxy`) in which the LLB-at-T=0 path must reproduce the LLG trajectory on a skyrmion-seed initial condition driven by a 10 mT transverse field.

**Full Atxitia longitudinal torque.** P3b shipped a phenomenological exponential relaxation:

    d|m|/dt|_long = (α_∥(T_s) / tau_long_base) · (m_e(T_s) − |m|) · m̂

F2 generalised this to a two-stage chain via the `m_target` proxy variable (see §7.2 for details), but the χ_∥-coupled Atxitia form remains a future upgrade: the `chi_par_table` storage buffer is bound and populated as a 2D (T, B) table after F1, but the shader's longitudinal torque still uses the phenomenological form. Drop-in replacement is mechanical when prioritised.

**Beaurepaire quantitative calibration.** At 7 mJ/cm² absorbed into a 20 nm single-cell Ni-parameterised film without lateral heat diffusion, the phonon bath equilibrates above T_c and stays there over the 10 ps window, pinning \|m\| at 0. Plan §1 already flagged in-plane ∇²T diffusion as P6+ scope. F3's Beer-Lambert improves volumetric-energy accounting in multilayers but does NOT add lateral diffusion — the Beaurepaire 7 mJ/cm² amplitude target remains model-limited. See §3.2.

## 3. Traces and numbers

### 3.1 GPU vs host M3TM — 1 mJ/cm² single-cell Ni, 3 ps

From `examples/test_m3tm_gpu_vs_host.rs`:

```
step   200 t=0.200 ps: host=314.56K gpu=314.56K
step   300 t=0.300 ps: host=735.94K gpu=735.94K
step   400 t=0.400 ps: host=952.02K gpu=952.02K
step   500 t=0.500 ps: host=907.72K gpu=907.72K
step   800 t=0.800 ps: host=778.32K gpu=778.32K
step  1500 t=1.500 ps: host=574.84K gpu=574.84K
step  2500 t=2.500 ps: host=469.78K gpu=469.78K

max relative T_e diff = 8.806e-7  (step 1302)
max absolute |m| diff  = 1.251e-6
```

### 3.2 Beaurepaire Ni — 7 mJ/cm², 20 nm, 100 fs FWHM

From `--benchmark beaurepaire-ni`:

```
t=0.2 ps: |m| = 0.935  T_e =  390.9 K  T_p =  300.4 K
t=0.3 ps: |m| = 0.000  T_e = 1825.4 K  T_p =  318.8 K
t=0.5 ps: |m| = 0.000  T_e = 2439.5 K  T_p =  426.3 K
t=1.0 ps: |m| = 0.000  T_e = 2154.2 K  T_p =  658.9 K
t=2.0 ps: |m| = 0.000  T_e = 1727.2 K  T_p =  953.1 K
t=5.0 ps: |m| = 0.000  T_e = 1267.9 K  T_p = 1197.3 K
t=10  ps: |m| = 0.000  T_e = 1219.5 K  T_p = 1218.6 K

|m| floor = 0.000 at t = 0.306 ps
|m|(500 fs)           = 0.000   (Beaurepaire target: 0.60 ± 0.05)
1/e recovery τ        = 9.69 ps from floor   (target: ~1 ps)

Energy balance:
  laser input   = 2.800·10⁻¹⁴ J/cell
  Δ(U_e + U_p)  = 2.800·10⁻¹⁴ J/cell
  ratio         = 1.000   (target 1.00 ± 0.05)
```

**Interpretation.** Energy conservation is exact to 3 decimal places — the M3TM + LLB integrator conserves energy between (electron bath, phonon bath, laser input) to better than 0.1 %. The magnetic reservoir doesn't absorb measurable energy because \|m\|=0 is achieved quickly and the LLB longitudinal work is small relative to electron-phonon thermalisation.

The amplitude miss is model-limited, not code-limited: with no lateral heat diffusion and no substrate heat sink, all 7 mJ/cm² stays in a 20 nm slab and the equilibrium phonon temperature is ≈1218 K (well above Ni T_c = 627 K). Real Beaurepaire Ni on a sapphire substrate at 7 mJ/cm² quickly dumps heat into the substrate and the in-plane surround. Reproducing the amplitude target will require either:
- heat-diffusion extension (P6+), or
- substrate thermal-coupling boundary condition, or
- parameter-fit at lower fluence where T_peak stays below T_c (Beaurepaire's 2 mJ/cm² traces).

A lower-fluence trace at 1 mJ/cm² (see `examples/test_llb_ni_demag.rs`) gives |m|_floor = 0 briefly at T_e peak, recovering to 0.82 at 10 ps — clean demag + recovery morphology without the over-heat saturation.

### 3.3 SP4 proxy — LLG vs LLB trajectories at T=0 K on a skyrmion

From `examples/test_sp4_proxy.rs`, 2000 steps @ dt = 10 fs, B_ext = (10 mT, 0, 0), `reset_skyrmion_seed(12 nm)`:

```
LLG final (avg_mx, avg_my, avg_mz) = (0.087391, 0.014785, 0.623434)
LLB final (avg_mx, avg_my, avg_mz) = (0.087546, 0.014738, 0.623432)
max |ΔM| components: mx = 1.545·10⁻⁴   my = 1.290·10⁻⁴   mz = 4.470·10⁻⁶
```

Residuals are dominated by the LLG post-normalize vs LLB no-normalize difference and the finite-precision ordering of cross-product evaluation; both paths are converging to the same magnetic equilibrium to the fourth decimal.

### 3.4 Throughput

- LLG only (thermal = None): **≈13 k steps/s** @ 256×256×1 fgt_default (unchanged vs pre-refactor baseline on the same GPU).
- LLG + M3TM (thermal Some, enable_llb = false): **≈8–10 k steps/s**.
- Full LLB + M3TM (enable_llb = true): **≈7 k steps/s**.

Well inside the plan's "3× LLG baseline" budget during thermal-active windows.

## 4. Open calibration / upgrade paths (not part of P3c)

- **Full Atxitia LLB** — replace the phenomenological longitudinal term with the χ_∥-coupled form, dimensionalising chi_par_table on upload. Current buffer is ready to be consumed.
- **Two-parameter Beaurepaire fit** (a_sf, R) against published amplitude and τ → delivers the calibrated Ni M3TM set.
- ~~**Heat-sink boundary condition**~~ — **done** post-P4 (see §6). `g_sub_phonon` per-layer term drains phonon energy toward T_ambient at rate `g_sub/c_p`.
- **FGT calibration** (P5) — same two-parameter fit against Zhou 2025.

## 5. Throughput at head (after substrate-sink addition)

- LLG only, 256² grid: ≈13 k steps/s (unchanged).
- LLG + M3TM (no LLB): ≈8–10 k steps/s.
- Full LLB + M3TM: ≈7 k steps/s.

## 6. Post-P4 addition — phonon–substrate coupling

**Rationale.** Without a heat-extraction channel, a single-cell magnetic thin film traps all absorbed laser energy forever; the equilibrium phonon bath stabilises above T_c for any non-trivial fluence, which prevents magnetisation recovery and blocks quantitative matches to Beaurepaire / Zhou.

**Implementation.** New field `LayerThermalParams.g_sub_phonon` [W/(m³·K)] with per-material defaults (≈3·10¹⁶ for Ni / Py / CoFeB on sapphire; ≈2·10¹⁶ for the FGT Ni-surrogate preset; 0 for YIG). Uploaded as a new per-layer vec4 (offset 656) plus a shared `thermal_globals.x = t_ambient` uniform at offset 672. Shader `m3tm_derivs` now subtracts `g_sub_phonon · (T_p − t_ambient) / c_p` from dT_p/dt. Host-reference `thermal::advance_m3tm_cell(..., t_ambient, dt)` mirrors the change (signature added a parameter; all call sites updated).

**Post-addition gates:**
- All P3/P4 regressions green (GPU vs host 1.0·10⁻⁶ / 9.5·10⁻⁷; SP4 proxy unchanged; LLB→LLG unchanged; pump-probe distinct responses).
- **Energy balance (Beaurepaire config, 7 mJ/cm² / 10 ps):** `Δ(U_e + U_p) + ∫ g_sub·(T_p − T_amb) dt · V_cell = 1.000 × laser_input` — closes to the third decimal when the substrate outflow channel is included.
- Substrate extracts **6.4 %** of laser input over 10 ps for Ni's default g_sub=3·10¹⁶ (cooling time constant ≈ 100 ps — expected for 20 nm film on sapphire).

**Beaurepaire quantitative miss persists at 500 fs** — substrate cooling is too slow to help in the first ps; the miss reflects the absence of optical-skin-depth modeling and lateral diffusion, not the newly-added sink. 7 mJ/cm² _absorbed_ into 20 nm raises T_e above 2400 K which pins |m|=0. Matching Beaurepaire now requires calibrating reflectivity (real Ni at 400 nm has R≈0.45) or using thicker films — parameter-fit work, not physics-model work.

## 5. How to reproduce

```
cargo test --release --lib
cargo run --release --example test_m3tm_gpu_vs_host
cargo run --release --example test_sp4_proxy
cargo run --release --example test_llb_reduces_to_llg
cargo run --release --example test_llb_ni_demag
cargo run --release --example test_beaurepaire_ni
cargo run --release --example test_llb_two_stage          # F2 acceptance
cargo run --release --example test_optical_skin_depth     # F3 acceptance
cargo run --release --example test_zhou_fgt_calibrate     # 81-trial Zhou recalibration (~13 min)
./target/release/magnonic-sim --benchmark beaurepaire-ni
./target/release/magnonic-sim --benchmark mumag-sp4-proxy
```

## 7. F-tier physics upgrades

After P3c the physics surface had three documented blockers:
field-degeneracy at T = T_c, single-timescale LLB, and uniform-absorption
energy normalisation. F1, F2, F3 closed each in turn. The numerical
results below all use commits at or after each F-tier landing — no
P3 regression is broken by the F additions.

### 7.1 F1 — field-dependent m_e(T, B) (`d67de1d`)

Replaces the 1D Brillouin / MFA tables with a 2D solve of
`m = tanh((m + h)·T_c/T)`, where `h = g·μ_B·B / (k_B·T_c)` is the
reduced Zeeman temperature. h = 0 reduces to the original equation
exactly.

Key numerical results (FGT preset, T_c = 210 K):

| (T, B) | m_e value | Physical meaning |
|---|---|---|
| (0 K, 0) | 1.0000 | Saturation |
| (T_c, 0) | 0.0000 | Critical point at B=0 (degenerate pre-F1) |
| (T_c, 1 T) | **0.264** | Field-induced equilibrium — what Zhou measures |
| (T_c, 10 T) | 0.604 | Saturation under strong field |
| (1.5·T_c, 0) | 0.0000 | Paramagnetic |

The (T_c, 1 T) cell going from 0 → 0.264 is what makes the literal-
operating-point Zhou calibration possible.

### 7.2 F2 — two-timescale longitudinal LLB (`97e686c`)

Replaces single-stage exponential relaxation with a two-stage chain:

    dm_target/dt = α_par(T_s) · (m_e(T_s, B) − m_target) / tau_fast_base
       d|m|/dt   = α_par(T_s) · (m_target  − |m|)        / tau_long_base

`m_target` is a per-cell proxy variable in a new GPU buffer (binding 11).
Updated in `advance_m3tm` via implicit Euler (unconditionally stable);
the LLB torque kernel reads `m_target` instead of computing `m_e` directly.
When `tau_fast_base ≤ 0` the chain collapses to F1's single-stage form.

Acceptance results (single-cell Ni / 1 mJ/cm² / 100 fs FWHM pulse):

| Case | tau_fast/tau_long | max\|m_target − m_e\| | \|m\|_floor | \|m\|@10 ps |
|---|---|---|---|---|
| A (F1 collapse) | 0 | **0.0000** (instant) | 0.000 | 0.832 |
| B | 5× | 0.7747 | 0.000 | 0.832 |
| C | 50× | 0.9419 | **0.105** | 0.831 |

Case C demonstrates the slow-stage rate-limiting effect: with τ_fast
much greater than the pulse duration, the proxy can't track m_e all
the way down, so |m| can't reach 0 either — exactly the bi-exponential
signature Zhou observes.

### 7.3 F3 — Beer-Lambert per-layer absorption (`99a507f`)

Per-layer attenuation factor pre-computed on host:

    uniform mode (δ ≤ 0): factor = 1/t_i    (pre-F3 normalisation preserved)
    Beer-Lambert (δ > 0): factor = (∏_{j > i} exp(−t_j / δ_j))
                                  · (1 − exp(−t_i / δ_i)) / t_i

Uploaded as `layer_optical_atten: vec4<f32>` at offset 720 (struct
720 → 736 B). Shader `laser_power_density_at` multiplies the surface
flux [W/m²] by this factor to recover volumetric power [W/m³].

Acceptance result (2-layer 20 nm × 20 nm Ni-like stack):

| Mode | Top T_e peak | Bottom T_e peak | Δ |
|---|---|---|---|
| Uniform (δ = 0) | 953.40 K | 953.40 K | 0 K |
| Beer-Lambert (δ = 14 nm) | **841 K** | **481 K** | **360 K** |

The 360 K split matches the expected `e^(-20/14) ≈ 0.24` transmittance
through the top 20 nm layer.

### 7.4 Zhou recalibration at the literal operating point (ADR-006)

After F1+F2+F3, `examples/test_zhou_fgt_calibrate` was rerun at the
**literal Zhou operating point** (T = T_c = 210 K, B = 1 T, F = 0.24
mJ/cm², 150 fs, 5 nm flake) over an 81-trial 4-axis grid:

| Parameter | Calibrated | Pre-F1 (ADR-005, T=150K shifted) |
|---|---|---|
| `tau_long_base` | 1.0 fs | 3.0 fs |
| `tau_fast_base` (F2) | 0.3 fs | 0 (collapsed) |
| `g_sub_phonon` | 3·10¹⁷ W/(m³·K) | 2·10¹⁷ |
| `optical_skin_depth_m` (F3) | 18 nm (fixed) | 0 (uniform) |
| operating T | **T_c = 210 K** | 150 K |

| Observable | Simulated | Zhou target | Residual |
|---|---|---|---|
| demag fraction | 0.776 | 0.79 | 1.4 pp |
| recovery@22 ps | 0.552 | 0.55 | 0.2 pp |
| **Loss** | **0.0002** | 0 | **35× tighter than pre-F1 best of 0.0070** |

Captured in `material_thermal::fgt_zhou_calibrated()`. ADR-006 records
the supersession of ADR-005's shifted-operating-point preset.
