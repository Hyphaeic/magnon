# FGT Zhou 2025 Morphology Calibration

**Date:** 2026-04-18 (initial Option-A); refreshed 2026-05-09 (post-F1/F2/F3).
**Author:** P5 calibration agent (from conversation with `volition-billy`)
**Status:** Recalibrated at the literal Zhou operating point (T = T_c = 210 K, B = 1 T) after F1+F2+F3 closed the structural blockers documented in §2 of the original Option-A. Parameters captured in `material_thermal::fgt_zhou_calibrated`. ADR-006 supersedes ADR-005 for the current preset.

**Not** a complete reproduction of Zhou et al. *Natl. Sci. Rev.* 12, nwaf185 (2025), but the residuals on the two aggregate observables are now within rounding (1.4 percentage points on demag, 0.2 on recovery). This document records what was done, what matched, and the structural assumptions under which the fit is valid.

---

## 1. What Zhou 2025 measured

TR-MOKE on a 5 nm Fe₃GeTe₂ flake with a gold auxiliary substrate, 400 nm pump, 150 fs FWHM, 0.24 mJ/cm² incident fluence, 1 kHz rep. rate. Scan axes: ambient temperature 1.5–300 K, external field 0–7 T perpendicular.

Headline number (cited by us as the calibration target):
- **at T = T_c = 210 K and B = 1 T: 79 % demagnetization within 22.2 ps.**

Morphology: two distinct demag timescales. τ₁ = 0.39–0.55 ps (fast). τ₂ = 5.1–22.2 ps (slow, temperature-dependent).

## 2. What our simulator can and can't do

**Can (physics present in the code at head):**
- Per-cell Koopmans M3TM: `(T_e, T_p, |m|)` evolution with electron-phonon coupling + Koopmans R source. Energy conservation gate passes to 0.1 %.
- LLB longitudinal relaxation of `|m|` toward `m_e(T_s)` with rate `α_∥(T) / tau_long_base`. At T = 0 it reduces to LLG.
- Substrate heat sink: `dT_p/dt |_sub = −g_sub_phonon·(T_p − T_ambient)/c_p`. Gives a tunable τ_sub for thermal recovery.
- Zeeman from a uniform external B.

**Cannot (known model gaps that bore on the original Option-A calibration; all three CLOSED post-F1/F2/F3):**
- ~~**Field-dependent equilibrium magnetization `m_e(T, B)`**~~ — **F1 added 2D (T, B) tables**. `m_e(T_c, 1T) ≈ 0.264` is well-defined.
- ~~**Two-stage longitudinal relaxation**~~ — **F2 introduced an `m_target` proxy variable** with its own time constant `tau_fast_base`, decoupling the τ_1 (fast) and τ_2 (slow) Zhou timescales.
- ~~**Optical skin depth**~~ — **F3 added Beer-Lambert per-layer absorption** with a per-material `optical_skin_depth_m` field. FGT-at-400 nm: 18 nm.

**What's still missing (lower-impact gaps that aren't blocking this calibration):**
- **Stochastic LLB** (Langevin thermal field) — deferred per plan.md §1; doesn't bear on aggregate-morphology fitting.
- **Full Atxitia LLB longitudinal torque** with χ_∥ as effective-field prefactor — the chi_par_table buffer is populated but the shader still uses the phenomenological (m_target − |m|)/τ form. F2 is sufficient for two-stage morphology; the χ_∥-coupled form remains a future upgrade.
- **a_sf is inert under LLB back-coupling** — covered in ADR-005 §Consequences/Neutral; structural feature of the M3TM ↔ LLB coupling architecture.

## 3. The calibration

### 3.1 Operating point — literal Zhou conditions (post-F1/F2/F3)

After F1+F2+F3 the calibration runs at **T = T_c = 210 K** with **B_z = 1 T** — the literal Zhou operating point. The pre-F1 calibration shifted to T = 150 K because the zero-field tables made T = T_c degenerate; that workaround is no longer needed.

| Variable | Value |
|---|---|
| Ambient T | 210 K (= T_c) |
| External field B_z | 1 T (perpendicular) |
| Fluence F | 0.24 mJ/cm² |
| Pulse FWHM | 150 fs |
| Sample thickness | 5 nm |
| Optical skin depth | 18 nm (FGT @ 400 nm physics estimate; F3) |

`m_e(210 K, 1 T) = 0.264` — well-defined pre-pulse state.

### 3.2 Fit targets

We fit two observables derived from Zhou's reported numbers:

- `demag_frac = (|m|_initial − |m|_floor) / |m|_initial`, target **0.79**
- `recovery_frac = (|m|_22ps − |m|_floor) / (|m|_initial − |m|_floor)`, target **0.55** (partial recovery consistent with τ₂ ≈ 22 ps)

Where `|m|_initial` is the steady-state value just before the pulse (3.5 ps into the run; LLB has relaxed to `m_e(T_ambient, B)`).

### 3.3 Fit parameters

4-axis grid search over the dynamic-physics knobs:

| Parameter | Meaning | Range |
|---|---|---|
| `tau_long_base` | LLB **slow** stage time base; τ_slow(T) = `tau_long_base / α_∥(T)` | 1.0, 3.0, 10.0 fs |
| `tau_fast_base` (F2) | LLB **fast** stage time base; τ_fast(T) = `tau_fast_base / α_∥(T)` | 0.1, 0.3, 1.0 fs |
| `reflectivity` | Pulse-side reflectivity (with F3, no longer compensating for /thickness mis-norm) | 0.30, 0.50, 0.70 |
| `g_sub_phonon` | Phonon→substrate coupling; recovery time is `c_p / g_sub_phonon` | 1·10¹⁷, 3·10¹⁷, 1·10¹⁸ W/(m³·K) |

`optical_skin_depth_m` is **not** in the grid — it has a direct physics estimate (18 nm for FGT at 400 nm); fitting it would conflate absorption physics with dynamics.

`a_sf` is **not** in the grid — under LLB back-coupling the Koopmans dm/dt term is bypassed, so `a_sf` is inert for demag depth (ADR-005 §Consequences/Neutral).

### 3.4 Loss function

```
L = (demag_frac − 0.79)² + 0.5·(recovery_frac − 0.55)²
```

Recovery target weighted 0.5× because Zhou's τ₂ varies with temperature (5–22 ps) — the target is more of a band than a point.

### 3.5 Result (post-F1/F2/F3, 81-trial grid)

**Best-fit point:**

| Parameter | Calibrated | Pre-F1 (ADR-005) |
|---|---|---|
| `tau_long_base` | **1.0 · 10⁻¹⁵ s** | 3.0 · 10⁻¹⁵ s |
| `tau_fast_base` (F2) | **0.3 · 10⁻¹⁵ s** | 0 (collapsed F1 path) |
| `reflectivity` (pulse) | **0.50** | 0.50 |
| `g_sub_phonon` | **3.0 · 10¹⁷ W/(m³·K)** | 2.0 · 10¹⁷ |
| `optical_skin_depth_m` (F3) | **18 nm (fixed)** | 0 (uniform mode) |
| Operating T | **210 K (= T_c)** | 150 K (shifted) |

**Achieved observables:**

| Feature | Simulated | Zhou target | Residual |
|---|---|---|---|
| demag fraction | 0.776 | 0.79 | 0.014 (1.4 pp) |
| recovery fraction at 22 ps | 0.552 | 0.55 | 0.002 (0.2 pp) |
| **Loss** | **0.0002** | 0 | — |

The pre-F1 best fit was Loss = 0.0070 with residuals (0.075, 0.05) at T = 150 K. Post-F1+F2+F3 the loss tightens by **35×** *and* the operating point matches Zhou's experiment.

**Top-4 cluster width:** Loss < 0.0024 for the four best trials, all sharing `(tau_long = 1 fs, g_sub = 3·10¹⁷)` and spanning `tau_fast ∈ {0.1, 0.3, 1} fs` × `R ∈ {0.3, 0.5}`. Two parameters (τ_long, g_sub) are well-constrained by the two observables; (τ_fast, R) trade off against each other and remain under-constrained.

### 3.6 Calibrated preset

`material_thermal::fgt_zhou_calibrated()` carries the new fit. CLI: `--thermal-preset fgt-zhou`. Use with `R = 0.50` on the pulse spec.

Diff vs `fgt_ni_surrogate()`:

- `t_c`: 220 → **210 K** (match Zhou's sample)
- `tau_long_base`: 0.3 fs → **1.0 fs**
- `tau_fast_base`: 0.0 → **0.3 fs** (engages F2 two-stage chain)
- `g_sub_phonon`: 2·10¹⁶ → **3·10¹⁷ W/(m³·K)**
- `optical_skin_depth_m`: 0.0 → **18 nm** (engages F3 Beer-Lambert)
- `m_e_table`, `chi_par_table`: regenerated as 2D (T, B) tables for the new T_c (F1)

## 4. Scope, validity, and what this calibration is not

**This preset is valid for** Zhou-style ultrafast-demag morphology studies in FGT at T in [100, 220] K under pulses that produce demag in the 50–90 % range. With F1/F2/F3 in play it now also captures:

- The literal T = T_c operating point (F1).
- A two-stage decay morphology that crudely matches Zhou's τ₁ + τ₂ split (F2).
- Beer-Lambert absorption rather than uniform-thickness normalisation (F3).

**This preset is still NOT validated for:**

- Cross-fluence prediction (different F values were not in the fit; the basin shape under varying F is unmeasured).
- B ≠ 1 T without re-fitting — the field-induced `m_e(T_c, B)` enters the dynamics and shifts the fit basin.
- Cross-temperature scans through the entire 1.5–300 K range Zhou measured — the fit was at one (T, B) point.
- Materials beyond FGT — the parameters are FGT-specific (optical skin depth, T_c, atomic moment, etc.).

## 5. Reproduction

```
cargo run --release --example test_zhou_fgt             # feasibility probe
cargo run --release --example test_zhou_fgt_calibrate   # 81-trial grid (~13 min on RTX 4060)
```

Best-fit numbers quoted above were generated on an RTX 4060 at commit landing ADR-006.

## 6. What would need to change for further calibration

In descending order of impact:

1. ~~**Field-dependent `m_e(T, B)` tables.**~~ ✅ **DONE in F1** (`d67de1d`). 2D Brillouin solve with finite reduced-Zeeman field `h = g·μ_B·B / (k_B·T_c)`; `m_e(T_c, 1T) ≈ 0.264`.
2. ~~**Two-timescale longitudinal LLB.**~~ ✅ **DONE in F2** (`97e686c`). Per-cell `m_target` proxy variable; `m_target` chases `m_e` on `tau_fast_base`, `|m|` chases `m_target` on `tau_long_base`.
3. ~~**Optical skin-depth-weighted absorption.**~~ ✅ **DONE in F3** (`99a507f`). Per-layer Beer-Lambert factors pre-computed on host; shader applies the per-layer attenuation on every cell.
4. **Finite-spin-bath thermalization delay.** Replace the instantaneous `|m| → m_e(T_s)` assumption with an explicit spin-bath temperature variable. F2's m_target proxy partially addresses this (decoupling the m response timescale from the m_e value), but a real spin-bath would have its own temperature evolved by M3TM and would alter the energy bookkeeping. Multi-day.
5. **Full Atxitia LLB longitudinal torque.** Replace the phenomenological `(m_target − |m|)/τ_∥` form with the χ_∥-coupled effective-field form. The chi_par_table buffer is populated but unread. ≈ 2 days plus full preset re-fit.
6. **Cross-fluence/cross-temperature calibration.** Single-operating-point fits don't constrain extrapolation. Adding 4–5 more (T, F) points from Zhou's data and refitting under multi-objective loss would dramatically tighten the fit basin. ≈ 1 day of compute, no code change.

Items 4–6 are the lowest-leverage upgrades remaining; the original three model gaps (1–3) that motivated this calibration are all closed. The most consequential next step is item 6 — additional fit points — which doesn't require any code work, only running the existing calibration script over a different grid.
