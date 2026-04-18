# FGT Zhou 2025 Morphology Calibration

**Date:** 2026-04-18
**Author:** P5 calibration agent (from conversation with `volition-billy`)
**Status:** Option-A calibration complete. Parameters captured in `material_thermal::fgt_zhou_calibrated`. ADR-005 records the decision context.

**Not** a literal reproduction of Zhou et al. *Natl. Sci. Rev.* 12, nwaf185 (2025). This document records what was done, what matched, and — critically — the structural assumptions under which the fit is valid.

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

**Cannot (known model gaps that bear on this calibration):**
- **Field-dependent equilibrium magnetization `m_e(T, B)`** — our Brillouin / MFA tables are zero-field. Real Zhou @ T_c relies on the 1 T field to give a finite pre-pulse MOKE signal; in our model, `m_e(T = T_c, B = 0) = 0` makes the operating point degenerate.
- **Two-stage longitudinal relaxation** — our phenomenological LLB has a single timescale by construction (`tau_long_base`). Zhou's τ₁ (fast) and τ₂ (slow) can't both be fit simultaneously; we pick an averaged single time constant.
- **Optical skin depth** — we assume uniform absorption across the 5 nm flake. Real 400 nm light has ≈15 nm penetration depth in metallic FGT, so this is an ≈0.7× over-estimate of the volumetric deposition for thin flakes. We fold that approximation into the fitted reflectivity (`R = 0.5` is the fit handle, not a measurement).

## 3. The calibration

### 3.1 Operating-point shift

We do **not** run at Zhou's literal T = T_c = 210 K. We run at **T = 150 K** (0.71·T_c), where:
- `m_e(150 K) ≈ 0.85` from our zero-field MFA — well-defined pre-pulse state.
- Zhou's own text notes the magneto-effect is pronounced across 90–210 K, so 150 K is in the experimentally relevant range.
- The field-induced correction to `m_e` at B = 1 T is <5 % here (Brillouin expansion), so zero-field tables are an acceptable approximation.

This is the key **assumption** the calibration rests on: **aggregate demag + recovery morphology at T = 150 K is representative of the T = 210 K experiment after removing the field-induced-equilibrium effect.**

### 3.2 Fit targets

We fit two observables derived from Zhou's reported numbers:

- `demag_frac = (|m|_initial − |m|_floor) / |m|_initial`, target **0.79**
- `recovery_frac = (|m|_22ps − |m|_floor) / (|m|_initial − |m|_floor)`, target **0.55** (partial recovery consistent with τ₂ ≈ 22 ps)

Where `|m|_initial` is the steady-state value just before the pulse (3.5 ps into the run; LLB has relaxed to `m_e(T_ambient)`).

### 3.3 Fit parameters

Grid search over three knobs:

| Parameter | Meaning | Range |
|---|---|---|
| `tau_long_base` | LLB longitudinal-relaxation time base; effective τ_∥(T_c) = `tau_long_base / α_∥(T_c)` | 0.1, 0.3, 1.0, 3.0, 10.0 fs |
| `reflectivity` | Absorbed-power derating on the pulse | 0.10, 0.30, 0.50, 0.70, 0.85 |
| `g_sub_phonon` | Phonon→substrate coupling; recovery time is `c_p / g_sub_phonon` | 5·10¹⁶, 1·10¹⁷, 2·10¹⁷, 5·10¹⁷, 1·10¹⁸ W/(m³·K) |

**`a_sf` is not a fit parameter** because the Koopmans dm/dt term is replaced by the LLB longitudinal torque once `enable_llb = true` — `a_sf` only affects `T_e` dynamics indirectly through a back-reaction term we already omit (standard Koopmans simplification). The effective *demag-depth* knob under LLB is `tau_long_base`, not `a_sf`.

### 3.4 Loss function

```
L = (demag_frac − 0.79)² + 0.5·(recovery_frac − 0.55)²
```

Recovery target weighted 0.5× because Zhou's τ₂ varies with temperature (5–22 ps) — the target is more of a band than a point.

### 3.5 Result

125 trials, 6 min wall-clock on an RTX 4060.

**Best-fit point:**
| Parameter | Value | Prior default |
|---|---|---|
| `tau_long_base` | **3.0 · 10⁻¹⁵ s** | 0.3 · 10⁻¹⁵ s (ferromagnet default) |
| `reflectivity` (on pulse) | **0.50** | N/A — not in the preset |
| `g_sub_phonon` | **2.0 · 10¹⁷ W/(m³·K)** | 2.0 · 10¹⁶ (10× weaker) |

**Achieved observables:**
| Feature | Simulated | Zhou target | Residual |
|---|---|---|---|
| demag fraction | 0.865 | 0.79 | +0.075 |
| recovery fraction at 22 ps | 0.601 | 0.55 | +0.05 |
| Loss | 0.0070 | 0 | — |

**Top-5 cluster width:** Loss < 0.025 for the five best trials spanning `tau_long_base ∈ {1, 3} fs`, `R ∈ {0.3, 0.5, 0.7}`, `g_sub_phonon ∈ {5·10¹⁶, 1·10¹⁷, 2·10¹⁷, 5·10¹⁷}`. The fit basin is **broad**; individual parameters are under-constrained (only two aggregate observables fit three parameters). A finer second-pass grid would not meaningfully reduce the uncertainty.

### 3.6 Calibrated preset

`material_thermal::fgt_zhou_calibrated()` is the calibrated preset, accessible as `--thermal-params-for fgt-zhou` / `--thermal-preset fgt-zhou` in the CLI. Use with `R = 0.5` on the pulse spec.

Diff vs `fgt_ni_surrogate()`:
- `t_c`: 220 → 210 K (match Zhou's sample)
- `tau_long_base`: 0.3 fs → 3.0 fs
- `g_sub_phonon`: 2·10¹⁶ → 2·10¹⁷ W/(m³·K)
- `m_e_table`, `chi_par_table`: regenerated for the new T_c
- Other fields unchanged

## 4. Scope, validity, and what this calibration is not

**This preset is valid for** qualitative ultrafast-demag morphology studies in FGT at T in [100, 200] K under pulses that produce demag in the 50–90 % range. The calibration captures the *ratio* of demag-to-recovery timescales observed in Zhou's data.

**This preset is NOT valid for:**
- Quantitative reproduction of Zhou at exactly T = T_c (requires field-dependent `m_e` tables).
- Two-stage recovery analysis (τ₁ and τ₂ separately resolved) — our LLB has one longitudinal time constant.
- Any simulation at B ≠ 1 T without re-fitting — the loss landscape would shift under varying field strength.
- Extrapolation to thicker FGT stacks without re-fitting `g_sub_phonon` (substrate coupling is geometry-dependent).

## 5. Reproduction

```
cargo run --release --example test_zhou_fgt             # feasibility probe
cargo run --release --example test_zhou_fgt_calibrate   # 125-trial grid (6 min)
```

Best-fit numbers quoted above were generated on an RTX 4060 at commit `<hash-of-this-commit>`.

## 6. What would need to change for a literal Zhou reproduction

In descending order of impact:

1. **Field-dependent `m_e(T, B)` tables.** Regenerate via Brillouin solution in a finite external field, include B as a per-layer scalar in the uniform, and interpolate 2D. ≈ half-day of work.
2. **Two-timescale longitudinal LLB.** Add a second characteristic time `tau_long_slow` representing the T_e → T_p → T_spin energy pathway independently from the direct LLB relaxation. ≈ one day.
3. **Optical skin-depth-weighted absorption.** Per-z absorption profile in multilayer stacks. Significant at 5 nm thickness. ≈ one day.
4. **Finite-spin-bath thermalization delay.** Replace the instantaneous `|m| → m_e(T_s)` assumption with an explicit spin-bath temperature variable. This is closer to a full M3TM with a spin reservoir rather than the current "T_s = T_e" approximation. ≈ multi-day.

Items 1 + 2 together would enable literal T = T_c calibration and resolve the two-stage morphology. Neither is currently in scope.
