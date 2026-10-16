# Calibration: Zhou-2025 FGT and how to fit new materials

Read [`physics.md`](physics.md) §6 first for the trust-boundary
discussion. Calibration here means **fitting the few free parameters of the
shipped model to a specific experiment** — not deriving parameters
from first principles. Since F4 the LLB longitudinal dynamics are the
Garanin/Atxitia form, so the fit handles are a bath coupling λ, the
pulse reflectivity R and the substrate sink g_sub.

---

## 1. The shipped FGT calibration

`material_thermal::fgt_zhou_calibrated()` is a three-parameter F4 fit to
Zhou et al. *Natl. Sci. Rev.* 12, nwaf185 (2025) at the literal
operating point.

### 1.1 Operating point

| Variable | Value |
|---|---|
| Ambient T | 210 K (= T_c) |
| External field B_z | 1 T (perpendicular) |
| Absorbed fluence F | 0.24 mJ/cm² |
| Pulse FWHM | 150 fs |
| Sample thickness | 5 nm |
| Optical skin depth (fixed) | 18 nm (FGT @ 400 nm physics estimate) |

`m_e(210 K, 1 T) = 0.264` — well-defined pre-pulse state (this number
was zero by construction pre-F1, hence the original calibration's
temporary shift to T = 150 K).

### 1.2 Fit targets and observables

```
demag_frac    = ( |m|_initial − |m|_floor ) / |m|_initial      target 0.79
recovery_frac = ( |m|_22ps    − |m|_floor ) / ( |m|_initial − |m|_floor )
                                                                target 0.55
```

`|m|_initial` is the steady-state value 3.5 ps into the run (after
the LLB has relaxed to the field-induced equilibrium). `|m|_floor` is
the post-pulse minimum; `|m|_22ps` is the value 22 ps after the pulse
peak.

Loss function:

```
L = ( demag_frac − 0.79 )² + 0.5 · ( recovery_frac − 0.55 )²
```

The recovery target carries 0.5× weight because Zhou's τ₂ varies
across temperature and the target is more of a band than a point.

### 1.3 Fit parameters and grid (F4)

3-axis grid over the dynamic-physics knobs:

| Parameter | Grid values | What it controls |
|---|---|---|
| `lambda_bath` (λ) | 0.03, 0.1, 0.3, 1.0 | α_∥ = 2λT/3T_c; longitudinal rate γα_∥/χ̃_∥ |
| `reflectivity` (pulse) | 0.30, 0.50, 0.70 | absorbed-power compensator |
| `g_sub_phonon` | 2·10¹⁶, 6·10¹⁶, 2·10¹⁷, 6·10¹⁷ W/(m³·K) | substrate sink → recovery |

4×3×4 = 48 trials, 1.6 s on the host reference (`--cpu`), a few
minutes on a GPU. The run starts at `|m| = m_e(210 K, 1 T) = 0.264`
(thermal equilibrium) rather than `|m| = 1`. κ = 3, δ = 18 nm fixed.

`tau_long_base` / `tau_fast_base` are no longer fit: they are inert
when `lambda_bath > 0`. `a_sf` remains inert under `enable_llb`.

### 1.4 Fit result (F4)

Best-fit point at the literal fluence (`fgt_zhou_calibrated()`):

| Parameter | Value | Physical expectation |
|---|---|---|
| `lambda_bath` | 0.3 | 0.02–0.3 for metals |
| `g_sub_phonon` | 6 · 10¹⁶ W/(m³·K) | G/t ≈ 1–2·10¹⁶ for G = 50–100 MW m⁻² K⁻¹, t = 5 nm |
| reflectivity (on pulse) | 0.70 | ≈ 0.5 expected |
| `optical_skin_depth_m` | 18 nm | measured optical constant |

| Observable | Simulated | Zhou target |
|---|---|---|
| demag_frac | 0.932 | 0.79 |
| recovery_frac at 22 ps | 0.471 | 0.55 |
| **Loss** | **0.023** | 0 |

The loss floor is set by the demag depth: every λ ≥ 0.1 gives
92–95 % because `|m|_floor` follows `m_e(T_e^peak, 1 T)` of the MFA
table. The pre-F4 four-knob fit reached loss 0.0002 by pushing
`g_sub` to 3·10¹⁷ (15× the Kapitza estimate) and using sub-fs τ's.

**Fluence scan.** Re-running the grid with the incident fluence
reduced (`--fluence`) isolates where the residual lives:

| F (mJ/cm²) | best λ | best R | best g_sub | demag | rec@22 ps | loss |
|---|---|---|---|---|---|---|
| 0.24 (nominal) | 0.3 | 0.70 | 6·10¹⁶ | 0.932 | 0.471 | 0.023 |
| 0.12 | 0.3 | 0.50 | 6·10¹⁶ | 0.916 | 0.527 | 0.016 |
| 0.06 | 0.3 | 0.50 | 2·10¹⁶ | 0.798 | 0.626 | 0.003 |

At a quarter of the nominal absorbed energy the fit is at the expected
R, at the Kapitza g_sub, and reproduces the demag depth. The open
question is therefore the energy actually deposited in the spin system
of a 5 nm flake (thin-film optics, substrate reflections) and the
shape of `m_e(T > T_c, B)` beyond spin-½ MFA — see
[`physics.md`](physics.md) §7.

### 1.5 Trust boundaries

The preset is calibrated at one operating point:
- T = 210 K
- B = 1 T
- F = 0.24 mJ/cm² (single-pulse absorbed fluence)
- 150 fs FWHM pulse on a 5 nm FGT flake

It is useful for qualitative Zhou-style ultrafast-demag morphology
exploration near that point, especially when the simulated response
stays in the same broad 50–90 % demag regime.

The preset is **NOT** validated for:
- **Cross-fluence prediction** at the same (T, B). The fit was
  single-fluence; the basin shape under varying F is unmeasured.
- **B ≠ 1 T** without re-fitting. The field-induced m_e(T_c, B) enters
  the dynamics and shifts the loss landscape.
- **Cross-temperature scans**, including the broader 100–220 K range and
  Zhou's full 1.5–300 K range.
- **Materials beyond FGT**.

For predictions outside the validated band, treat the simulation
output as exploratory and re-fit at the new operating point.

### 1.6 Reproduction

```bash
cargo run --release --example test_zhou_fgt           # feasibility probe (5 cases)
cargo run --release --example test_zhou_fgt_calibrate # full grid search
```

---

## 2. How to calibrate a new material

The procedure for adding a new material with experimental data:

### 2.1 Required experimental inputs

At least these scalar observables, ideally at multiple operating
points to break parameter degeneracies:

- An aggregate demag fraction (e.g., the floor of |m|/|m|_initial).
- A characteristic recovery time or recovery fraction at a known time.
- Pulse parameters: wavelength (→ skin depth), FWHM, absorbed fluence.
- Sample geometry: thickness, substrate (for `g_sub_phonon` estimation).
- Operating point: T_ambient, B_ext.

### 2.2 Required physics inputs

Before fitting, fix the parameters that have direct measurements:

| Parameter | Source |
|---|---|
| `t_c` (Curie temperature) | bulk literature |
| `gamma_e` (electron heat-capacity coefficient) | Lin/Zhigilei 2008 or DFT |
| `c_p` (phonon heat capacity) | Dulong-Petit or Debye model |
| `g_ep` (electron-phonon coupling) | Lin/Zhigilei 2008 or M3TM literature |
| `mu_atom_bohr`, `v_atom`, `theta_d` | bulk material constants |
| `optical_skin_depth_m` | optical-constants measurement at the laser wavelength |

### 2.3 Fit handles

The parameters that are genuinely fit:

| Parameter | Range to grid | What it shifts |
|---|---|---|
| `lambda_bath` | 0.03 – 1 | LLB bath coupling (longitudinal rate) |
| `g_sub_phonon` | 1·10¹⁶ – 1·10¹⁸ W/(m³·K) | recovery time τ_sub |
| `reflectivity` | 0.0 – 0.85 | absorbed-power compensator |

### 2.4 Procedure

1. Copy a near-relevant preset in `material_thermal.rs` and rename
   (e.g., `co_zhou_calibrated()`).
2. Set the physics-fixed parameters (§2.2) from literature.
3. Set the fit-handle defaults conservatively (tau values 0, R=0,
   reasonable g_sub).
4. Adapt `examples/test_zhou_fgt_calibrate.rs`:
   - Update operating-point constants (`T_AMBIENT_K`, `B_EXT_T`,
     `FLUENCE_MJ_CM2`, `PULSE_FWHM_FS`, `FLAKE_THICKNESS_NM`).
   - Update target observables (demag fraction, recovery fraction).
   - Adjust grid ranges if needed.
5. Run the grid:
   ```bash
   cargo run --release --example test_<material>_calibrate
   ```
6. Inspect top-5 cluster width. If too broad, add a third experimental
   constraint (e.g., demag at a second fluence) and tighten the loss.
7. Update the preset with the best-fit values.
8. Document the operating point, fit basin width, and trust boundaries
   in this calibration guide or an equivalent in-tree note.
9. Add a row to the preset table in [`usage.md`](usage.md) §5.3.

### 2.5 Honesty checklist before shipping a calibrated preset

- [ ] Operating point clearly named (T, B, F, pulse spec).
- [ ] Fit basin width reported (not just the point estimate).
- [ ] Trust-boundary statement ("valid for X; not validated for Y").
- [ ] Comparison against at least one independent prediction (e.g.,
  next-fluence test) — flag if absent.
- [ ] ADR documenting the procedure and limitations.

---

## 3. Cross-fluence validation: an open task

The current `fgt_zhou_calibrated()` preset has been fit at one (T, B, F)
point. Zhou 2025 reports data across multiple (T, F) combinations. A
multi-point fit would tighten the under-constrained
(`tau_fast`, `R`) plane and produce an extrapolation-trust map.

This is **pure compute work** (no code changes needed): the existing
calibration script accepts a different grid and a different loss
function. ≈ 1 day of wall-clock for a 3D (T, F, parameter) sweep on
the calibration grid.

A natural follow-up after this would be the same procedure on a second
material to test whether the parameter set transfers (it shouldn't —
each material's `(tau_long, tau_fast, g_sub)` is intrinsic).

---

## 4. Where this calibration could fail and how you'd notice

If a refit at a new operating point gives **dramatically different**
parameters from the existing one, the model is being asked to do
something it can't. Typical signatures:

- **Runaway λ → ∞** or a loss floor set by the demag depth: the
  equilibrium curve `m_e(T, B)` or the absorbed energy is wrong, not
  the relaxation rate (this is the F4 situation at nominal fluence).
- **Loss floor stays > 0.1**: the morphology can't be reproduced
  regardless of parameters. Likely missing physics — most likely
  candidates: optical skin-depth (verify δ at wavelength), demag
  field (CoFeB, Py at typical thicknesses), lateral diffusion (high F
  on small spots).
- **Discontinuous parameter sensitivity**: small grid step changes
  the best-fit drastically. Loss landscape is multi-modal; switch
  from grid search to gradient-based optimisation or report multiple
  basins.

In all three cases, the right move is documentation, not a heroic
parameter-tuning session: write up what failed and why, log it as a
known limitation in [`physics.md`](physics.md) §6, and either re-scope
the calibration or implement the missing physics.

---

## 5. References

See [`references.md`](references.md). The papers most directly
relevant to this calibration:

- Zhou et al. *Natl. Sci. Rev.* 12, nwaf185 (2025) — fit target.
- Koopmans et al. *Nat. Mater.* 9, 259 (2010) — M3TM equations.
- Atxitia & Chubykalo-Fesenko *Phys. Rev. B* 84, 144414 (2011) — LLB
  longitudinal-relaxation form.
- Leon-Brito et al. *J. Appl. Phys.* 120, 083903 (2016) — FGT bulk
  parameters (Ms, A, K_u, T_c).
- Lin, Zhigilei, Celli *Phys. Rev. B* 77, 075133 (2008) — γ_e, g_ep
  for general metals.
