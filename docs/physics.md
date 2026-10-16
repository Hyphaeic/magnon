# Physics model and architecture

Read in tandem with [`equations.md`](equations.md) for the formulas and
[`references.md`](references.md) for sources.

---

## 1. Scope

The simulator solves coupled magnetisation and ultrafast-thermal
dynamics on a 2D grid that may stack into a small (≤ 4) number of
discrete magnetic layers. It targets ultrafast-demagnetisation
experiments on van-der-Waals ferromagnets (FGT) and metal thin films
(Ni, Py, CoFeB, YIG) under femtosecond laser pulses at fluences
relevant to TR-MOKE / pump-probe measurements.

Two integration paths share the same effective-field assembly:

- **LLG path** (default): unit-magnitude `m`, projected Heun
  integration. No thermal coupling.
- **LLB + M3TM path** (opt-in): per-cell electron and phonon
  temperatures evolve under the Koopmans 2010 M3TM equations; |m| is
  free to vary and relaxes toward a field-dependent equilibrium under
  an Atxitia (χ_∥-driven) longitudinal torque; a phenomenological
  two-stage form remains selectable.

This document maps each piece of the model to the source files that
implement it. It is the place to start if you want to understand
**why** the code looks the way it does.

---

## 2. Coordinate, unit, and convention choices

### 2.1 Reduced-magnetisation convention

Every magnetic field in the code is in **Tesla** (B-form), not in A/m
(H-form). The conversion when comparing to literature in H-form is
`B = μ₀ · H`. The prefactors that look like `2A / Ms`, `2K_u / Ms`,
etc. derive from the B-form Landau-Lifshitz equation; the same
prefactors in OOMMF / MuMax3 docs are typically expressed in H-form
without the μ₀ factor.

`m` is reduced (`m = M / Ms`). Under LLG it stays unit-norm by
construction; under LLB it varies between 0 and m_e(T, B) ≤ 1.

### 2.2 Grid convention

```
flat_index = iz · (nx · ny) + ix · ny + iy
```

`iz = 0` is the bottom layer (substrate-side). Light is incident from
the **top** (highest iz). In-plane neighbour access uses Neumann
clamping at the grid edges.

### 2.3 Layer-0 substrate convention

Substrate-derived contributions (interfacial DMI, exchange bias,
α_pumping, K_u_surface, Ms_proximity, SOT spin-Hall angle) apply to
**layer 0 only**. The top of the stack is treated as free / vacuum.
Heterostructures with non-vacuum top interfaces are not modelled.

### 2.4 Cell-centred coordinates

Cell `(ix, iy)` centres at `((ix + 0.5)·dx, (iy + 0.5)·dx)` for the
purpose of evaluating spatial Gaussian profiles (pulse spot positions
etc.). dx is the in-plane cell size from `Geometry`.

---

## 3. Effective field assembly

The core LLG path executes four compute pipelines per Heun step:

```
field_torque_phase0 → heun_predict → field_torque_phase1 → heun_correct
```

`field_torque_phase0` reads `mag` and computes B_eff into `h_eff` plus
the LLG torque into `torque_0`. `heun_predict` does the explicit Euler
step into `mag_pred`. `field_torque_phase1` repeats with `mag_pred`
into `torque_1`. `heun_correct` computes the trapezoidal mean and
writes back to `mag`.

Each effective-field component is per-cell:

| Component | Formula | Source | When zero |
|---|---|---|---|
| Exchange (in-plane) | 5-point Laplacian × `2A/Ms·dx²` | `exchange_from_mag` | A = 0 |
| Interlayer exchange | z-Laplacian × `2A_inter/Ms·dz²` | `exchange_interlayer_from_mag` | single layer |
| Anisotropy (uniaxial) | `2K_u/Ms · (m·û)·û` | `anisotropy_field` | K_u = 0 |
| Zeeman | `B_ext` (global) | `zeeman_field` | B_ext = 0 |
| Interfacial DMI | derivative-of-m_z and div(m_x, m_y) | `dmi_from_mag` | D_DMI = 0 |
| Exchange-bias | per-layer constant | `exchange_bias_field` | layers other than 0 |
| Laser (IFE) | per-pulse spatial Gaussian × peak field | `laser_field_at` | pulse_count = 0 |
| SOT | Slonczewski (damping- + field-like) | folded into `llg_torque` | layers other than 0 or J = 0 |

`field_torque_phase{0,1}_llb` are the LLB-path counterparts that swap
`llg_torque` for `llb_torque`. The b_eff assembly itself is identical;
only the torque-from-b_eff conversion differs.

See [`equations.md`](equations.md) §2 for explicit formulas.

---

## 4. M3TM thermal-bath integration

When `photonic.thermal.is_some()` and `photonic.pulses` is non-empty,
the host adds a fifth pipeline pass per step: `advance_m3tm`. Per cell,
this kernel:

1. Reads `(T_e, T_p, m)` from `temp_e`, `temp_p`, and either `m_reduced`
   (P3a-style independent track) or `|mag|` (P3c back-coupling, when
   `enable_llb_flag = 1`).
2. Reads the per-pulse instantaneous absorbed surface-flux
   `pulse_directions[p].w` (written by the host each step) and applies
   the per-layer Beer-Lambert factor (F3) and the spatial Gaussian to
   get `P_laser` [W/m³].
3. Heun-integrates the (T_e, T_p, m) ODEs (Koopmans 2010 with
   substrate-sink extension on T_p).
4. Updates the F2 proxy variable `m_target` via implicit Euler:
   `m_target ← (m_target + α·dt·m_e) / (1 + α·dt)` where
   `α = α_∥(T_s) / tau_fast_base`. Implicit Euler is unconditionally
   stable for any dt; the form reduces to `m_target = m_e` instantly
   when `tau_fast_base ≤ 0`.
5. Writes back `(T_e_new, T_p_new, m_reduced_new, m_target_new)`.

The LLB longitudinal torque then drives `|m|`: in F4 Atxitia mode
(`lambda_bath > 0`) toward `m_e(T_e, B)` at the χ_∥-derived rate; in the
legacy mode toward `m_target` on the timescale `tau_long_base / α_∥(T)`.
The LLB path uses `T_e` as the effective spin temperature for
`α_⊥(T)`, `α_∥(T)`, and `m_e(T, B)`. This is a modelling approximation:
there is no separate spin-temperature state or spin-bath equilibration
equation.

When the LLG path runs (thermal off), this pipeline is not dispatched
and the four legacy pipelines execute exactly as before — the existing
P3 byte-stable regression confirms this.

See [`equations.md`](equations.md) §3 for the M3TM ODEs and §1.2 for
the LLB form.

---

## 5. F-tier upgrades (post-P3c)

These are physics additions, each independently regression-tested and
each defaulting to "off" so pre-upgrade behaviour is bit-stable.

### 5.1 F1 — field-dependent m_e(T, B) tables

Replaces the 1D Brillouin / MFA tables with 2D `[T, B]` solves. The
self-consistent equation gains a Zeeman term:

```
m_e(T, B) = tanh( ( m_e + h_B ) · T_c / T )
h_B       = g · μ_B · B / ( k_B · T_c )
```

The `B = 0` slice of the new table reproduces the old 1D table
exactly. The shader bilinear-interpolates over `(T, B)`. The longitudinal
field used in the lookup is `|B_ext.z|` (external Zeeman along the
easy axis); intra-cell anisotropy is already in B_eff and not in the
m_e lookup.

Why this matters: pre-F1 the equilibrium m_e at T = T_c with B = 0
was zero by construction. Zhou 2025 measures FGT at T = T_c with a
1 T perpendicular field; the field-induced equilibrium is what TR-MOKE
sees. Pre-F1 that operating point was degenerate. Post-F1 the
calibration runs at the literal experimental conditions.

Implementation: `material_thermal.rs::brillouin_tables_spin_half_2d` +
`solve_m_e_spin_half_with_field`; GPU upload via
`gpu.rs::pack_llb_tables`; shader lookup `llg.wgsl::sample_m_e`.

### 5.2 F2 — two-timescale longitudinal LLB

Splits the longitudinal relaxation into a fast and a slow stage,
mediated by the per-cell proxy variable `m_target`:

```
m_e(T_s, B)  ── fast (τ_fast) ──→  m_target  ── slow (τ_slow) ──→  |m|
```

Adds `tau_fast_base` to `LayerThermalParams` (default 0 = collapse to
single-stage F1 form) and a new GPU buffer `m_target` (binding 11).

Why this matters: Zhou 2025 reports two distinct decay timescales
(τ_1 ≈ 0.4 ps, τ_2 ≈ 5–22 ps). A single-timescale exponential
relaxation cannot reproduce the morphology; F2 introduces the
necessary degree of freedom.

Status after F4: this chain is the **legacy** path, selected by
`lambda_bath ≤ 0`. The Atxitia form (§5.4) produces the two timescales
without extra knobs — fast demag while T_e is hot, recovery limited by
critical slowing down near T_c and by substrate cooling.

### 5.4 F4 — Atxitia longitudinal torque, λ, K(T), A(T), α_⊥ above T_c

Four changes, all in the LLB path only:

1. **Longitudinal torque** from the MFA free energy
   (`H_∥ = (1 − m²/m_e²)/(2χ̃_∥)`, [`equations.md`](equations.md) §1.2),
   reading the χ_∥(T, B) table that F1 already uploaded. The
   sub-femtosecond `tau_long`/`tau_fast` constants are gone from the
   fitted preset.
2. **Separate bath coupling λ** (`lambda_bath`) for `α_∥ = 2λT/3T_c`.
   The FMR Gilbert damping `alpha_0` (0.001 for FGT) keeps governing
   `α_⊥`; λ for metals is 0.02–0.3 and is the calibration knob.
3. **Callen–Callen anisotropy** `K ∝ m_e^κ` (κ = 3) and
   `A ∝ m_e^a` (a = 2) via per-layer exponents. At the Zhou point
   (m_e ≈ 0.26) the anisotropy field drops from 7.8 T to ≈ 0.5 T.
4. **α_⊥ above T_c** switches to `α_0·2T/3T_c` instead of going
   negative and being clamped to zero.

The transverse damping in Atxitia mode is Garanin's
`−γ α_⊥ m̂×(m̂×B)` (no extra |m| factor, no `1/(1+α²)`).

Verification: `src/llb_host.rs` is a single-cell host integrator of
the same equations; `examples/test_llb_atxitia_gpu_vs_host.rs` shows the
GPU kernel tracks it to |Δ|m|| < 4·10⁻⁶ over 15 ps at the Zhou point.

### 5.3 F3 — Beer–Lambert per-layer absorption

The pre-F3 code divided the per-pulse absorbed power by
`layer0_thickness` and used the same volumetric P_laser for every
cell of every layer — incorrect for any non-trivial multilayer.

F3 replaces that with per-layer attenuation factors pre-computed on
the host. Light entering layer `i` is the surface flux times the
cumulative transmittance through all layers above; the per-layer
absorbed-volume fraction is `(1 − exp(−t_i/δ_i)) / t_i`. Setting
`optical_skin_depth_m = 0` reverts to uniform-mode `1/t_i`, preserving
pre-F3 behaviour byte-for-byte.

Why this matters: 5 nm FGT at 400 nm has skin depth ~18 nm. Uniform
mode dramatically over-estimates absorption in the thin layer. Without
F3, a pre-pulse equilibrium calibration at one fluence cannot be
extrapolated to another. With F3 the per-layer accounting is
physics-correct.

---

## 6. Honest limitations

These are **known structural gaps**, documented in-tree, not bugs. Read
this list before quoting simulation outputs as physical predictions.

### 6.1 No demagnetisation field

The shape-anisotropy / dipolar B contribution is not implemented. This
is out of scope per the original substrate plan and has not been
added. Practical consequences:

- µMAG SP4 (the standard correctness benchmark) cannot be reproduced
  literally. We use a deterministic LLG-vs-LLB-at-T=0 skyrmion proxy
  instead (`mumag-sp4-proxy`).
- Permalloy and CoFeB at typical thin-film thicknesses are dominated
  by shape anisotropy (~0.9 MJ/m³ for CoFeB). Without demag, K_u
  alone determines the easy-axis behaviour, which is qualitatively
  wrong for those materials. The presets exist for general-material
  experimentation but should not be trusted for quantitative thin-film
  results.

### 6.2 No lateral heat diffusion

The M3TM treats each cell as thermally isolated in-plane. Hot spots
do not spread; each cell only loses energy through (i) electron-phonon
coupling and (ii) the substrate sink term `g_sub_phonon · (T_p − T_amb)`.
The Beaurepaire 7 mJ/cm² Ni amplitude target (40 % demag at 500 fs)
remains unreachable for this reason — under uniform absorption all
cells overheat and stay hot. F3 fixes the volumetric distribution but
not the lateral transport. This is queued as a future P6+ item per the
original plan.

### 6.3 Legacy longitudinal torque is phenomenological

With `lambda_bath ≤ 0` the LLB uses the pre-F4 exponential relaxation
toward `m_target` with rate `α_∥(T) / tau_long_base`; `tau_long_base`
is a fit parameter masquerading as a material constant. All presets
except `fgt-zhou` still default to this path (λ = 0) so their gate
numbers are unchanged; set `lambda_bath > 0` to use the Atxitia form
(§5.4), which has no such constant.

Remaining simplifications of the F4 form: the longitudinal projections
of the exchange and anisotropy fields are dropped; `T_e` is used as the
spin temperature; the longitudinal rate is clipped at low T where the
MFA `χ̃_∥` underflows (§5.4). The stochastic (Langevin) terms are absent.

### 6.4 DMI has no chiral edge boundary condition

The 5-point stencil uses Neumann clamping at the grid edges (the same
BC used for all other in-plane neighbours). The Rohart-Thiaville
edge-canting boundary condition specific to interfacial DMI is not
implemented. Skyrmion-stability and edge-pinning studies are therefore
unreliable.

### 6.5 Interlayer exchange is bulk-like z-exchange

The implementation treats interlayer coupling as a z-direction Laplacian
weighted by `2·A_inter / (Ms·dz²)` — formally equivalent to a bulk
exchange-stiffness term across the interface. Standard practice is to
parameterise interfacial exchange coupling (IEC) as an interfacial
energy per area `J_int [J/m²]`, with effective field scaling as
`1 / (μ₀·Ms·t_layer)`. The two formulations have different thickness
dependences. Quantitative IEC studies should account for this
deviation.

### 6.6 Q-factor / linewidth metrics are exploratory

`metrics.rs::analyze_time_series` does a vanilla FFT on a power-of-two
truncated record, takes a half-max width, and fits a log envelope on
8 windows. This is **not** measurement-equivalent linewidth metrology.
For quantitative FMR / BLS / TR-MOKE linewidth comparisons, use
geometry-specific observables and fit models tied to the measurement
modality. The Q values reported by `magnonic-sweep` are useful for
trend exploration only.

### 6.7 `a_sf` is inert under the LLB back-coupling path

When `enable_llb = true`, the M3TM Koopmans dm/dt term is replaced by
the LLB longitudinal torque (either form), so `a_sf` does not affect
the demagnetisation depth in any LLB-active simulation. In the F4 form
the role of the Koopmans R is played by λ; a_sf only matters with
`enable_llb = false`.

### 6.9 LLB tables stop at 1.5·T_c

`m_e(T, B)` and `χ_∥(T, B)` are tabulated on `T ∈ [0, 1.5·T_c]`; hotter
cells read the last row. At the Zhou point T_e peaks at 310–420 K
(1.5·T_c = 315 K), so during the ~1 ps electron spike `m_e` is held at
its 315 K value (≈ 0.013 at 1 T) instead of falling further, and
`χ̃_∥` is up to 2× too large. This bounds the demag floor from below
and slightly slows the hot-phase relaxation. Extending the axis costs
T resolution near T_c unless the row count is raised.

### 6.8 Spin-pumping is thickness-independent

`α_total = α_bulk + α_pumping` regardless of Ms or layer thickness.
Standard Tserkovnyak-Brataas-Bauer scaling `Δα ∝ g_mix·μ_B / (Ms·t)`
gives a thickness-dependent contribution that we don't reproduce.
Thickness-sweep linewidth conclusions need this caveat.

---

## 7. Where calibration is honest, where it isn't

The `fgt_zhou_calibrated` preset is now an F4 fit with three handles
(λ, R, g_sub) instead of the pre-F4 four (`tau_long`, `tau_fast`, R,
g_sub). At the literal Zhou operating point (T = T_c = 210 K,
B_z = 1 T, F = 0.24 mJ/cm², 150 fs, 5 nm) the best fit is
λ = 0.3, R = 0.70, g_sub = 6·10¹⁶ with demag 93 % (target 79 %),
recovery 47 % (target 55 %), loss 0.023.

**The model now over-predicts the demag depth at every λ.** The floor
tracks `m_e(T_e^peak, 1 T)` of the spin-½ MFA table (0.013–0.025)
while Zhou's 79 % implies ≈ 0.055. Scanning the incident fluence shows
the discrepancy is in the energy delivered to the spins, not in the
relaxation dynamics: at F = 0.06 mJ/cm² the fit lands at λ = 0.3,
R = 0.50, g_sub = 2·10¹⁶ (the Kapitza estimate for a 5 nm vdW flake)
with demag 79.8 % and recovery 63 %, loss 0.003. Candidate causes, in
order of likelihood: thin-film optics (a 5 nm flake on a substrate does
not absorb like a semi-infinite Beer–Lambert slab), the MFA
`m_e(T > T_c, B)` under-estimating the field-induced moment of a
quasi-2D magnet, and the 1.5·T_c table clamp (§6.9).

Read as: the relaxation physics (λ ≈ 0.3, g_sub ≈ 2–6·10¹⁶) is now
in the physically expected range, and the residual is a well-posed
question about absorbed energy and the equilibrium curve. The preset is
still a single-operating-point fit and is not validated for
cross-fluence or cross-temperature extrapolation.

See [`calibration.md`](calibration.md) for the grids and
[`references.md`](references.md) for the Zhou citation.

---

## 8. Performance envelope

On an NVIDIA RTX 4060 with the current bind-group layout (12 entries,
GpuParams = 800 B):

| Path | Throughput @ 256² grid |
|---|---|
| LLG only (thermal = None) | ≈ 13 k steps/s |
| LLG + M3TM observability (LLB off) | ≈ 8–10 k steps/s |
| Full LLB + M3TM + F1 + F2 + F3 | ≈ 6–7 k steps/s |

Well inside the original plan's "≤ 3× LLG during thermal-active
windows" budget.

---

## 9. Where to read next

- [`equations.md`](equations.md) — every formula the code implements.
- [`usage.md`](usage.md) — CLI flags and dashboard keymap.
- [`calibration.md`](calibration.md) — Zhou recalibration + how to fit
  new materials.
- [`references.md`](references.md) — bibliography.
- [`references.md`](references.md) — bibliography plus notes on historical
  ADR references that are not included in this checkout.
