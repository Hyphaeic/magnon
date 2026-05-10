# Master list of equations

Every formula the simulator integrates or evaluates, with the source
file and (where applicable) the published reference. Read alongside
[`physics.md`](physics.md) for the prose model and
[`references.md`](references.md) for the citations.

Convention throughout: `m = M / Ms` is the **reduced** magnetisation
(unit vector under LLG; magnitude varies under LLB). Effective fields
are in **Tesla** (B-form, not H-form), with γ in rad / (s·T).

---

## 1. Magnetisation dynamics

### 1.1 Landau–Lifshitz–Gilbert (default path)

```
dm/dt = −γ / (1 + α²) · ( m × B_eff + α · m × (m × B_eff) )
```

- α: Gilbert damping (dimensionless, per layer).
- γ: gyromagnetic ratio, 1.7595·10¹¹ rad / (s·T).
- B_eff: sum of effective-field contributions enumerated in §2.

Implementation: `llg.wgsl::llg_torque`. Time integration: projected
Heun (explicit trapezoidal + post-stage `normalize`).

### 1.2 Landau–Lifshitz–Bloch (opt-in path, F2 phenomenological form)

Transverse part — like LLG but with temperature-dependent damping:

```
dm/dt|_trans = −γ / (1 + α_⊥(T_s)²) · ( m × B_eff + α_⊥(T_s) · m × (m × B_eff) )
```

Longitudinal part — drives |m| toward the proxy variable `m_target`:

```
d|m|/dt|_long = ( α_∥(T_s) / tau_long_base ) · ( m_target − |m| ) · m̂
```

The proxy itself relaxes toward the field-dependent equilibrium:

```
dm_target/dt = ( α_∥(T_s) / tau_fast_base ) · ( m_e(T_s, B) − m_target )
```

Two-stage chain: `m_e → m_target → |m|`. When `tau_fast_base ≤ 0` the
chain collapses (`m_target := m_e` instantly) and the longitudinal
relaxation reduces to the F1 single-stage form. At T = 0 K both
α_⊥(0) → α_0 and α_∥(0) → 0, so the longitudinal term vanishes and the
LLB integrator reproduces LLG (gate residual ≤ 1.5·10⁻⁴).

Damping temperature dependence (Atxitia 2011 standard MFA forms):

```
α_⊥(T) = α_0 · ( 1 − T / (3·T_c) )
α_∥(T) = α_0 · ( 2·T / (3·T_c) )
```

Implementation: `llg.wgsl::llb_torque` plus the proxy update inside
`llg.wgsl::advance_m3tm` (implicit Euler, unconditionally stable).
Phenomenological ≠ Atxitia: the χ_∥-coupled effective-field form is
not yet implemented; see [`physics.md`](physics.md) §6.3.

### 1.3 Heun integrator (LLG)

Predictor:
```
m* = normalize( m + dt · τ(m, B_eff(m)) )
```

Corrector:
```
m_new = normalize( m + 0.5 · dt · ( τ(m, B_eff(m)) + τ(m*, B_eff(m*)) ) )
```

For LLB the `normalize` calls are dropped (|m| is state); a 1·10⁻⁶
floor clamp guards against div-by-zero in the longitudinal torque.

Implementation: `llg.wgsl::heun_predict / heun_correct` and
`llb_predict / llb_correct`.

---

## 2. Effective-field components

`B_eff = B_exch + B_inter + B_anis + B_zee + B_dmi + B_bias + B_sot + B_laser`

Implementation: assembled per-cell in `llg.wgsl::field_torque_phase{0,1}`
(LLG path) and `field_torque_phase{0,1}_llb` (LLB path).

### 2.1 In-plane exchange (5-point Laplacian)

```
B_exch = ( 2·A / Ms ) · ∇²m
       = ( 2·A / (Ms · dx²) ) · ( m_{i+1,j} + m_{i−1,j} + m_{i,j+1} + m_{i,j−1} − 4 m_{i,j} )
```

Boundary conditions: Neumann (clamped neighbour at the edges).

`A`: exchange stiffness [J/m]. `dx`: in-plane cell size [m]. `Ms`:
saturation magnetisation [A/m].

Implementation: `llg.wgsl::exchange_from_mag / exchange_from_pred`.

### 2.2 Interlayer exchange (z-direction stencil)

```
B_inter(k) = pf_below(k) · (m_{k−1} − m_k)  +  pf_above(k) · (m_{k+1} − m_k)

pf_below(k) = 2·A_inter[k−1] / ( Ms_k · dz[k−1]² )       (zero if k = 0)
pf_above(k) = 2·A_inter[k]   / ( Ms_k · dz[k]² )         (zero if k = N−1)
```

Note: this is a **bulk-like** z-Laplacian, not a surface-IEC formulation.
Documented departure from standard practice; see
[`physics.md`](physics.md) §6.5.

Implementation: `llg.wgsl::exchange_interlayer_from_mag /
exchange_interlayer_from_pred`.

### 2.3 Uniaxial anisotropy

```
B_anis = ( 2·K_u / Ms ) · ( m · û ) · û
```

`û`: per-layer easy axis (unit vector). `K_u`: uniaxial anisotropy
constant [J/m³].

Implementation: `llg.wgsl::anisotropy_field`.

### 2.4 Zeeman (external field)

```
B_zee = B_ext
```

Global, applies to every layer.

Implementation: `llg.wgsl::zeeman_field`.

### 2.5 Interfacial DMI

```
B_dmi = ( D / (Ms · dx) ) · ( −∂m_z/∂x , −∂m_z/∂y , ∂m_x/∂x + ∂m_y/∂y )
```

Sign convention as in `llg.wgsl::dmi_from_mag` (interfacial DMI on
layer 0, applied via the substrate). Edge BC are Neumann (no Rohart-
Thiaville chiral edge condition); see [`physics.md`](physics.md) §6.4.

### 2.6 Exchange-bias (substrate-frozen field)

```
B_bias = b_bias_layer  (per-layer constant vector)
```

Implementation: `llg.wgsl::exchange_bias_field`. Set non-zero only for
layer 0 by default.

### 2.7 Spin–orbit torque (Slonczewski-like)

```
τ_DL = ( ℏ · θ_SH · |J| ) / ( 2 · e · Ms · t_layer )      (damping-like)
τ_FL = τ_DL · ratio                                        (field-like)
σ̂   = (−J_y, J_x, 0) / |(J_y, J_x)|                        (spin polarisation)
```

Torque contribution to dm/dt:

```
dm/dt|_SOT = −γ · ( τ_DL · m̂ × (m̂ × σ̂) · |m| + τ_FL · m × σ̂ )
```

Active only on layer 0. `θ_SH`: spin Hall angle. `t_layer`: layer
thickness [m]. `ℏ = 1.054572·10⁻³⁴ J·s`, `e = 1.602177·10⁻¹⁹ C`.

Implementation: `effective.rs::sot_coefficients`,
`llg.wgsl::llg_torque`.

### 2.8 IFE laser (coherent Inverse-Faraday-Effect contribution)

For each pulse `p`:

```
B_p(r, t) = A_p(t) · w_p(r) · k̂_p
A_p(t)    = peak_field_p · exp( −(t − t_center_p)² / (2 · σ_t_p²) )
w_p(r)    = exp( −(r − spot_center_p)² / (2 · σ_r_p²) )       if σ_r > 0
            1.0                                                otherwise
σ_t_p     = duration_fwhm_p / ( 2 · √(2 · ln 2) ) ≈ FWHM / 2.3548
```

Total IFE contribution: `B_laser(r, t) = Σ_p B_p(r, t)`.

Implementation: per-step `pulse_amplitudes[p]` rewrite in
`gpu.rs::step_n`; per-cell sum in `llg.wgsl::laser_field_at`. The
`peak_field` is taken as the IFE-equivalent B in Tesla — users
calibrate against fluence externally (see Appendix A in
[`physics.md`](physics.md)).

---

## 3. Microscopic three-temperature model (M3TM, Koopmans 2010)

Three coupled per-cell ODEs. The first two govern the electron and
phonon baths; the third governs the Koopmans `m` track (which is
**not** consumed by the LLB path — it's a history of the M3TM dm/dt
for observability under enable_llb=false, and a mirror of |mag| under
enable_llb=true).

```
C_e(T_e) · dT_e/dt = P_laser(r, t) − g_ep · ( T_e − T_p )
C_p       · dT_p/dt = g_ep · ( T_e − T_p ) − g_sub_phonon · ( T_p − T_ambient )
            dm/dt   = R · m · ( T_p / T_c ) · ( 1 − m · coth(m · T_c / T_e) )

C_e(T_e)  = γ_e · T_e
R         = 8 · a_sf · g_ep · k_B · T_c² · V_at / ( μ_at · E_D² )
E_D       = k_B · θ_D
```

Variables:
- `T_e, T_p`: electron / phonon temperatures [K].
- `m = |M| / Ms(0)`: reduced magnetisation magnitude (dimensionless).
- `γ_e`: electron heat-capacity coefficient [J / (m³·K²)].
- `C_p`: phonon heat capacity [J / (m³·K)].
- `g_ep`: electron-phonon coupling [W / (m³·K)].
- `g_sub_phonon`: phonon-substrate coupling [W / (m³·K)] (post-P4 addition).
- `T_ambient`: ambient bath temperature [K].
- `a_sf`: Koopmans Elliott-Yafet scaling (dimensionless).
- `μ_at`: atomic moment in units of μ_B (dimensionless).
- `V_at`: atomic volume [m³].
- `θ_D`: Debye temperature [K].
- `k_B = 1.380649·10⁻²³ J/K`.
- `μ_B = 9.274010·10⁻²⁴ J/T`.

Numerical guards: at small argument, `m · coth(m·T_c/T_e)` is
Taylor-expanded as `T_e/T_c + m²·T_c/(3·T_e)` to avoid the 1/x pole.

Time integration: Heun (CPU reference in `thermal.rs::advance_m3tm_cell`,
GPU in `llg.wgsl::advance_m3tm`).

---

## 4. Brillouin / mean-field-approximation tables (F1, 2D)

### 4.1 Self-consistent equilibrium magnetisation

For spin-1/2 (Weiss molecular field with applied longitudinal Zeeman):

```
m_e(T, B) = tanh( ( m_e + h_B ) · T_c / T )
h_B       = g · μ_B · B / ( k_B · T_c )            (g = 2 by convention)
```

`h_B` is a **reduced Zeeman temperature** in units of T_c. Solved by
fixed-point iteration with a critical-exponent or field-tilted initial
guess (≤ 200 iterations to 10⁻¹⁰ precision).

At `B = 0`, `T < T_c`: spontaneous-magnetisation branch; standard
Brillouin solve.
At `B = 0`, `T > T_c`: paramagnetic, m_e = 0.
At `B > 0`: unique non-zero solution at any T > 0.

Implementation: `material_thermal.rs::solve_m_e_spin_half_with_field`.

### 4.2 Longitudinal susceptibility (B = 0 form, dimensionless)

```
χ_∥(T) = ( 1 / T_c ) · ( 1 − m_e² ) / ( T/T_c − (1 − m_e²) )       T < T_c
χ_∥(T) = ( 1 / T_c ) / ( T/T_c − 1 )                                T > T_c
χ_∥(0) = 0    (mean-field zero at saturation)
```

Stored in the GPU as a 2D table `[i_T, i_B]` for consistency with the
m_e table; the B-dependent form is approximated by evaluating the same
expression at the field-shifted m_e. The shipped LLB longitudinal
torque does **not** consume this table — see [`physics.md`](physics.md)
§6.3 for the open Atxitia upgrade path.

Implementation: `material_thermal.rs::brillouin_tables_spin_half_2d`.

### 4.3 Bilinear interpolation

For arbitrary (T, B):

```
i_T_frac = clamp( T / (1.5·T_c), 0, 1 ) · (n_T − 1)
i_B_frac = clamp( |B| / b_max,   0, 1 ) · (n_B − 1)
```

Standard bilinear lookup over the four corner samples.

Implementation: `photonic.rs::sample_table_2d` (host),
`llg.wgsl::sample_m_e` (GPU).

---

## 5. Optical absorption (F3 Beer–Lambert)

Light is incident on the **top** of the stack (highest layer index
`nz − 1`) and propagates downward. For layer i with thickness t_i and
optical skin depth δ_i, the per-layer attenuation factor is

```
factor(i) = ( ∏_{j > i} exp( −t_j / δ_j ) ) · ( 1 − exp( −t_i / δ_i ) ) / t_i
```

In the special case `δ_i ≤ 0`, layer i runs in **uniform-absorption
mode** and `factor(i) = 1 / t_i`. This preserves the pre-F3 single-layer
normalisation byte-for-byte.

Per-cell volumetric power [W/m³] is then

```
P_vol(cell, t) = ( P_inst(t) / (√(2π) · σ_t) ) · w_p(r) · factor(iz)
P_inst(t)      = (1 − R) · F · envelope(t)
envelope(t)    = exp( −(t − t_center)² / (2·σ_t²) )
```

Where `F` [J/m²] is the absorbed peak fluence and `R` is the pulse-side
reflectivity. The host pre-computes `factor(iz)` once at upload time
and writes per-step `P_inst` into `pulse_directions[p].w` (re-using
that vec4's previously-padding `w` slot, ADR-003).

Implementation: `gpu.rs::compute_optical_atten` (host pre-compute);
`gpu.rs::step_n` (per-step write); `llg.wgsl::laser_power_density_at`
(per-cell apply).

---

## 6. Energy balance (regression gate)

For the M3TM bath energies plus substrate outflow:

```
ΔU_e         = (1/2) · γ_e · ( T_e_final² − T_e_init² ) · V_cell
ΔU_p         =        C_p   · ( T_p_final  − T_p_init ) · V_cell
E_substrate  = ∫₀ᵗ_end g_sub_phonon · ( T_p(τ) − T_ambient ) · V_cell · dτ

E_input = (1 − R) · F · area_cell    (per pulse)

Balance: ( ΔU_e + ΔU_p + E_substrate ) / E_input = 1.000 ± 0.001
```

Verified at the Beaurepaire-Ni operating point in
`examples/test_beaurepaire_ni.rs`; the gate closes to the third decimal.

---

## 7. Pump–probe sweep observables

Aggregate magnetisation features extracted from the post-pulse
free-decay window:

```
demag_frac      = ( |m|_initial − |m|_floor ) / |m|_initial
recovery_frac   = ( |m|_22ps    − |m|_floor ) / ( |m|_initial − |m|_floor )
```

`|m|_initial` is sampled just before the pulse peak (typically 3.5 ps
into the run, after the LLB has relaxed to the field-induced
equilibrium). These are the two scalar observables fit by the Zhou
calibration.

Loss function (Zhou calibration):

```
L = ( demag_frac − 0.79 )² + 0.5 · ( recovery_frac − 0.55 )²
```

The recovery target carries a 0.5× weight because Zhou's τ_2 varies
across temperature; the target is more of a band than a point.

Implementation: `examples/test_zhou_fgt_calibrate.rs`.

---

## 8. Q-factor / linewidth (sweep harness)

For a samples vector `s_n` (one of avg_mx, probe_mz, ...) recorded at
spacing dt_sample:

```
S_k = | FFT(s_n) |²                          (1D FFT, single-sided power)
f_peak           = argmax_k(S_k) · 1 / (N · dt_sample)
freq_width_GHz   = full-width at half-S_max  (linear interpolation)
Q_factor         = f_peak / freq_width_GHz
decay_time_ns    = − 1 / fit-slope of log( | envelope(s_n) | )
```

Truncated power-of-two record, simple half-max width, 8-window
log-envelope fit. Exploratory post-processing — not measurement-
equivalent FMR/BLS linewidth metrology. See [`physics.md`](physics.md)
§6.6.

Implementation: `metrics.rs::analyze_time_series`.

---

## 9. k-space dashboard view

For a single layer's in-plane magnetisation:

```
ψ(x, y) = m_x(x, y) + i · m_y(x, y)
ψ̂(k_x, k_y) = 2D-FFT( ψ(x, y) )
P(k_x, k_y) = | ψ̂(k_x, k_y) |²    (fft-shifted, k=0 at screen centre)
P_dB(k)     = 10 · log₁₀( P / P_peak )         (clamped at −60 dB floor)
```

Separable FFT (length-`nx` along x rows, length-`ny` along y after
transpose). Computed only when the dashboard's k-space tile is the
active view (single mode) or always (grid mode).

Implementation: `bin/dashboard.rs::KspaceEngine`.

---

## 10. Stability constraints

### 10.1 LLG / LLB explicit-Heun stability

```
dt · γ · |B_eff|_max  < 0.03    (empirical, conservative)
```

For fgt-bulk with B_anis ≈ 7.77 T this gives dt < 1·10⁻¹⁴ s; the
default `fgt_default()` uses dt = 1·10⁻¹⁴.

### 10.2 LLB longitudinal stability (during a thermal window)

```
dt · α_∥(T) / tau_long_base   < 0.1
```

Currently advisory: `thermal_dt_cap` is a stored field but the host
does not enforce sub-stepping. The implicit-Euler m_target update is
unconditionally stable.

### 10.3 Pulse temporal sampling

```
σ_t  ≥  5 · dt    (recommended; warning emitted if violated)
```

---

## 11. Physical constants used

| Symbol | Value | Unit | Use |
|---|---|---|---|
| k_B | 1.380649·10⁻²³ | J/K | Koopmans R, Brillouin solve |
| μ_B | 9.27401008·10⁻²⁴ | J/T | reduced Zeeman h_B |
| ℏ | 1.054571817·10⁻³⁴ | J·s | SOT damping-like coefficient |
| e | 1.602176634·10⁻¹⁹ | C | SOT (charge) |
| g (Landé) | 2 | — | Brillouin Zeeman; convention |
| γ default | 1.7595·10¹¹ | rad/(s·T) | gyromagnetic ratio (free electron) |

All material parameters live in `material.rs`, `material_thermal.rs`,
and `substrate.rs`. See [`references.md`](references.md) for primary
sources.
