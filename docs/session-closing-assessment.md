# Session Closing Assessment — Phases P3 → P5A → F1/F2/F3

**Date:** 2026-04-18 (initial P3→P5A) → 2026-05-09 (F1/F2/F3 + Zhou recalibration)
**Baseline:** `pre-thermal-baseline` tag (now consumed in subsequent rebase; see "commit train" below for the canonical sequence at head)
**Head:** F3 commit + cleanup
**Total commits since pre-thermal baseline:** 19

---

## 1. Headline

The simulator now reaches the literal Zhou 2025 operating point with a fit
loss 35× tighter than the pre-F1 calibration. Three structural blockers
that the original P5 Option-A documentation flagged ("what would need to
change for a literal Zhou reproduction") have all been closed:

1. **Zero-field m_e tables** → F1 added 2D (T, B) Brillouin-MFA tables.
2. **Single-timescale LLB** → F2 added `m_target` proxy with its own time
   constant `tau_fast_base`, decoupling the τ_1 and τ_2 Zhou observables.
3. **Uniform-absorption normalisation** → F3 added Beer-Lambert per-layer
   attenuation factors; thin-flake / multilayer absorption is now
   physics-correct.

Every existing P3/P4/D-tier regression remains byte-stable. `cargo build
--bins --examples` is warning-free; 12/12 lib unit tests pass.

Six ADRs (001 – 006) record the design decisions, including five
deliberate deviations from the plan document, each with rationale and
consequences.

## 2. Commit train (most-recent first)

```
99a507f  F3: optical skin-depth Beer-Lambert per-layer absorption profile
1c92ef0  F2: two-timescale longitudinal LLB via m_target proxy variable
1e6ae81  F1: field-dependent m_e(T, B) tables — closes T=Tc degeneracy
651192a  Dashboard: grid mode — all 8 heatmap views at once
b7bc6bc  D2: runtime pulse + thermal controls in the dashboard
a00a3d3  Dashboard: live k-space view via 2D FFT of (mx + i·my)
7ea27f5  Dashboard: symmetric auto-range for Mx/My/Mz heatmaps
d8f3b4a  D1: dashboard status panel + cyclable heatmap fields + thermal CLI
e979609  Add session closing assessment
38a23b1  Cleanup: silence dead_code warnings, update older examples for P3 API
29be3a8  P5 Option A: FGT Zhou-morphology calibration (limits-aware)
910a808  Add substrate-sink term g_sub_phonon on the phonon bath
22bc127  P4: pump-probe sequencer (multi-pulse CLI + sweep axis)
624b310  P3c: M3TM↔LLB coupling, SP4 proxy, Beaurepaire harness, validation doc
3d0d1e4  P3b: LLB integrator with phenomenological longitudinal relaxation
60adb2e  P3a: Koopmans M3TM solver (non-coupled to LLG torque)
```

Plus two micro-commits (`dad76ea` gitignore, `4fa2987` "niii"). Total ≈
4,500 net LOC across ~20 modified/new files.

## 3. Gate matrix at head

| Gate | Value | Threshold | Notes |
|---|---|---|---|
| LLG byte-for-byte when thermal disabled | exact match | exact | 202 rows, `min_norm`/`max_norm` cols |
| GPU M3TM vs host reference | 1.04·10⁻⁶ / 9.51·10⁻⁷ (T_e / \|m\|) | 1·10⁻³ | single-cell Ni, 3 ps, 1 mJ/cm² |
| LLB → LLG reduction at T=0 | max 1.5·10⁻⁴ | 1·10⁻³ | deterministic skyrmion, 2000 steps |
| SP4 proxy (skyrmion + transverse B) | mx 1.5·10⁻⁴ / my 1.3·10⁻⁴ / mz 4.5·10⁻⁶ | 1·10⁻³ | literal SP4 requires demag (ADR-004) |
| Energy balance (Beaurepaire, with substrate) | 1.000 ± 0.001 | ±0.05 | Δ(U_e + U_p) + ∫ g_sub · (T_p − T_a) dt |
| LLB demag + recovery morphology (1 mJ/cm² Ni) | \|m\| floor → 0, recovers to 0.83 @ 10 ps | no NaN | |
| Pump-probe multi-pulse distinct responses | 3 distinct \|avg_mx\| amplitudes | non-degenerate | single / Δ=0.5 ps / Δ=5 ps |
| **F1: m_e field-dependence monotone in B** | spot-check at 4 temperatures | **PASS** | new lib unit test |
| **F2: tau_fast=0 collapses to F1** | max \|m_target − m_e\| | **0.0000** | `test_llb_two_stage` case A |
| **F2: tau_fast > 0 produces lag** | max \|m_target − m_e\| @ 5×τ_long | **0.7747** | `test_llb_two_stage` case B |
| **F3: uniform mode → identical layer peaks** | \|ΔT_e\| 2-layer 20×20 nm | **0.00 K** | `test_optical_skin_depth` case A |
| **F3: Beer-Lambert produces top/bot split** | ΔT_e @ δ=14 nm | **360 K** | `test_optical_skin_depth` case B |
| **Zhou recalibration (T=Tc=210K, B=1T)** | demag / rec@22ps | **0.776 / 0.552** | targets 0.79 / 0.55, loss 0.0002 |
| Lib unit tests | 12 / 12 | all | including new `m_e_field_dependence_monotone` |

All green.

## 4. Capability delta vs pre-session

**Can do now that couldn't before:**
- Per-cell Koopmans M3TM integration with electron + phonon + spin-bath
  dynamics, either as an observability layer (LLB off) or fully coupled to
  the magnetic dynamics (LLB on).
- LLB longitudinal relaxation in **two stages** with independent fast and
  slow timescales (F2). Reduces to single-stage F1 form when
  `tau_fast_base = 0`; reduces to LLG when both timescales are infinite.
- Substrate thermal-sink coupling via per-layer `g_sub_phonon`.
- **Field-dependent equilibrium magnetization `m_e(T, B)`** (F1). Operating
  at T = T_c is no longer degenerate.
- **Beer-Lambert per-layer optical absorption** with per-material skin depth
  (F3). Multilayer thin-film studies now have physics-correct volumetric
  energy distribution.
- Pump-probe protocols in the sweep harness, with thermal observables.
- Two canonical benchmarks as one-line CLI invocations:
  `--benchmark beaurepaire-ni` and `--benchmark mumag-sp4-proxy`.
- One **literal-operating-point** Zhou calibration (`fgt_zhou_calibrated`)
  fitting demag fraction (77.6 %) and recovery@22ps (55.2 %) with loss
  0.0002 — within rounding of Zhou's headline numbers.
- Six material presets total (`ni_m3tm`, `py_m3tm`, `fgt_ni_surrogate`,
  `fgt_zhou_calibrated`, `yig_inert`, `cofeb_m3tm`).
- Interactive dashboard with 8-panel grid view of all field types,
  k-space FFT visualization, runtime thermal/LLB toggles, and pump-probe
  sequencer keys.

**What didn't change:**
- LLG path is the default — `SimConfig::fgt_default` produces a valid
  config with `photonic.thermal = None`.
- All P3/D-tier regression numerical results are bit-stable across F1, F2,
  and F3 (the new physics defaults to "off" / "uniform" / "collapsed F1").
- Pre-existing examples produce the same trajectories they always did.

## 5. Performance envelope (RTX 4060)

| Path | Throughput | Multiplier vs LLG |
|---|---|---|
| LLG only (`thermal = None`) | ≈ 13 k steps/s @ 256²×1 grid | 1× |
| LLG + M3TM (LLB off) | ≈ 8–10 k steps/s | 1.3–1.6× |
| Full LLB + M3TM + F1 + F2 + F3 | ≈ 6–7 k steps/s | 1.9–2.2× |

Well inside the plan's "≤ 3× LLG during thermal-active windows" budget.

## 6. Decisions recorded as ADRs

| ADR | Decision | Why |
|---|---|---|
| 001 | `fgt_bulk` (Leon-Brito 2016) as default, not `fgt_effective` (Garland) | Pre-session — scientific honesty in the default preset |
| 002 | Stack order is a first-class user input | Pre-session — multilayer heterostructures |
| 003 | P3a/P3b GPU layout: thermal appended at end; single shader file; pulse_directions.w reuse; raised storage-buffer limit | Session — minimise plan-vs-code drift risk |
| 004 | µMAG SP4 replaced by a proxy (skyrmion + transverse B @ T = 0) | Session — simulator has no demag field; literal SP4 not reachable |
| 005 | `fgt_zhou_calibrated` preset — morphology fit at shifted operating point | P5 Option A — at T = 150 K because pre-F1 zero-field tables made T = T_c degenerate |
| **006** | **`fgt_zhou_calibrated` recalibrated at literal T = T_c after F1/F2/F3** | **Loss 35× tighter; supersedes ADR-005** |

## 7. Honest limitations at head

These are **known structural gaps**, documented in-tree, not bugs:

1. **No demagnetization field** — out of scope since plan.md §5.7. Makes
   µMAG SP4 unreachable in the literal sense. Breaks any workflow where
   shape anisotropy dominates.
2. **No lateral heat diffusion** — single-cell and grid simulations alike
   treat cells as thermally isolated. For fluence calibration at high
   energies (Beaurepaire 7 mJ/cm²) this still traps heat indefinitely;
   F3 fixes the volumetric distribution but not the lateral transport.
3. **Phenomenological LLB longitudinal torque** — F2 added a two-stage
   chain via `m_target`, but the χ_∥-coupled Atxitia form is still not
   used. The `chi_par_table` buffer is populated as a 2D (T, B) table
   but the shader doesn't read it. Drop-in replacement is mechanical.
4. **`a_sf` is inert under the LLB back-coupling path** — follows from the
   M3TM ↔ LLB coupling architecture (P3c). Real influence on demag depth
   requires `enable_llb = false` or a full Atxitia LLB form.
5. **Single-operating-point Zhou calibration** — fitting two aggregate
   observables (demag fraction, recovery@22ps) gives a tight fit but the
   basin is broad in the (`tau_fast`, `R`) plane. Cross-fluence / cross-
   temperature predictions remain extrapolations.

The original limitations 4 ("zero-field m_e tables") and 5 ("no optical
skin depth") from the pre-F session have been **closed** by F1 and F3
respectively; this list reflects what's still genuinely open.

## 8. Where to go next (priority-ordered menu)

### If you want *better physics fidelity*

1. ~~**Field-dependent m_e(T, B) tables**~~ ✅ **DONE in F1.**
2. ~~**Two-timescale LLB**~~ ✅ **DONE in F2.**
3. ~~**Optical skin-depth absorption**~~ ✅ **DONE in F3.**
4. **Full Atxitia LLB longitudinal torque** — dimensionalise χ_∥, use as
   effective-field prefactor. Replaces the phenomenological
   `(m_target − |m|)/τ` form end-to-end. ≈ 2 days (physics + per-preset
   recalibration).
5. **Lateral heat diffusion** — adds an in-plane ∇²T term to T_p/T_e
   evolution. Multi-week project; would unblock the Beaurepaire 7 mJ/cm²
   amplitude target. Plan.md §5.7 carved this out of scope.
6. **Demagnetization field** — multi-week. Unblocks SP4, CoFeB / Permalloy
   / YIG quantitative use, edge effects.

### If you want *new experiments on the current model*

7. **Cross-fluence Zhou validation** — run the F1+F2+F3 calibration script
   over a grid of (T, F) values matching Zhou's published trace family.
   Tighten the fit basin and produce an extrapolation-trust map. ≈ 1 day
   compute, no code change.
8. **Multi-layer thermal coupling** — YIG/FGT heterostructure with
   independent thermal baths and inter-layer thermal coupling. Now genuinely
   physics-correct after F3 (per-layer Beer-Lambert factors actually mean
   what they should in multilayers).
9. **Fluence-response mapping** — sweep fluence × temperature on the
   current model and map the demag-fraction landscape. Good intuition-
   builder.
10. **TR-MOKE readout calibration** — calibrate the `probe_mz` observable
    amplitude so the dashboard reports MOKE rotation angles instead of
    reduced magnetization.

### If you want to *harden what exists*

11. **Host-side adaptive dt** — `thermal_dt_cap` is currently advisory.
    Add a per-step check against pulse windows.
12. **Regression suite as `cargo test`** — collapse the integration gates
    in `examples/` into `tests/` for one-line `cargo test --release`.
13. **Dashboard runtime mode-toggle** — `--grid` is launch-only; runtime
    toggle requires minifb window-recreate logic.
14. **Cross-fluence calibration as a first-class CLI** —
    `--calibrate-zhou` would automate the grid run.

### If you want to *stop here*

The simulator is self-consistent, energy-conservative, documented, and
every deviation from the plan has an ADR. Restarting from head is a
clean handoff — all state is in git, six ADRs, three extension plans
(`plan.md`, `plan-multilayer.md`, `plan-photonic.md`), and three results
docs (`llb_validation.md`, `zhou_fgt_calibration.md`,
`session-closing-assessment.md`).

## 9. Honest self-critique of the F1+F2+F3 work

**What worked well:**
- The "default to old behavior" pattern in F1/F2/F3 (B=0 column = old 1D
  table; tau_fast=0 = collapse to single-stage; skin_depth=0 = uniform-
  mode) preserved every existing regression byte-stable. Adding new
  physics without breaking old physics is hard; this paid off.
- Each F-tier produced a dedicated acceptance probe
  (`test_llb_two_stage`, `test_optical_skin_depth`) that proves the
  collapsed limit is exact AND the new physics is visible. Saved a lot
  of time when verifying composability.
- The Zhou recalibration converged in a single 81-trial grid with
  loss < 0.001. Pre-F1 took 125 trials to reach loss 0.007; F1+F2+F3
  reduced both the trial count and the residual.

**What's still under-constrained:**
- The (`tau_fast`, `R`) trade-off in the Zhou fit. Two aggregate
  observables can't constrain four free parameters; cross-fluence /
  cross-temperature data is the only way to break this degeneracy.
- The chi_par_table buffer has now been bound for three commits without
  being read by any shader code. Either use it (via the Atxitia form) or
  remove it — the inconsistency is starting to feel like dead weight.

**Biggest remaining risk to trust in results:**
The phenomenological longitudinal LLB form remains. F2's two-stage chain
fixes the τ_1/τ_2 *time scales* but the *physical mechanism* in our
shader is still "exponential relaxation of m to m_target" with rates
parameterised by (`α_∥(T) / tau_*_base`). The Atxitia form would be
"effective-field-driven dynamics with longitudinal field
χ_∥⁻¹·(1 − m²/m_e²)·m̂/2". Both can be calibrated to give similar
results in the regime we've fit, but only the latter is mechanistically
correct. Anyone extrapolating beyond the Zhou-fit operating point should
read `docs/zhou_fgt_calibration.md` §6 for the open paths.

---

**End of session.** Next engagement starts from `git log --oneline` and
this document. Recommended first steps for whoever picks it up: read this
doc, then `docs/llb_validation.md`, then `docs/zhou_fgt_calibration.md`,
then ADRs 001–006 in order. That's the complete context for the
M3TM+LLB+substrate+pump-probe+Zhou stack with F1+F2+F3 active at head.
