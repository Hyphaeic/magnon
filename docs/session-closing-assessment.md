# Session Closing Assessment — Phases P3 → P5A

**Date:** 2026-04-18
**Baseline:** `pre-thermal-baseline` tag (commit `6e96b24`)
**Head:** commit `e1166b3`
**Commits in session:** 7

---

## 1. Headline

Over one session we took the simulator from a pure LLG micromagnetic solver
(with a coherent-only IFE driver) to a thermally-coupled M3TM + LLB system
with a substrate heat sink, a pump-probe orchestrator, and a calibrated FGT
preset matching Zhou 2025 morphology. Every commit passes regression; the
baseline LLG path is bit-identical byte-for-byte on `min_norm`/`max_norm` vs
the pre-session tag. `cargo build --bins --examples` is warning-free.

Five ADRs (001 – 005) record the design decisions, including four deliberate
deviations from the plan document, each documented with rationale and
consequences.

## 2. Commit train

```
e1166b3  Cleanup: silence dead_code warnings, update older examples for P3 API
be5894c  P5 Option A: FGT Zhou-morphology calibration (limits-aware)
56b0916  Add substrate-sink term g_sub_phonon on the phonon bath
e75d4da  P4: pump-probe sequencer (multi-pulse CLI + sweep axis)
e09c912  P3c: M3TM↔LLB coupling, SP4 proxy, Beaurepaire harness, validation doc
58e0b88  P3b: LLB integrator with phenomenological longitudinal relaxation
c651f06  P3a: Koopmans M3TM solver (non-coupled to LLG torque)
6e96b24  pre-thermal-baseline
```

Size of the session: roughly 3,200 net LOC across 18 modified/new files.

## 3. Gate matrix at head

| Gate | Value | Threshold | Notes |
|---|---|---|---|
| LLG byte-for-byte when thermal disabled | exact match | exact | 202 rows, `min_norm`/`max_norm` cols |
| GPU M3TM vs host reference | 1.0·10⁻⁶ / 9.5·10⁻⁷ (T_e / \|m\|) | 1·10⁻³ | single-cell Ni, 3 ps, 1 mJ/cm² |
| LLB → LLG reduction at T=0 | max 1.5·10⁻⁴ | 1·10⁻³ | deterministic skyrmion, 2000 steps |
| SP4 proxy (skyrmion + transverse B) | mx 1.5·10⁻⁴ / my 1.3·10⁻⁴ / mz 4.5·10⁻⁶ | 1·10⁻³ | literal SP4 requires demag (ADR-004) |
| Energy balance (Beaurepaire, with substrate) | 1.000 ± 0.001 | ±0.05 | Δ(U_e + U_p) + ∫ g_sub · (T_p − T_a) dt |
| LLB demag + recovery morphology (1 mJ/cm² Ni) | |m| floor → 0, recovers to 0.83 @ 10 ps | no NaN | |
| Pump-probe multi-pulse distinct responses | 3 distinct \|avg_mx\| amplitudes | non-degenerate | single / Δ=0.5 ps / Δ=5 ps |
| FGT Zhou morphology fit | demag 86.5 % / recovery 60.1 % | targets 79 % / 55 % | residuals ≤ 0.075 / 0.05 |
| Lib unit tests | 12 / 12 | all | |

All green.

## 4. Capability delta vs pre-session

**Can do now that couldn't before:**
- Per-cell Koopmans M3TM integration with electron + phonon + spin-bath
  dynamics, either as an observability layer (LLB off) or fully coupled to
  the magnetic dynamics (LLB on).
- LLB longitudinal relaxation with a phenomenological single-timescale form
  that reduces cleanly to LLG at T = 0.
- Substrate thermal-sink coupling via per-layer `g_sub_phonon`.
- Pump-probe protocols in the sweep harness, with thermal observables
  (`max_te_k`, `min_m_reduced`, `first_pulse_t_ps`, `total_fluence_mj_cm2`,
  `pulse_count`) recorded per (design × delay) row.
- Two canonical benchmarks as one-line CLI invocations:
  `--benchmark beaurepaire-ni` and `--benchmark mumag-sp4-proxy`.
- One calibrated FGT preset (`fgt_zhou_calibrated`) + five material presets
  total (`ni_m3tm`, `py_m3tm`, `fgt_ni_surrogate`, `fgt_zhou_calibrated`,
  `yig_inert`, `cofeb_m3tm`).

**What didn't change:**
- LLG path is the default — `SimConfig::fgt_default` still produces a valid
  config with `photonic.thermal = None`, and the 4-pipeline LLG loop is
  exactly the pre-session code.
- Pre-existing examples (`test_m2_coupling`, `test_nz2`) produce the same
  trajectories they always did.
- Throughput on the LLG-only path is unchanged on an RTX 4060.

## 5. Performance envelope (RTX 4060)

| Path | Throughput | Multiplier vs LLG |
|---|---|---|
| LLG only (`thermal = None`) | ≈ 13 k steps/s @ 256²×1 grid | 1× |
| LLG + M3TM (LLB off) | ≈ 8–10 k steps/s | 1.3–1.6× |
| Full LLB + M3TM | ≈ 7 k steps/s | 1.9× |

Well inside the plan's "≤ 3× LLG during thermal-active windows" budget.

## 6. Decisions recorded as ADRs

| ADR | Decision | Why |
|---|---|---|
| 001 | `fgt_bulk` (Leon-Brito 2016) as default, not `fgt_effective` (Garland) | Pre-session — scientific honesty in the default preset |
| 002 | Stack order is a first-class user input | Pre-session — multilayer heterostructures |
| 003 | P3a/P3b GPU layout: thermal appended at end; single-shader file; pulse_directions.w reuse; raised storage-buffer limit | Session — minimise plan-vs-code drift risk |
| 004 | µMAG SP4 replaced by a proxy (skyrmion + transverse B @ T = 0) | Session — simulator has no demag field; literal SP4 not reachable |
| 005 | `fgt_zhou_calibrated` preset — morphology fit at shifted operating point | Session — literal T = T_c reproduction blocked by zero-field m_e tables |

## 7. Honest limitations at head

These are **known structural gaps**, documented in-tree, not-bugs:

1. **No demagnetization field** — out of scope since plan.md §5.7. Makes
   µMAG SP4 unreachable in the literal sense. Breaks any workflow where
   shape anisotropy dominates.
2. **No lateral heat diffusion** — single-cell and grid simulations alike
   treat cells as thermally isolated. For fluence calibration at high
   energies (Beaurepaire) this traps heat indefinitely.
3. **Phenomenological LLB** — single longitudinal timescale. Two-stage
   recovery (Zhou's τ₁/τ₂) cannot be resolved simultaneously. The χ_∥
   table is populated but unread; it's the drop-in point for a future
   full-Atxitia LLB upgrade.
4. **Zero-field m_e(T) tables** — makes T = T_c operating points degenerate.
   Field-induced magnetization near T_c is not representable.
5. **No optical-skin-depth absorption profile** — uniform absorption is
   assumed. At 5 nm flakes with 15 nm skin depth this is a ~30 % error in
   deposited volumetric energy.
6. **`a_sf` is inert under the LLB back-coupling path** — follows from the
   M3TM ↔ LLB coupling architecture (P3c). Real influence on demag depth
   requires `enable_llb = false` or a full Atxitia LLB form.

## 8. Where to go next (priority-ordered menu)

### If you want *better physics fidelity*

1. **Field-dependent m_e(T, B) tables** — regenerate Brillouin solution in
   a finite external field; store as 2D table. Unlocks literal T = T_c
   operation. ≈ ½ day.
2. **Two-timescale LLB** — add `tau_long_slow` for a second longitudinal
   relaxation channel; represent the τ₁ / τ₂ split Zhou observes. ≈ 1 day.
3. **Optical skin-depth absorption** — per-cell absorption profile along the
   stack z-axis. Significant at 5 nm thicknesses. ≈ 1 day.
4. **Full Atxitia LLB longitudinal torque** — dimensionalise χ_∥, use as
   effective-field prefactor. Replaces the phenomenological single-time
   form end-to-end. ≈ 2 days (physics + calibration re-run).

### If you want *new experiments on the current model*

5. **Multi-layer thermal coupling** — YIG/FGT heterostructure with
   independent thermal baths, inter-layer thermal coupling term. Enables
   hybrid-clock studies with realistic temperature dynamics.
6. **Fluence-response mapping** — reuse the pump-probe harness to scan
   fluence × temperature and map the demag-fraction landscape. Good for
   understanding parameter sensitivities without further code work.
7. **TR-MOKE readout calibration** — the `probe_mz` observable is
   structurally equivalent to TR-MOKE, but we've never calibrated the
   amplitude. A simple linear-response calibration would let us quote
   MOKE rotation angles instead of reduced magnetization.

### If you want to *harden what exists*

8. **Host-side adaptive dt** — `thermal_dt_cap` is currently advisory. Add
   a per-step check against pulse windows and auto-sub-loop with smaller
   dt when inside. Closes a soft requirement from plan §1 non-functional #5.
9. **Regression suite as `cargo test`** — currently the integration gates
   (SP4 proxy, M3TM vs host, LLB demag, pump-probe) live in `examples/` and
   must be invoked individually. Collapsing them into `tests/` would let
   `cargo test --release` run the full matrix.
10. **Dashboard binding update** — the GUI dashboard (`src/bin/dashboard.rs`)
    has never been touched during P3/P4/P5. It still uses the 6-binding BGL
    layout and will fail at runtime against the current shader. Either
    update or feature-gate it.

### If you want to *stop here*

The simulator is self-consistent, energy-conservative, documented, and every
deviation from the plan has an ADR. Restarting from head is a clean
handoff — all state is in git, 5 ADRs, 3 extension plans (`plan.md`,
`plan-multilayer.md`, `plan-photonic.md`), and two results docs
(`llb_validation.md`, `zhou_fgt_calibration.md`).

## 9. Honest self-critique of the session

**What worked well:**
- Commit-at-boundary discipline made each phase independently reviewable.
- ADRs caught three plan-vs-code contradictions before they calcified into
  silent bugs (GpuParams offset, shader file split, SP4 scope).
- Energy-balance gate at Beaurepaire turned out to be the single most
  important physics check in the session — it's what confirmed the
  substrate-sink term was dimensionally correct.
- The "fit doesn't hit the target, document why" response to the Beaurepaire
  500-fs miss (P3c) and the Zhou literal-T_c gap (P5A) kept the session
  honest — much better than forcing a heroic parameter tune.

**What I'd do differently next time:**
- The calibration grid search over `a_sf` was a wasted 125 trials — I should
  have realized `a_sf` was inert under LLB back-coupling *before* running
  the grid, not after seeing identical observables across rows. Five minutes
  of manual reasoning up-front would have saved six minutes of grid.
- Initial WGSL fields in `GpuParams` are now slightly sprawling (6 per-layer
  vec4s for thermal + 2 globals + 1 set of tables). One more addition and
  the uniform block starts feeling crowded. A struct split would help.
- The `chi_par_table` buffer is bound but unread — I kept it because the
  full-Atxitia LLB is a clean drop-in, but it's technically dead weight at
  head. Worth removing if the full-LLB path isn't on the near-term roadmap.
- The dashboard was never updated. Feature-gating or documenting its
  breakage should have been a checklist item at P3a completion.

**Biggest actual risk to trust in results:**
The LLB longitudinal form is phenomenological and under-motivated. It
reduces to LLG at T=0 cleanly, but the coefficient in front —
`α_∥(T) / tau_long_base` — has no direct derivation from the Atxitia
literature. `tau_long_base` is a fit parameter masquerading as a material
constant. Anyone trying to use this for predictions beyond the calibrated
operating point should read `docs/zhou_fgt_calibration.md` §2 and §6 before
interpreting results.

---

**End of session.** Next engagement can start from `git log --oneline` and
this document. Recommended first steps for whoever picks it up: read this
doc, then `docs/llb_validation.md`, then `docs/zhou_fgt_calibration.md`,
then the five ADRs in order. That's the complete context for the
M3TM+LLB+substrate+pump-probe+Zhou stack at head.
