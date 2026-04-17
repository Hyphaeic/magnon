# Substrate-Aware Parametric Explorer — Implementation Plan

**Document type:** Implementation plan
**Author:** system-architect (from conversation with volition-billy)
**Date:** 2026-04-17
**Status:** Approved — ready to execute
**Parent project:** `project-magnonic-clock-sim`
**Program:** `program-physical-substrates`

---

## 0. Metadata & Lookback

### Prior documents (read these first)
- [`docs/viability-report.md`](./viability-report.md) — SOTA audit, algorithm selection, FGT parameter assessment, build timeline
- [`README.md`](../README.md) — project overview, scientific status, tech stack
- [`../../../../corp/registry/entities/project-magnonic-clock-sim.yaml`](../../../../corp/registry/entities/project-magnonic-clock-sim.yaml) — registry entity, dependencies, provenance

### Lookback keywords (for conversation/context search)
`baked-in substrate parameters` · `parametric explorer reframe` · `bulk vs interface decomposition`
`thickness scaling for interface density` · `MaterialParams vacuum assumption`
`projected Heun + renormalization` · `pitchfork stabilization dead code`
`h_eff stage scratch not observable` · `B-vs-H convention` · `axis normalization enforcement`
`FGT on substrate X not FGT intrinsic` · `Garland 2026 parameters effective not fundamental`
`clock viability zone sweep` · `heterostructure device simulator` · `surface anisotropy density`
`interfacial DMI Rashba stencil` · `Slonczewski SOT` · `exchange bias pinning field`

### What exists now (baseline)
- Working 2D LLG solver: exchange + uniaxial anisotropy + Zeeman, projected Heun integrator, native + GUI binaries
- GPU: wgpu 27 + WGSL compute, ~12–15k steps/s at 256² on RTX 4060
- Validated against analytic expectations for relaxation and precession
- Code entry points: `src/config.rs` (parameters), `src/gpu.rs` (solver), `src/shaders/llg.wgsl` (kernels), `src/bin/dashboard.rs` (viz)
- Known limitations documented in viability-report §4 and §8

### The substrate problem in one paragraph
The current `MaterialParams` struct (`config.rs:22-35`) treats the ferromagnet as if it were suspended in vacuum. The "FGT default" values come from Garland 2026, which are *effective parameters* measured on a specific FGT-on-substrate sample at a specific thickness and temperature. Using them as if they were intrinsic material constants conflates material identity with experimental context. Two different sweep points in (α, K_u) space could physically correspond to "FGT on hBN" versus "Fe₃GaTe₂ on Pt" — the simulator cannot tell them apart, and neither can the experimenter reading the output. This defeats the parametric-explorer framing: we want to sweep over *design choices*, not over unlabeled effective parameters.

---

## 1. Requirements

### Functional requirements
1. Simulation inputs must express **physical design choices** (bulk material × substrate × geometry), not unlabeled effective parameters.
2. The GPU kernel must continue to receive flat effective parameters (no shader restructure in phase 1).
3. The decomposition must be **physically correct**: bulk contributions are volume densities, interface contributions are surface densities, combined via film thickness.
4. The simulator must produce output that labels each run with its physical design (material name, substrate name, thickness, not just effective numbers).
5. The existing binary interfaces (`magnonic-sim`, `magnonic-dashboard`) must continue to work. Existing behavior — FGT default → +z relaxation — must be preserved byte-for-byte where possible.
6. A new binary (`magnonic-sweep`) must iterate over (material × substrate × thickness × conditions) tuples and produce CSV suitable for post-hoc phase-diagram analysis.

### Non-functional requirements
1. Performance: the restructured code must not regress by more than ~5% on baseline sims.
2. Clarity: every effective parameter on the GPU must be traceable back to its bulk + interface contributions.
3. Extensibility: adding a new bulk material or substrate must require only library additions, no refactor.
4. Validation: every phase must include a regression test that confirms existing FGT-on-vacuum dynamics match the baseline.

### Out-of-scope for this plan
- Temperature / thermal LLG (stochastic Langevin term) — deferred, ~3-4 days when needed
- Vertical stacking / multilayer — deferred, ~2-3 weeks, structurally different simulator
- Full demag via FFT — deferred, see viability-report §2.3 and §7 critical path
- Magnetoelastic / strain coupling — deferred, 2-3 weeks, requires elasticity solver
- Spatially-varying parameters per cell — deferred to Phase 6 (below); listed in future work

---

## 2. Architecture

### The three-tier type hierarchy

The central insight is that **effective parameters decompose into three factors** that a physicist should be able to vary independently:

```
BulkMaterial  (intrinsic material properties — substrate-independent)
    +
Substrate     (substrate-contributed effects — material-independent coupling)
    + scaled by
Geometry      (film thickness, cell size, grid dimensions)
    ↓
EffectiveParams (the flat GPU-visible parameters)
    ↓
GpuParams     (shader-compatible f32 struct — unchanged from current)
```

### Decomposition rules (physics)

| Effective quantity | Bulk contribution | Interface contribution | Combination rule |
|---|---|---|---|
| `Ms` [A/m] | `ms_bulk` | `Δms_proximity` | `Ms_eff = ms_bulk + Δms_proximity` (additive) |
| `A_ex` [J/m] | `a_ex_bulk` | (none in leading order) | `A_eff = a_ex_bulk` |
| `K_u` [J/m³] | `k_u_bulk` | `K_u_surface` [J/m²] | `K_u_eff = k_u_bulk + K_u_surface / t_film` |
| `α` [dimensionless] | `alpha_bulk` | `alpha_pumping` | `α_eff = alpha_bulk + alpha_pumping` (additive) |
| `γ` [rad/(s·T)] | `gamma` | (none) | `γ_eff = gamma` |
| `D_DMI` [J/m²] | (zero in centrosymmetric vdW) | `d_dmi_surface` [J/m²] | `D_eff = d_dmi_surface / t_film` |
| `b_exchange_bias` [T] | (none) | `b_exchange_bias` | `B_bias = b_exchange_bias` |

Thickness-scaling intuition: interfaces contribute a fixed energy per unit area. The *volume-averaged* contribution grows inversely with film thickness. A monolayer (~0.7 nm) amplifies interface effects ~10× relative to a 10-layer flake (~7 nm). This is why monolayer FGT has different effective anisotropy than bulk FGT.

### Data flow

```
User selects (BulkMaterial, Substrate, Geometry)
        │
        ▼
EffectiveParams::from(&bulk, &substrate, &geometry)
        │
        ▼
GpuParams::from_effective(&eff)      ← unchanged shader interface
        │
        ▼
GPU compute pipelines                ← unchanged
        │
        ▼
Observables + labeled run metadata   ← includes material_name, substrate_name, thickness
        │
        ▼
CSV output with design-choice columns
```

### Backward-compatibility strategy

- Keep `MaterialParams` as a deprecated alias or thin wrapper during transition
- `SimConfig::fgt_default(nx, ny)` continues to work; internally constructs `(BulkMaterial::fgt_bulk(), Substrate::vacuum(), Geometry::thin_2d(nx, ny))`
- Existing CLI flags (`--alpha`, `--bz`, `--a-ex`) map to modifying the `EffectiveParams` post-construction OR to `BulkMaterial` overrides — decide per-flag in Phase 1
- Existing `GpuParams` layout is byte-identical; no shader changes in Phase 1

### File layout after Phase 1

```
src/
├── lib.rs
├── config.rs              # SimConfig, Geometry, CLI parsing
├── material.rs            # NEW: BulkMaterial, library presets
├── substrate.rs           # NEW: Substrate, library presets
├── effective.rs           # NEW: EffectiveParams, composition rules
├── gpu.rs                 # GpuSolver (minor: accepts EffectiveParams)
├── main.rs                # magnonic-sim (headless, unchanged behavior)
├── bin/
│   ├── dashboard.rs       # magnonic-dashboard (unchanged)
│   └── sweep.rs           # NEW in Phase 5: magnonic-sweep
└── shaders/
    └── llg.wgsl           # Unchanged through Phase 2; modified Phase 3+
```

---

## 3. Implementation Phases

### Phase 1 — Parameter Restructuring (2-3 days) — NO NEW PHYSICS

**Goal:** Make the bulk-vs-substrate decomposition explicit in code. The GPU sees identical effective parameters it sees today.

**Lookback:** This directly answers the "baked-in substrate parameters" concern raised in conversation. The fix is structural, not physical.

**Steps:**

1. Create `src/material.rs`:
   - `BulkMaterial` struct: `name, ms_bulk, a_ex_bulk, k_u_bulk, alpha_bulk, gamma, tc_bulk, notes`
   - Library constructors (all substrate-independent, from bulk single-crystal measurements where available):
     - `fgt_bulk()` — use Leon-Brito 2016 bulk values (K_u ≈ 1.46×10⁶ J/m³, not the Garland effective value)
     - `fga_te2_bulk()` — Fe₃GaTe₂, Tc ≈ 370K
     - `cri3_bulk()` — CrI₃ monolayer, Tc ≈ 45K
     - `cofeb_bulk()` — CoFeB magnonic workhorse
     - `permalloy_bulk()` — Ni₈₀Fe₂₀, K_u ≈ 0
   - `Display` impl for debugging

2. Create `src/substrate.rs`:
   - `Substrate` struct with fields initially zeroed (vacuum equivalent):
     ```
     name, k_u_surface [J/m²], d_dmi_surface [J/m²],
     alpha_pumping, delta_ms_proximity, b_exchange_bias [3 × T],
     sot_damping_like, sot_field_like
     ```
   - Library constructors (values cited from literature; document source per entry):
     - `vacuum()` — all zeros, reproduces current behavior
     - `sio2()` — weak, disordered interface; minimal contributions
     - `hbn()` — clean vdW interface, weak contributions
     - `pt_heavy_metal()` — strong interfacial DMI (~1 mJ/m²), large spin pumping
     - `wte2()` — topological, chiral DMI (Wu 2020)
     - `yig_insulator()` — magnon transport medium, negligible magnetic effects on FM
     - `irmn_afm()` — exchange bias pinning, moderate
   - `Display` impl

3. Create `src/effective.rs`:
   - `EffectiveParams` struct: flat view (same fields as current MaterialParams + DMI + bias + SOT)
   - `EffectiveParams::from(&BulkMaterial, &Substrate, &Geometry) -> Self`
   - Apply thickness-scaling rules from §2
   - `Display` impl showing the decomposition: "K_u_eff = 1.46e6 (bulk) + 0.05 / 0.7e-9 (interface/t) = 1.53e6 J/m³"

4. Modify `src/config.rs`:
   - Add `Geometry` struct: `thickness_nm, cell_size_nm, nx, ny`
   - Refactor `SimConfig` to hold `(BulkMaterial, Substrate, Geometry, b_ext, stab_coeff, readback_interval, total_steps, probe_idx)`
   - Update `fgt_default()` to use `(fgt_bulk(), vacuum(), Geometry { thickness: 0.7nm, cell: 2.5nm, nx, ny })` — produces the SAME effective parameters as before (since vacuum substrate contributes zero)
   - Update `exchange_prefactor()` and `anisotropy_prefactor()` to operate on `EffectiveParams`
   - Keep `print_summary()` but extend to show the decomposition

5. Modify `src/gpu.rs`:
   - `GpuParams::from_config` calls `EffectiveParams::from(...)` then flattens to f32 layout
   - No buffer layout changes in Phase 1
   - Existing runtime setters (`set_alpha`, `set_exchange_pf`, `set_b_ext`) continue to work — they operate on the flattened f32 values

6. Modify `src/main.rs`:
   - CLI flags for `--material <name>`, `--substrate <name>`, `--thickness <nm>`
   - Existing flags (`--alpha`, `--bz`, `--a-ex`) override bulk values post-lookup
   - `--list-materials` and `--list-substrates` to dump library

**Acceptance criteria:**
- `magnonic-sim` with no flags produces byte-identical CSV output to the pre-refactor version (checked via diff)
- `magnonic-sim --material fgt --substrate vacuum --thickness 0.7` produces identical output to default
- `--list-materials` shows 5 materials, `--list-substrates` shows 7 substrates
- `EffectiveParams::from(...).k_u` correctly computes `bulk + surface/thickness`
- All existing tests pass (there are none yet — add at minimum: one relaxation test, one precession period test, one |m|=1 preservation test)

**Risk checks:**
- f64 → f32 truncation for large surface / small thickness products. With K_u_surface = 1e-3 J/m² and t = 0.7e-9 m, surface contribution = 1.4e6 J/m³ — well within f32 range. Safe.
- Clippy lints on unused library entries (permit via `#[allow]` or `#[cfg(not(test))]`).

---

### Phase 2 — Exchange Bias (1 day)

**Goal:** Use the `Substrate.b_exchange_bias` field that Phase 1 added. Currently zero for all substrates — now becomes a real uniform field contribution.

**Lookback:** Simplest substrate physics; essentially another Zeeman term. Useful for modeling AFM-pinned reference layers (e.g., IrMn/CoFe in MTJ stacks).

**Steps:**

1. In `src/gpu.rs`, extend `GpuParams` with `b_bias_x, b_bias_y, b_bias_z, _pad` (3×f32 + padding). Total struct grows from 80 to 96 bytes.
2. In `src/shaders/llg.wgsl`:
   - Add matching `b_bias` fields to the `Params` struct
   - Create `fn exchange_bias_field() -> vec3<f32> { return vec3(params.b_bias_x, ...); }`
   - Sum into `b_eff` in both `field_torque_phase0` and `field_torque_phase1`
3. Populate `Substrate::irmn_afm()` with a reasonable exchange bias value (~0.02 T equivalent, easy axis along some direction)
4. Add a runtime setter `set_b_bias()` analogous to `set_b_ext()`

**Acceptance criteria:**
- With `--substrate vacuum`, dynamics unchanged from Phase 1
- With `--substrate irmn-afm`, magnetization in zero external field relaxes toward the exchange-bias direction, not the anisotropy easy axis (unless they coincide)
- Dashboard can optionally render the bias direction as an arrow overlay on the heatmap

**Validation:** single-cell analytic check: ωₚ = γ·B_total where B_total = anisotropy + exchange_bias. Compare simulated precession frequency to analytic.

---

### Phase 3 — Interfacial (Rashba) DMI (1-2 days)

**Goal:** Add the Dzyaloshinskii-Moriya interaction term for heavy-metal substrates. Unlocks skyrmions, chiral domain walls, and substrate-induced chirality.

**Lookback:** viability-report §3.3 (FGT DMI values from Wu 2020, Nguyen 2020, Park 2021). Discussed as one of the physics additions most critical for the heterostructure device concept.

**Math:** The interfacial (Rashba-type) DMI energy density for a 2D film with broken inversion symmetry normal to the plane:

```
e_DMI = D · [m_z·(∇·m_∥) - (m_∥·∇)m_z]
      = D · [m_z·(∂m_x/∂x + ∂m_y/∂y) - m_x·∂m_z/∂x - m_y·∂m_z/∂y]
```

Effective field (derived from δe/δm):

```
B_DMI_x = (2D/Ms) · ∂m_z/∂x
B_DMI_y = (2D/Ms) · ∂m_z/∂y
B_DMI_z = -(2D/Ms) · (∂m_x/∂x + ∂m_y/∂y)
```

Discrete stencil (central difference, on square grid with spacing dx):

```
∂m_z/∂x ≈ (m_z[i+1,j] - m_z[i-1,j]) / (2·dx)
∂m_z/∂y ≈ (m_z[i,j+1] - m_z[i,j-1]) / (2·dx)
```

**Steps:**

1. In `EffectiveParams`, add `d_dmi_eff` field computed as `substrate.d_dmi_surface / geometry.thickness`
2. In `src/shaders/llg.wgsl`:
   - Add `dmi_pf: f32` to `Params` (prefactor `2·D_eff / (Ms_eff · 2·dx)` to pre-fold the /2 from central difference)
   - Add `dmi_field_from_mag(idx)` and `dmi_field_from_pred(idx)` helper functions
   - Sum into `b_eff` in both phase 0 and phase 1 kernels
3. In `GpuParams`, add `dmi_pf` field and its byte offset
4. Populate substrate library:
   - `pt_heavy_metal()`: D ≈ 1 mJ/m² (typical Pt/Co interface)
   - `wte2()`: D ≈ 1 mJ/m² (Wu 2020 for WTe₂/FGT)
   - Others: zero or small positive

**Acceptance criteria:**
- With `--substrate vacuum` or any substrate with D = 0: results identical to Phase 2
- With `--substrate pt-heavy-metal` + initial uniform +z: isolated skyrmion seed (manually placed by a reset function) stabilizes instead of collapsing
- Skyrmion size matches approximate analytical prediction: `r ≈ π·A / D` (≈ few nm for typical values)

**Validation:**
- Analytic: domain wall width `π·√(A/K_eff)` vs Bloch wall width with DMI `π·√(A/K_eff - D²/(4A))`
- Compare skyrmion radius at equilibrium to analytical formula
- Regression: Phase 2 tests must still pass with vacuum substrate

**Reference implementations for cross-check:**
- MuMax3 DMI source: github.com/mumax/3/blob/master/cuda/dmibulk.cu
- mumax+ extension: github.com/mumax-plus/mumax-plus (newer DMI variants)

---

### Phase 4 — Spin-Orbit Torque / Slonczewski Form (2-3 days)

**Goal:** Model current-driven dynamics from spin current injected by heavy-metal substrates (Pt, Ta, W). This is the electrical input/output interface for any real device concept.

**Lookback:** Required for the "clock driven and read electrically" framing. Without SOT, the simulator can model dynamics but not the device's electrical interface.

**Math:** Spin-orbit torque in the Slonczewski form:

```
τ_SOT = -γ · [τ_DL · m×(m×σ) + τ_FL · (m×σ)]
```

where:
- `σ = ẑ × ĵ_c` is the spin polarization direction (charge current `ĵ_c` flows in-plane, spin polarization is in-plane and perpendicular to it)
- `τ_DL` is the damping-like torque coefficient, proportional to `θ_SH · J_c · ℏ / (2e·Ms·t)`
- `τ_FL` is the field-like torque coefficient (smaller, usually ~10-30% of τ_DL for Pt, but significant for Ta/W)
- `θ_SH` is the spin Hall angle of the substrate
- `J_c` is the applied charge current density [A/m²]
- `t` is the ferromagnet thickness

**Steps:**

1. Add to `Substrate`:
   - `spin_hall_angle: f64` (θ_SH: 0.07 for Pt, ~0.15 for Ta, ~0.3 for W)
   - `tau_fl_ratio: f64` (τ_FL / τ_DL, typically 0.1-0.3)
2. Add to `SimConfig`:
   - `j_current: [f64; 3]` — charge current vector in A/m²
3. In `EffectiveParams`, compute:
   - `tau_dl_eff = (ℏ · θ_SH · |J_c|) / (2 · e · Ms_eff · thickness)`
   - `sigma_direction = ẑ × ĵ_c.normalize()`
4. In `Params` uniform: add `tau_dl, tau_fl, sigma_x, sigma_y, sigma_z, _pad` (24 bytes)
5. In `llg_torque`:
   - After the LLG precession + damping computation, add `sot_torque`
   - `sot_torque = -γ · [τ_DL · m × (m × σ) + τ_FL · (m × σ)]`
6. Add runtime setters for current direction and magnitude

**Acceptance criteria:**
- With J_c = 0, dynamics identical to Phase 3
- With applied current above threshold on Pt substrate, initial +z magnetization switches to -z (current-induced switching)
- Switching threshold matches approximate analytical prediction from Slonczewski formula

**Validation:**
- Threshold current for switching vs analytical: `J_c^thresh ∝ (α · μ_0 · Ms · t · (H_K + H_ext/2)) / (ℏ · θ_SH / 2e)`
- Compare to published MuMax3 + SOT benchmarks (e.g., Lee 2014 PRL)

**Reference implementations:**
- MuMax3 Slonczewski extension: github.com/mumax/3/blob/master/engine/torque.go (function `Slonczewski`)
- BORIS: SOT documented in boris-spintronics.uk docs

**Physical constants to use:**
- `ℏ = 1.054571817e-34 J·s`
- `e = 1.602176634e-19 C`

---

### Phase 5 — Sweep Harness (2 days)

**Goal:** Make the parametric-explorer framing concrete. A new binary iterates over design choices and produces CSV for post-hoc analysis.

**Lookback:** This was the motivating use case from the start — "Parametric Explorer" vs "Physical Replica" discussion. The output format is what enables phase-diagram analysis.

**Steps:**

1. Create `src/bin/sweep.rs`:
   - CLI: `--materials <list>`, `--substrates <list>`, `--thicknesses <list>`, `--alphas <list>`, `--bz-range <min> <max> <steps>`, `--output <path>`
   - Iterator over Cartesian product of inputs
   - For each design point:
     - Construct `(BulkMaterial, Substrate, Geometry)`
     - Run sim for N steps (from equilibrium, apply test pulse, observe decay)
     - Extract metrics: precession frequency (FFT of probe_mz), Q-factor, decay time constant, final avg_mz, |m| drift
     - Append row to CSV

2. Parallelization: since each design point is independent, use `rayon` for parallel execution with one GpuSolver per thread. Add `rayon = "1.11"` dep.

3. Metrics module (`src/metrics.rs`):
   - `fft_peak_frequency(samples: &[f32], dt: f64) -> f64`
   - `fit_decay_time_constant(samples: &[f32]) -> f64`
   - `q_factor(freq: f64, decay_time: f64) -> f64`
   - `viability_score(...)` — composite metric (tunable weights)

4. CSV schema:
   ```
   material,substrate,thickness_nm,alpha_bulk,alpha_pumping,
   ms_eff,k_u_eff,d_dmi_eff,tau_dl,
   initial_state,pulse_strength,pulse_duration_ps,
   clock_freq_GHz,Q_factor,decay_time_ns,
   final_avg_mz,max_norm_deviation,viability_score
   ```

**Acceptance criteria:**
- Sweeping across default FGT/vacuum at different thicknesses shows thickness-dependent anisotropy in the output
- Sweeping FGT across substrates (vacuum, hBN, Pt, YIG) shows different Q-factors and decay times
- Total sweep time for 100 design points is < 2 minutes on RTX 4060
- CSV loads cleanly into Python/pandas or Julia DataFrames for analysis
- External viz demo: include a `scripts/plot_viability.py` that reads the CSV and produces phase diagrams

---

### Phase 6 — Spatially-Varying Parameters (deferred, 3-5 days when needed)

Listed for completeness but not in immediate scope. Required to model:
- Heterostructure layouts (e.g., "half the grid is on Pt, half on hBN")
- Gradient structures
- Defect patterns

Implementation: replace uniform `exchange_pf`, `anisotropy_pf`, `d_dmi_pf` in the uniform buffer with per-cell storage buffers. Each cell becomes fully independent in material properties. Requires extending buffer count (we have 6 bindings used, limit is 8 per bind group default, so headroom is tight).

Decision point: defer until sweep results from Phases 1-5 reveal whether homogeneous-material simulations are sufficient to answer the clock-viability question.

---

## 4. Validation Strategy

### Per-phase regression suite (to build up)
Each phase adds tests, none ever removed. Cumulative suite at end of Phase 5:

| Test | Phase added | Verifies |
|---|---|---|
| FGT vacuum relaxation to +z | 1 | No regression from baseline |
| Single-cell precession frequency = γB | 1 | LLG torque correctness |
| Single-cell damping time = 1/(αγB) | 1 | Gilbert damping correctness |
| |m|=1 preservation to 1e-7 over 10⁵ steps | 1 | Projected Heun robustness |
| Exchange bias shifts equilibrium | 2 | Bias field integration |
| Skyrmion stability on Pt substrate | 3 | DMI physics |
| Domain wall width with DMI matches analytic | 3 | DMI stencil correctness |
| Current-induced switching threshold matches Slonczewski formula | 4 | SOT physics |
| Thickness scaling: K_u_eff(0.7nm) vs K_u_eff(7nm) differs correctly | 5 | Geometry integration |
| Substrate comparison: YIG vs Pt give different decay times | 5 | Substrate library correctness |

### Baseline comparison (ongoing)
Every phase: `cargo run --release -- --material fgt --substrate vacuum --steps 10000 > current.csv`
Compare to `baseline.csv` captured at end of Phase 1. Allowable drift: <1e-5 in avg_mz at any step (numerical noise only).

### External reference benchmarks (Phase 3-4 cross-checks)
- Run a standard MuMax3 skyrmion problem on their online service, compare to ours
- Compare SOT switching threshold to published values (Lee 2014, Miron 2011)

---

## 5. References

### Project documents
- [`docs/viability-report.md`](./viability-report.md) — full SOTA + algorithm + FGT parameter review
- [`README.md`](../README.md) — project summary
- Registry entity: `corp/registry/entities/project-magnonic-clock-sim.yaml`

### Material-parameter literature (FGT-specific)
- Leon-Brito et al. (2016), "Magnetic microstructure and properties of Fe₃GeTe₂," J. Appl. Phys. 120, 083903 — bulk K_u = 1.46×10⁶ J/m³, the honest bulk value
- Fang et al. (2022), "Magnetic damping anisotropy in FGT," Phys. Rev. B 106, 134409 — bulk α predictions
- Li et al. (2025), "Ultralow Gilbert damping in vdW ferromagnets," PRL (arXiv:2411.12544) — monolayer damping
- Garland et al. (2026), "Thickness-dependent skyrmion evolution in FGT," Adv. Funct. Mater. — effective parameter set used in current code
- Nguyen et al. (2020), "Chiral spin spirals at FGT surface," Nano Lett. 20, 8563 — DMI values
- Wu et al. (2020), "Néel-type skyrmion in WTe₂/FGT," Nature Comms 11, 3860 — interfacial DMI ~1 mJ/m²

### Numerical methods
- Vansteenkiste et al. (2014), "The design and verification of MuMax3," AIP Adv. 4, 107133 — reference implementation of projected Heun for LLG
- Peiris et al. (2025), "Stabilisation of LLG equation via pitchfork bifurcation," Nature Sci. Rep. — alternative to renormalization (dormant in our code)

### SOT physics
- Slonczewski (1996), "Current-driven excitation of magnetic multilayers," JMMM 159, L1 — original formulation
- Lee et al. (2014), "Threshold current for switching of a perpendicular magnetic layer," PRL — benchmark for Phase 4 validation
- Miron et al. (2011), "Perpendicular switching of a single ferromagnetic layer induced by in-plane current injection," Nature 476 — experimental reference

### Code references for cross-check
- MuMax3: github.com/mumax/3 — gold-standard GPU micromagnetics, though CUDA not WebGPU
- mumax+: github.com/mumax-plus/mumax-plus — extensible successor
- BORIS: github.com/SerbanL/Boris2 — comprehensive multi-physics
- MicroMagnetic.jl: github.com/MagneticSimulation/MicroMagnetic.jl — Julia port, best vendor coverage

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase 1 refactor breaks existing binaries | Medium | High | Byte-level regression test vs baseline CSV as a blocking gate |
| f32 precision at small thickness × large surface density | Low | Medium | Document that thicknesses < 0.3 nm may have precision issues; warn at runtime |
| WGSL storage buffer count exhausted (Phase 6) | Medium | Medium | Current: 6/8 used. Phase 3 adds no new buffers. Phase 6 might require packing or bind group split. |
| DMI stencil at boundaries produces spurious accumulation | Medium | Medium | Use same clamping Neumann BC as exchange; validate domain wall at boundary |
| SOT current direction singular for σ_z component | Low | Low | `σ = ẑ × ĵ_c` is zero when current is along z; enforce in-plane current in validation |
| Substrate library values diverge from experimental reports as new papers appear | High | Low | Keep per-substrate citations inline in doc comments; review annually |
| Performance regression from bigger struct | Low | Low | Benchmark each phase; revert if > 5% regression at 256² |

---

## 7. Execution Protocol

When operating on this plan:

1. **Work one phase at a time.** Commit each phase before starting the next.
2. **Regression gate.** Run the cumulative validation suite after every phase. Baseline CSV comparison is a blocking gate.
3. **Document decisions inline.** When a substrate library value is chosen, cite the paper in the doc comment. When a derivation is non-obvious, add the math as a comment.
4. **Flag physics uncertainty.** If a parameter value has >2× scatter in the literature, use the lowest-damping / conservative value and tag with `// TODO: verify` comment.
5. **Update this plan as you go.** Treat this doc as a living record. Add "Implementation notes" subsections under each phase header as actual code lands. Preserve the original plan text for lookback.

### Checkpoint schedule
- After Phase 1: one commit, regression CSV as artifact, status note in registry provenance
- After Phase 2: one commit, bias-shift test result
- After Phase 3: one commit, skyrmion stability screenshot (commit PNG to repo)
- After Phase 4: one commit, switching threshold vs analytic plot
- After Phase 5: one commit, full sweep CSV (< 1 MB, commit inline)

---

## 8. Appendix

### Parameter conversion quick reference

| Have | Want | Formula |
|---|---|---|
| H_eff [A/m] | B_eff [T] | `B = μ₀ · H`, μ₀ = 4π×10⁻⁷ |
| K_u_surface [J/m²] at thickness t | K_u_eff [J/m³] | `K_u_eff = K_u_surface / t` |
| D_DMI_surface [J/m²] at thickness t | D_eff [J/m²] for 2D | Same — effective 2D DMI IS the surface value; thickness scaling enters via Ms·t in the prefactor |
| α_bulk + spin pumping | α_eff | `α_eff = α_bulk + g_eff↑↓ · γ · ℏ / (4π · Ms · t)` — for substrates where spin pumping is characterized by g↑↓ mixing conductance |
| J_c [A/m²], θ_SH | τ_DL [rad·T/s] | `τ_DL = (ℏ · θ_SH · J_c) / (2 · e · Ms · t)` |

### Bulk vs effective values for FGT (for audit purposes)

| Parameter | Current (effective) | Target (bulk) | Source |
|---|---|---|---|
| K_u | 4.0×10⁵ J/m³ | 1.46×10⁶ J/m³ | Leon-Brito 2016 bulk crystal |
| A_ex | 1.4×10⁻¹² J/m | 9.5×10⁻¹² J/m | Leon-Brito 2016 (note: simulation convention uses 1.4e-12) |
| Ms | 3.76×10⁵ A/m | 3.76×10⁵ A/m | Consistent across sources |
| α | 0.01 | 10⁻⁴ to 10⁻² (theory) | Fang 2022, Li 2025 |

Note: the "simulation convention" A_ex ≈ 1.4 pJ/m differs from Leon-Brito's bulk 9.5 pJ/m. This 7× discrepancy is itself one of the literature inconsistencies flagged in viability-report §3.2. Phase 1 should adopt the Leon-Brito bulk value as `fgt_bulk()` default and document the discrepancy.

### Lookback keyword index (for search)

- **vacuum assumption** → §0 "substrate problem", §1 requirements
- **baked-in parameters** → §0 "substrate problem", Appendix "Bulk vs effective"
- **parametric explorer** → §5 phase, §1 requirements
- **bulk-interface decomposition** → §2 architecture, §2 "decomposition rules"
- **thickness scaling** → §2 "decomposition rules", §8 appendix
- **projected Heun** → viability-report §2.1, PRIOR audit conversation
- **stabilization dead code** → llg.wgsl header comment, PRIOR audit
- **h_eff semantics** → llg.wgsl header comment, PRIOR audit
- **B vs H convention** → config.rs header block, PRIOR audit
- **FGT effective not intrinsic** → §0, §8 appendix
- **Garland 2026 values** → viability-report §3.1, Appendix
- **Leon-Brito 2016** → §5, §8 appendix
- **DMI Rashba stencil** → §3 Phase 3
- **Slonczewski SOT** → §3 Phase 4
- **exchange bias** → §3 Phase 2
- **spin pumping** → §2 decomposition, §8 appendix
- **clock viability zone** → §3 Phase 5

---

**End of plan document.**

When continuing from this plan in a new session, the minimum context you need is:
1. This document
2. `docs/viability-report.md`
3. The current state of `src/config.rs`, `src/gpu.rs`, `src/shaders/llg.wgsl`
4. Registry entity at `corp/registry/entities/project-magnonic-clock-sim.yaml`

Begin execution at §3 Phase 1.
