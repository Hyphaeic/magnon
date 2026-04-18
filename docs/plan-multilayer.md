# Discrete Multilayer Extension — Implementation Plan

**Document type:** Implementation plan (follows the pattern of `docs/plan.md`)
**Author:** system-architect (from conversation with volition-billy)
**Date:** 2026-04-17
**Status:** Draft — awaiting approval to execute
**Parent project:** `project-magnonic-clock-sim`
**Supersedes:** Phase 6 "Spatially-Varying Parameters" in `docs/plan.md` (this replaces that deferred phase with a higher-priority and structurally similar upgrade)

---

## 0. Metadata & Lookback

### Prior documents (read these first)
- [`docs/plan.md`](./plan.md) — substrate-extension plan, Phase 1-5 complete
- [`docs/viability-report.md`](./viability-report.md) — SOTA and FGT parameter review; flagged multilayer (YIG/FGT) as the most promising clock architecture
- [`docs/usage.md`](./usage.md) — user guide, covers current 2D capabilities
- Registry entity: `corp/registry/entities/project-magnonic-clock-sim.yaml`

### Motivation

The current simulator is strictly 2D: single magnetization vector per (x, y) cell, no z-dimension. The `thickness` parameter is a scaling factor for substrate-contribution averaging, not a third grid axis. This correctly captures thin monolayer dynamics (t ≪ exchange length) but **cannot represent stacked heterostructures** (YIG/FGT hybrid, synthetic AFM, MTJ reference layers).

The viability report specifically called out **YIG/FGT hybrid clocks** as the most promising architectural direction — YIG provides low-loss magnon transport, FGT provides non-linear coupling. This is unreachable without discrete multilayer support.

### Lookback keywords
`Nz dimension` · `interlayer exchange coupling` · `heterostructure stack` · `YIG-FGT hybrid` ·
`synthetic AFM` · `per-layer material parameters` · `z-Laplacian stencil` · `6-neighbor exchange` ·
`layer-dependent anisotropy` · `interlayer DMI` · `RKKY coupling` · `multilayer buffer layout`

### What exists now (baseline)
- 2D LLG solver with full substrate physics (Phases 1-5 of `plan.md`)
- `(BulkMaterial × Substrate × Geometry) → EffectiveParams` decomposition
- Three binaries: `magnonic-sim`, `magnonic-dashboard`, `magnonic-sweep`
- First experiment `experiment-clock-viability-maps` produced validated Q-factor maps

### The structural limit this plan addresses

| Physics | 2D captures | Reason |
|---------|-------------|--------|
| Monolayer precession + damping | ✅ | Film uniform through z |
| Exchange/anisotropy/Zeeman | ✅ | All can be expressed via effective params |
| Interfacial DMI / SOT | ✅ (for thin films) | Surface-injected, averaged into thin film |
| **Stacked heterostructures** | ❌ | Need per-layer state |
| **Interlayer exchange coupling** | ❌ | Need z-direction stencil |
| **Synthetic antiferromagnet** | ❌ | Need two sublattices in z |
| **Standing spin-wave modes through thickness** | ❌ | Need z-resolved magnetization |
| Demagnetization field | ❌ | Separate problem, not addressed here |

**This plan adds discrete multilayer support. It does NOT add demagnetization.** Demag remains deferred (viability report §2.3) as a separate large engineering problem.

---

## 1. Requirements

### Functional

1. The simulator must accept a **stack definition** of N layers, each with its own bulk material and thickness, plus interlayer exchange coupling parameters between adjacent layers.
2. Nz = 1 (single layer) must continue to work and produce byte-identical behavior to the current 2D simulator — no regression.
3. Each layer's magnetization must be independently evolvable under the full LLG + substrate physics from Phases 2-4.
4. **Interlayer exchange coupling** (direct `A_inter` at minimum; RKKY-style `J_inter` optional in future phase) must be a first-class effective field contribution.
5. Substrate effects apply to the **bottom layer only** (bottom-layer-substrate interface); the top surface is treated as free/vacuum. The "substrates" between layers are other layers.
6. Existing binaries must work with backward-compatible defaults. A multilayer config must be specifiable via CLI (e.g., `--stack "fgt-bulk:0.7,yig:2.0"`) or config file.
7. The sweep harness must be extensible to iterate over stack designs (e.g., bilayer material × spacer thickness).
8. The dashboard visualization must support layer selection — display heatmap for any chosen layer.

### Non-functional

1. **Memory:** total GPU buffer memory scales linearly with Nz. At Nx = Ny = 256, Nz = 4, total is ~32 MB of magnetization state. Within budget.
2. **Performance:** compute scales linearly with Nz. For Nz = 2, expect ~2× slowdown vs 2D. Acceptable.
3. **Clarity:** the per-layer physics must be inspectable from `print_summary` output.
4. **Backward compat:** all Phase 1-5 validations must still pass with Nz = 1 default.

### Out of scope for this plan

- **Interlayer DMI.** Different physics, not part of the initial multilayer MVP. Can be added as a follow-up if needed.
- **Demagnetization field.** Separate multi-week problem (viability report §2.3, plan §5.7 critical path).
- **Per-cell (in-plane) material variation.** This was the original deferred Phase 6 of plan.md. Replaced by this plan; can be revisited as a smaller follow-up.
- **RKKY oscillatory coupling vs thickness.** Start with simple direct exchange. RKKY (coupling strength oscillating with spacer thickness) is a published extension.
- **Standing-wave mode analysis on the analysis side.** The simulator will resolve these, but the post-hoc analysis in the Jupyter notebook can be deferred.

---

## 2. Architecture

### Data-layout tier: adding Nz

The current `mag_buf` has `Nx × Ny × 4` f32 elements indexed by `idx = ix * Ny + iy`. Multilayer generalizes this to:

```
cell_idx_3d = (iz * Nx + ix) * Ny + iy      [flat index into 3D buffer]
total_cells = Nx * Ny * Nz
```

Buffer sizes multiply by Nz. All existing storage buffers (mag, mag_pred, h_eff, torque_0, torque_1) grow proportionally. `Nz = 1` recovers the current 2D behavior exactly.

### Type tier: stack definition

Replace the single `BulkMaterial` field in `SimConfig` with a stack:

```rust
pub struct Layer {
    pub material: BulkMaterial,
    pub thickness: f64,           // layer thickness [m]
    pub u_axis: [f64; 3],          // anisotropy axis (can differ per layer)
}

pub struct Stack {
    pub layers: Vec<Layer>,        // ordered bottom → top
    pub interlayer_a: Vec<f64>,    // A_inter at each interface (len = layers.len() - 1) [J/m]
    pub layer_spacing: Vec<f64>,   // dz between adjacent layer centers (len = layers.len() - 1) [m]
}
```

`Substrate` continues to exist but applies only to the bottom of the stack (below `layers[0]`).

New `Geometry`:

```rust
pub struct Geometry {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,            // number of layers (= stack.layers.len())
    pub cell_size: f64,     // in-plane dx [m]
    // layer thickness is per-layer, from Stack
}
```

`EffectiveParams` becomes `EffectiveParams3D` — a Vec of per-layer effective-param records plus the interlayer coupling prefactors.

### GPU tier: shader changes

The `Params` uniform struct grows to hold **per-layer arrays**. WGSL arrays in uniforms need fixed size. We'll cap at `MAX_LAYERS = 4` for Phase M1 (can be raised later by increasing the array sizes and restriding):

```wgsl
struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    cell_count: u32,
    dt: f32,
    gamma: f32,
    stab_coeff: f32,
    cell_size_dx: f32,

    // Per-layer arrays (padded to MAX_LAYERS = 4)
    layer_alpha: array<f32, 4>,
    layer_exchange_pf: array<f32, 4>,
    layer_anisotropy_pf: array<f32, 4>,
    layer_u_axis: array<vec4<f32>, 4>,
    layer_thickness: array<f32, 4>,
    layer_dmi_pf: array<f32, 4>,
    layer_sot_tau_dl: array<f32, 4>,
    layer_sot_tau_fl: array<f32, 4>,
    layer_sigma: array<vec4<f32>, 4>,

    // Interlayer coupling
    interlayer_exchange_pf: array<f32, 4>,   // at interface between layer i and i+1

    // Bottom substrate (from the Substrate entity)
    b_ext: vec4<f32>,
    b_bias: vec4<f32>,
}
```

Shader kernels index their own layer via `iz = idx / (Nx * Ny)` and read per-layer params from the arrays:

```wgsl
fn field_torque_phase0(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.cell_count { return; }
    let iz = idx / (params.nx * params.ny);
    let in_plane_idx = idx % (params.nx * params.ny);

    let m = read_mag(idx);
    let b_eff =
          exchange_in_plane(idx, iz)            // per-layer prefactor
        + exchange_interlayer(idx, iz)          // z-direction coupling (new)
        + anisotropy_field(m, iz)
        + zeeman_field()
        + exchange_bias_field_if_bottom(iz)     // only layer 0
        + dmi_from_mag(idx, iz);
    let t = llg_torque(m, b_eff, iz);
    // ...
}
```

`exchange_interlayer(idx, iz)` reads the adjacent-layer cells (iz±1) at the same (ix, iy) and applies:

```
B_exch_z = (2·A_inter / (Ms · dz²)) · (m_{z-1} + m_{z+1} - 2·m_z)
```

At top/bottom layers, the missing neighbor uses the clamping Neumann BC (same approach as in-plane boundaries).

### Substrate-per-layer model

- Layer 0 (bottom) gets the full `Substrate` contribution in its effective params
- Layers 1..N get only bulk material contributions + interlayer exchange
- Optional Phase M5: per-layer substrate (if we add cap-layer effects), but not MVP

---

## 3. Implementation Phases

### Phase M1 — Restructure for Nz ≥ 1 (3-4 days)

**Goal:** Generalize buffer layout, indexing, and params struct to support Nz layers while preserving Nz = 1 behavior exactly.

**Steps:**

1. Add `Layer`, `Stack` types to `src/config.rs`
2. Extend `Geometry` with `nz` field
3. Refactor `SimConfig` to hold `Stack` instead of `BulkMaterial`
4. Add `stack_default_monolayer()` constructor that produces the current single-layer FGT-bulk default
5. Refactor `EffectiveParams` to `EffectiveParams3D` — per-layer arrays
6. Update `GpuParams` struct: flatten per-layer fields into fixed-size arrays (MAX_LAYERS = 4 initially)
7. Update WGSL `Params` struct to match
8. Update all compute kernels to index via `iz` and read per-layer params
9. Update `reset_*` magnetization methods to stride by layer
10. Update `readback_mag` + `observables` to handle Nz > 1 (initially average over all layers for `avg_mx/y/z`; probe is single cell on a specific layer)

**Acceptance criteria:**
- `magnonic-sim` with no flags produces the same relaxation trajectory as before (Nz = 1 fallback)
- Manually setting Nz = 2 with identical material in both layers and zero interlayer coupling should produce two independent copies of the 2D dynamics
- All binaries build without warnings
- Dashboard default run still works

### Phase M2 — Interlayer exchange (1-2 days)

**Goal:** Activate the z-Laplacian term in the exchange stencil. With nonzero `A_inter`, two layers with different materials should show coupled dynamics.

**Steps:**

1. Add `interlayer_a: f64` and `layer_spacing: f64` fields to `Stack` (Vec per interface)
2. Compute `interlayer_exchange_pf = 2·A_inter / (Ms_layer · dz²)` per interface in `EffectiveParams3D`
3. Add `exchange_interlayer(idx, iz)` helper to `llg.wgsl`
4. Sum its output into `b_eff` in both phase 0 and phase 1 kernels
5. Runtime setter for interlayer coupling

**Acceptance criteria:**
- Two identical-material layers with strong interlayer coupling precess in phase (as if single layer)
- Two identical-material layers with zero interlayer coupling precess independently
- Two different-material layers (YIG-like + FGT-like, coupled): frequency should lie between the two uncoupled frequencies
- Single-layer regression (Nz = 1 with no interlayer term) preserves Phase 1-5 behavior

### Phase M3 — Multilayer init + stack CLI (1 day)

**Goal:** User-facing way to specify a stack. Wire up the CLI and reset functions.

**Steps:**

1. Parse `--stack "MAT1:THICKNESS_NM,MAT2:THICKNESS_NM,..."` in main.rs
2. Parse `--interlayer-a F1,F2,...` and `--layer-spacing NM1,NM2,...`
3. Default: `--stack` absent → single-layer monolayer (current behavior)
4. Update `reset_uniform_z`, `reset_random`, `reset_stripe_domains`, `reset_skyrmion_seed` to handle Nz > 1 (apply same profile to each layer; or take a layer index for per-layer patterns)
5. New reset: `reset_uniform_z_alternating` for stripe-through-z (synthetic AFM init)
6. Update `print_summary` to show the stack as a table of layers with per-layer effective params

**Acceptance criteria:**
- `--stack "yig:2.0,fgt-bulk:0.7" --interlayer-a 1e-11 --layer-spacing 0.5` runs a YIG/FGT bilayer
- `--stack "fgt-bulk:0.7,fgt-bulk:0.7" --interlayer-a -1e-11` runs a synthetic AFM (antiferromagnetic coupling)
- Summary output clearly shows each layer's effective parameters

### Phase M4 — Sweep harness + visualization (2-3 days)

**Goal:** Make the multilayer capability usable from the sweep harness and the dashboard.

**Steps:**

1. Add CLI-level stack-design iteration to `sweep.rs` — the sweep axes can include stack configurations
2. Add per-layer readback support to `observables()` (pick a layer via config; or multi-layer metrics)
3. Update dashboard to show a layer-selector key (e.g., `1`/`2`/`3` to cycle layer heatmap)
4. Dashboard title bar shows current layer + per-layer avg_mz
5. Sweep CSV adds columns: `n_layers`, `stack_description`, `interlayer_a`
6. Optional: dashboard shows all layers as horizontal strips in the heatmap area (more complex, nice-to-have)

**Acceptance criteria:**
- Sweep can iterate over two stacks: `yig:2.0,fgt:0.7` vs `fgt:0.7,yig:2.0` (order matters for substrate effects)
- Dashboard shows layer-0 by default; pressing `2` switches to layer-1 heatmap
- CSV output differentiates multilayer runs from monolayer runs

### Phase M5 — First multilayer experiment (1-2 days)

**Goal:** Exercise the new capability via the HiR experiment flow.

**Steps:**

1. Create `experiment-yig-fgt-hybrid-clock` in `branches/hir/experiments/`
2. Charter: hypothesis that YIG/FGT bilayer outperforms either monolayer at the clock-viability question
3. Registry entity, workspace scope extension, run.sh
4. Notebook: compare YIG/FGT hybrid Q-factors to monolayer-only Q-factors from `clock-viability-maps`
5. Promote winning figure to `artifacts/cross-domain/`

**Acceptance criteria:**
- Bilayer Q-factor measured and compared to both component monolayers
- Results either confirm or refute the hypothesis; either outcome is a scientific result
- ADR-002 records whatever design lesson emerges (e.g., "stack ordering matters / doesn't matter for Q")

---

## 4. Validation Strategy

### Per-phase regression gates (cumulative)

| Test | Phase | Verifies |
|---|---|---|
| Nz = 1 default produces same trajectory as pre-M1 | M1 | No regression |
| Nz = 2 with identical materials + J_inter = 0 gives two decoupled copies | M1 | Layer independence |
| Nz = 2 with strong J_inter gives single-precession-mode coupled dynamics | M2 | Interlayer exchange works |
| Nz = 2, anti-ferro J_inter, initial sync → antiphase relaxation | M2 | Negative coupling works |
| Stack parser correctly maps `"A:1,B:2"` to Layer[A@1nm, B@2nm] | M3 | CLI parsing |
| Dashboard layer-cycle key works | M4 | UI |
| YIG/FGT hybrid experiment produces different Q than FGT alone | M5 | Physics |

### Analytic cross-checks

- Two identical coupled layers: frequency should increase by factor √(1 + 2·J_inter / (A_eff · k²)) or similar, matching known bilayer mode theory
- Synthetic AFM (J_inter < 0): ground state has alternating layers; applied field rotates them differently
- YIG/FGT coupled: lower-frequency mode is YIG-like, higher is FGT-like (avoid crossing near J_inter = 0)

---

## 5. References

### Prior work in this project
- `docs/plan.md` — substrate extension plan (Phases 1-5 complete)
- `docs/viability-report.md` §6 — future work lists vertical stacking / multilayer as "2-3 weeks, structurally different simulator"

### Multilayer micromagnetics literature
- MuMax3 extension for interlayer exchange: github.com/mumax/3 (`InterExchange` module)
- BORIS multilayer documentation: boris-spintronics.uk
- Stiles & McMichael (1999), "Model for exchange bias in polycrystalline ferromagnet-antiferromagnet bilayers," Phys. Rev. B — fundamental theory
- Slonczewski (1993), "Conductance and exchange coupling of two ferromagnets separated by a tunneling barrier," Phys. Rev. B — RKKY/tunneling coupling
- Hoffmann & Bader (2015), "Opportunities at the frontiers of spintronics," Phys. Rev. Appl. — multilayer review

### YIG/FGT hybrid specifically
- Viability report §3.3 flagged YIG as the low-damping transport layer
- Chumak et al. (2022), "Advances in magnetics: roadmap on spin-wave computing," IEEE Trans. Magn. — discusses hybrid structures

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Shader uniform size exceeds 64 KiB with MAX_LAYERS = 4 | Low | Medium | Current struct ~150 bytes × 4 = 600 bytes; plenty of headroom |
| Performance regression at Nz = 1 due to added array indexing | Low | Low | Benchmark; Nz = 1 path should be indistinguishable |
| Interlayer exchange sign convention flip (like DMI in Phase 3) | Medium | Low | Test both signs early; document explicitly |
| Layer-index boundary bugs at top/bottom | Medium | Medium | Same clamping-Neumann as in-plane boundaries; cover in tests |
| Dashboard layer-cycle UX is confusing | Low | Low | One key for cycling is simple; can defer to strip-view if needed |
| Stack CLI parser fragility | Low | Low | Simple format; enforce via early validation |
| Researchers confuse "thickness" in Geometry with layer thickness | High | Low | Remove old `thickness` from Geometry (it was per-layer in the old model too, effectively); document clearly |

---

## 7. Execution Protocol

- Work one phase at a time. Commit at each phase boundary.
- Regression gate after every phase: `magnonic-sim --nx 64 --ny 64 --steps 1000 --interval 100` must produce monotonic relaxation to +z with `|m| = 1` preserved.
- Document design decisions inline in code. If the interlayer coupling sign turns out to match the literature opposite of our derivation (likely, given Phase 3 precedent), record in ADR-002.
- Update this plan as execution proceeds — add "Implementation notes" subsections under each phase as they complete.

### Suggested execution order

**Before Phase M1:** decide on MAX_LAYERS (4 is a reasonable starting point; 8 would double the uniform size but still fit).

**Phase M1 first, review, then M2:** the restructure is the riskiest phase. Validate Nz = 1 matches before adding interlayer physics.

**Phase M5 can be deferred** if intermediate validation reveals issues. The new capability is useful even before a formal experiment is built on top of it.

---

## 8. Appendix — Formula reference

### In-plane exchange (unchanged from Phase 1)
```
B_exch_xy[i,j] = (2·A_layer / (Ms_layer · dx²)) · (Σ_4-nbrs m - 4·m_center)
```

### Interlayer exchange (new)
```
B_exch_z[i,j,k] = (2·A_inter[k↔k±1] / (Ms_layer · dz²)) · (m_{k-1} + m_{k+1} - 2·m_k)
```
with clamping Neumann BC at top (k = Nz - 1) and bottom (k = 0). `A_inter` is per-interface; `Ms_layer` is the layer in question (for asymmetric interfaces, the interpretation matters — use the harmonic mean of adjacent Ms for smoothness across the interface).

### Total effective field (per cell)
```
B_eff[i,j,k] = B_exch_xy[i,j,k] + B_exch_z[i,j,k]
             + B_anis[i,j,k] + B_zeeman
             + {B_bias[i,j,k] if k=0 else 0}
             + B_DMI[i,j,k]
             + ... (other phase terms)
```

### Memory estimate for a 256 × 256 × 4 stack
- mag, mag_pred, h_eff, torque_0, torque_1: 5 × 256² × 4 × 16 bytes = ~40 MB total on GPU
- Params uniform: ~1 KB
- Within the 128 MiB storage buffer limit per binding; within 64 KiB uniform limit

---

**End of plan document.**

Ready to execute Phase M1 on approval.
