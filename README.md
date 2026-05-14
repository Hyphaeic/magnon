<div align="center">
  <img src="https://src.hyphaeic.com/website/img/logo.png" alt="Hyphaeic" width="80"/>
  
  # MAGNON
  
  [![Hyphaeic](https://img.shields.io/badge/HYPHAEIC-research-41efa4?style=flat-square&labelColor=1a1a1a)](https://github.com/Hyphaeic)
  [![arXiv](https://img.shields.io/badge/arXiv-2506.09499-b31b1b?style=flat-square&labelColor=1a1a1a)](https://arxiv.org/abs/2506.09499)
  [![Tests](https://img.shields.io/badge/tests-220%2B_passing-41efa4?style=flat-square&labelColor=1a1a1a)](https://github.com/Hyphaeic/stok-core)
  [![License](https://img.shields.io/badge/license-HPL-41efa4?style=flat-square&labelColor=1a1a1a)](https://github.com/hyphaeic/hpl)
  
  **A GPU-accelerated micromagnetic and ultrafast-thermal simulator built in Rust, optimized for 2D van-der-Waals ferromagnets like Fe₃GeTe₂. It supports standard LLG and opt-in thermal LLB pathways, complete with optical driving and a real-time visualization dashboard**
  
  [Paper](https://arxiv.org/abs/2506.09499) · [Documentation](docs) · [Examples](examples)
  
  [HPL License](https://github.com/hyphaeic/hpl) · [Local License](LICENSE)

</div>

---

**A GPU-accelerated micromagnetic + ultrafast-thermal simulator targeting
nanostructured 2D van-der-Waals ferromagnets (Fe₃GeTe₂ in particular).
Implements the Landau–Lifshitz–Gilbert (LLG) equation as a default path
and the Landau–Lifshitz–Bloch (LLB) equation with a Koopmans M3TM
thermal-bath coupling as an opt-in path. Adds a phonon→substrate sink
(F2-prep), field-dependent equilibrium magnetisation `m_e(T, B)` (F1),
two-timescale longitudinal LLB with a proxy variable (F2), and
Beer-Lambert per-layer optical absorption (F3).**

> Calibrated against Zhou et al. *Natl. Sci. Rev.* 12, nwaf185 (2025) at the literal experimental operating point (T = T_c = 210 K, B = 1 T,
F = 0.24 mJ/cm²) with loss 0.0002.*

---

## What this is, briefly

- **Default path:** projected Heun LLG on a 2D + multilayer-z grid.
  Bit-stable against pre-thermal baseline.
- **Thermal/LLB path:** opt-in via `--enable-thermal` / `--enable-llb`.
  Per-cell (T_e, T_p, |m|, m_target) state evolved by Koopmans M3TM
  electron+phonon dynamics + a phenomenological two-stage longitudinal
  relaxation toward the field-dependent equilibrium `m_e(T, B)`.
- **Optical drive:** up to 4 simultaneous laser pulses with Gaussian
  spatial + temporal profiles. Pulses can carry coherent IFE field, an
  M3TM thermal source (with absorbed fluence), or both. F3
  Beer-Lambert per-layer absorption profile for multilayer studies.
- **Observability:** dashboard with 8 cyclable heatmap fields including
  live 2D-FFT k-space view, plus a 4×2 grid mode that shows all fields
  simultaneously. Status panel reports time, integrator, thermal state,
  current field range.

## What this *isn't*

This is a **reduced micromagnetic model**, not a general-purpose
simulator like OOMMF or MuMax3. It explicitly omits:

- **Demagnetisation field** — out of scope; magnetostatics deferred.
- **Lateral heat diffusion (∇²T)** — single-cell or grid cells are
  thermally isolated. Multi-cell hot-spot studies trap heat indefinitely.
- **Stochastic thermal LLB (Langevin noise)** — deterministic
  integration only.
- **Full Atxitia LLB longitudinal torque** — the χ_∥-coupled effective-
  field form. The shipped form is phenomenological exponential
  relaxation toward a target. The χ_∥ table is computed and bound on
  the GPU but currently unread.

See [`docs/physics.md`](docs/physics.md) §6 for the full limitations
list and [`docs/calibration.md`](docs/calibration.md) §4 for the trust-
boundary discussion of the calibrated FGT preset.

---

## Quick start

```bash
# Default LLG run, FGT-bulk on vacuum, 256² grid, 10k steps.
cargo run --release --bin magnonic-sim -- --steps 10000 --interval 100

# Beaurepaire Ni canonical benchmark (1-line CLI).
./target/release/magnonic-sim --benchmark beaurepaire-ni

# Interactive dashboard with grid mode (all 8 fields visible).
cargo build --release --bin magnonic-dashboard --features gui
./target/release/magnonic-dashboard --grid --nx 128 --ny 128

# Engage thermal + LLB at runtime via CLI flags.
./target/release/magnonic-dashboard --grid \
    --enable-llb --thermal-preset fgt-zhou --t-ambient 210 --bz 1.0
# Inside the dashboard: Space to run, L to fire IFE pulse, V to cycle fields.
```

[`docs/usage.md`](docs/usage.md) covers the full CLI surface and the
dashboard keymap.

---

## Status at head

All passing:

| Gate | Result | Source |
|---|---|---|
| LLG byte-for-byte when thermal off | exact | `min_norm`/`max_norm` cols |
| GPU M3TM vs host (Ni 1 mJ/cm²) | 1.0e-6 / 9.5e-7 | `examples/test_m3tm_gpu_vs_host.rs` |
| LLB → LLG at T = 0 K | 1.5e-4 (avg_mz residual) | `examples/test_sp4_proxy.rs` |
| Energy balance (Beaurepaire + substrate sink) | 1.000 ± 0.001 | `examples/test_beaurepaire_ni.rs` |
| Pump-probe distinct responses | 3 amplitudes | `examples/test_pump_probe.rs` |
| F1 m_e field-dependence monotone | PASS | lib unit test |
| F2 collapsed = F1 single-stage | gap = 0.0000 | `examples/test_llb_two_stage.rs` |
| F3 Beer-Lambert top/bot split | 360 K @ δ=14 nm | `examples/test_optical_skin_depth.rs` |
| Zhou recalibration at literal T = T_c | demag 77.6%, rec@22ps 55.2%, loss 0.0002 | `examples/test_zhou_fgt_calibrate.rs` |
| 12 lib unit tests | 12/12 | `cargo test --release --lib` |

`cargo build --release --bins --examples` is warning-free.

---

## Architecture in one figure

```
                  ┌──────────────────────────────┐
                  │  SimConfig (host, Rust)      │
                  │  ├ Stack (layers, materials) │
                  │  ├ Substrate                 │
                  │  ├ Geometry (nx, ny, dx)     │
                  │  ├ B_ext, j_current          │
                  │  └ PhotonicConfig            │
                  │     ├ pulses: Vec<LaserPulse>│
                  │     └ thermal: Option<...>   │
                  └──────────┬───────────────────┘
                             │ effective() + GpuParams
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│   GPU buffers (binding group, 12 entries)                        │
│                                                                  │
│   0  params (uniform, 736 B) ────────────────────────────┐       │
│   1  mag      (cell × vec4 m, 16 B/cell)                 │       │
│   2  mag_pred (Heun predictor scratch)                   │       │
│   3  h_eff    (B_eff scratch)                            │       │
│   4  torque_0 / 5 torque_1  (Heun stages)                │       │
│   6  temp_e   (electron T per cell)        ◀─── M3TM ──┐│       │
│   7  temp_p   (phonon  T per cell)         ◀───────────┤│       │
│   8  m_reduced (M3TM |m| or mirror of |mag|) ◀─────────┤│       │
│   9  m_e_table (2D, [T,B], read-only)        ──────────┤│       │
│  10  chi_par_table (2D, [T,B], read-only, F1)──────────┤│       │
│  11  m_target  (F2 proxy, |m| chases this)   ◀─────────┘│       │
│                                                          │       │
└──────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┴───────────────┐
              ▼                              ▼
   ┌────────────────────┐         ┌────────────────────┐
   │  LLG path          │         │  LLB+M3TM path     │
   │  (default,         │         │  (opt-in)          │
   │   thermal=None)    │         │                    │
   │                    │         │  advance_m3tm:     │
   │  4 pipeline pass:  │         │    update Te,Tp,m_target
   │   ft_phase0,       │         │  ft_phase0_llb     │
   │   heun_predict,    │         │  llb_predict       │
   │   ft_phase1,       │         │  ft_phase1_llb     │
   │   heun_correct     │         │  llb_correct       │
   │                    │         │                    │
   │  |m| stays         │         │  |m| varies        │
   │  unit-norm         │         │  with thermal      │
   └────────────────────┘         └────────────────────┘
```

Detail: [`docs/physics.md`](docs/physics.md).
Equations: [`docs/equations.md`](docs/equations.md).

---

## Documentation

| Document | What's in it |
|---|---|
| [`docs/physics.md`](docs/physics.md) | Model, conventions, data flow, F-tier upgrades, limitations |
| [`docs/equations.md`](docs/equations.md) | Master list of every equation the code implements |
| [`docs/usage.md`](docs/usage.md) | CLI flags, dashboard keymap, preset library |
| [`docs/calibration.md`](docs/calibration.md) | Zhou-2025 calibration procedure + how to fit new materials |
| [`docs/references.md`](docs/references.md) | Bibliography (every cited paper) |
| `branches/hir/decisions/ADR-*` | Six architecture-decision records (006 supersedes 005 for the FGT preset) |

---

## Code layout

```
src/
├── lib.rs                 module index
├── config.rs              SimConfig, Stack, Layer, Geometry
├── material.rs            BulkMaterial presets (FGT, Py, YIG, CoFeB, ...)
├── material_thermal.rs    LayerThermalParams presets + 2D Brillouin tables
├── substrate.rs           Substrate presets (vacuum, Pt, WTe2, YIG, ...)
├── effective.rs           (BulkMaterial × Substrate × Geometry) → effective params
├── photonic.rs            LaserPulse, PhotonicConfig, ThermalConfig
├── thermal.rs             Host-side M3TM reference integrator (test oracle)
├── metrics.rs             FFT / Q-factor / decay-time analysis for the sweep harness
├── gpu.rs                 GpuSolver (wgpu), buffer layout, dispatch
├── shaders/llg.wgsl       All compute kernels: LLG, LLB, M3TM, advance_m3tm
├── main.rs                magnonic-sim CLI
└── bin/
    ├── sweep.rs           magnonic-sweep parametric sweep harness
    ├── dashboard.rs       magnonic-dashboard interactive viewer
    └── dashboard_text.rs  5×7 bitmap font + draw helpers
```

`examples/` holds the regression-gate / acceptance probes
(`test_m3tm_gpu_vs_host`, `test_sp4_proxy`, `test_pump_probe`,
`test_llb_two_stage`, `test_optical_skin_depth`,
`test_zhou_fgt_calibrate`, ...).

---

## Build / run / test

```bash
# Headless binaries (default, fast).
cargo build --release --bins

# GUI dashboard (requires X / Wayland and the `gui` feature).
cargo build --release --bin magnonic-dashboard --features gui

# Unit tests.
cargo test --release --lib

# All acceptance probes.
for e in test_m3tm_gpu_vs_host test_sp4_proxy test_llb_reduces_to_llg \
         test_llb_ni_demag test_pump_probe test_llb_two_stage \
         test_optical_skin_depth test_beaurepaire_ni; do
    cargo run --release --example $e
done

# Zhou re-calibration (longest, ~13 min on RTX 4060).
cargo run --release --example test_zhou_fgt_calibrate
```

---

## License

Internal research code; no public license attached.

## Acknowledgements

Closest architectural analog: Boris (Lepadatu 2020). Calibration target:
Zhou et al. (2025). All references in [`docs/references.md`](docs/references.md).
