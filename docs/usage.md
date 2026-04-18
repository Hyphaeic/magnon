# Magnonic Clock Simulator — Usage Guide

**A Rust/WebGPU parametric explorer for magnonic time crystal behavior in 2D van der Waals magnet heterostructures.**

This guide covers: how to build it, how to run each binary, how to compose design choices, and how to interpret what comes out. For the physics derivations and material parameter literature review, see `docs/plan.md` and `docs/viability-report.md`.

---

## 1. What this simulator does

It answers three classes of questions about thin-film ferromagnet design:

1. **Dynamics:** given a material, substrate, and geometry, how does the magnetization evolve? What frequencies does it precess at? How quickly does it damp?
2. **Stability:** can a specific configuration (uniform state, domain wall, skyrmion) survive in this parameter regime?
3. **Viability:** across a sweep of design choices, which combinations produce a usable magnonic clock (high Q-factor, stable frequency)?

The core is a 2D projected-Heun LLG solver running on wgpu compute shaders. Substrate physics (interfacial anisotropy, spin pumping, exchange bias, interfacial DMI, spin-orbit torque) is separated from bulk material properties and composed via film thickness — the "substrate-aware parametric explorer" design of `docs/plan.md`.

### Three binaries

| Binary | Purpose | Feature required |
|--------|---------|------------------|
| `magnonic-sim` | Headless single run, prints CSV to stdout | (default) |
| `magnonic-dashboard` | Interactive GUI with live heatmap + plot strip | `--features gui` |
| `magnonic-sweep` | Parametric sweep, writes CSV to file | (default) |

---

## 2. Prerequisites & building

### System requirements

- **Rust** 1.85+ (project uses edition 2021)
- **GPU** with Vulkan / Metal / DX12 support and WebGPU-compatible drivers
  - NVIDIA: any recent driver with Vulkan
  - AMD: RADV driver or official
  - Intel: ANV driver on Linux, recent Windows drivers
  - Apple: Metal (all modern Macs)
- **Linux X11/Wayland** if running the dashboard (uses `minifb`)

### Build commands

Project lives at `branches/hir/projects/magnonic-clock-sim/`. From that directory:

```bash
# Headless + sweep only (no GUI)
cargo build --release

# Everything including dashboard
cargo build --release --features gui
```

The release build takes 30–60 seconds from a clean state. Binaries land in `target/release/`.

### Quick sanity check

```bash
./target/release/magnonic-sim --nx 64 --ny 64 --steps 1000 --interval 100
```

Should produce ~12 lines of CSV and finish in under a second. If this works, the GPU is detected and the solver compiles.

---

## 3. Design choices: material × substrate × geometry

Every simulation is parameterized by three tiers, which combine to produce the effective parameters the GPU actually sees. This lets you vary **physical design intent** (which bulk material? on which substrate? how thick?) instead of wrangling unlabeled effective numbers.

### Bulk materials

```bash
./target/release/magnonic-sim --list-materials
```

| Key | Origin | Use for |
|-----|--------|---------|
| `fgt-bulk` (default) | Leon-Brito 2016 bulk crystal | Substrate-free FGT reference |
| `fgt-effective` (alias `fgt-garland`) | Garland 2026 sim values | Reproducing published simulations |
| `fga-te2-bulk` | Fe₃GaTe₂, room-temperature vdW ferromagnet | Room-T operation |
| `cri3-bulk` | Monolayer CrI₃ | Strong PMA, cryogenic |
| `cofeb-bulk` | CoFeB thin-film | PMA workhorse (MTJs, MRAM) |
| `permalloy-bulk` (alias `py`) | Ni₈₀Fe₂₀ | Isotropic, soft, magnonics reference |

### Substrates

```bash
./target/release/magnonic-sim --list-substrates
```

| Key | Contributes |
|-----|-------------|
| `vacuum` (default) | Nothing — baseline isolated film |
| `sio2` | Weak, disordered interface |
| `hbn` | Clean vdW interface, minimal coupling |
| `pt` | Strong interfacial DMI, large spin pumping, SOT via spin Hall |
| `wte2` | Topological substrate, chiral DMI (Wu 2020) |
| `yig` | Insulator, ultra-low damping magnon transport |
| `irmn` | Antiferromagnetic pinning (exchange bias) |

Each substrate carries: surface anisotropy density, DMI density, spin pumping, proximity moment, exchange bias vector, spin Hall angle, field-like torque ratio. These are composed with the bulk material via thickness to yield effective parameters.

### Thickness scaling

Interface effects scale inversely with film thickness:

```
K_u_eff     = K_u_bulk  + K_u_surface / t
α_eff       = α_bulk    + α_pumping          (additive, not thickness-scaled)
D_DMI_eff   = D_DMI_surface                  (MuMax3 convention, already effective)
τ_DL(J_c)   = ℏ·θ_SH·|J_c| / (2·e·Ms·t)      (1/t scaling for SOT)
```

A 0.7 nm monolayer amplifies surface anisotropy ~10× relative to a 7 nm flake. The effect on SOT is similar: thin films respond more strongly to a given current density.

---

## 4. Running a single simulation (`magnonic-sim`)

### Basic invocation

```bash
./target/release/magnonic-sim
```

Without arguments: 256×256 grid of fgt-bulk on vacuum, 10 000 steps. Prints the effective parameter decomposition to stderr and a CSV time series to stdout.

### Key options

```
Design choices:
  --material NAME        (default fgt-bulk)
  --substrate NAME       (default vacuum)
  --thickness NM         (default 0.7)
  --cell-size NM         (default 2.5)
  --nx N / --ny N        (default 256)

Conditions:
  --bx / --by / --bz F   External field components [T]
  --jx / --jy / --jz F   Charge current density [A/m²]
  --dt F                 Timestep [s] (default 1e-13)

Initial state:
  --init MODE            uniform | random | stripe | skyrmion
  --skyrmion-radius F    Skyrmion seed radius [nm]

Material overrides (post-lookup):
  --alpha F              Bulk Gilbert damping
  --a-ex F               Exchange stiffness [J/m]
  --k-u F                Uniaxial anisotropy [J/m³]
  --ms F                 Saturation magnetization [A/m]
  --d-dmi F              DMI [J/m²]
  --theta-sh F           Spin Hall angle

Run control:
  --steps N              Total steps (default 10 000)
  --interval N           Readback interval (default 100)
```

### Output format

stderr: parameter summary + progress notices.
stdout: CSV with one line per readback:

```
step,time_ps,avg_mx,avg_my,avg_mz,min_norm,max_norm,probe_mz
```

| Column | Meaning |
|--------|---------|
| `step` | Cumulative integration step count |
| `time_ps` | Simulation time in picoseconds |
| `avg_mx/y/z` | Spatially averaged magnetization components |
| `min_norm` / `max_norm` | Smallest/largest `|m|` across the grid (should stay at 1.0 ± f32 noise) |
| `probe_mz` | Single-cell MTJ-style readout at grid center |

### Recipe: free relaxation

```bash
./target/release/magnonic-sim --nx 128 --ny 128 --steps 5000 --interval 500
```

A fresh uniform +z state (with 5° random perturbation) relaxing to the anisotropy easy axis. Used for sanity checks.

### Recipe: current-driven dynamics

```bash
./target/release/magnonic-sim \
  --material fgt-effective --substrate pt \
  --jx 5e13 \
  --nx 32 --ny 32 --steps 10000 --dt 1e-15
```

Dramatic SOT-driven dynamics — watch `avg_mx` swing through large values. At these currents τ_DL approaches H_K and you see persistent limit-cycle oscillation (auto-oscillator regime, the basis for spin-torque nano-oscillators).

### Recipe: skyrmion stability test

```bash
./target/release/magnonic-sim \
  --material fgt-effective --substrate wte2 \
  --init skyrmion \
  --nx 128 --ny 128 --steps 20000 --interval 2000
```

Seeds a Néel skyrmion, watches its evolution. `avg_mz` near 0.56 sustained over thousands of steps = skyrmion stable. `avg_mz` climbing to +1 = skyrmion collapsing.

---

## 5. Interactive dashboard (`magnonic-dashboard`)

Requires `--features gui`. Opens a window with the magnetization heatmap live-animated on top of a time-series plot.

### Launch

```bash
./target/release/magnonic-dashboard --nx 128 --ny 128
```

Accepts many of the same design flags as `magnonic-sim` (`--material`, `--substrate`, `--thickness`, `--alpha`, `--bz`, `--dt`, `--nx/ny`).

### Window layout

| Area | Content |
|------|---------|
| Top ~70% | Magnetization heatmap — one pixel per cell, colored by Mz. Blue = Mz=-1, white = Mz=0, red = Mz=+1 |
| Thin middle bar | Color legend dots for the plot lines (red=avg Mz, blue=avg Mx, green=avg My, amber=probe Mz). Turns dark-orange during a pulse |
| Bottom ~30% | Time series plot. Y axis: [-1, +1]. Horizontal lines at 0, ±0.5, ±1 for reference |

The window title shows live stats: step, sim time, avg Mz, probe Mz, alpha, Bz, steps/sec.

### Keyboard controls

| Key | Action |
|-----|--------|
| **Space** | Play / pause |
| **P** | Fire transverse Bx pulse (excites precession) |
| **← / →** | Adjust pulse strength (±0.5 T) |
| **A / Z** | Increase / decrease Gilbert damping α (×1.5 per press) |
| **B / N** | Increase / decrease external Bz (±0.1 T per press) |
| **↑ / ↓** | Double / halve steps-per-frame (simulation speed) |
| **R** | Reset to random magnetization |
| **D** | Reset to stripe domains |
| **U** | Reset to uniform +z |
| **S** | Reset to Néel skyrmion seed |
| **C** | Clear plot history |
| **Esc** | Quit |

### Typical dashboard session

1. Launch with desired material/substrate
2. Press **Space** to start
3. Press **P** to pulse — watch the plot lines oscillate as precession damps
4. Press **Z** several times to drop alpha below 0.001 — precession rings much longer
5. Press **R** to randomize — watch spatial domains form in the heatmap as exchange coupling enforces order
6. Press **S** then **P** to seed a skyrmion and drive it

The heatmap becomes visually interesting with non-uniform initial states or after a strong pulse destabilizes the uniform phase.

---

## 6. Parametric sweeps (`magnonic-sweep`)

This is where the parametric-explorer framing pays off: iterate over design choices, produce CSV, analyze with pandas/Julia/spreadsheet.

### Per-design-point protocol

1. Construct SimConfig from (bulk, substrate, thickness, conditions)
2. Relax system for N_relax steps (default 5 000)
3. Apply transverse Bx pulse for N_pulse steps (default 100 = 10 ps at dt=1e-13)
4. Turn pulse off; sample avg_mx every M steps (default 20) for N_sample samples (default 1 024)
5. FFT the samples → dominant frequency (GHz)
6. Envelope fit → decay time constant (ns)
7. Q-factor = π · f · τ

### Minimal sweep

```bash
./target/release/magnonic-sweep \
  --materials fgt-bulk \
  --substrates vacuum,pt,wte2,yig \
  --thicknesses 0.7,2.1 \
  --output sweep.csv
```

8 design points (1 × 4 × 2 × 1 × 1). Runtime ~10 seconds.

### Full 5D sweep

```bash
./target/release/magnonic-sweep \
  --materials fgt-bulk,fgt-effective,cofeb,permalloy \
  --substrates vacuum,sio2,hbn,pt,wte2,yig,irmn \
  --thicknesses 0.7,2.1,7.0 \
  --bz-values 0.0,0.5 \
  --jx-values 0,1e12,5e12 \
  --nx 32 --ny 32 \
  --num-samples 512 \
  --output full-sweep.csv
```

504 design points. Runtime ~8 minutes on RTX 4060.

### All sweep flags

```
Design axes (comma-separated):
  --materials M1,M2,...       Bulk material names
  --substrates S1,S2,...      Substrate names
  --thicknesses T1,T2,...     Thicknesses [nm]
  --bz-values B1,B2,...       External Bz [T]
  --jx-values J1,J2,...       Current density Jx [A/m²]

Geometry:
  --nx N / --ny N             Grid size (default 32)
  --cell-size F               Cell size [nm] (default 2.5)
  --dt-ps F                   Timestep [ps] (default 0.1)

Measurement protocol:
  --relax-steps N             Pre-pulse equilibration (default 5000)
  --pulse F                   Transverse Bx pulse amplitude [T] (default 1.0)
  --pulse-steps N             Pulse duration in steps (default 100)
  --sample-interval N         Steps between probe samples (default 20)
  --num-samples N             Samples collected after pulse (default 1024)

Output:
  --output / -o PATH          CSV file (default sweep.csv)
```

### CSV columns

```
material, substrate, thickness_nm,
ms_eff, a_eff, k_u_eff, alpha_eff, d_dmi_eff,   ← effective params (what the GPU sees)
bz_T, jx_A_per_m2,                               ← design conditions
freq_GHz, freq_width_GHz,                        ← dominant precession freq + FWHM
q_factor, decay_time_ns,                         ← clock quality metrics
final_value, late_amplitude,                     ← steady-state diagnostics
dt_sample_ps                                      ← sampling interval for reference
```

### Quick analysis in Python

```python
import pandas as pd
df = pd.read_csv("sweep.csv")

# Top-10 designs by Q-factor
df[df.q_factor != float("inf")].nlargest(10, "q_factor")[
    ["material", "substrate", "thickness_nm", "freq_GHz", "q_factor", "decay_time_ns"]
]

# Material × substrate Q heatmap at fixed thickness
pivot = df[df.thickness_nm == 0.7].pivot_table(
    index="material", columns="substrate", values="q_factor"
)
pivot.style.background_gradient(cmap="viridis")
```

### Quick analysis in the shell

```bash
# Q > 100 designs
awk -F, 'NR==1 || ($13 != "inf" && $13+0 > 100)' sweep.csv | column -t -s,

# Best Q for each material
awk -F, 'NR > 1 {
  key = $1
  if ($13 != "inf" && $13+0 > best[key]) { best[key] = $13+0; line[key] = $0 }
} END { for (k in line) print line[k] }' sweep.csv | column -t -s,
```

---

## 7. Interpreting output

### The effective parameter summary

Every run of `magnonic-sim` or `magnonic-dashboard` starts by printing the decomposition to stderr:

```
Effective params (from fgt-bulk bulk + pt substrate @ t=0.70nm):
  Ms_eff    = 3.810e5 = bulk 3.760e5 + prox 5.000e3 [A/m]
  A_eff     = 9.500e-12 [J/m]
  K_u_eff   = 3.317e6 = bulk 1.460e6 + surf/t 1.857e6 [J/m³]
  α_eff     = 0.00700 = bulk 0.00100 + pump 0.00600
  D_DMI_eff = 1.000e-3 J/m² [active]
  θ_SH      = 0.0700, τ_FL/τ_DL = 0.150 [active]
Prefactors: B_exch=7.979 T, B_anis=17.413 T
```

Read this carefully — it tells you exactly what physics is active for this run. Interface-dominated K_u, spin-pumping-enhanced damping, nonzero DMI, and SOT-capable substrate all show up here. Anything labeled `[active]` is being computed in the GPU kernel; anything missing is inactive (zero contribution).

### |m| preservation

The CSV's `min_norm` and `max_norm` columns should stay within ~1e-7 of unity for the entire run. If you see drift beyond that, either:
- dt is too large for the field magnitudes involved (reduce `--dt`)
- There's a bug — report it

Typical healthy range: min_norm ≈ 0.99999988, max_norm ≈ 1.00000012 (f32 rounding).

### Sweep metrics

| Metric | What it means | Typical range for FGT-based designs |
|--------|---------------|-------------------------------------|
| `freq_GHz` | Dominant precession frequency | 5–250 GHz depending on B_anis |
| `freq_width_GHz` | FWHM of the spectral peak | Narrower = more coherent |
| `q_factor` | π · f · τ_decay | Higher = better clock. Real devices target > 10³ |
| `decay_time_ns` | Amplitude e-fold time of free precession | 0.1 ns (lossy) to 100+ ns (clean) |
| `final_value` | Mean of last 10% of samples | ≈0 after damped precession; nonzero = persistent tilt |
| `late_amplitude` | Peak-to-peak swing in the second half | Tiny = damped; large = persistent oscillation |

`q_factor = inf` or `decay_time = inf` means the sweep protocol couldn't detect decay within the sample window — usually because damping is very low (α ≪ 0.001) and the window is short. Run longer (`--num-samples 2048`) to resolve.

---

## 8. Common workflows

### Workflow A: single-design exploration

You have one heterostructure in mind (say fgt-bulk on Pt at 1 nm). Want to understand its dynamics.

```bash
# 1. Inspect effective parameters
./target/release/magnonic-sim --material fgt-bulk --substrate pt --thickness 1.0 \
  --nx 16 --ny 16 --steps 1 2>&1 | head -20

# 2. Interactive visual exploration
./target/release/magnonic-dashboard --material fgt-bulk --substrate pt --thickness 1.0 \
  --nx 128 --ny 128

# 3. Quantitative time-series
./target/release/magnonic-sim --material fgt-bulk --substrate pt --thickness 1.0 \
  --nx 64 --ny 64 --steps 20000 --interval 200 > single-run.csv
```

### Workflow B: thickness scaling study

How does Q-factor change as I vary film thickness for a fixed material × substrate?

```bash
./target/release/magnonic-sweep \
  --materials fgt-bulk \
  --substrates pt \
  --thicknesses 0.35,0.7,1.4,2.1,3.5,7.0,14.0 \
  --output pt-thickness-study.csv
```

7 rows, load in pandas, plot `q_factor vs thickness_nm`.

### Workflow C: material × substrate phase diagram

Which combinations sustain clock behavior at all?

```bash
./target/release/magnonic-sweep \
  --materials fgt-bulk,fgt-effective,cofeb,permalloy,cri3 \
  --substrates vacuum,hbn,pt,wte2,yig \
  --thicknesses 1.0 \
  --output phase-diagram.csv
```

25 rows. Pivot into a material × substrate matrix of Q-factors. Visual heatmap answers "which cells are viable."

### Workflow D: SOT drive threshold

At what current density does the magnetization start oscillating persistently?

```bash
./target/release/magnonic-sweep \
  --materials fgt-effective \
  --substrates pt \
  --thicknesses 0.7 \
  --jx-values 0,1e11,5e11,1e12,5e12,1e13,5e13 \
  --pulse 0.0 --pulse-steps 0 \
  --num-samples 2048 \
  --output sot-threshold.csv
```

Watch `late_amplitude` climb as J_c increases — that's the anti-damping SOT sustaining oscillation.

### Workflow E: skyrmion stability regime

For what (K_u, D) combinations are skyrmions metastable?

This is not yet directly supported by the sweep (which always does uniform-start + pulse). For now, use single-run skyrmion seeding:

```bash
for K in 2e5 5e5 1e6 3e6; do
  for D in 0.5e-3 1e-3 2e-3; do
    ./target/release/magnonic-sim --init skyrmion \
      --material fgt-effective --substrate wte2 \
      --k-u $K --d-dmi $D \
      --nx 128 --ny 128 --steps 50000 --interval 5000 \
      --dt 1e-13 \
      2>/dev/null | tail -1
  done
done
```

Last line's `avg_mz` column tells you whether the skyrmion survived (~0.5-0.7) or collapsed (~1.0).

---

## 9. Extending the system

### Adding a new bulk material

Edit `src/material.rs`:

1. Add a new constructor:
   ```rust
   pub fn cr_te2_bulk() -> Self {
       Self {
           name: "crte2-bulk",
           ms_bulk: 3.2e5,
           a_ex_bulk: 1.5e-12,
           k_u_bulk: 3.0e5,
           alpha_bulk: 0.01,
           gamma: 1.7595e11,
           tc_bulk: 300.0,
           notes: "CrTe2 — above-room-T 2D ferromagnet.",
       }
   }
   ```
2. Add to `lookup()`:
   ```rust
   "crte2" | "crte2-bulk" => Some(Self::cr_te2_bulk()),
   ```
3. Add to `list_all()`.

No other changes needed. `magnonic-sim --material crte2` now works.

### Adding a new substrate

Edit `src/substrate.rs` the same way. Fields accepted:
- `k_u_surface [J/m²]`, `d_dmi_surface [J/m²]`, `alpha_pumping`, `delta_ms_proximity [A/m]`, `b_exchange_bias [T; 3]`, `spin_hall_angle`, `tau_fl_ratio`

### Adding a new initial condition

Edit `src/gpu.rs` — add a `reset_*()` method alongside the existing ones (`reset_uniform_z`, `reset_random`, `reset_stripe_domains`, `reset_skyrmion_seed`). The method writes initial magnetization to `self.mag_buf` via `queue.write_buffer`.

Expose via the CLI in `src/main.rs`'s `--init` handler.

### Adding new physics

The shader `src/shaders/llg.wgsl` has a clean field-sum structure:

```wgsl
let b_eff = exchange_from_mag(idx) + anisotropy_field(m)
          + zeeman_field() + exchange_bias_field() + dmi_from_mag(idx);
```

Add a new `my_field(m, idx)` function, append it to this sum in both `field_torque_phase0` and `field_torque_phase1`. Supporting uniform parameters go in the `Params` struct (matching Rust `GpuParams`). See `docs/plan.md` §3 for the byte-layout protocol.

---

## 10. Troubleshooting

### `No GPU adapter` on launch

The wgpu backend couldn't find a compatible device. Options:

- Linux: install Vulkan drivers (`vulkan-tools` package, `vulkaninfo` should succeed)
- Verify GPU is not exclusively assigned to another process
- Try `WGPU_BACKEND=gl` as a fallback (slower, but works without Vulkan)

### Build fails on `winit 0.30.13`

This is an old issue that Phase 1 avoided by using `minifb` instead of winit+egui. If you hit it while adding dependencies, pin or replace the offending crate.

### |m| drifts above 1e-5

Reduce `--dt`. If problem persists at dt=1e-15, it may be a field-magnitude issue (very large τ_DL from high current). Reduce current or increase damping.

### Sweep is slow

Per-point cost is dominated by:
1. Pipeline creation on solver instantiation (~200 ms)
2. GPU compute (scales with `nx*ny * (relax+pulse+sample*interval)` steps)
3. Readback latency (scales with num_samples)

To speed up:
- Smaller grids (`--nx 16 --ny 16` gives ~8× speedup at same step count)
- Fewer samples (`--num-samples 256` is usually enough for FFT peak detection)
- Fewer relax steps (`--relax-steps 2000`) if you don't need full equilibrium

### Frequency values near FFT bin resolution

If sweep reports `freq_GHz` at a small integer multiple of a base frequency, the FFT may be resolving only the lowest bin (the true precession is below the sample window's resolution). Increase `--num-samples` or `--sample-interval`.

---

## 11. Where to look next

- `docs/plan.md` — the phased implementation plan with the full physics decomposition rules, byte layout, and references
- `docs/viability-report.md` — SOTA audit, FGT literature review, SOT/DMI/material parameter assessment
- `src/shaders/llg.wgsl` — the LLG kernel itself with inline derivations for each field term
- `src/effective.rs` — how `(BulkMaterial + Substrate + Geometry) → EffectiveParams` composition works
- `corp/registry/entities/project-magnonic-clock-sim.yaml` — project registry record

## 12. One-line recipes cheat sheet

```bash
# Default run, just watch it relax
./target/release/magnonic-sim --nx 64 --ny 64 --steps 5000 --interval 500

# See all materials
./target/release/magnonic-sim --list-materials

# See all substrates
./target/release/magnonic-sim --list-substrates

# FGT on Pt heterostructure, full summary
./target/release/magnonic-sim --material fgt-bulk --substrate pt --thickness 1.0 --steps 1 2>&1 | head -20

# Interactive — play with parameters live
./target/release/magnonic-dashboard --material fgt-bulk --substrate wte2

# Skyrmion stability check
./target/release/magnonic-sim --init skyrmion --material fgt-effective --substrate wte2 --nx 128 --ny 128 --steps 20000 --interval 2000 > skx.csv

# Quick 16-point sweep (~10s)
./target/release/magnonic-sweep --materials fgt-bulk --substrates vacuum,pt,wte2,yig --thicknesses 0.7,2.1 -o sweep.csv

# Big parameter sweep (~8min)
./target/release/magnonic-sweep --materials fgt-bulk,fgt-effective,cofeb,permalloy --substrates vacuum,sio2,hbn,pt,wte2,yig,irmn --thicknesses 0.7,2.1,7.0 --bz-values 0,0.5 -o full.csv
```
