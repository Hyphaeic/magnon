# Usage guide

Current CLI surface, dashboard keymap, preset libraries.

---

## 1. Binaries

| Binary | Purpose | Build |
|---|---|---|
| `magnonic-sim` | headless solver, dump CSV | `cargo build --release --bin magnonic-sim` |
| `magnonic-sweep` | parametric sweep harness | `cargo build --release --bin magnonic-sweep` |
| `magnonic-dashboard` | interactive viewer | `cargo build --release --bin magnonic-dashboard --features gui` |

Examples (regression-gate / acceptance probes) live in `examples/` and
run with `cargo run --release --example <name>`.

---

## 2. `magnonic-sim` — headless solver

```bash
./target/release/magnonic-sim [options]
```

CSV is written to stdout; status / diagnostics go to stderr.

### 2.1 Geometry / material

```
--nx N                  in-plane cells along x         (default 256)
--ny N                  in-plane cells along y         (default 256)
--cell-size NM          in-plane cell size [nm]         (default 2.5)
--material NAME         single-layer bulk material     (default fgt-bulk)
--substrate NAME        substrate (bottom interface)   (default vacuum)
--thickness NM          first-layer thickness [nm]     (default 0.7)
--stack "M1:T,M2:T,..." multi-layer spec, comma-separated
                        e.g. "yig:2.0,fgt-bulk:0.7"
--interlayer-a F1,F2,…  interlayer A [J/m] (N_layers − 1 values)
--layer-spacing NM,…    per-interface z-spacing [nm]
--probe-layer N         which layer the probe samples  (default 0)
```

Run `--list-materials` and `--list-substrates` to print the catalogue
with their parameters.

### 2.2 Conditions

```
--bx F / --by F / --bz F   external B [T]
--jx F / --jy F / --jz F   charge current density [A/m²] (SOT drive of layer 0)
--dt F                     timestep [s]                (default 1e-14)
--alpha F                  override Gilbert α (first layer)
--a-ex F                   override A_ex
--k-u F                    override K_u
--ms F                     override Ms
--d-dmi F                  override substrate DMI
--theta-sh F               override substrate spin-Hall angle
--init MODE                uniform | random | stripe | skyrmion | alternating
--skyrmion-radius NM       (for --init skyrmion)
```

### 2.3 Run control

```
--steps N           total steps        (default 10000)
--interval N        readback interval  (default 100)
```

### 2.4 Photonic / IFE pulses

```
--pulse "t=T,fwhm=W,peak=B,dir=D[,x=X,y=Y,sigma=S][,fluence=F[,R=Rho]]"
```

Keys:

| Key | Meaning | Default unit |
|---|---|---|
| `t` | pulse centre time | ps |
| `fwhm` | temporal FWHM | fs |
| `peak` | IFE-equivalent peak field | T |
| `dir` | polarisation: `x`,`y`,`z`,`-x`,`-y`,`-z` or `a,b,c` | unit vec |
| `x`, `y` | spot centre | nm |
| `sigma` | 1-σ beam radius (0 = uniform) | nm |
| `fluence` | absorbed peak fluence (engages M3TM source) | mJ/cm² |
| `R` | reflectivity 0..1 | dimensionless |

Up to 4 pulses; max enforced.

### 2.5 Thermal / LLB (Phase P3+)

```
--enable-thermal             attach a default ThermalConfig (M3TM observables)
--enable-llb                 engage LLB integrator (implies --enable-thermal)
--thermal-params-for KEY     ni | py | fgt | fgt-zhou | yig | cofeb (default fgt)
--t-ambient K                ambient bath T (default 300)
--thermal-dt-cap F           timestep cap during pulse window [s] (advisory)
```

### 2.6 Canonical benchmarks

One-line shortcuts:

```
--benchmark beaurepaire-ni     20 nm Ni, 100 fs, 7 mJ/cm², T_amb=300 K, LLB on
--benchmark mumag-sp4-proxy    deterministic skyrmion + 10 mT bias, LLB at T=0
```

Output schema:

```
step,time_ps,avg_mx,avg_my,avg_mz,min_norm,max_norm,probe_mz,max_t_e,max_t_p,min_m_reduced
```

---

## 3. `magnonic-sweep` — parametric sweeps

```bash
./target/release/magnonic-sweep [options] -o sweep.csv
```

### 3.1 Design-axes (comma-separated lists)

```
--materials M1,M2,…              (default fgt-bulk,fgt-effective)
--substrates S1,S2,…             (default vacuum,pt,wte2,yig)
--thicknesses T1,T2,…            in nm (default 0.7,2.1)
--bz-values B1,B2,…              in T  (default 0)
--jx-values J1,J2,…              in A/m² (default 0)
```

Or pin a stack and sweep other axes:

```
--stack-spec "MAT1:T_nm,MAT2:T_nm,..."
--stack-interlayer-a F1,F2,…
--stack-spacing T1,T2,…  (nm)
```

### 3.2 Geometry / protocol

```
--nx / --ny N            grid (default 32)
--cell-size F            cell size [nm] (default 2.5)
--dt-ps F                timestep [ps] (default 0.01)
                         WARNING: legacy default was 0.1 which is unstable
                         for fgt-bulk; corrected to 0.01.
--relax-steps N          equilibration steps (default 5000)
--pulse F                transverse Bx pulse amplitude [T] (default 1.0)
--pulse-steps N          pulse duration [steps] (default 100)
--sample-interval N      steps between probe samples (default 20)
--num-samples N          samples after pulse (default 1024)
```

### 3.3 Pump-probe sequencer (P4)

```
--pump-probe-mode                       enable pump-probe protocol
--pump-t-center PS                      pump centre time [ps] (default 200)
--pump-fwhm FS / --probe-fwhm FS        temporal FWHM [fs]   (default 100)
--pump-peak T / --probe-peak T          IFE peak field [T]    (default 0.5 / 0.1)
--pump-fluence MJ / --probe-fluence MJ  absorbed fluence [mJ/cm²] (engages M3TM)
--pump-reflectivity F / --probe-reflectivity F   reflectivity 0..1
--pump-probe-delay-range START END N    delay axis [ps]
```

### 3.4 Thermal (sweep-level)

```
--enable-thermal             attach ThermalConfig
--enable-llb                 engage LLB (implies --enable-thermal)
--thermal-preset KEY         ni | py | fgt | fgt-zhou | yig | cofeb
--t-ambient K                ambient bath T (default 300)
```

CSV header (extended for pump-probe mode):

```
material,substrate,thickness_nm,n_layers,stack_desc,
ms_eff,a_eff,k_u_eff,alpha_eff,d_dmi_eff,bz_T,jx_A_per_m2,
freq_GHz,freq_width_GHz,q_factor,decay_time_ns,
final_value,late_amplitude,dt_sample_ps,
delay_ps,pulse_count,first_pulse_t_ps,total_fluence_mj_cm2,
max_te_k,min_m_reduced
```

---

## 4. `magnonic-dashboard` — interactive viewer

Requires the `gui` cargo feature and a desktop with X / Wayland:

```bash
cargo build --release --bin magnonic-dashboard --features gui
./target/release/magnonic-dashboard [options]
```

### 4.1 CLI options

All `magnonic-sim` flags work, plus:

```
--grid / -g                  grid layout: all 8 heatmap fields visible at once

--pump-fwhm FS               pump pulse temporal FWHM [fs] (L key fires this)
--pump-peak T                pump IFE peak field [T]
--pump-dir x|y|z|-x|-y|-z    pump polarisation
--pump-fluence MJ_CM2        absorbed fluence (engages M3TM source on L pulse)
--pump-reflectivity F        reflectivity 0..1
--pump-delay PS              fire at t_now + delay (default 1.0)
```

### 4.2 Heatmap fields (cycle with `V`)

| Field | Meaning | Colormap | Range |
|---|---|---|---|
| `Mz` | out-of-plane component | diverging blue-white-red | symmetric auto |
| `Mx` | in-plane x | diverging | symmetric auto |
| `My` | in-plane y | diverging | symmetric auto |
| `\|m\|` | magnitude | viridis-ish | [0, 1] |
| `K-space` | 2D-FFT of (m_x + i·m_y), fft-shifted | hot dB | [-60, 0] dB |
| `Te` | electron temperature [K] | hot | auto, ≥ T_amb |
| `Tp` | phonon temperature [K] | hot | auto, ≥ T_amb |
| `m_red` | M3TM reduced magnetisation | viridis-ish | [0, 1] |

Thermal fields render dim grey when `--enable-thermal` is off. K-space
view always works (uses only the in-plane magnetisation).

### 4.3 Keybindings

| Key | Action |
|---|---|
| `Space` | play / pause |
| `V` | cycle heatmap field (single mode) / focus tile (grid mode) |
| `↑` / `↓` | steps per frame ×2 / ÷2 |
| `Esc` | quit |
| | |
| `P` | fire transverse Bx pulse with current pulse-strength |
| `←` / `→` | pulse strength ∓ 0.5 T |
| `L` | fire IFE laser pulse (uses `--pump-*` defaults) |
| `X` | clear all queued pulses |
| | |
| `T` | toggle thermal M3TM on/off |
| `G` | toggle LLB within thermal |
| `I` | cycle thermal preset (fgt → fgt-zhou → ni → py → cofeb → yig) |
| `[` / `]` | T_ambient ∓ 10 K |
| | |
| `A` / `Z` | α ×1.5 / ÷1.5 |
| `B` / `N` | Bz ± 0.1 T |
| | |
| `R` | reset: random magnetisation |
| `D` | reset: stripe domains (width 8) |
| `U` | reset: uniform +z (5° cone) |
| `S` | reset: Néel skyrmion seed (radius ≈ 1/6 grid size) |
| `K` | reset: alternating ±z per layer (synthetic AFM) |
| `1` / `2` / `3` / `4` | display layer (multilayer only) |
| `C` | clear plot history |

### 4.4 Layout

**Single mode** (default):

```
┌──────────────┬─────────────┐
│              │   status    │
│   heatmap    │   panel     │
│   (active    │             │
│    field)    │   t, step,  │
│              │   integ,    │
│              │   thermal,  │
│              │   range,    │
│              │   ctrls     │
├──────────────┴─────────────┤
│   label bar (legend)       │
├────────────────────────────┤
│   plot strip (mx,my,mz,    │
│      probe_mz history)     │
└────────────────────────────┘
```

**Grid mode** (`--grid`):

```
┌────────┬────────┬────────┬────────┬───────────┐
│  Mz    │  Mx    │  My    │ |m|    │  status   │
├────────┼────────┼────────┼────────┤  panel    │
│  Te    │  Tp    │ m_red  │ |M(k)|²│           │
├────────┴────────┴────────┴────────┤           │
│   label bar                       │           │
├───────────────────────────────────┤           │
│   plot strip                      │           │
└────────────────────────────────────────────────┘
```

Window dims scale with `nx, ny`. Thermal tiles render dim grey when
thermal is off (so the layout stays consistent).

---

## 5. Material catalogue

### 5.1 Bulk materials (`material.rs`, `--material` flag)

| Key | Ms (A/m) | A (J/m) | K_u (J/m³) | α | T_c (K) | Notes |
|---|---|---|---|---|---|---|
| `fgt-bulk` (default) | 3.76·10⁵ | 9.5·10⁻¹² | 1.46·10⁶ | 0.001 | 220 | Leon-Brito 2016 single crystal |
| `fgt-effective` | 3.76·10⁵ | 1.4·10⁻¹² | 4·10⁵ | 0.01 | 220 | Garland 2026 effective |
| `fga-te2-bulk` | 5.5·10⁵ | 2·10⁻¹² | 3·10⁵ | 0.005 | 370 | Fe₃GaTe₂ (50% uncertainty) |
| `cri3-bulk` | 1.4·10⁵ | 0.75·10⁻¹² | 3·10⁵ | 0.001 | 45 | Ising-like 2D, cryogenic |
| `cofeb-bulk` | 1.2·10⁶ | 1.5·10⁻¹¹ | 4.5·10⁵ | 0.006 | 1100 | Thin-film MTJ workhorse |
| `yig-bulk` | 1.4·10⁵ | 3.65·10⁻¹² | 0 | 3·10⁻⁵ | 560 | Ferrimagnet, ultra-low damping |
| `permalloy-bulk` (`py`) | 8·10⁵ | 1.3·10⁻¹¹ | 0 | 0.008 | 870 | Ni₈₀Fe₂₀ |

### 5.2 Substrates (`substrate.rs`, `--substrate` flag)

| Key | Description |
|---|---|
| `vacuum` (default) | inert, no contributions |
| `pt` | Pt with strong SOT (θ_SH ~ 0.1) and proximity effect |
| `pd` | Pd, milder SOT |
| `ta` | Ta, opposite-sign θ_SH |
| `wte2` | WTe₂, exotic Berry-phase-derived torques |
| `yig` | YIG underlayer for spin-pumping into magnonic substrate |
| `mgo` | MgO insulator with K_u_surface contribution |
| `irmn` | IrMn antiferromagnet, exchange-bias provider |

`--list-substrates` prints the full numeric set.

### 5.3 Thermal presets (`material_thermal.rs`, `--thermal-preset`)

| Key | T_c (K) | a_sf | g_ep (W/m³K) | Source |
|---|---|---|---|---|
| `ni` | 627 | 0.185 | 8·10¹⁷ | Koopmans 2010 verified |
| `py` (= `permalloy`) | 870 | 0.08 | 1·10¹⁸ | Battiato 2010 |
| `fgt` (= `fgt-ni-surrogate`) | 220 | 0.185 | 5·10¹⁷ | Ni surrogate; uncalibrated for FGT |
| `fgt-zhou` (= `fgt-calibrated`) | 210 | 0.185 | 5·10¹⁷ | Zhou 2025 morphology fit, ADR-006 |
| `yig` (= `yig-inert`) | 560 | 0 | 0 | Insulator; no M3TM dynamics |
| `cofeb` | 1100 | 0.15 | 1.1·10¹⁸ | Sato 2018 |

The `fgt-zhou` preset additionally sets `tau_long_base = 1.0 fs`,
`tau_fast_base = 0.3 fs`, `g_sub_phonon = 3·10¹⁷`,
`optical_skin_depth_m = 18 nm`. See [`calibration.md`](calibration.md)
for the procedure that derived these values.

---

## 6. Common workflows

### 6.1 Reproduce the LLG bit-stable baseline

```bash
./target/release/magnonic-sim --steps 2000 --interval 10 > new_baseline.csv
diff <(cut -d, -f6,7 new_baseline.csv) <(cut -d, -f6,7 reference_baseline.csv)
# (diff should be empty on the min_norm/max_norm columns)
```

### 6.2 Run the canonical benchmarks

```bash
./target/release/magnonic-sim --benchmark beaurepaire-ni --steps 10000
./target/release/magnonic-sim --benchmark mumag-sp4-proxy
```

### 6.3 Reproduce the Zhou calibration

```bash
cargo run --release --example test_zhou_fgt_calibrate
```

(81 trials, ≈13 minutes on RTX 4060.)

### 6.4 Pump-probe sweep with thermal active

```bash
./target/release/magnonic-sweep \
    --pump-probe-mode \
    --pump-probe-delay-range 0.5 10 5 \
    --pump-peak 0.0 --pump-fluence 0.24 --pump-reflectivity 0.5 \
    --enable-thermal --enable-llb --thermal-preset fgt-zhou --t-ambient 210 \
    --materials fgt-bulk --substrates vacuum --thicknesses 0.5 \
    --bz-values 1 --jx-values 0 \
    --nx 1 --ny 1 --cell-size 20 --dt-ps 0.005 \
    --relax-steps 800 --num-samples 256 --sample-interval 5 \
    -o /tmp/zhou_sweep.csv
```

### 6.5 Interactive exploration

```bash
./target/release/magnonic-dashboard --grid --nx 128 --ny 128 \
    --pump-peak 1.0 --pump-fluence 0.5 --pump-fwhm 100
# Inside: Space, then L (fires the configured laser pulse),
# then T (engages thermal), then V (cycles focus tile in grid mode).
```

---

## 7. Run regression / acceptance tests

All tests are deterministic-or-better.

```bash
cargo test --release --lib

# Acceptance probes — each is a self-checking example.
cargo run --release --example test_m3tm_gpu_vs_host    # P3a gate
cargo run --release --example test_sp4_proxy           # P3c gate
cargo run --release --example test_llb_reduces_to_llg  # P3b gate
cargo run --release --example test_llb_ni_demag        # P3 morphology smoke
cargo run --release --example test_pump_probe          # P4 gate
cargo run --release --example test_llb_two_stage       # F2 gate
cargo run --release --example test_optical_skin_depth  # F3 gate
cargo run --release --example test_beaurepaire_ni      # energy-balance gate
cargo run --release --example test_zhou_fgt            # Zhou feasibility
cargo run --release --example test_zhou_fgt_calibrate  # full Zhou refit (long)
```

---

## 8. Where to read next

- [`physics.md`](physics.md) for what's computed and why.
- [`equations.md`](equations.md) for every formula.
- [`calibration.md`](calibration.md) for fitting new materials.
- [`references.md`](references.md) for citations.
