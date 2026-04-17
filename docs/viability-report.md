# Viability Report: Nanostructured Magnonic Clock Simulator

**Date:** 2026-04-16
**Author:** system-architect (research synthesis)
**Requested by:** volition-billy
**Program:** physical-substrates
**Status:** Viability review complete

---

## Executive Summary

**Verdict: 3-4 month research-engineering project, not a 1-month build.**

A custom Rust/WebGPU LLG solver is technically viable. The software path is clear: Heun's method on wgpu compute shaders, with a 2D MVP achievable in 2-3 weeks. However, the material science is the binding constraint -- Fe3GeTe2 damping has never been directly measured by FMR, exchange stiffness has a 10x literature scatter, and magnon BEC has never been demonstrated in any vdW ferromagnet. The simulation would be exploratory/theoretical, not predictive.

The critical path is: WGSL FFT implementation (no library exists) -> demag convolution -> adaptive timestepping -> WASM integration. A 2D exchange-only MVP bypasses the hardest engineering problem (FFT) and delivers a working simulator in weeks.

---

## 1. State of the Art Audit

### 1.1 Rust Micromagnetics Ecosystem

**No Rust micromagnetic crates exist.** crates.io has zero results for micromagnetic simulation, LLG solving, or spin dynamics. This is a completely open niche.

### 1.2 Existing GPU Micromagnetic Solvers

| Solver | GPU Backend | Language | Notes |
|--------|------------|----------|-------|
| MuMax3 | CUDA | Go + CUDA C | Gold standard. f32. Heun default. |
| mumax+ | CUDA | C++ + CUDA | Extensible successor to MuMax3. |
| BORIS | CUDA | C++ + CUDA | Single + multi-GPU. |
| Spirit | CUDA + OpenMP | C++ | Spin dynamics focused. |
| magnum.af | CUDA, OpenCL, CPU | C++ (ArrayFire) | Most backend-flexible. |
| magnum.np | CUDA/ROCm (PyTorch) | Python | Validated against MuMax3. |
| MicroMagnetic.jl | CUDA, ROCm, oneAPI, Metal | Julia | Broadest GPU vendor coverage. |
| OOMMF | CPU only | C++ + Tcl | NIST reference. |

**No WebGPU port exists for any micromagnetic solver.** This project would be the first.

### 1.3 WebGPU Physics Solvers (Reference Architectures)

| Repo | What | Relevance |
|------|------|-----------|
| dimforge/wgsparkl | MPM physics on wgpu (Rust) | Architecture reference for Rust/wgpu compute |
| scttfrdmn/webgpu-compute-exploration | SPH, MD, wave equation (JS/WGSL + Rust/WASM) | Closest pattern: multi-pass compute, 60fps |
| jeantimex/fluid | SPH + PIC/FLIP on WebGPU | GPU fluid sim architecture |

---

## 2. Algorithm Viability

### 2.1 Recommended Integration Scheme: Heun's Method (RK12)

| Criterion | Assessment |
|-----------|-----------|
| Correctness (|m|=1) | Renormalization keeps norm at machine precision |
| GPU parallelism | All operations embarrassingly parallel per-cell |
| Implementation complexity | 2 stages, minimal storage, no linear solves |
| Production validation | MuMax3 default, validated on muMAG standard problems |
| f32 compatibility | MuMax3 proves f32 is sufficient |
| Adaptive stepping | Built-in error estimate from predictor-corrector |
| Thermal/stochastic support | Compatible |

**Why Heun over alternatives:**
- **RK4**: 4 FFT passes/step vs 2 for Heun. Higher order but 2x compute cost.
- **Adams-Bashforth**: Only 1 FFT/step but untested for LLG in production; stability concerns.
- **Alouges projection**: Geometrically exact but requires sparse linear solve on GPU -- impractical in WGSL.
- **Semi-implicit Crank-Nicolson**: Same sparse solver problem.

**Enhancement (drop-in):** A 2025 paper (Nature Sci. Rep.) proposes a stabilization term `A * m * (1 - |m|^2)` that makes |m|=1 an asymptotically stable fixed point, eliminating renormalization drift. Trivially parallel per-cell addition to the torque computation.

### 2.2 WGSL-Specific Constraints

**f32 precision: Sufficient.** MuMax3 runs entirely in f32 and is validated against standard problems. Relative differences between f32 and f64: ~1e-6 for statics, ~0.04 for dynamics. Mitigations: Kahan summation for global reductions, compute demag kernel on CPU in f64 then truncate.

**WebGPU compute limits (guaranteed defaults):**

| Limit | Value | Impact |
|-------|-------|--------|
| max_compute_invocations_per_workgroup | 256 | Standard; use @workgroup_size(64,1,1) |
| max_compute_workgroup_storage_size | 16 KB | Limits FFT butterfly to ~2048 complex values/pass |
| max_storage_buffers_per_shader_stage | 8 | Use packed vec4 buffers + multiple bind groups |
| max_buffer_size | 256 MiB | ~3.7M cells with Heun solver |
| max_compute_workgroups_per_dimension | 65535 | Up to ~16.7M cells per dispatch |

### 2.3 The Hard Problem: Demagnetization Field FFT

Standard approach (MuMax3/OOMMF): zero-pad to 2N, 3D FFT on 3 components, pointwise kernel multiply, inverse 3D FFT. Total: 6 FFTs per demag evaluation.

**GPU FFT options for WebGPU:**

| Option | Status | Effort |
|--------|--------|--------|
| gpu-fft crate (CubeCL/wgpu) | 1D only, 32 GitHub stars, early | Prototype-viable |
| VkFFT (via Vulkan backend) | Production-quality, no WGSL target | Native-only, needs Rust bindings |
| Custom WGSL Stockham FFT | No existing implementation | 2-4 weeks to implement |
| CPU FFT (rustfft) + upload | Works today | Kills scaling for large grids |
| Direct summation O(N^2) | Trivial on GPU | Only viable for <32^3 grids |

**Recommendation:** Start without demag (exchange-dominated dynamics are physically valid for thin-film nanoscale geometries). Add direct summation for small grids, then GPU FFT when validated.

### 2.4 Memory Budget (Heun Solver)

Per cell: ~72 bytes (60 if material params in uniform buffer).
At 256 MiB max buffer: ~3.7M cells (155^3 grid at 2-5nm cell size = 310-775nm per side).

---

## 3. Material Constraints: Fe3GeTe2

### 3.1 Compiled Parameter Table

| Parameter | Value | Units | Source | T (K) |
|-----------|-------|-------|--------|--------|
| Ms (saturation mag.) | 3.76 x 10^5 | A/m | Leon-Brito (2016) | 5 |
| Ku (uniaxial aniso., bulk) | 1.46 x 10^6 | J/m^3 | Leon-Brito (2016) | 5 |
| Ku (thin flake) | 2.3 x 10^5 | J/m^3 | Nguyen (2020) | 120 |
| A (exchange, bulk expt.) | 9.5 pJ/m | J/m | Leon-Brito (2016) | 5 |
| A (exchange, sim consensus) | 1.0-1.4 pJ/m | J/m | Various sims | -- |
| alpha (theory, monolayer OOP) | < 10^-3 | -- | Li PRL (2025) | low |
| alpha (sim assumed) | 0.01 | -- | Garland (2026) | -- |
| Tc (bulk) | 220-230 | K | Multiple | -- |
| Tc (monolayer) | 75-130 | K | Multiple | -- |
| D (DMI, WTe2/FGT interface) | 1.0 | mJ/m^2 | Wu (2020) | <100 |
| Spin gap | 3.7-3.9 | meV | Calder INS (2019) | low |
| Domain wall width | 2.5 | nm | Leon-Brito (2016) | 5 |
| Exchange length | 2.3 | nm | Leon-Brito (2016) | 5 |

### 3.2 Critical Assessment

**Gilbert damping (alpha): NO direct FMR measurement exists.** Theory predicts ultralow values in ideal monolayers (< 10^-3) with extreme anisotropy (200-10,000% depending on magnetization direction). The simulation community uses alpha = 0.01 as an ungrounded estimate. This is the single most critical parameter for magnon dynamics and it is essentially unconstrained experimentally.

**Exchange stiffness: 10x scatter.** Bulk experimental (9.5 pJ/m from Leon-Brito 2016) vs simulation convention (1.0-1.4 pJ/m). This directly controls magnon dispersion, domain wall width, and critical wavelengths.

**DMI: Highly sample-dependent.** Ranges from ~0 (ideal centrosymmetric) to 1.0 mJ/m^2 (heterostructures). Depends on Fe vacancy concentration, interface chemistry, and thermal history. Not an intrinsic material constant.

**All parameters are stoichiometry-dependent.** Fe vacancies (common, hard to control) shift Tc by 40-70 K and modify all magnetic parameters.

**Cryogenic only.** Bulk Tc ~220 K. All magnonic physics must operate below ~200 K unless ionic gating extends Tc to ~300 K.

### 3.3 Magnon BEC and Time Crystals in vdW Magnets

**Magnon BEC has NOT been demonstrated in FGT or any vdW ferromagnet.**

Magnon BEC is established in:
- YIG (Demokritov et al. 2006, Nature) -- room temperature
- Superfluid 3He-B (Autti et al. 2022, Nature Comms) -- demonstrated as time crystal

Magnonic time crystals specifically:
- Demonstrated in YIG (Divinskiy et al. 2021, PRL) -- space-time crystal in magnon BEC
- Demonstrated in superfluid 3He (Autti et al. 2025, Nature Comms)
- **No papers exist on magnonic time crystals in FGT or any vdW magnet**

The 2024 Magnonics Roadmap (Flebus et al., J. Phys.: Condens. Matter 36, 363501) notes vdW magnets as promising but acknowledges high damping and cryogenic operation as barriers. Fe5GeTe2 spin wave decay length is ~0.7 um vs millimeters in YIG -- three orders of magnitude worse.

### 3.4 Verdict on Material Readiness

**Marginal for predictive simulation. Viable for exploratory/theoretical study.**

A simulation must:
- Treat alpha as a sweep parameter (10^-4 to 10^-2), not a fixed value
- Acknowledge factor-of-2+ uncertainty in exchange stiffness
- State that magnon BEC in FGT is speculative/undemonstrated
- Compare against YIG baseline (where parameters are known to ~1%) to establish what FGT would need to achieve

---

## 4. Hardware/Software Mapping

### 4.1 WASM + WebGPU Data Flow

The hot loop (field computation + time integration) stays entirely on-GPU. The WASM-GPU boundary is crossed only for:
- Initial upload (once)
- Per-step observable readback (scalar/small vector -- amortize by reading every N steps)
- Occasional full-field snapshots for visualization

`queue.writeBuffer()` is the preferred WASM->GPU path (per Chrome WebGPU team). Use `StagingBelt` for reusable buffer pools. Double-buffer readback staging buffers to hide map_async latency.

### 4.2 WASM Memory: Not the Bottleneck

4 GB wasm32 limit is irrelevant -- host side holds only parameter arrays and readback buffers. GPU buffers (where the heavy data lives) are outside WASM linear memory. The real constraint is GPU VRAM.

### 4.3 Native vs Browser Performance Gap

Estimated 10-30% overhead in-browser vs native wgpu for pure compute, primarily from IPC and validation. Acceptable for a research tool.

### 4.4 Recommended Architecture

```
Host (Rust/WASM)                    Device (WGSL Compute)
-----------------                   --------------------
Grid setup                          Buffers:
Parameter init         ------>       magnetization[2] (ping-pong)
Timestep orchestration               effective_field
Observable readback    <------       demag_kernel (readonly)
egui dashboard                       fft_workspace[2]
                                     params (uniform)
                                     observables (readback)

                                    Pipelines:
                                     exchange_field
                                     anisotropy_field
                                     zeeman_field
                                     llg_integrate (Heun stages)
                                     normalize
                                     reduce_observables
                                     (fft_forward, demag_multiply,
                                      fft_inverse -- Phase 2)
```

Web deployment: solver runs in a dedicated Web Worker owning the GPUDevice. Main thread handles egui UI only.

---

## 5. Crate Ecosystem

### Core (required)

| Crate | Version | Role |
|-------|---------|------|
| wgpu | 29.0.1 | GPU abstraction, compute pipelines, dual native+WASM |
| naga | 29.0.1 | WGSL shader compilation/validation (bundled with wgpu) |
| bytemuck | 1.25.0 | Zero-copy GPU buffer marshaling |
| glam | 0.32.1 | Host-side f32 vector math |

### Recommended

| Crate | Role |
|-------|------|
| realfft / rustfft | CPU-side FFT for demag (fallback / validation) |
| egui + egui-wgpu | Native + web visualization dashboard |
| winit | Window management (native) |
| wasm-bindgen | WASM JS interop |

### Optional / Future

| Crate | Role |
|-------|------|
| nalgebra | Higher-precision validation, matrix decompositions |
| CubeCL | Rust-to-GPU kernel generation (alternative to hand-written WGSL) |
| gpu-fft | WGSL FFT (early-stage, 1D only) |

---

## 6. Key References

### Micromagnetic Methods
- Vansteenkiste et al. (2014), "The design and verification of MuMax3," AIP Advances 4, 107133
- Bruckner et al. (2025), "mumax+: extensible GPU-accelerated micromagnetics," arXiv:2411.18194
- Leliaert et al. (2019), "Fast micromagnetic simulations on GPU," J. Phys. D

### LLG Numerics
- Peiris et al. (2025), "Stabilisation of the LLG equation via pitchfork bifurcation," Nature Sci. Rep. (PMC12056204)
- Abert et al. (2019), "Micromagnetics and spintronics: models and numerical methods," Eur. Phys. J. B
- Garcia-Cervera & E (2003), "Numerical micromagnetics: a review"

### Fe3GeTe2 Material Parameters
- Leon-Brito et al. (2016), "Magnetic microstructure and properties of Fe3GeTe2," J. Appl. Phys. 120, 083903
- Fang et al. (2022), "Magnetic damping anisotropy in FGT from first principles," Phys. Rev. B 106, 134409
- Li et al. (2025), "Symmetry-forbidden intraband transitions leading to ultralow Gilbert damping in vdW ferromagnets," PRL (arXiv:2411.12544)
- Garland et al. (2026), "Thickness-dependent skyrmion evolution in FGT," Adv. Funct. Mater.
- Nguyen et al. (2020), "Chiral spin spirals at the surface of FGT," Nano Lett. 20, 8563
- Calder et al. (2019), "Magnetic excitations in Fe3-xGeTe2," arXiv:1812.00519
- Wu et al. (2020), "Neel-type skyrmion in WTe2/FGT heterostructure," Nature Comms 11, 3860

### Magnon BEC and Time Crystals
- Demokritov et al. (2006), "BEC of quasi-equilibrium magnons at room temperature," Nature 443
- Autti et al. (2022), "Nonlinear two-level dynamics of quantum time crystals," Nature Comms 13, 3090
- Divinskiy et al. (2021), "Magnon interaction with driven space-time crystals," PRL 126, 057201
- Flebus et al. (2024), "The 2024 magnonics roadmap," J. Phys.: Condens. Matter 36, 363501
- Newton (2025), "vdW magnets in magnon spintronics" (review)
- Schulz et al. (2023), "Propagating spin waves in Fe5GeTe2," Nano Lett.

### WebGPU / WASM
- Toji (Brandon Jones), "WebGPU Buffer Upload Best Practices" (toji.dev)
- Chrome (2024), "WASM+WebGPU Enhancements for Web AI" (developer.chrome.com)
- scttfrdmn, "WebGPU Compute Exploration" (GitHub)

---

## 7. Build Timeline Assessment

### MVP: 2D Exchange-Only Solver (2-3 weeks)

- 2D grid, no demag, Heun integrator, fixed dt
- Exchange + anisotropy + Zeeman fields
- egui dashboard with magnetization color map
- Native target only

This is a **build**, not research. The patterns are well-established from MuMax3 and existing wgpu compute examples.

### Phase 2: Add Demag + 3D (4-6 weeks additional)

- Direct summation demag for small grids (<32^3)
- OR GPU FFT (2-4 weeks alone for correct 3D FFT in WGSL)
- 3D grid support
- Adaptive timestepping (RK23 or RK45)
- GPU-side error reduction

This is **engineering with research risk** -- the FFT is the hard part.

### Phase 3: Full Solver + WASM (4-6 weeks additional)

- WASM dual-target build
- Web worker architecture
- Browser dashboard
- DMI, thermal noise, multirate integration
- Validation against muMAG standard problems

### Phase 4: FGT-Specific Physics (ongoing research)

- Parameter sweep framework (alpha, A, K as variables)
- Magnon dispersion analysis
- BEC threshold exploration
- Comparison to YIG baseline

**Total to validated, dual-target solver: ~3-4 months.**
**Total to novel FGT magnonic clock results: 4-6+ months** (depends on physics discoveries).

### Critical Path

```
Week 1-3:  MVP (2D, no demag, Heun, native)
Week 4-8:  GPU FFT + 3D demag convolution [CRITICAL PATH]
Week 6-10: Adaptive timestepping + WASM build
Week 8-12: Validation + FGT parameter sweeps
Week 12+:  Magnonic clock-specific physics
```

The GPU FFT is the single hardest engineering problem and gates everything that requires demagnetization. Skipping demag for thin-film geometries (where shape anisotropy can be approximated) is a viable indefinite workaround for exchange-dominated physics.

---

## 8. Recommendation

**Proceed.** This fills a genuine gap (no Rust micromagnetics, no WebGPU micromagnetics anywhere). The software path is clear and the existing hydro-object-sim project proves the team can build Rust/wgpu compute solvers.

However, frame this correctly:
1. The **solver** is an engineering project with known patterns.
2. The **FGT magnonic clock physics** is speculative research -- magnon BEC in vdW magnets is undemonstrated and material parameters have order-of-magnitude uncertainty.
3. The simulation's value is exploratory: mapping what parameter regimes would need to hold for magnonic clock behavior, not predicting whether FGT actually exhibits it.

The project should declare its scientific status as `speculative` and treat material parameters as sweep variables, not constants.
