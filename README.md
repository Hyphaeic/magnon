# Project: Magnonic Clock Simulator

## Summary

Custom Rust/WebGPU solver for the Landau-Lifshitz-Gilbert (LLG) equation,
targeting simulation of nanostructured magnonic time crystal behavior in
van der Waals magnets (Fe3GeTe2).

## Program

`program-physical-substrates` -- simulating nanoscale material physics to
identify computational structures (magnonic clocks) emerging from magnetic
dynamics in 2D vdW ferromagnets.

## Thesis

Magnon Bose-Einstein condensates exhibit spontaneous coherent precession
at well-defined frequencies, breaking continuous time-translation symmetry
(magnonic time crystals). This has been demonstrated in YIG and superfluid
He-3 but not in vdW magnets. FGT's 2D tunability (thickness, gating,
heterostructure engineering) could enable miniaturized frequency references
if the material parameters support it.

This project builds the simulation tool to explore that question.

## Scientific Status

**Speculative.** Magnon BEC has not been demonstrated in any vdW
ferromagnet. FGT's Gilbert damping has never been directly measured by FMR.
Exchange stiffness has a 10x literature scatter. The simulation is
exploratory -- mapping parameter regimes, not predicting outcomes.

## Tech Stack

- **Language:** Rust
- **GPU Backend:** wgpu / WebGPU (WGSL compute shaders)
- **Runtime:** Native + WebAssembly (WASM)
- **Integration:** Heun's method (MuMax3-validated)
- **Visualization:** egui + egui-wgpu

## Key Documents

- `docs/viability-report.md` -- Full viability review with SOTA audit,
  algorithm analysis, material constraints, and build timeline.

## Workspace Dependencies

- `workspace-mech-eng-env` (Rust toolchain, wgpu)

## Status

Charter stage. Viability review complete (2026-04-16).
