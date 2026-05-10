# References

Bibliography of every paper, dataset, or external resource cited in
the code, ADRs, or other docs. Organised by topic; cross-referenced
to the section of [`equations.md`](equations.md) or
[`physics.md`](physics.md) that consumes it.

---

## 1. Landau–Lifshitz–Gilbert / micromagnetic foundations

- **Landau, L. D.; Lifshitz, E.** "On the theory of the dispersion of
  magnetic permeability in ferromagnetic bodies." *Phys. Z. Sowjet.* 8,
  153 (1935). Original LLG equation.

- **Gilbert, T. L.** "A phenomenological theory of damping in
  ferromagnetic materials." *IEEE Trans. Magn.* 40, 3443 (2004) —
  retrospective formalisation. Gilbert form of the damping term used
  here.

- **Brown, W. F.** *Micromagnetics*. Wiley (1963). Standard reference
  for the effective-field decomposition (exchange, anisotropy, Zeeman,
  dipolar). We omit dipolar; everything else is canonical.

- **Abert, C.** "Micromagnetics and spintronics: models and numerical
  methods." *Eur. Phys. J. B* 92, 120 (2019).
  [DOI](https://doi.org/10.1140/epjb/e2019-90599-6). Modern
  comprehensive review; cross-check on integration schemes and
  effective-field forms.

- **Vansteenkiste, A. et al.** "The design and verification of MuMax3."
  *AIP Advances* 4, 107133 (2014).
  [DOI](https://doi.org/10.1063/1.4899186). Closest large-scale
  micromagnetic simulator; reference for the Heun integrator
  validation.

- **OOMMF (NIST).** Object Oriented MicroMagnetic Framework.
  https://math.nist.gov/oommf/. Reference solver for µMAG benchmarks;
  cited as the cross-simulator validation target in
  [`physics.md`](physics.md) §6.1.

Used in: [`equations.md`](equations.md) §1, §2.

---

## 2. Microscopic three-temperature model (M3TM)

- **Beaurepaire, E. et al.** "Ultrafast spin dynamics in ferromagnetic
  nickel." *Phys. Rev. Lett.* 76, 4250 (1996). The original
  ultrafast-demagnetisation observation in Ni (40 % drop at 500 fs,
  7 mJ/cm²). Cited as the canonical benchmark target in
  `examples/test_beaurepaire_ni.rs` and
  [`physics.md`](physics.md) §6.2.

- **Koopmans, B. et al.** "Explaining the paradoxical diversity of
  ultrafast laser-induced demagnetization." *Nat. Mater.* 9, 259
  (2010). The microscopic three-temperature model implemented as
  `advance_m3tm`. Single Elliott-Yafet parameter `a_sf` parameterises
  the spin-flip rate.

- **Battiato, M.; Carva, K.; Oppeneer, P. M.** "Superdiffusive spin
  transport as a mechanism of ultrafast demagnetization." *Phys. Rev.
  Lett.* 105, 027203 (2010). Alternative framework; cited for context
  but not adopted here.

- **Pankratova, M. et al.** "Heat-conserving three-temperature model
  for ultrafast demagnetization in nickel." *Phys. Rev. B* 106, 134401
  (2022). Energy-balance refinement we partially adopt via the
  substrate-sink term.

- **Lin, Z.; Zhigilei, L. V.; Celli, V.** "Electron-phonon coupling
  and electron heat capacity of metals under conditions of strong
  electron-phonon nonequilibrium." *Phys. Rev. B* 77, 075133 (2008).
  γ_e and g_ep values for general metals; source of the Ni / Py /
  CoFeB defaults in `material_thermal.rs`.

Used in: [`equations.md`](equations.md) §3.

---

## 3. Landau–Lifshitz–Bloch (finite-temperature LLG)

- **Garanin, D. A.** "Fokker-Planck and Landau-Lifshitz-Bloch equations
  for classical ferromagnets." *Phys. Rev. B* 55, 3050 (1997).
  Original LLB derivation.

- **Atxitia, U.; Chubykalo-Fesenko, O.; Chantrell, R. W.; Nowak, U.;
  Rebei, A.** "Multiscale modeling of magnetic materials: Temperature
  dependence of the exchange stiffness." *Phys. Rev. B* 82, 134440
  (2010). MFA derivation of α_⊥(T) and α_∥(T).

- **Atxitia, U.; Chubykalo-Fesenko, O.** "Ultrafast magnetization
  dynamics rates within the Landau-Lifshitz-Bloch model." *Phys. Rev.
  B* 84, 144414 (2011).
  [DOI](https://doi.org/10.1103/PhysRevB.84.144414). The standard MFA
  forms `α_⊥(T) = α_0·(1 − T/(3·T_c))` and
  `α_∥(T) = α_0·(2T/(3·T_c))` used in `llg.wgsl::alpha_perp` and
  `alpha_par`. The χ_∥-coupled effective-field longitudinal torque
  (eq. 22) is **not** implemented; see [`physics.md`](physics.md) §6.3.

- **Evans, R. F. L. et al.** "Stochastic form of the Landau-Lifshitz-
  Bloch equation." *Phys. Rev. B* 85, 014433 (2012). Stochastic LLB
  (Langevin term); deterministic mode here, stochastic deferred.

- **Lepadatu, S.** "Boris computational spintronics — high
  performance multi-mesh magnetic and spin transport modelling
  software." *J. Appl. Phys.* 128, 243902 (2020). The closest
  architectural analog to this simulator on the GPU side; reference
  for the dispatch ordering of M3TM + LLB kernels.

- **Leliaert, J.; Mulkers, J.; De Clercq, J.; Coene, A.; Dvornik, M.;
  Van Waeyenberge, B.** "Adaptively time stepping the stochastic
  Landau-Lifshitz-Gilbert equation at non-zero temperature." *AIP
  Advances* 7, 125010 (2017). Informs the (currently advisory)
  `thermal_dt_cap` field.

Used in: [`equations.md`](equations.md) §1.2 and §4.

---

## 4. Material parameters

### 4.1 Fe₃GeTe₂ (FGT)

- **Leon-Brito, N. et al.** "Magnetic and electronic properties of the
  Fe₃GeTe₂ single crystal." *J. Appl. Phys.* 120, 083903 (2016).
  Bulk Ms = 3.76·10⁵ A/m, A = 9.5·10⁻¹² J/m, K_u = 1.46·10⁶ J/m³,
  T_c = 220 K. Source of `BulkMaterial::fgt_bulk()`.

- **Garland, A. et al.** *Adv. Funct. Mater.* (2026, projected).
  Effective values used in `BulkMaterial::fgt_effective()` for
  pre-Phase-1 reproduction.

- **Zahn, P. et al.** "Magnetism and electronic structure of
  Fe₃GeTe₂ thin films." *Phys. Rev. B* 105, 014426 (2022).
  Atomic moment, Tc dependence on layer count. Sets `mu_atom_bohr`.

- **Lichtenberg, T.; Zhou, et al.** "Recent FGT thin-film
  characterization studies." Various, 2024–2025. Background for
  thickness-dependent parameter ranges.

- **★ Zhou, Wang et al.** "Acceleration of ultrafast demagnetization
  in van der Waals ferromagnet Fe₃GeTe₂ in high magnetic field."
  *Natl. Sci. Rev.* 12, 7, nwaf185 (2025).
  [URL](https://academic.oup.com/nsr/article/12/7/nwaf185/8142530).
  **Calibration target**: 79 % demag at 22.2 ps under (T = T_c =
  210 K, B = 1 T, F = 0.24 mJ/cm², 150 fs FWHM, 5 nm flake). The
  fit lives in `material_thermal::fgt_zhou_calibrated()`. ADR-006
  documents the recalibration after F1+F2+F3.

Used in: `BulkMaterial::fgt_bulk`, `LayerThermalParams::fgt_*`,
[`calibration.md`](calibration.md).

### 4.2 Other ferromagnets

- **Sato, H. et al.** "Comprehensive study of CoFeB-MgO magnetic
  tunnel junction characteristics with single- and double-interface
  structures." *Phys. Rev. B* 97, 014433 (2018). β = 1.73 Bloch
  exponent, a_sf calibration source for `cofeb_m3tm()`.

- **CoFeB and Permalloy thin-film parameters** are well-tabulated in
  the materials-characterisation literature; the values shipped here
  are mid-range published estimates with explicit notes when α or K_u
  depends on annealing / interface conditions.

- **YIG** (Y₃Fe₅O₁₂) parameters are textbook (e.g., Wigen 1994,
  *Nonlinear Phenomena and Chaos in Magnetic Materials*).

### 4.3 vdW ferromagnets

- **Fei, Z. et al.** "Two-dimensional itinerant ferromagnetism in
  atomically thin Fe₃GeTe₂." *Nat. Mater.* 17, 778 (2018).
  Discovery of monolayer FGT ferromagnetism.

- **Huang, B. et al.** "Layer-dependent ferromagnetism in a van der
  Waals crystal down to the monolayer limit." *Nature* 546, 270
  (2017). CrI₃ context, justifies `cri3_bulk()`.

Used in: motivation discussion; not directly consumed by the model.

---

## 5. Magneto-optics and inverse Faraday effect

- **Pitaevskii, L. P.** "Electric forces in a transparent dispersive
  medium." *Sov. Phys. JETP* 12, 1008 (1961). Original IFE theory.

- **van der Ziel, J. P.; Pershan, P. S.; Malmstrom, L. D.** "Optically-
  induced magnetization resulting from the inverse Faraday effect."
  *Phys. Rev. Lett.* 15, 190 (1965). First experimental IFE in
  diamagnetic glass.

- **Stanciu, C. D. et al.** "All-optical magnetic recording with
  circularly polarized light." *Phys. Rev. Lett.* 99, 047601 (2007).
  Single-pulse all-optical switching in GdFeCo. Background for the
  IFE-driver model.

- **Kimel, A. V. et al.** "Ultrafast non-thermal control of
  magnetization by instantaneous photomagnetic pulses." *Nature* 435,
  655 (2005). IFE dynamics in DyFeO₃.

- **Mangin, S. et al.** "Engineered materials for all-optical
  helicity-dependent magnetic switching." *Nat. Mater.* 13, 286
  (2014). Materials engineering for AOS.

Used in: `LaserPulse` documentation; `B_IFE` calibration discussion.

---

## 6. Spintronics torques

- **Tserkovnyak, Y.; Brataas, A.; Bauer, G. E. W.** "Enhanced Gilbert
  damping in thin ferromagnetic films." *Phys. Rev. Lett.* 88, 117601
  (2002). [DOI](https://doi.org/10.1103/PhysRevLett.88.117601).
  Standard spin-pumping derivation; the `1/(Ms·t)` thickness scaling
  we **don't** implement (cited as a known limitation in
  [`physics.md`](physics.md) §6.8).

- **Slonczewski, J. C.** "Current-driven excitation of magnetic
  multilayers." *J. Magn. Magn. Mater.* 159, L1 (1996). Original
  spin-transfer torque formulation; basis for the SOT terms in
  `effective.rs::sot_coefficients`.

- **Liu, L. et al.** "Spin-torque switching with the giant spin Hall
  effect of tantalum." *Science* 336, 555 (2012). Spin-Hall-effect
  derivation of the τ_DL coefficient.

- **Manchon, A. et al.** "Current-induced spin-orbit torques in
  ferromagnetic and antiferromagnetic systems." *Rev. Mod. Phys.* 91,
  035004 (2019). Comprehensive SOT review.

Used in: [`equations.md`](equations.md) §2.7.

---

## 7. Interlayer exchange and DMI

- **Richardson, D. C. et al.** "Interlayer Exchange Coupling in
  Magnetic Hard-Soft Bilayered Structures." *Phys. Rev. Applied* 11,
  044016 (2019).
  [DOI](https://doi.org/10.1103/PhysRevApplied.11.044016). Reference
  for proper interfacial-IEC formulation; we use a bulk-like
  z-Laplacian approximation (cited as a known deviation in
  [`physics.md`](physics.md) §6.5).

- **Bruno, P.** "Theory of interlayer magnetic coupling." *Phys. Rev.
  B* 52, 411 (1995). Theoretical basis for IEC oscillation with
  spacer thickness; not consumed by our (non-RKKY) implementation.

- **Rohart, S.; Thiaville, A.** "Skyrmion confinement in ultrathin
  film nanostructures in the presence of Dzyaloshinskii-Moriya
  interaction." *Phys. Rev. B* 88, 184422 (2013).
  [DOI](https://doi.org/10.1103/PhysRevB.88.184422). The chiral
  edge-canting boundary condition we **don't** implement (cited in
  [`physics.md`](physics.md) §6.4).

- **Dzyaloshinskii, I. E.** "A thermodynamic theory of weak
  ferromagnetism of antiferromagnetics." *J. Phys. Chem. Solids* 4,
  241 (1958). Original DMI.

- **Moriya, T.** "Anisotropic superexchange interaction and weak
  ferromagnetism." *Phys. Rev.* 120, 91 (1960). Microscopic DMI
  mechanism.

Used in: [`equations.md`](equations.md) §2.2 (interlayer), §2.5 (DMI).

---

## 8. Magnon BEC and time crystals

- **Demokritov, S. O. et al.** "Bose-Einstein condensation of quasi-
  equilibrium magnons at room temperature under pumping." *Nature*
  443, 430 (2006). Magnon BEC in YIG. Cited only as physics context;
  this simulator does not have the dipolar dispersion needed to
  reproduce it (the k-space dashboard view shows what we *can*
  visualise; see [`usage.md`](usage.md) §4.2).

- **Bunkov, Yu. M.; Volovik, G. E.** "Magnon BEC in superfluid
  ³He-B." *J. Phys. Condens. Matter* 22, 164210 (2010).
  Time-crystal-like coherent precession in superfluid helium.

- **Wilczek, F.** "Quantum time crystals." *Phys. Rev. Lett.* 109,
  160401 (2012). Original time-crystal proposal.

Used in: README motivation; not directly consumed by the model.

---

## 9. Numerical methods and benchmarks

- **µMAG Standard Problem 4 (NIST).**
  https://www.ctcms.nist.gov/~rdm/mumag.org.html. Standard demag-
  inclusive LLG benchmark. Not literally implementable here without a
  demagnetisation field; replaced by a deterministic LLG-vs-LLB
  proxy. ADR-004 documents the substitution.

- **Heun's method** (predictor-corrector, second-order) is textbook;
  no specific reference used. The projected variant (post-stage
  `normalize`) is standard in micromagnetics.

- **rustfft.** https://crates.io/crates/rustfft. Used for the
  dashboard k-space view (2D-FFT) and `metrics.rs::analyze_time_series`
  (1D-FFT).

- **wgpu / WebGPU.** https://wgpu.rs / https://www.w3.org/TR/webgpu/.
  GPU API and shading language (WGSL).

---

## 10. ADR-internal references

The decision records in `branches/hir/decisions/ADR-001..006`
reference each other and several papers above. Key ADR ↔ reference
pairs:

| ADR | Topic | Key external reference |
|---|---|---|
| 001 | `fgt_bulk` as default preset | Leon-Brito 2016 (§4.1) |
| 002 | Stack-order is first-class | none external |
| 003 | P3a/P3b GPU layout deviations | none external |
| 004 | µMAG SP4 substitution | µMAG SP4 spec (§9) |
| 005 | Original FGT calibration (T = 150 K) | Zhou 2025 (§4.1) |
| 006 | FGT recalibration at literal T = T_c | Zhou 2025; supersedes ADR-005 |

---

## 11. How to add a reference

When citing a new paper or dataset:

1. Add it to the appropriate section above.
2. Cross-reference the section of [`equations.md`](equations.md) or
   [`physics.md`](physics.md) that consumes it.
3. If it informs a parameter value, link from the preset's `notes`
   field in `material*.rs`.
4. If it justifies a model deviation, cite from the relevant ADR.

The goal is a one-look bibliography: anyone reading a parameter or an
equation should be able to trace it back to a published source (or
flag it as conjectural / fitted if no source applies).
