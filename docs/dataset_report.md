# UrbanTALES Dataset Discovery & Technical Evaluation Report

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Task:** Task 3 — Dataset Discovery and Data Contract  
**Owner:** Project Lead  
**Document Version:** 2.0.0  
**Branch:** `feature/dataset-pipeline`  
**Status:** Approved Reference Report  

---

## 1. Executive Summary & Authoritative Verification

The course project evaluates urban microclimate prediction around simplified and realistic building layouts, comparing:
1. An automated machine learning (AML) baseline using geometric summary features to predict an area-averaged flow quantity.
2. A deep neural surrogate using multi-channel spatial raster inputs to predict a continuous 2D flow field.

We investigated the **UrbanTALES** dataset (*Urban Turbulence Analyses from Large-Eddy Simulations*) recommended by the college brief to determine its suitability, physical fidelity, access parameters, and operational constraints.

### Authoritative Verification Summary:
- **Title:** UrbanTALES: A Comprehensive LES Dataset for Urban Canopy Layer Turbulence Analyses and Parameterization Development
- **Lead Investigators:** Negin Nazarian (UNSW Sydney), Maarten van Reeuwijk (Imperial College London), and collaborators.
- **Publications:** 
  - *Bulletin of the American Meteorological Society (BAMS)*, DOI: [10.1175/BAMS-D-25-0061.1](https://doi.org/10.1175/BAMS-D-25-0061.1) (2025).
  - *EarthArXiv Preprint*, DOI: [10.31223/X58X3B](https://doi.org/10.31223/X58X3B).
- **Official Access Portal:** [https://urbantales.vercel.app/](https://urbantales.vercel.app/) / UrbanTALES dissemination platform.
- **Underlying Codebase:** Built on **uDALES** (open-source urban extension of Dutch Atmospheric Large-Eddy Simulation, GitHub: [uDALES/u-dales](https://github.com/uDALES/u-dales)).
- **License & Legal Attribution:** Distributed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. Academic and educational reuse is permitted provided formal attribution is given to Nazarian et al. (2025). Commercial utilization is prohibited without express author consent.

---

## 2. Technical Specifications of the UrbanTALES Dataset

| Technical Dimension | Authoritative Specification in UrbanTALES | Implications for Machine Learning & Surrogates |
|---|---|---|
| **Layout Count & Scope** | **538 total configurations**: 224 idealized building arrays and 314 realistic neighborhood cutouts across 62 global cities. | Provides sufficient sample diversity to evaluate both in-distribution interpolation and out-of-distribution family holdout. |
| **Geometry Representation** | 2D raster matrices of building top heights above ground level $H(x, y)$ [$\text{m}$] derived from OpenStreetMap footprints and LiDAR building heights. Ground terrain = $0.0\text{ m}$. | Directly rasterizable into continuous 2D height maps and binary building masks without loss of polygonal fidelity. |
| **Plan Area Density ($\lambda_p$)** | Ranging from sparse suburban ($\lambda_p \approx 0.06$) to extremely dense urban cores ($\lambda_p \approx 0.64$). | Spans all three classical flow regimes: isolated roughness, wake interference, and skimming flow. |
| **Building Heights ($H$)** | Mean building heights $H_{\text{mean}} \in [4\text{ m}, 70\text{ m}]$. Idealized layouts include both uniform and variable height arrays ($H_{\text{std}} / H_{\text{mean}}$). | Allows testing surrogate sensitivity to height variance and skyline heterogeneity. |
| **Spatial Resolution** | <ul><li>*Idealized Arrays:* Uniform $\Delta x = \Delta y = \Delta z = 0.5\text{ m}$.</li><li>*Realistic Cities:* $\Delta x = \Delta y = 1.0\text{ m}$, $\Delta z = 0.5\text{ m}$.</li></ul> | Sub-meter to 1-meter horizontal resolution captures pedestrian sidewalk scales and narrow street canyons. |
| **Coordinate System & Units** | Right-handed Cartesian $(x, y, z)$. $z = 0$ is ground level. Standard SI units ($\text{m}, \text{m/s}, \text{m}^2/\text{s}^2, \text{degrees}$). | Universal SI unit consistency across all data ingestion and normalization pipelines. |
| **CFD Solver & Turbulence** | uDALES (parallelized incompressible Navier-Stokes solver using fractional-step method, FFT Poisson solver, and Deardorff/Smagorinsky subgrid-scale turbulence closure). | High-fidelity LES resolving anisotropic turbulent eddies and transient wake vortex shedding. |
| **Boundary Conditions** | <ul><li>*Bottom/Ground:* Wall roughness length $z_0$ and no-slip wall functions.</li><li>*Top:* Free-slip rigid lid ($\partial u / \partial z = 0$) at domain top ($L_z \approx 130\text{ m}$).</li><li>*Lateral:* Periodic/cyclic for idealized arrays; buffer fringe zones for realistic city cutouts.</li></ul> | Well-characterized numerical boundary conditions that prevent unphysical blockage and domain confinement ($A_{\text{frontal}} / A_{\text{domain}} < 3\%$). |

---

## 3. Dual-Modality Support: Area-Averaged vs. Spatial Field Prediction

UrbanTALES natively supports both target modalities required by the project brief:

### 3.1 Spatial-Field Prediction (Target for Model 2: Neural Surrogate)
- **Pedestrian Wind Speed Slices:** UrbanTALES provides horizontal 2D planar velocity slices sampled at pedestrian height:
  $$z = 1.75\text{ m}$$
- **Components Available:** Time-averaged velocity components $\bar{u}(x, y), \bar{v}(x, y), \bar{w}(x, y)$ and scalar speed magnitude $U(x, y) = \sqrt{\bar{u}^2 + \bar{v}^2 + \bar{w}^2}$ [$\text{m/s}$].
- **Turbulence Slices:** Second-order turbulent kinetic energy (TKE) $k(x, y)$ [$\text{m}^2/\text{s}^2$].
- **Raster Grid Compatibility:** Directly aligned with 2D building height maps $H(x, y)$, making them ideal target tensors for 2D convolutional encoder-decoders (U-Nets).

### 3.2 Area-Averaged Prediction (Target for Model 1: Traditional AML Baseline)
- **Canopy-Averaged Bulk Statistics:** The dataset provides volume-averaged flow speed $\langle U \rangle_{\text{canopy}}$ from ground level up to mean building height $H_{\text{mean}}$.
- **Pedestrian Outdoor Surface Averaging:** Alternatively, a pedestrian-level area-averaged wind speed is calculated by integrating the pedestrian slice strictly across the outdoor fluid domain:
  $$\bar{U}_{\text{ped}} = \frac{\iint_{\Omega_{\text{outdoor}}} U(x, y, z=1.75) \, dx \, dy}{\iint_{\Omega_{\text{outdoor}}} dx \, dy} = \frac{\sum_{i,j} (1 - M_{i,j}) U_{i,j} \Delta A_{i,j}}{\sum_{i,j} (1 - M_{i,j}) \Delta A_{i,j}}$$
  where $M_{i,j} \in \{0, 1\}$ is the building footprint mask ($1$ inside solid building, $0$ in fluid).
- **Physical Validity:** This dual representation enables a direct, fair benchmark: Model 1 predicts $\bar{U}_{\text{ped}}$ from 11 scalar morphometrics, while Model 2 predicts $U(x, y)$, which is then integrated to compare against Model 1.

---

## 4. Dataset Strategy: UrbanTALES vs. Local OpenFOAM Sweep

We conducted a comparative trade-off analysis between utilizing UrbanTALES versus generating a local CFD simulation sweep using OpenFOAM, in direct consultation with the CFD teammate's findings in [`docs/cfd_feasibility.md`](docs/cfd_feasibility.md) and [`docs/cfd_methodology.md`](docs/cfd_methodology.md):

| Dimension | Option A: Existing UrbanTALES Corpus | Option B: Local OpenFOAM Sweep |
|---|---|---|
| **Turbulence Fidelity** | **High-Fidelity 3D LES** (resolves multi-scale turbulent eddies, unsteady wake shedding, and Reynolds stresses). | **Steady 2D/3D RANS** ($k$-$\omega$ SST) (averages turbulence into static eddy viscosity; underestimates wake fluctuations). |
| **Geometry Diversity** | **538 Layouts** across 62 real cities and canonical arrays ($\lambda_p \in [0.06, 0.64]$). | **10–30 Simple Blocks** (generating complex 3D meshes for hundreds of city layouts is intractable locally). |
| **Computational Feasibility** | **Zero local compute required** (pre-computed using 3+ million CPU hours on supercomputers). | **High Local Bottleneck**: Developer workstation runs macOS Apple Silicon without native Linux ABI or container runtimes (Docker/Podman). |
| **Generation Time** | Fast download of curated 2D slices ($< 1\text{ GB}$). | 2–5 hours for small 2D sweep; days/weeks for 3D sweep with high thermal throttling. |
| **Thermal Field Availability** | **None** (isothermal neutral ABL only). | Possible in theory with `buoyantSimpleFoam`, but requires complex solar radiation coupling. |
| **Scientific Validation** | Peer-reviewed and published in *BAMS* (Nazarian et al., 2025). | Self-generated; requires extensive experimental wind-tunnel validation (e.g. CEDVAL). |

### Team Consensus: Dual-Pillar Strategy
1. **Primary Ground Truth (UrbanTALES):** Adopt UrbanTALES as the primary corpus for training and benchmarking both the AML baseline (Model 1) and the neural surrogate (Model 2) across the full spectrum of 538 layouts.
2. **Verification Anchor (Local OpenFOAM Prototype):** Maintain the production-grade 2D OpenFOAM benchmark case ([`cfd/cases/simple2d_building_block/`](../cfd/cases/simple2d_building_block/)) and the automated bridge script ([`cfd/scripts/foam_to_contract.py`](../cfd/scripts/foam_to_contract.py)) as an independent physical verification tool to test boundary-layer inlet profiles and CFD-to-contract transformations.

---

## 5. Critical Limitations and Operational Guidelines

1. **Absence of Thermal / Temperature Data:**
   - UrbanTALES is strictly an **isothermal, aerodynamic** simulation dataset. It does **not** solve energy transport or surface radiation.
   - *Operational Rule:* The primary prediction target is established as **pedestrian wind speed magnitude ($U$ at $z = 1.75\text{ m}$)**. Temperature prediction is formally documented as excluded from the UrbanTALES scope.
2. **Monolithic Storage Avoidance (~35 TB):**
   - The complete 3D time-dependent flow fields total ~35 TB.
   - *Operational Rule:* Team members must **never attempt to download the full 3D volume**. Only the 2D pedestrian horizontal slices ($z = 1.75\text{ m}$) and 2D building height ASCII grids should be retrieved for target subsets (e.g., 60–100 layouts).
3. **Solid Geometry Masking:**
   - Fluid velocity is undefined or zero inside building volumes. Treating building interiors as valid zero-speed fluid cells distorts loss gradients.
   - *Operational Rule:* The pipeline enforces strict boolean building masking across all loaders, normalizers, and loss evaluators.
