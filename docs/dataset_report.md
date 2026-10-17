# UrbanTALES Dataset Investigation & Evaluation Report

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Role:** Dataset and Data Engineering Lead  
**Date:** October 2026  
**Status:** Evaluated and Documented (No large raw files downloaded locally)

---

## 1. Executive Summary

The project brief references the **UrbanTALES** dataset (*Urban Turbulence Analyses from Large-Eddy Simulations*) as a candidate benchmark for modeling wind and temperature fields around building layouts. 

UrbanTALES provides high-resolution Large-Eddy Simulation (LES) data across **538 distinct urban layouts** (both idealized arrays and real city neighborhoods). However, our investigation identifies key capabilities and fundamental limitations:
1. **Strengths:** High spatial fidelity (0.5 m – 1.0 m resolution), rigorous time-averaged LES flow moments, pedestrian-level horizontal slices ($z = 1.75\text{ m}$), and well-characterized layout families.
2. **Key Physical Limitation:** UrbanTALES simulates **isothermal / neutrally stratified atmospheric boundary layer flows**. It does **not** include active thermal scalar transport, solar surface radiation, or temperature fields. Therefore, for this dataset, **wind velocity magnitude and turbulent kinetic energy (TKE) are valid target variables**, while temperature is **not** supported directly.
3. **Storage & Access:** The complete 3D dataset comprises **~35 TB** across 3 million CPU hours of computation. For our neural surrogate and ML baseline pipeline, we must retrieve only the **2D pedestrian-level slices ($z = 1.75\text{ m}$)** or downsampled canopy slices via the dissemination portal rather than attempting a monolithic raw download.

---

## 2. Source, Access, and Licensing

| Item | Details |
|---|---|
| **Dataset Title** | UrbanTALES: Urban Turbulence Analyses from Large-Eddy Simulations |
| **Principal Authors** | Negin Nazarian, Maarten van Reeuwijk, and collaborators (UNSW Sydney, Imperial College London, etc.) |
| **Primary Publications** | <ul><li>*BAMS Journal Paper:* DOI [10.1175/BAMS-D-25-0061.1](https://doi.org/10.1175/BAMS-D-25-0061.1)</li><li>*EarthArXiv Preprint:* DOI [10.31223/X58X3B](https://doi.org/10.31223/X58X3B)</li></ul> |
| **Access Portal** | [https://urbantales.vercel.app/](https://urbantales.vercel.app/) / UrbanTALES dissemination platform |
| **Simulation Codebase** | Built on **uDALES** (open-source urban LES model derived from Dutch Atmospheric Large-Eddy Simulation): [GitHub - uDALES](https://github.com/uDALES/u-dales) |
| **License** | **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** |
| **Attribution Requirement** | Commercial use prohibited without explicit permission. Any academic publication, report, or open repository must cite Nazarian et al. (2025). |

---

## 3. Layouts, Geometry, and Available Physical Variables

### 3.1 Layout Categorization (538 Total Configurations)
The dataset is structured into two core layout families:

1. **Idealized Building Arrays (224 configurations):**
   - Canonical building layouts: **aligned** and **staggered** cubical/rectangular block arrays.
   - Plan area densities $\lambda_p \in [0.06, 0.44]$.
   - Uniform height arrays ($H = \text{const}$) and variable height arrays with controlled height variance ($H_{\text{std}} / H_{\text{mean}}$).
   - Inflow angles tested: perpendicular ($\delta = 0^\circ$) and oblique ($\delta = 15^\circ, 30^\circ, 45^\circ$).
2. **Realistic Urban Neighborhoods (314 configurations):**
   - Extracted from real-world urban topologies across **62 cities in 20 countries** using OpenStreetMap building footprints and LiDAR height data.
   - Plan area densities $\lambda_p \in [0.06, 0.64]$.
   - Mean building heights $H_{\text{mean}} \in [4\text{ m}, 70\text{ m}]$, capturing low-rise suburban, mid-rise commercial, and high-rise dense downtown districts.

### 3.2 Physical Variables Provided
- **First-Order Flow Fields (Time-Averaged):**
  - Streamwise velocity $\bar{u}(x, y, z)$ [$\text{m/s}$]
  - Spanwise velocity $\bar{v}(x, y, z)$ [$\text{m/s}$]
  - Vertical velocity $\bar{w}(x, y, z)$ [$\text{m/s}$]
  - Total velocity magnitude $U(x, y, z) = \sqrt{\bar{u}^2 + \bar{v}^2 + \bar{w}^2}$ [$\text{m/s}$]
  - Pedestrian-level wind speed slice at $z = 1.75\text{ m}$ [$\text{m/s}$]
- **Second- and Third-Order Turbulence Moments:**
  - Reynolds normal stresses: $\overline{u'u'}$, $\overline{v'v'}$, $\overline{w'w'}$ [$\text{m}^2/\text{s}^2$]
  - Turbulent kinetic energy (TKE): $k = \frac{1}{2}(\overline{u'u'} + \overline{v'v'} + \overline{w'w'})$ [$\text{m}^2/\text{s}^2$]
  - Momentum fluxes (Reynolds shear stresses): $\overline{u'w'}$, $\overline{v'w'}$ [$\text{m}^2/\text{s}^2$]
- **Summary & Profile Outputs:**
  - Horizontally-averaged vertical velocity and flux profiles $U(z), \tau(z)$.
  - Canopy-volume-averaged wind speed and drag coefficients.

---

## 4. Coordinate System, Units, and Resolution

- **Coordinate System:** Right-handed Cartesian $(x, y, z)$.
  - $x$: Streamwise direction (aligned with principal wind direction or coordinate axis).
  - $y$: Spanwise / lateral horizontal direction.
  - $z$: Vertical direction ($z = 0$ is ground level).
- **Units:** Standard SI units (lengths in $\text{m}$, velocities in $\text{m/s}$, stresses/TKE in $\text{m}^2/\text{s}^2$, angles in degrees).
- **Grid Resolution:**
  - *Idealized Arrays:* Uniform grid spacing $\Delta x = \Delta y = 0.5\text{ m}$, $\Delta z = 0.5\text{ m}$.
  - *Realistic Neighborhoods:* $\Delta x = \Delta y = 1.0\text{ m}$, $\Delta z = 0.5\text{ m}$.
  - Domain height $L_z \approx 130\text{ m}$ (typically $5 - 8 \times H_{\text{mean}}$ to prevent artificial boundary confinement).

---

## 5. CFD Solver Settings and Boundary Conditions

- **Solver:** uDALES (LES solver using MPI parallelization, fractional-step Navier-Stokes solver, and filtered subgrid-scale turbulence closures such as Deardorff / Smagorinsky models).
- **Boundary Conditions:**
  - **Bottom (Ground and Building Surfaces):** Immersed boundary method or surface wall functions with aerodynamic roughness length $z_0$ (typically $0.01 - 0.05\text{ m}$ on ground/walls).
  - **Top Boundary ($z = L_z$):** Free-slip / rigid lid with zero vertical gradient ($\partial u / \partial z = 0$).
  - **Lateral Boundaries:** Periodic / cyclic boundary conditions for idealized repeating arrays; turbulent inflow generator or buffer fringe zones for realistic city cutouts.
- **Time Averaging:** Simulations run until boundary layer flow reaches quasi-steady statistical equilibrium, followed by time-averaging over thousands of time steps exceeding eddy turnover times.

---

## 6. Critical Limitations and Team Decision Points

1. **No Temperature / Thermal Data:**
   - UrbanTALES is an **isothermal, aerodynamic** LES dataset.
   - *Recommendation:* Our project scope should establish **pedestrian wind speed ($U_{1.75\text{m}}$)** as the primary target variable and **turbulent kinetic energy ($k$)** as the secondary target. If temperature prediction is strictly required by the course brief, we must augment or replace UrbanTALES with a thermal CFD dataset (e.g. coupled OpenFOAM with Boussinesq buoyant flow or ENVI-met simulations).
2. **Dataset Size (~35 TB):**
   - The full 3D transient or time-averaged fields cannot be stored locally on student workstations.
   - *Recommendation:* Download only the processed **2D pedestrian slices ($z = 1.75\text{ m}$)** and building height ASCII grids from `urbantales.vercel.app` for a curated subset of 50–100 layouts.
3. **Solid Building Interiors:**
   - In LES, building cells have zero velocity or are excluded from the fluid grid. In neural networks, treating building interiors as arbitrary zeroes can skew mean/std statistics and MSE loss gradients.
   - *Solution in our data pipeline:* We implemented explicit building masking in `FieldNormalizer` and `LayoutRasterizer`, enabling loss functions to mask out building interiors during training.
