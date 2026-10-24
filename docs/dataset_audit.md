# Urban Microclimate Modelling: Dataset Audit Report

**Author / Role:** Team Member A — Dataset Audit & Data Validation (@Aditya)  
**Date:** October 24, 2026  
**Branch:** `feature/dataset-audit`  
**Document Version:** 1.0.0  
**Status:** Complete Audit & Provenance Verification  

---

## 1. Executive Summary & Beginner-Friendly Overview

### What is this project doing?
This project develops and evaluates fast surrogate models for urban microclimates. Specifically, we are comparing two approaches:
1. **Traditional Machine Learning Baseline:** Uses 11 geometric summary numbers (such as how dense buildings are, how tall they are, and frontal wind areas) to predict a single summary number: the **average outdoor pedestrian wind speed**.
2. **Deep Neural Surrogate (U-Net):** Takes a 2D map of building shapes and inflow winds (a multi-channel 2D image-like tensor) and predicts the full **2D spatial wind speed map** at walking height ($1.75\text{ m}$ above the ground).

### What data exists right now, what can it do, and what is missing?
- **What data is actually on our machines right now?**
  Only **software test fixtures and canonical benchmark files** exist locally. There are **no massive real-world CFD runs downloaded yet** (which is intentional to keep the repository lightweight and adhere to the project's data-storage policy).
- **What candidate external dataset has been identified?**
  The **UrbanTALES** dataset (Nazarian et al., 2025; BAMS DOI: [10.1175/BAMS-D-25-0061.1](https://doi.org/10.1175/BAMS-D-25-0061.1)). It contains 538 Large-Eddy Simulations (LES) across idealized building arrays and 62 global cities.
- **What can it support?**
  It directly supports **pedestrian wind speed ($z = 1.75\text{ m}$)** and **turbulent kinetic energy (TKE)** across diverse building layouts.
- **What is missing or cannot be supported?**
  **Temperature is NOT present.** UrbanTALES is purely aerodynamic and isothermal. It does not model heat transfer, solar radiation, or thermal air temperature. Therefore, temperature prediction cannot be performed using UrbanTALES.
  Furthermore, the full 3D simulation files are ~35 TB, so the team must only download **2D pedestrian horizontal slices ($z = 1.75\text{ m}$)** for a curated subset of 50–100 layouts.

---

## 2. Categorization of Repository Data Assets

To ensure strict scientific integrity, all data assets within or referenced by this repository are formally categorized into four distinct classes:

```
+----------------------------------------------------------------------------------------------------+
|                                    DATA ASSET TAXONOMY                                             |
+------------------------------------+---------------------------------------------------------------+
| Category                           | Current Assets & Location                                     |
+------------------------------------+---------------------------------------------------------------+
| 1. Verified Project Data (Local)   | None committed locally (data/ is gitignored for raw files)     |
| 2. Candidate External Datasets     | UrbanTALES (online), CEDVAL Wind Tunnel (online reference)    |
| 3. CFD Simulation Outputs          | simple2d_building_block (OpenFOAM case setup, unexecuted local)|
| 4. Synthetic Test Fixtures         | tests/fixtures/synthetic_data.py, canonical_block_reference.npz|
+------------------------------------+---------------------------------------------------------------+
```

### Category 1: Verified Project Data (Local)
- **Status:** **None currently downloaded or committed.**
- **Details:** The repository `.gitignore` deliberately excludes `data/raw/`, `data/processed/`, `*.npz`, `*.h5`, `*.nc`, and `*.npy`. This prevents multi-gigabyte files from bloating the Git repository.

### Category 2: Candidate External Reference Datasets
- **Primary Source: UrbanTALES**
  - *Full Name:* Urban Turbulence Analyses from Large-Eddy Simulations.
  - *Lead Investigators:* Negin Nazarian (UNSW Sydney), Maarten van Reeuwijk (Imperial College London), et al.
  - *Portal:* `https://urbantales.vercel.app/`
  - *License:* Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
  - *Status:* **Candidate external data.** Verified via BAMS publication and online portal.
- **Validation Benchmark: CEDVAL Wind Tunnel Database**
  - *Source:* Meteorological Institute, University of Hamburg.
  - *Portal:* `https://www.mi.uni-hamburg.de/en/arbeitsgruppen/meteorologie/forschung/windtunnel/cedval.html`
  - *Status:* Candidate experimental validation source for physical flow verification (Cases A1-1 isolated building, B1-1 street canyon).

### Category 3: CFD Solver Output
- **Status:** **Case configured; local solver run unexecuted.**
- **Location:** `cfd/cases/simple2d_building_block/`
- **Details:** Contains a complete OpenFOAM case directory (`0/`, `constant/`, `system/`, `Allrun`, `Allclean`). Because the local development machine (macOS Apple Silicon) lacks native OpenFOAM or container runtimes (Docker/Podman), local solver execution has not yet occurred.
- **Scientific Integrity Note:** The presence of solver configuration files must never be reported as completed solver runs.

### Category 4: Synthetic Software-Test Fixtures
- **Fixture 1:** `tests/fixtures/synthetic_data.py` (`create_synthetic_sample`)
  - Deterministic mathematical shapes used purely for automated software unit testing (shape consistency, mask orientation, data loading, train/test splitting).
- **Fixture 2:** `cfd/outputs/canonical_block_reference.npz` (and `canonical_block_field.svg`)
  - Analytically synthesized flow field matching published wind-tunnel flow features (stagnation divergence, corner acceleration jets, $4.5H$ wake recirculation). Used to test the `cfd/scripts/foam_to_contract.py` data bridge.
  - Clearly marked in metadata as `city_or_source="CANONICAL_AERODYNAMIC_BENCHMARK_REFERENCE"`. **Never to be cited as real experimental or CFD solver output.**

---

## 3. Deep-Dive Audit of Candidate External Data: UrbanTALES

### 3.1 Provenance, Access, and License Verification

| Audit Item | Verified Finding | Evidence & Source |
|---|---|---|
| **Official Name** | UrbanTALES: Urban Turbulence Analyses from Large-Eddy Simulations | BAMS Paper DOI: 10.1175/BAMS-D-25-0061.1 |
| **Principal Authors** | Assoc. Prof. Negin Nazarian, Dr. Maarten van Reeuwijk, Dr. Jiachen Lu, Dr. Mathew Lipson, et al. | Dissemination portal & BAMS publication |
| **Official URL** | [https://urbantales.vercel.app/](https://urbantales.vercel.app/) | Verified live HTTP response |
| **Hosting Organization** | Climate-Resilient Cities (CRC) Lab, University of New South Wales (UNSW), Australia | Portal metadata |
| **License** | **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International) | Explicitly stated in portal footer and publication |
| **Commercial Restriction** | Commercial use prohibited without explicit permission. Educational/academic reuse permitted with mandatory citation. | License terms |
| **Underlying Solver** | **uDALES** (open-source Fortran 90/MPI urban atmospheric LES code) | GitHub: `uDALES/u-dales` |

### 3.2 Available Layout Families and Geometries

UrbanTALES provides **538 total urban configurations** partitioned into two fundamental families:

1. **Idealized Building Arrays (224 configurations):**
   - *Morphologies:* Uniform and staggered regular arrays of rectangular/cubical building blocks.
   - *Plan Area Density:* $\lambda_p \in [0.0625, 0.64]$ (spanning isolated roughness, wake interference, and skimming flow regimes).
   - *Height Variations:* Uniform height arrays ($H_{\text{std}} = 0\text{ m}$) and heterogeneous height distributions ($H_{\text{std}} \in [2.8\text{ m}, 5.6\text{ m}]$).
   - *Wind Directions:* Perpendicular ($\delta = 0^\circ$) and oblique ($\delta = 15^\circ, 30^\circ, 45^\circ$) to building alignments.
   - *Grid Resolution:* Uniform $\Delta x = \Delta y = \Delta z = 0.5\text{ m}$. Domain height $L_z \approx 130\text{ m}$.
2. **Realistic Urban Neighborhoods (314 configurations):**
   - *Topologies:* Real building footprints extracted from OpenStreetMap and LiDAR height models across **62 global cities in 20 countries**.
   - *Plan Area Density:* $\lambda_p \in [0.05, 0.75]$.
   - *Building Heights:* Mean heights $H_{\text{mean}} \in [4\text{ m}, 70\text{ m}]$ (suburban residential to high-rise downtown cores).
   - *Grid Resolution:* $\Delta x = \Delta y = 1.0\text{ m}, \Delta z = 0.5\text{ m}$.

### 3.3 Available Flow Variables and Physical Completeness

| Physical Variable | Status in UrbanTALES | Units | Suitability for Project |
|---|---|---|---|
| **Streamwise Velocity ($\bar{u}$)** | Available | $\text{m/s}$ | Time-averaged horizontal velocity component. |
| **Spanwise Velocity ($\bar{v}$)** | Available | $\text{m/s}$ | Time-averaged lateral velocity component. |
| **Vertical Velocity ($\bar{w}$)** | Available | $\text{m/s}$ | Time-averaged vertical velocity component. |
| **Pedestrian Wind Speed ($U_{1.75\text{m}}$)** | **Available** | $\text{m/s}$ | **Primary Spatial Target** for Neural Surrogate: $U = \sqrt{\bar{u}^2 + \bar{v}^2 + \bar{w}^2}$ at $z = 1.75\text{ m}$. |
| **Area-Averaged Outdoor Wind Speed** | **Computable** | $\text{m/s}$ | **Primary Scalar Target** for ML Baseline: surface integration over outdoor cells via `src/data/area_averaging.py`. |
| **Turbulent Kinetic Energy ($k$)** | Available | $\text{m}^2/\text{s}^2$ | Secondary spatial field: $k = \frac{1}{2}(\overline{u'u'} + \overline{v'v'} + \overline{w'w'})$. |
| **Reynolds Stresses ($\overline{u'w'}, \overline{v'w'})$** | Available | $\text{m}^2/\text{s}^2$ | Vertical momentum flux profiles. |
| **Air Temperature ($T$)** | **NOT AVAILABLE** | $\text{K}$ or $^\circ\text{C}$ | **Absent.** UrbanTALES is strictly isothermal. |
| **Surface Heat Fluxes / Radiation** | **NOT AVAILABLE** | $\text{W/m}^2$ | **Absent.** Solar heating and thermal buoyancy are not modelled. |

---

## 4. Verification Status: Verified vs. Unverified Details

To adhere to the scientific integrity mandate (*"If you cannot verify a detail, explicitly mark it 'unverified' and explain how the team can verify it"*):

### Verified Details:
1. **Dataset Name and Citations:** Confirmed via BAMS journal publication (DOI: 10.1175/BAMS-D-25-0061.1).
2. **License:** Confirmed as CC BY-NC 4.0 via portal and publication.
3. **Absence of Thermal Data:** Confirmed via peer-reviewed literature; uDALES was executed in neutral, isothermal atmospheric mode.
4. **Resolution and Layout Count:** Confirmed (224 idealized, 314 realistic, totaling 538).
5. **Horizontal Slices:** Confirmed that horizontal 2D planar arrays are generated at $z = 1.75\text{ m}$.

### Unverified Details (and Verification Protocols):
1. **Exact File Format on Web Dissemination Portal [UNVERIFIED]:**
   - *Current State:* The web portal `urbantales.vercel.app/download_ideal` uses interactive Next.js client-side components to serve files. The exact file container (NetCDF `.nc`, HDF5 `.h5`, or compressed ASCII grids) has not been inspected directly via local download.
   - *How the team can verify:* Download a single sample layout zip/archive from `https://urbantales.vercel.app/download_ideal` and run `ncdump -h <file>` or Python `h5py` / `netCDF4` inspection script.
2. **Inflow Roughness Length ($z_0$) and Friction Velocity ($u_*$) in File Headers [UNVERIFIED]:**
   - *Current State:* While the LES boundary conditions are documented as a rough-wall bottom boundary in uDALES, whether each individual layout file contains explicit header metadata for $z_0$ and $u_*$ is unverified until a raw file is unpacked.
   - *How the team can verify:* Inspect the metadata dictionary or attributes of the downloaded NetCDF/ASCII file. If missing, use the default values reported in Nazarian et al. ($z_0 = 0.05\text{ m}$, $U_{\text{ref}} = 5.0\text{ m/s}$).
3. **Precise Spatial Coordinate Anchors [UNVERIFIED]:**
   - *Current State:* It is unverified whether the 2D matrices have bottom-left origin $(0, 0)$ or top-left origin $(0, 0)$, or whether row index $i$ corresponds to $+y$ (north) or $-y$ (south).
   - *How the team can verify:* Compare the building height raster against the pedestrian velocity wake. Inflow travels from west to east ($+x$); the wake cavity must appear downwind (to the right of the obstacle).

---

## 5. Support for Agreed Target Variables & Spatial Prediction Task

### 5.1 Primary Spatial Target: Pedestrian Wind Speed ($z = 1.75\text{ m}$)
- **Surrogate Compatibility:** **Full Support.**
- The neural surrogate takes an input tensor $\mathbf{X} \in \mathbb{R}^{7 \times N_y \times N_x}$ (building height, building mask, signed distance field, inflow velocity components, normalized coordinates) and outputs predicted spatial field $\hat{\mathbf{Y}} \in \mathbb{R}^{1 \times N_y \times N_x}$.
- Because UrbanTALES provides horizontal 2D velocity slices at $z = 1.75\text{ m}$, the target field aligns cell-by-cell with the building height raster.

### 5.2 Primary Scalar Target: Area-Averaged Wind Speed ($\bar{U}_{\text{area}}$)
- **Baseline ML Compatibility:** **Full Support.**
- The traditional ML baseline (Random Forest / Gradient Boosted Trees) predicts a single scalar $\bar{U}_{\text{area}}$ from 11 tabular geometric features (plan density $\lambda_p$, frontal density $\lambda_f$, mean height $H_{\text{mean}}$, roughness $z_0$, displacement height $d$, etc.).
- Using `src/data/area_averaging.py`, the target scalar is rigorously computed by mask-tracked surface integration strictly over fluid cells:
  $$\bar{U}_{\text{ped}} = \frac{\sum_{i,j} (1 - M_{i,j}) U_{i,j} \Delta A_{i,j}}{\sum_{i,j} (1 - M_{i,j}) \Delta A_{i,j}}$$
- Solid building interior cells ($M_{i,j} = 1$) are excluded to prevent zero-velocity bias.

### 5.3 Thermal / Temperature Prediction: **UNSUPPORTED**
- **Impact on Course Brief:** If the course brief suggests predicting temperature, the team must explicitly state that **UrbanTALES does not contain thermal data**.
- Claiming that wind-only aerodynamic CFD or UrbanTALES produces temperature fields is physically impossible and violates scientific integrity.
- If temperature prediction is strictly required by the course examiners, the project must incorporate an alternative thermal CFD dataset (e.g., coupled OpenFOAM `buoyantBoussinesqSimpleFoam` or ENVI-met simulations).

---

## 6. Alignment with Repository Data Contract

The repository defines its authoritative data schema in `src/data/contract.py` and `docs/data_schema.md`. UrbanTALES aligns with this schema without requiring competing abstractions:

| Data Contract Class | Repository Implementation | UrbanTALES Mapping |
|---|---|---|
| **`LayoutMetadata`** | `src/data/contract.py` | `layout_id`: unique configuration code.<br>`layout_family`: `idealized_aligned`, `idealized_staggered`, `realistic_neighborhood`.<br>`domain_size_m`: $(L_x, L_y, L_z)$.<br>`resolution_m`: $(0.5, 0.5, 0.5)$ or $(1.0, 1.0, 0.5)$.<br>`grid_shape`: $(N_y, N_x)$. |
| **`GeometryData`** | `src/data/contract.py` | `height_map`: $H(x, y) \ge 0$.<br>`building_mask`: $H(x, y) > 10^{-4}\text{ m}$. |
| **`EnvironmentalConditions`** | `src/data/contract.py` | `inflow_speed_mps`: approach wind speed.<br>`wind_direction_deg`: $0^\circ, 15^\circ, 30^\circ, 45^\circ$.<br>`roughness_length_z0_m`: terrain roughness. |
| **`CFDSolverMetadata`** | `src/data/contract.py` | `solver_name="uDALES"`<br>`turbulence_model="LES-Deardorff"` |
| **`TargetFields`** | `src/data/contract.py` | `pedestrian_wind_speed`: $(N_y, N_x)$ float32 array at $z = 1.75\text{ m}$.<br>`area_averaged_wind_speed`: float32 scalar.<br>`turbulent_kinetic_energy`: optional $(N_y, N_x)$ array. |

---

## 7. Storage, Reproducibility & Holdout Strategy

### 7.1 Monolithic Storage Avoidance (~35 TB)
- The raw 3D time-resolved LES velocity fields across 538 simulations exceed **35 Terabytes**.
- **Actionable Rule:** The team will **never attempt a raw bulk download**.
- We only retrieve the **2D pedestrian-level horizontal slices ($z = 1.75\text{ m}$)** and building height ASCII grids. A curated sample of 50–100 layouts is roughly **50–100 Megabytes**, which easily fits into local storage while remaining gitignored.

### 7.2 Holdout Strategy (Zero Spatial Leakage)
- **Prohibited:** Random pixel-level splitting. Splitting neighboring pixels within the same city neighborhood into train and test sets causes massive spatial autocorrelation leakage.
- **Enforced Protocol:** Splitting is performed strictly at the **`layout_id`** or **`layout_family`** level:
  - *Standard Split:* 70% Train, 15% Validation, 15% Test (stratified by plan area density $\lambda_p$).
  - *Generalization Holdout:* Train on `idealized_aligned` and `idealized_staggered`; hold out `realistic_neighborhood` to test true out-of-distribution generalization to complex real-world city geometry.

---

## 8. Summary of Data Gaps & Actionable Recommendations

| Identified Data Gap | Severity | Practical Resolution |
|---|---|---|
| **No temperature / heat transfer data in UrbanTALES** | High | Formally document pedestrian wind speed magnitude ($U_{1.75\text{m}}$) as the primary project target. Exclude thermal claims. |
| **~35 TB raw dataset size** | Medium | Download only curated 2D pedestrian slices ($z = 1.75\text{ m}$) for 50–100 layouts via web portal. |
| **Local OpenFOAM solver unexecuted** | Low | Maintain OpenFOAM case (`cfd/cases/simple2d_building_block/`) as a verification reference once Docker is approved. Use UrbanTALES as the primary corpus. |
| **Coordinate orientation in raw downloads unverified** | Low | Implement coordinate orientation validation check during ingestion to ensure wake cavity is downstream of building. |
| **Building interior non-zero velocities** | Low | Enforce strict boolean building masking in `LayoutRasterizer`, `FieldNormalizer`, and loss functions (`MaskedMSELoss`). |
