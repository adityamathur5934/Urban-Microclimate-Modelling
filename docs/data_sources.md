# Data Sources & Provenance Registry

**Author / Role:** Team Member A — Dataset Audit & Data Validation (@Aditya)  
**Date:** October 24, 2026  
**Document Version:** 1.0.0  
**Branch:** `feature/dataset-audit`  
**Status:** Approved Reference Registry  

---

## 1. Overview

This document provides a single, authoritative provenance registry for all data sources, candidate external corpora, experimental benchmarks, simulation setups, and synthetic fixtures utilized across the **Urban Microclimate Surrogate Project**.

Every data asset must be traceable to its original creator, legal license, official access method, and physical assumptions.

---

## 2. Master Data Sources Registry

| Source ID | Name | Category | Primary URL / Access | License / Terms | Variables Provided | Verification Status |
|---|---|---|---|---|---|---|
| `SRC-01` | **UrbanTALES** | Candidate External CFD (LES) | [https://urbantales.vercel.app/](https://urbantales.vercel.app/) | CC BY-NC 4.0 (Non-Commercial, Attribution) | Pedestrian wind speed ($U_{1.75\text{m}}$), velocity vectors $(\bar{u}, \bar{v}, \bar{w})$, TKE ($k$), Reynolds stresses | **Verified** via BAMS DOI: [10.1175/BAMS-D-25-0061.1](https://doi.org/10.1175/BAMS-D-25-0061.1). Temperature absent. |
| `SRC-02` | **CEDVAL Database** | Candidate Experimental (Wind Tunnel) | [CEDVAL Portal](https://www.mi.uni-hamburg.de/en/arbeitsgruppen/meteorologie/forschung/windtunnel/cedval.html) | Free Academic Use (Univ. of Hamburg) | Laser Doppler Anemometry (LDA) velocity profiles, turbulence intensities | **Verified** experimental benchmark (Cases A1-1, B1-1). |
| `SRC-03` | **OpenFOAM 2D Block Case** | Local CFD Simulation Setup | Local: `cfd/cases/simple2d_building_block/` | GPL-3.0 (OpenFOAM open source) | RANS $k$-$\omega$ SST steady fields: $U, p, k, \omega, \nu_t$ | **Setup Verified; Solver Run Unexecuted** (blocked by macOS/Docker). |
| `SRC-04` | **Canonical Aerodynamic Reference** | Analytical Benchmark Fixture | Local: `cfd/outputs/canonical_block_reference.npz` | Project Internal (MIT/Apache compatible) | Pedestrian wind speed, velocity components, TKE, building mask | **Verified Benchmark Fixture** (synthesized from empirical aerodynamics). |
| `SRC-05` | **Software Test Fixtures** | Synthetic Test Fixtures | Local: `tests/fixtures/synthetic_data.py` | Project Internal | Toy building blocks (aligned, staggered, random) with potential-flow wakes | **Verified Software Fixture** (strictly for unit tests). |
| `SRC-06` | **OpenStreetMap & LiDAR** | Urban Morphology | [OpenStreetMap](https://www.openstreetmap.org/) & Global LiDAR portals | ODbL / Open Data Commons | Polygonal building footprints, roof heights, surface roughness | **Underlying source for UrbanTALES geometries**. |

---

## 3. Detailed Source Profiles

### 3.1 SRC-01: UrbanTALES (Primary Target Corpus)
- **Official Title:** UrbanTALES: A Comprehensive LES Dataset for Urban Canopy Layer Turbulence Analyses and Parameterization Development
- **Lead Authors:** Assoc. Prof. Negin Nazarian (UNSW Sydney), Dr. Maarten van Reeuwijk (Imperial College London), Dr. Jiachen Lu, Dr. Mathew Lipson, et al.
- **Peer-Reviewed Reference:** *Bulletin of the American Meteorological Society (BAMS)*, DOI: [10.1175/BAMS-D-25-0061.1](https://doi.org/10.1175/BAMS-D-25-0061.1) (2025).
- **Online Access:** [https://urbantales.vercel.app/](https://urbantales.vercel.app/)
- **Underlying Solver:** uDALES (urban Dutch Atmospheric Large-Eddy Simulation, Fortran 90 / MPI).
- **Physical Scope:** Neutral, isothermal atmospheric boundary layer flow past 538 building configurations (224 idealized block arrays, 314 realistic neighborhood geometries from 62 global cities).
- **License:** CC BY-NC 4.0. Mandatory citation to Nazarian et al. (2025). Commercial use requires author authorization.
- **Physical Limitations:**
  - Strictly aerodynamic: **Zero thermal or temperature variables**.
  - Total 3D dataset size is ~35 TB; only **2D pedestrian slices ($z = 1.75\text{ m}$)** should be downloaded locally.

---

### 3.2 SRC-02: CEDVAL Compilation of Experimental Data for Validation
- **Hosting Organization:** Meteorological Institute, University of Hamburg, Germany.
- **Access URL:** `https://www.mi.uni-hamburg.de/en/arbeitsgruppen/meteorologie/forschung/windtunnel/cedval.html`
- **Validation Cases:**
  - *Case A1-1:* Flow past an isolated surface-mounted cube ($H = W = L = 0.125\text{ m}$ model scale) in a thick boundary layer ($Re \approx 3.7 \times 10^4$).
  - *Case B1-1:* 2D street canyon arrays ($H/W = 1$).
- **Instrumentation:** High-frequency 2-component Laser Doppler Anemometry (LDA). Measurement uncertainty estimated at $\pm 2\%$.
- **Project Role:** Quality reference for validating the recirculation cavity length ($L_r \approx 4.5 H$) and corner acceleration jets.

---

### 3.3 SRC-03: Local OpenFOAM 2D Block Case
- **Location:** `cfd/cases/simple2d_building_block/`
- **Solver:** `simpleFoam` (Steady-state incompressible Navier-Stokes solver using SIMPLE pressure-velocity coupling).
- **Turbulence Closure:** Menter Shear Stress Transport ($k$-$\omega$ SST RANS).
- **Inflow Boundary Condition:** Richards & Hoxey (1993) Atmospheric Boundary Layer equilibrium profile ($U_{\text{ref}} = 5.0\text{ m/s}, z_0 = 0.05\text{ m}$).
- **Mesh:** Multi-block structured hexahedral mesh (`blockMeshDict`), $L_x = 420\text{ m}, L_z = 120\text{ m}, L_y = 1\text{ m}$ (quasi-2D, 1 cell thick).
- **Status:** **Configuration complete; local solver execution deferred.** Developer laptop runs macOS ARM64 without container runtimes. The case is ready to execute on Docker/OrbStack or cloud Linux instances.

---

### 3.4 SRC-04: Canonical Aerodynamic Reference Benchmark
- **Generator Script:** `cfd/scripts/generate_canonical_reference.py`
- **Output Artifacts:** `cfd/outputs/canonical_block_reference.npz`, `cfd/outputs/canonical_block_field.svg`
- **Physical Foundation:** Analytically synthesized flow field incorporating empirical bluff-body aerodynamic parameters (Castro & Robins 1977, Martinuzzi & Tropea 1993, COST Action 732):
  - Inflow logarithmic profile at $U_{\text{ref}} = 5.0\text{ m/s}$ ($u_* \approx 0.3865\text{ m/s}$).
  - Front stagnation divergence at $z \approx 0.65 H$.
  - Lateral corner acceleration jets ($1.3 - 1.5 \times U_{\text{ref}}$).
  - Leeward recirculation cavity extending $4.5 H$ downstream with reverse flow.
  - TKE peak along the separated shear layers.
- **Scientific Integrity Policy:** This dataset is an analytical benchmark fixture. It is clearly designated as `city_or_source="CANONICAL_AERODYNAMIC_BENCHMARK_REFERENCE"` and must never be cited as real sensor data or direct CFD output.

---

### 3.5 SRC-05: Software Test Fixtures
- **Location:** `tests/fixtures/synthetic_data.py`
- **Role:** Deterministic toy samples used exclusively for unit testing shape matching, mask orientation, data loaders, split functions, and normalizers.
- **Scientific Integrity Policy:** Marked with explicit docstring disclaimers: *"DISCLAIMER: All data generated in this module are synthetic mathematical fixtures designed exclusively for software testing. THEY DO NOT REPRESENT REAL CFD OBSERVATIONS."*

---

## 4. Summary of Data Compliance Rules

1. **Never Invent Data:** Fabricating CFD solver runs, convergence logs, or experimental measurements is strictly prohibited.
2. **Explicit Labeling:** Every dataset ingested into `.npz` format must record its `solver_name` and `city_or_source` metadata.
3. **No Temperature Claims:** Because UrbanTALES is wind-only, the project target is established as **pedestrian wind speed magnitude ($U$ at $z = 1.75\text{ m}$)**. Temperature prediction is documented as excluded.
4. **Git Storage Hygiene:** Raw simulation files, large `.npz` corpora, and `.nc` files must remain strictly within `data/` and excluded from Git commits via `.gitignore`.
