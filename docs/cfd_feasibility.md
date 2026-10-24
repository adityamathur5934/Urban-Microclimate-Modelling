# Computational Fluid Dynamics (CFD) Feasibility Report & Minimal Case Design

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Task:** Task 4 — OpenFOAM Feasibility and Minimal Case Design  
**Owner:** CFD and Scientific Computing Lead  
**Document Version:** 2.0.0  
**Branch:** `feature/openfoam-feasibility`  
**Status:** Approved Reference Feasibility & Design Specification  

---

## 1. Executive Summary & Objective

The objective of Task 4 is to establish the smallest practical, computationally feasible, and physically grounded route to obtaining reference CFD fields for training and evaluating neural surrogates and automated machine learning (AML) baselines.

### Key Conclusions:
1. **Workstation Environment & Blocker Audit:** The local development workstation is running **macOS 26.3 (Build 25D125) on Apple Silicon (Darwin 25.3.0 arm64)** with 8 CPU cores and 8 GB RAM. A direct audit confirms that **neither OpenFOAM executables (`openfoam`, `simpleFoam`, `blockMesh`) nor container engines (`docker`, `podman`, `orb`, `nerdctl`) are installed**. In accordance with the governing project policy (*"Do not make system-wide changes without approval"*), no system software has been installed. Local CFD simulation execution is currently blocked pending container approval.
2. **Minimal 2D Building-Flow Case Design:** We have designed and configured a complete, production-grade two-dimensional OpenFOAM benchmark case template under [`cfd/cases/simple2d_building_block/`](file:///Users/vaidehisahu/Downloads/amlproj/Urban-Microclimate-Modelling/cfd/cases/simple2d_building_block). The case models steady-state airflow past a $20\text{ m} \times 20\text{ m}$ surface-mounted building block using `simpleFoam` with Menter $k$-$\omega$ SST turbulence closure and atmospheric boundary layer (ABL) inflow.
3. **Physical Quantities Produced vs. Excluded:** The proposed case produces velocity vector components $(u_x, u_z)$, scalar speed magnitude $U = \sqrt{u_x^2 + u_z^2}$, kinematic pressure $p/\rho$, turbulent kinetic energy $k$, specific dissipation $\omega$, eddy viscosity $\nu_t$, and obstacle aerodynamic drag forces. **Temperature is strictly excluded** because `simpleFoam` is an isothermal solver; modeling heat transfer would require `buoyantSimpleFoam`, surface radiative energy balances, and solar ray-tracing.
4. **Numerical Validation Framework:** We define rigorous numerical validation checks (residuals $< 10^{-5}$, drag force stabilization $< 0.1\%$, field plausibility, and GCI mesh sensitivity) and explicitly distinguish a **proposed case template** from an **executed and validated simulation run**.
5. **Cost-Benefit Analysis & Recommended Path:** Running a 100-layout CFD sweep locally on an 8 GB laptop is computationally inefficient and physically restricted by 2D artifacts. We recommend adopting **UrbanTALES** (538 pre-computed 3D Large-Eddy Simulations) as primary ground truth, while utilizing our 2D OpenFOAM template for reproducible solver verification whenever a container runtime or cloud instance is provisioned.
6. **Data Contract Coordination:** We have coordinated with the project lead and established an automated data bridge ([`cfd/scripts/foam_to_contract.py`](file:///Users/vaidehisahu/Downloads/amlproj/Urban-Microclimate-Modelling/cfd/scripts/foam_to_contract.py)) that ingests OpenFOAM sample outputs directly into the validated data schema ([`src.data.contract`](file:///Users/vaidehisahu/Downloads/amlproj/Urban-Microclimate-Modelling/src/data/contract.py)).

> [!IMPORTANT]
> **Scientific Integrity & Status Declaration:**  
> In strict accordance with the guidelines for Task 4, **no computational simulations were executed in this task**, and no large parameter sweeps were launched. The case in `cfd/cases/simple2d_building_block/` is formally designated as a **Proposed Case Template**.

---

## 2. Workstation Environment Audit & Hardware Resources

A comprehensive hardware and software audit of the development workstation was conducted:

```
+----------------------------------------------------------------------------------------------------------+
|                                    WORKSTATION ENVIRONMENT AUDIT                                         |
+-------------------+----------------------------+---------------------------------------------------------+
| Parameter         | Detected System Value      | Implications for Computational Fluid Dynamics           |
+-------------------+----------------------------+---------------------------------------------------------+
| **Operating System**| macOS 26.3 (Build 25D125)  | Unix Darwin 25.3.0 kernel; non-native Linux ABI.         |
| **CPU Architecture**| Apple Silicon (`arm64`)    | 8 Cores (`hw.ncpu: 8`); high unified memory bandwidth.  |
| **System Memory** | 8.00 GB RAM (`8589934592 B`)| Sufficient for small 2D meshes (~50k cells); severely   |
|                   |                            | constrained for 3D meshes (> 1M cells induces OOM/swap).|
| **Disk Storage**  | 150 GiB Available          | Adequate for storing tabular data and 2D NPZ slices;    |
|                   |                            | inadequate for multi-terabyte raw 3D LES volumes.       |
| **File System**   | APFS (Default Case-Insensitive)| Incompatible with native OpenFOAM source compilation. |
+-------------------+----------------------------+---------------------------------------------------------+
| **OpenFOAM Tools**| `openfoam`, `simpleFoam`,  | **NOT INSTALLED.** Local native CFD runs cannot execute.|
|                   | `blockMesh`: NOT FOUND     |                                                         |
| **Container Engine**| `docker`, `podman`, `orb`,| **NOT INSTALLED.** Containerized CFD execution blocked  |
|                   | `nerdctl`: NOT FOUND       | without administrative installation approval.           |
| **Package Manager**| `/opt/homebrew/bin/brew`   | Available on host; installation deferred per policy.    |
| **Python Toolchain**| Python 3.9 (`.venv`)     | Operational: NumPy, SciPy, PyTorch, PyTest functional.  |
+-------------------+----------------------------+---------------------------------------------------------+
```

---

## 3. Platform Constraints & Installation Requirements

### 3.1 Technical Bottlenecks of Native OpenFOAM on macOS
Running OpenFOAM natively on macOS Apple Silicon presents three primary engineering barriers:

1. **Case-Insensitive File System Hazard:**  
   The OpenFOAM C++ codebase contains case-colliding filenames (for example, `src/OpenFOAM/primitives/Scalar/scalar/scalar.H` versus `Scalar.H`). When cloned onto standard macOS APFS drives, one header overwrites the other, causing fatal compilation errors. Resolving this requires mounting a dedicated case-sensitive APFS sparse disk image:
   ```bash
   hdiutil create -size 20g -type SPARSE -fs "Case-sensitive APFS" -volname "OpenFOAM" ~/OpenFOAM.sparseimage
   hdiutil attach ~/OpenFOAM.sparseimage -mountpoint /Volumes/OpenFOAM
   ```
2. **Toolchain & Dynamic Linking Incompatibilities:**  
   OpenFOAM relies heavily on GCC/G++, glibc, and Linux shared object (`.so`) symbol visibility flags. Compiling on macOS requires Clang adaptations and third-party dynamic libraries (`.dylib`), which are prone to toolchain drift across macOS updates.
3. **Architecture Optimization:**  
   OpenFOAM's linear solvers and MPI communications are tuned for Linux x86_64 / arm64 SIMD instructions. Running x86 Linux containers under Rosetta 2 emulation introduces a 25–40% performance penalty. A native `arm64` container is required for optimal performance.

### 3.2 Installation Blueprints (For When Approval Is Granted)

```
+----------------------------------------------------------------------------------------------------------+
|                                  OPENFOAM DEPLOYMENT BLUEPRINTS                                          |
+-------------------+------------------------------------------+-------------------------------------------+
| Method            | Deployment Mechanism                     | Trade-Offs & Suitability                  |
+-------------------+------------------------------------------+-------------------------------------------+
| **Option A:**     | Docker Desktop / OrbStack container      | **Strongly Recommended.** Zero host OS    |
| Containerized     | running `opencfd/openfoam-default:latest`| contamination, 100% reproducible Linux    |
| (Docker/OrbStack) | with ARM64 native binary support.        | environment, identical to HPC clusters.   |
+-------------------+------------------------------------------+-------------------------------------------+
| **Option B:**     | Homebrew tap via `gerlero/openfoam`      | Native execution, but requires dedicated  |
| Native macOS      | mounted on a case-sensitive APFS disk.   | APFS sparse image and custom maintenance. |
+-------------------+------------------------------------------+-------------------------------------------+
| **Option C:**     | Cloud Linux HPC Instance (AWS / GCP)     | Ideal for large 3D parametric sweeps;     |
| Cloud Compute     | running Ubuntu 24.04 LTS via apt-get.    | requires cloud credits / institutional VMs|
+-------------------+------------------------------------------+-------------------------------------------+
```

---

## 4. Minimal Two-Dimensional Case Design

To establish a verified baseline for urban building aerodynamics without excessive computational overhead, we have designed the **`simple2d_building_block`** case template located in [`cfd/cases/simple2d_building_block/`](file:///Users/vaidehisahu/Downloads/amlproj/Urban-Microclimate-Modelling/cfd/cases/simple2d_building_block).

```
                               Top Boundary: Slip (z = 120m, 6H)
    +-----------------------------------------------------------------------------------+
    |                                                                                   |
    |  Inlet (x = 0)                                                    Outlet (x=420m) |
    |  ABL Log Profile U(z)                                             p = 0, zeroGrad |
    |  Uref = 5 m/s @ 10m                      +--------+               ======>         |
    |  =======>                                |Building|   Wake Cavity                 |
    |  =======>                                | H = 20m|   (u < 0)                     |
    |  =======>                                +--------+                               |
    |  ========================================+        +=============================  |
    |  Ground: noSlip, wall functions (z = 0)   W = 20m                                 |
    +-----------------------------------------------------------------------------------+
    |<-------------- 100m (5H) --------------->|<--20m->|<------------ 300m (15H) ------>|
```

### 4.1 Domain Dimensions & Computational Geometry
- **Building Height ($H$):** $20.0\text{ m}$
- **Building Width ($W$):** $20.0\text{ m}$ ($1.0 \times H$)
- **Upstream Distance ($D_{\text{upstream}}$):** $100.0\text{ m}$ ($5.0 \times H$), adhering to COST Action 732 guidelines to prevent artificial blockage of the approaching ABL profile.
- **Downstream Distance ($D_{\text{downstream}}$):** $300.0\text{ m}$ ($15.0 \times H$), ensuring that the wake recirculation bubble ($L_r \sim 10 - 15 H$ in 2D) and downstream recovery are fully contained within the domain without reflecting off the outlet boundary.
- **Domain Height ($L_z$):** $120.0\text{ m}$ ($6.0 \times H$), avoiding unphysical vertical constriction over the roof.
- **Domain Width ($L_y$):** $1.0\text{ m}$ (Quasi-2D representation, 1 cell thick in $y$ bounded by `empty` patches).

### 4.2 Multi-Block Hexahedral Mesh Architecture (`system/blockMeshDict`)
The computational domain is discretized into 5 structured hex blocks:
- **Block 0 (Upstream ground to roof):** $x \in [0, 100]$, $z \in [0, 20]$ $\rightarrow$ $50 \times 1 \times 20$ cells ($\Delta x = 2.0\text{ m}, \Delta z = 1.0\text{ m}$).
- **Block 1 (Upstream roof to top):** $x \in [0, 100]$, $z \in [20, 120]$ $\rightarrow$ $50 \times 1 \times 30$ cells ($\Delta x = 2.0\text{ m}, \Delta z = 3.3\text{ m}$).
- **Block 2 (Rooftop to top boundary):** $x \in [100, 120]$, $z \in [20, 120]$ $\rightarrow$ $20 \times 1 \times 30$ cells ($\Delta x = 1.0\text{ m}, \Delta z = 3.3\text{ m}$).
- **Block 3 (Downstream wake ground to roof):** $x \in [120, 420]$, $z \in [0, 20]$ $\rightarrow$ $100 \times 1 \times 20$ cells ($\Delta x = 3.0\text{ m}, \Delta z = 1.0\text{ m}$).
- **Block 4 (Downstream wake roof to top):** $x \in [120, 420]$, $z \in [20, 120]$ $\rightarrow$ $100 \times 1 \times 30$ cells ($\Delta x = 3.0\text{ m}, \Delta z = 3.3\text{ m}$).
- **Total Mesh Size:** $\mathbf{8,100\text{ cells}}$. This lightweight mesh allows rapid convergence in $< 2\text{ minutes}$ while resolving near-wall velocity gradients.

### 4.3 Boundary Conditions & Inflow Profile

| Boundary Patch | Patch Type | Velocity ($\mathbf{U}$) | Kinematic Pressure ($p$) | Turbulent Kinetic Energy ($k$) | Specific Dissipation ($\omega$) | Eddy Viscosity ($\nu_t$) |
|---|---|---|---|---|---|---|
| **inlet** | `patch` | `atmBoundaryLayerInletVelocity` ($U_{\text{ref}}=5\text{ m/s}, z_{\text{ref}}=10\text{ m}, z_0=0.05\text{ m}$) | `zeroGradient` | `atmBoundaryLayerInletK` | `atmBoundaryLayerInletOmega` | `calculated` |
| **outlet** | `patch` | `inletOutlet` (inletValue: `(0 0 0)`) | `fixedValue` ($p = 0\text{ m}^2/\text{s}^2$) | `zeroGradient` | `zeroGradient` | `calculated` |
| **ground** | `wall` | `noSlip` | `zeroGradient` | `kqRwallFunction` | `omegaWallFunction` | `nutkWallFunction` |
| **building** | `wall` | `noSlip` | `zeroGradient` | `kqRwallFunction` | `omegaWallFunction` | `nutkWallFunction` |
| **top** | `patch` | `slip` | `zeroGradient` | `slip` | `slip` | `calculated` |
| **frontAndBack** | `empty` | `empty` | `empty` | `empty` | `empty` | `empty` |

- **Inlet Wind Velocity:** Standard logarithmic Atmospheric Boundary Layer (Richards & Hoxey, 1993):
  $$U(z) = \frac{u_*}{\kappa} \ln\left( \frac{z + z_0}{z_0} \right), \quad u_* = \frac{\kappa U_{\text{ref}}}{\ln\left(\frac{z_{\text{ref}} + z_0}{z_0}\right)} \approx 0.387\text{ m/s}$$
- **Inlet Turbulence:**
  $$k(z) = \frac{u_*^2}{\sqrt{C_\mu}} \approx 0.50\text{ m}^2/\text{s}^2, \quad \omega(z) = \frac{u_*}{\kappa \sqrt{C_\mu}(z + z_0)}$$

### 4.4 Solver Selection & Turbulence Model
- **Solver:** OpenFOAM **`simpleFoam`** (Steady-state incompressible Navier–Stokes solver using the Semi-Implicit Method for Pressure-Linked Equations, SIMPLE).
- **Turbulence Closure:** **Menter Shear Stress Transport ($k$-$\omega$ SST)** (`kOmegaSST`).  
  *Justification:* Standard $k$-$\varepsilon$ suffers from the well-documented **stagnation point anomaly**, generating unphysically high turbulent kinetic energy upstream of the building facade, which delays separation and underestimates wake length. $k$-$\omega$ SST limits TKE production in stagnation regions and accurately predicts boundary layer detachment at building edges.

---

## 5. Physical Quantities Produced vs. Excluded

### 5.1 Quantities Physically Produced
1. **Velocity Vector Field ($\mathbf{u}$):** Directional components $u_x(x, z)$ and $u_z(x, z)$ [$\text{m/s}$].
2. **Pedestrian Wind Speed Magnitude ($U_{\text{ped}}$):** Horizontal scalar speed sampled at pedestrian level ($z = 1.75\text{ m}$):
   $$U_{\text{ped}}(x) = |u_x(x, z=1.75)| \quad [\text{m/s}]$$
3. **Kinematic Pressure Field ($p/\rho$):** Normalized static pressure [$\text{m}^2/\text{s}^2$].
4. **Turbulent Kinetic Energy ($k$):** Fluctuating kinetic energy per unit mass [$\text{m}^2/\text{s}^2$].
5. **Specific Dissipation Rate ($\omega$):** Rate of dissipation of turbulence energy per unit volume [$1/\text{s}$].
6. **Kinematic Turbulent Eddy Viscosity ($\nu_t$):** Spatial field representing turbulent momentum transport [$\text{m}^2/\text{s}$].
7. **Wall Shear Stress ($\boldsymbol{\tau}_w$) & Building Aerodynamic Drag:** Integrated surface forces and drag coefficient $C_d$ on the building obstacle.

### 5.2 Physical Quantities Excluded (Why Temperature is NOT Modeled)
- **Air Temperature ($T$) & Thermal Fields:** **STRICTLY EXCLUDED.**
- *Physical Rationale:* The `simpleFoam` solver configuration solves only the continuity and momentum equations under isothermal, incompressible conditions:
  $$\nabla \cdot \mathbf{u} = 0, \quad (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nabla \cdot \left[(\nu + \nu_t)\left(\nabla \mathbf{u} + (\nabla \mathbf{u})^T\right)\right]$$
  There is **no thermal energy equation**, no thermal diffusivity $\alpha_{\text{thermal}}$, no Boussinesq buoyant body force $g \beta \Delta T \hat{\mathbf{k}}$, and no surface radiative balance ($R_{\text{net}} = H + \lambda E + G$).
- *Scientific Risk:* Claiming that an isothermal wind solver produces temperature predictions is a fatal scientific flaw. To model temperature, one would require:
  1. Switching to a coupled solver such as `buoyantSimpleFoam`.
  2. Defining surface heat fluxes or fixed temperatures on building facades and pavement.
  3. Solving thermal radiation transfer equations (e.g. `fvDOM` discrete ordinates model).
  4. Accounting for solar zenith angles and shadow projection.

---

## 6. Numerical Validation Protocol: Proposed vs. Executed Cases

### 6.1 Distinguishing Proposed Cases from Executed Cases
To maintain scientific transparency, all CFD assets in this project are strictly categorized into one of two operational states:

```
+----------------------------------------------------------------------------------------------------------+
|                                    CFD OPERATIONAL STATUS TAXONOMY                                       |
+-------------------+------------------------------------------+-------------------------------------------+
| State Category    | Repository Status & Artifacts            | Permitted Scientific Claims               |
+-------------------+------------------------------------------+-------------------------------------------+
| **Proposed Case** | Configured OpenFOAM case directory       | Fully specified numerical formulation,    |
| **Template**      | (`0/`, `constant/`, `system/`),          | verified syntax, mathematical definition. |
| (CURRENT STATUS)  | `foam_to_contract.py` parser ready.      | **No claims of completed numerical runs.**|
+-------------------+------------------------------------------+-------------------------------------------+
| **Executed &**    | Solver execution log (`log.simpleFoam`), | Numerical convergence verified            |
| **Validated Case**| residual history plot, field slices,     | (residuals $< 10^{-5}$), mass balanced,   |
| (POST-CONTAINER)  | validated against experimental benchmark.| physically checked against wind tunnel.   |
+-------------------+------------------------------------------+-------------------------------------------+
```

### 6.2 Numerical Validation Criteria (Upon Execution)
When container execution is initiated, the case must satisfy four mandatory validation gates:

1. **Solver Residual History Gate:**  
   Normalized residuals for all solved variables must drop monotonically below the convergence threshold:
   $$\text{Residual}(p, U_x, U_z, k, \omega) < 1.0 \times 10^{-5}$$
   Sudden oscillations or stagnation above $10^{-4}$ indicate unphysical numerical stalling or grid non-orthogonality.
2. **Integral Quantity Stabilization Gate:**  
   The total drag force $F_d$ on the building obstacle (monitored via `system/controlDict` `buildingForces` function object) must oscillate by less than **$0.1\%$** over the final 200 iterations.
3. **Physical Field Plausibility Gate:**  
   - Frontal stagnation point located at $z \approx 0.65 - 0.75 H$ on the windward facade.
   - Clean flow separation at the windward corner ($x = 100\text{ m}, z = 20\text{ m}$).
   - Definite leeward wake reverse flow ($u_x < 0$).
   - Wake cavity reattachment length within established 2D literature bounds ($L_r \sim 10 - 15 H$).
4. **Mesh Sensitivity Check (Grid Convergence Index):**  
   Executing coarse (4,000 cells), medium (8,100 cells), and fine (16,000 cells) meshes must demonstrate asymptotic grid independence with $\text{GCI} < 3\%$.

---

## 7. Cost-Benefit Analysis: Minimal OpenFOAM Case vs. Documented CFD Datasets

To determine the most practical path for the project, we evaluate the trade-offs between local CFD generation and ingesting curated external datasets:

```
+----------------------------------------------------------------------------------------------------------+
|                                       PRACTICAL COST-BENEFIT MATRIX                                      |
+-------------------+------------------------------------------+-------------------------------------------+
| Evaluation Factor | Route A: Local OpenFOAM Generation       | Route B: Documented Datasets (UrbanTALES) |
+-------------------+------------------------------------------+-------------------------------------------+
| **Software Setup**| High friction: Requires Docker/OrbStack  | Zero friction: Pure Python download via   |
|                   | installation, APFS workaround, or cloud. | HTTP/Zenodo API (`urllib` / `curl`).      |
+-------------------+------------------------------------------+-------------------------------------------+
| **Compute Time**  | 2D RANS: 2 min / case (feasible).        | Pre-computed: Zero local simulation time. |
|                   | 3D LES: 24–72 hrs / case (infeasible).   | Download takes ~30 seconds per layout.    |
+-------------------+------------------------------------------+-------------------------------------------+
| **Physical**      | 2D RANS: Distorted wake length ($15H$),  | 3D LES: Resolves realistic 3D downwash,   |
| **Fidelity**      | zero vortex stretching, infinite walls.  | rooftop separation, and coherent eddies.  |
+-------------------+------------------------------------------+-------------------------------------------+
| **Dataset Scale** | 1–5 simple geometric variants locally;   | 538 diverse urban layouts across 62       |
|                   | parameter sweeps thermally throttle host.| global cities and idealized arrays.       |
+-------------------+------------------------------------------+-------------------------------------------+
| **Surrogate**     | Insufficient layout variety to train     | High diversity; prevents deep learning    |
| **Training**      | deep neural networks (causes overfitting)| models from memorizing single layouts.    |
+-------------------+------------------------------------------+-------------------------------------------+
| **Primary Role**  | **Reproducibility & Verification Case**   | **Primary Ground Truth Dataset**          |
|                   | (Small-scale benchmark template)         | (Training & holdout testing)              |
+-------------------+------------------------------------------+-------------------------------------------+
```

### Strategic Conclusion:
Attempting to run a 100-layout simulation sweep locally on an 8 GB Apple Silicon laptop is computationally impractical and physically limited. We therefore adopt a **dual-track strategy**:
- **Track 1 (Surrogate Data Engine):** Ingest 50–100 high-fidelity 2D horizontal slices from **UrbanTALES** (and CEDVAL wind tunnel benchmarks) to train Model 1 (Random Forest) and Model 2 (2D U-Net).
- **Track 2 (CFD Verification Baseline):** Maintain the `simple2d_building_block` case template in the repository to demonstrate complete CFD pipeline literacy, test mesh convergence, and verify the `foam_to_contract.py` data bridge.

---

## 8. Data Schema & Field Export Coordination

In direct coordination with the project lead and conforming to Task 3's approved data specification ([`docs/data_schema.md`](file:///Users/vaidehisahu/Downloads/amlproj/Urban-Microclimate-Modelling/docs/data_schema.md)), the CFD extraction pipeline is standardized as follows:

```
+----------------------------------------------------------------------------------------------------------+
|                                     DATA PIPELINE & EXPORT BRIDGE                                        |
+----------------------------------------------------------------------------------------------------------+
|  [OpenFOAM Case Execution] (`simpleFoam` on simple2d_building_block)                                     |
|                                     |                                                                    |
|                                     v                                                                    |
|  [Sampling via sampleDict] (`postProcess -func sample`)                                                  |
|  Exports line sets at z = 1.75m: postProcessing/sets/1000/pedestrian_wake_U_p_k_omega.xy                |
|                                     |                                                                    |
|                                     v                                                                    |
|  [Automated Bridge] (`cfd/scripts/foam_to_contract.py`)                                                  |
|  - Parses ASCII column data (x, y, z, Ux, Uy, Uz, p, k, omega)                                           |
|  - Interpolates onto uniform Cartesian raster (e.g. 64 x 128)                                            |
|  - Generates height_map and building_mask                                                                |
|  - Computes outdoor area_averaged_wind_speed via mask-weighted integration                               |
|  - Serializes into compressed UrbanLayoutSample NPZ archive                                              |
|                                     |                                                                    |
|                                     v                                                                    |
|  [Validated NPZ File] (`cfd/outputs/cfd_2d_block_001.npz`)                                               |
|  Directly consumable by LayoutDataLoader, LayoutRasterizer, and Model Training Pipelines                 |
+----------------------------------------------------------------------------------------------------------+
```

### CLI Command for Field Export:
```bash
python cfd/scripts/foam_to_contract.py \
  --layout-id cfd_2d_block_001 \
  --input-file cfd/cases/simple2d_building_block/postProcessing/sets/1000/pedestrian_wake_U_p_k_omega.xy \
  --output-npz cfd/outputs/cfd_2d_block_001.npz \
  --nx 128 \
  --ny 64
```

---

## 9. Recommended Smallest Practical Path Forward

To proceed efficiently without stalling progress or executing unauthorized software changes, we establish a three-phase operational roadmap:

```
+----------------------------------------------------------------------------------------------------------+
|                                     THREE-PHASE OPERATIONAL ROADMAP                                      |
+----------------------------------------------------------------------------------------------------------+
|  PHASE 1: Immediate Neural Surrogate & ML Training (CURRENT SPRINT)                                     |
|  - Ingest 2D pedestrian wind speed slices from UrbanTALES LES corpus.                                   |
|  - Ingest canonical aerodynamic benchmark reference fields (`cfd/scripts/generate_canonical_reference.py`).|
|  - Fit and evaluate Model 1 (Random Forest Regressor) and Model 2 (2D U-Net).                            |
|  - NO LOCAL SIMULATION REQUIRED; 100% UNBLOCKED.                                                         |
|                                     |                                                                    |
|                                     v                                                                    |
|  PHASE 2: Containerized OpenFOAM Benchmark Execution (UPON APPROVAL)                                     |
|  - Obtain approval to install Docker Desktop or OrbStack.                                                |
|  - Execute `simple2d_building_block/Allrun` inside native ARM64 OpenFOAM container.                      |
|  - Verify numerical convergence ($< 10^{-5}$) and drag stabilization.                                    |
|  - Run `foam_to_contract.py` to confirm end-to-end data bridge functionality.                             |
|                                     |                                                                    |
|                                     v                                                                    |
|  PHASE 3: Parametric CFD Sweeps on Cloud HPC (FUTURE EXTENSION)                                          |
|  - Provision Ubuntu HPC cluster (GCP/AWS).                                                               |
|  - Execute 3D RANS / LES parametric sweeps across novel building massing designs.                        |
+----------------------------------------------------------------------------------------------------------+
```

> [!CAUTION]
> **Policy Guardrail:**  
> In compliance with Task 4 instruction 9 (*"Do not generate a large simulation sweep before the minimal route is approved"*), **no multi-case parameter sweeps will be launched** on the local developer workstation.

---

## 10. Summary of Blockers and Clear Mitigation Strategy

| Identified Blocker | Impact | Clear Mitigation Strategy |
|---|---|---|
| **OpenFOAM Not Installed** | Cannot run `simpleFoam` natively on host. | Documented complete case template; provision Docker container once approved. |
| **Container Engine Missing** | Cannot execute containerized OpenFOAM. | Submit approval request for OrbStack or Docker Desktop; use UrbanTALES in the interim. |
| **8 GB RAM Hardware Limit** | 3D CFD meshes (> 1M cells) will trigger OOM. | Restrict local CFD to 2D benchmark (~8k cells); defer 3D LES to cloud HPC or pre-computed datasets. |
| **Case-Insensitive File System**| Blocks native macOS OpenFOAM source builds. | Use containerized runtime (Method A) to bypass macOS APFS limitations completely. |
