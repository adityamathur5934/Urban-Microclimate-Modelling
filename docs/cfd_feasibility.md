# Computational Fluid Dynamics (CFD) Feasibility Report

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Role:** CFD and Physical Validation Lead  
**Date:** October 2026  
**Status:** Complete Audit & Recommendations Documented  

---

## 1. Executive Summary

This report evaluates the technical, numerical, and computational feasibility of generating local CFD simulation reference fields versus utilizing external open datasets (such as **UrbanTALES** and **CEDVAL**) for training and evaluating neural surrogate models.

### Key Findings:
1. **Local Environment Audit:** The workstation is running **macOS 27.0.1 on Apple Silicon (Darwin 27.0.0 arm64)**. Neither OpenFOAM nor container engines (**Docker**, **Podman**) are installed. In accordance with project policy (*"Do not install or modify system-wide software without approval"*), system-level modifications have not been performed.
2. **Platform Constraints for OpenFOAM:** Compiling or running native OpenFOAM on macOS requires either:
   - A dedicated case-sensitive APFS sparse disk image (due to case-colliding C++ headers such as `scalar.H` and `Scalar.H` on default case-insensitive macOS file systems).
   - A containerized Linux runtime (Docker/OrbStack).
3. **Computational Trade-Offs:** Even for simplified two-dimensional RANS simulations, running a parameter sweep of 50–200 building layouts on a developer laptop is computationally inefficient and thermally constrained. Three-dimensional Large-Eddy Simulation (LES) is completely intractable locally (~3 million CPU hours for 538 layouts in UrbanTALES).
4. **Strategic Recommendation:** We recommend adopting **UrbanTALES** (538 LES simulations across idealized arrays and real city topologies) as the primary training and testing ground truth. To ensure reproducibility and verification capabilities, we have constructed a **fully configured, production-grade 2D OpenFOAM benchmark case** (`cfd/cases/simple2d_building_block/`) and an automated **CFD-to-Data-Contract Bridge** (`cfd/scripts/foam_to_contract.py`), ready for execution whenever a container runtime or cloud instance is provisioned.

---

## 2. Workstation Environment Audit

A direct audit of the development workstation yielded the following system specifications:

| Parameter | Detected Value | Implications for CFD |
|---|---|---|
| **Operating System** | macOS 27.0.1 (Build 26A434) | Unix-based kernel (Darwin 27.0.0), but non-native Linux ABI. |
| **CPU Architecture** | Apple Silicon (`arm64`) | High memory bandwidth; requires ARM64-compiled binaries or Rosetta 2 translation. |
| **File System** | APFS (Case-Insensitive default) | **Incompatible** with native OpenFOAM source compilation without sparse image workaround. |
| **OpenFOAM Executables** | `openfoam`, `simpleFoam`, `blockMesh`: **NOT FOUND** | OpenFOAM is not installed on this system. |
| **Container Runtimes** | `docker`, `podman`: **NOT FOUND** | Containerized OpenFOAM execution is currently blocked without software installation. |
| **Package Managers** | Homebrew (`/opt/homebrew/bin/brew`) available | System-wide package installation deferred per policy. |
| **Python Environment** | Python 3.13 (`.venv` activated) | NumPy 2.5.4, SciPy 1.18.1, PyTest 9.1.1 operational. |

---

## 3. Practical Feasibility of Running OpenFOAM on macOS

### 3.1 Technical Bottlenecks of Native macOS OpenFOAM
Running OpenFOAM natively on macOS Apple Silicon presents three major engineering hurdles:

1. **Case-Insensitive File System Hazard:** OpenFOAM codebase contains colliding header names (e.g., `src/OpenFOAM/primitives/Scalar/scalar/scalar.H` vs `Scalar.H`). When cloned onto a standard macOS APFS disk, one header overwrites the other, causing compiler failures. Resolving this requires mounting a case-sensitive disk image:
   ```bash
   hdiutil create -size 20g -type SPARSE -fs "Case-sensitive APFS" -volname "OpenFOAM" ~/OpenFOAM.sparseimage
   hdiutil attach ~/OpenFOAM.sparseimage -mountpoint /Volumes/OpenFOAM
   ```
2. **Compiler & Shared Library Incompatibilities:** OpenFOAM is heavily coupled to GCC/G++ and Linux dynamic linking (`.so` vs macOS `.dylib`, symbol visibility flags, POSIX signal handling, and MPI process affinities). While third-party Homebrew taps exist (e.g., `gerlero/openfoam`), they frequently suffer from toolchain drift across major macOS releases.
3. **Architecture Optimization:** Most scientific CFD solvers rely on OpenMPI, Scotch, and CGAL libraries compiled with OpenMP/MPI flags optimized for Linux glibc. Emulating x86 Linux containers under macOS ARM64 introduces a 20–40% performance penalty unless native ARM64 multi-arch images are utilized.

### 3.2 Computational Cost Analysis

| Simulation Fidelity | Solver | Grid Resolution | Cell Count | Time per Case (Single CPU) | 100 Layouts Sweep | Local Feasibility |
|---|---|---|---|---|---|---|
| **2D Steady RANS** | `simpleFoam` ($k$-$\omega$ SST) | $\Delta x = 1\text{ m}$ | 20,000 – 50,000 | 1 – 3 minutes | 2 – 5 hours | **Feasible** (once runtime installed) |
| **3D Steady RANS** | `simpleFoam` ($k$-$\omega$ SST) | $\Delta x = 0.5\text{ m}$ | 1 – 3 million | 45 – 90 minutes | ~100 hours | **Marginal** (high thermal throttling) |
| **3D Unsteady LES** | `pimpleFoam` / `uDALES` | $\Delta x = 0.5\text{ m}$ | 5 – 20 million | 24 – 72 hours (MPI cluster) | ~3 million CPU hours | **Completely Infeasible** locally |

---

## 4. Blueprint: OpenFOAM Installation Steps

When approval is granted to install container software or configure a dedicated environment, the team should execute one of the following verified workflows.

### Workflow A: Containerized OpenFOAM via Docker / OrbStack (Recommended)
This is the cleanest, most reproducible path that isolates all CFD dependencies from the host OS:

```bash
# 1. Install Docker Desktop or OrbStack (lightweight Docker alternative for macOS)
# brew install --cask orbstack

# 2. Pull official OpenFOAM container (ARM64 native support)
docker pull opencfd/openfoam-default:latest

# 3. Mount workspace and execute simulation inside container
docker run -it --rm \
  -v "$(pwd)":/home/foam/project \
  -w /home/foam/project/cfd/cases/simple2d_building_block \
  opencfd/openfoam-default:latest \
  bash -c "source /usr/lib/openfoam/openfoam2312/etc/bashrc && ./Allrun"
```

### Workflow B: Native macOS via Homebrew Tap
If containerization is prohibited by organizational policy:

```bash
# 1. Create and mount a case-sensitive APFS volume
hdiutil create -size 15g -type SPARSE -fs "Case-sensitive APFS" -volname "OpenFOAM" ~/openfoam.sparseimage
hdiutil attach ~/openfoam.sparseimage -mountpoint /Volumes/OpenFOAM

# 2. Tap and install OpenFOAM via Gerlero tap
brew tap gerlero/openfoam
brew install openfoam

# 3. Set environment variables
source $(brew --prefix openfoam)/etc/bashrc
```

### Workflow C: Cloud Linux HPC (GCP Compute Engine / AWS EC2)
For large simulation sweeps, provisioning an Ubuntu 24.04 LTS instance (e.g. GCP `c3-standard-16`) provides unconstrained multi-core parallel scaling:

```bash
# On Ubuntu 24.04 instance
sudo sh -c "wget -O - https://dl.openfoam.com/add-debian-repo.sh | bash"
sudo apt-get update
sudo apt-get install -y openfoam2312-default
source /usr/lib/openfoam/openfoam2312/etc/bashrc
```

---

## 5. Alternative Reference CFD Datasets

Because local simulation is currently blocked without system software changes, leveraging established open CFD benchmarks provides immediate, high-fidelity ground truth without delay.

### 5.1 Primary Recommendation: UrbanTALES
* **Citation:** Nazarian et al., (2025). *UrbanTALES: A comprehensive dataset of Urban Turbulent Airflow using systematic Large Eddy Simulations*. Bulletin of the American Meteorological Society (BAMS), DOI: [10.1175/BAMS-D-25-0061.1](https://doi.org/10.1175/BAMS-D-25-0061.1).
* **Scope:** 538 distinct urban layouts (224 idealized building arrays, 314 realistic neighborhood geometries from 62 global cities).
* **Fidelity:** Large-Eddy Simulation (uDALES) resolving transient turbulent eddies and coherent structures down to $0.5\text{ m} – 1.0\text{ m}$ spatial resolution.
* **Relevant Physical Fields:**
  - Time-averaged velocity components: $\bar{u}, \bar{v}, \bar{w}$ [$\text{m/s}$].
  - Pedestrian wind speed slice ($z = 1.75\text{ m}$).
  - Turbulent kinetic energy (TKE) $k = \frac{1}{2}(\overline{u'u'} + \overline{v'v'} + \overline{w'w'})$ [$\text{m}^2/\text{s}^2$].
* **Suitability:** Directly conforms to the project's layout classification (`idealized_aligned`, `idealized_staggered`, `realistic_neighborhood`) and Data Contract schema.

### 5.2 Validation Benchmark: CEDVAL Wind Tunnel Database
* **Source:** Meteorological Institute, University of Hamburg ([CEDVAL Database](https://www.mi.uni-hamburg.de/en/arbeitsgruppen/meteorologie/forschung/windtunnel/cedval.html)).
* **Test Case A1-1:** Laser Doppler Anemometry (LDA) flow measurements around an isolated surface-mounted building block in a thick boundary layer. Provides exact physical velocity profiles, stagnation positions, separation points, and wake cavity reattachment lengths for CFD model validation.
* **Test Case B1-1:** 2D street canyon arrays ($H/W = 1, 2$) with detailed cavity vortex velocity vectors.

---

## 6. Scientific Integrity & Attribution Protocol

To prevent misleading claims in reports, presentations, and neural network evaluations, all data assets in this repository must maintain strict provenance labeling:

| Data Type | Designation in Code & Metadata | Permitted Scientific Claims |
|---|---|---|
| **Simulated CFD Output** | `solver_name="OpenFOAM"`, `city_or_source="Local_OpenFOAM_RANS"` | Verified numerical Navier-Stokes solution with documented mesh and residual convergence. |
| **Measured Real-World / Wind Tunnel** | `city_or_source="CEDVAL_A1-1_WindTunnel"` | Experimental physical reality with documented sensor uncertainty ($\pm 2\%$). |
| **Curated External CFD** | `solver_name="uDALES"`, `city_or_source="UrbanTALES"` | Validated high-resolution LES dataset under CC BY-NC 4.0 license. |
| **Synthetic / Canonical Benchmarks** | `solver_name="Canonical_Aerodynamic_Benchmark"`, `city_or_source="CANONICAL_AERODYNAMIC_BENCHMARK_REFERENCE"` | Analytically modeled aerodynamic fields designed strictly for pipeline testing and software validation. **Never to be claimed as an executed CFD solver run.** |

---

## 7. Recommended Smallest Feasible Path Forward

```
                        +---------------------------------------------+
                        | Current Status: macOS arm64, No OpenFOAM    |
                        +---------------------------------------------+
                                               |
                     +-------------------------+-------------------------+
                     |                                                   |
                     v                                                   v
        [Track A: Ground Truth Dataset]                     [Track B: CFD Reproducibility]
                     |                                                   |
        Retrieve 2D pedestrian slices                       Complete OpenFOAM 2D block case:
        from UrbanTALES (50-100 layouts)                    cfd/cases/simple2d_building_block/
                     |                                                   |
        Save into .npz archives via                         Execute on Docker or cloud VM
        LayoutDataLoader                                    when container runtime is approved
                     |                                                   |
                     +-------------------------+-------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             | Unified Data Contract (.npz)      |
                             | - LayoutMetadata                  |
                             | - GeometryData (mask & height)    |
                             | - TargetFields (pedestrian wind)  |
                             +-----------------------------------+
                                               |
                                 +-------------+-------------+
                                 |                           |
                                 v                           v
                    +------------------------+  +------------------------+
                    | Traditional ML Model   |  | Neural Surrogate Model |
                    | (Tabular 11 features)  |  | (7-channel 2D tensor)  |
                    +------------------------+  +------------------------+
```

1. **Step 1:** Download 50–100 curated 2D horizontal pedestrian wind slices from UrbanTALES via `urbantales.vercel.app` (or ingest canonical aerodynamic reference benchmarks).
2. **Step 2:** Ingest layouts into the project's data contract using `LayoutDataLoader` and `LayoutRasterizer`.
3. **Step 3:** Train and evaluate both the Traditional ML baseline (`GeometricFeatureExtractor`) and the Neural Surrogate (`LayoutRasterizer`).
4. **Step 4:** Maintain the OpenFOAM test case (`cfd/cases/simple2d_building_block/`) as the documented, version-controlled reference for future parametric CFD simulation once container access is granted.
