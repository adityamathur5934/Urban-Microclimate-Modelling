# CFD Simulation & Physical Validation Methodology

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Role:** CFD and Physical Validation Lead  
**Document Version:** 1.0.0  
**Status:** Approved Reference Methodology  

---

## 1. Scope and Theoretical Objectives

The objective of this methodology is to define a mathematically rigorous, physically validated protocol for generating and verifying computational fluid dynamics (CFD) reference fields around urban obstacle configurations. 

These reference flow fields serve as ground truth targets for:
1. **Model 1: Traditional ML Baseline** (extracting 11 tabular morphological features to predict scalar canopy wind speed).
2. **Model 2: Neural Surrogate** (using 7-channel rasterized input tensors to predict spatial pedestrian wind speed fields).

```
+---------------------------------------------------------------------------------------+
|                                    CFD DOMAIN                                         |
|                                                                                       |
|  Inlet (ABL Profile)             Obstacle (H x W)                 Outlet (p = 0)      |
|  U(z), k(z), omega(z)            ================                 Zero-Gradient       |
|  ===================>            |              |     Wake Cavity                     |
|                                  |   BUILDING   |     Recirculation                   |
|                                  |   (No-Slip)  |     (u < 0)                         |
|  Ground: Wall Function (z0)      +--------------+---------------------------------    |
|  =================================================================================    |
|  <------- 5 H ------->           <----- W ----->      <------------- 15 H ------->    |
+---------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation & Governing Equations

### 2.1 Incompressible Navier-Stokes Equations
Under neutral atmospheric boundary layer conditions (wind speed $U_{\text{ref}} \ge 3\text{ m/s}$, Mach number $\text{Ma} < 0.03$), thermal buoyancy is negligible and airflow is treated as steady, incompressible, and isothermal:

$$\nabla \cdot \mathbf{U} = 0$$

$$(\mathbf{U} \cdot \nabla)\mathbf{U} = -\frac{1}{\rho}\nabla p + \nabla \cdot \left[ (\nu + \nu_t) \left( \nabla \mathbf{U} + (\nabla \mathbf{U})^T \right) \right]$$

where:
- $\mathbf{U} = (u, v, w)$ is the Reynolds-averaged velocity vector [$\text{m/s}$].
- $p$ is the modified kinematic pressure $p/\rho$ [$\text{m}^2/\text{s}^2$].
- $\nu$ is the molecular kinematic viscosity of air ($1.5 \times 10^{-5}\text{ m}^2/\text{s}$ at $20^\circ\text{C}$).
- $\nu_t$ is the kinematic turbulent eddy viscosity [$\text{m}^2/\text{s}$].

### 2.2 Turbulence Closure: $k\text{-}\omega$ SST vs. Standard $k\text{-}\varepsilon$

In urban aerodynamics, bluff-body obstacle flow exhibits sharp leading-edge separation, streamline curvature, adverse pressure gradients, and an unsteady recirculating wake.

We select the **Menter Shear Stress Transport ($k\text{-}\omega$ SST)** two-equation closure (Menter, 1994) for the following physical reasons:

1. **Stagnation Point Anomaly Resolution:** Standard $k\text{-}\varepsilon$ computes eddy viscosity as $\nu_t = C_\mu k^2 / \varepsilon$. In stagnation flow upstream of a building's windward facade, the irrotational strain rate $S = \sqrt{2 S_{ij}S_{ij}}$ is extremely high. Standard $k\text{-}\varepsilon$ generates an unphysical surge in turbulent kinetic energy $k$, causing excessive eddy viscosity that distorts stagnation pressure and delays flow separation at roof edges.
2. **Near-Wall Treatment without Damping:** $k\text{-}\omega$ SST formulates turbulence directly down to the wall using the specific dissipation rate $\omega = \varepsilon / (C_\mu k)$, eliminating complex non-linear wall-damping functions.
3. **Cross-Diffusion Blending:** SST employs a blending function $F_1$ that switches smoothly from the Wilcox $k\text{-}\omega$ formulation in the near-wall boundary layer ($F_1 = 1$) to the standard $k\text{-}\varepsilon$ model in the free-stream far field ($F_1 = 0$), avoiding the notorious free-stream sensitivity of pure $k\text{-}\omega$.
4. **Shear Stress Limiter:** SST enforces Bradshaw's assumption that turbulent shear stress in adverse pressure gradients is proportional to turbulent kinetic energy ($\tau = \rho a_1 k$), preventing premature flow reattachment on building roofs and capturing the correct wake cavity reattachment length ($L_r \approx 4 - 6 H$).

#### Transport Equations for $k\text{-}\omega$ SST:

$$\frac{\partial (\rho k)}{\partial t} + \nabla \cdot (\rho \mathbf{U} k) = \tilde{P}_k - \beta^* \rho k \omega + \nabla \cdot \left[ (\mu + \sigma_k \mu_t)\nabla k \right]$$

$$\frac{\partial (\rho \omega)}{\partial t} + \nabla \cdot (\rho \mathbf{U} \omega) = \alpha \frac{\rho}{\mu_t} \tilde{P}_k - \beta \rho \omega^2 + \nabla \cdot \left[ (\mu + \sigma_\omega \mu_t)\nabla \omega \right] + 2 (1 - F_1) \rho \sigma_{\omega 2} \frac{1}{\omega} \nabla k \cdot \nabla \omega$$

$$\nu_t = \frac{a_1 k}{\max(a_1 \omega, S F_2)}$$

### 2.3 RANS vs. Large-Eddy Simulation (LES) Limitations

While RANS ($k\text{-}\omega$ SST) provides efficient, converged steady solutions, its physical limitations must be acknowledged:
- **Time-Averaged Wake Dynamics:** Bluff-body wakes inherently shed transient vortex streets (von Kármán unsteadiness). Steady RANS time-averages these dynamics into a stationary symmetric recirculation cavity, typically underestimating turbulent fluctuation intensity in the near wake.
- **Role of LES (e.g. UrbanTALES):** LES explicitly resolves turbulent eddies larger than the grid filter width $\Delta = (\Delta x \Delta y \Delta z)^{1/3}$. UrbanTALES provides time-averaged LES moments that inherently capture wake intermittency, making it an ideal high-fidelity target dataset.

---

## 3. Computational Domain and Geometric Specifications

Following international best practice guidelines (Franke et al., 2007, **COST Action 732**; Tominaga et al., 2008, **AIJ Guidelines**):

### 3.1 Domain Dimensions (for Obstacle of Height $H = 20\text{ m}$, Width $W = 20\text{ m}$):
- **Upstream Fetch:** $5 H = 100\text{ m}$. Allows the equilibrium atmospheric boundary layer to develop without unphysical distortion from the building stagnation pressure field.
- **Downstream Wake Distance:** $15 H = 300\text{ m}$. Ensures the wake recirculation cavity ($4 - 6 H$) reattaches and recovers before encountering the outlet boundary, preventing artificial pressure reflection.
- **Domain Height:** $6 H = 120\text{ m}$. Ensures the blockage ratio $A_{\text{frontal}} / A_{\text{domain}} < 3\%$, eliminating spurious artificial acceleration over the roof caused by top boundary confinement.
- **Total Domain Length:** $L_x = 420\text{ m}$.
- **Quasi-2D Extrusion:** $L_y = 1.0\text{ m}$ (single cell thick with `empty` front and back patches).

### 3.2 Computational Mesh Configuration:
The structured multi-block hexahedral mesh is defined via `system/blockMeshDict` with 5 orthogonal blocks:
- **Block 0:** Upstream lower domain ($x \in [0, 100]$, $z \in [0, 20]$). $50 \times 1 \times 20$ cells.
- **Block 1:** Upstream upper domain ($x \in [0, 100]$, $z \in [20, 120]$). $50 \times 1 \times 30$ cells.
- **Block 2:** Overhead domain ($x \in [100, 120]$, $z \in [20, 120]$). $20 \times 1 \times 30$ cells.
- **Block 3:** Downstream wake cavity ($x \in [120, 420]$, $z \in [0, 20]$). $100 \times 1 \times 20$ cells.
- **Block 4:** Downstream upper domain ($x \in [120, 420]$, $z \in [20, 120]$). $100 \times 1 \times 30$ cells.
- **Solid Obstacle:** The building footprint ($x \in [100, 120]$, $z \in [0, 20]$) is excluded from the fluid domain.
- **Total Cell Count:** 7,100 structured hexahedral cells (2D prototype) or up to 50,000 cells for fine grid studies.

---

## 4. Boundary Conditions & Atmospheric Boundary Layer (ABL)

To ensure the inflow boundary layer does not decay horizontally before reaching the building (the "horizontal inhomogeneity" problem identified by Blocken et al., 2007), boundary conditions are formulated according to **Richards & Hoxey (1993)** equilibrium profiles:

### 4.1 Inflow Profiles ($x = 0$):
For reference wind speed $U_{\text{ref}} = 5.0\text{ m/s}$ at $z_{\text{ref}} = 10.0\text{ m}$ over terrain with aerodynamic roughness $z_0 = 0.05\text{ m}$:

$$u_* = \frac{\kappa U_{\text{ref}}}{\ln\left(\frac{z_{\text{ref}} + z_0}{z_0}\right)} = \frac{0.41 \times 5.0}{\ln(10.05 / 0.05)} \approx 0.3865\text{ m/s}$$

$$U(z) = \frac{u_*}{\kappa} \ln\left(\frac{z + z_0}{z_0}\right)$$

$$k(z) = \frac{u_*^2}{\sqrt{C_\mu}} \approx 0.498\text{ m}^2/\text{s}^2 \quad (\text{constant with height})$$

$$\omega(z) = \frac{u_*}{\kappa \sqrt{C_\mu} (z + z_0)}$$

$$\varepsilon(z) = \frac{u_*^3}{\kappa (z + z_0)}$$

where $\kappa = 0.41$ (von Kármán constant) and $C_\mu = 0.09$.

### 4.2 Wall Boundary Conditions:
- **Ground ($z = 0$):** `noSlip` for velocity; roughness wall function `atmNutkRoughWallFunction` with roughness length $z_0 = 0.05\text{ m}$. OpenFOAM computes equivalent sand-grain roughness via:
  $$k_s = \frac{9.793 z_0}{C_s} \approx 0.979\text{ m} \quad (\text{for } C_s = 0.5)$$
- **Building Facades & Roof:** `noSlip` for velocity; standard smooth wall functions: `kqRWallFunction` for $k$, `omegaWallFunction` for $\omega$, `nutkWallFunction` for $\nu_t$.
- **Top Boundary ($z = 120\text{ m}$):** `slip` (zero normal velocity $w = 0$, zero shear stress $\partial u / \partial z = 0$).
- **Outlet ($x = 420\text{ m}$):** `inletOutlet` with $p = 0$ gauge pressure (`fixedValue uniform 0`).
- **Front & Back Patches:** `empty` (enforces 2D planar solution).

---

## 5. Numerical Schemes and Solver Configuration

The OpenFOAM case utilizes the steady-state, incompressible **`simpleFoam`** solver with the SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm:

### 5.1 Discretization Schemes (`system/fvSchemes`):
- **Time Derivative (`ddtSchemes`):** `steadyState`.
- **Gradient (`gradSchemes`):** `Gauss linear` with `cellLimited Gauss linear 1` on velocity to prevent unphysical cell-to-cell overshoots near building corners.
- **Convection (`divSchemes`):**
  - Velocity convection $\nabla \cdot (\mathbf{U}\mathbf{U})$: `bounded Gauss linearUpwind grad(U)` (second-order upwind scheme essential to prevent artificial numerical diffusion).
  - Turbulence convection $\nabla \cdot (\mathbf{U}k)$, $\nabla \cdot (\mathbf{U}\omega)$: `bounded Gauss upwind` (first-order upwind to strictly guarantee positive boundedness of $k$ and $\omega$).
- **Laplacian (`laplacianSchemes`):** `Gauss linear corrected` (second-order with non-orthogonal correction).

### 5.2 Linear Solvers & Convergence (`system/fvSolution`):
- **Pressure Equation:** GAMG (Geometric-Algebraic Multigrid) solver with Gauss-Seidel smoother; convergence tolerance $10^{-6}$.
- **Momentum & Turbulence Equations:** Smooth solver with symmetric Gauss-Seidel (`symGaussSeidel`); convergence tolerance $10^{-6}$.
- **Relaxation Factors:**
  - Pressure field: $\alpha_p = 0.3$
  - Velocity field: $\alpha_U = 0.7$
  - Turbulence fields: $\alpha_k = 0.7, \alpha_\omega = 0.7$
- **Stopping Criteria:** Iterations proceed until scaled residuals for all fields ($p, U_x, U_z, k, \omega$) drop below $10^{-5}$ and monitor quantities stabilize.

---

## 6. Reference Data Contract Integration

To allow seamless data ingestion by the surrogate modeling pipeline, all CFD reference outputs are standardized into the project's data schema:

### 6.1 Coordinate System & Units:
- **Coordinates:** Right-handed Cartesian $(x, y, z)$. $x$: streamwise [$\text{m}$], $y$: spanwise [$\text{m}$], $z$: vertical height [$\text{m}$]. Ground is at $z = 0.0\text{ m}$.
- **Physical Units:** Standard SI ($m, m/s, m^2/s^2, Pa$).
- **Pedestrian Slice Height:** $z = 1.75\text{ m}$ above ground (standard meteorological pedestrian level).

### 6.2 Data Container (`UrbanLayoutSample` -> `.npz`):
The bridge script `cfd/scripts/foam_to_contract.py` packages CFD outputs into compressed `.npz` files:
```
sample_<layout_id>.npz
├── height_map                (float32, [Ny, Nx])
├── building_mask              (bool,    [Ny, Nx])
├── pedestrian_wind_speed     (float32, [Ny, Nx])   <-- Magnitude at z=1.75m
├── velocity_u                (float32, [Ny, Nx])   <-- Streamwise component
├── velocity_v                (float32, [Ny, Nx])   <-- Spanwise component
├── turbulent_kinetic_energy  (float32, [Ny, Nx])   <-- TKE at z=1.75m
└── _metadata_json            (JSON string containing layout, domain, solver metadata)
```

### 6.3 Downstream Consumption:
1. **Traditional ML Baseline:** `src/data/features.py` extracts 11 geometric summary features (plan area density $\lambda_p$, frontal area density $\lambda_f$, height statistics, aerodynamic roughness $z_0$, displacement height $d$) and pairs them with `area_averaged_wind_speed`.
2. **Neural Surrogate:** `src/data/rasterizer.py` converts the sample into a 7-channel spatial tensor:
   $$\mathbf{X} \in \mathbb{R}^{7 \times N_y \times N_x} = [\text{height}, \text{mask}, \text{SDF}, u_{\text{inflow}}, v_{\text{inflow}}, x_{\text{norm}}, y_{\text{norm}}]$$
   and masks building interiors to zero during loss evaluation.

---

## 7. Numerical and Physical Validation Checklist

Before accepting any CFD simulation or benchmark reference field as ground truth for surrogate training, verify every item on this checklist:

### Part A: Numerical Quality Verification
- [ ] **Domain Dimensions:** Upstream fetch $\ge 5H$, downstream wake $\ge 15H$, domain height $\ge 6H$.
- [ ] **Blockage Ratio:** Frontal area blockage $A_{\text{built}} / A_{\text{domain}} < 3\%$.
- [ ] **Mesh Non-Orthogonality:** Maximum face non-orthogonality $< 70^\circ$; average $< 30^\circ$ (verified via `checkMesh`).
- [ ] **Aspect Ratio & Skewness:** Maximum cell aspect ratio $< 50$ (near walls); skewness $< 2.0$.
- [ ] **Residual Convergence:** Scaled residuals for $p, U, k, \omega$ must reach $< 10^{-5}$ or plateau smoothly without oscillations.
- [ ] **Physical Monitor Invariance:** Drag coefficient $C_d$ on the building and velocity at downstream wake monitoring points change by $< 0.1\%$ over 200 iterations.
- [ ] **Grid Convergence:** Multi-grid verification (Grid Convergence Index $\text{GCI} < 5\%$ across coarse, medium, fine resolutions).

### Part B: Physical Aerodynamic Plausibility
- [ ] **Stagnation Zone:** Upstream stagnation pressure rise located at $z \approx 0.6 - 0.7 H$ on the windward facade.
- [ ] **Corner Separation:** Flow separates cleanly at the sharp leading edge of the windward roof/corners.
- [ ] **Corner Acceleration (Venturi Effect):** Pedestrian wind speed at lateral building edges accelerates to $1.3 - 1.5 \times U_{\text{ref}}$.
- [ ] **Leeward Wake Cavity:** Low-velocity recirculating wake cavity extends $4 - 6 H$ downstream of the building with reverse velocity ($u < 0$).
- [ ] **Far-Wake Recovery:** Velocity deficit monotonically recovers downstream according to turbulent wake self-similarity ($U(x) \propto x^{-1/2}$).
- [ ] **Building Interior Masking:** Fluid velocity inside the solid building footprint is exactly $0.0\text{ m/s}$ and masked to prevent gradient corruption.
- [ ] **Atmospheric Boundary Layer Homogeneity:** In an empty domain test, the vertical profiles of $U(z)$ and $k(z)$ must remain horizontally invariant ($\Delta U < 3\%$) between inlet and building location.
