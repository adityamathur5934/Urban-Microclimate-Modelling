# Physical Assumptions, Boundary Physics, and Modeling Limitations in Urban Microclimate

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Task:** Task 2 — CFD Fundamentals and Physical Assumptions  
**Owner:** CFD and Scientific Computing Lead  
**Document Version:** 1.0.0  
**Branch:** `feature/cfd-fundamentals`  
**Status:** Approved Reference Specification  

---

## 1. Executive Summary & Problem Scope

Computational modeling of urban microclimates requires balancing physical fidelity with computational tractability. When training data-driven machine learning surrogates (such as 2D U-Nets) or tabular baselines (Random Forests / XGBoost), the surrogate inherits the exact physical assumptions, simplifications, and limitations of the underlying computational fluid dynamics (CFD) dataset.

This document systematically analyzes the physical assumptions governing our computational framework:
1. **Flow Field Taxonomy:** Distinguishing between wind velocity vector fields, wind speed magnitude fields, and temperature fields.
2. **Thermal Physics & The Energy Equation:** Analyzing the mathematical coupling between momentum and heat transfer, the Boussinesq approximation, thermal boundary conditions, and why an isothermal (wind-only) solver cannot be substituted for a temperature simulation.
3. **Dimensional Constraints:** Demonstrating why two-dimensional (2D) simulations fundamentally distort three-dimensional bluff-body vortex dynamics, wake recirculation lengths, and street canyon channelling.
4. **Verification vs. Validation:** Clarifying why numerical convergence (residuals $< 10^{-5}$) does not establish physical reality, and outlining the role of experimental benchmarks (such as CEDVAL and AIJ).
5. **Surrogate Mapping:** Explaining how these physical assumptions constrain the neural network loss functions, input representations (Signed Distance Fields), and data contracts (`src.data.contract`).

> [!IMPORTANT]
> **Task Scope & Simulation Execution Disclaimer:**  
> In strict compliance with the task guidelines, **no simulations were executed during this task**. This document establishes the theoretical, physical, and methodological boundaries required to validate and interpret urban microclimate simulations and surrogate models.

---

## 2. Taxonomy of Microclimate Fields: Speed vs. Velocity vs. Temperature

Urban microclimatic analysis frequently refers to "wind" and "temperature," but these represent distinct mathematical objects with different conservation laws, boundary conditions, and regulatory relevance.

```
+---------------------------------------------------------------------------------------------------------------+
|                                      MICROCLIMATE FIELD TAXONOMY MATRIX                                       |
+-------------------+----------------------------+-----------------------+--------------------------------------+
| Field Type        | Mathematical Object        | Dimensionality & Unit | Physical Role & Regulatory Use       |
+-------------------+----------------------------+-----------------------+--------------------------------------+
| **Wind Speed**    | Scalar Field               | Rank 0 Tensor         | Primary index for pedestrian comfort |
| **Magnitude**     | $U = \sqrt{u^2 + v^2}$     | $[\text{m/s}]$        | and mechanical safety criteria       |
|                   | (Strictly non-negative)    | ($U \ge 0$)           | (Lawson, Davenport, NEN 8100).      |
+-------------------+----------------------------+-----------------------+--------------------------------------+
| **Wind Velocity** | Vector Field               | Rank 1 Tensor         | Governs momentum transport, streamline|
| **Components**    | $\mathbf{u} = (u, v, w)$   | $[\text{m/s}]$        | topology, recirculation, vorticity   |
|                   | (Directional & signed)     | ($u, v, w \in \mathbb{R}$) | $\boldsymbol{\omega} = \nabla \times \mathbf{u}$, and stagnation. |
+-------------------+----------------------------+-----------------------+--------------------------------------+
| **Air Temperature**| Thermodynamic Scalar     | Rank 0 Tensor         | Governs urban heat island (UHI),     |
| **Field**         | $T(\mathbf{x})$            | $[\text{K}]$ or       | thermal buoyancy plumes, and human   |
|                   | (Absolute thermodynamic)   | $[^\circ\text{C}]$    | thermal comfort (UTCI / PMV).        |
+-------------------+----------------------------+-----------------------+--------------------------------------+
```

### 2.1 Wind Speed Magnitude ($U$)

The wind speed magnitude is defined as the Euclidean $L_2$ norm of the velocity vector:

$$U_{\text{ped}}(x, y) = \sqrt{u(x, y)^2 + v(x, y)^2} \quad \text{or} \quad U_{\text{3D}}(x, y, z) = \sqrt{u^2 + v^2 + w^2}$$

In urban microclimate modeling at pedestrian level ($z = 1.75\text{ m}$), horizontal wind speed magnitude $U_{\text{ped}}$ is the **primary regulatory standard**. International criteria for pedestrian wind comfort and safety (e.g., Lawson Criteria, Davenport Comfort Scale, Dutch Standard NEN 8100) are formulated exclusively in terms of exceedance probabilities of scalar speed thresholds:
- **Sitting Long Duration:** $U \le 2.5\text{ m/s}$
- **Strolling / Walking:** $2.5\text{ m/s} < U \le 5.0\text{ m/s}$
- **Business Walking:** $5.0\text{ m/s} < U \le 7.5\text{ m/s}$
- **Pedestrian Safety Limit (Hazardous):** $U > 15.0\text{ m/s}$

*Mathematical Characteristics for Surrogates:*  
- Strictly non-negative: $U(x, y) \ge 0$ everywhere.
- Loss of directional orientation: knowing $U = 4\text{ m/s}$ does not reveal whether air is flowing forward, backward in a recirculating wake cavity, or upward in a buoyant plume.

### 2.2 Wind Velocity Vector Field ($\mathbf{u} = (u, v, w)$)

The velocity vector field contains both speed and direction:

$$\mathbf{u}(x, y, z) = u \, \hat{\mathbf{i}} + v \, \hat{\mathbf{j}} + w \, \hat{\mathbf{k}}$$

*Physical Significance:*
1. **Recirculation and Wake Bubbles:** In the leeward wake cavity, the streamwise component is negative ($u < 0$), proving reverse flow. A scalar speed magnitude obscures this critical topological distinction because $|-2\text{ m/s}| = |+2\text{ m/s}| = 2\text{ m/s}$.
2. **Vorticity and Circulation:** Vorticity $\boldsymbol{\omega} = \nabla \times \mathbf{u}$ requires directional velocity derivatives ($\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$). Vorticity cannot be reconstructed from scalar speed fields alone.
3. **Mass Conservation:** The continuity equation $\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$ applies strictly to vector components. A scalar speed field cannot satisfy a divergence operator.

### 2.3 Temperature Field ($T$)

Temperature is an intensive scalar state variable governing internal thermal energy:

$$e_{\text{thermal}} = c_v T \quad [\text{J/kg}]$$

Unlike wind speed, which is generated by kinematic boundary conditions (atmospheric wind profiles) and pressure gradients, the temperature field is governed by thermal radiation, sensible and latent heat exchange, and conductive heat storage in urban construction materials (asphalt, concrete, brick).

---

## 3. The Energy Equation and Coupled Thermal-Fluids Physics

### 3.1 The Thermal Energy Transport Equation

The conservation of thermal energy in an incompressible fluid is given by:

$$\rho c_p \left( \frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T \right) = \nabla \cdot (k_{\text{thermal}} \nabla T) + \Phi_{\text{visc}} + S_h$$

where:
- $\rho$ is air density ($1.205\text{ kg/m}^3$).
- $c_p$ is the specific heat capacity of air at constant pressure ($1005\text{ J}/(\text{kg}\cdot\text{K})$).
- $k_{\text{thermal}}$ is the thermal conductivity of air ($0.0257\text{ W}/(\text{m}\cdot\text{K})$ at $20^\circ\text{C}$).
- $\Phi_{\text{visc}} = \tau_{ij} \frac{\partial u_i}{\partial x_j}$ is viscous dissipation (negligible in low-speed atmospheric flows).
- $S_h$ is the volumetric thermal heat source [$\text{W/m}^3$] (e.g., radiative absorption, anthropogenic emissions).

Dividing by $\rho c_p$ yields the thermal advection-diffusion equation:

$$\frac{\partial T}{\partial t} + u_j \frac{\partial T}{\partial x_j} = \alpha_{\text{thermal}} \frac{\partial^2 T}{\partial x_j \partial x_j} + \frac{S_h}{\rho c_p}$$

where $\alpha_{\text{thermal}} = \frac{k_{\text{thermal}}}{\rho c_p} \approx 2.12 \times 10^{-5}\text{ m}^2/\text{s}$ is the thermal diffusivity of air.

### 3.2 The Boussinesq Approximation and Buoyant Coupling

In non-isothermal urban environments, temperature differences generate density differences ($\Delta \rho$). Gravity acts on these density differences, producing a vertical buoyant body force.

Under the **Boussinesq Approximation**, density variations are assumed negligible everywhere *except* in the gravitational body force term of the vertical momentum equation:

$$\rho \approx \rho_0 \left[ 1 - \beta (T - T_0) \right]$$

where:
- $\rho_0$ is the reference fluid density at reference temperature $T_0$ (e.g., $293.15\text{ K}$).
- $\beta = -\frac{1}{\rho_0} \left( \frac{\partial \rho}{\partial T} \right)_p = \frac{1}{T_0}$ is the thermal volumetric expansion coefficient for an ideal gas ($3.41 \times 10^{-3}\text{ K}^{-1}$ at $20^\circ\text{C}$).

Substituting into the vertical momentum equation ($z$-direction):

$$\rho_0 \left( \frac{\partial w}{\partial t} + \mathbf{u} \cdot \nabla w \right) = -\frac{\partial p}{\partial z} + \mu \nabla^2 w - \rho g$$

Separating total pressure into hydrostatic equilibrium $p_{\text{hyd}}$ and hydrodynamic motion pressure $p'$:

$$-\frac{\partial p_{\text{hyd}}}{\partial z} - \rho_0 g = 0$$

The remaining momentum equation retains the **buoyant body force**:

$$\frac{\partial w}{\partial t} + \mathbf{u} \cdot \nabla w = -\frac{1}{\rho_0} \frac{\partial p'}{\partial z} + \nu \nabla^2 w + g \beta (T - T_0)$$

```
                                  TWO-WAY BUOYANCY COUPLING
                                  
          +-------------------------------------------------------+
          |               Energy Equation: T(x, y, z)             |
          |       u_j (dT / dx_j) = alpha (d^2 T / dx_j^2)        |
          +-------------------------------------------------------+
                     |                                 ^
                     | Buoyancy Force                  | Convective Advection
                     | f_b = rho_0 * g * beta * DeltaT | u_j (dT / dx_j)
                     v                                 |
          +-------------------------------------------------------+
          |              Momentum Equation: u(x, y, z)            |
          |   rho (u_j du_i/dx_j) = -dp/dx_i + mu d^2u_i + f_b    |
          +-------------------------------------------------------+
```

This represents **two-way physical coupling**:
1. Velocity $\mathbf{u}$ advects temperature $T$ through the convection term $u_j \frac{\partial T}{\partial x_j}$.
2. Temperature $T$ drives velocity $w$ through the buoyant body force $g \beta (T - T_0) \hat{\mathbf{k}}$.

### 3.3 Dimensionless Governing Numbers & Convective Regimes

The relative importance of thermal buoyancy versus inertial wind forces is quantified by four dimensionless numbers:

```
+---------------------------------------------------------------------------------------------------------------+
|                                      DIMENSIONLESS THERMAL-FLUID NUMBERS                                      |
+-------------------+----------------------------+--------------------------------------------------------------+
| Dimensionless No. | Mathematical Definition    | Physical Meaning & Urban Significance                        |
+-------------------+----------------------------+--------------------------------------------------------------+
| **Reynolds (Re)** | $\text{Re} = \frac{U_{\text{ref}} H}{\nu}$ | Ratio of inertial forces to viscous forces. In urban canopies, |
|                   |                            | $\text{Re} \sim 10^6 - 10^7$ (fully turbulent flow).         |
+-------------------+----------------------------+--------------------------------------------------------------+
| **Grashof (Gr)**  | $\text{Gr} = \frac{g \beta \Delta T H^3}{\nu^2}$ | Ratio of buoyant forces to viscous forces driving natural   |
|                   |                            | convection along heated facades and asphalt.                 |
+-------------------+----------------------------+--------------------------------------------------------------+
| **Richardson (Ri)**| $\text{Ri} = \frac{\text{Gr}}{\text{Re}^2} = \frac{g \beta \Delta T H}{U_{\text{ref}}^2}$ | Ratio of buoyant forces to inertial aerodynamic forces;    |
|                   |                            | determines the atmospheric stability regime.                 |
+-------------------+----------------------------+--------------------------------------------------------------+
| **Prandtl (Pr)**  | $\text{Pr} = \frac{\nu}{\alpha_{\text{thermal}}} = \frac{\mu c_p}{k_{\text{thermal}}}$ | Ratio of momentum diffusivity to thermal diffusivity;       |
|                   |                            | $\text{Pr} \approx 0.71$ for air at standard conditions.      |
+-------------------+----------------------------+--------------------------------------------------------------+
```

#### Atmospheric Convective Regimes:
- **Forced Convection ($\text{Ri} < 0.1$):**  
  Inertial wind forces dominate completely. Buoyant thermal plumes are swept away by the wind. The airflow field is decoupled from the temperature field, and the isothermal assumption is physically valid. (Occurs during moderate-to-strong winds: $U_{\text{ref}} \ge 3\text{ m/s}$).
- **Mixed Convection ($0.1 \le \text{Ri} \le 10$):**  
  Both buoyancy and wind inertia exert equal influence. Updrafts along solar-heated facades alter the roof-level shear layer, shifting street-canyon vortex centers and increasing rooftop air exchange.
- **Free / Natural Convection ($\text{Ri} > 10$):**  
  Buoyant forces dominate. Under calm summer conditions ($U_{\text{ref}} < 1\text{ m/s}$), airflow is driven entirely by thermal updrafts rising off hot asphalt and sunlit building walls.

---

## 4. Why an Isothermal Wind Solver Cannot Be Treated as a Temperature Simulation

It is tempting in early machine learning experiments to assume that scalar fields are interchangeable—for example, treating a CFD wind speed field as a proxy for urban surface temperature. **Doing so is a fundamental physical and scientific error.**

```
+---------------------------------------------------------------------------------------------------------------+
|                        PHYSICAL DIVERGENCE: WIND-ONLY SOLVER VS. THERMAL-COUPLED SOLVER                       |
+--------------------------------+--------------------------------------+---------------------------------------+
| Feature / Physical Mechanism   | Isothermal Wind Solver (simpleFoam)   | Non-Isothermal Buoyant Solver         |
|                                |                                      | (buoyantSimpleFoam)                   |
+--------------------------------+--------------------------------------+---------------------------------------+
| **Governing Equations**        | Mass Continuity + Momentum (RANS)    | Mass + Momentum + Thermal Energy Eq.  |
| **Coupling Mechanism**         | None (Kinematic & Pressure driven)   | Two-way Boussinesq Buoyant Coupling   |
| **Surface Boundary Physics**   | No-slip wall: $\mathbf{u} = 0$       | Surface Energy Balance: Radiation,    |
|                                |                                      | Conduction, Convection, Evaporation   |
| **Source Terms**               | None in fluid domain ($S = 0$)       | Solar radiation absorption ($S_h \ne 0$)|
| **Canyon Flow Physics**        | Fixed trapped vortex at high $H/W$   | Thermal updrafts destroy or invert    |
|                                |                                      | canyon vortex under solar heating     |
| **Computational Cost**         | Baseline ($1\times$)                 | $3\times - 5\times$ compute time;     |
|                                |                                      | requires radiation sub-models (fvDOM) |
+--------------------------------+--------------------------------------+---------------------------------------+
```

### 4.1 Asymmetry of Surface Energy Balances

In an isothermal wind simulation, the boundary condition at building facades and the ground plane is purely kinematic:

$$\mathbf{u}_{\text{wall}} = (0, 0, 0)$$

In reality, surface temperature $T_{\text{surf}}$ is governed by the **Surface Energy Balance (SEB)**:

$$R_{\text{net}} = H_{\text{sensible}} + \lambda E_{\text{latent}} + G_{\text{conductive}}$$

where:
1. $R_{\text{net}} = (1 - \alpha_{\text{albedo}}) S_{\text{solar}\downarrow} + L_{\text{atm}\downarrow} - \epsilon_{\text{surf}} \sigma T_{\text{surf}}^4$ is the net all-wave radiation.
2. $H_{\text{sensible}} = h_c (T_{\text{surf}} - T_{\text{air}})$ is the turbulent sensible convective heat flux to the air.
3. $\lambda E_{\text{latent}}$ is the latent heat flux from evaporating surfaces or urban vegetation.
4. $G_{\text{conductive}} = -k_{\text{mat}} \left. \frac{\partial T}{\partial n} \right|_{\text{solid}}$ is the conductive heat storage entering the building envelope or pavement.

*Why Wind Solvers Cannot Predict This:*  
A wind solver has no knowledge of solar zenith angles, shadow casting, building albedo, surface emissivity, or material thermal conductivity. On a sunny day, a sunlit south-facing concrete wall can reach $55^\circ\text{C}$ while an adjacent shadowed north-facing wall remains at $25^\circ\text{C}$, even though the local wind speed is identical ($1\text{ m/s}$) along both walls.

### 4.2 Inversion and Destruction of Street-Canyon Vortices

In an isothermal wind simulation of a dense street canyon (aspect ratio $H/W = 1.0$), the skimming flow across rooftops drives a single, clockwise-rotating trapped vortex inside the canyon.

However, when thermal effects are simulated (as proven in wind tunnel studies by Allegrini et al., 2013 and Offerle et al., 2007):
1. When the **windward wall is heated by direct sunlight**, the upward buoyant plume along the wall opposes the downward flow of the mechanical vortex. The mechanical vortex is weakened or split into two counter-rotating secondary vortices.
2. When the **street pavement is intensely heated**, a strong thermal updraft develops from the canyon floor, pushing upward against the rooftop shear layer and completely altering the air exchange rate with the free stream.

An isothermal dataset (such as **UrbanTALES**) strictly assumes neutral atmospheric boundary layer conditions ($\text{Ri} \approx 0$). Presenting UrbanTALES wind speed outputs as a temperature prediction would constitute a catastrophic scientific failure and violation of academic integrity.

---

## 5. Dimensional Constraints: Critical Limitations of 2D Simulations

In early CFD studies, developers frequently set up two-dimensional (2D) planar simulations to reduce runtimes from hours to minutes. In our repository, `cfd/cases/simple2d_building_block/` provides an illustrative 2D educational benchmark.

However, **2D CFD simulations of urban airflow contain severe, fundamental physical errors** and cannot be used as training targets for realistic city surrogates.

```
+---------------------------------------------------------------------------------------------------------------+
|                                       3D REALITY VS. 2D SIMULATION ARTIFACTS                                  |
+---------------------------------------------------------------------------------------------------------------+
|                                                                                                               |
|    (a) 3D REALITY: Flow Escapes in 3 Dimensions          (b) 2D SIMULATION: Artificial Lateral Channelling    |
|                                                                                                               |
|               Flow over Roof                                            NO VERTICAL FLOW POSSIBLE             |
|                  ======>                                                                                      |
|               +-----------+                                             +-------------------------+           |
|   Wind ====>  |           | ====> Wake Recirculation        Wind =====> |                         |           |
|               | BUILDING  |       Length L_r ~ 4 - 6 H                  |    INFINITE CYLINDER    |           |
|               |  (3D Box) |                                             |      (No Roof, No Base) |           |
|               +-----------+                                             +-------------------------+           |
|              /           /                                                           |                        |
|             +-----------+                                                            v                        |
|        Downwash Impingement & Horseshoe Vortex                      Mass Conservation Forces ALL Flow         |
|        Flow escapes OVER and AROUND building                        into Lateral Streets (Venturi Jet x3)     |
|                                                                     Wake Length Overestimated: L_r ~ 15 - 20 H|
+---------------------------------------------------------------------------------------------------------------+
```

### 5.1 Vortex Stretching: The Fundamental Difference

The vorticity transport equation is obtained by taking the curl of the Navier–Stokes equations:

$$\frac{\partial \boldsymbol{\omega}}{\partial t} + (\mathbf{u} \cdot \nabla)\boldsymbol{\omega} = (\boldsymbol{\omega} \cdot \nabla)\mathbf{u} + \nu \nabla^2 \boldsymbol{\omega}$$

where $\boldsymbol{\omega} = \nabla \times \mathbf{u}$ is the vorticity vector.

The term $(\boldsymbol{\omega} \cdot \nabla)\mathbf{u}$ represents **vortex stretching and tilting**:
- In **three-dimensional (3D) turbulence**, when fluid elements are stretched along vortex lines by velocity gradients, their cross-sectional area contracts, increasing vorticity magnitude to conserve angular momentum. Vortex stretching is the fundamental physical mechanism by which kinetic energy cascades from large turbulent eddies down to the microscopic dissipative Kolmogorov scales.
- In **two-dimensional (2D) flows**, $\mathbf{u} = (u(x, y), v(x, y), 0)$ and $\boldsymbol{\omega} = (0, 0, \omega_z)$. Therefore:
  $$(\boldsymbol{\omega} \cdot \nabla)\mathbf{u} = \omega_z \frac{\partial \mathbf{u}}{\partial z} \equiv 0$$
  **Vortex stretching is identically zero in 2D.**

#### The Inverse Energy Cascade:
Because vortex stretching cannot occur in 2D, energy does not cascade from large to small scales. Instead, 2D turbulence exhibits an **inverse energy cascade** (Kraichnan, 1967; Batchelor, 1969): small eddies merge into artificially gigantic, coherent, long-lived vortices that do not exist in the physical world.

### 5.2 Geometric Distortion: The Infinite Cylinder Fallacy

In a 2D planar simulation (in the horizontal $x$-$y$ plane), the vertical coordinate $z$ is treated as invariant ($\frac{\partial}{\partial z} \equiv 0$):
1. **Buildings become infinitely tall cylinders extending to $z = \pm \infty$**:
   In reality, when oncoming wind strikes a building facade of height $H$, a large fraction of the oncoming air escapes vertically **over the roof** (forming the rooftop separation bubble) and downward (forming the downwash jet).
2. **Artificial Lateral Mass Channelling:**
   In 2D, because air cannot escape over the roof or down to the ground, **100% of the intercepted mass flux must squeeze through the lateral passages between buildings**. By continuity ($\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$), street canyon velocities are artificially accelerated by $150\% - 300\%$ over realistic 3D values.
3. **Severe Overestimation of Wake Recirculation Length ($L_r$):**
   In 3D, the leeward wake cavity is rapidly ventilated and reattached by downwash descending from the roof and lateral flow wrapping around flanks, resulting in a reattachment length of $L_r \approx 4.0 - 6.0 H$.  
   In 2D, wake cavity ventilation from above is physically impossible. The leeward separation bubble extends downstream to **$15 - 25 \times H$**, creating an unphysically elongated stagnation wake.

### 5.3 Legitimate Role of 2D Simulations in our Workflow

Given these severe physical limitations, why does the repository include a 2D OpenFOAM benchmark case (`cfd/cases/simple2d_building_block/`)?
1. **Numerical Pipeline Verification:** Testing grid generation (`blockMesh`), numerical schemes (`fvSchemes`), linear solvers (`fvSolution`), and convergence scripting in seconds rather than hours.
2. **Contract Pipeline Testing:** Validating that data-bridge scripts (`cfd/scripts/foam_to_contract.py`) correctly parse OpenFOAM sampling sets into validated `UrbanLayoutSample` objects.
3. **Educational Baseline:** Providing a lightweight testbed for students to understand boundary conditions and residuals on a laptop without requiring high-performance computing clusters.

> [!TIP]
> **Ground Truth Rule for Neural Surrogate Training:**  
> For actual machine learning surrogate training, ground truth data must originate from **3D simulations** (such as the 3D LES dataset **UrbanTALES**), sliced horizontally at the pedestrian height ($z = 1.75\text{ m}$). A 2D slice of a 3D simulation preserves the physical downwash, rooftop overflow, and correct wake reattachment lengths.

---

## 6. Numerical Convergence Does Not Establish Physical Accuracy

A common misconception among beginner CFD and machine learning practitioners is that if a simulation converges—meaning the residuals drop below $10^{-5}$ and the log file displays `End`—the simulation results must represent physical reality.

**Numerical convergence is a necessary condition for mathematical consistency, but it is entirely insufficient to establish physical accuracy.**

```
                               VERIFICATION VS. VALIDATION
                               
               +---------------------------------------------------------------+
               |                       REALITY (PHYSICS)                       |
               +---------------------------------------------------------------+
                                               ^
                                               |
                                     VALIDATION (Experiments)
                               "Are we solving the RIGHT equations?"
                                               |
                                               v
               +---------------------------------------------------------------+
               |                       MATHEMATICAL MODEL                      |
               |        (Navier-Stokes, k-omega SST, Incompressible)           |
               +---------------------------------------------------------------+
                                               ^
                                               |
                                    VERIFICATION (Numerics)
                               "Are we solving the equations RIGHT?"
                                               |
                                               v
               +---------------------------------------------------------------+
               |                       COMPUTER CODE (CFD)                     |
               |             Discretization, Meshes, Residuals < 1e-5          |
               +---------------------------------------------------------------+
```

### 6.1 Verification vs. Validation (Roache, 1998; ASME V&V 20)

- **Verification:** The process of determining that a computational model implementation accurately represents the developer's conceptual description and specification.  
  *Question:* "Are we solving the equations right?"  
  *Checks:* Residual convergence, grid independence tests (GCI $< 2\%$), absence of coding bugs, spatial order of accuracy testing.
- **Validation:** The process of determining the degree to which a model is an accurate representation of the real world from the perspective of the intended uses of the model.  
  *Question:* "Are we solving the right equations?"  
  *Checks:* Comparison against physical wind tunnel measurements (e.g., CEDVAL A1-1 isolated building, AIJ building cluster benchmarks).

### 6.2 Failure Modes Where a Fully Converged CFD Solution is Physically False

A CFD simulation can converge to machine precision ($10^{-7}$) while being completely physically invalid due to:

1. **Stagnation Point Anomaly in Turbulence Modeling:**
   Running standard $k$-$\varepsilon$ on a building cluster will converge smoothly and quickly. However, the solution will overpredict turbulent kinetic energy at windward facades by $400\%$, suppress corner separation, and miss localized pedestrian hazard zones. The math converged, but the turbulence physics is wrong.
2. **Domain Truncation and Excessive Blockage:**
   If a domain is built with only $3H$ downstream fetch (instead of $15H$), the leeward wake hits the outlet boundary. A fixed gauge pressure boundary ($p = 0$) forces unphysical streamline deflection. The simulation will converge, but the predicted wake length and street canyon pressure drops are artifacts of the artificial boundary.
3. **Uniform Inflow Instead of Logarithmic ABL:**
   Applying a uniform inlet velocity ($U(z) = \text{const}$) eliminates the natural vertical shear of atmospheric wind. Without vertical shear, frontal stagnation pressure is uniform, eliminating downwash and the ground-level horseshoe vortex.
4. **Omission of Buoyancy in Low-Wind Conditions:**
   Under low-speed summer conditions ($U_{\text{ref}} = 1\text{ m/s}$), solar radiation heats asphalt to $50^\circ\text{C}$ ($\text{Ri} > 1.0$). Running an isothermal simulation converges easily, but completely misses the buoyant updrafts that ventilate the street canyon.

---

## 7. Mapping CFD Assumptions to Neural Surrogate Loss Formulation

When training deep neural networks (such as a 2D U-Net) to act as surrogates for urban CFD, the loss function and input representations must reflect the underlying fluid physics.

### 7.1 Hard Physical Constraints vs. Soft Penalties

In deep learning, physical constraints can be enforced via two distinct mechanisms:

```
+---------------------------------------------------------------------------------------------------------------+
|                                    PHYSICS ENFORCEMENT IN NEURAL SURROGATES                                   |
+-------------------+------------------------------------------+------------------------------------------------+
| Approach          | Implementation Mechanism                 | Impact on Surrogate Physical Validity          |
+-------------------+------------------------------------------+------------------------------------------------+
| **Hard Physical** | Architecture-level masking:              | **Guarantees exact physical compliance.**      |
| **Constraints**   | $\hat{\mathbf{Y}} = (1 - \mathbf{M}) \odot \mathcal{M}_\theta(\mathbf{X})$ | Solid building interiors are identically zero.  |
|                   | Clamping non-negative speeds: $\max(0, \hat{Y})$ | Zero chance of unphysical non-zero velocities  |
|                   |                                          | inside solid concrete walls.                   |
+-------------------+------------------------------------------+------------------------------------------------+
| **Soft Loss**     | Multi-objective composite loss:          | Encourages the network to learn physical       |
| **Penalties**     | $\mathcal{L} = \mathcal{L}_1 + \lambda_{\text{grad}} \mathcal{L}_{\text{Sobel}}$ | gradients (shear layers, separation boundaries)|
|                   | Continuity penalty: $\|\nabla \cdot \hat{\mathbf{u}}\|_2^2$ | without causing optimization instability.      |
+-------------------+------------------------------------------+------------------------------------------------+
```

### 7.2 Inductive Bias: The Role of Signed Distance Fields (SDF)

A standard binary mask $M(x, y) \in \{0, 1\}$ provides only a discontinuous step function indicating whether a cell is solid or fluid.

In fluid dynamics, the effect of a solid boundary propagates continuously into the flow field through viscous shear stress and boundary layer growth:

$$u(y) \approx \frac{u_*}{\kappa} \ln\left( \frac{y + z_0}{z_0} \right)$$

By augmenting the neural network input with an exact **Signed Distance Field (SDF)**:

$$\phi_{\text{SDF}}(x, y) = \begin{cases} -d((x, y), \partial \Omega) & (x, y) \in \text{building} \\ +d((x, y), \partial \Omega) & (x, y) \in \text{fluid} \end{cases}$$

we provide convolutional kernels with direct, continuous spatial gradients representing wall proximity. This provides the network with the inductive bias required to learn near-wall velocity recovery without requiring excessively deep receptive fields.

---

## 8. Comprehensive Glossary of Physical & Thermodynamic Terms

- **Advective Acceleration ($(\mathbf{u} \cdot \nabla)\mathbf{u}$):** Acceleration of a fluid particle caused by changes in velocity over position in a non-uniform flow field.
- **Boussinesq Approximation:** Simplification for buoyant flows where density variations are neglected in inertia terms and retained only in the gravitational body force term.
- **Dynamic Pressure ($q = \frac{1}{2}\rho U^2$):** Kinetic energy per unit volume of moving fluid.
- **Energy Cascade:** Process in 3D turbulent flow where kinetic energy is transferred inviscidly from large eddies down through intermediate scales to microscopic Kolmogorov scales where it is dissipated by viscosity.
- **Forced Convection:** Heat transfer regime dominated entirely by fluid inertia (high wind speeds) where buoyant forces are negligible ($\text{Ri} < 0.1$).
- **Grashof Number ($\text{Gr}$):** Dimensionless number representing the ratio of buoyant forces to viscous forces in natural convection.
- **Inverse Energy Cascade:** Phenomenon unique to 2D turbulence where kinetic energy transfers from small to large scales due to the absence of 3D vortex stretching.
- **Isothermal Flow:** Flow regime characterized by constant, uniform temperature throughout the domain, decoupling thermodynamics from momentum.
- **Kinematic Pressure ($p/\rho$):** Thermodynamic pressure normalized by fluid density, having dimensions of $[\text{m}^2/\text{s}^2]$.
- **Knudsen Number ($Kn$):** Dimensionless ratio of molecular mean free path to characteristic physical length scale; verifies the continuum hypothesis ($Kn < 0.001$).
- **Natural / Free Convection:** Fluid motion and heat transfer driven entirely by density differences caused by temperature gradients under gravity ($\text{Ri} > 10$).
- **Prandtl Number ($\text{Pr}$):** Dimensionless ratio of momentum diffusivity (kinematic viscosity) to thermal diffusivity ($\text{Pr} = \nu / \alpha$).
- **Reynolds Number ($\text{Re}$):** Dimensionless ratio of inertial forces to viscous forces ($\text{Re} = U L / \nu$).
- **Richardson Number ($\text{Ri} = \text{Gr} / \text{Re}^2$):** Dimensionless ratio characterizing the relative importance of natural convection (buoyancy) to forced convection (wind inertia).
- **Signed Distance Field (SDF):** Continuous field encoding the shortest Euclidean distance from every grid pixel to the nearest obstacle boundary, with negative signs inside solid geometry.
- **Surface Energy Balance (SEB):** Thermodynamic equilibrium equation balancing net radiation, sensible heat flux, latent heat flux, and conductive heat storage at urban surfaces.
- **Thermal Diffusivity ($\alpha_{\text{thermal}} = k_{\text{thermal}} / \rho c_p$):** Material property measuring the rate of heat transfer through diffusion relative to heat storage capacity.
- **Universal Thermal Climate Index (UTCI):** Bio-meteorological index measuring human physiological thermal stress based on temperature, wind speed, mean radiant temperature, and relative humidity.
- **Validation:** Procedure evaluating whether a computational model accurately reflects the physical real-world phenomenon ("solving the right equations").
- **Verification:** Procedure evaluating whether numerical algorithms solve the underlying mathematical model equations accurately ("solving the equations right").
- **Vortex Stretching ($(\boldsymbol{\omega} \cdot \nabla)\mathbf{u}$):** Mechanism in 3D turbulence where velocity gradients stretch vorticity filaments, concentrating angular momentum and driving the forward energy cascade.
- **Vorticity ($\boldsymbol{\omega} = \nabla \times \mathbf{u}$):** Mathematical vector representing the local angular rotation of a fluid element.

---

## 9. Relevant Verified References

1. **ASME (2009).** *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer.* ASME V&V 20-2009, American Society of Mechanical Engineers, New York.
2. **Allegrini, J., Dorer, V., & Carmeliet, J. (2013).** *Wind tunnel measurements of buoyant flows in street canyons.* Journal of Wind Engineering and Industrial Aerodynamics, 122, 118–126.
3. **Offerle, B., Eliasson, I., Grimmond, C. S. B., & Holmer, B. (2007).** *Surface heat flux and energy balance of a high-latitude city: Göteborg, Sweden.* Journal of Applied Meteorology and Climatology, 46(4), 564–579.
4. **Kraichnan, R. H. (1967).** *Inertia ranges in two-dimensional turbulence.* The Physics of Fluids, 10(7), 1417–1423.
5. **Batchelor, G. K. (1969).** *Computation of the energy spectrum in homogeneous two-dimensional turbulence.* The Physics of Fluids, 12(12), II-233.
6. **Blocken, B., Stathopoulos, T., & Carmeliet, J. (2007).** *CFD simulation of the atmospheric boundary layer: wall function problems.* Atmospheric Environment, 41(2), 238–252.
7. **Franke, J., Hellsten, A., Schlünzen, H., & Carissimo, B. (2007).** *Best practice guideline for the CFD simulation of flows in the urban environment.* COST Action 732: Quality Assurance and Improvement of Microclimate Models.
8. **Tominaga, Y., & Stathopoulos, T. (2013).** *CFD simulation of near-field pollutant dispersion in the urban environment: A review of current modeling techniques.* Atmospheric Environment, 79, 716–730.

---

## 10. Viva Voce Examination Questions & Model Answers

### Question 1: Why is it physically impermissible to use an isothermal wind CFD simulation to predict urban air temperature?
**Answer:**  
An isothermal wind simulation solves only the conservation of mass and linear momentum under the assumption of constant density and temperature. Temperature is governed by a separate partial differential equation—the thermal energy equation—which involves distinct physical phenomena that have no counterpart in isothermal aerodynamics:
1. **Surface Energy Balance and Solar Radiation:** Surface temperatures depend heavily on solar irradiation, building surface albedo, shading geometry, material heat capacity, and thermal conductivity. An isothermal solver has zero representation of solar radiative fluxes.
2. **Two-Way Buoyancy Coupling:** Temperature differences induce density variations that generate vertical buoyant forces ($g \beta \Delta T$). In low-to-moderate wind conditions ($\text{Ri} > 0.1$), buoyant updrafts along heated walls significantly alter street-canyon circulation patterns, displacing or destroying the mechanical vortex.  
Attempting to predict temperature from an isothermal wind solver lacks the fundamental physical governing equations, boundary conditions, and source terms, rendering the model physically invalid.

### Question 2: What is the vortex stretching term in the 3D vorticity equation, and why does its absence in 2D simulations distort urban wake cavities?
**Answer:**  
In the 3D vorticity transport equation, the vortex stretching term is $(\boldsymbol{\omega} \cdot \nabla)\mathbf{u}$. Physically, when fluid elements are elongated along vortex lines by velocity gradients, their cross-sectional diameter shrinks, intensifying the local vorticity to conserve angular momentum. This is the primary physical engine driving the forward energy cascade, transferring kinetic energy from large eddies down to small scales where viscous dissipation destroys them.  
In 2D simulations, velocity is restricted to the plane ($w = 0$) and vorticity is perpendicular ($\boldsymbol{\omega} = (0, 0, \omega_z)$). Consequently, the vortex stretching term is identically zero:
$$(\boldsymbol{\omega} \cdot \nabla)\mathbf{u} = \omega_z \frac{\partial \mathbf{u}}{\partial z} \equiv 0$$
Without vortex stretching, 2D flows cannot dissipate energy down to small scales; instead, they undergo an artificial inverse energy cascade where vortices merge into unrealistically large, stable whirlpools. Furthermore, in 2D, oncoming air cannot escape over the building roof, forcing 100% of the air to squeeze around the building sides. This artificially exaggerates lateral street canyon velocities and prolongs the leeward wake cavity to $15 - 25 H$ downstream, compared to $4 - 6 H$ in physical 3D reality.

### Question 3: What is the Richardson number ($\text{Ri}$), and how does it determine whether thermal buoyancy can be neglected in urban wind modeling?
**Answer:**  
The Richardson number is the dimensionless ratio of buoyant forces to inertial aerodynamic forces:
$$\text{Ri} = \frac{\text{Gr}}{\text{Re}^2} = \frac{g \beta \Delta T H}{U_{\text{ref}}^2}$$
where $g$ is gravity, $\beta$ is thermal expansion, $\Delta T$ is surface-to-air temperature difference, $H$ is building height, and $U_{\text{ref}}$ is reference wind speed.  
- When $\text{Ri} < 0.1$, the flow is in the **forced convection regime**. Inertial wind forces dominate completely, and buoyant forces are negligible. Under this condition, an isothermal CFD simulation (such as standard OpenFOAM `simpleFoam` or UrbanTALES) accurately represents the flow physics.
- When $0.1 \le \text{Ri} \le 10$, the flow is in the **mixed convection regime**. Both wind and thermal buoyancy are equally important; buoyancy modifies canyon shear layers and rooftop exchange.
- When $\text{Ri} > 10$, the flow is in the **free/natural convection regime** (typical of calm summer heat waves with $U_{\text{ref}} < 1\text{ m/s}$), where airflow is driven primarily by thermal updrafts rather than wind.

### Question 4: Explain the difference between CFD code verification and CFD model validation. Can a verified code yield an invalid simulation?
**Answer:**  
According to ASME V&V 20 guidelines:
- **Verification** evaluates whether the computational algorithm accurately solves the underlying mathematical equations ("solving the equations right"). It is demonstrated through numerical residual convergence ($< 10^{-5}$), code bug fixes, and grid convergence studies (GCI $< 2\%$).
- **Validation** evaluates whether the mathematical equations accurately represent the physical phenomenon of interest in the real world ("solving the right equations"). It is established by benchmarking computational results against independent physical experiments (e.g., wind tunnel measurements from CEDVAL).  
**Yes, a verified code can easily yield a physically invalid simulation.** For example, running an OpenFOAM simulation using the standard $k$-$\varepsilon$ turbulence model on a coarse, truncated 2D domain may achieve perfect mathematical convergence with residuals dropping to $10^{-7}$ (verification passed). However, because standard $k$-$\varepsilon$ suffers from the stagnation point anomaly and the 2D domain prohibits roof-level flow escape, the resulting wind velocity field deviates by over $100\%$ from physical reality (validation failed).

### Question 5: Why is the Signed Distance Field (SDF) an effective inductive bias when training convolutional neural networks on urban fluid flow?
**Answer:**  
A binary building footprint mask provides only discontinuous step values ($0$ for fluid, $1$ for building). Convolutional kernels applied to binary masks only detect the immediate 1-pixel boundary interface, forcing the network to use multiple deep convolutional layers to propagate wall proximity information into the fluid domain.  
In fluid mechanics, the presence of a solid wall exerts viscous shear stress that governs boundary layer velocity development across a wide distance from the wall ($u(y) \propto \ln(y)$). A Signed Distance Field (SDF) provides an explicit, smooth, continuous Euclidean distance gradient from every fluid pixel to the nearest wall surface. This provides the neural network with direct geometric knowledge of wall distance at every layer, acting as a strong inductive bias that helps the network learn near-wall boundary layer shear profiles with significantly lower error and faster training convergence.
