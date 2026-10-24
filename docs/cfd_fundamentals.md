# Computational Fluid Dynamics (CFD) Fundamentals & Numerical Formulation for Urban Microclimate

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Task:** Task 2 — CFD Fundamentals and Physical Assumptions  
**Owner:** CFD and Scientific Computing Lead  
**Document Version:** 1.0.0  
**Branch:** `feature/cfd-fundamentals`  
**Status:** Approved Reference Specification  

---

## 1. Executive Summary & Purpose

The purpose of this document is to establish the mathematical, physical, and numerical foundations of Computational Fluid Dynamics (CFD) required to model urban canopy microclimates and provide reference ground truth for neural surrogate architectures.

A neural surrogate is only as dependable as the physics encoded in its training targets. To train, evaluate, and diagnose data-driven deep learning models (such as 2D U-Nets) and automated machine learning baselines (Random Forests / XGBoost), we must precisely understand:
1. What the Navier–Stokes equations represent and how they govern fluid motion through complex obstacle arrays.
2. How turbulent bluff-body aerodynamic phenomena (separation, stagnation, recirculation bubbles, vortex shedding, and street-canyon channelling) emerge from these governing equations.
3. How continuous differential equations are discretized into finite-volume algebraic systems over spatial meshes.
4. How numerical convergence, residual monitoring, and mesh sensitivity govern simulation reliability.
5. How high-dimensional CFD solution fields are sampled, interpolated, and mapped into structured raster tensors conforming to the project's data contracts (`src.data.contract`).

> [!IMPORTANT]
> **Task Scope & Simulation Execution Disclaimer:**  
> In strict accordance with the project guidelines for Task 2, **no computational simulations were executed during this task**. This document establishes the theoretical, numerical, and engineering specifications. All references to solvers, boundary conditions, meshes, and field exports define the verified operational protocol and mathematical framework for the project.

---

## 2. Fluid Mechanics & Governing Equations of Urban Airflow

### 2.1 The Continuum Hypothesis and Fluid Properties

At the scale of urban microclimate modeling (domain extents $L \sim 10^2 - 10^3\text{ m}$, building heights $H \sim 10 - 100\text{ m}$, street widths $W \sim 10 - 50\text{ m}$), air is treated as a continuous medium rather than a collection of discrete molecules. This assumption is verified by evaluating the **Knudsen number** ($Kn$):

$$Kn = \frac{\lambda_{\text{mfp}}}{L_{\text{char}}}$$

where $\lambda_{\text{mfp}} \approx 6.8 \times 10^{-8}\text{ m}$ is the mean free path of air molecules at standard temperature and pressure ($T = 293.15\text{ K}$, $p = 101.325\text{ kPa}$), and $L_{\text{char}}$ is the characteristic length scale (such as minimum building width or boundary layer thickness, $\sim 1\text{ m}$). Because $Kn \ll 10^{-3}$, macroscopic fluid variables (velocity, pressure, density, temperature) are smooth, continuously differentiable spatial and temporal fields.

Key physical properties of air at standard atmospheric conditions ($20^\circ\text{C}$):
- **Density ($\rho$):** $1.205\text{ kg/m}^3$ (assumed constant under incompressible flow).
- **Dynamic Viscosity ($\mu$):** $1.81 \times 10^{-5}\text{ Pa}\cdot\text{s}$ ($1.81 \times 10^{-5}\text{ kg}/(\text{m}\cdot\text{s})$).
- **Kinematic Viscosity ($\nu = \mu / \rho$):** $1.50 \times 10^{-5}\text{ m}^2/\text{s}$.

### 2.2 Velocity Field, Pressure, and Viscous Stresses

The kinematic state of the fluid is described by the continuous three-dimensional velocity vector field:

$$\mathbf{u}(\mathbf{x}, t) = \begin{bmatrix} u(x, y, z, t) \\ v(x, y, z, t) \\ w(x, y, z, t) \end{bmatrix} = u_i(\mathbf{x}, t) \quad (i = 1, 2, 3)$$

- **Streamwise velocity component ($u$ or $u_1$):** Velocity along the primary wind direction (typically aligned with the positive $x$-axis in canonical setups).
- **Spanwise / Lateral velocity component ($v$ or $u_2$):** Cross-stream horizontal velocity along the $y$-axis, resolving cross-canyon jets and vortex lateral spreading.
- **Vertical velocity component ($w$ or $u_3$):** Velocity normal to the ground along the $z$-axis, capturing frontal downwash, rooftop up-drafts, and buoyant thermal plumes.

#### Pressure Fields in Atmospheric Flows:
1. **Static Pressure ($p$):** Thermodynamic fluid pressure acting isotropically across any control surface [$\text{Pa}$ or $\text{N/m}^2$]. In incompressible solvers like OpenFOAM, pressure is normalized by density to yield the **kinematic pressure**:
   $$p_{\text{kin}} = \frac{p}{\rho} \quad \left[\frac{\text{m}^2}{\text{s}^2}\right]$$
2. **Dynamic Pressure ($q$):** Kinetic energy per unit volume:
   $$q = \frac{1}{2}\rho |\mathbf{u}|^2 = \frac{1}{2}\rho (u^2 + v^2 + w^2) \quad [\text{Pa}]$$
3. **Total (Stagnation) Pressure ($p_0$):**
   $$p_0 = p + \frac{1}{2}\rho |\mathbf{u}|^2$$
   When approaching wind decelerates to a complete stop against the windward facade of a building ($\mathbf{u} \to 0$), all dynamic pressure converts into static pressure, generating a localized high-pressure stagnation zone.

#### Viscous Stresses and the Cauchy Stress Tensor:
Forces acting on a fluid control volume consist of surface forces and body forces. Surface forces are parameterized by the symmetric **Cauchy stress tensor** $\boldsymbol{\sigma}$:

$$\sigma_{ij} = -p \delta_{ij} + \tau_{ij}$$

where $\delta_{ij}$ is the Kronecker delta, and $\tau_{ij}$ is the viscous shear stress tensor. Under Stokes' hypothesis for an isotropic Newtonian fluid:

$$\tau_{ij} = 2\mu S_{ij} - \frac{2}{3}\mu (\nabla \cdot \mathbf{u}) \delta_{ij}$$

where $S_{ij} = \frac{1}{2} \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)$ is the symmetric strain-rate tensor. For incompressible flow ($\nabla \cdot \mathbf{u} = 0$), this simplifies to:

$$\tau_{ij} = \mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)$$

---

### 2.3 Conservation of Mass (The Continuity Equation)

Consider an arbitrary, fixed control volume $V$ bounded by a closed surface $A$ with outward unit normal $\mathbf{n}$. The principle of mass conservation dictates that the rate of increase of mass inside $V$ must equal the net mass flux entering through boundary $A$:

$$\frac{\partial}{\partial t} \int_V \rho \, dV + \oint_A \rho (\mathbf{u} \cdot \mathbf{n}) \, dA = 0$$

Applying Gauss's Divergence Theorem to convert the surface integral to a volume integral:

$$\int_V \left[ \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) \right] dV = 0$$

Because this volume is arbitrary, the integrand must vanish identically everywhere, producing the differential continuity equation:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0 \quad \iff \quad \frac{\partial \rho}{\partial t} + \frac{\partial (\rho u_i)}{\partial x_i} = 0$$

#### The Incompressibility Condition in Urban Wind:
The Mach number in the atmospheric canopy layer is given by:

$$\text{Ma} = \frac{U_{\text{ref}}}{c_{\text{sound}}} \approx \frac{10\text{ m/s}}{343\text{ m/s}} \approx 0.029 \ll 0.3$$

Because $\text{Ma} < 0.3$, density variations due to dynamic pressure fluctuations are negligible ($\Delta \rho / \rho \sim \mathcal{O}(\text{Ma}^2) < 0.001$). Under neutral atmospheric stability, density $\rho$ is constant, and the continuity equation reduces to the **solenoidal (divergence-free) condition**:

$$\nabla \cdot \mathbf{u} = 0 \quad \iff \quad \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0 \quad \iff \quad \frac{\partial u_i}{\partial x_i} = 0$$

*Physical Interpretation:* In an incompressible flow, fluid cannot be compressed or concentrated into a point. Any local deceleration or convergence in one direction must be instantaneously compensated by acceleration or divergence in orthogonal directions.

---

### 2.4 Conservation of Momentum & The Navier–Stokes Equations

Newton's Second Law applied to a fluid particle states that the rate of change of linear momentum equals the sum of all surface stresses and volumetric body forces:

$$\rho \frac{D\mathbf{u}}{Dt} = \nabla \cdot \boldsymbol{\sigma} + \mathbf{f}_{\text{body}}$$

where $\frac{D}{Dt} = \frac{\partial}{\partial t} + (\mathbf{u} \cdot \nabla)$ is the material (substantial) derivative, capturing both local unsteady acceleration and non-linear advective transport.

Substituting the Newtonian stress tensor $\sigma_{ij} = -p\delta_{ij} + \mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)$ and assuming constant viscosity $\mu$:

$$\nabla \cdot \boldsymbol{\sigma} = -\nabla p + \mu \nabla^2 \mathbf{u} + \mu \nabla (\nabla \cdot \mathbf{u})$$

Since $\nabla \cdot \mathbf{u} = 0$, the last term vanishes. Dividing by density $\rho$ and introducing kinematic viscosity $\nu = \mu / \rho$, we arrive at the **Incompressible Navier–Stokes Equations**:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$

In index (Einstein summation) notation:

$$\frac{\partial u_i}{\partial t} + u_j \frac{\partial u_i}{\partial x_j} = -\frac{1}{\rho}\frac{\partial p}{\partial x_i} + \nu \frac{\partial^2 u_i}{\partial x_j \partial x_j} + f_i$$

```
+----------------------------------------------------------------------------------------------------------+
|                                    NAVIER-STOKES TERMS AND PHYSICAL ROLES                                |
+-------------------+-----------------------------+--------------------------------------------------------+
| Term              | Mathematical Expression     | Physical Meaning & Aerodynamic Role in Urban Canopy     |
+-------------------+-----------------------------+--------------------------------------------------------+
| Unsteady Rate of  |   \partial u_i / \partial t | Temporal acceleration; vanishes in steady-state RANS,  |
| Change            |                             | captures turbulent eddy shedding in LES / URANS.        |
+-------------------+-----------------------------+--------------------------------------------------------+
| Advective /       |   u_j (\partial u_i /       | Non-linear momentum transport by the fluid's own       |
| Convective        |        \partial x_j)        | velocity; drives flow around corners and momentum      |
| Acceleration      |                             | exchange between rooftop free stream and canyon.       |
+-------------------+-----------------------------+--------------------------------------------------------+
| Pressure Gradient | -(1 / \rho) (\partial p /   | Driving force created by pressure differences; drives  |
| Force             |              \partial x_i)  | frontal downwash and suction into leeward wake cavity. |
+-------------------+-----------------------------+--------------------------------------------------------+
| Molecular Viscous |  \nu (\partial^2 u_i /      | Molecular momentum diffusion and viscous dissipation;  |
| Diffusion         |       \partial x_j \partial x_j)| enforces the no-slip boundary condition at walls.   |
+-------------------+-----------------------------+--------------------------------------------------------+
| Body Forces       |   f_i                       | External volume forces (gravitational buoyancy in      |
|                   |                             | non-isothermal flows, Coriolis force in mesoscale).    |
+-------------------+-----------------------------+--------------------------------------------------------+
```

---

## 3. Aerodynamic Flow Phenomena Around Urban Bluff Bodies

Unlike streamlined aerodynamic bodies (such as airfoils) where flow adheres smoothly along the surface, buildings are sharp-edged **bluff bodies**. Bluff bodies impose severe adverse pressure gradients that trigger immediate boundary layer detachment.

```
                                  APPROACHING WIND U(z)
                                            |
                                            v
    +-----------------------------------------------------------------------------------+
    |                               Free Stream Flow U_inf                              |
    |                               ======================>                             |
    |                                                                                   |
    |                             Roof Separation Bubble                                |
    |                                  (~~~~~~~)                                        |
    |                           +-----------------------+                               |
    |                           |                       |                               |
    |        Stagnation         |                       |      Wake Cavity              |
    |        Point (~0.7 H) --->*       BUILDING        |   Recirculation Vortex        |
    |          |                |        BLOCK          |          (u < 0)              |
    |          | (Downwash)     |                       |          (<====)              |
    |          v                |                       |                               |
    |      Horseshoe Vortex     +-----------------------+---------- Reattachment ------- |
    |      (@ ground base)                                       (4 - 6 H Downstream)   |
    |    =============================================================================  |
    |                                  Ground Plane (z = 0)                             |
    +-----------------------------------------------------------------------------------+
```

### 3.1 Flow Separation and Adverse Pressure Gradients

Boundary layer theory demonstrates that when a moving fluid encounters an **adverse pressure gradient** ($\frac{\partial p}{\partial x} > 0$), momentum in the near-wall region is rapidly depleted by both viscous wall shear and the opposing pressure force.

According to the boundary layer momentum equation at the solid wall ($y = 0$, where $u = v = 0$):

$$\mu \left. \frac{\partial^2 u}{\partial y^2} \right|_{y=0} = \frac{\partial p}{\partial x}$$

1. In a **favorable pressure gradient** ($\frac{\partial p}{\partial x} < 0$), $\left. \frac{\partial^2 u}{\partial y^2} \right|_{y=0} < 0$, ensuring a full, stable velocity profile.
2. In an **adverse pressure gradient** ($\frac{\partial p}{\partial x} > 0$), $\left. \frac{\partial^2 u}{\partial y^2} \right|_{y=0} > 0$. Because $\frac{\partial^2 u}{\partial y^2}$ must eventually become negative to match the free stream far from the wall, an **inflection point** ($\frac{\partial^2 u}{\partial y^2} = 0$) appears in the velocity profile.
3. According to **Rayleigh's Inflection Point Theorem**, an inflection point renders the shear flow inviscidly unstable.
4. Downstream, the wall velocity gradient decreases until it reaches zero:
   $$\tau_w = \mu \left. \frac{\partial u}{\partial y} \right|_{y=0} = 0$$
   This location defines the **point of flow separation**. Beyond this point, $\left. \frac{\partial u}{\partial y} \right|_{y=0} < 0$, signifying that the fluid near the surface reverses direction ($u < 0$), detaching the boundary layer from the solid wall and projecting a turbulent shear layer into the free stream.

At the sharp 90-degree corners and roof parapets of buildings, the radius of curvature approaches zero ($r \to 0$). Fluid inertia makes it impossible for streamlines to turn the corner, causing **fixed, geometry-induced separation** that is virtually independent of Reynolds number.

### 3.2 Key Bluff-Body Aerodynamic Structures

1. **Frontal Stagnation & Downwash Jet:**
   The incoming atmospheric wind profile $U(z)$ exhibits increasing speed with height. Decelerating against the windward facade produces a vertical stagnation pressure gradient $\frac{\partial p}{\partial z} > 0$. Maximum pressure occurs at $z \approx 0.65 - 0.75 H$. Fluid below this point is driven downward by the adverse pressure gradient, forming a powerful **downwash jet** that strikes the ground at high speeds ($0.5 - 0.8 U_{\text{ref}}$).
2. **Horseshoe Vortex System:**
   The downwash jet impinging on the ground rolls up into a counter-rotating vortex in front of the windward base. This vortex is stretched laterally by the oncoming flow, wrapping around both flanks of the building like a horseshoe and generating severe pedestrian-level wind speeds near the building base.
3. **Corner Peeling Streams:**
   Flow detaching at the windward vertical edges constricts into high-speed corner streams. The local velocity accelerates significantly:
   $$U_{\text{corner}} \approx (1.4 - 2.2) \times U_{\text{ref}}$$
   These corner jets represent the single most dangerous microclimatic hazard for pedestrian safety.
4. **Leeward Wake Recirculation Cavity:**
   Behind the building's leeward wall, flow separation produces a sheltered, low-pressure cavity. Fluid reverses direction ($u < 0$) to form a recirculating bubble. The ground-level reattachment point occurs at:
   $$L_r \approx (4.0 - 6.5) \times H$$
   downstream of the leeward face. Within this cavity, wind speeds are extremely low, creating chronic stagnation zones that trap vehicular emissions.
5. **Rooftop Separation Bubble:**
   Flow detaching from the front roof edge forms a separation bubble over the roof. If the building length $L \ge 2H$, the bubble may reattach to the roof before reaching the trailing edge; for shorter roofs, it merges directly into the wake.

---

### 3.3 Street Canyon Flow Regimes

When buildings are grouped into urban blocks, the wake generated by an upstream building alters the inflow to downstream buildings. Oke (1988) classified street canyon airflow into three distinct regimes parameterized by the building height-to-width ratio ($H/W$) and building packing density ($\lambda_p = A_{\text{built}} / A_{\text{total}}$):

```
(a) Isolated Roughness (H/W < 0.3)      (b) Wake Interference (0.3 < H/W < 0.6)     (c) Skimming Flow (H/W > 0.65)
    ========>                               ========>                                   ========>
   +---+       +---+                       +---+       +---+                           +---+  ~~~~~~  +---+
   |   |       |   |                       |   |       |   |                           |   | (( o ))  |   |
   |   | (o)   |   |                       |   |  (o)  |   |                           |   |  Vortex  |   |
   +---+-------+---+                       +---+-------+---+                           +---+----------+---+
   Wakes do not interact                   Wake hits downstream facade                 Stable trapped vortex
```

1. **Isolated Roughness Flow ($H/W < 0.3$, $\lambda_p < 0.10$):**
   Buildings are spaced far apart. The wake cavity behind the windward building reattaches fully to the ground and recovers before reaching the next building. Airflow around each structure behaves as if isolated.
2. **Wake Interference Flow ($0.3 \le H/W \le 0.65$, $0.10 \le \lambda_p \le 0.25$):**
   The distance between buildings is less than the reattachment length ($W < L_r$). The recirculating wake of the upstream building impinges directly on the windward facade of the downstream building, creating complex three-dimensional vortex interactions, elevated turbulence, and secondary downwash structures.
3. **Skimming Flow ($H/W > 0.65$, $\lambda_p > 0.25$):**
   In dense urban layouts, the bulk flow skims over the rooftops without penetrating the street canyon. A stable, recirculating cavity vortex is driven purely by shear across the roof-level shear layer. Air within the canyon is decoupled from the free stream, resulting in minimal pedestrian wind speeds, poor air exchange, and severe accumulation of pollutants and heat.

---

## 4. Turbulence and Reynolds-Averaged Navier–Stokes (RANS)

### 4.1 The Physics of Turbulence and the Energy Cascade

Urban airflow occurs at extremely high Reynolds numbers:

$$\text{Re} = \frac{U_{\text{ref}} H}{\nu} \approx \frac{(5\text{ m/s})(20\text{ m})}{1.5 \times 10^{-5}\text{ m}^2/\text{s}} \approx 6.7 \times 10^6$$

At this Reynolds number, laminar flow is completely unstable. Flow is characterized by three-dimensional, chaotic, broadband velocity fluctuations across a continuous spectrum of length scales:
- **Integral scales ($l_0 \sim H \sim 10 - 100\text{ m}$):** Large, anisotropic eddies containing most of the kinetic energy, produced by mean flow shear and bluff-body separation.
- **Inertial subrange:** Energy is transferred inviscidly from larger to smaller eddies without loss, following Kolmogorov's $-5/3$ spectral scaling law: $E(\kappa) \propto \kappa^{-5/3}$.
- **Kolmogorov microscale ($\eta = (\nu^3 / \varepsilon)^{1/4} \sim 0.1 - 1.0\text{ mm}$):** Smallest scales where molecular viscosity dissipates kinetic energy into internal thermal energy.

Resolving all scales down to $\eta$ in a Direct Numerical Simulation (DNS) requires a spatial grid count scaling as $N_{\text{cells}} \propto \text{Re}^{9/4} \approx (6.7 \times 10^6)^{2.25} \sim 10^{15}\text{ cells}$, which is computationally impossible for urban engineering.

### 4.2 Reynolds Decomposition and Reynolds Stresses

To make computation tractable, Osborne Reynolds proposed decomposing any instantaneous flow variable $\phi(\mathbf{x}, t)$ into a time-averaged mean component $\bar{\phi}(\mathbf{x})$ and a turbulent fluctuating component $\phi'(\mathbf{x}, t)$:

$$u_i(\mathbf{x}, t) = \bar{u}_i(\mathbf{x}) + u_i'(\mathbf{x}, t), \quad p(\mathbf{x}, t) = \bar{p}(\mathbf{x}) + p'(\mathbf{x}, t)$$

where the time average over an averaging window $\Delta t$ is defined as:

$$\bar{u}_i(\mathbf{x}) = \frac{1}{\Delta t} \int_{t_0}^{t_0 + \Delta t} u_i(\mathbf{x}, t) \, dt, \quad \text{such that} \quad \overline{u_i'} \equiv 0$$

Substituting this decomposition into the incompressible Navier–Stokes equations and taking the time average yields the **Reynolds-Averaged Navier–Stokes (RANS) equations**:

$$\frac{\partial \bar{u}_i}{\partial x_i} = 0$$

$$\rho \bar{u}_j \frac{\partial \bar{u}_i}{\partial x_j} = -\frac{\partial \bar{p}}{\partial x_i} + \frac{\partial}{\partial x_j} \left[ \mu \left( \frac{\partial \bar{u}_i}{\partial x_j} + \frac{\partial \bar{u}_j}{\partial x_i} \right) - \rho \overline{u_i' u_j'} \right]$$

The non-linear convective term gives rise to the term $-\rho \overline{u_i' u_j'}$, known as the **Reynolds Stress Tensor** $\tau_{ij}^R$:

$$\boldsymbol{\tau}^R = -\rho \begin{bmatrix} \overline{u'^2} & \overline{u'v'} & \overline{u'w'} \\ \overline{u'v'} & \overline{v'^2} & \overline{v'w'} \\ \overline{u'w'} & \overline{v'w'} & \overline{w'^2} \end{bmatrix}$$

This symmetric tensor contains six independent unknown turbulent correlation terms representing convective momentum transport by turbulent fluctuations. The appearance of 6 new unknowns without 6 new equations is the classic **Turbulence Closure Problem**.

---

### 4.3 The Boussinesq Eddy-Viscosity Hypothesis

In 1877, Joseph Boussinesq proposed that turbulent stresses are proportional to the mean rate of strain, by analogy to molecular viscosity:

$$-\rho \overline{u_i' u_j'} = \mu_t \left( \frac{\partial \bar{u}_i}{\partial x_j} + \frac{\partial \bar{u}_j}{\partial x_i} \right) - \frac{2}{3} \rho k \delta_{ij}$$

where:
- $\mu_t$ is the **turbulent dynamic eddy viscosity** [$\text{Pa}\cdot\text{s}$], and $\nu_t = \mu_t / \rho$ is the **kinematic eddy viscosity** [$\text{m}^2/\text{s}$].
- $k = \frac{1}{2} \overline{u_i' u_i'} = \frac{1}{2} (\overline{u'^2} + \overline{v'^2} + \overline{w'^2})$ is the **Turbulent Kinetic Energy (TKE)** [$\text{m}^2/\text{s}^2$].
- The second term $-\frac{2}{3}\rho k \delta_{ij}$ ensures that the trace of the tensor equals $-2\rho k$, satisfying mathematical consistency.

> [!NOTE]
> Unlike molecular viscosity $\nu$, which is an intrinsic thermodynamic property of air, **turbulent viscosity $\nu_t$ is a property of the flow field**, varying by multiple orders of magnitude across the domain ($0 \le \nu_t \le 10^4 \nu$).

---

### 4.4 Turbulence Closure Selection: $k$-$\omega$ SST vs. Standard $k$-$\varepsilon$

To compute $\nu_t$, two-equation eddy-viscosity models solve two additional transport equations.

```
+---------------------------------------------------------------------------------------------------------+
|                               COMPARISON OF TWO-EQUATION TURBULENCE MODELS                               |
+-------------------+----------------------------+--------------------------------------------------------+
| Model             | Solved Quantities          | Performance in Urban Bluff-Body Aerodynamics           |
+-------------------+----------------------------+--------------------------------------------------------+
| Standard          | $k$ (Kinetic Energy)       | **Severe Stagnation Anomaly:** Overpredicts $k$ by      |
| $k$-$\varepsilon$ | $\varepsilon$ (Dissipation)| up to 400% in frontal stagnation zones, causing       |
| (Launder-Spalding)|                            | excessive $\nu_t$, delayed corner separation, and      |
|                   |                            | severely underestimated wake cavity lengths ($< 2.5H$).|
+-------------------+----------------------------+--------------------------------------------------------+
| Realizable        | $k$ (Kinetic Energy)       | Enforces realizability constraint ($C_\mu$ varies with |
| $k$-$\varepsilon$ | $\varepsilon$ (Dissipation)| strain rate); mitigates stagnation error but remains   |
| (Shih et al.)     |                            | sensitive to adverse pressure gradients near walls.    |
+-------------------+----------------------------+--------------------------------------------------------+
| Standard          | $k$ (Kinetic Energy)       | Robust in near-wall viscous boundary layers; however,  |
| $k$-$\omega$      | $\omega$ (Specific         | notoriously hypersensitive to arbitrary free-stream    |
| (Wilcox)          | Dissipation Rate)          | inlet values of $\omega_\infty$ (up to 100% variance). |
+-------------------+----------------------------+--------------------------------------------------------+
| **Menter $k$-$\omega$ | $k$ and $\omega$ with  | **Optimal Choice:** Blends Wilcox near walls with     |
| SST**             | Cross-Diffusion Blending   | $k$-$\varepsilon$ in free stream; incorporates         |
| (Menter, 1994)    | & Shear Stress Limiter     | Bradshaw shear-stress limiter; accurately predicts     |
|                   |                            | corner separation and wake length ($4.5 - 6 H$).       |
+-------------------+----------------------------+--------------------------------------------------------+
```

#### Why $k$-$\omega$ SST is Chosen for our Reference CFD Cases:
1. **Resolution of the Stagnation Anomaly:**
   In standard $k$-$\varepsilon$, eddy viscosity is $\nu_t = C_\mu k^2 / \varepsilon$. In irrotational stagnation flow upstream of a building facade, normal strain rate $S = \sqrt{2 S_{ij} S_{ij}}$ is enormous while vorticity $\Omega = 0$. Standard $k$-$\varepsilon$ produces unphysical turbulence generation $P_k \approx \nu_t S^2 \propto k^2$, leading to an explosive overprediction of $k$. $k$-$\omega$ SST incorporates a production limiter:
   $$P_k = \min(G_k, 10 \beta^* \rho k \omega)$$
   preventing turbulent energy build-up in stagnation zones.
2. **Cross-Diffusion Blending Function ($F_1$):**
   Menter introduced a smooth blending function $F_1$:
   - Near walls ($F_1 \to 1$), the model activates the Wilcox $k$-$\omega$ equations, which can be integrated directly through the viscous sublayer without numerical damping functions.
   - In the free stream far from obstacles ($F_1 \to 0$), the model transforms mathematically into the standard $k$-$\varepsilon$ formulation, eliminating the free-stream inlet sensitivity of pure $k$-$\omega$.
3. **Bradshaw Shear-Stress Limiter ($F_2$):**
   In adverse pressure gradients (boundary layers under deceleration), standard eddy-viscosity models overpredict shear stresses, delaying separation. SST modifies the eddy viscosity:
   $$\nu_t = \frac{a_1 k}{\max(a_1 \omega, \, S F_2)}$$
   where $a_1 = 0.31$ and $F_2$ is a second blending function. This limiter enforces Bradshaw's empirical observation that turbulent shear stress in boundary layers is bounded by $\tau \le \rho a_1 k$, allowing the model to accurately capture boundary layer separation on building facades and roofs.

---

## 5. CFD Domain Design, Meshing, and Boundary Conditions

To perform reliable CFD simulations that adhere to international guidelines (such as **COST Action 732** and **AIJ Japanese Guidelines for Urban CFD**), domain geometry and grid topology must follow strict geometric rules.

```
+----------------------------------------------------------------------------------------------------------+
|                                    COMPUTATIONAL DOMAIN DIMENSIONS                                       |
|                                                                                                          |
|   Top Boundary (Slip / Symmetry): H_domain >= 5 H_max                                                    |
|   +--------------------------------------------------------------------------------------------------+   |
|   |                                                                                                  |   |
|   |                                                                                                  |   |
|   |  Inflow Boundary                                                       Outflow Boundary          |   |
|   |  ABL Profile U(z), k(z), omega(z)                                      p = 0, zeroGradient       |   |
|   |  ======>                                  +-------+                    ======>                   |   |
|   |  ======>                                  |       |                    ======>                   |   |
|   |  ======>                                  |   H   |                    ======>                   |   |
|   |  ======>                                  |       |                    ======>                   |   |
|   |  =========================================+-------+============================================  |   |
|   |  Ground Boundary (Wall Function, z0)                                                             |   |
|   +--------------------------------------------------------------------------------------------------+   |
|   |<------------- 5 H ------------->|<--- W --->|<--------------------- 15 H ----------------------->|   |
|                                                                                                          |
+----------------------------------------------------------------------------------------------------------+
```

### 5.1 Domain Dimensions and Blockage Ratio

Let $H_{\text{max}}$ be the height of the tallest building in the cluster, and $A_{\text{frontal}}$ be the projected frontal area of all buildings facing the wind:
- **Upstream Fetch ($D_{\text{upstream}} \ge 5 H_{\text{max}}$):** Allows the atmospheric boundary layer profile to adapt to the domain without unphysical distortion before encountering obstacles.
- **Downstream Fetch ($D_{\text{downstream}} \ge 15 H_{\text{max}}$):** Ensures that the wake recirculation cavity ($L_r \sim 5 H$) and downstream wake recovery occur entirely within the domain without encountering the artificial numerical influence of the outlet boundary.
- **Top Boundary Clearance ($H_{\text{domain}} \ge 5 H_{\text{max}}$):** Prevents artificial vertical flow acceleration across rooftops.
- **Lateral Clearance ($D_{\text{lateral}} \ge 5 H_{\text{max}}$):** Prevents artificial side-wall blockage.
- **Blockage Ratio Rule:** The total projected obstacle area divided by the domain cross-sectional area must not exceed 3%:
  $$\text{Blockage Ratio} = \frac{\sum A_{\text{obstacle}}}{A_{\text{domain}}} < 3\%$$
  Exceeding 3% induces an artificial Venturi acceleration around building flanks, invalidating the calculated wind speed field.

### 5.2 Atmospheric Boundary Layer (ABL) Boundary Conditions

Urban wind does not enter as a uniform "plug" flow. Friction against upstream planetary terrain establishes an **Atmospheric Boundary Layer (ABL)** logarithmic velocity and turbulence profile (Richards & Hoxey, 1993):

1. **Velocity Profile ($U(z)$):**
   $$U(z) = \frac{u_*}{\kappa} \ln\left( \frac{z + z_0}{z_0} \right)$$
   where:
   - $u_*$ is the atmospheric friction velocity [$\text{m/s}$]:
     $$u_* = \frac{\kappa U_{\text{ref}}}{\ln\left(\frac{z_{\text{ref}} + z_0}{z_0}\right)}$$
   - $\kappa \approx 0.41$ is the von Kármán constant.
   - $z_0$ is the aerodynamic roughness length [$\text{m}$] ($z_0 \approx 0.03\text{ m}$ for open flat terrain, $z_0 \approx 0.5 - 1.5\text{ m}$ for dense urban terrain).
2. **Turbulent Kinetic Energy Profile ($k(z)$):**
   $$k(z) = \frac{u_*^2}{\sqrt{C_\mu}}$$
   where $C_\mu = 0.09$. Under neutral ABL assumptions, $k(z)$ is constant with height.
3. **Specific Dissipation Rate Profile ($\omega(z)$):**
   $$\omega(z) = \frac{u_*}{\kappa \sqrt{C_\mu} (z + z_0)}$$

### 5.3 Wall Boundary Conditions & Wall Functions

At solid surfaces (building walls and ground plane), viscous fluid adheres to the boundary, establishing the **no-slip condition**:

$$\mathbf{u}_{\text{wall}} = (0, 0, 0)$$

Near the wall, steep velocity gradients produce a thin boundary layer characterized by the dimensionless wall distance $y^+$:

$$y^+ = \frac{u_\tau y}{\nu}, \quad u_\tau = \sqrt{\frac{\tau_w}{\rho}}$$

```
+----------------------------------------------------------------------------------------------------------+
|                                    NEAR-WALL BOUNDARY LAYER STRUCTURE                                    |
+-------------------+---------------+----------------------------------------------------------------------+
| Layer             | $y^+$ Range   | Dominant Physics & Mathematical Behavior                             |
+-------------------+---------------+----------------------------------------------------------------------+
| Viscous Sublayer  | $y^+ < 5$     | Molecular viscous stresses dominate; linear velocity profile         |
|                   |               | $u^+ = y^+$. Reynolds stresses are negligible.                       |
+-------------------+---------------+----------------------------------------------------------------------+
| Buffer Layer      | $5 < y^+ < 30$| Both molecular viscosity and turbulent stresses are of equal order;   |
|                   |               | highly non-linear transitional dynamics.                             |
+-------------------+---------------+----------------------------------------------------------------------+
| Log-Law Layer     | $30 < y^+ <300$| Turbulent Reynolds stresses dominate; logarithmic velocity profile: |
|                   |               | $u^+ = \frac{1}{\kappa} \ln(E y^+)$.                                 |
+-------------------+---------------+----------------------------------------------------------------------+
```

In urban-scale CFD, resolving $y^+ < 1$ requires sub-millimeter cell sizes at building surfaces, yielding hundreds of millions of cells. We employ **standard/scalable wall functions** in OpenFOAM:
- `nutkWallFunction`: Computes turbulent viscosity $\nu_t$ at wall-adjacent centroid nodes based on the log-law, placing the first cell centroid at $30 < y^+ < 150$.
- `omegaWallFunction`: Computes $\omega$ at walls using blended formulas between viscous and logarithmic asymptotes.

---

## 6. Numerical Discretization, Residuals, Convergence, and Mesh Sensitivity

### 6.1 The Finite Volume Method (FVM) Discretization

CFD solvers like OpenFOAM discretize the continuous Navier–Stokes equations using the **Finite Volume Method (FVM)**:
1. The domain is divided into a mesh of contiguous polyhedral control volumes $V_P$.
2. The differential equations are integrated over each volume:
   $$\int_{V_P} \nabla \cdot (\rho \mathbf{u} \phi) \, dV = \int_{V_P} \nabla \cdot (\Gamma_\phi \nabla \phi) \, dV + \int_{V_P} S_\phi \, dV$$
3. Applying Gauss's Theorem converts volume integrals of divergence into surface flux sums over cell faces $f$:
   $$\sum_{f} (\rho \mathbf{u} \cdot \mathbf{n}_f \phi_f A_f) = \sum_{f} (\Gamma_\phi \nabla \phi \cdot \mathbf{n}_f A_f) + S_P V_P$$
4. Face values $\phi_f$ are evaluated using interpolation schemes:
   - **First-order Upwind:** Unconditionally stable, but introduces heavy **numerical diffusion** (artificial damping of vortices).
   - **Second-order Linear Upwind (OpenFOAM `linearUpwind`):** Second-order accurate; recommended for urban canopy flows to preserve sharp corner shear layers.

### 6.2 The SIMPLE Algorithm for Pressure-Velocity Coupling

Because incompressible flow has no independent equation of state for pressure (density is constant), pressure must serve as an internal constraint to enforce mass conservation ($\nabla \cdot \mathbf{u} = 0$).

OpenFOAM's steady solver `simpleFoam` uses the **Semi-Implicit Method for Pressure-Linked Equations (SIMPLE)**:
1. Initialize fields $\mathbf{u}^0, p^0$.
2. **Momentum Predictor:** Solve the discretized momentum equations using the current pressure field $p^*$ to obtain an intermediate velocity $\mathbf{u}^*$.
3. **Pressure Correction (Poisson Equation):** Enforce continuity $\nabla \cdot \mathbf{u} = 0$ by substituting $\mathbf{u}^*$ into continuity, yielding a Poisson equation for kinematic pressure:
   $$\nabla^2 p = \nabla \cdot \left( \frac{\mathbf{H}(\mathbf{u})}{a_P} \right)$$
4. **Velocity and Pressure Update:** Correct velocities and pressure using explicit under-relaxation factors ($\alpha_p \approx 0.3, \alpha_U \approx 0.7$):
   $$p^{n+1} = p^n + \alpha_p (p^* - p^n), \quad \mathbf{u}^{n+1} = \mathbf{u}^n + \alpha_U (\mathbf{u}^* - \mathbf{u}^n)$$
5. **Turbulence Transport:** Solve transport equations for $k$ and $\omega$.
6. Repeat until residuals satisfy convergence criteria.

### 6.3 Residuals and Convergence Criteria

Let the discretized linear system at iteration $m$ for cell $P$ be:

$$a_P \phi_P^m = \sum_N a_N \phi_N^m + b_P$$

The **unscaled local residual** $r_P^m$ is the imbalance:

$$r_P^m = b_P + \sum_N a_N \phi_N^m - a_P \phi_P^m$$

The normalized domain-wide residual $R^\phi$ reported in OpenFOAM log files is:

$$R^\phi = \frac{\sum_{P} |r_P^m|}{\sum_{P} |a_P \phi_P^m|}$$

#### Objective Convergence Criteria:
A simulation cannot be deemed converged solely because residuals dip below an arbitrary threshold. We require **three simultaneous criteria**:
1. **Residual Thresholds:** Normalized residuals for all variables ($p, U_x, U_y, U_z, k, \omega$) drop monotonically below **$10^{-5}$** (or reach a machine precision plateau $< 10^{-4}$).
2. **Integral Quantity Stabilization:** Key integral targets (total building drag coefficient $C_d$, canopy-averaged wind speed $\bar{U}_{\text{canopy}}$, and pressure drops across street canyons) vary by less than **$0.1\%$ over 200 consecutive iterations**.
3. **Point Probe Stationarity:** Velocity and pressure monitored at critical local probe points (in the wake cavity and corner streams) stabilize with zero drift.

### 6.4 Mesh Sensitivity Analysis and Grid Convergence Index (GCI)

To eliminate numerical discretization error (spatial truncation error), every CFD setup must undergo a systematic **Mesh Sensitivity Analysis** using the **Roache (1998) Grid Convergence Index (GCI)**:
1. Generate three systematically refined meshes: **Coarse** ($N_3$), **Medium** ($N_2$), and **Fine** ($N_1$), maintaining a grid refinement ratio:
   $$r = \frac{h_2}{h_1} = \left( \frac{N_1}{N_2} \right)^{1/D} \ge 1.30$$
2. Compute the apparent order of formal accuracy $p_{\text{order}}$:
   $$p_{\text{order}} = \frac{|\ln|(\phi_3 - \phi_2) / (\phi_2 - \phi_1)| + q(p)|}{\ln(r)}$$
3. Compute Richardson Extrapolation for the continuum zero-grid estimate $\phi_{\text{ext}}^{21}$:
   $$\phi_{\text{ext}}^{21} = \frac{r^p \phi_1 - \phi_2}{r^p - 1}$$
4. Calculate the Grid Convergence Index for the fine mesh:
   $$\text{GCI}_{\text{fine}}^{21} = \frac{1.25 |\phi_1 - \phi_2| / \phi_1}{r^p - 1} \times 100\%$$
If $\text{GCI}_{\text{fine}} < 2\%$, the solution is verified to be within the asymptotic range of grid convergence, ensuring that spatial discretization artifacts will not contaminate neural network targets.

---

## 7. Connecting CFD Fields to Neural Network Target Arrays

### 7.1 Field Slicing and Coordinate Transformation

OpenFOAM solves on an unstructured polyhedral mesh. Neural surrogates (such as 2D U-Nets) operate on uniform Cartesian 2D matrices ($\mathbf{Y} \in \mathbb{R}^{H \times W}$). The mapping process follows a verified mathematical interpolation pipeline:

```
+---------------------------------------------------------------------------------------------------+
|                                 CFD TO SURROGATE DATA EXTRACTION PIPELINE                         |
+---------------------------------------------------------------------------------------------------+
|  [3D Unstructured CFD Field] (simpleFoam / UrbanTALES LES)                                       |
|  Polyhedral cell centroids (x_c, y_c, z_c) with vector values U = (u, v, w)                        |
|                                             |                                                     |
|                                             v                                                     |
|  [Planar Slicing at Pedestrian Height]                                                            |
|  Extract cutting plane at z = 1.75 m (pedestrian chest height)                                    |
|                                             |                                                     |
|                                             v                                                     |
|  [Bilinear / Delaunay Rasterization onto Uniform Lattice]                                         |
|  Interpolate scattered points onto grid: Delta_x = Delta_y = 1.0 m (e.g., 128 x 128 or 256 x 256) |
|                                             |                                                     |
|                                             v                                                     |
|  [Field Extraction & Formatting]                                                                  |
|  1. Horizontal speed magnitude: U_mag = sqrt(u^2 + v^2)                                           |
|  2. Vector components: u(x, y), v(x, y)                                                           |
|  3. Turbulent Kinetic Energy: k(x, y)                                                             |
|  4. Canopy Average: y_scalar = (1 / N_fluid) * sum(U_mag)                                          |
|                                             |                                                     |
|                                             v                                                     |
|  [Building Interior Masking & Contract Packaging]                                                 |
|  Set building cells to zero: U_mag[building_mask == True] = 0.0                                   |
|  Package into validated src.data.contract.UrbanLayoutSample (.npz)                                |
+---------------------------------------------------------------------------------------------------+
```

### 7.2 The Mathematical and Practical Necessity of Masking

Inside solid building footprints, fluid flow does not exist ($\mathbf{u} \equiv 0$). If building interior pixels are left unmasked during neural surrogate training:
1. **Loss Dilution:** In a typical dense urban layout with packing density $\lambda_p = 0.40$, $40\%$ of all pixels in the tensor are solid building interiors with target value $0.0$. An unweighted Mean Squared Error (MSE) loss:
   $$\mathcal{L}_{\text{unmasked}} = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W (Y_{ij} - \hat{Y}_{ij})^2$$
   rewards the network heavily for memorizing trivial zeros inside solid blocks, artificially deflating loss values while failing to resolve complex shear layers in narrow street canyons.
2. **Masked Loss Formulation:**
   We enforce strict **fluid-domain masking** across all loss functions and evaluation metrics:
   $$\mathcal{L}_{\text{masked}} = \frac{\sum_{i=1}^H \sum_{j=1}^W \mathbb{I}_{M_{ij} = 0} \cdot |Y_{ij} - \hat{Y}_{ij}|}{\sum_{i=1}^H \sum_{j=1}^W \mathbb{I}_{M_{ij} = 0}}$$
   where $M_{ij} \in \{0, 1\}$ is the binary building mask ($1$ for solid building, $0$ for fluid).

---

## 8. Comprehensive Glossary of CFD & Aerodynamic Terms

- **Adverse Pressure Gradient ($\frac{\partial p}{\partial x} > 0$):** Pressure increasing in the direction of flow, causing fluid deceleration, boundary layer thickening, and separation.
- **Atmospheric Boundary Layer (ABL):** The bottom layer of the troposphere directly influenced by contact with the Earth's planetary surface, characterized by turbulent mixing and a logarithmic velocity shear profile.
- **Bluff Body:** An unstreamlined aerodynamic body with sharp or steep edges where flow cannot adhere, causing prominent separation bubbles and broad turbulent wakes.
- **Bradshaw Assumption:** The empirical principle that in turbulent boundary layers under adverse pressure gradients, shear stress is directly proportional to turbulent kinetic energy ($\tau = \rho a_1 k$).
- **Continuity Equation:** The mathematical statement of mass conservation ($\nabla \cdot \mathbf{u} = 0$ for incompressible flow).
- **Corner Peeling Jet:** High-velocity localized flow acceleration around the sharp vertical edges of a building, often reaching $1.5 - 2.2 \times U_{\text{ref}}$.
- **Downwash:** Downward jet of air along the windward facade driven by the vertical stagnation pressure gradient, impinging on the ground base.
- **Finite Volume Method (FVM):** Numerical discretization technique that solves conservation laws in integral form over discrete control volumes, guaranteeing exact conservation of fluxes.
- **Grid Convergence Index (GCI):** Standardized metric based on Richardson extrapolation quantifying spatial discretization uncertainty between refined grid levels.
- **Horseshoe Vortex:** A rolled-up vortex system formed at the windward base of a building by downwash impingement, wrapping around the building flanks.
- **Incompressible Flow:** Flow regime where fluid density variations are negligible ($\text{Ma} < 0.3$), decoupling pressure from thermodynamic state equations.
- **$k$-$\omega$ SST Model:** Menter's Shear Stress Transport two-equation eddy-viscosity model, blending $k$-$\omega$ near walls with $k$-$\varepsilon$ in the free stream while limiting shear stress.
- **Kinematic Pressure ($p/\rho$):** Static pressure divided by fluid density, with dimensions of velocity squared [$\text{m}^2/\text{s}^2$].
- **Kinematic Viscosity ($\nu$):** Ratio of dynamic viscosity to fluid density ($\nu = \mu / \rho$), representing momentum diffusivity.
- **No-Slip Condition:** Physical requirement that viscous fluid at a solid boundary has zero relative velocity to the wall ($\mathbf{u} = 0$).
- **Recirculation Cavity:** Under-pressured leeward wake bubble where flow reverses direction ($u < 0$) back toward the obstacle base.
- **Reynolds Number ($\text{Re}$):** Dimensionless ratio of inertial forces to viscous forces ($\text{Re} = U L / \nu$).
- **Reynolds Stresses ($-\rho \overline{u_i' u_j'}$):** Apparent stresses arising from time-averaging turbulent convective momentum fluctuations.
- **SIMPLE Algorithm:** Semi-Implicit Method for Pressure-Linked Equations, iterative predictor-corrector procedure for solving incompressible Navier–Stokes equations.
- **Skimming Flow:** Urban street canyon regime ($H/W > 0.65$) where bulk flow skims over rooftops, driving a stable trapped recirculating cavity vortex below.
- **Stagnation Point:** Location on an obstacle surface where fluid velocity drops to zero and static pressure reaches its local maximum.
- **Turbulent Kinetic Energy ($k$):** Mean kinetic energy per unit mass associated with turbulent velocity fluctuations ($k = \frac{1}{2} \overline{u_i' u_i'}$).
- **Wall Function:** Semi-empirical algebraic equations bridging the log-law layer to solid walls without explicitly resolving the viscous sublayer.
- **$y^+$ (Dimensionless Wall Distance):** Normalized distance from a solid wall ($y^+ = u_\tau y / \nu$), used to determine boundary layer mesh resolution.

---

## 9. Relevant Verified References

1. **Franke, J., Hellsten, A., Schlünzen, H., & Carissimo, B. (2007).** *Best practice guideline for the CFD simulation of flows in the urban environment.* COST Action 732: Quality Assurance and Improvement of Microclimate Models.
2. **Tominaga, Y., Mochida, A., Yoshie, R., Kataoka, H., Nozu, T., Yoshikawa, M., & Shirasawa, T. (2008).** *AIJ guidelines for practical applications of CFD to pedestrian wind environment around buildings.* Journal of Wind Engineering and Industrial Aerodynamics, 96(10-11), 1749–1761.
3. **Menter, F. R. (1994).** *Two-equation eddy-viscosity turbulence models for engineering applications.* AIAA Journal, 32(8), 1596–1605.
4. **Richards, P. J., & Hoxey, R. P. (1993).** *Appropriate boundary conditions for computational wind engineering models using the $k$-$\varepsilon$ turbulence model.* Journal of Wind Engineering and Industrial Aerodynamics, 46, 145–153.
5. **Oke, T. R. (1988).** *Street design and urban canopy layer climate.* Energy and Buildings, 11(1-3), 103–113.
6. **Blocken, B. (2014).** *50 years of Computational Wind Engineering: Past, present and future.* Journal of Wind Engineering and Industrial Aerodynamics, 129, 69–102.
7. **Roache, P. J. (1998).** *Verification and Validation in Computational Science and Engineering.* Hermosa Publishers, Albuquerque, NM.
8. **Launder, B. E., & Spalding, D. B. (1974).** *The numerical computation of turbulent flows.* Computer Methods in Applied Mechanics and Engineering, 3(2), 269–289.

---

## 10. Viva Voce Examination Questions & Model Answers

### Question 1: What is the physical meaning of the convective acceleration term $(\mathbf{u} \cdot \nabla)\mathbf{u}$ in the Navier–Stokes equations, and why does it cause numerical difficulties?
**Answer:**  
The convective acceleration term $(\mathbf{u} \cdot \nabla)\mathbf{u}$ (or $u_j \frac{\partial u_i}{\partial x_j}$) represents the rate of change of velocity of a fluid particle as it moves through space due to spatial gradients in the velocity field. Physically, it is responsible for non-linear momentum transport—such as wind speeding up as it turns around a sharp building corner or fluid being entrained into a wake cavity.  
Numerically, this term is non-linear (velocity multiplied by its own spatial derivative). This non-linearity prevents analytical solutions, necessitates iterative solver algorithms (like SIMPLE), produces asymmetric discretized matrix coefficients, and introduces numerical instabilities (such as unphysical oscillations) unless bounded discretization schemes (like upwinding or TVD limiters) are employed.

### Question 2: Why does the standard $k$-$\varepsilon$ model fail around the windward corners of bluff buildings, and how does the $k$-$\omega$ SST model resolve this?
**Answer:**  
In the stagnation zone upstream of a building's windward facade, the fluid undergoes strong normal strain without rotation ($\Omega \approx 0$). The standard $k$-$\varepsilon$ model computes turbulence production as $P_k \approx \nu_t S^2$, with $\nu_t = C_\mu k^2 / \varepsilon$. Because the irrotational strain rate $S$ is very large, the model produces an unphysical surge in turbulent kinetic energy ($P_k \propto k^2$). This is known as the **stagnation point anomaly**. The resulting excessive eddy viscosity artificially diffuses momentum, retards flow acceleration, suppresses sharp separation at windward corners, and severely underpredicts the wake cavity length downstream.  
The $k$-$\omega$ SST model resolves this by: (1) implementing a turbulent production limiter ($P_k \le 10 \beta^* \rho k \omega$), preventing excessive TKE generation in stagnation zones, and (2) incorporating Bradshaw's shear stress limiter ($\nu_t = a_1 k / \max(a_1 \omega, S F_2)$), which prevents unphysical shear stress growth in adverse pressure gradients, thereby accurately capturing boundary layer separation and realistic wake reattachment lengths ($4.5 - 6 H$).

### Question 3: Why cannot a CFD simulation be declared "converged" based solely on scaled residuals dropping below $10^{-4}$?
**Answer:**  
Residuals measure only the numerical imbalance in the discretized algebraic equations ($R = b - A\mathbf{x}$); they are normalized relative to the domain-integrated flux. In large computational domains or recirculating flows:
1. Residuals can drop below $10^{-4}$ while global integral quantities (such as total building drag force or leeward recirculation bubble length) are still drifting or oscillating.
2. In highly under-relaxed iterative schemes (e.g., small under-relaxation factors $\alpha \ll 1$), the change between successive iterations ($\Delta \phi = \phi^{m+1} - \phi^m$) becomes artificially small, making residuals appear low even though the solution is far from the true steady-state equilibrium.  
Therefore, standard best practice (COST 732, AIJ) mandates that true convergence requires both: (a) residual reduction below threshold ($10^{-5}$), (b) asymptotic flattening of target physical quantities (drag coefficient and canopy-average wind speed changing by $< 0.1\%$ over several hundred iterations), and (c) constant readings at local monitor probes.

### Question 4: How does the presence of an adverse pressure gradient lead to boundary layer separation at building walls?
**Answer:**  
In a fluid boundary layer, fluid particles near the wall have low kinetic energy due to viscous friction. When the flow encounters an adverse pressure gradient ($\frac{\partial p}{\partial x} > 0$), the increasing pressure exerts a force opposing the direction of motion. Because near-wall fluid has already lost its momentum, it cannot overcome this opposing pressure force. As a result, the near-wall velocity profile decelerates rapidly until the velocity gradient normal to the wall reaches zero:
$$\tau_w = \mu \left. \frac{\partial u}{\partial y} \right|_{y=0} = 0$$
Immediately downstream of this point, the opposing pressure gradient forces the near-wall flow to reverse direction ($u < 0$). The oncoming forward stream is pushed away from the wall surface, causing the boundary layer to detach and enter the free stream as a turbulent free shear layer, forming a recirculating separation cavity.

### Question 5: Why must CFD target fields sampled on unstructured meshes be interpolated and masked before being used in neural surrogate training?
**Answer:**  
Unstructured CFD meshes contain irregular, variable-density polyhedral cells concentrated near walls, whereas convolutional neural networks (like 2D U-Nets) require uniform, grid-structured 2D/3D tensors with fixed spatial resolution ($\Delta x, \Delta y$). Slicing and bilinear interpolation project the irregular CFD solution onto a Cartesian lattice.  
Masking is mathematically and physically mandatory because inside solid building walls, fluid flow does not physically exist ($\mathbf{u}$ is zero or undefined). If building interior pixels are left unmasked during neural network training, the loss function evaluates errors over solid geometry. In dense layouts ($30\% - 50\%$ building volume), an unmasked model achieves an artificially low MSE by simply outputting zero inside building footprints while failing to learn steep velocity gradients in narrow street canyons. Applying a binary building mask ensures that loss optimization and evaluation metrics focus strictly on fluid cells where urban microclimate physics occurs.
