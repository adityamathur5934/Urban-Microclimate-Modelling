# Project Scope & Problem Formulation

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Task:** Task 1 — Research and Experiment Design  
**Owner:** Project Lead  
**Document Version:** 1.0.0  
**Branch:** `feature/project-research`  
**Status:** Approved Reference Scope  

---

## 1. Executive Overview & Problem Context

Urban microclimate refers to the localized atmospheric conditions—specifically airflow patterns, wind speeds, turbulence, temperature distributions, and solar exposure—occurring within the **Urban Canopy Layer (UCL)**, extending from the ground up to the mean roof height of buildings.

As global urbanization accelerates and climate change intensifies, urban centers experience acute environmental challenges:
- **Urban Heat Island (UHI) Effect:** Trapping of shortwave and longwave radiation in dense concrete and asphalt geometries, exacerbated by low convective cooling.
- **Pedestrian Wind Comfort & Safety:** Tall or poorly oriented buildings channel high-altitude winds down to ground level (downwash) and create hazardous high-speed wind jets through narrow street passages (Venturi and corner channelling effects), posing safety hazards to pedestrians and cyclists.
- **Urban Ventilation & Pollutant Dispersion:** Deep street canyons and recirculating wake cavities trap vehicular emissions and anthropogenic heat, preventing natural flushing by ambient winds.

Designing climate-resilient cities requires architects and urban planners to evaluate the aerodynamic and thermal impacts of building massing during early-stage master planning.

---

## 2. Aerodynamic Fundamentals: Building-Induced Flow Phenomena

When approaching wind encounters a bluff body (such as an isolated building or a cluster of buildings), the flow cannot follow the sharp contours of the solid obstacle, producing distinct, highly non-linear flow structures:

```
                          APPROACHING WIND U(z)
                                    |
                                    v
     +-------------------------------------------------------------------------+
     |                                                                         |
     |                     Free Stream Wind (Inflow)                           |
     |                       =======================>                          |
     |                                                                         |
     |                               Roof Separation Bubble                    |
     |                                     (~~~)                               |
     |                       +---------------+                                 |
     |     Stagnation        |               |                                 |
     |     Point (0.7 H) --->*   BUILDING    |       Wake Recirculation        |
     |       |               |    BLOCK      |            Cavity               |
     |       | (Downwash)    |               |            (u < 0)              |
     |       v               |               |            (<====)              |
     |   Horseshoe Vortex    |               |                                 |
     |   (@ ground level)    +---------------+---------- Reattachment -------- |
     |      (( o ))                              (4 - 6 H Downstream)          |
     |  =====================================================================  |
     |                               Ground Level                              |
     +-------------------------------------------------------------------------+
```

### Key Aerodynamic Mechanisms:

1. **Frontal Stagnation & Downwash:**
   Approaching wind decelerates against the windward facade, converting kinetic energy into pressure. Maximum static pressure occurs at approximately $0.6 - 0.7 \times H$ (building height). The downward vertical pressure gradient drives air toward the ground, forming a strong downward jet known as **downwash**.
2. **Horseshoe Vortex System:**
   The downwash jet strikes the ground at the windward base, rolls up into an upstream vortex, and wraps around both sides of the building base in the shape of a horseshoe, causing localized ground-level wind amplification.
3. **Corner Streams & Flow Separation:**
   At the sharp vertical edges of the windward facade, the flow detaches abruptly. Streamlines constrict rapidly around the corners, accelerating local velocities by a factor of $1.5 - 2.2 \times U_{\text{ref}}$, generating severe wind hazards for pedestrians.
4. **Wake Recirculation Cavity & Reverse Flow:**
   A large, under-pressured separation bubble forms directly behind the building's leeward facade. Flow reverses direction ($u < 0$), recirculating back toward the building base before reattaching to the ground surface at $4 - 6 \times H$ downstream. In dense urban layouts, this wake cavity traps heat and particulate matter.
5. **Channelling & Venturi Effect:**
   When wind enters narrow gaps between parallel buildings or street canyons, mass conservation forces flow acceleration:
   $$U_2 = U_1 \left( \frac{A_1}{A_2} \right)$$
   producing high-velocity wind jets along street corridors.
6. **Skimming Flow vs. Wake Interference vs. Isolated Roughness:**
   Depending on the building packing density ($\lambda_p = A_{\text{built}} / A_{\text{total}}$) and aspect ratio ($H/W$):
   - **Isolated Roughness Flow ($\lambda_p < 0.10$):** Wakes do not interact.
   - **Wake Interference Flow ($0.10 \le \lambda_p \le 0.25$):** The wake of an upstream building impinges on the windward face of the downstream building.
   - **Skimming Flow ($\lambda_p > 0.25$):** Flow skims across rooftops, establishing a stable, isolated recirculating vortex inside the street canyon below, severely reducing ground-level ventilation.

---

## 3. Computational Fluid Dynamics (CFD) vs. Neural Surrogates

### 3.1 The Role and Limitation of Traditional CFD
To resolve these complex flow structures, engineers use CFD solvers (such as OpenFOAM, ANSYS Fluent, or uDALES). CFD discretizes the continuous incompressible Navier-Stokes equations over thousands or millions of spatial grid cells:

$$\nabla \cdot \mathbf{U} = 0$$
$$\frac{\partial \mathbf{U}}{\partial t} + (\mathbf{U} \cdot \nabla)\mathbf{U} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{U} + \mathbf{f}$$

While CFD provides high physical fidelity, it has prohibitive computational drawbacks for architectural design:
- **Computational Cost:** A single steady-state 3D RANS simulation requires 1–3 hours on multi-core CPUs; a high-fidelity 3D Large-Eddy Simulation (LES) takes 24–72 hours on high-performance computing (HPC) clusters.
- **Iterative Bottleneck:** Urban master planning is an exploratory process requiring rapid evaluation of hundreds of building placement alternatives. Running full CFD for every geometry variant is intractable.

### 3.2 What is a Neural Surrogate?
A **neural surrogate model** is a deep learning model trained on a curated corpus of high-fidelity CFD simulations. Once trained, the neural network acts as a data-driven approximation of the Navier-Stokes operator $\mathcal{N}$:

$$\hat{\mathbf{Y}} = \mathcal{M}_\theta(\mathbf{X})$$

where:
- $\mathbf{X}$ represents the input geometry and inflow boundary conditions.
- $\hat{\mathbf{Y}}$ is the predicted spatial flow field (e.g. pedestrian wind velocity).
- $\theta$ are the learned parameters of the network.

**The Surrogate Advantage:**
While training a neural surrogate requires upfront compute, **inference is virtually instantaneous ($< 50\text{ ms}$ on GPU/CPU)**. This represents an acceleration of **$10^3 - 10^5\times$**, enabling real-time interactive microclimate feedback directly inside CAD tools.

---

## 4. Two Modalities: Summary-Based vs. Spatial Prediction

Our project directly compares two distinct paradigms for predicting urban microclimate:

```
+-----------------------------------------------------------------------------------------------+
|                                      URBAN GEOMETRY INPUT                                     |
+-----------------------------------------------------------------------------------------------+
                               |                                 |
                               v                                 v
     +-----------------------------------+             +----------------------------------+
     |         PARADIGM 1: AML           |             |       PARADIGM 2: SURROGATE      |
     |     Summary-Based Modeling        |             |      Spatial Field Prediction    |
     +-----------------------------------+             +----------------------------------+
                       |                                                 |
                       v                                                 v
     +-----------------------------------+             +----------------------------------+
     | Feature Extractor:                |             | Rasterizer:                      |
     | - Plan area density (lambda_p)    |             | - Building Height Map H(x, y)    |
     | - Frontal area density (lambda_f) |             | - Binary Building Mask M(x, y)   |
     | - Mean & Std Building Height      |             | - Signed Distance Field (SDF)    |
     | - Aerodynamic Roughness z0        |             | - Inflow U, V velocity channels  |
     | - Inflow speed & wind direction   |             | - Normalized coordinates (x, y)  |
     +-----------------------------------+             +----------------------------------+
                       |                                                 |
                       v                                                 v
     +-----------------------------------+             +----------------------------------+
     | Regressor (Tabular ML):           |             | Deep Neural Network (Vision/CNN):|
     | - Random Forest Regressor         |             | - 2D U-Net Architecture          |
     | - XGBoost Gradient Boosting       |             | - Multi-Scale Encoder-Decoder    |
     +-----------------------------------+             +----------------------------------+
                       |                                                 |
                       v                                                 v
     +-----------------------------------+             +----------------------------------+
     | OUTPUT: Scalar Quantity           |             | OUTPUT: Continuous 2D Field      |
     | Area-Averaged Wind Speed [m/s]    |             | Pedestrian Wind Speed Field      |
     | (Over entire urban canyon)        |             | U(x, y) at z = 1.75 m [m/s]      |
     +-----------------------------------+             +----------------------------------+
```

### Detailed Modality Comparison:

| Feature / Aspect | Model 1: Traditional AML Baseline | Model 2: Neural Spatial Surrogate |
|---|---|---|
| **Input Representation** | 1D Tabular feature vector $\mathbf{x} \in \mathbb{R}^{11}$ containing morphometric summary metrics ($\lambda_p, \lambda_f, H_{\text{mean}}, H_{\text{std}}, z_0, d, U_{\text{ref}}, \dots$). | Multi-channel 2D raster tensor $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ ($C=7$: height, mask, signed distance field, inflow vectors, coordinates). |
| **Model Architecture** | Tree-based ensembles: Random Forest Regressor or XGBoost. | Deep convolutional encoder-decoder: 2D U-Net with skip connections. |
| **Output Target** | Single scalar $y \in \mathbb{R}$: Canopy area-averaged wind speed $\bar{U}_{\text{ped}}$ [$\text{m/s}$]. | High-resolution 2D continuous matrix $\hat{\mathbf{Y}} \in \mathbb{R}^{H \times W}$: Pedestrian wind speed field at $z = 1.75\text{ m}$. |
| **Spatial Awareness** | **None.** Aggregates the entire neighborhood into bulk density numbers; loses all spatial layout, obstacle orientations, and street alignments. | **Full spatial context.** Resolves exact local building footprints, wall orientations, street widths, and windward/leeward spatial relationships. |
| **Operational Use-Case** | Regional screening, city-scale zoning guidelines, rapid macro-roughness estimation. | Microscale architectural design, pinpointing pedestrian danger spots, evaluating courtyard ventilation, building entryway comfort. |
| **Computational Footprint** | Microseconds to train and infer; minimal memory. | Requires GPU for training (~1–2 hours); inference takes 10–50 ms. |

---

## 5. Candidate Target Variable Analysis

The project brief considers three candidate target fields:
1. **2D Wind Speed Magnitude Field $U(x, y)$ at pedestrian height ($z = 1.75\text{ m}$)**
2. **2D Wind Velocity Vector Field $(u(x, y), v(x, y))$ at pedestrian height**
3. **2D Temperature Field $T(x, y)$ at pedestrian height**

### Systematic Trade-Off Matrix:

| Evaluation Criterion | Candidate A: Wind Speed Magnitude $U(x, y)$ | Candidate B: Velocity Vector Field $(u, v)$ | Candidate C: Temperature Field $T(x, y)$ |
|---|---|---|---|
| **Physical Meaning** | Scalar speed $U = \sqrt{u^2 + v^2}$ governing convective wind sensation. | Directional horizontal airflow vector $(u, v)$ resolving recirculations and stagnation. | Convective and radiative thermal field governing Urban Heat Island effect. |
| **Pedestrian Comfort Relevance** | **Direct.** Industry comfort standards (Lawson, Davenport, NEN 8100) are defined strictly on scalar wind speed thresholds. | Indirect (magnitude must be derived from $\sqrt{u^2 + v^2}$). | **Direct.** Governs Universal Thermal Climate Index (UTCI) and thermal comfort. |
| **Data Availability (UrbanTALES)** | **100% Available.** Pre-calculated in UrbanTALES across all 538 layouts. | **100% Available.** $u, v, w$ components available in LES outputs. | **0% Available.** UrbanTALES is an **isothermal** dataset (neutral ABL, no thermal scalar). |
| **Local CFD Feasibility** | Highly feasible in OpenFOAM ($k$-$\omega$ SST `simpleFoam`). | Highly feasible in OpenFOAM ($k$-$\omega$ SST `simpleFoam`). | Requires coupled buoyant solver (`buoyantSimpleFoam`) + surface solar radiation models. |
| **Surrogate Complexity** | **Moderate.** Single target channel ($C_{\text{out}} = 1$), strictly non-negative ($U \ge 0$). | **Slightly Higher.** Multi-channel ($C_{\text{out}} = 2$), signed values, requires vector coherence. | **High.** Requires modeling coupled buoyancy, material albedo, and solar radiation geometry. |
| **Recommendation** | **PRIMARY TARGET** | **SECONDARY AUXILIARY TARGET** | **DEFERRED / OUT OF SCOPE** |

### Definitive Recommendation & Scientific Justification:
- **Primary Target:** We select **pedestrian wind-speed magnitude ($U$ at $z = 1.75\text{ m}$)** as our primary prediction target. It is directly applicable to pedestrian comfort criteria, strictly supported by the UrbanTALES ground truth dataset, and physically realistic.
- **Secondary Target:** Velocity components $(u, v)$ are supported in our data pipeline as auxiliary multi-task channels to encourage the network to learn directional streamline continuity.
- **Temperature Status:** Because UrbanTALES does not model heat transfer, and generating hundreds of buoyant non-isothermal simulations locally on a laptop is computationally unfeasible, **temperature prediction is formally ruled out of the core scope** unless the college provides an external thermal CFD corpus.

---

## 6. Project Scope Boundaries, Assumptions, and Non-Goals

### In-Scope:
- Ingestion and normalization of urban building layouts across idealized arrays and realistic city neighborhood topologies.
- Feature extraction of 11 morphometric parameters for traditional tabular ML.
- Rasterization of building heights, masks, and signed distance fields (SDF) for deep learning.
- Implementation of Model 1 (Random Forest / XGBoost baseline) predicting canopy-averaged wind speed.
- Implementation of Model 2 (2D U-Net architecture) predicting continuous pedestrian wind speed fields.
- Rigorous holdout testing on unseen building layouts and unseen layout families.
- Quantitative evaluation using Field MAE, Normalized RMSE, $R^2$, and 2D spatial error maps.
- Initial uncertainty quantification via Monte Carlo Dropout and spatial residual analysis.

### Out-of-Scope (Non-Goals):
- **No Large Raw Downloads:** Downloading the full 35 TB 3D UrbanTALES volume is strictly prohibited. Only 2D pedestrian slices ($z = 1.75\text{ m}$) are used.
- **No Local 3D LES Runs:** Running 3D unsteady Large-Eddy Simulations locally is impossible on student hardware.
- **No Active Thermal / Radiation Modelling:** Excluded due to lack of thermal data in the primary reference dataset.
- **No Complex Vegetation / Tree Canopies:** Porous media vegetation models are excluded; buildings are treated as solid impervious bluff bodies.

---

## 7. Pedagogical Guide & Viva Voce Q&A

### Simple Explanation of the Core Concept
> *Imagine wind blowing through a city like water rushing through rocks in a stream. In front of a large boulder, water hits the wall and pushes down, while at the edges it speeds up into rapid jets. Behind the boulder, there is a quiet, swirling whirlpool. CFD calculates how every drop of water moves by solving complex physics equations step by step over hours. A neural surrogate takes an aerial photograph of the boulders and instantly draws where the fast streams and quiet whirlpools are based on what it learned from hundreds of previous simulations.*

### Viva Voce Questions & Model Answers

**Q1: Why is an area-averaged wind speed insufficient for evaluating pedestrian wind safety?**  
*Answer:* Wind hazards are localized phenomena. A building's sharp corners can generate localized wind jets exceeding $12\text{ m/s}$ (causing pedestrian instability), while directly behind the building the air is stagnant ($< 1\text{ m/s}$). An area-averaged metric averages these extremes together to report a benign $4\text{ m/s}$, completely masking localized hazard zones and dangerous stagnation pockets.

**Q2: What is the physical role of the Signed Distance Field (SDF) in the raster input?**  
*Answer:* A binary building mask only tells the neural network whether a pixel is solid or fluid. However, fluid dynamics is governed by wall-bounded boundary layers: viscous friction and shear stresses depend strongly on the distance to solid obstacles. The SDF provides an explicit, smooth continuous gradient representing proximity to walls, giving convolutional filters direct geometric information about obstacle proximity without having to infer it through deep receptive fields.

**Q3: Why must building interiors be masked out during neural network loss computation?**  
*Answer:* Inside solid building walls, fluid velocity is physically zero or undefined. If building interiors are included in unmasked mean squared error (MSE) loss, the network spends capacity memorizing trivial zero-valued pixels inside building footprints rather than optimizing complex flow gradients in street canyons. Masking ensures that training loss and evaluation metrics reflect only the fluid domain where microclimate physics actually occurs.
