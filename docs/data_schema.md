# Urban Microclimate Modelling: Data Contract & Schema Specification

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Task:** Task 3 — Dataset Discovery and Data Contract  
**Owner:** Project Lead  
**Document Version:** 2.0.0  
**Branch:** `feature/dataset-pipeline`  
**Status:** Approved Reference Contract  

---

## 1. Overview and Core Architectural Principles

To ensure complete scientific reproducibility, eliminate silent dimension mismatches, and prevent data leakage across team members, all data ingested, processed, and evaluated in this project must strictly comply with this **Data Contract**:

1. **Strict Layout-Level & Family-Level Holdout (Zero Spatial Leakage):**
   - Individual pixels, street corridors, or sub-blocks of a single urban layout must **never** be split across train, validation, and test sets. Because elliptic Navier-Stokes operators induce strong spatial autocorrelation across neighboring cells, pixel-level splitting leads to data leakage and artificially optimistic test errors.
   - Partitions are enforced strictly on unique **`layout_id`** or **`layout_family`** identifiers.
2. **Train-Only Normalization Fitting:**
   - Normalization statistics (means, standard deviations, min, max) must be fitted **exclusively on designated training layouts**. Test and validation layouts must be scaled using saved training statistics.
3. **Explicit Fluid Masking & Solid Geometry Preservation:**
   - Solid building interiors are boolean-flagged (`building_mask = True`). All loss functions, error metrics, and statistical normalizations evaluate exclusively over the outdoor fluid domain ($\text{building\_mask} = \text{False}$).
4. **Dimensional and Unit Consistency:**
   - All spatial dimensions are in meters [$\text{m}$], velocities in meters per second [$\text{m/s}$], angles in degrees [$^\circ$], and turbulent kinetic energy in square meters per second squared [$\text{m}^2/\text{s}^2$].

---

## 2. Complete Schema Specification

### 2.1 Layout Metadata (`LayoutMetadata`)
Describes domain bounds, spatial grid resolution, and architectural categorization.

| Field Name | Type | Unit | Constraints | Description |
|---|---|---|---|---|
| `layout_id` | `str` | — | Non-empty unique string | Unique identifier (e.g. `idealized_aligned_H25_01`, `realistic_sydney_042`). |
| `layout_family` | `LayoutFamily` | — | `idealized_aligned`, `idealized_staggered`, `realistic_neighborhood` | Morphological family of the building configuration. |
| `city_or_source` | `str` | — | Non-empty string | Real city source (e.g. `Sydney`, `London`) or synthetic label. |
| `domain_size_m` | `Tuple[float, float, float]` | $\text{m}$ | $L_x, L_y, L_z > 0$ | Bounding domain length, width, and height $(L_x, L_y, L_z)$. |
| `resolution_m` | `Tuple[float, float, float]` | $\text{m}$ | $\Delta x, \Delta y, \Delta z > 0$ | Spatial grid cell spacing (e.g. $(0.5, 0.5, 0.5)$ or $(1.0, 1.0, 0.5)$). |
| `grid_shape` | `Tuple[int, int]` | pixels | $N_y, N_x > 0$ | 2D horizontal grid dimensions $(N_y, N_x)$ where $N_y = L_y/\Delta y, N_x = L_x/\Delta x$. |

### 2.2 Geometry Data (`GeometryData`)
Defines the topographical 3D building morphology as 2D raster representations.

| Field Name | Type | Shape | Unit | Description |
|---|---|---|---|---|
| `height_map` | `float32` | $(N_y, N_x)$ | $\text{m}$ | Building top height above ground level at each cell. Ground = $0.0\text{ m}$. Values must be $\ge 0$. |
| `building_mask` | `bool` | $(N_y, N_x)$ | boolean | `True` where building is present ($H > 0$); `False` in outdoor fluid flow space. |

*Validation Rule:* $\forall (y, x): \text{height\_map}[y, x] > 10^{-4} \implies \text{building\_mask}[y, x] = \text{True}$. Negative heights or NaNs are forbidden.

### 2.3 Environmental & Inflow Conditions (`EnvironmentalConditions`)
Atmospheric driving conditions governing the microclimate flow.

| Field Name | Type | Unit | Allowed Range | Description |
|---|---|---|---|---|
| `inflow_speed_mps` | `float32` | $\text{m/s}$ | $> 0$ | Approach reference wind speed $U_{\text{ref}}$ (typically $3.0 - 10.0\text{ m/s}$). |
| `wind_direction_deg` | `float32` | degrees | $[0, 360)$ | Inflow angle $\delta$ ($0^\circ$: west-to-east $+x$; $90^\circ$: south-to-north $+y$). |
| `reference_height_m` | `float32` | $\text{m}$ | $> 0$ | Measurement height $z_{\text{ref}}$ where $U_{\text{ref}}$ is defined. |
| `roughness_length_z0_m` | `float32` | $\text{m}$ | $> 0$ | Aerodynamic surface roughness $z_0$ of ground terrain ($0.01 - 0.1\text{ m}$). |
| `ambient_temperature_k` | `Optional[float]` | $\text{K}$ | $> 0$ | Background temperature (default $293.15\text{ K}$, for thermal extensions). |

### 2.4 CFD Solver Metadata (`CFDSolverMetadata`)

| Field Name | Type | Description |
|---|---|---|
| `solver_name` | `str` | CFD simulation code (e.g. `uDALES`, `OpenFOAM`, `PALM`). |
| `turbulence_model` | `str` | Turbulence closure scheme (e.g. `LES-Deardorff`, `LES-Smagorinsky`, `RANS-k-omega-SST`). |
| `boundary_conditions` | `Dict[str, str]` | Inlet, outlet, ground, lateral, and top boundary condition specifications. |
| `time_averaging_window_s` | `Optional[float]` | Duration of stationary time averaging for LES flow statistics [$\text{s}$]. |
| `convergence_criteria` | `Optional[str]` | Convergence residual threshold (e.g. `1e-5` for RANS). |

### 2.5 Target Fields (`TargetFields`)

| Target Field | Type | Shape | Unit | Usage in Project |
|---|---|---|---|---|
| `pedestrian_wind_speed` | `float32` | $(N_y, N_x)$ | $\text{m/s}$ | **Primary Spatial Target** for Neural Surrogate (wind speed magnitude at $z = 1.75\text{ m}$). |
| `area_averaged_wind_speed` | `float32` | scalar | $\text{m/s}$ | **Primary Scalar Target** for Traditional AML Baseline (mean outdoor canopy wind speed). |
| `velocity_u` | `Optional[float32]` | $(N_y, N_x)$ | $\text{m/s}$ | Streamwise horizontal velocity component. |
| `velocity_v` | `Optional[float32]` | $(N_y, N_x)$ | $\text{m/s}$ | Spanwise horizontal velocity component. |
| `turbulent_kinetic_energy` | `Optional[float32]` | $(N_y, N_x)$ | $\text{m}^2/\text{s}^2$ | Secondary scalar field for turbulence intensity. |

---

## 3. Spatial Field to Area-Averaged Target Conversion

A critical bridge between the spatial neural surrogate and the traditional AML baseline is the **area-averaging transformation**. Converting a continuous 2D spatial field $U(x, y)$ into a single scalar $\bar{U}_{\text{area}}$ must rigorously preserve physical units, account for differential grid cell areas, and avoid solid-interior bias.

### 3.1 Mathematical Formulation
Let $\Omega$ be the full horizontal domain, and let $M(x, y) \in \{0, 1\}$ be the building footprint mask ($1$ inside solid building, $0$ in fluid). The integration domain is strictly the outdoor fluid space:
$$\Omega_{\text{outdoor}} = \{ (x, y) \in \Omega \mid M(x, y) = 0 \}$$

The area-averaged wind speed is defined by the surface integral:
$$\bar{U}_{\text{area}} = \frac{\iint_{\Omega_{\text{outdoor}}} U(x, y) \, dA}{\iint_{\Omega_{\text{outdoor}}} dA} = \frac{\sum_{i=1}^{N_y} \sum_{j=1}^{N_x} (1 - M_{i,j}) U_{i,j} \Delta A_{i,j}}{\sum_{i=1}^{N_y} \sum_{j=1}^{N_x} (1 - M_{i,j}) \Delta A_{i,j}}$$

where $\Delta A_{i,j} = \Delta x_{i,j} \times \Delta y_{i,j}$ is the differential area of grid cell $(i, j)$ in $\text{m}^2$.

```
+-------------------------------------------------------------------------------+
| UNMASKED (INCORRECT) INTEGRATION            MASK-TRACKED (CORRECT) INTEGRATION|
|                                                                               |
| Fluid: U = 4.0 m/s                          Fluid: U = 4.0 m/s                |
| Building: U = 0.0 m/s (zero-padded)         Building: M = True (excluded)     |
| Average over all cells:                     Average over fluid cells only:    |
| (4.0 + 4.0 + 0.0 + 0.0)/4 = 2.0 m/s <---    (4.0 + 4.0) / 2 = 4.0 m/s   <---  |
| ARTIFICIALLY DEPRESSED BY DENSITY!          TRUE PHYSICAL SPEED PRESERVED!    |
+-------------------------------------------------------------------------------+
```

### 3.2 Key Rules Implemented in `src/data/area_averaging.py`:
1. **Mask Tracking:** Zero-valued building cells are strictly omitted from the numerator and denominator, preventing artificial dilution of wind speeds as building density $\lambda_p$ increases.
2. **Differential Cell Weighting ($dA$):** For non-uniform grids, explicit cell area weights $\Delta A_{i,j}$ are applied. For uniform Cartesian grids, $\Delta x \Delta y$ factors out, maintaining exact physical unit consistency ($\text{m/s}$).
3. **Boundary Condition Exception Handling:** If a layout contains 100% solid building cells ($N_{\text{fluid}} = 0$), the function raises a descriptive `ValueError` rather than silently returning a `ZeroDivisionError` or `NaN`.

---

## 4. Multi-Channel Spatial Raster Alignment (Neural Surrogate Inputs)

The neural surrogate (2D U-Net) requires multi-channel spatial input tensors $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times N_y \times N_x}$ perfectly aligned with the target field $\mathbf{Y} \in \mathbb{R}^{C_{\text{out}} \times N_y \times N_x}$:

```
                              INPUT CHANNELS (C_in = 7)
+---------------------------------------------------------------------------------------+
| Ch 0: Height Map [m]         | Ch 1: Building Mask [0, 1] | Ch 2: Signed Distance [m] |
| H(x, y) building heights     | 1.0 = solid, 0.0 = fluid   | >0 fluid, <0 solid        |
|------------------------------+----------------------------+---------------------------|
| Ch 3: Inflow U [m/s]         | Ch 4: Inflow V [m/s]       | Ch 5 & 6: Coordinates     |
| U_ref * cos(delta)           | U_ref * sin(delta)         | x/Lx, y/Ly in [0, 1]      |
+---------------------------------------------------------------------------------------+
                                          |
                                          v  [2D U-Net Forward Pass]
                                          |
                               TARGET FIELD (C_out = 1)
+---------------------------------------------------------------------------------------+
| Ch 0: Pedestrian Wind Speed [m/s] (z = 1.75 m)                                        |
| Outdoor fluid flow speed (building interior masked to 0.0 m/s)                        |
+---------------------------------------------------------------------------------------+
```

### The Signed Distance Field (SDF):
Computed via `scipy.ndimage.distance_transform_edt`:
$$\text{SDF}(x, y) = \text{dist}((x, y), \text{building}) - \text{dist}((x, y), \text{fluid})$$
- $\text{SDF} > 0$ in the fluid flow domain (distance to nearest building wall in meters).
- $\text{SDF} < 0$ inside solid building volumes.
- $\text{SDF} = 0$ along the exact building facade boundary.
*Aerodynamic Purpose:* Fluid viscous shear stress and boundary-layer development are direct functions of wall distance. The SDF provides convolutional layers with smooth, continuous geometric gradients without requiring an excessively deep receptive field.

---

## 5. Normalization, Inverse Transformation, and Train-Only Statistics

### 5.1 Training-Only Statistical Fitting
To prevent data contamination, scalers are fitted **exclusively on the training split**:
$$\mu_c = \frac{1}{\sum_{k \in \text{Train}} N_{\text{fluid}}^{(k)}} \sum_{k \in \text{Train}} \sum_{(i,j) \in \text{fluid}} X_{c,i,j}^{(k)}$$
$$\sigma_c = \sqrt{\frac{1}{\sum_{k \in \text{Train}} N_{\text{fluid}}^{(k)}} \sum_{k \in \text{Train}} \sum_{(i,j) \in \text{fluid}} (X_{c,i,j}^{(k)} - \mu_c)^2}$$

### 5.2 Transformation & Exact Inversion Equations
- **Standard (Z-Score) Normalization:**
  $$\tilde{X}_{c,i,j} = \frac{X_{c,i,j} - \mu_c}{\sigma_c + \epsilon}$$
  $$X_{c,i,j} = \tilde{X}_{c,i,j} \cdot \sigma_c + \mu_c$$
- **Min-Max Scaling ($[0, 1]$):**
  $$\tilde{X}_{c,i,j} = \frac{X_{c,i,j} - X_{c,\text{min}}}{X_{c,\text{max}} - X_{c,\text{min}} + \epsilon}$$
  $$X_{c,i,j} = \tilde{X}_{c,i,j} \cdot (X_{c,\text{max}} - X_{c,\text{min}}) + X_{c,\text{min}}$$

### 5.3 Mask Preservation Rule:
After transformation and after inverse transformation:
$$\forall (i, j) \in \text{building}: \quad \tilde{X}_{c,i,j} = 0.0, \quad X_{c,i,j} = 0.0$$
Solid building cells remain strictly zero-valued, ensuring that denormalized predictions match physical boundaries.

---

## 6. End-to-End Walkthrough: From Source Data to Model Inputs & Targets

The following trace illustrates how a sample moves from raw disk storage through both modeling pipelines:

```
[RAW SOURCE DATA]
sample_london_042.npz
├── height_map: (64, 64) float32 (building heights 0 m to 45 m)
├── building_mask: (64, 64) bool (True inside building footprints)
├── pedestrian_wind_speed: (64, 64) float32 (LES reference slice at z = 1.75 m)
└── _metadata_json: {U_ref: 5.0 m/s, delta: 45 deg, dx: 1.0 m, dy: 1.0 m}
                        |
                        v
          [LayoutDataLoader.load_sample_npz]
       Creates validated UrbanLayoutSample object
                        |
       +----------------+----------------+
       |                                 |
       v                                 v
[MODEL 1: TRADITIONAL AML]        [MODEL 2: NEURAL SURROGATE]
GeometricFeatureExtractor()        LayoutRasterizer()
       |                                 |
Extracts 11 morphometrics:         Rasterizes 7 spatial channels:
1. lambda_p = 0.28                 1. height_map_m (0 - 45 m)
2. lambda_f = 0.19                 2. building_mask (0 or 1)
3. H_mean = 22.4 m                 3. signed_distance_m (-15 m to +35 m)
4. H_std = 8.6 m                   4. inflow_u = 3.535 m/s
5. H_max = 45.0 m                  5. inflow_v = 3.535 m/s
6. lambda_w = 0.52                 6. coord_x_norm (0 to 1)
7. z0_macdonald = 0.42 m           7. coord_y_norm (0 to 1)
8. d_macdonald = 14.1 m                  |
9. U_ref = 5.0 m/s                 Raster input: (7, 64, 64) float32
10. cos(delta) = 0.707             Aligned target: (1, 64, 64) float32
11. sin(delta) = 0.707                   |
       |                           FieldNormalizer.transform()
Target calculation:                Scale fluid cells with train mu, sigma;
compute_area_averaged_wind_speed() Building cells kept at 0.0
Scalar target: y = 3.12 m/s              |
       |                           Model Forward Pass: 2D U-Net
FeatureNormalizer.transform()      Predicts normalized field Y_tilde
Feature vector: x in R^11                |
       |                           FieldNormalizer.inverse_transform()
Model: Random Forest / XGBoost     Recovers physical wind speeds in m/s
Predicts: y_hat = 3.08 m/s         Evaluates MAE, nRMSE, & error maps
```

---

## 7. Storage Format & Configuration Persistence

- **Disk Archive (`.npz`):** Samples are stored as compressed NumPy archives containing array blocks alongside serialized JSON metadata (`_metadata_json`).
- **Configuration Persistence (`data/preprocessing_config.json`):** Saves training normalization statistics (channel-wise means, standard deviations, min, max) and layout partition assignments (`train_layout_ids`, `val_layout_ids`, `test_layout_ids`), allowing deterministic test-time evaluation without re-fitting.
