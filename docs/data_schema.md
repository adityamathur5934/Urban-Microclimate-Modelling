# Urban Microclimate Modelling: Data Contract & Schema Specification

**Document Version:** 1.0.0 (Proposal for Team Coordination)  
**Authors:** Dataset and Data Engineering Team  
**Scope:** Data interface definition between CFD data ingestion, Traditional ML Baseline, and Neural Surrogates.

---

## 1. Overview and Core Principles

To prevent data contamination, silent dimension mismatches, and reproducibility failures between team members, all data ingested and processed in this project must adhere to this **Data Contract**:

1. **Layout-Level Holdout (No Pixel-Level Splitting):**
   - Individual pixels or sub-regions of a single urban layout must **never** be split across train, validation, and test sets. Doing so causes spatial autocorrelation leakage and gives artificially optimistic performance metrics.
   - Splits must be performed exclusively at the **layout ID** level or **layout family** level.
2. **Train-Only Normalization:**
   - All scalers (mean, standard deviation, min, max) must be fitted strictly on the designated training split. Test layouts must be normalized using saved training parameters.
3. **Explicit Masking of Solid Geometry:**
   - Solid building interiors are masked out (`building_mask = True`). Fluid cells are evaluated using outdoor masks.

---

## 2. Layout Schema Specification

### 2.1 Metadata (`LayoutMetadata`)
Describes the spatial domain, layout categorization, and spatial resolution.

| Field Name | Type | Unit | Allowed Values / Constraints | Description |
|---|---|---|---|---|
| `layout_id` | `string` | — | Non-empty unique string (e.g. `idealized_aligned_001`) | Unique identifier for the layout. |
| `layout_family` | `enum` | — | `idealized_aligned`, `idealized_staggered`, `realistic_neighborhood` | Morphological family of the urban configuration. |
| `city_or_source` | `string` | — | e.g. `Sydney`, `London`, `IdealizedArray` | City of origin or synthetic benchmark label. |
| `domain_size_m` | `(float, float, float)` | $\text{m}$ | $L_x, L_y, L_z > 0$ | Total domain bounding box dimensions $(L_x, L_y, L_z)$. |
| `resolution_m` | `(float, float, float)` | $\text{m}$ | $\Delta x, \Delta y, \Delta z > 0$ | Spatial grid resolution (e.g. $(0.5, 0.5, 0.5)$ or $(1.0, 1.0, 0.5)$). |
| `grid_shape` | `(int, int)` | pixels | $N_y, N_x > 0$ | Dimensions of horizontal 2D grid $(N_y, N_x)$. |

### 2.2 Geometry (`GeometryData`)
Defines the topographical 3D building morphology as 2D raster representations.

| Field Name | Type | Shape | Unit | Description |
|---|---|---|---|---|
| `height_map` | `float32` | $(N_y, N_x)$ | $\text{m}$ | Building height above ground level at each $(y, x)$ coordinate. Ground = $0.0\text{ m}$. |
| `building_mask` | `bool` | $(N_y, N_x)$ | boolean | `True` where building is present ($H > 0$); `False` in outdoor fluid flow space. |

*Validation Rule:* $\forall (y, x): \text{height\_map}[y, x] > 0 \implies \text{building\_mask}[y, x] = \text{True}$. Negative heights are forbidden.

### 2.3 Environmental & Inflow Conditions (`EnvironmentalConditions`)
Atmospheric driving conditions governing the microclimate simulation.

| Field Name | Type | Unit | Range | Description |
|---|---|---|---|---|
| `inflow_speed_mps` | `float32` | $\text{m/s}$ | $> 0$ | Reference approach wind speed $U_{\text{ref}}$ (e.g., $5.0\text{ m/s}$). |
| `wind_direction_deg`| `float32` | degrees | $[0, 360)$ | Wind direction $\delta$ ($0^\circ$: west-to-east $+x$; $90^\circ$: south-to-north $+y$). |
| `reference_height_m`| `float32` | $\text{m}$ | $> 0$ | Reference height $z_{\text{ref}}$ where $U_{\text{ref}}$ is specified. |
| `roughness_length_z0_m`| `float32` | $\text{m}$ | $> 0$ | Aerodynamic surface roughness $z_0$ of ground terrain. |
| `ambient_temperature_k`| `Optional[float]` | $\text{K}$ | $> 0$ | Ambient background temperature (default $293.15\text{ K}$). |

### 2.4 CFD Solver Metadata (`CFDSolverMetadata`)

| Field Name | Type | Description |
|---|---|---|
| `solver_name` | `string` | Name of CFD code (e.g., `uDALES`, `PALM`, `OpenFOAM`). |
| `turbulence_model` | `string` | Turbulence closure scheme (e.g., `LES-Deardorff`, `LES-Smagorinsky`, `RANS-k-epsilon`). |
| `boundary_conditions` | `dict[str, str]` | Inflow, outflow, ground, lateral, and domain top boundary conditions. |
| `time_averaging_window_s`| `float` | Averaging duration in seconds for stationary flow statistics. |

### 2.5 Target Fields (`TargetFields`)

| Target Field | Type | Shape | Unit | Usage |
|---|---|---|---|---|
| `pedestrian_wind_speed` | `float32` | $(N_y, N_x)$ | $\text{m/s}$ | **Primary spatial target** for Neural Surrogate: wind speed magnitude at $z = 1.75\text{ m}$. |
| `area_averaged_wind_speed`| `float32` | scalar | $\text{m/s}$ | **Primary scalar target** for Traditional ML baseline: mean wind speed over outdoor pedestrian cells. |
| `velocity_u` | `Optional[float32]` | $(N_y, N_x)$ | $\text{m/s}$ | Optional streamwise horizontal component. |
| `velocity_v` | `Optional[float32]` | $(N_y, N_x)$ | $\text{m/s}$ | Optional spanwise horizontal component. |
| `turbulent_kinetic_energy`| `Optional[float32]` | $(N_y, N_x)$ | $\text{m}^2/\text{s}^2$ | Optional secondary target for turbulence intensity. |

---

## 3. Data Representation for Models

### 3.1 Model 1: Traditional ML Baseline (Tabular Summary Features)
Extracts a 1D feature vector $\mathbf{x} \in \mathbb{R}^{11}$ per layout to predict scalar $y \in \mathbb{R}$ (`area_averaged_wind_speed`):

1. `plan_area_density_lambda_p`: $\lambda_p = \frac{A_{\text{built}}}{A_{\text{total}}}$
2. `frontal_area_density_lambda_f`: $\lambda_f = \frac{A_{\text{frontal}}}{A_{\text{total}}}$ (projected along wind angle $\delta$)
3. `mean_building_height_m`: $H_{\text{mean}}$ (over built footprint)
4. `std_building_height_m`: $H_{\text{std}}$
5. `max_building_height_m`: $H_{\text{max}}$
6. `wall_area_density_lambda_w`: Total vertical facade surface area / total lot area
7. `roughness_length_macdonald_m`: Aerodynamic roughness length $z_0$ computed via MacDonald et al. (1998)
8. `displacement_height_d_m`: Zero-plane displacement height $d$ (MacDonald et al., 1998)
9. `inflow_speed_mps`: Inflow velocity $U_{\text{ref}}$
10. `wind_direction_cos`: $\cos(\delta)$
11. `wind_direction_sin`: $\sin(\delta)$

### 3.2 Model 2: Neural Surrogate (Spatial Multi-Channel 2D Tensor)
Rasterized input tensor $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times N_y \times N_x}$ to predict spatial field $\mathbf{Y} \in \mathbb{R}^{C_{\text{out}} \times N_y \times N_x}$:

- **Input Channels ($C_{\text{in}} = 7$):**
  1. `height_map_m`: Raw building heights in meters.
  2. `building_mask`: $1.0$ for building footprint, $0.0$ for fluid cells.
  3. `signed_distance_m`: Signed distance transform in meters ($>0$ fluid, $<0$ building interior).
  4. `inflow_u_mps`: $U_{\text{ref}} \cos(\delta)$ spatially tiled.
  5. `inflow_v_mps`: $U_{\text{ref}} \sin(\delta)$ spatially tiled.
  6. `coord_x_norm`: Normalized horizontal coordinate $x / L_x \in [0, 1]$.
  7. `coord_y_norm`: Normalized vertical coordinate $y / L_y \in [0, 1]$.
- **Target Channels ($C_{\text{out}} = 1$ to $4$):**
  1. `pedestrian_wind_speed_mps`: Wind speed at $z = 1.75\text{ m}$ (zero inside building footprints).
  2. (Optional auxiliary) `velocity_u`, `velocity_v`, `turbulent_kinetic_energy`.

---

## 4. Train, Validation, and Test Partitioning Contract

Partitions are specified via `DatasetSplit` containing:
- `train_layout_ids`: List of layout IDs strictly assigned to training.
- `val_layout_ids`: List of layout IDs used for early stopping and hyperparameter tuning.
- `test_layout_ids`: List of layout IDs held out for final benchmark evaluation.

### Partitioning Protocols:
- **Standard Protocol (Ratio Holdout):** 70% Train, 15% Validation, 15% Test, stratified across plan area density bins.
- **Generalization Protocol (Family Holdout):** Train on `idealized_aligned` and `idealized_staggered` arrays; hold out `realistic_neighborhood` in test set to evaluate model generalizability to real-world complex geometry.

---

## 5. File System & Storage Standard

Individual layout samples are saved as compressed NumPy archives (`.npz`) containing:
```
sample_<layout_id>.npz
├── height_map                (float32, [Ny, Nx])
├── building_mask              (bool,    [Ny, Nx])
├── pedestrian_wind_speed     (float32, [Ny, Nx])
├── velocity_u (optional)     (float32, [Ny, Nx])
├── velocity_v (optional)     (float32, [Ny, Nx])
├── turbulent_kinetic_energy  (float32, [Ny, Nx])
└── _metadata_json            (string JSON containing metadata, environment, CFD solver, and scalar targets)
```

Fitted normalizer statistics and split assignments are persisted to `data/preprocessing_config.json`.
