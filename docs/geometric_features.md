# Geometric Features Reference

**Branch:** `feature/geometric-features`  
**Author:** Team Member B — Geometric Features & ML Baseline  
**Date:** 2026-10-26  
**Status:** Implemented & Tested  

---

## 1. Overview

This document defines every geometric and environmental feature extracted by
`GeometricFeatureExtractor` in [`src/data/features.py`](../src/data/features.py).

Features are computed exclusively from fields available **at prediction time**:

| Source | Fields used |
|---|---|
| `GeometryData` | `height_map` (Ny, Nx) float32 [m], `building_mask` (Ny, Nx) bool |
| `LayoutMetadata` | `resolution_m` (dx, dy, dz) [m], `domain_size_m` (Lx, Ly, Lz) [m] |
| `EnvironmentalConditions` | `inflow_speed_mps`, `wind_direction_deg`, `reference_height_m`, `roughness_length_z0_m` |

**No target fields** (`pedestrian_wind_speed`, `area_averaged_wind_speed`, `turbulent_kinetic_energy`,
velocity components) are used in any feature formula.

The `target_area_averaged_wind_speed` scalar is stored in `GeometricFeatures` as a
**metadata field only** and is explicitly excluded from `to_feature_vector()` and
`feature_names()`. The `layout_id` string is likewise excluded.

---

## 2. Unresolved Dependencies & Data Status

> [!IMPORTANT]
> **No real CFD data is available locally.** All features have been implemented and
> tested against synthetic test fixtures (`tests/fixtures/synthetic_data.py`).
> Physical plausibility validation against the UrbanTALES dataset cannot be performed
> until a curated set of layouts is downloaded (see `docs/dataset_audit.md`).

> [!WARNING]
> **Building count via connected components:** `building_count` and footprint
> statistics are computed using `scipy.ndimage.label` applied to `building_mask`.
> This treats each **contiguous group of True cells** as one building. This
> assumption is valid for idealized rectangular building arrays (UrbanTALES idealized
> family) but may over-count L-shaped or irregular polygon edges as separate objects
> if rasterization introduces gaps. Validate this assumption visually once real data
> is available.

---

## 3. Feature Reference Table

### Group 1 — Existing Features (11 features, unchanged)

| Feature Name | Formula | Unit | Edge Case |
|---|---|---|---|
| `plan_area_density_lambda_p` | $\lambda_p = N_\text{built} / N_\text{total}$ | — | 0.0 if no buildings |
| `frontal_area_density_lambda_f` | Wind-projected step increases in `height_map` / total area; see §4.1 | — | 0.0 if no buildings |
| `mean_building_height_m` | $\bar{H} = \text{mean}(\text{height\_map}[\text{mask}])$ | m | 0.0 if no buildings |
| `std_building_height_m` | $\sigma_H = \text{std}(\text{height\_map}[\text{mask}])$ | m | 0.0 if no buildings |
| `max_building_height_m` | $H_\text{max} = \text{max}(\text{height\_map}[\text{mask}])$ | m | 0.0 if no buildings |
| `wall_area_density_lambda_w` | Total absolute height-step facade area / total plan area; see §4.2 | — | 0.0 if no buildings |
| `roughness_length_macdonald_m` | MacDonald et al. (1998) $z_0$ parameterisation; see §4.3 | m | Falls back to `roughness_length_z0_m` if $\lambda_f$ or $\bar{H}$ ≈ 0 |
| `displacement_height_d_m` | MacDonald et al. (1998) $d$ parameterisation; see §4.3 | m | 0.0 if no buildings |
| `inflow_speed_mps` | $U_\text{ref}$ — reference approach wind speed | m/s | Must be > 0 (contract-enforced) |
| `wind_direction_cos` | $\cos(\delta)$, $\delta$ = `wind_direction_deg` in radians | — | Always in [−1, 1] |
| `wind_direction_sin` | $\sin(\delta)$ | — | Always in [−1, 1] |

---

### Group 2 — Height Distribution Quantiles (5 new features)

| Feature Name | Formula | Unit | Edge Case |
|---|---|---|---|
| `min_building_height_m` | $H_\text{min} = \text{min}(\text{height\_map}[\text{mask}])$ | m | 0.0 if no buildings |
| `p25_building_height_m` | $H_{25} = \text{percentile}(\text{height\_map}[\text{mask}],\ 25)$ | m | 0.0 if no buildings |
| `p75_building_height_m` | $H_{75} = \text{percentile}(\text{height\_map}[\text{mask}],\ 75)$ | m | 0.0 if no buildings |
| `height_iqr_m` | $\text{IQR} = H_{75} - H_{25}$ | m | 0.0 if no buildings or uniform height |
| `height_cv` | $C_V = \sigma_H / \bar{H}$ | — | 0.0 if $\bar{H} \le 10^{-9}$ m (guarded against division by zero) |

**Physical interpretation:**  
`height_iqr_m` and `height_cv` quantify **roof-level heterogeneity**.
Uniform-height arrays (IQR = 0) support stable skimming flow regimes.
Heterogeneous heights (large IQR / CV) promote turbulent vertical momentum
exchange between the canopy and the free stream, increasing pedestrian-level
ventilation. These features cannot resolve *which* specific tall buildings
generate local wake acceleration — that requires the full spatial field.

---

### Group 3 — Building Count & Footprint Statistics (5 new features)

Computed by `scipy.ndimage.label(building_mask)` with default 2D connectivity
(4-connected structuring element).

| Feature Name | Formula | Unit | Edge Case |
|---|---|---|---|
| `building_count` | $N_b$ = number of connected components of `building_mask` | — (stored as float) | 0.0 if no buildings |
| `mean_footprint_area_m2` | $\bar{A} = \text{mean}(\{A_k\}) \cdot \Delta x \cdot \Delta y$ | m² | 0.0 if no buildings |
| `std_footprint_area_m2` | $\sigma_A = \text{std}(\{A_k\}) \cdot \Delta x \cdot \Delta y$ | m² | 0.0 if ≤ 1 building |
| `max_footprint_area_m2` | $A_\text{max} = \text{max}(\{A_k\}) \cdot \Delta x \cdot \Delta y$ | m² | 0.0 if no buildings |
| `footprint_area_cv` | $C_{V,A} = \sigma_A / \bar{A}$ | — | 0.0 if ≤ 1 building or $\bar{A} \le 10^{-9}$ |

where $A_k$ = number of raster cells in connected component $k$.

**Physical interpretation:**  
Building count and footprint spread describe **macro-scale layout diversity**.
A layout with many small buildings and one very large anchor (high $C_{V,A}$)
will produce different bulk drag and wake interference patterns than a uniform
grid. These features cannot capture the *location* or *orientation* of specific
wake interactions.

---

### Group 4 — Layout Extent & Domain Summary (3 new features)

| Feature Name | Formula | Unit | Edge Case |
|---|---|---|---|
| `domain_aspect_ratio` | $\text{AR} = L_x / L_y$ | — | 1.0 if $L_y \le 10^{-9}$ m (guarded) |
| `built_area_total_m2` | $A_\text{built} = N_\text{built} \cdot \Delta x \cdot \Delta y$ | m² | 0.0 if no buildings |
| `mean_height_times_lambda_p` | $\bar{H} \cdot \lambda_p$ | m | 0.0 if no buildings; proportional to canopy volume fraction |

**Physical interpretation:**  
`domain_aspect_ratio` encodes whether the computational domain is elongated
streamwise or spanwise — relevant for channel geometry effects on bulk flow
resistance. `mean_height_times_lambda_p` is a surrogate for the canopy volume
fraction and correlates with displacement height magnitude.

---

### Group 5 — Environmental Boundary Inputs (2 new features)

| Feature Name | Source | Unit | Edge Case |
|---|---|---|---|
| `reference_height_m` | `environment.reference_height_m` | m | Must be > 0 (contract-enforced) |
| `roughness_z0_env_m` | `environment.roughness_length_z0_m` | m | Must be > 0 (contract-enforced) |

**Distinction from `roughness_length_macdonald_m`:**  

| Feature | Physical meaning |
|---|---|
| `roughness_z0_env_m` | Terrain surface roughness of the underlying ground; an **input boundary condition** to the CFD solver (e.g. grass: 0.01 m, suburban: 0.05 m, urban: 0.1–0.5 m). |
| `roughness_length_macdonald_m` | Effective aerodynamic roughness of the **building canopy**, derived from the MacDonald (1998) parameterisation using $\lambda_f$, $\bar{H}$, and $d$. This is a computed output representing the bulk canopy drag seen by the above-roof flow. |

Both features are physically non-redundant and are available at prediction time
from `EnvironmentalConditions`, which does not contain target fields.

---

## 4. Exact Formulae

### 4.1 Frontal Area Density $\lambda_f$

Wind direction $\delta$ (degrees, 0° = west→east +x, 90° = south→north +y):

$$
\lambda_f = \frac{|\cos\delta| \cdot A_{f,x} + |\sin\delta| \cdot A_{f,y}}{A_\text{total}}
$$

where

$$
A_{f,x} = \Delta y \sum_{i,j} \max(0,\ H_{i,j+1} - H_{i,j}) + \Delta y \sum_i H_{i,0}
$$

$$
A_{f,y} = \Delta x \sum_{i,j} \max(0,\ H_{i+1,j} - H_{i,j}) + \Delta x \sum_j H_{0,j}
$$

The left-column and bottom-row boundary terms account for buildings that reach the
inlet face of the domain. Units: m² (numerator), m² (denominator) → dimensionless.

### 4.2 Wall Area Density $\lambda_w$

$$
\lambda_w = \frac{A_\text{facade}}{A_\text{total}}
$$

$$
A_\text{facade} = \Delta y \sum_{i,j} |H_{i,j+1} - H_{i,j}|
+ \Delta y \sum_i (H_{i,0} + H_{i,N_x-1})
+ \Delta x \sum_{i,j} |H_{i+1,j} - H_{i,j}|
+ \Delta x \sum_j (H_{0,j} + H_{N_y-1,j})
$$

All height steps (in absolute value) contribute to total vertical facade area.

### 4.3 MacDonald et al. (1998) Canopy Parameterisation

**Displacement height** $d$:

$$
\frac{d}{\bar{H}} = 1 + \alpha^{-\lambda_p}(\lambda_p - 1), \quad \alpha = 4.0
$$

Clamped: $d \in [0,\ 0.95\bar{H}]$.

**Aerodynamic roughness length** $z_0$ (MacDonald 1998, Eq. 11):

$$
\frac{z_0}{\bar{H}} = \left(1 - \frac{d}{\bar{H}}\right) \exp\!\left(-\left(\frac{1}{2}\frac{\beta C_D}{\kappa^2}\left(1-\frac{d}{\bar{H}}\right)\lambda_f\right)^{-1/2}\right)
$$

with $\beta = 1.0$, $C_D = 1.2$ (bluff body), $\kappa = 0.40$.

**Fallback:** if $\lambda_f \le 10^{-4}$, $\bar{H} \le 0$, or the inner term $\le 10^{-4}$,
$z_0$ falls back to `environment.roughness_length_z0_m` (terrain surface roughness).

---

## 5. Missing-Value Handling Summary

| Condition | Affected features | Behaviour |
|---|---|---|
| No buildings (`mask` all False) | All building-derived features | Set to 0.0 |
| Single building (`building_count` = 1) | `std_footprint_area_m2`, `footprint_area_cv` | 0.0 |
| Uniform height ($\sigma_H = 0$) | `height_cv`, `height_iqr_m` | 0.0 |
| Zero mean height ($\bar{H} = 0$) | `height_cv`, `displacement_height_d_m`, `roughness_length_macdonald_m` | 0.0 (CV), 0.0 (d), fallback to `roughness_length_z0_m` |
| Zero frontal area ($\lambda_f \le 10^{-4}$) | `roughness_length_macdonald_m` | Falls back to `roughness_length_z0_m` |
| Square domain ($L_x = L_y$) | `domain_aspect_ratio` | 1.0 |
| Degenerate domain ($L_y \approx 0$) | `domain_aspect_ratio` | 1.0 (guarded) |

---

## 6. Target Leakage Audit

| Check | Status |
|---|---|
| No target field in formula inputs | ✅ All features use only `GeometryData` + `EnvironmentalConditions` + `LayoutMetadata` |
| `target_area_averaged_wind_speed` excluded from `feature_names()` | ✅ Confirmed by `test_target_not_in_feature_names` |
| `layout_id` excluded from `feature_names()` | ✅ Confirmed by `test_layout_id_not_in_feature_names` |
| `layout_family` excluded from feature vector | ✅ Not computed or stored numerically |
| `pedestrian_wind_speed` field not accessed in extract() | ✅ Only accessed via `area_averaging.compute_area_averaged_wind_speed` when target scalar is None (which itself sums over fluid cells — not available at inference without the full CFD result) |

> [!CAUTION]
> The fallback path in `extract()` calls `compute_area_averaged_wind_speed`
> from `targets.pedestrian_wind_speed` only when `targets.area_averaged_wind_speed`
> is `None`. This fallback **must never be triggered** at inference time — the
> `area_averaged_wind_speed` must be pre-computed and stored in the NPZ archive.
> The fallback exists solely to handle legacy or partially-loaded samples during
> data preparation, and produces the **training-time target scalar**, not a feature.

---

## 7. Deferred Features

| Feature | Reason for Deferral |
|---|---|
| Street / canyon orientation histogram | Requires vector street-network geometry (polygon centroids, street graph). Not present in `height_map + building_mask` raster schema. |
| Building-to-building spacing / gap statistics | Defensible via Euclidean distance transform on labeled components, but adds complexity for marginal gain at raster resolution ≥ 0.5 m. Deferred to future iteration once schema includes per-building polygon IDs. |
| Sky View Factor (SVF) | Requires hemisphere integration / 3D ray-casting from `height_map`. Outside scope of tabular ML feature set. |
| Reynolds number $Re = U_\text{ref} \bar{H} / \nu$ | Collinear with `inflow_speed_mps × mean_building_height_m`. Excluded to avoid multicollinearity. |
| Per-building aspect ratio $H/W$ | Requires per-building footprint width measurement from polygon geometry. Not computable from raster mask alone. |
| Temperature-derived features | UrbanTALES is isothermal; no thermal data available (see `docs/dataset_audit.md`). |

---

## 8. Scale Interpretation

### What these features capture (large-scale / canopy-averaged effects)

- **$\lambda_p$:** Controls flow regime — isolated roughness ($\lambda_p < 0.10$),
  wake interference ($0.10 \le \lambda_p \le 0.25$), skimming flow ($\lambda_p > 0.25$).
- **$\lambda_f$:** Controls bulk aerodynamic drag and momentum extraction from the
  approaching flow.
- **Height statistics ($\bar{H}$, $\sigma_H$, IQR, CV):** Determine displacement height
  $d$ and aerodynamic roughness $z_0$, which set the log-law profile shape above
  the canopy and thus the approach flow momentum available at pedestrian level.
- **$U_\text{ref}$, $\cos\delta$, $\sin\delta$:** Set the kinetic energy and direction
  of the approach flow — first-order determinants of canopy-averaged wind speed.
- **Building count / footprint:** Encode macro-layout diversity and packing
  regularity at the neighbourhood scale.
- **$L_x/L_y$:** Captures whether the domain is streamwise- or spanwise-elongated.

### What these features CANNOT capture (local wake / canyon effects)

These are the precisely the phenomena requiring the **neural surrogate** (U-Net):

| Physical effect | Why bulk features cannot resolve it |
|---|---|
| **Corner acceleration jets** | Purely local; depends on exact windward building edge geometry |
| **Wake cavity length** ($4$–$6H$) | Specific to each building's dimensions and upstream spacing |
| **Horseshoe vortex system** | Ground-level vortex structure at individual building base |
| **Street canyon channelling / Venturi effect** | Requires precise gap width between adjacent specific buildings |
| **Recirculation asymmetry** | Caused by irregular arrangements not captured by density averages |
| **Local TKE peak at roof edges** | Point-specific shear layer — invisible to layout-averaged features |

> [!NOTE]
> Documenting these limitations explicitly is not a weakness — it is the
> scientific rationale for building the neural surrogate alongside the ML baseline.
> The baseline quantifies what bulk morphometrics can predict; the surrogate
> demonstrates what spatial context additionally enables.

---

## 9. Feature Vector Specification

**Total features in `to_feature_vector()`:** 26  
**Feature names (ordered):** see `GeometricFeatures.feature_names()` in [`src/data/features.py`](../src/data/features.py)

```
Index  Name                           Group   Unit
  0    plan_area_density_lambda_p     1       -
  1    frontal_area_density_lambda_f  1       -
  2    mean_building_height_m         1       m
  3    std_building_height_m          1       m
  4    max_building_height_m          1       m
  5    wall_area_density_lambda_w     1       -
  6    roughness_length_macdonald_m   1       m
  7    displacement_height_d_m        1       m
  8    inflow_speed_mps               1       m/s
  9    wind_direction_cos             1       -
 10    wind_direction_sin             1       -
 11    min_building_height_m          2       m
 12    p25_building_height_m          2       m
 13    p75_building_height_m          2       m
 14    height_iqr_m                   2       m
 15    height_cv                      2       -
 16    building_count                 3       -
 17    mean_footprint_area_m2         3       m²
 18    std_footprint_area_m2          3       m²
 19    max_footprint_area_m2          3       m²
 20    footprint_area_cv              3       -
 21    domain_aspect_ratio            4       -
 22    built_area_total_m2            4       m²
 23    mean_height_times_lambda_p     4       m
 24    reference_height_m             5       m
 25    roughness_z0_env_m             5       m
```

---

## 10. References

- MacDonald, R. W., Griffiths, R. F., & Hall, D. J. (1998). An improved method for the estimation of surface roughness of obstacle arrays. *Atmospheric Environment*, 32(11), 1857–1864.
- Nazarian, N., van Reeuwijk, M., Lu, J., et al. (2025). UrbanTALES: Urban Turbulence Analyses from Large-Eddy Simulations. *Bulletin of the American Meteorological Society*. DOI: 10.1175/BAMS-D-25-0061.1
- Grimmond, C. S. B., & Oke, T. R. (1999). Aerodynamic properties of urban areas derived from analysis of surface form. *Journal of Applied Meteorology*, 38(9), 1262–1292.
