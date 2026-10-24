# Urban Microclimate Modelling: Data Validation & Preprocessing Hardening

**Author / Role:** Team Member A — Dataset Audit & Data Validation (@Aditya)  
**Date:** October 24, 2026  
**Branch:** `feature/data-validation-hardening`  
**Document Version:** 1.0.0  
**Status:** Implemented & Verified in Test Suite (92/92 tests passing)  

---

## 1. Overview & Objective

To prevent silent failures, corrupted gradients, and deceptive scientific results in microclimate neural surrogates, all incoming simulation data and preprocessed tensors must pass strict, multi-tiered validation barriers before model ingestion.

This document details the **hardened validation and leakage-safe preprocessing pipeline** implemented across [`src/data/contract.py`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py), [`src/data/normalization.py`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/normalization.py), [`src/data/rasterizer.py`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/rasterizer.py), and [`src/data/loader.py`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/loader.py).

```
 Incoming Raw CFD / NPZ Archives
                │
                ▼
 ┌──────────────────────────────────────────────┐
 │  Pre-flight Archive Validation                │
 │  LayoutDataLoader.validate_npz_archive()     │
 └──────────────────────┬───────────────────────┘
                        │ Pass
                        ▼
 ┌──────────────────────────────────────────────┐
 │  Data Contract & Physical Invariants         │
 │  UrbanLayoutSample.validate()                │
 │  - Shape & Coordinate Grid Consistency      │
 │  - Finite Real Values (NaN/Inf Detection)    │
 │  - SI Units & Boundary Condition Sanity      │
 └──────────────────────┬───────────────────────┘
                        │ Pass
                        ▼
 ┌──────────────────────────────────────────────┐
 │  Spatial Alignment & Rasterization            │
 │  LayoutRasterizer.rasterize()                │
 │  verify_spatial_alignment()                  │
 │  - Multi-channel Tensor Grid Alignment       │
 │  - Solid Mask Target Interior Zeroing        │
 │  - Monotonic Normalized Coordinates [0, 1]   │
 └──────────────────────┬───────────────────────┘
                        │ Pass
                        ▼
 ┌──────────────────────────────────────────────┐
 │  Leakage-Free Normalization                  │
 │  FieldNormalizer / FeatureNormalizer         │
 │  - Fitted strictly on Training Layout IDs    │
 │  - Runtime check: verify_no_leakage()        │
 └──────────────────────────────────────────────┘
```

---

## 2. Actionable Exception Hierarchy

Generic Python exceptions (`KeyError`, `ValueError`) obscure root causes during large-scale dataset ingestion. We enforce a dedicated, actionable exception hierarchy rooted at [`DataValidationError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L29-L31):

| Exception Class | Base Class | Trigger Condition | Diagnostic Payload |
|---|---|---|---|
| [`DataValidationError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L29-L31) | `ValueError` | Base class for all contract violations | Human-readable explanation and corrective guidance |
| [`MissingFieldValidationError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L34-L37) | `DataValidationError`, `KeyError` | Missing metadata section, empty layout ID, or missing array keys | Exact missing key names and archive path |
| [`ShapeMismatchValidationError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L39-L42) | `DataValidationError` | 2D/3D tensor shape differs from `(Ny, Nx)` in metadata | Received shape vs expected `grid_shape` |
| [`NonFiniteValueError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L44-L47) | `DataValidationError` | Floating-point `NaN` or `±Inf` encountered in arrays | Exact counts of NaNs and Infs per field |
| [`PhysicalConstraintValidationError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L49-L52) | `DataValidationError` | Negative building height, negative wind speed, wind angle $\notin [0, 360^\circ)$, or mask/height mismatch | Minimum/maximum offending values and cell counts |
| [`CoordinateGridValidationError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L54-L57) | `DataValidationError` | Domain dimensions $(L_x, L_y)$ inconsistent with $(N_x \cdot dx, N_y \cdot dy)$ | Theoretical domain bounds vs declared size |
| [`IncompatibleUnitsError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L59-L62) | `DataValidationError` | Non-SI or unconvertible units encountered | Declared unit vs required SI unit |
| [`DataLeakageError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/normalization.py#L9-L12) | `RuntimeError` | Overlap detected between training fitted layout IDs and holdout layout IDs | List of overlapping layout IDs |

---

## 3. Data Validation Rules & Invariants

### 3.1 Metadata & Coordinate System
- **Layout ID:** Must be a non-empty string identifier (e.g., `"idealized_aligned_001"`, `"london_marylebone_012"`).
- **Layout Family:** Must be a valid member of `LayoutFamily`:
  - `IDEALIZED_ALIGNED`
  - `IDEALIZED_STAGGERED`
  - `REALISTIC_NEIGHBORHOOD`
- **Domain vs Resolution Consistency:**
  The physical domain $(L_x, L_y)$ and grid cell resolution $(dx, dy)$ must satisfy:
  $$|L_x - N_x \cdot dx| \le 0.05 \cdot L_x \quad \text{and} \quad |L_y - N_y \cdot dy| \le 0.05 \cdot L_y$$
  Any discrepancy exceeding 5% raises a [`CoordinateGridValidationError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L54-L57).

### 3.2 Geometry & Physical Constraints
- **Building Height Map (`height_map`):**
  - Dimensions: `(Ny, Nx)` float32 in meters.
  - Non-negativity: $h(y, x) \ge 0.0\text{ m}$. Negative values raise [`PhysicalConstraintValidationError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L49-L52).
  - Finiteness: All values must be real; `NaN` or `Inf` raises [`NonFiniteValueError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L44-L47).
- **Building Mask (`building_mask`):**
  - Dimensions: `(Ny, Nx)` boolean (`True` for solid building, `False` for fluid domain).
  - Mask Consistency: Where $h(y, x) > 10^{-4}\text{ m}$, the mask **must** be `True`. If $h > 0$ while `mask` is `False`, [`PhysicalConstraintValidationError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L49-L52) is raised.

### 3.3 Target Quantities & Building Interior Zeroing
- **Pedestrian Wind Speed (`pedestrian_wind_speed`):**
  - Magnitude of horizontal or 3D velocity at pedestrian height $z = 1.75\text{ m}$ (in $\text{m/s}$).
  - Non-negativity: Scalar wind speed cannot be negative ($s \ge -10^{-4}\text{ m/s}$).
  - Solid Interior Zeroing: Fluid flow inside buildings is physically nonexistent. To prevent neural networks from learning arbitrary artifacts within solids, target fields inside building footprints ($M = \text{True}$) are explicitly set to $0.0\text{ m/s}$.

### 3.4 CFD Solver Metadata
- **Solver Identification:** `solver_name` (e.g. `"uDALES"`, `"OpenFOAM"`) and `turbulence_model` (e.g. `"LES-Deardorff"`, `"RANS-k-epsilon"`) must be specified and non-empty. Empty fields raise [`MissingFieldValidationError`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/contract.py#L34-L37).

---

## 4. Leakage-Safe Preprocessing & Normalization

### 4.1 The Threat of Spatial Autocorrelation Leakage
In urban fluid mechanics, adjacent grid cells within the same urban layout exhibit high spatial autocorrelation. Performing random pixel-level or patch-level train/test splits would leak spatial boundary conditions and wake structures directly into the evaluation set.

**Strict Mandate:** All splits must be partitioned strictly by **Layout ID** or by entire **Layout Family**.

### 4.2 Normalizer Tracking & Verification
Both [`FieldNormalizer`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/normalization.py#L14) and [`FeatureNormalizer`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/normalization.py#L228) have been hardened with:
1. **Fitted Layout Tracking (`fitted_layout_ids_`):** When `.fit()` is called on training samples, the unique layout identifiers of the training set are recorded.
2. **Leakage Assertion (`verify_no_leakage(test_layout_ids)`):**
   ```python
   overlap = set(self.fitted_layout_ids_).intersection(set(test_layout_ids))
   if overlap:
       raise DataLeakageError(f"Data leakage detected! Normalizer fitted on {sorted(list(overlap))}")
   ```
3. **Serialization Integrity:** `to_dict()` and `from_dict()` preserve `fitted_layout_ids_`, ensuring that checkpoints and saved normalizer artifacts carry their provenance and can be verified across training pipelines.
4. **Fluid Masking in Statistics:** `FieldNormalizer` masks out solid building cells ($M = \text{True}$) during statistics calculation so that zeroed interiors do not bias the computed fluid mean and standard deviation.

---

## 5. Spatial Tensor Alignment Verification

The [`verify_spatial_alignment()`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/rasterizer.py#L159-L265) utility provides an exhaustive post-rasterization check:

```python
from src.data.rasterizer import LayoutRasterizer, verify_spatial_alignment

rasterizer = LayoutRasterizer(zero_inside_buildings=True)
rasterized = rasterizer.rasterize(sample)

# Verifies tensor shape, mask identity, finite values, coordinate bounds, and interior zeros
verify_spatial_alignment(rasterized, sample, check_zero_interiors=True)
```

Checks executed by `verify_spatial_alignment()`:
1. **Layout ID Identity:** Matches between sample metadata and rasterized output.
2. **Rank & Dimensions:** `channels` is `(C_in, Ny, Nx)`, `target_field` is `(C_out, Ny, Nx)`, and `building_mask` is `(Ny, Nx)`.
3. **Array Equality:** Rasterized `building_mask` exactly matches `sample.geometry.building_mask`.
4. **NaN/Inf Screening:** Both inputs and targets are verified to contain zero non-finite values.
5. **Coordinate Channel Bounds:** Normalized spatial coordinates (`coord_x_norm`, `coord_y_norm`) are strictly bounded in $[0.0, 1.0]$ and monotonic.
6. **Building Interior Zeroing:** When `check_zero_interiors=True`, verifies that target velocity magnitudes within solid building cells are $\le 10^{-6}\text{ m/s}$.

---

## 6. Pre-flight NPZ Archive Validation

Before loading multi-megabyte simulation arrays into memory, pipelines can execute fast header and structure inspection using [`LayoutDataLoader.validate_npz_archive()`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/src/data/loader.py#L23-L74):

```python
from src.data.loader import LayoutDataLoader

is_valid, issues = LayoutDataLoader.validate_npz_archive("data/sample_001.npz")
if not is_valid:
    print(f"Skipping corrupt archive: {issues}")
```

This inspection verifies:
- File existence and readability.
- Presence of `_metadata_json` with valid JSON containing required sections (`metadata`, `environment`, `cfd_solver`).
- Presence of required spatial arrays (`height_map`, `building_mask`, `pedestrian_wind_speed`).
- Matching array shapes against declared `grid_shape`.
- Absence of `NaN` or `Inf` values in floating-point fields.

---

## 7. Developer Instructions & Test Execution

### 7.1 Running the Hardened Validation Test Suite
To run all 19 dedicated validation hardening tests:
```bash
.venv/bin/pytest tests/test_validation_hardening.py -v
```

To run the complete repository test suite:
```bash
.venv/bin/pytest -v
```

### 7.2 Testing vs Real Simulation Policy
- **Synthetic Test Fixtures:** Files created by [`tests/fixtures/synthetic_data.py`](file:///Users/omkarshukla/Urban-Microclimate-Modelling/tests/fixtures/synthetic_data.py) are small, deterministic test arrays. They are strictly for automated testing and **must never be cited or used as physical CFD validation**.
- **Real Datasets (UrbanTALES):** When external `.npz` slices are ingested, they must pass `validate_npz_archive()` and `UrbanLayoutSample.validate()` before entry into training splits.
