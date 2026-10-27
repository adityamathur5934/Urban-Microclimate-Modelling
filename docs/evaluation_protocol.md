# Evaluation Protocol: Layout-Aware Splitting & Spatial Microclimate Metrics

**Project:** Urban Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Role:** Machine Learning Evaluation Engineer (Team Member C)  
**Document Version:** 1.0.0  
**Branch:** `feature/evaluation-metrics`  
**Status:** Approved Reference Protocol  

---

## 1. Executive Summary & Purpose

Evaluating machine learning models and deep neural surrogates on physical continuum simulations (such as CFD and atmospheric boundary layer flows) requires fundamentally different protocols than standard tabular or image classification tasks. Due to spatial autocorrelation induced by elliptic Navier-Stokes pressure equations, random pixel-wise or cell-wise data splitting causes severe data leakage, resulting in artificially near-zero test errors that fail to generalize to real urban geometries.

This document formalizes:
1. **Deterministic, layout-aware and family-aware dataset partitioning** with strict non-leakage guarantees.
2. **Standardized quantitative regression metrics** evaluated exclusively over outdoor fluid cells with explicit units and equations.
3. **Transparent, non-ambiguous denominators for Normalized Root Mean Squared Error (NRMSE)** across aerodynamic and statistical conventions.
4. **Invariant enforcement**, including spatial grid alignment, mask semantics, non-finite value rejection, and empty-mask safeguards.
5. **A clear delineation between software smoke fixtures and scientific accuracy claims**.

---

## 2. Deterministic Layout & Family Partitioning Protocol

### 2.1 The Spatial Autocorrelation Leakage Problem
In CFD simulations, pressure and momentum are governed by elliptic partial differential equations. Adjacent cells in the same flow domain share boundary layer history, recirculating vortices, and upstream wake structures. 

If pixels from the same simulation domain are randomly assigned across training and testing sets:
- The model memorizes neighboring fluid cell solutions rather than learning generalized Navier-Stokes flow physics.
- Test metrics exhibit severe optimistic bias (often reporting $R^2 > 0.99$ on train-test splits of the same layout).
- When evaluated on a completely unseen layout, performance degrades catastrophically.

**Rule:** Partitions must be defined strictly at the **layout identifier (`layout_id`)** or **layout family (`layout_family`)** level. Individual pixels, grid cells, or localized sub-blocks of a single layout must **never** be divided across splits.

```
+-----------------------------------------------------------------------------------------+
|                                FULL DATASET OF URBAN LAYOUTS                            |
+-----------------------------------------------------------------------------------------+
                                             |
             +-------------------------------+-------------------------------+
             |                               |                               |
             v                               v                               v
    [TRAINING PARTITION]          [VALIDATION PARTITION]             [TEST PARTITION]
       70% of Layouts                 15% of Layouts                  15% of Layouts
     (No Shared Layouts)            (Model Selection Only)         (Zero-Leakage Benchmark)
```

### 2.2 Partitioning Strategies Supported in `src.data.split`

The `LayoutSplitter` module implements three deterministic splitting strategies:

1. **Ratio Holdout (`split_by_ratio`):**
   - Partitions unique layout IDs into training (default $70\%$), validation ($15\%$), and testing ($15\%$) sets.
   - If a `group_by` mapping is supplied (e.g., grouping multi-angle wind runs of the same base geometry), entire groups remain indivisible.
2. **Family Holdout (`split_by_family`):**
   - Holds out an entire morphological family (e.g. `realistic_neighborhood`) exclusively for testing, while training on canonical arrays (`idealized_aligned`, `idealized_staggered`).
   - Evaluates true out-of-distribution (OOD) architectural generalization from canonical building matrices to real-world urban topographies.
3. **Group Holdout (`split_by_group`):**
   - Groups related runs (e.g., parametric height sweeps, grid resolution variations, or inflow speed variations) by group key.
   - Guarantees all layouts belonging to a group remain in the exact same partition.

### 2.3 Determinism and Reproducibility Guarantees
All splitting routines in `LayoutSplitter`:
- Extract unique identifiers and sort them alphabetically (`sorted(unique_ids)`) prior to shuffling.
- Use an isolated pseudo-random number generator instance `random.Random(seed)` initialized with a configurable integer seed (default `seed = 42`).
- Never rely on non-deterministic hash iterations or global random state.
- Guarantee identical partition assignments across operating systems and execution environments.

### 2.4 Metadata Tracking and JSON Serialization
Every `DatasetSplit` instance includes partition metadata and provides export/import methods:
- `split.save_json(output_path)`: Exports layout assignments, split strategy, seed, and timestamps to formatted JSON.
- `DatasetSplit.load_json(input_path)`: Loads and immediately validates the split against leakage.

```json
{
  "train_layout_ids": ["synth_layout_000", "synth_layout_001"],
  "val_layout_ids": ["synth_layout_002"],
  "test_layout_ids": ["synth_layout_003"],
  "metadata": {
    "split_strategy": "ratio",
    "seed": 42,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "created_at": "2026-10-27T18:00:00Z"
  }
}
```

### 2.5 Explicit Non-Leakage Assertions
`DatasetSplit.assert_no_overlap(check_families=True)` verifies:
- $\text{Train} \cap \text{Val} = \emptyset$
- $\text{Train} \cap \text{Test} = \emptyset$
- $\text{Val} \cap \text{Test} = \emptyset$
- If `layout_to_family` or group mapping is provided under family/group holdout, no family or group identifier crosses split boundaries.
- Any violation immediately raises `ValueError("Data leakage detected! ...")`.

---

## 3. Quantitative Regression Metric Formulations

All spatial metrics are evaluated **strictly over the outdoor fluid domain** ($\Omega_{\text{fluid}}$), where $M_{\text{building}}(y, x) = 0$. Solid building interiors are masked out before metric computation to prevent zero-padding or interior default values from distorting errors.

Let $N_{\text{fluid}} = |\Omega_{\text{fluid}}|$ be the number of active outdoor cells, $y_i$ be the CFD ground truth velocity magnitude at cell $i$, and $\hat{y}_i$ be the predicted velocity magnitude.

### 3.1 Field Mean Absolute Error (MAE)
Measures the average magnitude of absolute velocity errors across pedestrian fluid cells.

$$\text{MAE} = \frac{1}{N_{\text{fluid}}} \sum_{i=1}^{N_{\text{fluid}}} |y_i - \hat{y}_i|$$

- **Target Units:** Meters per second [$\text{m/s}$].
- **Function:** `compute_field_mae(y_true, y_pred, mask=building_mask)` (alias `field_mae`).

### 3.2 Field Root Mean Squared Error (RMSE)
Applies a quadratic penalty to deviations, making it sensitive to localized high-velocity discrepancies (such as missed corner vortex jets or boundary-layer separation zones).

$$\text{RMSE} = \sqrt{\frac{1}{N_{\text{fluid}}} \sum_{i=1}^{N_{\text{fluid}}} (y_i - \hat{y}_i)^2}$$

- **Target Units:** Meters per second [$\text{m/s}$].
- **Function:** `compute_field_rmse(y_true, y_pred, mask=building_mask)` (alias `field_rmse`).

### 3.3 Normalized Root Mean Squared Error (NRMSE) Conventions
Different sub-disciplines (wind engineering, statistical ML, climate physics) normalize RMSE by different denominators. **Our evaluation engine never silently defaults to an arbitrary denominator.** The denominator is explicitly documented and selectable via `normalization`:

| Normalization Key | Denominator ($D$) | Formula | Typical Field of Use | Output Units |
|---|---|---|---|---|
| `normalization="range"` | Target Dynamic Range | $D = \max(y_{\text{true}}) - \min(y_{\text{true}})$ | Microclimate and general ML benchmarks | Ratio $[0, 1]$ or $[\%]$ |
| `normalization="inflow"` | Approach Inflow Speed | $D = U_{\text{ref}} > 0$ | Wind Engineering & Bluff-Body Aerodynamics | Ratio $[0, 1]$ or $[\%]$ |
| `normalization="std"` | Target Standard Deviation | $D = \sigma_{y_{\text{true}}}$ | Statistical modeling ($\approx \sqrt{1 - R^2}$) | Ratio $[0, 1]$ or $[\%]$ |
| `normalization="mean"` | Target Absolute Mean | $D = \|\bar{y}_{\text{true}}\|$ | Sensor validation & tabular metrics | Ratio $[0, 1]$ or $[\%]$ |

- **Equations:**
  - $\text{NRMSE}_{\text{range}} = \frac{\text{RMSE}}{\max_{i \in \text{fluid}}(y_i) - \min_{i \in \text{fluid}}(y_i)}$
  - $\text{NRMSE}_{\text{inflow}} = \frac{\text{RMSE}}{U_{\text{ref}}}$
  - $\text{NRMSE}_{\text{std}} = \frac{\text{RMSE}}{\sqrt{\frac{1}{N_{\text{fluid}}} \sum_{i=1}^{N_{\text{fluid}}} (y_i - \bar{y})^2}}$
- **Function:** `compute_normalized_rmse(y_true, y_pred, mask=building_mask, normalization="range", u_ref=10.0, as_percent=True)` (alias `field_nrmse`).
- **Safeguard:** If $D \le 10^{-8}$ (e.g. constant field), raises `ValueError` if $\text{RMSE} > 10^{-8}$, or returns $0.0$ if $\text{RMSE} \le 10^{-8}$.

### 3.4 Relative $L_2$ Error Norm ($\epsilon_{L_2}$)
Standard in computational physics and PDE operator learning (e.g. FNO / DeepONet benchmarks):

$$\epsilon_{L_2} = \frac{\|\mathbf{y}_{\text{true}} - \mathbf{y}_{\text{pred}}\|_2}{\|\mathbf{y}_{\text{true}}\|_2} = \sqrt{\frac{\sum_{i=1}^{N_{\text{fluid}}} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N_{\text{fluid}}} y_i^2}}$$

- **Units:** Dimensionless ratio.
- **Function:** `compute_relative_l2(y_true, y_pred, mask=building_mask)`.

### 3.5 Coefficient of Determination ($R^2$)
Measures the proportion of variance in fluid velocity explained by the surrogate:

$$R^2 = 1 - \frac{\sum_{i=1}^{N_{\text{fluid}}} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N_{\text{fluid}}} (y_i - \bar{y})^2}$$

- **Units:** Dimensionless ($1.0$ is perfect; $\le 0.0$ indicates performance worse than the spatial mean predictor).
- **Function:** `compute_r2_score(y_true, y_pred, mask=building_mask)`.

### 3.6 Spatial Absolute Error Map
Generates a 2D raster field of absolute point-wise errors for spatial inspection:

$$\Delta U(y, x) = \begin{cases} |y_{\text{true}}(y, x) - y_{\text{pred}}(y, x)| & \text{if } M_{\text{building}}(y, x) = 0 \\ 0.0 & \text{if } M_{\text{building}}(y, x) = 1 \end{cases}$$

- **Target Units:** Meters per second [$\text{m/s}$].
- **Function:** `compute_spatial_error_map(y_true, y_pred, building_mask)`.

---

## 4. Robust Validation & Invariant Enforcement

To eliminate subtle evaluation bugs in production or testing, `_align_and_extract_fluid_arrays` enforces the following invariants:

```
+-----------------------------------------------------------------------------------------+
|                                    INPUT VERIFICATION                                   |
|   1. Squeeze leading singleton dims (1, H, W) -> (H, W)                                 |
|   2. Verify shape(y_true) == shape(y_pred) == shape(mask)                               |
|   3. Resolve mask semantics (fluid_mask vs building_mask)                               |
|   4. Verify non-zero fluid cells (num_fluid > 0)                                        |
|   5. Verify all values in fluid domain are finite (no NaN / Inf)                        |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
                      [EXACT 1D ARRAYS OVER FLUID DOMAIN]
```

1. **Spatial Shape Alignment:**
   - Leading singleton dimensions (e.g., PyTorch batch/channel `(1, 1, H, W)` or `(1, H, W)`) are squeezed to spatial grid shape `(H, W)`.
   - If spatial dimensions differ between `y_true`, `y_pred`, and `mask`, raises `ValueError("Spatial alignment error: ...")`.
2. **Mask Semantics:**
   - `building_mask`: Boolean array where `True` denotes solid building cells and `False` denotes fluid flow cells.
   - `fluid_mask`: Boolean array where `True` denotes fluid flow cells and `False` denotes solid building cells.
   - Functions support `building_mask=...`, `fluid_mask=...`, or generic `mask=...` with `mask_is_fluid=False` (default).
3. **Empty Mask Guard:**
   - If a domain contains $0$ valid fluid cells (e.g. all-building domain), raises `ValueError("Empty mask error: domain contains 0 valid fluid cells...")`.
4. **Strict Non-Finite Value Check:**
   - Any `NaN` or `Inf` in evaluated fluid cells of targets or predictions raises an explicit `ValueError` identifying the exact count of non-finite elements. Targets and predictions must be strictly finite real numbers.
   - If mask contains `NaN` values, raises `ValueError("Mask contains NaN values...")`.

---

## 5. Synthetic Test Fixtures vs. Scientific Validation

### 5.1 Clear Separation of Purpose
- **Synthetic Test Fixtures (`tests/fixtures/synthetic_data.py`):**
  - Synthetically generated arrays with hand-calculated analytical metric values.
  - Used strictly for software unit tests, shape compatibility tests, gradient flow checks, and pipeline verification.
  - **Explicit Disclaimer:** Synthetic fixtures must never be reported in papers, PR descriptions, or presentations as evidence of neural surrogate predictive accuracy.
- **Scientific Validation:**
  - Evaluated on verified CFD simulation runs (UrbanTALES LES or OpenFOAM RANS) that adhere to `src.data.contract.UrbanLayoutSample`.

### 5.2 Hand-Calculated Benchmark Reference Values
The `create_synthetic_metric_benchmark_fixture()` provides a $4 \times 4$ analytical test field with $12$ fluid cells and $4$ building cells, where fluid cells have a known ground truth sequence $y_i \in [2.0, 13.0]\text{ m/s}$ and predictions have uniform error $+1.5\text{ m/s}$:

- $\text{MAE} = 1.5000\text{ m/s}$
- $\text{RMSE} = 1.5000\text{ m/s}$
- $\text{Max Error} = 1.5000\text{ m/s}$
- $\text{NRMSE}_{\text{range}} = \frac{1.5}{13.0 - 2.0} = \frac{3}{22} \approx 13.64\%$
- $\text{NRMSE}_{\text{inflow}} (U_{\text{ref}} = 10\text{ m/s}) = \frac{1.5}{10.0} = 15.00\%$
- $\text{Relative } L_2 = \sqrt{\frac{27}{818}} \approx 0.1817$
- $R^2 = \frac{116}{143} \approx 0.8112$

---

## 6. Example Usage and API Quickstart

### 6.1 Partitioning a Dataset and Exporting Metadata
```python
from src.data.split import LayoutSplitter

layout_ids = [f"layout_{i:03d}" for i in range(100)]

# 1. Ratio split (70 / 15 / 15) with fixed random seed
split = LayoutSplitter.split_by_ratio(
    layout_ids=layout_ids,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
)

# 2. Strict non-overlap verification
split.assert_no_overlap()

# 3. Export split metadata to JSON
split.save_json("configs/splits/baseline_split_seed42.json")
```

### 6.2 Evaluating a Spatial Prediction Field
```python
import numpy as np
from src.evaluation.metrics import compute_regression_metrics

# Load CFD target, surrogate prediction, and building mask
y_true = np.load("sample_true.npy")       # (Ny, Nx) in m/s
y_pred = np.load("sample_pred.npy")       # (Ny, Nx) in m/s
building_mask = np.load("mask.npy")       # (Ny, Nx) bool (True for building)

metrics = compute_regression_metrics(
    y_true=y_true,
    y_pred=y_pred,
    building_mask=building_mask,
    u_ref=5.0,  # Ambient inflow speed in m/s
)

print(f"MAE:  {metrics['mae_mps']:.3f} m/s")
print(f"RMSE: {metrics['rmse_mps']:.3f} m/s")
print(f"NRMSE (range):  {metrics['nrmse_range_percent']:.2f}%")
print(f"NRMSE (inflow): {metrics['nrmse_inflow_percent']:.2f}%")
print(f"R2:   {metrics['r2_score']:.3f}")
```

---

## 7. Limitations & Open Assumptions

1. **Pedestrian Height Focus ($z = 1.75\text{ m}$):**
   - All spatial metrics currently evaluate on 2D horizontal slices at pedestrian height. Vertical velocities ($W$) and 3D volumetric field evaluation are out of scope for the current benchmark.
2. **Statistical Uncertainty on Small Datasets:**
   - Small held-out test splits ($N < 20$ layouts) exhibit wider confidence intervals. Results should report mean $\pm$ standard deviation across multi-seed evaluations.
3. **Isothermal Flow Assumption:**
   - Current targets reflect aerodynamic velocity fields. Thermal buoyancy extensions (surface temperature, Richardson number) will reuse the same masked evaluation interface with temperature units [$\text{K}$].
