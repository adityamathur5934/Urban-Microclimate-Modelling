# Field Visualization & Spatial Error Analysis Guide

**Project:** Urban Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Role:** Machine Learning Evaluation Engineer (Team Member C)  
**Document Version:** 1.0.0  
**Branch:** `feature/field-visualization`  
**Status:** Approved Reference Guide  

---

## 1. Executive Summary & Purpose

Spatial field visualization is critical for diagnosing microclimatic flow behavior, identifying localized surrogate prediction failures, and verifying aerodynamic boundary-layer physics (such as corner separation jets, leeward wake recirculation cavities, and street canyon channeling).

This guide documents:
1. **Coordinate and orientation conventions** ensuring physical fidelity without unverified rotations or spatial flipping.
2. **Unified color scaling rules** guaranteeing fair visual comparison between reference CFD and model predictions.
3. **Directional error analysis** using signed error maps ($y_{\text{pred}} - y_{\text{true}}$) alongside absolute error maps ($|y_{\text{true}} - y_{\text{pred}}|$).
4. **Distinct visual representation of solid building morphology** to prevent confusion with fluid stagnation zones.
5. **How to run the visualization pipeline** on both real verified CFD data and synthetic software test fixtures.

---

## 2. Spatial Coordinate & Grid Conventions

To prevent silent geometric distortions, all plotting utilities strictly adhere to the project's data schema ([`docs/data_schema.md`](file:///Users/prateekshukla/Urban-Microclimate-Modelling/docs/data_schema.md)):

```
       +y (North, Row Index)
        ^
        |   (0, Ny-1) ---------------- (Nx-1, Ny-1)
        |       |                             |
        |       |        FLOW DOMAIN          |
        |       |    (Streamwise: +x)         |
        |       |                             |
        |     (0, 0) ------------------- (Nx-1, 0)
        +----------------------------------------> +x (East, Column Index)
```

- **Array Indexing:** 2D horizontal slices are indexed as `[y, x]`, where $y \in [0, N_y-1]$ is the vertical/row dimension and $x \in [0, N_x-1]$ is the horizontal/column dimension.
- **Matplotlib Origin:** All routines strictly set `origin="lower"`. Row index $y = 0$ is placed at the bottom, pointing upwards along $+y$ (south-to-north). Column index $x = 0$ is placed at the left, pointing rightwards along $+x$ (west-to-east).
- **Physical Extent:** If physical cell sizes $(\Delta x, \Delta y, \Delta z)$ are provided, axes are automatically calibrated in meters:
  $$\text{extent} = [0, N_x \Delta x, 0, N_y \Delta y]$$
- **No Uncontrolled Resampling:** Plots set `interpolation="nearest"` (or `None`) to preserve raw computational grid cells and prevent false spatial smoothing across solid building boundaries.

---

## 3. Scale Fairness & Colorbar Conventions

A common pitfall in surrogate modeling is rendering ground truth and predictions with independent colorbars, artificially masking surrogate dampening or extreme-value underestimation.

Our visualization pipeline enforces:

### 3.1 Shared Color Limits ($v_{\min}, v_{\max}$)
Panels displaying the reference CFD flow and surrogate predictions share identical color limits:
$$v_{\min} = \min\left(\min(y_{\text{true}}), \min(y_{\text{pred}})\right)$$
$$v_{\max} = \max\left(\max(y_{\text{true}}), \max(y_{\text{pred}})\right)$$
- If the true peak is $7.5\text{ m/s}$ and the prediction only reaches $5.0\text{ m/s}$, the predicted panel visually appears lighter/cooler, immediately exposing the peak attenuation to domain experts.

### 3.2 Absolute vs. Signed Error Maps

| Error Map | Formula | Colormap | Zero Position | Purpose |
|---|---|---|---|---|
| **Absolute Error** | $|y_{\text{true}} - y_{\text{pred}}|$ | `magma` | Left boundary ($0.0$) | Highlights overall magnitude of local discrepancy; annotated with field MAE. |
| **Signed Error** | $y_{\text{pred}} - y_{\text{true}}$ | `coolwarm` / `RdBu_r` | Symmetric center ($0.0$) | Identifies **directional bias**: <br>• **Red ($> 0$):** Overprediction (e.g. corner jets) <br>• **Blue ($< 0$):** Underprediction (e.g. wake stagnation). |

---

## 4. Building Footprint & Mask Overlay

In urban microclimate modeling, velocity inside solid buildings is non-physical. If building cells are filled with $0.0\text{ m/s}$, readers cannot distinguish solid walls from aerodynamic stagnation zones ($U \approx 0$).

- Solid cells (`building_mask == True`) are converted to `np.nan` in plotting arrays.
- Subplot backgrounds are explicitly set to a dark neutral charcoal `#2b2b2b`.
- Solid walls render as distinct dark blocks, allowing immediate visual separation between urban obstacles and low-velocity flow recirculation zones.

---

## 5. Physical Variables & Supported Fields

Figures configure colormaps and units dynamically based on the modeled physical quantity:

| `field_type` | Variable Name | Physical Units | Default Colormap | Notes |
|---|---|---|---|---|
| `"wind_speed"` | Pedestrian Wind Speed | $\text{m/s}$ | `viridis` | Standard target evaluated at $z = 1.75\text{ m}$. |
| `"velocity_u"` | Streamwise Velocity | $\text{m/s}$ | `viridis` | Horizontal $+x$ component. |
| `"velocity_v"` | Spanwise Velocity | $\text{m/s}$ | `viridis` | Lateral $+y$ component. |
| `"turbulent_kinetic_energy"` | TKE | $\text{m}^2/\text{s}^2$ | `cividis` | Turbulence energy $k = \frac{1}{2}(\overline{u'^2} + \overline{v'^2} + \overline{w'^2})$. |
| `"temperature"` | Ambient Temperature | $\text{K}$ | `inferno` | **Only used when thermal fields are explicitly present.** Never implied for velocity-only runs. |

---

## 6. How to Run Visualizations

### 6.1 Using the CLI Tool (`scripts/visualize_fields.py`)

#### Option A: Software Verification on Synthetic Fixture (No Data Required)
```bash
python scripts/visualize_fields.py --synthetic --include-signed-error --output-dir outputs/plots/
```
- Generates a 4-panel comparison: [CFD Reference] | [Surrogate Prediction] | [Signed Error] | [Absolute Error].
- Automatically labeled with `[SYNTHETIC_TEST_FIXTURE]`.

#### Option B: On a Verified Real CFD Sample (`UrbanLayoutSample` NPZ)
```bash
python scripts/visualize_fields.py \
    --sample-path data/samples/sydney_neighborhood_01.npz \
    --include-signed-error \
    --provenance VERIFIED_CFD_DATA \
    --output-dir outputs/plots/
```

#### Option C: From Saved Prediction & Ground-Truth Arrays
```bash
python scripts/visualize_fields.py \
    --true-npy outputs/predictions/true_field.npy \
    --pred-npy outputs/predictions/unet_pred.npy \
    --mask-npy outputs/predictions/building_mask.npy \
    --layout-id idealized_aligned_H25 \
    --include-signed-error \
    --output-dir outputs/plots/
```

### 6.2 Python API Usage

```python
import numpy as np
from src.evaluation.plotting import plot_field_triplet, plot_field_quad

# Load fields
y_true = np.load("target.npy")
y_pred = np.load("pred.npy")
building_mask = np.load("mask.npy")

# 1. Generate 3-panel comparison (Reference, Prediction, Absolute Error)
fig, out_path = plot_field_triplet(
    y_true=y_true,
    y_pred=y_pred,
    mask=building_mask,
    layout_id="real_sydney_042",
    field_type="wind_speed",
    inflow_speed=5.0,
    provenance_tag="VERIFIED_CFD_DATA",
    output_path="outputs/plots/sydney_triplet.png",
)

# 2. Generate 4-panel comparison with signed error for bias analysis
fig_quad, quad_path = plot_field_quad(
    y_true=y_true,
    y_pred=y_pred,
    mask=building_mask,
    layout_id="real_sydney_042",
    field_type="wind_speed",
    inflow_speed=5.0,
    provenance_tag="VERIFIED_CFD_DATA",
    output_path="outputs/plots/sydney_quad.png",
)
```

---

## 7. Data Provenance & Scientific Integrity

In accordance with [`docs/ui-scientific-audit.md`](file:///Users/prateekshukla/Urban-Microclimate-Modelling/docs/ui-scientific-audit.md), every generated figure incorporates a **Provenance Tag** in its header:

1. **`VERIFIED_CFD_DATA`:** Direct output from converged OpenFOAM solver cases or ingested UrbanTALES LES cases.
2. **`CANONICAL_BENCHMARK`:** Validated analytical bluff-body aerodynamics benchmark (e.g. Castro & Robins, Martinuzzi & Tropea).
3. **`SURROGATE_PREDICTION`:** Forward inference from trained PyTorch surrogate (`CompactUNet`) or tabular ML baseline (`RandomForestBaseline`).
4. **`SYNTHETIC_TEST_FIXTURE`:** Mathematical smoke-test arrays used exclusively to verify software plotting code without physical claims.

> [!CAUTION]
> Never report synthetic test fixtures in publications or slide decks as evidence of neural surrogate predictive accuracy. Always use verified CFD test splits.
