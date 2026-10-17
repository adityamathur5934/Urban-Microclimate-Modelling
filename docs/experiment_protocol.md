# Experiment Protocol & Rigorous Evaluation Guidelines

**Course Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Role:** Machine Learning & Deep Learning Lead  
**Scope:** Preventing Misleading Evaluation, Metric Definitions, and Scientific Comparison Protocol  

---

## 1. Principles of Fair & Leakage-Free Evaluation

In spatial CFD and microclimate surrogate modeling, conventional random data splitting creates catastrophic data leakage due to spatial autocorrelation. This protocol establishes strict safeguards:

### 1.1 Complete Layout Holdout (Zero Spatial Leakage)
- **Rule:** Individual pixels, grid cells, or localized patches from a single urban layout must **never** be divided across training, validation, and testing sets.
- **Why Random Pixel Splitting is Invalid:** Nearby fluid cells in the same CFD simulation share pressure gradients and boundary conditions. Splitting pixels allows a model to memorize neighboring points from the test domain, reporting artificially near-zero errors that fail completely on any new building layout.
- **Enforcement:** Splitting is performed strictly at the **layout ID** level via `src.data.LayoutSplitter`.

### 1.2 Two Benchmark Partitioning Protocols
1. **Ratio Holdout (Within-Family Generalization):**
   - 70% Train, 15% Validation, 15% Test.
   - Stratified across plan area density bins ($\lambda_p \in [0.06, 0.44]$) to ensure diverse building packing in all splits.
2. **Family Holdout (Out-of-Distribution Generalization):**
   - Train on `idealized_aligned` and `idealized_staggered` arrays.
   - Held-out test set consists exclusively of `realistic_neighborhood` geometries.
   - Tests whether neural representations learned on regular blocks generalize to complex real-world city geometry.

### 1.3 Preprocessing & Normalization Isolation
- All scalers (mean, variance, min-max bounds) must be fitted strictly on the designated training split via `FieldNormalizer` and `FeatureNormalizer`.
- Validation and test layouts must be transformed using the frozen training parameters.
- Normalizer states are persisted to `data/preprocessing_config.json` for reproducible inference.

### 1.4 Test Set Isolation During Model Selection
- The test partition remains completely untouched during model training and hyperparameter selection.
- Early stopping, learning rate schedules, and model selection must use the validation partition only.

---

## 2. Documented Metric Definitions

### 2.1 Spatial Field Metrics (Neural Surrogate)
All field metrics are evaluated strictly over outdoor fluid cells ($\Omega_{\text{outdoor}} = \{(y, x) \mid M_{\text{building}}(y, x) = 0\}$).

1. **Masked Field Mean Absolute Error (MAE):**
   $$\text{MAE}_{\text{field}} = \frac{1}{|\Omega_{\text{outdoor}}|} \sum_{(y, x) \in \Omega_{\text{outdoor}}} |Y(y, x) - \hat{Y}(y, x)|$$
   *Physical Meaning:* Average absolute discrepancy in wind speed ($\text{m/s}$) across the pedestrian domain.

2. **Masked Field Root Mean Squared Error (RMSE):**
   $$\text{RMSE}_{\text{field}} = \sqrt{\frac{1}{|\Omega_{\text{outdoor}}|} \sum_{(y, x) \in \Omega_{\text{outdoor}}} (Y(y, x) - \hat{Y}(y, x))^2}$$
   *Physical Meaning:* Quadratic penalty weighting large local errors (e.g. missed corner gusts).

3. **Normalized Root Mean Squared Error (NRMSE):**
   To prevent ambiguity across different literature conventions, three explicit normalization bases are provided in [src/training/metrics.py](file:///Users/vaidehisahu/Downloads/amlproj/Urban-Microclimate-Modelling/src/training/metrics.py):

   - **Range Normalization ($\text{NRMSE}_{\text{range}}$) — Default:**
     $$\text{NRMSE}_{\text{range}} = \frac{\text{RMSE}_{\text{field}}}{Y_{\max} - Y_{\min}}$$
     Measures error as a fraction of the observed velocity dynamic range in outdoor cells.

   - **Inflow Normalization ($\text{NRMSE}_{\text{inflow}}$) — Aerodynamic Standard:**
     $$\text{NRMSE}_{\text{inflow}} = \frac{\text{RMSE}_{\text{field}}}{U_{\text{ref}}}$$
     Standard in wind engineering: normalizes prediction error by the ambient approach wind speed $U_{\text{ref}}$.

   - **Standard Deviation Normalization ($\text{NRMSE}_{\text{std}}$):**
     $$\text{NRMSE}_{\text{std}} = \frac{\text{RMSE}_{\text{field}}}{\sigma_Y}$$
     Statistical normalization equivalent to $\sqrt{1 - R^2}$ when predictions are unbiased.

---

## 3. Resolving the AML vs. DL Resolution Gap

### 3.1 The Representation Difference
- The **AML Baseline** operates on a 1D tabular vector $\mathbf{x} \in \mathbb{R}^{11}$ and predicts a **scalar aggregate**: $\hat{y}_{\text{AML}} \in \mathbb{R}$ (`area_averaged_wind_speed`).
- The **Neural Surrogate** operates on a multi-channel tensor $\mathbf{X} \in \mathbb{R}^{7 \times H \times W}$ and predicts a **2D spatial field**: $\hat{\mathbf{Y}}_{\text{DL}} \in \mathbb{R}^{H \times W}$ (`pedestrian_wind_speed`).

### 3.2 The Spatial Aggregation Bridge for Fair Comparison
Direct comparison at the field pixel level is impossible because the AML baseline does not output a 2D field. To conduct a scientifically valid, apples-to-apples comparison on held-out test layouts:

The neural surrogate's 2D field prediction must be spatially aggregated over outdoor fluid cells:

$$\hat{\bar{U}}_{\text{surrogate}} = \frac{1}{|\Omega_{\text{outdoor}}|} \sum_{(y, x) \in \Omega_{\text{outdoor}}} \hat{Y}_{\text{DL}}(y, x)$$

Implemented in `src.training.aggregate_spatial_field_to_scalar`.

Both models can then be evaluated against the CFD ground truth scalar target $\bar{U}_{\text{CFD}}$ using identical scalar metrics on the held-out test partition:

| Model | Predicted Entity | Bridge to Scalar | Evaluated Target | Test MAE ($\text{m/s}$) | Test RMSE ($\text{m/s}$) | Test $R^2$ |
|---|---|---|---|---|---|---|
| **AML Baseline (RF)** | Scalar $\hat{y}$ | Direct output | $\bar{U}_{\text{CFD}}$ | Evaluated | Evaluated | Evaluated |
| **Neural Surrogate (U-Net)** | 2D Field $\hat{Y}$ | Spatial outdoor average $\hat{\bar{U}}$ | $\bar{U}_{\text{CFD}}$ | Evaluated | Evaluated | Evaluated |

### 3.3 What the Neural Surrogate Uniquely Provides
Even if the scalar area-averaged MAE between the AML baseline and DL surrogate is comparable, the neural surrogate provides critical spatial intelligence that tabular models cannot:
1. **Local Wake Zones:** Low-velocity recirculation cavities downstream of buildings.
2. **Street Canyon Venturi Effects:** Local velocity acceleration through narrow building gaps.
3. **Corner Vortices:** Peak pedestrian wind hazard zones near building corners.

---

## 4. Limits of Conclusions from Small Datasets & Synthetic Tensors

1. **Software Smoke Tests vs. Scientific Validation:**
   - Synthetic fixtures in `tests/fixtures/synthetic_data.py` are strictly mathematical test fixtures to verify software pipelines, tensor shapes, gradient flow, and checkpointing.
   - **Never cite synthetic test metrics as scientific findings.**
2. **Statistical Uncertainty on Small CFD Sweeps:**
   - When evaluating on small datasets (e.g. 10 to 30 CFD simulations), test set metrics have wide confidence intervals. Report confidence intervals (mean $\pm$ std across 5-fold cross-validation) rather than a single point estimate.
3. **No Premature Outperformance Claims:**
   - Do not claim the neural surrogate outperforms the baseline until both are evaluated on the identical held-out test layouts using the agreed data contract.

---

## 5. Unresolved Decisions & Coordination Checklist

Prior to final dataset integration, the team must confirm:
- [ ] **CFD Sample Ingestion Format:** Finalize the export script from UrbanTALES / OpenFOAM runs to compressed `.npz` conforming to `UrbanLayoutSample`.
- [ ] **Grid Resolution Consensus:** Confirm default raster grid shape (e.g., $64 \times 64$ or $128 \times 128$) and physical resolution ($\Delta x = 1.0\text{ m}$ vs. $0.5\text{ m}$).
- [ ] **Target Selection Confirmation:** UrbanTALES is isothermal; confirm pedestrian wind speed ($U_{1.75\text{m}}$) as primary target across team deliverables, maintaining thermal configuration hooks in `TargetConfig` for future extensions.
