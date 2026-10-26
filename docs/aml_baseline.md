# Traditional Machine Learning Baseline: Random Forest Regression

## 1. Executive Summary & Scope

This document defines the architecture, data contracts, evaluation protocol, and software smoke test for the **Traditional Machine Learning (AML) Baseline** in the Urban Microclimate Modelling framework.

### Baseline Scope & Physical Boundaries
> [!IMPORTANT]
> **Scope Boundary:**
> The AML baseline predicts a **single scalar aggregate quantity** — the canopy area-averaged pedestrian wind speed $\langle U_{\text{ped}} \rangle_{\text{fluid}}$ [m/s] — for an entire urban domain given 1D geometric summary statistics and atmospheric inflow boundary conditions.
>
> **What this baseline CANNOT do:**
> - It **cannot reproduce local spatial flow structures**, such as recirculating wake cavities downwind of buildings, corner separation jets, horseshoe vortex systems, or canyon Venturi channelling.
> - High-resolution 2D spatial field predictions require the spatial neural surrogate architecture (e.g. UNet / Fourier Neural Operator), which operates on 2D raster inputs.
>
> The purpose of this baseline is to establish a rigorous, transparent lower bound on predictive accuracy and to quantify how much microclimate variance can be explained by zero-dimensional bulk aerodynamic parameters alone.

---

## 2. Target Formulation & Physical Consistency

### 2.1 Target Definition
The prediction target is the **canopy area-averaged outdoor pedestrian wind speed**:

$$\langle U_{\text{ped}} \rangle_{\text{fluid}} = \frac{\int_{\Omega_{\text{fluid}}} U_{\text{ped}}(\mathbf{x}) \, dA}{\int_{\Omega_{\text{fluid}}} dA}$$

On a discrete computational grid $(N_y, N_x)$ with differential cell areas $\Delta A_{i,j}$:

$$\langle U_{\text{ped}} \rangle_{\text{fluid}} = \frac{\sum_{i=1}^{N_y} \sum_{j=1}^{N_x} U_{i,j} \cdot (1 - M_{i,j}) \cdot \Delta A_{i,j}}{\sum_{i=1}^{N_y} \sum_{j=1}^{N_x} (1 - M_{i,j}) \cdot \Delta A_{i,j}}$$

where:
- $U_{i,j} \ge 0$: Pedestrian-level horizontal wind speed magnitude at $z = 1.75\text{ m}$ [m/s].
- $M_{i,j} \in \{0, 1\}$: Boolean building footprint mask ($1 = \text{building interior}$, $0 = \text{outdoor fluid cell}$).
- $\Delta A_{i,j} = \Delta x \cdot \Delta y$: Horizontal cell area [$\text{m}^2$]. For uniform grids, $\Delta A_{i,j}$ cancels out, reducing to the arithmetic mean over outdoor cells.

### 2.2 Fluid-Masking Safeguards
Implemented in [`src/data/area_averaging.py`](file:///Users/yachna./Desktop/Urbanmodelling/Urban-Microclimate-Modelling/src/data/area_averaging.py):
1. **Solid Interior Exclusion:** Building interior cells ($M_{i,j} = 1$) are strictly masked out. Including solid cells (which are set to zero in CFD fields) would artificially dilute the canopy wind speed as building packing density $\lambda_p$ increases.
2. **Zero Fluid Cell Guard:** Domains with $100\%$ building coverage raise an explicit `ValueError`.
3. **Finite Value Validation:** Any NaN or Inf values in the flow field trigger immediate validation failure.

---

## 3. Feature Space & Leakage Safeguards

### 3.1 26-Dimensional Predictor Vector
The baseline ingests exclusively the 26 tabular features extracted by [`GeometricFeatureExtractor`](file:///Users/yachna./Desktop/Urbanmodelling/Urban-Microclimate-Modelling/src/data/features.py):

| Group | Features | Description | Units |
|---|---|---|---|
| **Bulk Aerodynamics** | `plan_area_density_lambda_p`, `frontal_area_density_lambda_f`, `wall_area_density_lambda_w`, `roughness_length_macdonald_m`, `displacement_height_d_m` | MacDonald et al. (1998) morphometric roughness & displacement | —, —, —, m, m |
| **Height Statistics** | `mean_building_height_m`, `std_building_height_m`, `max_building_height_m`, `min_building_height_m`, `p25_building_height_m`, `p75_building_height_m`, `height_iqr_m`, `height_cv` | Canopy roof-level elevation distribution and vertical heterogeneity | m (quantiles), — (CV) |
| **Component Morphometry** | `building_count`, `mean_footprint_area_m2`, `std_footprint_area_m2`, `max_footprint_area_m2`, `footprint_area_cv`, `built_area_total_m2` | Connected-component footprint statistics (via 8-connectivity labeling) | count, $\text{m}^2$, — (CV) |
| **Domain Geometry** | `domain_aspect_ratio_yx`, `mean_height_times_lambda_p_m` | Domain planform elongation and canopy volume proxy | —, m |
| **Environmental Inflow** | `inflow_speed_mps`, `wind_direction_cos`, `wind_direction_sin`, `reference_height_m`, `roughness_length_z0_m` | Boundary conditions at computational domain inflow boundary | m/s, —, —, m, m |

### 3.2 Target Leakage Safeguards
- **Predictor Isolation:** Metadata (`layout_id`) and the target itself (`target_area_averaged_wind_speed`) are explicitly excluded from `to_feature_vector()` and `feature_names()`.
- **Preprocessing Isolation:** The preprocessing pipeline uses `StandardScaler`, which is **fitted strictly on the training partition**. Normalization parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are held constant during validation and testing.

---

## 4. Dataset Partitioning Strategy

### 4.1 Whole-Layout & Family-Level Holdout
Randomly partitioning pixels or adjacent wind angles across train and test sets leads to catastrophic spatial autocorrelation leakage. The baseline strictly partitions by whole layout ID:

1. **Ratio-Based Layout Split (`split_by_ratio`):**
   - Unique layout IDs are randomly assigned to Train ($70\%$), Validation ($15\%$), and Test ($15\%$) using a fixed random seed.
   - Guarantees zero sample overlap across splits:
     $$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

2. **Family-Based Holdout (`split_by_family`):**
   - Holds out an entire architectural family (e.g., `REALISTIC_NEIGHBORHOOD`) exclusively for testing, assessing model generalizability to unseen urban typologies.

---

## 5. Model Architecture & Naive Benchmark

### 5.1 Model Selection Rationale: Random Forest
- **Why Random Forest:** Scikit-Learn `RandomForestRegressor` is already installed in the environment, highly transparent, naturally invariant to monotonic feature scaling, and provides robust out-of-bag feature importances.
- **XGBoost Status:** XGBoost is not pre-installed. In accordance with project instructions ("prefer the simpler available dependency if there is no reason to add another"), Random Forest is chosen as the primary baseline.

### 5.2 Model Hyperparameters
- `n_estimators`: 100 decision trees
- `max_depth`: 12 levels
- `min_samples_split`: 2
- `min_samples_leaf`: 1
- `random_state`: 42 (ensuring exact reproducibility)
- `n_jobs`: -1 (parallel CPU tree construction)

### 5.3 Zero-Skill Naive Benchmark (`NaiveMeanBaseline`)
To ensure scientific rigor, all evaluations compare the Random Forest against a **Naive Mean Baseline**:

$$\hat{y}_{\text{naive}} = \bar{y}_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i \in \mathcal{S}_{\text{train}}} y_i$$

The naive baseline achieves $R^2 = 0.0$ on the training split. Any viable ML baseline must demonstrate substantial error reduction and a strictly positive $R^2$ on held-out test data relative to this benchmark.

---

## 6. Regression Evaluation Metrics & Normalization Denominators

All evaluation metrics are computed in true physical units:

1. **Mean Absolute Error (MAE) [m/s]:**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

2. **Root Mean Squared Error (RMSE) [m/s]:**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

4. **Normalized RMSE by Range ($\text{NRMSE}_{\text{range}}$) [%]:**
   $$\text{NRMSE}_{\text{range}} = \frac{\text{RMSE}}{\max(y) - \min(y)} \times 100\%$$
   *Normalization Denominator:* Target dynamic range $(y_{\max} - y_{\min})$ over the partition [m/s].

5. **Normalized RMSE by Mean ($\text{NRMSE}_{\text{mean}}$) [%]:**
   $$\text{NRMSE}_{\text{mean}} = \frac{\text{RMSE}}{|\bar{y}|} \times 100\%$$
   *Normalization Denominator:* Ground-truth mean target speed $|\bar{y}|$ over the partition [m/s].

6. **Maximum Absolute Error ($\text{Max Error}$) [m/s]:**
   $$\text{Max Error} = \max_{i=1 \dots N} |y_i - \hat{y}_i|$$

---

## 7. Reproducible CLI Entry Point

The baseline pipeline is executed via [`scripts/run_aml_baseline.py`](file:///Users/yachna./Desktop/Urbanmodelling/Urban-Microclimate-Modelling/scripts/run_aml_baseline.py):

```bash
# Run software smoke test with synthetic fixtures
python scripts/run_aml_baseline.py --synthetic --num-samples 30 --output-dir outputs/baseline --seed 42

# Run on verified CFD NPZ archive directory (when available)
python scripts/run_aml_baseline.py --data-dir path/to/npz --output-dir outputs/baseline
```

### Persisted Artifacts (Stored in git-ignored `outputs/`):
- `outputs/baseline/split_assignments.json`: Exact layout IDs for train, val, test, and random seed.
- `outputs/baseline/rf_baseline.joblib`: Serialized Scikit-Learn Pipeline (`StandardScaler` + `RandomForestRegressor`).
- `outputs/baseline/baseline_metrics.json`: Full evaluation results, relative improvements, and top feature importances.

---

## 8. Software Smoke Test Execution & Provenance

> [!WARNING]
> **SOFTWARE SMOKE TEST ONLY — NOT SCIENTIFIC CFD PERFORMANCE**
> Because local access to verified UrbanTALES CFD raw data is pending download, the following run was executed using the synthetic data generator ([`tests/fixtures/synthetic_data.py`](file:///Users/yachna./Desktop/Urbanmodelling/Urban-Microclimate-Modelling/tests/fixtures/synthetic_data.py)).
> These scores demonstrate pipeline functionality, dimensional correctness, and zero-leakage isolation. **They must not be quoted as scientific model accuracy.**

### Smoke Test Results (30 Synthetic Layouts, Seed 42):

| Partition | Model | MAE [m/s] | RMSE [m/s] | $R^2$ | $\text{NRMSE}_{\text{range}}$ | $\text{NRMSE}_{\text{mean}}$ |
|---|---|---|---|---|---|---|
| **Train** (21 layouts) | **Random Forest** | **0.0761** | **0.0903** | **0.9984** | **1.41%** | **2.02%** |
| | Naive Mean Baseline | 1.9371 | 2.2297 | 0.0000 | 34.75% | 49.95% |
| **Val** (4 layouts) | **Random Forest** | **0.0520** | **0.0799** | **0.9963** | **2.48%** | **2.04%** |
| | Naive Mean Baseline | 1.3321 | 1.4238 | -0.1799 | 44.27% | 36.43% |
| **Test** (5 layouts) | **Random Forest** | **0.1162** | **0.1258** | **0.9981** | **1.96%** | **2.49%** |
| | Naive Mean Baseline | 2.8672 | 2.9595 | -0.0403 | 46.12% | 58.64% |

### Test Split Performance Gain over Naive Benchmark:
- **MAE Reduction:** $95.95\%$ improvement
- **RMSE Reduction:** $95.75\%$ improvement
- **$R^2$ Delta:** $+1.0384$

### Top-5 Identified Feature Importances:
1. `inflow_speed_mps`: $98.99\%$ (Primary driver of flow kinetic energy)
2. `wall_area_density_lambda_w`: $0.20\%$
3. `roughness_length_macdonald_m`: $0.16\%$
4. `frontal_area_density_lambda_f`: $0.13\%$
5. `displacement_height_d_m`: $0.09\%$
