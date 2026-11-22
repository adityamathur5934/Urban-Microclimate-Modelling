# CFD-to-Surrogate Experiment: Implementation Evidence

**Status:** Experiment implemented and executed. Models genuinely trained and evaluated.  
**Critical Provenance:** Reference fields are **ANALYTICAL** (Castro & Robins 1977 formulation), **not** solver CFD output.  
**CFD Execution Status:** BLOCKED — OpenFOAM not available (see §11).

---

## 1. Experiment Overview

This document records what was actually built and run. The objective is a reproducible
pipeline that generates reference wind-speed fields, trains the existing three models,
and evaluates them on entirely unseen layouts with uncertainty estimates. All code is in
`src/experiment/` and `cfd/scripts/`.

**What this is:**  
A full CFD-to-surrogate pipeline using documented *analytical* aerodynamic reference
fields (adapted Castro & Robins 1977 / Martinuzzi & Tropea 1993 formulation) instead
of solver output. The method, labelling, and limitations are documented explicitly.

**What this is NOT:**  
- Not OpenFOAM solver output.
- Not UrbanTALES LES data.
- Not trained or fabricated to look like converged CFD.

---

## 2. Reference Dataset

| Property | Value |
|---|---|
| Generator | `src/experiment/reference.py` → `generate_reference_dataset()` |
| n_layouts | **18** |
| Grid shape | **64 × 64** cells |
| Domain size | **128 × 128 m**, Lz = 60 m |
| Resolution | **2 m × 2 m** (dx = dy = 128/64 = 2 m) |
| Families | `IDEALIZED_ALIGNED` (2×2 array), `IDEALIZED_STAGGERED` (2×2 staggered), `REALISTIC_NEIGHBORHOOD` (single block) |
| Heights varied | 12, 20, 28 m |
| Inflow speeds | 3.5, 5.0, 7.0 m/s |
| Wind directions | 0°, 15°, 30° |
| Building gaps | 6, 14, 24 m (dense → sparse) |
| Provenance label | `ANALYTICAL_REFERENCE` on every sample |
| Solver metadata | `Analytical_Aerodynamic_Reference (Castro & Robins 1977)` |
| CFD claimed | **No** |

The analytical model computes: upstream ABL log law → windward stagnation → lateral corner
(Venturi) acceleration jets → leeward recirculating wake → street-canyon channeling. This
is the same formulation used by the existing canonical benchmark in `cfd/outputs/`.

---

## 3. Train / Validation / Test Split

Split by complete layouts (zero leakage), seed=42, 66/17/17%:

| Split | Layout IDs | Count |
|---|---|---|
| **Train** | `ref_aligned_03`, `ref_aligned_12`, `ref_aligned_15`, `ref_single_02`, `ref_single_05`, `ref_single_11`, `ref_single_14`, `ref_single_17`, `ref_staggered_01`, `ref_staggered_04`, `ref_staggered_10`, `ref_staggered_13` | **12** |
| **Val** | `ref_aligned_06`, `ref_staggered_07`, `ref_staggered_16` | **3** |
| **Test** | `ref_aligned_00`, `ref_aligned_09`, `ref_single_08` | **3** |

Split leakage verified via `FieldNormalizer.verify_no_leakage()`.

---

## 4. Normalisation

- **Input normalizer:** `FieldNormalizer(method="standard", mask_buildings=True)` — fitted on 12 training layouts only. Channel-wise z-score over fluid cells; building cells clamped to 0.0.
- **Target normalizer:** same approach, fitted on the pedestrian wind speed channel only.
- Normalizers serialized inside the experiment manifest (`configs/cfd_surrogate_experiment.json`).
- Leakage-free: `verify_no_leakage()` checks that no val/test layout ID appeared during `fit()`.

---

## 5. Models Trained

### 5.1 Random Forest Baseline

| Setting | Value |
|---|---|
| Architecture | `RandomForestBaseline` (scikit-learn `RandomForestRegressor` + `StandardScaler`) |
| Features | 26-dimensional geometric morphometric vector (from `GeometricFeatureExtractor`) |
| Target | Scalar area-averaged outdoor wind speed (m/s) |
| n_estimators / max_depth | 100 / 12 |
| Checkpoint | `checkpoints/cfd_surrogate_experiment/rf_baseline.joblib` |
| `_is_fitted` | **True** |

### 5.2 U-Net Ensemble (3 independently seeded members)

| Setting | Value |
|---|---|
| Architecture | `CompactUNet(in_channels=7, out_channels=1, base_channels=16, depth=3)` |
| Input | 7-channel raster (height map, building mask, SDF, inflow u/v, coord x/y) |
| Target | Pedestrian wind speed at z = 1.75 m, (64 × 64) field |
| Ensemble size | **3** members (seeds 142, 143, 144) |
| Loss | `CombinedSurrogateLoss` (masked L1 α=0.7 + MSE α=0.3 + gradient) |
| Epochs | 35 (with early stopping, patience=12) |
| Optimizer | AdamW lr=1e-3, weight_decay=1e-4; CosineAnnealingLR |
| Checkpoints | `checkpoints/cfd_surrogate_experiment/compactunet_seed142/best_checkpoint.pt`, `..._seed143/...`, `..._seed144/...` |
| Loaded for eval | **Yes** — each checkpoint reloaded via `CompactUNet.load_checkpoint()` before inference |

### 5.3 Dilated CNN

| Setting | Value |
|---|---|
| Architecture | `CompactCNN(in_channels=7, out_channels=1, hidden_channels=16, num_layers=4)` |
| Same input/target as U-Net | Yes |
| Epochs | 35 |
| Checkpoint | `checkpoints/cfd_surrogate_experiment/compactcnn_seed242/best_checkpoint.pt` |
| Loaded for eval | **Yes** |

---

## 6. Test Metrics (Held-Out Layouts: `ref_aligned_00`, `ref_aligned_09`, `ref_single_08`)

All metrics evaluated strictly over outdoor fluid cells (`building_mask = False`).
Inference timing: single-process CPU forward pass, per-layout, platform: macOS Apple Silicon CPU.

| Model | MAE (m/s) | RMSE (m/s) | NRMSE_range (%) | NRMSE_Uref (%) | R² | Max Error (m/s) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **U-Net Ensemble Mean** | **0.1053** | **0.1496** | **3.85** | **3.37** | **0.8894** | **1.2288** | 6.74 |
| U-Net Member 0 (single) | 0.1288 | 0.1829 | 4.61 | 3.97 | 0.8412 | 1.1888 | 2.13 |
| Dilated CNN | 0.2132 | 0.3265 | 8.22 | 7.05 | 0.4954 | 1.7783 | 0.50 |
| RF Broadcast Spatial* | 0.2619 | 0.4524 | — | 10.11 | — | 2.2697 | — |
| RF Scalar (area-avg) | MAE=0.019 m/s | RMSE=0.024 m/s | — | — | R²=0.9996 | — | — |

*RF spatial metrics are computed by broadcasting the scalar prediction over the entire fluid domain (not a valid spatial surrogate; shown only for comparison).

**NRMSE conventions:**
- `NRMSE_range` = RMSE / (max(U_true) − min(U_true)) over fluid cells, expressed in %.
- `NRMSE_Uref` = RMSE / U_ref (inflow speed), expressed in %. Standard wind-engineering normalisation.
- `NRMSE_inflow` as reported here uses the per-layout U_ref from `environment.inflow_speed_mps`.

**Interpretation notes:**
- R² ≈ 0.89 for the ensemble mean indicates strong linear correlation with the reference field.
- The CNN converges slower and scores lower R² (0.50) with 35 epochs at base_channels=16; more training or a larger model would improve this.
- The RF scalar achieves R² ≈ 1.0 for the area-averaged target over only 3 test layouts — a valid result for a scalar regression task.
- **These metrics evaluate the surrogate vs the analytical reference, not vs CFD solver output.** See §11 for the blocker.

---

## 7. Uncertainty Analysis

**Method:** Ensemble predictive disagreement (per-pixel standard deviation across 3 independently trained U-Nets with different random seeds). This quantifies epistemic spread only — where model members disagree.

**What it is NOT:** Not a calibrated confidence interval. Calibration would require empirical coverage verification against CFD ground truth, which is blocked (see §11).

| Test layout | Mean pixel std (m/s) | Max pixel std (m/s) | P95 pixel std (m/s) |
|---|---|---|---|
| `ref_aligned_00` | 0.0483 | 0.6040 | 0.1367 |
| `ref_aligned_09` | 0.0483 | 0.6040 | 0.1367 |
| `ref_single_08` | 0.1230 | 0.6218 | 0.2474 |

The single-block layout (`ref_single_08`) shows higher ensemble disagreement (mean std 0.12 m/s) than the aligned-array layouts (0.048 m/s), consistent with a harder-to-predict geometry appearing fewer times in the training set. Maximum disagreement concentrates in the immediate wake region (as expected from the aerodynamic model).

Visualisations: `outputs/cfd_surrogate_experiment/uncertainty_<layout_id>.png`

---

## 8. Visualisations Generated

All in `outputs/cfd_surrogate_experiment/`:

| File pattern | Content |
|---|---|
| `triplet_unet_ensemble_mean_<lid>.png` | CFD reference vs ensemble mean vs absolute error |
| `triplet_unet_member0_<lid>.png` | CFD reference vs U-Net member 0 vs absolute error |
| `triplet_cnn_<lid>.png` | CFD reference vs CNN vs absolute error |
| `triplet_rf_broadcast_<lid>.png` | CFD reference vs RF broadcast vs absolute error |
| `scatter_rf_area_averaged.png` | RF scalar: predicted vs reference area-averaged speed |
| `training_curves_unet_member*.png` | Train/val loss and NRMSE for each U-Net member |
| `training_curves_cnn.png` | CNN training curves |
| `uncertainty_<lid>.png` | Ensemble mean + ensemble std (predictive disagreement) |

---

## 9. Artifacts and Paths

| Artifact | Path |
|---|---|
| Experiment manifest (JSON) | `configs/cfd_surrogate_experiment.json` |
| RF checkpoint | `checkpoints/cfd_surrogate_experiment/rf_baseline.joblib` |
| U-Net member 0 | `checkpoints/cfd_surrogate_experiment/compactunet_seed142/best_checkpoint.pt` |
| U-Net member 1 | `checkpoints/cfd_surrogate_experiment/compactunet_seed143/best_checkpoint.pt` |
| U-Net member 2 | `checkpoints/cfd_surrogate_experiment/compactunet_seed144/best_checkpoint.pt` |
| CNN checkpoint | `checkpoints/cfd_surrogate_experiment/compactcnn_seed242/best_checkpoint.pt` |
| Figures | `outputs/cfd_surrogate_experiment/*.png` |
| CFD case templates | `cfd/cases/cfd_experiment/case_0{1–5}_*/` |

---

## 10. Exact Reproduction Commands

```bash
# Activate environment
source .venv/bin/activate

# Re-run the full experiment (generates dataset, trains all models, evaluates, saves artifacts)
python _run_exp.py

# Or using the Python API directly:
from src.experiment.runner import ExperimentConfig, run_experiment
result = run_experiment(ExperimentConfig(seed=42, n_layouts=18, epochs=35, ensemble_size=3))

# Run only the new experiment tests
pytest tests/test_cfd_surrogate_experiment.py -v

# Run the full test suite (286 tests)
pytest -q

# Generate EDA visualisations for the UA0625_d00 dataset
python scripts/ua0625_eda.py

# Generate OpenFOAM case files (no solver execution)
python cfd/scripts/generate_cfd_cases.py
```

---

## 11. CFD Execution: BLOCKED

**Reason:** OpenFOAM (`simpleFoam`, `blockMesh`) is not available in this execution environment (macOS Apple Silicon, no native Linux ABI and no OpenFOAM binary/package installed). Docker is present but no OpenFOAM container image is available. This was confirmed at experiment start by `command -v simpleFoam` returning NOT FOUND.

**Consequence:** No CFD simulation was executed. All reference fields used for training are analytical, not CFD. Every metric and figure is "surrogate vs analytical reference", never "surrogate vs CFD".

**What remains to complete the original CFD objective:**

1. **Obtain an OpenFOAM environment:**
   - Recommended: `docker pull opencfd/openfoam-run:2312` (or install via [openfoam.com](https://www.openfoam.com))
   - Alternative: run on a Linux workstation/HPC with OpenFOAM sourced.

2. **Run the generated cases (quasi-2D, 5 parameter variations):**
   ```bash
   cd cfd/cases/cfd_experiment/case_01_h20_u5
   ./Allrun
   # repeat for case_02 … case_05
   ```
   Expected runtime per case: ~5 min on 4-core workstation.

3. **Limitation of quasi-2D cases:** The existing cases are vertical-plane (x-z) cases. The z = 1.75 m pedestrian slice from a 2D case is a 1D profile over x, **not** a 2D (x, y) horizontal field. Producing the 2D horizontal pedestrian target that the neural surrogate predicts (shape Ny × Nx) requires 3D cases using `snappyHexMesh` over a building array.

4. **Convert CFD output to the data contract:**
   ```bash
   python cfd/scripts/foam_to_contract.py \
     --input-file cfd/cases/cfd_experiment/case_01_h20_u5/<time>/postProcessing/surfaces/U_z1.75.raw \
     --output-npz data/processed/cfd_case_01.npz \
     --layout-id cfd_case_01_h20_u5 --nx 64 --ny 64
   ```

5. **Load CFD NPZ into the experiment pipeline** — the runner already supports `UrbanLayoutSample` objects from any source (analytical or CFD); replace `generate_reference_dataset()` with `LayoutDataLoader.load_directory("data/processed/")`.

---

## 12. Known Limitations

| Limitation | Evidence |
|---|---|
| Reference fields are analytical, not CFD | Every sample carries `city_or_source="ANALYTICAL_REFERENCE"` and `solver_name` containing `"Analytical"` |
| Only 3 test layouts | Dataset is 18 layouts (limited by analytical generation); larger dataset improves reliability of test metrics |
| NRMSE/R² is vs analytical reference | The 3.85% NRMSE_range and R²=0.89 should be re-evaluated against genuine CFD once available |
| Uncertainty is not calibrated | `method: ensemble_predictive_disagreement` not empirically verified against CFD |
| No CFD speedup claimed | No OpenFOAM timing measurement was made; timing comparison is impossible |
| CNN underperforms U-Net | At base_channels=16, 4 dilated layers, 35 epochs; more capacity or training would improve it |
| 2D cases only | The CFD case generator produces quasi-2D vertical-plane cases; 3D cases needed for 2D horizontal target |
