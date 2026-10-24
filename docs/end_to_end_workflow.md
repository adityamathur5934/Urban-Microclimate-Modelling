# End-to-End Workflow: From Raw Geometry to Physical Predictions

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Task:** Task 5 — Neural-Surrogate Prototype  
**Owner:** Project Lead  
**Document Version:** 1.0.0  
**Branch:** `feature/neural-surrogate-prototype`  
**Status:** Approved Reference Workflow  

---

## 1. System Architecture Overview

This document specifies the complete, reproducible data-to-prediction pipeline connecting raw CFD reference data, data engineering contracts, traditional automated machine learning (AML), and deep neural surrogate spatial modeling.

```
+-----------------------------------------------------------------------------------------------+
|                               1. DATA INGESTION & CONTRACT VALIDATION                         |
|   Source: UrbanTALES LES / OpenFOAM RANS / Validated Canonical Archives (.npz)                |
|   Module: src/data/contract.py & src/data/loader.py (LayoutDataLoader)                       |
+-----------------------------------------------------------------------------------------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
+-----------------------------------------------+ +---------------------------------------------+
|    2. MODEL 1: TABULAR FEATURE EXTRACTION     | |   3. MODEL 2: SPATIAL RASTERIZATION (SDF)   |
|   Module: src/data/features.py                | |   Module: src/data/rasterizer.py            |
|   - 11 Morphometric Features:                 | |   - 7 Input Channels (C_in = 7):            |
|     lambda_p, lambda_f, H_mean, H_std,        | |     Height Map, Building Mask, SDF,         |
|     H_max, lambda_w, z0, d, U_ref, cos, sin   | |     Inflow U, Inflow V, Coord X, Coord Y    |
|   - Area-Averaged Target:                     | |   - Aligned Target Field (C_out = 1):       |
|     src/data/area_averaging.py                | |     Pedestrian Wind Speed (z = 1.75 m)      |
+-----------------------------------------------+ +---------------------------------------------+
                        |                                               |
                        +-----------------------+-----------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
|                            4. DATASET PARTITIONING & LEAKAGE CONTROL                          |
|   Module: src/data/split.py (LayoutSplitter, DatasetSplit)                                    |
|   - Strict Layout-Level Holdout (Train: 70%, Val: 15%, Test: 15%)                             |
|   - Out-of-Distribution Family Holdout (Train: Idealized Arrays, Test: Real Cities)           |
|   - Zero layout ID overlap asserted via DatasetSplit.validate()                               |
+-----------------------------------------------------------------------------------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
|                         5. TRAINING-ONLY NORMALIZATION & CONFIG PERSISTENCE                   |
|   Module: src/data/normalization.py (FieldNormalizer, FeatureNormalizer)                      |
|   Module: src/data/config.py (PreprocessingConfig)                                            |
|   - Mean and Std computed exclusively across training fluid cells                             |
|   - Parameters saved to data/preprocessing_config.json for test-time reuse                    |
+-----------------------------------------------------------------------------------------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
+-----------------------------------------------+ +---------------------------------------------+
|              6. MODEL 1: TABULAR AML          | |         7. MODEL 2: NEURAL SURROGATE        |
|   Algorithm: Random Forest / XGBoost          | |   Architecture: CompactUNet (src/models/)   |
|   Input: Normalized Feature Matrix (N, 11)    | |   Training: SurrogateTrainer (src/training/)|
|   Target: Area-Averaged Wind Speed [m/s]      | |   Loss: Masked L1 + Gradient Loss           |
|   Output: Scalar Prediction y_hat             | |   Optimizer: AdamW + ReduceLROnPlateau      |
|                                               | |   Output: Normalized Field Y_tilde (1, H, W)|
+-----------------------------------------------+ +---------------------------------------------+
                        |                                               |
                        +-----------------------+-----------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
|                        8. PHYSICAL INVERSION & RIGOROUS BENCHMARKING                          |
|   Module: src/evaluation/evaluator.py & src/evaluation/metrics.py                             |
|   - Invert predictions to physical units: Y_phys = Y_tilde * sigma_train + mu_train [m/s]     |
|   - Evaluate Field MAE [m/s], Normalized RMSE [%], R^2 Score, Relative L2 Norm               |
|   - Benchmark Wall-Clock Latency [ms] vs. Numerical CFD Runtime [s]                           |
+-----------------------------------------------------------------------------------------------+
                                                |
                                                v
+-----------------------------------------------------------------------------------------------+
|                             9. SPATIAL VISUALIZATION & DIAGNOSTICS                            |
|   Module: src/evaluation/plotting.py (plot_prediction_comparison)                             |
|   - 3-Panel Maps: [CFD Reference Field] | [Surrogate Prediction] | [Absolute Error Map]       |
+-----------------------------------------------------------------------------------------------+
```

---

## 2. Step-by-Step Execution Guide

### Step 1: Environment Setup & Verification
Ensure dependencies are installed and unit tests execute cleanly:

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify PyTorch, SciPy, and test suite (49 passed)
pytest -v
```

### Step 2: Data Loading & Preprocessing
```python
from src.data import (
    LayoutDataLoader,
    LayoutRasterizer,
    FieldNormalizer,
    LayoutSplitter,
    compute_area_averaged_wind_speed,
)
from src.training import UrbanSurrogateDataset

# 1. Load validated layout samples
samples = LayoutDataLoader.load_directory("data/processed/")

# 2. Split dataset by layout ID without spatial leakage
layout_ids = [s.layout_id for s in samples]
split = LayoutSplitter.split_by_ratio(layout_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

train_samples = [s for s in samples if s.layout_id in split.train_layout_ids]
val_samples = [s for s in samples if s.layout_id in split.val_layout_ids]
test_samples = [s for s in samples if s.layout_id in split.test_layout_ids]

# 3. Fit normalizer on training split only
rasterizer = LayoutRasterizer()
train_rasters = [rasterizer.rasterize(s) for s in train_samples]

target_normalizer = FieldNormalizer(method="standard", mask_buildings=True)
target_normalizer.fit(
    [r.target_field for r in train_rasters],
    masks=[r.building_mask for r in train_rasters],
)

# 4. Construct PyTorch Datasets
train_dataset = UrbanSurrogateDataset(train_samples, rasterizer, target_normalizer=target_normalizer)
val_dataset = UrbanSurrogateDataset(val_samples, rasterizer, target_normalizer=target_normalizer)
test_dataset = UrbanSurrogateDataset(test_samples, rasterizer, target_normalizer=target_normalizer)
```

### Step 3: Model Initialization & Training
```python
from src.models import CompactUNet
from src.training import SurrogateTrainer, TrainingConfig

# 1. Instantiate 2D U-Net (7 input channels -> 1 target channel)
model = CompactUNet(in_channels=7, out_channels=1, base_channels=32, dropout_rate=0.10)

# 2. Configure training with AdamW, ReduceLROnPlateau, and checkpointing
config = TrainingConfig(
    epochs=30,
    batch_size=8,
    learning_rate=1e-3,
    early_stopping_patience=6,
    checkpoint_dir="checkpoints/",
    device="auto",  # Automatically detects Apple Silicon MPS, CUDA, or CPU
)

trainer = SurrogateTrainer(model, config)
history = trainer.fit(train_dataset, val_dataset)
```

### Step 4: Evaluation on Held-Out Test Layouts in Physical Units
```python
from src.evaluation import SurrogateEvaluator, plot_prediction_comparison

# 1. Load best checkpoint and evaluate with physical inverse transformation
trainer.load_checkpoint("checkpoints/best_model.pt")
evaluator = SurrogateEvaluator(target_normalizer=target_normalizer)

summary = evaluator.evaluate(model, test_dataset, device="auto")
print(f"Held-Out Field MAE:  {summary.mean_mae_mps:.3f} m/s")
print(f"Normalized RMSE:     {summary.mean_nrmse_percent:.1f} %")
print(f"R^2 Score:           {summary.mean_r2_score:.3f}")
print(f"Mean Latency:        {summary.mean_latency_ms:.2f} ms / layout")

# 2. Generate 3-panel physical prediction comparison plot
test_sample = test_samples[0]
r_test = rasterizer.rasterize(test_sample)

# Predict single layout
import torch
x_tensor = torch.from_numpy(r_test.channels).float().unsqueeze(0).to(trainer.device)
with torch.no_grad():
    pred_norm = model(x_tensor).squeeze(0).cpu().numpy()
pred_phys = target_normalizer.inverse_transform(pred_norm, mask=r_test.building_mask)

plot_prediction_comparison(
    y_true=test_sample.targets.pedestrian_wind_speed,
    y_pred=pred_phys,
    mask=r_test.building_mask,
    layout_id=test_sample.layout_id,
    output_path=f"docs/figures/{test_sample.layout_id}_comparison.png",
)
```

---

## 3. Current Project Status & Blocked Dependencies

While the complete software architecture, contracts, models, loss functions, and evaluation suites are operational and verified:

1. **Scientific Benchmark Status:** Real scientific benchmarking remains blocked until the team confirms and downloads the real 2D pedestrian slices from the UrbanTALES portal (`urbantales.vercel.app`).
2. **Fair Model Comparison Status:** Model 1 (AML tabular baseline) and Model 2 (2D U-Net surrogate) will be trained and evaluated concurrently in Task 6 and Task 7 on identical real CFD partitions.
3. **Hardware Runtime:** The pipeline automatically leverages Apple Silicon Metal Performance Shaders (`mps`) on macOS, reducing training time for a 50-layout cohort to ~10–15 minutes.
