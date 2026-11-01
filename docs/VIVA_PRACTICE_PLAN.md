# Viva Practice Plan: Urban Microclimate Modelling

This hands-on preparation plan guides you through 7 structured technical exercises to ensure you thoroughly understand the mathematical formulations, architecture, code implementation, and scientific limitations of the project before facing examiners.

---

## Exercise 1: Repository Architecture & Data Flow Walkthrough

### Objective
Navigate the codebase confidently, explain the role of each directory, and trace the lifecycle of a layout from geometry to evaluation.

### Tasks to Perform
1. Open the repository root in your terminal or editor.
2. Trace the data contract in `src/data/contract.py`:
   - Inspect `LayoutMetadata`, `GeometryData`, `EnvironmentalConditions`, `CFDSolverMetadata`, and `TargetFields`.
   - Identify how shape mismatch, negative heights, and non-finite values are caught.
3. Trace the feature extractor in `src/data/features.py`:
   - Look at `extract_features()`: How are the 26 morphometric features (e.g., `lambda_p`, `lambda_f`, `mean_height`, `std_height`) computed?
4. Trace the multi-channel rasterizer in `src/data/rasterizer.py`:
   - Look at `rasterize()`: Identify the 7 input channels:
     1. Building height map $H(x, y)$
     2. Building binary mask $M(x, y)$
     3. Signed Distance Field (SDF) $\phi(x, y)$ via Euclidean Distance Transform
     4. Inflow streamwise velocity $U_{\text{in}}(x, y)$
     5. Inflow spanwise velocity $V_{\text{in}}(x, y)$
     6. Normalized Cartesian coordinate $X/L_x$
     7. Normalized Cartesian coordinate $Y/L_y$

### Expected Viva Mastery
- **You should be able to explain without notes:**
  > *"A layout is represented in two parallel ways: as a 26-dimensional morphometric summary vector for the Random Forest baseline, and as a 7-channel $128 \times 128$ spatial tensor for the neural surrogate. The Signed Distance Field provides smooth geometric boundary gradient information to convolutional filters."*

---

## Exercise 2: Backend Startup and API Endpoint Verification

### Objective
Start the backend server, execute live HTTP calls, and verify responses and error handling.

### Tasks to Perform
1. Start the server:
   ```bash
   ./.venv/bin/python -m src.api.server
   ```
2. In a second terminal, query key endpoints using `curl` or `python`:
   ```bash
   # Check hardware and corpus count
   curl -s http://127.0.0.1:8000/api/health | jq .

   # Check provenance inventory
   curl -s http://127.0.0.1:8000/api/inventory/datasets | jq '{total_loaded, canonical_benchmark_count, cfd_solver_output_count}'

   # Check checkpoint training status
   curl -s http://127.0.0.1:8000/api/inventory/checkpoints | jq .
   ```
3. Test deliberate error handling:
   ```bash
   # Unknown model ID returns HTTP 404 (not silent fallback)
   curl -s -w "\nHTTP Status: %{http_code}\n" -X POST http://127.0.0.1:8000/api/surrogates/predict \
     -H "Content-Type: application/json" \
     -d '{"layout_id": "layout_042_staggered_canyon", "model_id": "unknown-model-xyz"}'

   # Unsupported scalar field returns HTTP 400
   curl -s -w "\nHTTP Status: %{http_code}\n" -X POST http://127.0.0.1:8000/api/surrogates/predict \
     -H "Content-Type: application/json" \
     -d '{"layout_id": "layout_042_staggered_canyon", "model_id": "unet-v2-resblock4", "scalar_field": "temperature"}'
   ```

### Expected Viva Mastery
- **You should be able to explain without notes:**
  > *"The API strictly validates client inputs. An unknown model ID returns HTTP 404 rather than silently defaulting to U-Net. Requesting temperature returns HTTP 400 because UrbanTALES and our aerodynamic models are strictly isothermal, predicting pedestrian wind speed $U_{1.75\text{m}}$."*

---

## Exercise 3: Neural Surrogate Inference vs. Baseline Walkthrough

### Objective
Understand the exact forward pass mechanism, outdoor building masking, and baseline prediction logic.

### Tasks to Perform
1. Inspect `src/models/neural_unet.py`:
   - Identify `CompactUNet`: 4-level encoder-decoder with residual blocks and skip connections.
   - Look at `forward()`: Notice how the output is masked: `out = out * (~building_mask)`.
2. Inspect `src/models/baseline_rf.py`:
   - Look at `predict_layout()`: Notice that the Random Forest outputs a single scalar $\hat{U}_{\text{area}}$ (area-averaged wind speed) and broadcasts it across fluid cells.
3. Compare the predictions live via Python:
   ```bash
   ./.venv/bin/python -c "
   from src.api.service import service
   unet_res = service.predict_surrogate('layout_042_staggered_canyon', 'unet-v2-resblock4')
   rf_res = service.predict_surrogate('layout_042_staggered_canyon', 'rf-baseline-morph')
   print('U-Net MAE:', unet_res['field_mae'], 'Type:', unet_res['prediction_type'])
   print('RF Baseline MAE:', rf_res['field_mae'], 'Type:', rf_res['prediction_type'])
   "
   ```

### Expected Viva Mastery
- **You should be able to explain without notes:**
  > *"The Random Forest is a tabular baseline predicting an area-averaged scalar value from 26 morphometric features. It cannot resolve local recirculations or wakes. The U-Net is a spatially resolved neural surrogate predicting cell-by-cell velocities directly from 2D spatial context."*

---

## Exercise 4: Dissecting an Evaluation Plot (Ground Truth, Prediction, Error)

### Objective
Interpret 3-panel microclimate comparison plots, understand colorbar scales, and explain error patterns.

### Tasks to Perform
1. Generate a publication triplet figure:
   ```bash
   ./.venv/bin/python -c "
   from src.api.service import service
   fig = service.export_triplet_figure('layout_042_staggered_canyon', 'unet-v2-resblock4')
   print('Figure exported to:', fig)
   "
   ```
2. Open the image in `outputs/figures/triplet_layout_042_staggered_canyon_unet-v2-resblock4.png`.
3. Analyze the three panels:
   - **Panel 1 (Reference Ground Truth):** Inflow boundary on the left ($5.0\,\text{m/s}$). Low-velocity wake cavity downstream of building blocks. High-velocity jet in the narrow canyon between B1 and B3. Solid buildings rendered in dark slate grey ($U = 0.0\,\text{m/s}$).
   - **Panel 2 (Surrogate Prediction):** Spatial velocity field output by the neural surrogate.
   - **Panel 3 (Absolute Error Map $|y - \hat{y}|$):** Highlight where errors concentrate (shear layers, building corners, stagnation points).

### Expected Viva Mastery
- **You should be able to explain without notes:**
  > *"Errors in neural surrogates naturally peak in regions of high spatial velocity gradients $\nabla U$, specifically the separated shear layers shedding from sharp building corners and the stagnation points on upstream facades. Bulk flow far from buildings has lower error."*

---

## Exercise 5: Mock Viva Question Sequence

Conduct this self-interview without consulting notes:

1. **Q1: Why not just run OpenFOAM CFD simulations whenever a prediction is needed?**
   - *Model Answer:* OpenFOAM 3D steady RANS takes 15–45 minutes per layout on multi-core workstations, and transient Large-Eddy Simulations (LES) take 6–24 hours per case. In iterative architectural urban design, thousands of configurations must be evaluated. Neural surrogates evaluate in milliseconds ($<20\,\text{ms}$ on GPU), enabling real-time interactive design exploration.

2. **Q2: Why is pixel-level random splitting invalid in spatial microclimate modelling?**
   - *Model Answer:* Adjacent grid cells within the same urban canyon have high spatial autocorrelation governed by Navier-Stokes continuity. If cell $(i, j)$ is in the training set and cell $(i+1, j)$ is in the test set, the model merely interpolates neighboring pixels rather than learning fluid physics. We partition strictly by entire layouts or layout families (zero spatial leakage).

3. **Q3: What does the Signed Distance Field (SDF) channel add that the binary building mask does not?**
   - *Model Answer:* A binary building mask has discontinuous derivative transitions ($0 \to 1$). Standard convolutional filters with small kernels ($3 \times 3$) only detect the boundary when directly overlapping it. The SDF provides smooth Euclidean distance gradients throughout the fluid domain, informing every fluid cell of its exact distance and orientation to the nearest solid obstacle.

4. **Q4: Why does the Random Forest baseline fail to predict urban street canyon jets?**
   - *Model Answer:* The Random Forest operates on aggregated scalar features (e.g., plan area fraction $\lambda_p = 0.14$). It has no mechanism to resolve local street orientations, canyon aspect ratios ($H/W$), or spatial channeling. It can only predict a domain-wide scalar average.

5. **Q5: Can an untrained neural checkpoint be used to claim surrogate accuracy?**
   - *Model Answer:* Absolutely not. An untrained model's weights are randomly initialized. Any reported metrics merely measure the distance between random initial outputs and the reference field. We explicitly report `is_trained=False` for initial weights to preserve scientific integrity.

---

## Exercise 6: Debugging & Failure Handling Exercise

### Objective
Understand how the codebase responds to missing files, broken inputs, and hardware fallback.

### Tasks to Perform
1. Run the data validation hardening tests:
   ```bash
   ./.venv/bin/pytest tests/test_validation_hardening.py -v
   ```
2. Verify:
   - What happens if building heights are negative? $\to$ `PhysicalConstraintValidationError`
   - What happens if the grid shape does not match resolution $\times$ domain size? $\to$ `CoordinateGridValidationError`
   - What happens if NaN or Inf values appear in wind fields? $\to$ `NonFiniteValueError`
3. Verify PyTorch device fallback in `src/api/service.py`:
   - Checks CUDA first $\to$ MPS (Apple Silicon) second $\to$ CPU fallback.

### Expected Viva Mastery
- **You should be able to explain without notes:**
  > *"The scientific data layer implements an actionable exception hierarchy inheriting from `DataValidationError`. Corrupted CFD exports, dimension mismatches, and negative physical quantities are rejected immediately before entering training or inference pipelines."*

---

## Exercise 7: Final 10-Minute Demonstration Rehearsal

### Objective
Rehearse the complete viva demonstration using the exact script in `docs/DEMO_RUNBOOK.md`.

### Timing Breakdown
- **0:00 – 1:30:** Motivation, problem statement, and strict data provenance.
- **1:30 – 3:30:** Architecture overview (Random Forest baseline vs. U-Net surrogate).
- **3:30 – 6:00:** Live application demo: layout selection, flow field inspection, U-Net inference, probe, and transect.
- **6:00 – 8:00:** Background experiment training run and publication triplet export.
- **8:00 – 10:00:** Scientific limitations, blocked dependencies (UrbanTALES license / native OpenFOAM binaries), and future work.

