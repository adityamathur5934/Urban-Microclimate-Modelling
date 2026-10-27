# Dynamic Scientific Research UI Integration Plan

**Urban Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)**
*Repository Integration Specification & Architectural Blueprint*

---

## 1. Executive Summary & Existing Repository Audit

### 1.1 Git Status & Branch Context
- **Active Working Branch:** `feature/dynamic-stitch-ui` branched from `origin/main` (commit `5e50931`).
- **Pre-existing Python Test Suite:** 139 passed tests across data contracts, geometry feature extractors (26 features), multi-channel rasterizer (7 channels), normalizers, splitters, baseline RF, neural U-Net/CNN, and evaluation metrics.
- **Pre-existing Frontend:** None existed in the repository prior to this implementation. No HTML, CSS, JavaScript, React, Vue, or Next.js code was present.
- **Pre-existing Web Backend:** No HTTP server (FastAPI, Flask, Starlette) existed in the repository prior to this implementation. All logic lived in modular Python library code under `src/` and CLI scripts under `scripts/` and `cfd/scripts/`.

### 1.2 Inspection of Existing Scientific Modules
1. **Data Contract & Ingestion (`src/data/contract.py`, `src/data/loader.py`):**
   - Dataclasses: `LayoutMetadata`, `GeometryData`, `EnvironmentalConditions`, `CFDSolverMetadata`, `TargetFields`, `UrbanLayoutSample`.
   - Layout Families: `IDEALIZED_ALIGNED`, `IDEALIZED_STAGGERED`, `REALISTIC_NEIGHBORHOOD`.
   - NPZ Archive loader and strict zero-leakage validator.
2. **Geometric & Environmental Feature Extraction (`src/data/features.py`):**
   - 26 morphometric features (plan area density $\lambda_p$, frontal area density $\lambda_f$, mean/std/max/p25/p75 building heights, wall area density $\lambda_w$, MacDonald roughness $z_0$, displacement height $d$, building count, footprint statistics, inflow conditions).
3. **Spatial Multi-Channel Rasterizer (`src/data/rasterizer.py`):**
   - Rasterizes geometries into 7 channels: height map, building mask, signed distance field (SDF), normalized grid coordinate offsets ($dx, dy$), and directional inflow velocity components ($u_{\text{in}}, v_{\text{in}}$).
4. **Machine Learning & Neural Surrogates (`src/models/`):**
   - **Baseline:** `RandomForestBaseline` mapping 26 tabular features to area-averaged pedestrian wind speed.
   - **Deep Surrogate:** `CompactUNet` & `CompactCNN` operating directly on $7 \times H \times W$ spatial rasters with outdoor building mask enforcement ($U = 0.0$ inside structures).
5. **CFD Reference Generation & Post-Processing (`cfd/scripts/`):**
   - `generate_canonical_reference.py`: Aerodynamic benchmark flow generator simulating upstream ABL, front stagnation zone, horseshoe vortex divergence, corner acceleration jets ($1.3 - 1.5\times$ speed-up), and leeward recirculating wake cavity ($\sim 4.5H$ downstream).
   - `foam_to_contract.py`: Bridge interpolating OpenFOAM cell data to structured Cartesian grids.
6. **Evaluation & Visualization (`src/training/metrics.py`, `src/evaluation/plotting.py`):**
   - Scientific metrics: Masked Field MAE, Field RMSE, Normalized RMSE (NRMSE), Peak Local Error ($|\Delta U|_{\text{max}}$), valid fluid cell fraction.
   - Matplotlib headless generator: 3-panel triplet figures `[CFD Reference] | [Surrogate Prediction] | [Absolute Error Field]`.

---

## 2. Visual Reference Analysis (Stitch Design)

The provided visual reference consists of two cohesive, high-density scientific interfaces:

### Screen A: Research Overview Dashboard
- **Top Navigation Bar:**
  - Workspace selector (`Microclimate Lab - CFD & Neural Surrogates`, Active Session).
  - Global Search bar (layouts, runs, models).
  - Backend status indicator (`Backend: Connected / Port 8000`, latency indicator).
  - User identity (`Dr. E. Vance`, Urban Physics Group).
- **Left Sidebar Navigation:**
  - `RESEARCH WORKSPACE`: Overview (active), Urban Layouts, CFD Simulations, Surrogate Models, Field Comparison, Experiments, Evaluation, Dataset.
  - `SYSTEM`: Documentation, Settings.
- **Header & Metric Cards:**
  - Title: *Urban Microclimate Overview*. Subtitle explaining CFD reference fields and neural surrogates.
  - Primary Action Buttons: `+ New Experiment`, `Explore Layouts`, `View Evaluation`.
  - 4 KPI Stat Cards:
    1. *Geometric Corpus:* Layout count (e.g., 128 geometries, 4 distinct families A–D).
    2. *Ground Truth Solvers:* Converged CFD cases (e.g., 96/128 RANS $k$-$\epsilon$ solutions, Cartesian cell resolution).
    3. *Neural Architecture Suite:* Available model checkpoints (U-Net, FNO, PINN-NS).
    4. *Zero-Leakage Benchmark:* Latest held-out layout test MAE (e.g., 0.142 m/s, strict Family D holdout).
- **Main Spatial Velocity Magnitude Field (|U|) Viewer:**
  - Active layout header (e.g., Layout #042 - Staggered Array).
  - Mode toggles: `[CFD Reference (OpenFOAM)]` | `[Surrogate Prediction (U-Net)]` | `[Absolute Residual (|ΔU|)]`.
  - Display options: Domain dimension tags (`200m × 200m`), `[Mask]`, `[Streamlines]`, `[Grid]`.
  - Interactive Floating Probe Card: Live probe coordinates ($X, Y$), CFD ground truth, surrogate prediction, absolute and relative residuals.
  - Continuous scientific colormap bar with physical units (m/s) and domain min/mean/max readouts.
- **Neural Surrogates & Generalization Benchmark Table:**
  - Table of recent experiments with Experiment ID, Model Architecture, Dataset Split (e.g., Family-D Held-out), Inflow Condition, Held-Out MAE, Max $|\Delta U|$, Status, and Action buttons.
  - Footer with dataset provenance, stack versions, and protocol disclaimers.

### Screen B: Field Comparison Studio
- **Header & Action Bar:**
  - Title: *Field Comparison Studio*, `[Validation Mode]`, Synchronized 1:1 indicator.
  - Action buttons: `Point Probe`, `Sample Transect`, `Download .npz`, `Export Triplet Figure`.
- **Control Bar:**
  - Domain selector: Active layout dropdown.
  - Surrogate model selector: Checkpoint/model dropdown.
  - Scalar field selector: Velocity Magnitude $|U|$ [m/s], Velocity $U_x$, Velocity $U_y$, TKE.
  - Range lock indicator, toggle switches (`Mask Solid`, `Sync Cursor`, `Grid`), live probe coordinate and values readout.
- **Synchronized Triplet Viewport (3-Panel):**
  - Panel 1: **CFD Reference** (OpenFOAM RANS, cell count, inflow condition, panel mean/peak/divergence).
  - Panel 2: **Neural Surrogate** (U-Net, inference latency, speedup factor vs CFD, cosine similarity).
  - Panel 3: **Spatial Error Residual** ($|\Delta U|$, peak error location, global MAE, peak local $\Delta$, $R^2$).
  - Synchronized cursor crosshair across all 3 panels simultaneously.
- **Analytical & Diagnostic Drawer:**
  - **Quantitative Error Matrix:** Field MAE, Field RMSE, Normalized RMSE (NRMSE), Peak Local $|\Delta U|$, Active fluid cells count/percentage, 95% Confidence Interval.
  - **Canyon Centerline Transect Chart:** Interactive line graph showing along-canyon transect (CFD Truth vs Surrogate Prediction) with building wall boundaries and divergence callouts.
  - **Scientific Protocol Notes:** High-frequency shear gradient attenuation, zero-leakage verification, physical simulation assumptions, reference DOI link.

---

## 3. Architecture & Technology Stack

### 3.1 Backend Service Layer (`src/api/`)
To satisfy the requirement that data is **genuinely dynamic and backend-connected**:
- **Framework:** `FastAPI` + `uvicorn` (asynchronous, high-performance, strictly typed with Pydantic).
- **Core Endpoints:**
  - `GET /api/overview`: Computes actual repository corpus metrics, solvers count, checkpoint inventory, and zero-leakage test MAE from real layout samples and models.
  - `GET /api/layouts`: Lists all available urban layout samples with family classifications, domain sizes, and physical parameters.
  - `GET /api/layouts/{layout_id}`: Retrieves full metadata, 26 geometric features, and environmental conditions.
  - `GET /api/layouts/{layout_id}/fields`: Returns 2D arrays (height map, building mask, CFD velocity components $u, v$, speed magnitude $|U|$, TKE) and Cartesian coordinates.
  - `POST /api/surrogates/predict`: Takes layout ID and model ID, runs inference with the real PyTorch `CompactUNet` or `RandomForestBaseline` model, and computes exact pixel-wise difference field, MAE, RMSE, NRMSE, and $R^2$.
  - `GET /api/layouts/{layout_id}/probe`: Returns exact interpolated CFD and surrogate values at any $(x, y)$ coordinate.
  - `GET /api/layouts/{layout_id}/transect`: Returns 1D slice profile data along a specified coordinate line ($Y = y_0$ or $X = x_0$).
  - `GET /api/experiments`: Returns experiment records with genuine metrics computed on held-out test splits.
  - `POST /api/experiments/run`: Triggers a training run or evaluation job asynchronously with live status tracking.
  - `GET /api/export/triplet/{layout_id}`: Generates and streams a high-resolution publication-quality PNG/SVG figure using `src.evaluation.plotting.plot_prediction_comparison`.
  - `GET /api/health`: Server status, loaded PyTorch device (`cpu`, `mps`, or `cuda`), and memory usage.

### 3.2 Frontend Architecture (`src/ui/` / Static Assets)
Following the web application development guidelines:
- **Core:** Clean semantic HTML5 and modular vanilla JavaScript (ES6+), avoiding heavy frontend framework bloat while providing reactive state management.
- **Styling (CSS):** Vanilla CSS design system with CSS custom properties (`--bg-primary`, `--accent-blue`, `--border-subtle`, `--surface-card`, etc.), strict typography (Inter / system font stack), zero TailwindCSS, matching the Stitch aesthetic faithfully.
- **Scientific Visualization:**
  - **Spatial 2D Field Rendering:** HTML5 Canvas / WebGL accelerated 2D continuous scientific scalar raster rendering with Viridis, Plasma, Magma, and Coolwarm color lookup tables (LUTs), crisp vector overlays for building geometry boundaries, stream tracer lines, and synchronized multi-panel crosshairs.
  - **Transect Line Charts:** Precision vector SVG rendering with exact physical units, building occlusion shading, and interactive hover tooltips.
- **State Management & Dynamic Polling:**
  - Central reactive state store managing active view (`overview` vs `studio`), selected layout ID, selected model, field type, probe coordinates, sync toggles, and polling intervals for ongoing background jobs.
  - Auto-refresh mechanisms for layout changes, experiment status transitions, and parameter updates.

---

## 4. Implementation Steps & Validation Protocol

1. **Step 1: Install backend dependencies (`fastapi`, `uvicorn`) in `.venv` and verify environment.**
2. **Step 2: Implement FastAPI Backend Service (`src/api/server.py`, `src/api/routes.py`, `src/api/schemas.py`):**
   - Direct integration with `src.data.loader`, `cfd.scripts.generate_canonical_reference`, `src.models.neural_unet`, `src.models.baseline_rf`, and `src.training.metrics`.
   - Comprehensive test coverage for all API endpoints (`tests/test_api_endpoints.py`).
3. **Step 3: Construct the Design System & Scientific UI (`src/ui/`):**
   - `index.html`: Structure covering both Research Overview Dashboard and Field Comparison Studio.
   - `styles.css`: Stitch-faithful design tokens, layout grids, cards, data tables, colorbar scales, floating tooltips.
   - `app.js`: State management, API client, view switching, synchronized multi-panel canvas rendering, transect plotting, probe interactions, and auto-refresh logic.
4. **Step 4: Real Model & CFD Data Seeding:**
   - Pre-generate canonical and synthetic layout cohorts spanning all 4 layout families (`idealized_aligned`, `idealized_staggered`, `realistic_neighborhood`, `staggered_canyon_block`).
   - Train and save actual model weights (`checkpoints/unet_best.pt`, `checkpoints/baseline_rf.joblib`) so inference is authentic.
5. **Step 5: End-to-End Verification:**
   - Verify all existing 139 tests + new API tests pass.
   - Start the server and use headless browser / curl testing to verify live responses, field generation, transect queries, and UI rendering.
   - Capture screenshots of the running application for documentation and user verification.
