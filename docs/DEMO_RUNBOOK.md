# Urban Microclimate Modelling — Demonstration Runbook

This runbook provides a complete, repeatable, step-by-step sequence to launch, demonstrate, and defend the Urban Microclimate Modelling application during a live academic presentation or viva examination.

---

## 1. Prerequisites and Environment Verification

Ensure your environment is ready before starting the viva session:

```bash
# 1. Navigate to the repository root
cd "/Users/aditya.2024/Documents/Projects/Urban Microclimate Modelling"

# 2. Verify virtual environment and python version (3.11+)
./.venv/bin/python --version

# 3. Verify PyTorch device and test suite
./.venv/bin/pytest tests/ -q
```
*Expected test suite output:* `207 passed, 1 warning in ~25s`.

---

## 2. Launching the Application

Start the FastAPI backend with the static UI mount:

```bash
# Option A: Python module execution (recommended)
./.venv/bin/python -m src.api.server

# Option B: Direct uvicorn server invocation
./.venv/bin/uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

Once running, access the services:
- **Interactive Web Application:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive OpenAPI Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Direct Health & Hardware Endpoint:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 3. Step-by-Step Live Demonstration Script

Follow this 10-step sequence during the live presentation:

### Step 1: Open the Application & Check the Provenance Banner
- Navigate to `http://127.0.0.1:8000/` in Chrome or Safari.
- Point out the amber **Data Provenance & Transparency Banner** at the top:
  > *"Synthetic Aerodynamic Benchmark (Castro & Robins 1977 / COST 732). Real OpenFOAM cases located in `cfd/cases/simple2d_building_block`. Predictions computed live with PyTorch."*
- **What to say to the examiner:**
  > *"We enforce strict scientific data provenance. The system clearly distinguishes synthetic procedural test fixtures and analytical aerodynamic benchmark fields from real OpenFOAM solver runs. No synthetic field is ever dishonestly labelled as CFD ground truth (`has_cfd_ground_truth=False`)."*

### Step 2: System Overview & KPI Cards
- In the top KPI section, highlight the four metric cards:
  1. **Geometric Corpus:** 25 building geometries (3 topological families: Aligned Canyon, Staggered Canyon, Dense Courtyard).
  2. **Ground Truth Solvers:** 0 executed OpenFOAM solutions / 25 analytical & procedural benchmarks (honest reporting of solver status).
  3. **Neural Architecture Suite:** 3 implemented model definitions (`CompactUNet`, `CompactCNN`, `RandomForestBaseline`).
  4. **Zero-Leakage Benchmark:** Held-Out Split MAE reported honestly as *Unavailable* (since no trained checkpoint exists for real CFD data, rather than showing a fabricated 0.142 m/s MAE).

### Step 3: Layout Catalog & Morphometric Inspection
- Switch to the **Layout Catalog** view (or scroll to the catalog grid).
- Select **Layout #042: Staggered Street Canyon** (`layout_042_staggered_canyon`).
- Inspect the geometric summary features:
  - Domain size: $200\,\text{m} \times 200\,\text{m} \times 80\,\text{m}$ ($128 \times 128$ Cartesian grid).
  - Building count: 4 structures (B1 through B4, heights 12m to 45m).
  - Plan area fraction: $\lambda_p \approx 0.14$.
  - Inflow wind: $5.0\,\text{m/s}$ streamwise ($0^\circ$).

### Step 4: Spatial Field Studio (Reference Flow Field)
- Switch to the **Comparison Studio** tab.
- Panel 1 shows the **Aerodynamic Benchmark Reference Field** ($U_{1.75\text{m}}$):
  - Notice the stagnation zones upstream of tall buildings B3 and B4.
  - Notice the low-velocity wake cavity recirculations downstream.
  - Notice the high-velocity channeling jet in the narrow canyon between B1 and B3.
  - Toggle **Streamlines** to show 2D velocity vector trajectories advecting around solid buildings.

### Step 5: Live Neural Surrogate Prediction (U-Net)
- Select Model: `CompactUNet (ResBlock-4)` (`unet-v2-resblock4`).
- Click **Run Live Inference**.
- Observe Panel 2 (**Surrogate Field**) and Panel 3 (**Absolute Residual Error Map**):
  - Live PyTorch forward pass runs on device (`mps` / `cuda` / `cpu`).
  - Strict solid masking enforces $U = 0.0\,\text{m/s}$ inside all solid building cells.
  - Metrics are computed over valid fluid cells only: Field MAE, RMSE, NRMSE, Peak Local Error, and $R^2$.
- **What to say to the examiner:**
  - *"The checkpoint `checkpoints/unet_best.pt` contains initial random weights, so the resulting MAE reflects an untrained neural architecture. We do not invent performance; the system accurately reports `is_trained_checkpoint=False`."*

### Step 6: Model Comparison (CNN vs Random Forest Baseline)
- Change model selector to `Dilated-CNN-Surrogate-v1` (`compact-cnn-dilated`):
  - Re-run inference; observe the receptive-field response of the 4 dilated 2D convolutional layers.
- Change model selector to `Random Forest Morphometric Baseline` (`rf-baseline-morph`):
  - Observe how the Random Forest takes the 26-dimensional morphometric vector and predicts an area-averaged scalar wind speed, which is broadcast across fluid cells.
  - Point out that the RF baseline cannot resolve spatial recirculations, illustrating the fundamental need for spatial convolutional surrogates.

### Step 7: Point Probe Query (Interactive Flow Inspection)
- Click anywhere in the canyon channel on the reference field canvas (e.g., $X = 52.4\,\text{m}, Y = 48.1\,\text{m}$).
- The **Point Probe Inspector** dynamically queries `/api/layouts/{id}/probe`:
  - Physical coordinates: $X, Y$.
  - Cell status: *Valid Fluid Cell* or *Masked Solid Cell (U = 0.0 m/s)*.
  - Reference speed: e.g. $4.82\,\text{m/s}$.
  - Surrogate speed: e.g. $2.14\,\text{m/s}$.
  - Absolute residual: e.g. $2.68\,\text{m/s}$.

### Step 8: Canyon Centerline 1D Transect Cut
- View the **Transect Profile** chart below the studio:
  - Axis: $Y = 50.0\,\text{m}$ (horizontal cut through the street canyon).
  - Blue line: Analytical reference velocity profile.
  - Purple dashed line: Neural surrogate prediction.
  - Shaded grey bars: Solid building cross-sections where velocity is physically forced to zero.
  - Dynamic metrics: Max divergence and Canyon jet match percentage.

### Step 9: Launch a Background Experiment
- Navigate to the **Model Benchmarks** view.
- Click **+ New Experiment**.
- Configure:
  - Architecture: `CompactUNet` or `CompactCNN`.
  - Epochs: `3` (for live demo brevity).
  - Learning Rate: `0.001`.
  - Holdout Family: `idealized_staggered`.
- Click **Launch Training Run**:
  - The API returns HTTP 200 with an experiment ID (e.g. `EXP-2026-004`).
  - The UI background poller detects active training: `<span class="badge badge-amber pulse">Training (1/3)</span>`.
  - Real `NeuralSurrogateTrainer` trains with `torch.optim.AdamW` and `CombinedSurrogateLoss` on background thread.
  - Upon completion, real validation MAE is calculated and displayed without fabricated constants.

### Step 10: Publication Figure Export
- Click **Export Triplet PNG** button in the Studio or visit:
  `http://127.0.0.1:8000/api/export/triplet/layout_042_staggered_canyon?model_id=unet-v2-resblock4`
- The system renders a 300 DPI, 3-panel figure (Ground Truth, Surrogate Prediction, Absolute Error Map) using `src.evaluation.plotting.plot_prediction_comparison` and downloads it directly.

---

## 4. Fallback Demonstration Strategy

If an unexpected environmental issue occurs during the viva:

| Issue | Cause | Fallback Action |
|-------|-------|-----------------|
| Port 8000 already in use | Stale process | Run on port 8080: `./.venv/bin/uvicorn src.api.server:app --port 8080` |
| PyTorch device error | MPS device mismatch | The service auto-falls back to CPU; verify with `/api/health` |
| UI browser cache stale | Browser caching old JS | Hard refresh browser: `Cmd + Shift + R` (Mac) or `Ctrl + Shift + R` |
| Live training takes too long | High epoch count requested | Request 1 or 2 epochs for quick demonstration |
| OpenFOAM not installed | Host system lacks native binaries | Show `cfd/cases/simple2d_building_block` template and explain containerized execution (`docker run -v $(pwd)/cfd:/case openfoam/openfoam9-paraview56`) |

---

## 5. Summary of Truthful Statements for the Viva

1. **Surrogate Architecture:** We implemented both a multi-scale U-Net (`CompactUNet`, 4-level encoder-decoder with skip connections) and a dilated CNN (`CompactCNN`, receptive field 31 cells), taking 7-channel raster inputs (height, mask, SDF, inflow $U$, inflow $V$, $X$-grid, $Y$-grid).
2. **Baseline Model:** We implemented a Scikit-learn `RandomForestBaseline` with 26 morphometric features (plan area fraction, frontal area index, building height statistics, distance to inflow).
3. **Data Provenance:** The 25 current layouts consist of 24 procedural geometries and 1 canonical aerodynamic reference field (Castro & Robins 1977). OpenFOAM case templates are built in `cfd/cases/simple2d_building_block/` ready for LES/RANS meshing.
4. **Leakage Prevention:** Normalization and holdout splitting are strictly partitioned at the complete layout or family level. Zero pixel-level autocorrelation leaks across train and test partitions.

