# Execution State — Urban Microclimate Modelling

_Last updated: October 28, 2026 (Final Project Audit & Demo Readiness)_

---

## 1. Current Project Status

| Phase / Deliverable | Status | Notes |
|---------------------|--------|-------|
| **Priority A — Demo-Ready Application** | ✅ Verified Working | Live startup, UI, probes, transects, export verified |
| **Priority B — Scientific Investigation** | ✅ Verified & Hardened | Measured MAE/RMSE comparisons; honest in-sample vs held-out reporting |
| **Priority C — Project Report (DOCX & PDF)** | ✅ Generated & Formatted | `deliverables/Urban_Microclimate_Modelling_Project_Report.docx` (680 KB) & `.pdf` (840 KB) |
| **Priority D — Project Presentation (PPTX)** | ✅ Generated & Formatted | `deliverables/Urban_Microclimate_Modelling_Project_Presentation.pptx` (705 KB, 16 slides, speaker notes) |
| **Priority E — Viva Voce Defense Guide** | ✅ Generated & Formatted | `deliverables/Urban_Microclimate_Modelling_Viva_Guide.docx` (55 KB) & `.pdf` (7.4 KB) |
| **Priority F — Demo Runbook & Practice Plan** | ✅ Complete & Verified | `docs/DEMO_RUNBOOK.md` & `docs/VIVA_PRACTICE_PLAN.md` |
| **Automated Test Suite** | ✅ 100% Passing | **207 passed, 0 failed, 1 warning in 30.55s** |
| **Repository Cleanup** | ✅ Safe Cleanup Complete | Disposable caches, empty test dirs, and obsolete outputs removed |
| **External CFD / Dataset Access** | ⛔ Documented Limitations | UrbanTALES license required; native OpenFOAM not installed on macOS |

---

## 2. Verified Startup & Tested Application Workflows

### Startup Commands
```bash
# Option A: Module execution (recommended)
./.venv/bin/python -m src.api.server

# Option B: Direct uvicorn execution
./.venv/bin/uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```
- **Web UI URL:** `http://127.0.0.1:8000/`
- **Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **Health Endpoint:** `http://127.0.0.1:8000/api/health`

### Features Verified Live
1. **Health & Hardware Detection:** Returns `status: "healthy"`, device (`mps` / `cuda` / `cpu`), `corpus_size: 25`, `models_count: 3`.
2. **System Overview:** Displays honest provenance banner, 25 layout count, 0 executed OpenFOAM solutions, zero-leakage benchmark reported as *Unavailable* (no false MAE).
3. **Layout Catalog & Fields:** 25 layouts with geometric summary features, domain bounds, and $128 \times 128$ Cartesian grid arrays with physical outdoor masking ($U = 0.0$ in solid buildings).
4. **Surrogate Predictions (3 Models):**
   - `unet-v2-resblock4` (`CompactUNet`): Real PyTorch forward pass, MAE = 2.608 m/s, `is_trained=False`.
   - `compact-cnn-dilated` (`CompactCNN`): Real PyTorch forward pass with dilated convolutions, MAE = 2.909 m/s, `is_trained=False`.
   - `rf-baseline-morph` (`RandomForestBaseline`): 26D morphometric vector prediction, scalar broadcast across fluid cells, MAE = 1.127 m/s.
5. **Interactive Point Probes (`/api/layouts/{id}/probe`):** Continuous $(x, y)$ coordinate interpolation, reporting fluid vs solid status, reference velocity, surrogate velocity, and absolute residual.
6. **1D Canyon Transect Cuts (`/api/layouts/{id}/transect`):** Extracts 200 points along coordinate planes (e.g. $Y = 50.0\,\text{m}$), identifying building intervals, maximum divergence, and canyon jet matching.
7. **Inventory Endpoints:**
   - `/api/inventory/datasets`: 25 loaded layouts (24 procedural, 1 canonical reference), all `has_cfd_ground_truth=False`.
   - `/api/inventory/checkpoints`: Scans disk for training metadata; accurately flags untrained weights.
   - `/api/inventory/cfd`: Reports OpenFOAM solver binary availability and template case status.
8. **Asynchronous Experiment Execution:** Real background thread training using `NeuralSurrogateTrainer`, updating training progress and calculating genuine completion metrics.
9. **Publication Triplet Export (`/api/export/triplet/{id}`):** Generates and downloads 300 DPI 3-panel comparison figures with Viridis/Magma colorbars.

---

## 3. Confirmed Model & Checkpoint Status

- **`checkpoints/unet_best.pt`:** Contains raw PyTorch state dict weights without training epoch metadata. Correctly flagged as `is_trained=False`.
- **`checkpoints/baseline_rf.joblib`:** Fitted on the in-memory cohort of 25 layouts. Flagged as `is_trained=True` (`fitted=True`).
  - *Critical Audit Finding:* **Layout 042 was NOT held out from the default loaded Random Forest baseline checkpoint** (it was part of the in-memory training cohort when `baseline_rf.joblib` was created). This in-sample training fit directly explains why the Random Forest achieves a lower numeric MAE ($1.127\,\text{m/s}$) than held-out models!
- **`compact-cnn-dilated`:** Instantiated dynamically in evaluation mode (`torch.no_grad()`), `is_trained=False`.

---

## 4. Dataset Provenance & CFD Status

- **Loaded Data:** 24 procedural geometries + 1 canonical aerodynamic reference (`cfd_canonical_block_01`, Castro & Robins 1977 analytical formulation on $64 \times 128$ grid).
- **Ground Truth Flag:** `has_cfd_ground_truth = False` across all layout summaries, detail records, and spatial fields.
- **CFD Solver Availability:** Native `simpleFoam` and `blockMesh` binaries are not installed in PATH on macOS. Case templates are verified in `cfd/cases/simple2d_building_block/` for Docker container execution.
- **External Dataset Status:** UrbanTALES (538 LES layouts) requires an institutional access agreement; not downloaded offline.

---

## 5. Safe Repository Cleanup Log

### Items Removed (SAFE TO REMOVE)
1. **`.DS_Store`:** macOS filesystem metadata file (disposable OS artefact).
2. **`checkpoints/exp_EXP-2026-004/`, `005/`, `006/`:** Empty test directories created dynamically during `test_p1_inventory.py::TestRealTrainingThread`.
3. **`outputs/figures/test_triplet.png`:** Obsolete pre-audit test image from Oct 27 (superseded by active model triplet figures).

### Items Deliberately Preserved (KEEP)
- **Source Code (`src/`):** All modules (`api`, `data`, `evaluation`, `models`, `training`, `ui`).
- **Tests (`tests/`):** All 207 automated unit, integration, and integrity tests.
- **CFD Assets (`cfd/`):** OpenFOAM case template, canonical reference NPZ, and bridge scripts.
- **Checkpoints (`checkpoints/`):** `baseline_rf.joblib` and `unet_best.pt`.
- **Deliverables (`deliverables/`):** Report (DOCX/PDF), Presentation (PPTX), and Viva Guide (DOCX/PDF).
- **Documentation (`docs/`):** All 30 engineering and audit reports, including runbooks and plans.
- **Scripts (`scripts/`):** All report, presentation, diagram, and baseline CLI scripts.

### Items for Manual Review (REVIEW MANUALLY)
- `docs/frontend-integration-plan.md`: Historical planning document (retained for project history).
- `outputs/plots/extent_test_triplet.png`: Ignored test output generated by `test_visualization.py`.

---

## 6. Latest Test Suite Execution

```
Command: ./.venv/bin/pytest tests/ -v
Result:  collected 207 items
         207 passed, 0 failed, 0 skipped, 1 warning in 30.55s
```
Warnings: 1 StarletteDeprecationWarning regarding httpx in FastAPI testclient (harmless third-party deprecation notice).

---

## 7. Remaining Items for Final Submission & Viva

1. Personalize placeholder details on Title Pages of Report, PPTX, and Viva Guide:
   - `[Student Name]` $\to$ Your Full Name
   - `[Roll Number]` $\to$ Your Student ID
   - `[Department Name]` $\to$ Your Academic Department
   - `[Institution Name]` $\to$ Your University / College
   - `[Supervisor Name]` $\to$ Your Faculty Advisor
2. Practice the 10-step demonstration using [`docs/DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md).
3. Review the top 10 defense questions in [`docs/VIVA_PRACTICE_PLAN.md`](VIVA_PRACTICE_PLAN.md).
