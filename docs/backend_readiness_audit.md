# Backend Readiness Audit — Urban Microclimate Modelling

**Audit date:** 1 November 2026  
**Git revision:** `main` — up to date with remote (`git pull`: "Already up to date.")  
**Python env:** `.venv` — Python 3.13.3, pytest 9.1.1, torch (MPS), FastAPI  
**Scope:** (1) Full backend readiness audit. (2) EDA and preprocessing evidence for the presentation.  
**Constraint:** No application code was modified. No models were retrained. Findings only.

---

## Overall Verdict

The repository contains a **working FastAPI service** (all 207 tests pass) backed by a **sound Python computation library**. The backend is **not** ready as a scientifically honest production service because:

- The in-memory corpus is **24 procedurally synthesised layouts + 1 analytical reference** (25 total), not UrbanTALES.
- **0 OpenFOAM runs** have been executed on this machine. The CFD case is an unexecuted template.
- The U-Net checkpoint is **randomly initialised** (all `num_batches_tracked = 0`). No training history is stored.
- The Random Forest checkpoint **is** fitted (on the 24 synthetic layouts), but that is not CFD-quality ground truth.
- Several UI metrics are hardcoded, bound to wrong JSON keys, or fabricated.

The provenance banners in the API already advertise `is_demonstration_fixture: true`. This audit confirms and documents the exact gaps.

---

## Part 1 — Backend Readiness Audit

### 1. Existing backend components

#### 1.1 Computation library (Python modules)

| Component | Location | Status |
|---|---|---|
| Data contract & validation | `src/data/contract.py` | Implemented & tested |
| NPZ load / save / archive check | `src/data/loader.py` | Implemented & tested; `data/processed/` is **empty** |
| Geometric features (26-D) | `src/data/features.py` | Implemented & tested |
| Rasterizer (7-channel) | `src/data/rasterizer.py` | Implemented & tested; channels: `height_map_m`, `building_mask`, `signed_distance_m`, `inflow_u_mps`, `inflow_v_mps`, `coord_x_norm`, `coord_y_norm` |
| Normalization (field + feature) | `src/data/normalization.py` | Implemented & tested; leakage guards present |
| Layout / family split | `src/data/split.py` | Implemented & tested; ratio + whole-family holdout |
| Area averaging | `src/data/area_averaging.py` | Implemented & tested |
| Random Forest baseline | `src/models/baseline_rf.py` | Implemented & tested; fitted checkpoint exists |
| CompactUNet / CompactCNN | `src/models/neural_unet.py`, `src/models/unet.py` | Implemented; **UNet checkpoint is randomly initialised** |
| Training loop + losses | `src/training/trainer.py`, `losses.py`, `dataset.py` | Implemented; smoke-test only, no production run |
| Evaluation metrics | `src/evaluation/metrics.py` | Implemented & tested |
| Error-map / triplet plots | `src/evaluation/plotting.py`, `src/training/visualize.py` | Implemented; trip let PNGs generated on disk |
| CFD case template | `cfd/cases/simple2d_building_block/` | **Template only — never executed** |
| Foam → contract bridge | `cfd/scripts/foam_to_contract.py` | Implemented; demo path tested; no real Foam samples |
| Analytical reference | `cfd/scripts/generate_canonical_reference.py` | Analytical Castro-style wake — not OpenFOAM |
| Preprocessing config persistence | `src/data/config.py` | Implemented; not used by the API |

#### 1.2 HTTP service

- **Framework:** FastAPI + uvicorn (`fastapi>=0.110.0`, `uvicorn>=0.28.0`, `httpx>=0.27.0`)
- **App:** `src/api/server.py`, version `1.4.0`
- **Router:** `src/api/routes.py`, prefix `/api`
- **Service singleton:** `src/api/service.py` (`MicroclimateService`)
- **Schemas:** `src/api/schemas.py` (all Pydantic v2)
- **Frontend:** `src/ui/index.html` + `app.js` + `styles.css`, mounted at `/` and `/static`
- **Start command (not in README):** `python -m src.api.server` (binds `127.0.0.1:8000`)

---

### 2. Computation layer verification

**Test run (1 Nov 2026):**
```
python -m pytest tests/ -v --tb=short
207 passed, 1 warning in 11.19 s
```
One warning: `starlette.testclient` / httpx deprecation (non-breaking).

These tests prove **software contracts on synthetic fixtures**, not scientific model performance.

| Capability | Library works? | Evidence | Scientific performance? |
|---|---|---|---|
| Dataset loading & validation | ✅ | `test_loader.py`, `test_contract_and_validation.py`, `test_validation_hardening.py` | No real corpus on disk (`data/processed/` empty) |
| Geometric feature extraction (26-D) | ✅ | `test_geometry_features.py` — 26 names, no NaNs, no leakage | Synthetic masks, not measured cities |
| Rasterization (7-channel, 128×128) | ✅ | `test_rasterizer.py` | Same |
| Normalization (field + feature) | ✅ | `test_normalization.py`, `test_validation_hardening.py` | Same |
| Layout splitting (ratio + family) | ✅ | `test_split.py` — disjoint IDs, whole-family holdout | Splitter unused by the API corpus |
| RF train / predict | ✅ (sklearn) | `test_baseline_rf.py` | Fitted on 24 synthetic layouts, not UrbanTALES |
| U-Net / CNN forward pass | ✅ (smoke) | `test_neural_architecture.py`, `test_neural_training.py`, `test_surrogate.py` | **Checkpoint randomly initialised**; API does call `model(x)` (confirmed in service) |
| Evaluation metrics & error maps | ✅ | `test_metrics_and_evaluation.py`, `test_visualization.py` | Hand-calculated synthetic fixtures |
| CFD pipeline bridge | ✅ (demo path) | `test_cfd_pipeline_bridge.py` | No real OpenFOAM samples; `simpleFoam` / `blockMesh` not on PATH |

---

### 3. API endpoints (verified)

| Method | Path | Response schema | Notes |
|---|---|---|---|
| `GET` | `/` | HTML or JSON stub | Serves `src/ui/index.html` |
| `GET` | `/static/*` | CSS/JS | `StaticFiles` on `src/ui/` |
| `GET` | `/api/health` | `{status, device, corpus_size, models_count}` | Live: 25 layouts, 3 models |
| `GET` | `/api/overview` | `SystemOverview` | `is_demonstration_fixture: true`; `ground_truth_solvers_count: 0` |
| `GET` | `/api/layouts` | `List[LayoutSummary]` | 25 layouts (24 synthetic + 1 canonical NPZ) |
| `GET` | `/api/layouts/{layout_id}` | `LayoutDetail` | 26 geometric features |
| `GET` | `/api/layouts/{layout_id}/fields` | `SpatialFieldData` | Flat arrays; `has_cfd_ground_truth: false` |
| `POST` | `/api/surrogates/predict` | `SurrogatePredictionResponse` | Calls `model.forward()`; `is_trained_checkpoint: false` |
| `GET` / `POST` | `/api/layouts/{layout_id}/probe` | `ProbeQueryResponse` | Bilinear interpolation at physical coords |
| `GET` | `/api/layouts/{layout_id}/transect` | `TransectQueryResponse` | 1-D cut profile |
| `GET` | `/api/experiments` | `List[ExperimentRecord]` | 3 fixture rows (all "Untrained" / "Baseline") |
| `POST` | `/api/experiments/run` | `{status, experiment_id, message}` | Starts background training thread (real `NeuralSurrogateTrainer`) |
| `GET` | `/api/export/triplet/{layout_id}` | `image/png` | Generates publication triplet figure |
| `GET` | `/api/inventory/datasets` | `DatasetInventory` | 25 total; 0 CFD solver outputs |
| `GET` | `/api/inventory/checkpoints` | `CheckpointInventory` | UNet: `is_trained=false`; RF: fitted |
| `GET` | `/api/inventory/cfd` | `CFDInventory` | `openfoam_available: false`; 1 template case |

**Not existing (live 404):** `/api/ready`, `/api/models`, `/api/datasets`, `/api/cfd`, `/api/cfd/cases`, `/api/checkpoints`

---

### 4. End-to-end behaviour (live HTTP)

Verified by running the service and inspecting responses directly:

| Check | Result |
|---|---|
| `GET /api/health` | **200** `{status:healthy, device:mps, corpus_size:25, models_count:3}` |
| `GET /api/overview` | **200** `is_demonstration_fixture:true`, `ground_truth_solvers_count:0`, `synthetic_fixture_count:24` |
| `GET /api/layouts` | **200** 25 items |
| `GET /api/layouts/layout_042_staggered_canyon` | **200** 26 features |
| `GET /api/layouts/does_not_exist` | **404** |
| `GET .../fields` | **200** 128×128 fluid stats: min 0.169, mean 2.913, max 5.237 m/s |
| `POST /api/surrogates/predict` (unet, layout 042) | **200** MAE 2.608, `is_trained_checkpoint:false`, latency 487 ms |
| `POST /api/surrogates/predict` missing layout | **404** |
| `GET /api/inventory/checkpoints` | **200** `unet_best.pt: is_trained=false, epoch=null, val_mae=null`; RF: fitted |
| `GET /api/inventory/cfd` | **200** `openfoam_available:false`, 1 template, `solver_available:false` |
| `GET /api/inventory/datasets` | **200** 25 total, 24 synthetic, 1 canonical, 0 cfd_solver_outputs |

#### 4.1 CFD execution status

- `simpleFoam` and `blockMesh`: **not on PATH**
- `Allrun` script would exit with "OpenFOAM environment not sourced"
- Docker: CLI present (`Docker 28.3.3`) but **daemon not running**
- The API **never** calls `Allrun`. Fields come from NumPy procedural synthesis.
- `cfd/outputs/canonical_block_reference.npz` exists: 64×128 grid, 400 building cells, 7,792 fluid cells, wind speed 0.82–4.90 m/s — **analytical Castro–Robins formulation**, not a solver run.

#### 4.2 Checkpoint status

- `checkpoints/unet_best.pt`: raw `state_dict` (86 tensors), **all `num_batches_tracked = 0`** — randomly initialised. No training metadata is embedded.
- `checkpoints/baseline_rf.joblib`: sklearn `Pipeline(StandardScaler + RandomForestRegressor)`, `_is_fitted=True`, trained on the 24 in-memory synthetic layouts.

---

### 5. Dataset provenance

| Tier | API / UI claim | Actual |
|---|---|---|
| UrbanTALES (538 LES) | Target corpus; "UrbanFlow-2D v1.4" | Not downloaded; `data/processed/` is empty |
| OpenFOAM RANS ground truth | Overview banner, `has_cfd_ground_truth` | 0 executed cases; API now sets `has_cfd_ground_truth:false` |
| Canonical aerodynamic benchmark | Castro & Robins analytical reference | Analytical NumPy field; correctly flagged `CANONICAL_AERODYNAMIC_BENCHMARK` |
| API corpus (24 layouts) | Synthetic procedural fixture | Correctly flagged `is_demonstration_fixture:true` |
| Software test fixtures | `tests/fixtures/synthetic_data.py` | Clearly labelled; not to be cited as model skill |

---

### 6. UI-to-API integration gaps

The frontend (`src/ui/app.js`) calls **all** API endpoints correctly via `fetch`. The main integration gaps are field-name mismatches and static decorative content:

| UI element | API field | Gap |
|---|---|---|
| Inference badge | reads `pred.inference_time_ms` | API returns **`inference_latency_ms`** → shows "—" |
| Speedup badge | reads `pred.speedup_ratio` | API returns **`speedup_factor`** → shows "—" |
| R² panel | reads `pred.r_squared` | API returns **`r2_score`** → shows "—" |
| Layout card buildings count | reads `num_buildings` | Field exists in `LayoutSummary`; **was missing in prior audit, now present** |
| Layout card λp | reads `plan_area_fraction` | Field exists in `LayoutSummary`; now present |
| Building bounding boxes | `building_boxes` | Only populated for `layout_042_staggered_canyon`; empty array for all others |
| Overview provenance banner | expects `data.data_provenance.solver_status` | Nested object **present** in current `SystemOverview`; banner should work |
| Colorbar min/mean/max | static HTML `3.81` / `7.24` | Live field stats: `2.913` / `5.237` m/s |
| Experiment table MAE | hardcoded fixture rows | All experiments have `held_out_mae_mps: null` ("Untrained" status) |
| Models catalog | hardcoded HTML | Claims 419 201 weights; actual CompactUNet = **1 949 441**; CNN = **21 025** |

#### Unsupported claims still visible in UI static HTML

1. "24 converged RANS" / "OpenFOAM-v2312 simpleFoam"
2. "65,536 Cartesian cells" (128×128 = 16,384; OpenFOAM template ≈ 8,100)
3. Four topological families A–D (only 3 `LayoutFamily` values exist)
4. Zero-leakage MAE 0.142 m/s as a trained result
5. −18.4% improvement badge
6. 840× / 950× / 1,200× speedup (not measured on this machine)
7. "Epoch 250", "14.2M parameters" (actual UNet: 1.95M; CNN: 21K)
8. Canyon protocol "9.2% shear attenuation"

---

### 7. Prioritised missing work

**P0 — Correctness (required before any public demonstration)**

1. Fix the three broken JSON key bindings: `inference_latency_ms` → `inference_time_ms` in frontend, or rename API field.
2. Remove / gate the remaining hardcoded UI metric claims behind `is_demonstration_fixture` provenance banner.
3. Document `python -m src.api.server` in README; update test count to 207; add `src/api/` and `src/ui/` to project map.

**P1 — Library-as-service (recommended next sprint)**

4. Persist layouts as NPZ under `data/processed/` via `LayoutDataLoader`; `GET /api/inventory/datasets` already exists and will populate automatically.
5. Add `GET /api/models` returning checkpoint metadata (param count, is_trained, split hash).
6. Add `GET /api/ready` (corpus loaded, models loadable) distinct from `/api/health`.
7. Wire `scalar_field` parameter; return 404 for unknown `model_id`; populate `building_boxes` from actual masks for all layouts.

**P2 — Scientific data (post-submission)**

8. Ingest a documented UrbanTALES subset into the contract.
9. Execute `simple2d_building_block` in Docker OpenFOAM; export via `foam_to_contract.py`; serve as the sole CFD reference.
10. Train U-Net/RF on a real layout-level holdout; store checkpoints with split hashes; report only those metrics.

---

## Part 2 — EDA and Preprocessing Evidence for the Presentation

### Section A — EDA analyses already available

| Analysis | Exists? | Source file / function | Output on disk? |
|---|---|---|---|
| Building-layout visualisations | ✅ | `src/evaluation/plotting.plot_field_triplet` + `plot_single_field`; `scripts/visualize_fields.py --synthetic` | `outputs/figures/triplet_layout_042_staggered_canyon_*.png` (3 files) |
| Building mask (occupancy) | ✅ | `src/data/contract.GeometryData.building_mask` (bool array); rendered in all triplet plots | Embedded in triplet PNGs |
| Outdoor fluid mask | ✅ | `~building_mask`; used in every metric computation | Embedded in triplet PNGs |
| Wind-speed heatmap (canonical) | ✅ | `cfd/outputs/canonical_block_field.svg` | **`cfd/outputs/canonical_block_field.svg`** — use directly in PPT |
| Wind-speed spatial field (layout 042) | ✅ | `service.samples['layout_042_staggered_canyon'].targets.pedestrian_wind_speed`; rendered via `plot_field_triplet` | `outputs/figures/triplet_layout_042_staggered_canyon_unet-v2-resblock4.png` |
| Building-height distribution | ✅ (data exists) | `sample.geometry.height_map[building_mask]`; **no standalone histogram plot saved** | Not on disk — **can be generated safely** |
| Geometric feature distributions (26-D) | ✅ (data exists) | `GeometricFeatureExtractor.extract()`; `GeometricFeatures.to_dict()` | Not on disk — **can be generated safely** |
| Summary statistics across layouts | ✅ (data exists) | 25 layouts in `MicroclimateService.samples` | Not on disk — **can be generated safely** |
| Wind-speed histogram | ✅ (data exists) | `targets.pedestrian_wind_speed[~mask]` for all layouts | Not on disk — **can be generated safely** |

**Existing plot paths usable directly in PowerPoint:**
```
outputs/figures/triplet_layout_042_staggered_canyon_unet-v2-resblock4.png    (196 KB)
outputs/figures/triplet_layout_042_staggered_canyon_compact-cnn-dilated.png  (206 KB)
outputs/figures/triplet_layout_042_staggered_canyon_rf-baseline-morph.png    (199 KB)
outputs/figures/data_workflow.png                                             (101 KB)
outputs/figures/system_architecture.png                                       (192 KB)
cfd/outputs/canonical_block_field.svg                                         (SVG — import as image)
outputs/plots/extent_test_triplet.png                                         (121 KB)
```

---

### Section B — EDA plots that can safely be generated

All plots below use **only** the in-memory synthetic corpus (25 layouts) or the analytical canonical reference NPZ. No model training, no CFD simulations. Each must be labelled **"Synthetic Procedural Corpus"**.

**Plot 1 — Building-height distribution (histogram)**
- Data: `sample.geometry.height_map[sample.geometry.building_mask]` across all 25 layouts
- Function: `src/training/visualize.py` or plain matplotlib
- Corpus heights range: 12.0 m – 65.0 m, mean ≈ 26.2 m

**Plot 2 — Wind-speed histogram (fluid cells)**
- Data: `sample.targets.pedestrian_wind_speed[~building_mask]` across all layouts
- Inflow speeds span 3.5 – 8.5 m/s (4 distinct values)
- Shows the distribution of the **target variable** being predicted

**Plot 3 — Plan area density λp distribution (corpus scatter / bar)**
- Data: `GeometricFeatureExtractor.extract(sample).plan_area_density_lambda_p` per layout
- Corpus range: 0.049 – 0.219, mean ≈ 0.179
- Groups by family (idealized_aligned 9, idealized_staggered 7, realistic_neighborhood 9)

**Plot 4 — Canonical reference field panels (already exists as SVG)**
- `cfd/outputs/canonical_block_field.svg` — building mask + wind speed heatmap
- Generated by `cfd/scripts/generate_canonical_reference.py`
- Grid: 64×128, 400 building cells, fluid wind 0.82–4.90 m/s

---

### Section C — Exact preprocessing sequence

Each step is implemented and tested. Status uses: ✅ Implemented & tested | ⚠️ Implemented, smoke-tested only | ❌ Not implemented.

| Step | What it does | Source file / function | Status | Key values |
|---|---|---|---|---|
| 1. Layout loading/generation | Load NPZ from disk **or** build synthetic geometry in RAM | `src/data/loader.LayoutDataLoader.load_directory` / `service._initialize_corpus` | ✅ (synthetic path); ⚠️ (disk path — `data/processed/` empty) | 25 layouts in corpus |
| 2. Schema validation | Check field existence, shapes, non-finite values, physical bounds | `src/data/contract.UrbanLayoutSample.validate` | ✅ | Rejects negative heights, wind, shape mismatches, NaN/Inf |
| 3. Building mask generation | Boolean array from height_map > 0 | `src/data/contract.GeometryData` | ✅ | layout 042: 11.3% building, 88.7% fluid |
| 4. 7-channel raster creation | Stack height, mask, SDF, inflow_u, inflow_v, coord_x, coord_y | `src/data/rasterizer.LayoutRasterizer.rasterize` | ✅ | Output: `(7, Ny, Nx)` float32 |
| 5. Signed Distance Transform | Euclidean distance from building boundary (metres) | `rasterizer.compute_signed_distance_transform` | ✅ | Positive outside, negative inside buildings |
| 6. Building-interior zeroing | Zero all target values inside building mask | `rasterizer.rasterize` (zero_inside_buildings=True) | ✅ | Prevents gradient leakage through solid walls |
| 7. 26-D geometric feature extraction | λp, λf, heights, z0/d (MacDonald), footprints, inflow, etc. | `src/data/features.GeometricFeatureExtractor.extract` | ✅ | 26 named scalars per layout |
| 8. Target field normalisation | Fit `FieldNormalizer` on training split only; transform all splits | `src/data/normalization.FieldNormalizer.fit / transform` | ✅ | Methods: standard, minmax, robust |
| 9. Feature normalisation | Fit `FeatureNormalizer` (StandardScaler or minmax) on train only | `src/data/normalization.FeatureNormalizer.fit / transform` | ✅ | Leakage guard: raises `DataLeakageError` if test IDs seen at fit |
| 10. Training / val / test split | Family-level holdout (idealized_staggered held out) or ratio split | `src/data/split.LayoutSplitter.split_by_family` | ✅ | Zero layout-overlap enforced by `DatasetSplit.assert_no_overlap` |
| 11. Outdoor masking during evaluation | All metrics computed only over `~building_mask` fluid cells | `src/evaluation/metrics._align_and_extract_fluid_arrays` | ✅ | Prevents solid-interior zeros from inflating R²/NRMSE |

---

### Section D — Data splitting and leakage prevention

Three mechanisms, all implemented and covered by dedicated tests:

1. **Layout-level split** (`LayoutSplitter.split_by_family`): entire topological families are held out. No layout appears in more than one partition. `DatasetSplit.assert_no_overlap()` raises if IDs overlap.
2. **Normaliser fitted on train only** (`FieldNormalizer.fit`, `FeatureNormalizer.fit`): both classes store the list of training layout IDs. `verify_no_leakage(test_layout_ids)` raises `DataLeakageError` if any test layout was seen during fitting.
3. **Outdoor masking** (`metrics._align_and_extract_fluid_arrays`): all scalar metrics and error maps are computed strictly over outdoor fluid cells, preventing the uniform zero-field inside buildings from deflating the error.

Test coverage: `test_split.py::test_leakage_detection_raises_error`, `test_validation_hardening.py::test_field_normalizer_leakage_detection`, `test_validation_hardening.py::test_feature_normalizer_leakage_detection`.

**Status caveat:** The API service builds a single in-memory corpus without calling `LayoutSplitter`. The split logic is correct software but is not yet applied to the live API corpus.

---

### Section E — Model preparation configuration

#### E.1 Random Forest baseline

| Parameter | Value | Source |
|---|---|---|
| Input | 26-D `GeometricFeatures` vector | `src/models/baseline_rf.py` + `configs/baseline_rf.json` |
| Output | Scalar area-averaged outdoor wind speed (m/s) | `area_averaging.py` |
| Estimators | 100 | `configs/baseline_rf.json` |
| Max depth | 12 | same |
| Preprocessing | `StandardScaler` (inside sklearn `Pipeline`) | `baseline_rf.RandomForestBaseline` |
| Checkpoint | `checkpoints/baseline_rf.joblib` — `_is_fitted: True` | Fitted on 24 synthetic layouts |
| Validation metrics | None recorded in checkpoint | **Not a CFD-quality result** |

#### E.2 CompactUNet (primary neural surrogate)

| Parameter | Value | Source |
|---|---|---|
| Input | `(7, Ny, Nx)` float32 raster | `src/models/neural_unet.CompactUNet` |
| Output | `(1, Ny, Nx)` wind speed field (m/s, normalised) | same |
| Architecture | 4-level encoder-decoder + skip connections | `CompactUNet.__init__` |
| Base channels | 32 | default |
| Trainable parameters | **1,949,441** | `CompactUNet.get_param_count()` — verified |
| Loss | `CombinedSurrogateLoss` (α·L1 + (1−α)·MSE + gradient penalty) | `src/training/losses.py` |
| Optimiser | Not set in checkpoints; `Adam(lr=1e-3)` in trainer default | `src/training/trainer.TrainerConfig` |
| Checkpoint | `checkpoints/unet_best.pt` — **randomly initialised** (`num_batches_tracked=0` for all BN layers) | Confirmed in this audit |
| Validation metrics | **None** (not trained) | — |

#### E.3 CompactCNN (dilated baseline)

| Parameter | Value | Source |
|---|---|---|
| Input / Output | Same as UNet (7-channel in, 1-channel out, same grid) | `src/models/neural_unet.CompactCNN` |
| Trainable parameters | **21,025** | `CompactCNN.get_param_count()` — verified |
| No skip connections | Ablation model vs UNet | — |
| Checkpoint | **None on disk** | Not trained |

---

### Section F — Missing evidence and limitations

1. **No real CFD ground truth**: all target fields are analytical/procedural. UrbanTALES is not loaded.
2. **UNet checkpoint is random init**: the MAE reported by `POST /api/surrogates/predict` (≈2.6 m/s) is from a random-weight forward pass, not a trained model. It **must not** be cited as model performance.
3. **RF generalisation unknown on real data**: the RF is fitted and will produce predictions, but the 24-layout synthetic training set is too small and too homogeneous to claim CFD-level accuracy.
4. **No EDA plots saved to disk** (except the triplet PNGs and the canonical SVG): histograms, feature distributions, and family-level summaries must be generated before use.
5. **No training curves exist**: the `plot_training_curves` function works, but there is no training history because no real training run was completed.
6. **Data split not applied to API corpus**: the `LayoutSplitter` is correct but is not called by `MicroclimateService`. The API derives `dataset_split` heuristically per-layout, not via a formal split object.

---

### Section G — Recommended 3–4 visualisations for the PPT

| # | What to show | Data to use | Exists? | Generation path |
|---|---|---|---|---|
| G1 | **Layout mosaic**: 3-panel — building mask, height map, wind-speed heatmap for layout_042_staggered_canyon | `service.samples['layout_042_staggered_canyon']` | **Exists** — `outputs/figures/triplet_layout_042_staggered_canyon_unet-v2-resblock4.png` | Use directly |
| G2 | **Canonical flow reference** (wind speed field around isolated block) | `cfd/outputs/canonical_block_reference.npz` | **Exists** — `cfd/outputs/canonical_block_field.svg` | Use directly; label "Analytical Castro–Robins Reference (Synthetic)" |
| G3 | **Building-height histogram** across the 25-layout synthetic corpus | `sample.geometry.height_map[building_mask]` for all samples | Not on disk | Generate via `src/training/visualize.py` or matplotlib; label "Synthetic Corpus (25 layouts)" |
| G4 | **Plan area density λp by family** (bar/scatter grouped by layout family) | `GeometricFeatureExtractor.extract(s).plan_area_density_lambda_p` for each of 25 layouts | Not on disk | Generate via `src/data/features.GeometricFeatureExtractor`; label "Synthetic Corpus — 26-D Morphometric Features" |

**Corpus numbers for caption text (all verified in this audit):**
- 25 in-memory layouts: 9 idealized_aligned, 7 idealized_staggered, 9 realistic_neighborhood (+ 1 canonical analytical reference, currently counted in the 25)
- Grid: 128×128 (primary layouts), 64×64 (procedural variants), 64×128 (canonical reference)
- Building heights: 12 – 65 m, mean ≈ 26 m
- λp range: 0.049 – 0.219, mean ≈ 0.179
- Inflow speeds: 3.5, 5.0, 7.0, 7.5, 8.5 m/s
- All data is **synthetic/procedural** — must be labelled accordingly in every slide

---

## Appendix — Test inventory (207 tests, all passing)

```
tests/test_api_endpoints.py            12 tests — HTTP routes via TestClient
tests/test_baseline_rf.py             17 tests — RF pipeline, metrics, persistence
tests/test_cfd_pipeline_bridge.py      7 tests — foam→contract, canonical NPZ round-trip
tests/test_contract_and_validation.py  8 tests — schema validation edge cases
tests/test_data.py                    17 tests — schema, area-avg, raster, normalizers, split, NPZ
tests/test_geometry_features.py       31 tests — all 26 features, edge cases, leakage
tests/test_loader.py                   3 tests — NPZ round-trip, corrupt file, directory batch
tests/test_metrics_and_evaluation.py  20 tests — hand-calculated metrics, mask semantics, errors
tests/test_model_interfaces.py         2 tests — target config validation
tests/test_neural_architecture.py      6 tests — UNet/CNN shapes, mask enforcement, checkpoint
tests/test_neural_training.py          3 tests — dataset, smoke training, reproducibility
tests/test_normalization.py            3 tests — field/feature normalizer inversion
tests/test_p0_integrity.py            10 tests — provenance honesty, no hardcoded metrics
tests/test_p1_inventory.py            14 tests — inventory endpoints, canonical NPZ loading, experiment thread
tests/test_rasterizer.py               3 tests — channels, SDF, indoor zeroing
tests/test_split.py                    8 tests — ratio, reproducibility, group/family holdout, leakage
tests/test_surrogate.py                7 tests — UNet/CNN forward, masked loss, training, evaluator, plot
tests/test_validation_hardening.py    16 tests — all error types, leakage guards, NPZ archive checks
tests/test_visualization.py           10 tests — triplet/quad/single field plots, shapes, extents
```
