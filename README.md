# Microclimate Modelling via Neural Surrogates and CFD

Course project comparing a traditional ML baseline (using geometric summary features) against a neural surrogate (using multi-channel rasterized inputs) for urban microclimate wind and flow prediction.

---

## 1. Environment Setup

### Prerequisites
- Python 3.11+ (tested on macOS Apple Silicon with Python 3.13 / 3.14)
- Virtual environment tool (`venv`)

### Installation Commands
Run the following commands to create an isolated virtual environment and install all dependencies:

```bash
# 1. Create a virtual environment
python3.13 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install scientific dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Repository Structure

```
Urban-Microclimate-Modelling/
├── docs/
│   ├── dataset_report.md        # Technical evaluation of UrbanTALES dataset, access, and variables
│   ├── data_schema.md           # Agreed Data Contract, schemas, and holdout protocols
│   ├── model_design.md          # Architectural specifications for AML baseline & Neural Surrogates
│   ├── experiment_protocol.md   # Rigorous evaluation guidelines, NRMSE formulas & leakage prevention
│   ├── cfd_feasibility.md       # CFD numerical feasibility analysis
│   └── cfd_methodology.md       # OpenFOAM LES/RANS simulation setup documentation
├── src/
│   ├── api/                     # FastAPI backend and scientific service facade
│   │   ├── server.py            # FastAPI server entry point and static UI mount
│   │   ├── routes.py            # REST endpoints (/api/overview, /api/layouts, /api/surrogates/predict, etc.)
│   │   ├── schemas.py           # Pydantic data schemas with strict data provenance
│   │   └── service.py           # In-memory service facade managing corpus, models, and spatial evaluation
│   ├── data/                    # Data ingestion and preprocessing package
│   │   ├── contract.py          # Data contract, schemas, and shape validation
│   │   ├── features.py          # Geometric summary feature extractor (26 morphometric features)
│   │   ├── rasterizer.py        # Spatial multi-channel rasterizer & signed distance transforms (7 channels)
│   │   ├── normalization.py     # Train-fitted normalizers with exact inversion
│   │   ├── split.py             # Layout-level and family holdout splitters (zero leakage)
│   │   ├── loader.py            # Serializer and validated NPZ data loader
│   │   └── config.py            # Preprocessing configuration and statistics persistence
│   ├── evaluation/              # Quantitative evaluation metrics and publication plotting
│   │   ├── metrics.py           # Outdoor-masked MAE, RMSE, documented NRMSE conventions, R², and error maps
│   │   ├── evaluator.py         # SurrogateEvaluator orchestrating inverse normalization and layout metrics
│   │   └── plotting.py          # 3-panel publication comparison and residual maps
│   ├── models/                  # Machine Learning & Neural Surrogate model definitions
│   │   ├── interfaces.py        # Explicit TargetConfig and abstract model interfaces
│   │   ├── baseline_rf.py       # Scikit-learn Random Forest regression pipeline
│   │   └── neural_unet.py       # Compact U-Net & Dilated CNN PyTorch architectures with outdoor masking
│   ├── training/                # Training loops, loss functions, metrics, and visualization
│   │   ├── losses.py            # Outdoor-masked L1, MSE, and composite gradient losses
│   │   ├── metrics.py           # Masked Field MAE, RMSE, documented NRMSE formulas, & aggregation bridge
│   │   ├── trainer.py           # Reproducible PyTorch training loop, validation checkpointing, & seeds
│   │   └── visualize.py         # 3-panel field prediction comparison, error maps, & scatter plots
│   └── ui/                      # Scientific research frontend (HTML5/Canvas/CSS)
├── tests/                       # Unit and integration test suite
│   ├── fixtures/
│   │   └── synthetic_data.py    # Synthetic test fixtures (clearly labelled for testing only)
│   ├── test_api_endpoints.py
│   ├── test_baseline_rf.py
│   ├── test_cfd_pipeline_bridge.py
│   ├── test_contract_and_validation.py
│   ├── test_data.py
│   ├── test_geometry_features.py
│   ├── test_loader.py
│   ├── test_metrics_and_evaluation.py
│   ├── test_model_interfaces.py
│   ├── test_neural_architecture.py
│   ├── test_neural_training.py
│   ├── test_normalization.py
│   ├── test_rasterizer.py
│   ├── test_split.py
│   ├── test_surrogate.py
│   ├── test_validation_hardening.py
│   └── test_visualization.py
├── requirements.txt             # Core dependencies (NumPy, SciPy, PyTest, Scikit-learn, PyTorch, Matplotlib, FastAPI)
├── .gitignore                   # Excludes caches, venvs, and raw/processed data
└── README.md
```

---

## 3. Running the API and Scientific Frontend

The backend API is implemented with FastAPI and serves the Microclimate Lab static UI:

```bash
# Start server using module entry point (binds to http://127.0.0.1:8000)
python -m src.api.server

# Or directly with uvicorn
uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

Once started:
- **Interactive Web UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Live OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Overview Endpoint:** [http://127.0.0.1:8000/api/overview](http://127.0.0.1:8000/api/overview)

### Core REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health status and PyTorch hardware acceleration device |
| `GET` | `/api/overview` | Aggregated scientific metrics, active dataset, and solver provenance |
| `GET` | `/api/layouts` | Catalog of layout summaries with geometric features and splits |
| `GET` | `/api/layouts/{id}` | Detailed layout morphometrics and boundary conditions |
| `GET` | `/api/layouts/{id}/fields` | 2D spatial grid arrays ($128\times 128$) with fluid masking |
| `POST` | `/api/surrogates/predict` | Genuine PyTorch forward pass (`CompactUNet`/`CompactCNN`) with outdoor masking |
| `GET` | `/api/layouts/{id}/probe` | Physical $(x, y)$ coordinate interpolation and absolute residual |
| `GET` | `/api/layouts/{id}/transect` | 1D transect profile extraction across a specified coordinate line |
| `GET` | `/api/inventory/datasets` | Inventory of loaded layouts, provenance tiers, and ground truth flags |
| `GET` | `/api/inventory/checkpoints` | Checkpoint status scan (`is_trained`, epoch, validation metrics) |
| `GET` | `/api/inventory/cfd` | OpenFOAM case templates, solver availability, and pre-computed outputs |
| `GET` | `/api/experiments` | Benchmark experiment records and convergence metrics |
| `POST` | `/api/experiments/run` | Launch background training run with `NeuralSurrogateTrainer` |
| `GET` | `/api/export/triplet/{id}` | Generate and export 3-panel publication comparison figure |

---

## 4. Running Tests

To run the complete data engineering, API, and ML/DL test suite:

```bash
# Using active virtual environment:
pytest -v

# Or directly via venv path:
./.venv/bin/pytest -q --tb=line
```

The test suite validates:
- Model shape conformity and genuine forward passes on multi-channel spatial inputs.
- Accurate metrics evaluated over valid outdoor fluid cells using `src.evaluation.metrics`.
- Strict data provenance checking: synthetic procedural fields are never labelled as OpenFOAM solver output (`has_cfd_ground_truth=False`).
- Real model checkpoint inventory reporting (`is_trained=False` for untrained weights).
- Background training thread execution with `NeuralSurrogateTrainer`.
- Scikit-learn pipeline train-only normalization and layout-level evaluation.
- Physical outdoor building masking ($U = 0.0$ inside solid structures).
- Reproducible random seeding and validation-based checkpointing.
- Headless figure generation (comparison plots, error maps, training curves).
- Complete layout-level holdout (zero spatial autocorrelation leakage).

---

## 5. Usage Example

### 4.1 AML Baseline (Random Forest)
```python
from src.models import RandomForestBaseline
from src.data import LayoutSplitter, GeometricFeatureExtractor
from tests.fixtures.synthetic_data import create_synthetic_sample

# Load samples conforming to the data contract
samples = [create_synthetic_sample(f"layout_{i}") for i in range(10)]
samples_by_id = {s.layout_id: s for s in samples}

# Partition at complete layout level (zero spatial leakage)
split = LayoutSplitter.split_by_ratio(list(samples_by_id.keys()), train_ratio=0.7, test_ratio=0.3)

# Train pipeline strictly on train partition and evaluate held-out test layouts
baseline = RandomForestBaseline(n_estimators=100, max_depth=10, random_state=42)
results = baseline.evaluate_dataset_split(samples_by_id, split)
print(f"Test MAE: {results['test']['mae']:.4f} m/s | Test RMSE: {results['test']['rmse']:.4f} m/s")
```

### 4.2 Deep Neural Surrogate (Compact U-Net)
```python
import torch
from src.models import CompactUNet
from src.training import SurrogateDataset, NeuralSurrogateTrainer, TrainerConfig
from tests.fixtures.synthetic_data import create_synthetic_sample

# Prepare samples
train_samples = [create_synthetic_sample(f"train_{i}") for i in range(6)]
val_samples = [create_synthetic_sample(f"val_{i}") for i in range(2)]

train_dataset = SurrogateDataset(train_samples)
val_dataset = SurrogateDataset(val_samples)

# Initialize compact model (7 input channels -> 1 output channel)
model = CompactUNet(in_channels=7, out_channels=1, base_channels=16, depth=3)

# Train with reproducible seed, masked loss, and validation checkpointing
trainer = NeuralSurrogateTrainer(model=model, config=TrainerConfig(epochs=10, seed=42))
history = trainer.fit(train_dataset, val_dataset)
print(f"Best Val NRMSE: {trainer.best_val_nrmse:.4f}")
```

---

## 6. Dataset Investigation Highlights

- **Dataset:** [UrbanTALES](https://urbantales.vercel.app/) (Nazarian et al., 2025; BAMS DOI: 10.1175/BAMS-D-25-0061.1).
- **License:** CC BY-NC 4.0.
- **Physical Scope:** Isothermal, aerodynamic Large-Eddy Simulations across 538 layouts.
- **Important Note:** Temperature is **not** included in UrbanTALES; therefore, pedestrian-level wind speed ($U_{1.75\text{m}}$) is designated as the primary target variable. Full details are documented in [`docs/dataset_report.md`](docs/dataset_report.md).

---

## 7. Project Deliverables & Viva Preparation

All academic project deliverables, presentations, and defense materials have been generated and verified:

| Deliverable | Location | Description |
|-------------|----------|-------------|
| **Project Report (DOCX)** | [`deliverables/Urban_Microclimate_Modelling_Project_Report.docx`](deliverables/Urban_Microclimate_Modelling_Project_Report.docx) | Full 8-chapter academic course report with figures and tables (680 KB) |
| **Project Report (PDF)** | [`deliverables/Urban_Microclimate_Modelling_Project_Report.pdf`](deliverables/Urban_Microclimate_Modelling_Project_Report.pdf) | Publication-formatted PDF report with page numbers and certificates (840 KB) |
| **Project Presentation (PPTX)** | [`deliverables/Urban_Microclimate_Modelling_Project_Presentation.pptx`](deliverables/Urban_Microclimate_Modelling_Project_Presentation.pptx) | 16-slide widescreen presentation with full speaker notes on every slide (705 KB) |
| **Viva Defense Guide (DOCX)** | [`deliverables/Urban_Microclimate_Modelling_Viva_Guide.docx`](deliverables/Urban_Microclimate_Modelling_Viva_Guide.docx) | Comprehensive first-principles guide, code walkthrough, and 50-question bank (55 KB) |
| **Viva Defense Guide (PDF)** | [`deliverables/Urban_Microclimate_Modelling_Viva_Guide.pdf`](deliverables/Urban_Microclimate_Modelling_Viva_Guide.pdf) | Printable technical summary and top 10 defense questions (7.4 KB) |
| **Live Demonstration Runbook** | [`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md) | 10-step live viva demonstration script and fallback instructions |
| **Viva Practice Plan** | [`docs/VIVA_PRACTICE_PLAN.md`](docs/VIVA_PRACTICE_PLAN.md) | 7 technical hands-on exercises for self-rehearsal |
| **Project Completion Report** | [`docs/project_completion_report.md`](docs/project_completion_report.md) | Detailed technical verification summary and findings |
