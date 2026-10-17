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
│   ├── dataset_report.md        # Technical evaluation of UrbanTALES dataset, access, license, and variables
│   └── data_schema.md           # Agreed Data Contract, schemas, and holdout protocols
├── src/
│   └── data/                    # Data ingestion and preprocessing package
│       ├── contract.py          # Data contract, schemas, and shape validation
│       ├── features.py          # Geometric summary feature extractor for ML baseline
│       ├── rasterizer.py        # Spatial multi-channel rasterizer & signed distance transforms
│       ├── normalization.py     # Train-fitted normalizers with exact inversion
│       ├── split.py             # Layout-level and family holdout splitters (zero leakage)
│       ├── loader.py            # Serializer and validated NPZ data loader
│       └── config.py            # Preprocessing configuration and statistics persistence
├── tests/                       # Unit test suite
│   ├── fixtures/
│   │   └── synthetic_data.py    # Synthetic test fixtures (clearly labelled for testing only)
│   ├── test_contract_and_validation.py
│   ├── test_geometry_features.py
│   ├── test_rasterizer.py
│   ├── test_normalization.py
│   ├── test_loader.py
│   └── test_split.py
├── requirements.txt             # Core dependencies (NumPy, SciPy, PyTest)
├── .gitignore                   # Excludes caches, venvs, and raw/processed data
└── README.md
```

---

## 3. Running Tests

To run the complete data engineering test suite:

```bash
# Using active virtual environment:
pytest -v

# Or directly via venv path:
./.venv/bin/pytest -v
```

All 23 test cases validate:
- Shape matching and coordinate consistency.
- Rejection of invalid/negative heights and speeds.
- Invertibility of spatial and tabular normalizers.
- Signed distance field and spatial alignment.
- Layout-level and family-level holdout (ensuring zero data leakage).

---

## 4. Usage Example

```python
from src.data import (
    GeometricFeatureExtractor,
    LayoutRasterizer,
    FieldNormalizer,
    LayoutSplitter,
)
from tests.fixtures.synthetic_data import create_synthetic_sample

# 1. Load or generate a validated sample conforming to the contract
sample = create_synthetic_sample("layout_001")
sample.validate()

# 2. Extract tabular summary features for traditional ML baseline
extractor = GeometricFeatureExtractor()
features = extractor.extract(sample)
feature_vector = features.to_feature_vector()
target_scalar = features.target_area_averaged_wind_speed
print(f"Extracted {len(feature_vector)} features for layout {sample.layout_id}")

# 3. Rasterize inputs and aligned target fields for Neural Surrogate
rasterizer = LayoutRasterizer()
raster_data = rasterizer.rasterize(sample)
print(f"Raster input shape: {raster_data.channels.shape}")  # (C_in, Ny, Nx)
print(f"Raster target shape: {raster_data.target_field.shape}")  # (C_out, Ny, Nx)
```

---

## 5. Dataset Investigation Highlights

- **Dataset:** [UrbanTALES](https://urbantales.vercel.app/) (Nazarian et al., 2025; BAMS DOI: 10.1175/BAMS-D-25-0061.1).
- **License:** CC BY-NC 4.0.
- **Physical Scope:** Isothermal, aerodynamic Large-Eddy Simulations across 538 layouts.
- **Important Note:** Temperature is **not** included in UrbanTALES; therefore, pedestrian-level wind speed ($U_{1.75\text{m}}$) is designated as the primary target variable. Full details are documented in [`docs/dataset_report.md`](docs/dataset_report.md).
