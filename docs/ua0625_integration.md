# UA0625_d00 Real UrbanTALES Dataset Integration

**Status:** Integrated (geometry + 1D vertical profile, read-only)
**Module:** [`src/data/ua0625.py`](../src/data/ua0625.py)
**EDA script:** [`scripts/ua0625_eda.py`](../scripts/ua0625_eda.py)
**Tests:** [`tests/test_ua0625_integration.py`](../tests/test_ua0625_integration.py)

This document describes the integration of the real **UrbanTALES** LES sample
`UA0625_d00` shipped in the repository root, and the scientific constraints
enforced so that no data is fabricated and no metadata is assumed.

---

## 1. Source Files (preserved, never modified)

| File | Role | Format |
|------|------|--------|
| `UA0625_d00_topo.csv` | 2D building topology / height grid (geometry) | header-less numeric grid |
| `profile-UA0625_d00.csv` | 1D horizontally-averaged vertical profile | CSV with header row |
| `UA0625_d00.png` | High-resolution geometry rendering (visualisation) | RGBA PNG |

The original files are read-only inputs; the loader and EDA script never
overwrite them. EDA figures are written to `outputs/ua0625/` (git-ignored, as
`outputs/` is excluded by `.gitignore`).

---

## 2. Inspection Results (verbatim from the files)

### 2.1 Topology grid — `UA0625_d00_topo.csv`
- **Shape:** 384 rows (y) × 640 columns (x) — a pure numeric grid with **no header row** and **no spacing metadata**.
- **Values:** exactly two unique values — `0.0` (ground) and `16.0` (building).
- **Plan-area density** `lambda_p` (computed from the raster, not assumed): **0.0625** — matches the `0625` filename label.
- **Building cells:** 15,360 (6.25% of the 245,760-cell domain); **15 connected building components**.
- **Building height:** uniform 16.0 m (interpreted as metres, consistent with the profile `z` range 0–131.5 m and UrbanTALES conventions; the CSV carries no explicit unit field).
- **Physical spacing:** **not recorded in the file** → `resolution_m = domain_size_m = None`.

### 2.2 Vertical profile — `profile-UA0625_d00.csv`
- **Rows:** 130 data levels (1 header row + 130 rows); **31 columns**.
- **Header (by name):** `z, u, v, w, U, s, TKE, Duw, Tuw, Dvw, Tvw, Mom, ILs, vel, stdS, cdz, uGrad, sgsR, sgsM, turbT, lm, dispT, presT, shearP, wakeP, formA, Km, cklk, lece, cmu` (plus a leading index column).
- **Height axis `z`:** 0.0 → 131.497 m, **strictly increasing**, all unique.
- **Ground level (z = 0 m):** `u, v, w, U, TKE` are **NaN** — the no-slip ground has no defined mean velocity (physically correct). All other levels are finite for these columns (129/130 finite).
- **`U` is the dataset-reported speed magnitude.** It is **not** equal to `sqrt(u² + v² + w²)` for this sample (max relative deviation ≈ 18%); the loader preserves the reported value verbatim rather than recomputing it. (`vel` equals `|u|`, the streamwise magnitude.)
- **`TKE`** may take **negative** values near the domain top (dataset-specific definition); values are loaded as-is and never clamped.

### 2.3 Geometry image — `UA0625_d00.png`
- **Size:** 5087 × 1972 px, **mode RGBA**, `uint8`. Used for geometry visualisation only (no quantitative inference).

---

## 3. Scientific Constraints Enforced

These follow the project's data-contract principles ([`docs/data_schema.md`](data_schema.md)) and the integration brief:

1. **No fabricated 2D wind field.** The vertical profile is a 1D, height-dependent, horizontally-averaged quantity. It is **never** broadcast into a 2D pedestrian-level field. The dataset container exposes `has_2d_wind_field = False`, and the topology deliberately has no wind-speed attribute.
2. **No assumed grid resolution / orientation / spacing.** The topology CSV carries no spacing metadata, so `resolution_m` and `domain_size_m` are `None` (not guessed). `lambda_p` is *computed* from the raster (verifiable) and cross-checked against the filename label.
3. **No CFD / training claims.** The data originates from UrbanTALES LES run elsewhere; `locally_simulated = False`, and **no model is trained** on this single sample.
4. **Honest units.** The building height (16.0) is recorded as *interpreted* metres (UrbanTALES convention + profile z-range), not as a confirmed file attribute. The speed/TKE definitions are taken from the dataset, not imposed.
5. **Preserved originals.** Source files are read-only inputs.

Because the project's full [`UrbanLayoutSample`](../src/data/contract.py) contract requires physical spacing metadata **and** a 2D pedestrian-wind-speed target — neither of which is available here — the integration deliberately does **not** construct a full sample. Instead, the geometry is validated against the **same** `GeometryData` consistency rules (non-negative finite heights, mask/height agreement) used by the rest of the pipeline, keeping the integration consistent with the data contract without fabricating the missing pieces.

---

## 4. Module API (`src.data.ua0625`)

| Symbol | Purpose |
|--------|---------|
| `UA0625Topology` | 2D height map + building mask + spacing-free geometry summary; reuses `GeometryData` validation. |
| `UA0625VerticalProfile` | 1D `z, u, v, w, speed, tke` arrays (NaN-aware); validates monotonic `z` only. |
| `UA0625Image` | Geometry PNG (visualisation). |
| `UA0625Provenance` | Honest provenance + capability flags (`has_2d_wind_field=False`, …). |
| `UA0625Dataset` | Container for all three artefacts + `limitations`. |
| `load_ua0625_topology` / `load_ua0625_profile` / `load_ua0625_image` | Individual loaders. |
| `load_ua0625_dataset` | Loads all three from default repo locations. |
| `parse_ua0625_dataset_id` | Parses `UA0625_d00` → `lambda_p`/direction labels (filename convention). |

All symbols are re-exported from `src.data`.

```python
from src.data import load_ua0625_dataset

ds = load_ua0625_dataset()            # UA0625_d00
ds.topology.plan_area_density         # 0.0625 (computed)
ds.topology.resolution_m              # None (not assumed)
ds.profile.z[:3]                      # array([0.  , 0.25, 0.75])
ds.has_2d_wind_field                   # False  (no fabricated 2D field)
```

---

## 5. EDA Visualisations (`outputs/ua0625/`)

Generated by `python scripts/ua0625_eda.py`:

1. `ua0625_01_topology_height.png` — building-height topology grid (axes in cell indices; spacing unavailable).
2. `ua0625_02_building_mask.png` — binary building footprint.
3. `ua0625_03_vertical_wind_profile.png` — `u, v, w, U` vs height (vertical wind analysis).
4. `ua0625_04_tke_profile.png` — TKE vs height (turbulence analysis).
5. `ua0625_05_geometry_image.png` — the geometry rendering (downscaled for display; original preserved).

The topology axes are deliberately shown in **grid-cell indices**, not metres,
because physical spacing is unknown — this is annotated on each figure.

---

## 6. API Integration (additive, read-only)

A new, self-contained endpoint surfaces the real dataset without touching the
synthetic corpus or existing endpoints:

```
GET /api/inventory/urbantales → UrbanTALESInventory
```

It loads `UA0625_d00` on demand and returns honest provenance, the geometry
summary, the profile extent, and the limitations list. If the source files are
absent it returns `available: false` (HTTP 200) rather than erroring. The
existing `/api/inventory/datasets|checkpoints|cfd` endpoints and the synthetic
corpus are unchanged.

---

## 7. Known Limitations

- No 2D pedestrian-level wind field is available; the vertical profile is 1D and horizontally averaged.
- Grid resolution / physical spacing is unknown (`resolution_m = domain_size_m = None`).
- Wind-direction label (`d00` → 0°) is a filename convention, not verifiable from the topology file alone.
- Building height unit is interpreted (metres), not confirmed by file metadata.
- The dataset-reported `U` ≠ `sqrt(u² + v² + w²)` and is preserved verbatim; `TKE` may be negative near the domain top.
- No CFD was run locally and no model is trained on this single sample; it supplements (does not replace) the synthetic/canonical corpus.

---

## 8. Provenance & Attribution

- **Source:** UrbanTALES — *Urban Turbulence Analyses from Large-Eddy Simulations* (Nazarian et al., 2025; BAMS DOI: [10.1175/BAMS-D-25-0061.1](https://doi.org/10.1175/BAMS-D-25-0061.1)).
- **License:** CC BY-NC 4.0.
- **Solver (origin):** uDALES LES. We loaded published output; we did not run the simulation locally.
