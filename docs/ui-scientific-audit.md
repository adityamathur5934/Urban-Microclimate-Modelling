# Scientific UI Audit: Scientific Validity, Dynamic Data, and Provenance

**Project:** Urban Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)
**Document:** `docs/ui-scientific-audit.md`
**Date:** October 2026
**Auditor:** Scientific Software Engineering Audit

---

## 1. Executive Summary

This audit evaluates the scientific validity, data provenance, and frontend-backend data flow of the Microclimate Lab UI against the repository's verified numerical assets and computational fluid dynamics (CFD) code.

### Summary of Key Findings
1. **Provenance Misalignment:** The application presents demonstration aerodynamic fixtures and synthetic test geometries as verified OpenFOAM CFD solutions. While the mathematical flow models in `cfd/scripts/` and `src/api/service.py` reproduce physical phenomena (boundary-layer stagnation, corner separation jets, and leeward wake cavities), they are analytical aerodynamic benchmarks rather than full 3D OpenFOAM solver output across 128 layouts.
2. **Unsupported Benchmark Claims:** Numbers from the visual reference mockup (e.g. "96/128 converged solutions", "6 model checkpoints including FNO and PINN", "840× speedup", "-18.4% improvement") were propagated into API responses or hardcoded in the frontend. FNO and PINN-NS are completely absent from the codebase.
3. **Frontend Hardcoding Bug:** Several diagnostic metrics (inference latency: `14.2 ms`, cosine similarity: `0.954`, $R^2$: `0.988`, peak local error: `0.892 m/s`) were hardcoded in HTML markup instead of dynamically reflecting the `POST /api/surrogates/predict` response.
4. **Navigation Active-State Bug:** Multiple sidebar navigation items shared the same `data-view="studio"` attribute (`nav-layouts`, `nav-cfd`, `nav-models`, `nav-studio`, `nav-evaluation`), causing all five items to highlight simultaneously as active when viewing the Field Comparison Studio.
5. **Streamline & Field Realism:** The overview canvas rendered hardcoded spline curves over the scalar field rather than genuine streamlines integrated from the horizontal and vertical velocity component fields ($u_x, u_y$).

---

## 2. Granular Audit of Target Values & Claims

| Displayed Value / Claim | UI Location | Implementation Source | Audit Classification | Finding & Remediation Plan |
|---|---|---|---|---|
| **128 Layouts** | Overview KPI Card | `routes.py:56` (`geometric_corpus_count=128`) | **Unsupported Target** | The repository currently generates 24 synthetic/canonical layouts. The full UrbanTALES corpus (538 layouts) has not been ingested. **Fix:** Display actual count (`24`) with an honest badge: `Synthetic Benchmark Cohort (24 Layouts) • Target Corpus: 538 Layouts`. |
| **96 / 128 Converged Solutions** | Overview KPI Card | `routes.py:58` (`ground_truth_solvers_count=96`) | **Unsupported Mock** | The repository has only 1 setup case (`cfd/cases/simple2d_building_block`) and 1 canonical analytical reference. 96 converged RANS cases do not exist. **Fix:** Replace with actual status: `1 CFD Case Setup / 24 Synthetic Benchmark Solutions`. |
| **6 Checkpoints (U-Net, FNO, PINN-NS)** | Overview KPI Card & Table | `routes.py:61`, `service.py:610` | **Unsupported Architectural Scope** | Project scope is strictly: (1) Random Forest ML baseline (26 features), (2) U-Net / CNN neural surrogate (7-channel raster), and (3) OpenFOAM CFD reference. FNO and PINN are not implemented. **Fix:** Restrict model inventory strictly to implemented architectures (`CompactUNet`, `CompactCNN`, `RandomForestBaseline`). Mark others as `Planned / Not Implemented`. |
| **0.142 m/s MAE** | Overview KPI & Table | `routes.py:47`, `service.py:558` | **Synthetic Fixture / Demonstration** | Derived from the visual reference mockup. The freshly initialized U-Net was not trained across 500+ held-out layouts. **Fix:** Label explicitly as `Demonstration Baseline Benchmark` or compute from actual trained baseline evaluation. Ensure Overview and Studio display the exact same evaluation for the selected layout/model. |
| **-18.4% Improvement** | Overview KPI Badge | `routes.py:64` (`zero_leakage_change_pct=-18.4`) | **Hardcoded Fixture** | Hardcoded visual fixture value. **Fix:** Compute relative difference dynamically against the Traditional Random Forest Baseline ($MAE_{\text{RF}} - MAE_{\text{UNet}}) / MAE_{\text{RF}}$ or label as `Reference Target`. |
| **14.2 ms Inference Latency** | Studio Panel 2 Badge | `index.html:353` | **Hardcoded in HTML** | Hardcoded text `Inf: 14.2 ms` in static HTML, ignoring `data.inference_latency_ms` calculated by `service.predict_surrogate()`. **Fix:** Bind dynamically to `data.inference_latency_ms`. |
| **840× Speedup** | Studio Panel 2 Badge | `index.html:354`, `service.py:559` | **Unsupported Hypothesis** | Estimated ratio without a measured OpenFOAM execution benchmark run on the same hardware. **Fix:** Display actual inference time with hardware descriptor (`Apple Silicon M-series / MPS` or `CPU`), labeling solver speedup as `Estimated vs 2D RANS (~12s solve)`. |
| **0.954 Cosine Similarity** | Studio Panel 2 Stats | `index.html:367` | **Hardcoded in HTML** | Static string in HTML. **Fix:** Bind dynamically to `data.cosine_similarity` returned from backend. |
| **0.988 R-squared ($R^2$)** | Studio Panel 3 Stats | `index.html:399` | **Hardcoded in HTML** | Static string in HTML. **Fix:** Bind dynamically to `data.r2_score` computed from valid fluid cells. |
| **0.892 m/s Peak Local Error** | Studio Panel 3 Badge & Stats | `index.html:387`, `398`, `437` | **Hardcoded in HTML** | Static strings in HTML. **Fix:** Bind dynamically to `data.peak_local_error` from backend evaluation. |
| **9.2% Boundary-Layer Attenuation** | Studio Protocol Note 1 | `index.html:463` | **Unverified Scientific Claim** | Unverified hypothesis from visual reference. **Fix:** Label as `Observed in deep street canyon LES studies (e.g. Nazarian et al. 2025); surrogate models exhibit underestimation in high-shear corner separation zones`. |

---

## 3. Data Provenance Taxonomy

To ensure scientific integrity, every visualization and dataset shown in the application will carry an unambiguous provenance indicator:

1. **`VERIFIED_CFD_DATA`**: Output from converged OpenFOAM solver runs (`simpleFoam` / `pimpleFoam`) or ingested raw UrbanTALES NetCDF/NPZ cases.
2. **`CANONICAL_AERODYNAMIC_BENCHMARK`**: Physically grounded analytical aerodynamic benchmark (using Castro & Robins, Martinuzzi & Tropea, and COST 732 formulations for stagnation, Venturi jets, and leeward wake recirculation).
3. **`SYNTHETIC_TEST_FIXTURE`**: Software testing fixtures generated for pipeline verification (clearly labeled, zero physical validity).
4. **`SURROGATE_PREDICTION`**: Direct forward inference from deep learning models (`CompactUNet`, `CompactCNN`) or tabular ML baseline (`RandomForestBaseline`).
5. **`DEMO_ESTIMATE`**: Estimated metrics clearly marked when verified solver logs are pending.

---

## 4. Architectural & UI Corrections Plan

### Phase 2: Provenance & Metrics Unification
- Update `src/api/routes.py` and `src/api/schemas.py` to include explicit `data_provenance` metadata in every layout and field response.
- Expose verified model inventory (`CompactUNet`, `CompactCNN`, `RandomForestBaseline`), removing non-existent FNO and PINN models from active selection.
- Ensure the selected experiment/model in Overview and Studio dynamically shares the exact same metric calculation source.

### Phase 3: Route & Navigation Fix
- Implement client-side URL hash routing (`#/overview`, `#/studio`, `#/layouts`, `#/models`, `#/experiments`, `#/docs`).
- Ensure exactly one navigation item is marked `active` corresponding to the route.
- Synchronize browser back/forward buttons, direct URL bookmarking, and page refreshes.

### Phase 4: Numerical Field & Streamline Rendering
- Render the 2D field strictly from raw Cartesian grid values with Viridis and Magma color maps.
- Streamlines: Remove hardcoded spline curves. Implement Runge-Kutta / Euler field line integration using genuine $u_x$ and $u_y$ velocity vectors returned in `SpatialFieldData`. If vector components are absent, disable streamlines cleanly.
- Absolute error map: Apply building mask strictly, so solid building cells are rendered in neutral dark grey and excluded from error scales.

### Phase 5: Interactive Comparison Studio
- Dynamically bind all panel stats: Mean, Peak, Latency, Speedup, Cosine Similarity, MAE, RMSE, NRMSE, $R^2$, and Peak Local Error.
- Probe tool: Compute true physical coordinates $(X, Y)$ from canvas clicks via inverse bilinear transformation, showing "Solid Building — Masked ($U = 0$)" when clicking inside obstacles.
- Transect tool: Update SVG line profile dynamically on layout or model change.

### Phase 6: Test Suite Expansion
- Add pytest tests for data provenance tags, route consistency, error calculation, mask enforcement, and dynamic metric binding.
