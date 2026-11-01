"""API routes for Urban Microclimate Modelling research application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from src.api.schemas import (
    CheckpointInventory,
    CFDInventory,
    DatasetInventory,
    ExperimentRecord,
    ExperimentRunRequest,
    LayoutDetail,
    LayoutSummary,
    ProbeQueryRequest,
    ProbeQueryResponse,
    SpatialFieldData,
    SurrogatePredictionRequest,
    SurrogatePredictionResponse,
    SystemOverview,
    TransectQueryRequest,
    TransectQueryResponse,
    UrbanTALESInventory,
    UrbanTALESDatasetSummary,
)
from src.api.service import service

router = APIRouter(prefix="/api")


@router.get("/health")
def get_health():
    """Health check endpoint indicating active PyTorch compute hardware."""
    return {
        "status": "healthy",
        "device": str(service.device),
        "corpus_size": len(service.samples),
        "models_count": len(service.models),
    }


@router.get("/overview", response_model=SystemOverview)
def get_system_overview():
    """Aggregates high-level scientific metrics driving the research dashboard."""
    completed_maes = [e["held_out_mae_mps"] for e in service.experiments if e.get("held_out_mae_mps") is not None]
    best_mae = min(completed_maes) if completed_maes else None

    distinct_families = len(set(s.metadata.layout_family.value for s in service.samples.values()))
    actual_count = len(service.samples)

    provenance_info = {
        "solver_status": "Synthetic Aerodynamic Benchmark (Castro & Robins 1977 formulation)",
        "active_cfd_case": "cfd/cases/simple2d_building_block (unexecuted template)",
        "real_rans_count": 0,
        "synthetic_fixture_count": actual_count,
        "target_dataset_size": 538,
        "is_demonstration_fixture": True,
    }

    return SystemOverview(
        domain="Computational Wind Engineering",
        active_dataset="Synthetic Benchmark Corpus (In-Memory)",
        dataset_version="v0.9.4-research",
        stack_info=f"PyTorch ({service.device}) • OpenFOAM Template (Unexecuted)",
        geometric_corpus_count=actual_count,
        corpus_target_total=538,
        corpus_provenance="SYNTHETIC_PROCEDURAL_FIXTURE",
        corpus_status_desc=f"{actual_count} Synthetic Benchmarks Loaded • Target UrbanTALES: 538 Layouts (Not Loaded)",
        topological_families_count=distinct_families,
        ground_truth_solvers_count=0,
        solvers_status_desc="0 Executed OpenFOAM Solutions • 24 Analytical Flow Benchmarks",
        total_solvers_target=538,
        mesh_resolution_desc="16,384 Cartesian cells (128x128)",
        model_checkpoints_count=len(service.models),
        model_architectures=["CompactUNet (ResBlock-4)", "CompactCNN (Dilated-2D)", "Random Forest (26D Morphometric)"],
        zero_leakage_mae_mps=best_mae,
        zero_leakage_change_pct=None,
        test_partition_desc="Held-out Family D (Procedural Holdout)",
        selected_layout_id="layout_042_staggered_canyon",
        backend_status="Connected (Synthetic Benchmark Mode)",
        device=str(service.device),
        is_demonstration_fixture=True,
        data_provenance=provenance_info,
    )


@router.get("/layouts", response_model=List[LayoutSummary])
def list_layouts():
    """Returns summaries for all available layout samples in the corpus."""
    summaries = []
    for lid, sample in service.samples.items():
        mask = sample.geometry.building_mask
        fluid_pct = float(np.sum(~mask) / mask.size) if hasattr(mask, "size") else 0.85
        is_held_out = "staggered" in lid.lower() or sample.metadata.layout_family.value == "idealized_staggered"

        features = service.feature_extractor.extract(sample)
        plan_area = float(features.plan_area_density_lambda_p)
        b_count = int(features.building_count)
        split = "test" if is_held_out else ("val" if "courtyard" in lid.lower() else "train")

        summaries.append(
            LayoutSummary(
                layout_id=sample.layout_id,
                layout_family=sample.metadata.layout_family.value,
                city_or_source=sample.metadata.city_or_source,
                domain_size_m=sample.metadata.domain_size_m,
                resolution_m=sample.metadata.resolution_m,
                grid_shape=sample.metadata.grid_shape,
                inflow_speed_mps=sample.environment.inflow_speed_mps,
                wind_direction_deg=sample.environment.wind_direction_deg,
                is_held_out=is_held_out,
                has_cfd_ground_truth=False,
                active_fluid_fraction=round(fluid_pct, 4),
                data_provenance="SYNTHETIC_PROCEDURAL_FIXTURE",
                provenance_label="Synthetic Procedural Fixture",
                num_buildings=b_count,
                plan_area_fraction=round(plan_area, 3),
                roughness_length_z0_m=sample.environment.roughness_length_z0_m,
                dataset_split=split,
            )
        )
    return summaries


@router.get("/layouts/{layout_id}", response_model=LayoutDetail)
def get_layout_detail(layout_id: str):
    """Retrieves full metadata and 26 morphometric features for a specific layout."""
    if layout_id not in service.samples:
        raise HTTPException(status_code=404, detail=f"Layout '{layout_id}' not found.")

    sample = service.samples[layout_id]
    features = service.feature_extractor.extract(sample)
    feature_dict = {name: float(val) for name, val in zip(features.feature_names(), features.to_feature_vector())}

    # Extract distinct building heights
    h_map = sample.geometry.height_map
    unique_h = sorted(list(set(float(h) for h in h_map[sample.geometry.building_mask])))

    return LayoutDetail(
        layout_id=sample.layout_id,
        layout_family=sample.metadata.layout_family.value,
        city_or_source=sample.metadata.city_or_source,
        domain_size_m=sample.metadata.domain_size_m,
        resolution_m=sample.metadata.resolution_m,
        grid_shape=sample.metadata.grid_shape,
        inflow_speed_mps=sample.environment.inflow_speed_mps,
        wind_direction_deg=sample.environment.wind_direction_deg,
        reference_height_m=sample.environment.reference_height_m,
        roughness_length_z0_m=sample.environment.roughness_length_z0_m,
        cfd_solver_name=sample.cfd_solver.solver_name,
        turbulence_model=sample.cfd_solver.turbulence_model,
        is_held_out=("staggered" in layout_id.lower()),
        geometric_features=feature_dict,
        buildings_count=int(features.building_count),
        building_heights=unique_h,
        has_cfd_ground_truth=False,
        data_provenance="SYNTHETIC_PROCEDURAL_FIXTURE",
        provenance_label="Synthetic Aerodynamic Benchmark (Castro & Robins 1977 formulation)",
    )


@router.get("/layouts/{layout_id}/fields", response_model=SpatialFieldData)
def get_layout_fields(layout_id: str):
    """Retrieves 2D spatial grid arrays for scientific visualization."""
    if layout_id not in service.samples:
        raise HTTPException(status_code=404, detail=f"Layout '{layout_id}' not found.")

    sample = service.samples[layout_id]
    ny, nx = sample.metadata.grid_shape
    lx, ly, _ = sample.metadata.domain_size_m
    dx, dy, _ = sample.metadata.resolution_m

    mask = sample.geometry.building_mask
    h_map = sample.geometry.height_map
    ped_wind = sample.targets.pedestrian_wind_speed
    valid = ~mask

    stats = {
        "min": float(np.min(ped_wind[valid])) if np.any(valid) else 0.0,
        "mean": float(np.mean(ped_wind[valid])) if np.any(valid) else 0.0,
        "max": float(np.max(ped_wind[valid])) if np.any(valid) else 0.0,
    }

    # Building bounding boxes for crisp vector overlay rendering
    building_boxes = []
    if layout_id == "layout_042_staggered_canyon":
        building_boxes = [
            {"id": "B1", "h": 12.0, "x_min": 40.0, "x_max": 75.0, "y_min": 120.0, "y_max": 160.0},
            {"id": "B2", "h": 12.0, "x_min": 40.0, "x_max": 75.0, "y_min": 15.0, "y_max": 42.0},
            {"id": "B3", "h": 45.0, "x_min": 120.0, "x_max": 155.0, "y_min": 120.0, "y_max": 160.0},
            {"id": "B4", "h": 45.0, "x_min": 120.0, "x_max": 155.0, "y_min": 15.0, "y_max": 42.0},
        ]

    return SpatialFieldData(
        layout_id=sample.layout_id,
        grid_shape=(ny, nx),
        domain_extent_m=(lx, ly),
        dx_m=round(dx, 3),
        dy_m=round(dy, 3),
        height_map=h_map.flatten().tolist(),
        building_mask=mask.flatten().tolist(),
        cfd_velocity_mag=ped_wind.flatten().tolist(),
        cfd_velocity_u=sample.targets.velocity_u.flatten().tolist() if sample.targets.velocity_u is not None else None,
        cfd_velocity_v=sample.targets.velocity_v.flatten().tolist() if sample.targets.velocity_v is not None else None,
        cfd_tke=sample.targets.turbulent_kinetic_energy.flatten().tolist() if sample.targets.turbulent_kinetic_energy is not None else None,
        stats=stats,
        building_boxes=building_boxes,
        has_cfd_ground_truth=False,
        data_provenance="CANONICAL_AERODYNAMIC_BENCHMARK",
        provenance_description="Analytical Aerodynamic Benchmark Reference Field (Castro & Robins 1977 formulation)",
        has_velocity_components=True,
    )


@router.post("/surrogates/predict", response_model=SurrogatePredictionResponse)
def predict_surrogate(req: SurrogatePredictionRequest):
    """Executes neural surrogate or AML baseline inference on a layout."""
    try:
        res = service.predict_surrogate(
            layout_id=req.layout_id,
            model_id=req.model_id,
            scalar_field=req.scalar_field,
        )
        return SurrogatePredictionResponse(**res)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/layouts/{layout_id}/probe", response_model=ProbeQueryResponse)
def query_probe_get(
    layout_id: str,
    x: float = Query(..., description="X coordinate in meters"),
    y: float = Query(..., description="Y coordinate in meters"),
    model_id: str = Query("unet-v2-resblock4", description="Surrogate model identifier"),
):
    """Query flow values at specific physical coordinates (GET)."""
    if layout_id not in service.samples:
        raise HTTPException(status_code=404, detail=f"Layout '{layout_id}' not found.")
    res = service.query_probe(layout_id=layout_id, x_m=x, y_m=y, model_id=model_id)
    return ProbeQueryResponse(**res)


@router.post("/layouts/{layout_id}/probe", response_model=ProbeQueryResponse)
def query_probe_post(layout_id: str, req: ProbeQueryRequest):
    """Query flow values at specific physical coordinates (POST)."""
    if layout_id not in service.samples:
        raise HTTPException(status_code=404, detail=f"Layout '{layout_id}' not found.")
    res = service.query_probe(layout_id=layout_id, x_m=req.x_m, y_m=req.y_m, model_id=req.model_id or "unet-v2-resblock4")
    return ProbeQueryResponse(**res)


@router.get("/layouts/{layout_id}/transect", response_model=TransectQueryResponse)
def query_transect(
    layout_id: str,
    axis: str = Query("y", description="Axis of cut: 'y' for horizontal cut at fixed Y, 'x' for vertical cut at fixed X"),
    coord: float = Query(50.0, description="Fixed coordinate in meters (e.g. Y=50.0m)"),
    samples: int = Query(200, description="Number of regular sampling points along the cut"),
    model_id: str = Query("unet-v2-resblock4", description="Surrogate model identifier"),
):
    """Extracts 1D transect profile along a cut plane."""
    if layout_id not in service.samples:
        raise HTTPException(status_code=404, detail=f"Layout '{layout_id}' not found.")
    res = service.query_transect(
        layout_id=layout_id,
        axis=axis,
        coordinate_m=coord,
        num_samples=samples,
        model_id=model_id,
    )
    return TransectQueryResponse(**res)


@router.get("/experiments", response_model=List[ExperimentRecord])
def list_experiments():
    """Returns generalization benchmark experiment records."""
    with service.lock:
        return [ExperimentRecord(**exp) for exp in service.experiments]


@router.post("/experiments/run")
def run_experiment(req: ExperimentRunRequest):
    """Launches an asynchronous training/evaluation experiment."""
    exp_id = service.run_experiment_background(
        architecture=req.architecture,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        holdout_family=req.holdout_family,
    )
    return {"status": "started", "experiment_id": exp_id, "message": f"Experiment {exp_id} launched in background."}


@router.get("/export/triplet/{layout_id}")
def export_triplet(layout_id: str, model_id: str = "unet-v2-resblock4"):
    """Generates and downloads publication-quality 3-panel figure."""
    if layout_id not in service.samples:
        raise HTTPException(status_code=404, detail=f"Layout '{layout_id}' not found.")
    fig_path = service.export_triplet_figure(layout_id, model_id=model_id)
    return FileResponse(
        path=str(fig_path),
        media_type="image/png",
        filename=f"microclimate_triplet_{layout_id}_{model_id}.png",
    )


# ============================================================================
# Inventory Endpoints (P1)
# ============================================================================

@router.get("/inventory/datasets", response_model=DatasetInventory)
def get_dataset_inventory():
    """Returns a structured inventory of all loaded layouts, their provenance, and availability."""
    return DatasetInventory(**service.get_dataset_inventory())


@router.get("/inventory/checkpoints", response_model=CheckpointInventory)
def get_checkpoint_inventory():
    """Scans checkpoints/ directory and reports training status of each .pt file."""
    return CheckpointInventory(**service.get_checkpoint_inventory())


@router.get("/inventory/cfd", response_model=CFDInventory)
def get_cfd_inventory():
    """Reports available OpenFOAM case templates, solver availability, and pre-computed outputs."""
    return CFDInventory(**service.get_cfd_inventory())


@router.get("/inventory/urbantales", response_model=UrbanTALESInventory)
def get_urbantales_inventory():
    """Read-only inventory of the real UrbanTALES sample(s) shipped on disk.

    Loads ``UA0625_d00`` (topology + vertical profile + geometry image) on
    demand and reports honest provenance. This is geometry and a 1D vertical
    profile only -- no 2D wind field is fabricated and no grid spacing is
    assumed. The synthetic corpus and existing endpoints are untouched.
    """
    from src.data.ua0625 import default_ua0625_paths, load_ua0625_dataset

    paths = default_ua0625_paths(Path(__file__).resolve().parent.parent.parent)
    topo_path = paths["topology"]
    profile_path = paths["profile"]
    image_path = paths["image"]

    if not (topo_path.exists() and profile_path.exists()):
        return UrbanTALESInventory(
            available=False,
            datasets=[],
            provenance_source="UrbanTALES (not loaded)",
            notes="UA0625_d00 source files not found in repository root.",
        )

    try:
        ds = load_ua0625_dataset(project_root=Path(__file__).resolve().parent.parent.parent)
    except Exception as exc:  # pragma: no cover - defensive: never 500 on a read-only probe
        return UrbanTALESInventory(
            available=False,
            datasets=[],
            notes=f"Failed to load UA0625_d00 dataset: {exc}",
        )

    geom = ds.topology.geometry_summary()
    summary = UrbanTALESDatasetSummary(
        dataset_id=ds.dataset_id,
        has_geometry=ds.provenance.has_geometry,
        has_vertical_profile=ds.provenance.has_vertical_profile,
        has_2d_wind_field=ds.provenance.has_2d_wind_field,
        has_geometry_image=ds.image is not None,
        topology_grid_shape=ds.topology.grid_shape,
        plan_area_density=round(ds.topology.plan_area_density, 6),
        building_height_values_m=[float(h) for h in ds.topology.building_height_values],
        building_count=int(geom["building_count"]),
        profile_num_levels=ds.profile.num_levels,
        profile_z_min_m=round(ds.profile.z_min, 4),
        profile_z_max_m=round(ds.profile.z_max, 4),
        resolution_m=ds.topology.resolution_m,
        domain_size_m=ds.topology.domain_size_m,
        wind_direction_label_deg=ds.provenance.wind_direction_label_deg,
        limitations=ds.limitations,
    )
    return UrbanTALESInventory(
        available=True,
        datasets=[summary],
        provenance_source=ds.provenance.source,
        license=ds.provenance.license,
        locally_simulated=ds.provenance.locally_simulated,
        notes=(
            "Real UrbanTALES LES sample (geometry + 1D vertical profile). "
            "No 2D wind field is available/fabricated; grid spacing is unknown."
        ),
    )
