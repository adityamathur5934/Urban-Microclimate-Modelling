"""Pydantic schemas for the Urban Microclimate Scientific API with strict scientific data provenance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class LayoutSummary(BaseModel):
    """Concise metadata summary for a layout in list views."""
    layout_id: str
    layout_family: str
    city_or_source: str
    domain_size_m: Tuple[float, float, float]
    resolution_m: Tuple[float, float, float]
    grid_shape: Tuple[int, int]
    inflow_speed_mps: float
    wind_direction_deg: float
    is_held_out: bool = False
    has_cfd_ground_truth: bool = False
    active_fluid_fraction: float = 0.85
    data_provenance: str = "SYNTHETIC_PROCEDURAL_FIXTURE"
    provenance_label: str = "Synthetic Procedural Fixture"
    # Fields aligned with UI catalog card
    num_buildings: int = 0
    plan_area_fraction: float = 0.0
    roughness_length_z0_m: float = 0.05
    dataset_split: str = "train"


class LayoutDetail(BaseModel):
    """Detailed layout information including geometric features."""
    layout_id: str
    layout_family: str
    city_or_source: str
    domain_size_m: Tuple[float, float, float]
    resolution_m: Tuple[float, float, float]
    grid_shape: Tuple[int, int]
    inflow_speed_mps: float
    wind_direction_deg: float
    reference_height_m: float
    roughness_length_z0_m: float
    cfd_solver_name: str
    turbulence_model: str
    is_held_out: bool = False
    geometric_features: Dict[str, float] = Field(default_factory=dict)
    buildings_count: int = 0
    building_heights: List[float] = Field(default_factory=list)
    has_cfd_ground_truth: bool = False
    data_provenance: str = "SYNTHETIC_PROCEDURAL_FIXTURE"
    provenance_label: str = "Synthetic Aerodynamic Benchmark (Castro & Robins 1977 formulation)"


class SpatialFieldData(BaseModel):
    """2D spatial grid fields and bounds for rendering."""
    layout_id: str
    grid_shape: Tuple[int, int]  # (Ny, Nx)
    domain_extent_m: Tuple[float, float]  # (Lx, Ly)
    dx_m: float
    dy_m: float
    # Arrays serialized as row-major flat lists for efficient JSON transfer
    height_map: List[float]
    building_mask: List[bool]
    cfd_velocity_mag: List[float]
    cfd_velocity_u: Optional[List[float]] = None
    cfd_velocity_v: Optional[List[float]] = None
    cfd_tke: Optional[List[float]] = None
    # Pre-calculated field statistics
    stats: Dict[str, float]
    building_boxes: List[Dict[str, Any]] = Field(default_factory=list)
    has_cfd_ground_truth: bool = False
    data_provenance: str = "CANONICAL_AERODYNAMIC_BENCHMARK"
    provenance_description: str = "Analytical Aerodynamic Benchmark Reference Field (Castro & Robins 1977 formulation)"
    has_velocity_components: bool = True


class SurrogatePredictionRequest(BaseModel):
    """Request payload to run surrogate inference on a layout."""
    layout_id: str
    model_id: str = "unet-v2-resblock4"
    scalar_field: str = "velocity_magnitude"  # "velocity_magnitude" supported


class SurrogatePredictionResponse(BaseModel):
    """Inference output with predicted field, error field, and validation metrics."""
    layout_id: str
    model_id: str
    scalar_field: str
    grid_shape: Tuple[int, int]
    # Serialized predicted field and absolute error
    predicted_field: List[float]
    abs_error_field: List[float]
    # Metrics
    field_mae: float
    field_rmse: float
    field_nrmse: float  # Percentage (e.g. 3.04)
    peak_local_error: float
    cosine_similarity: float
    r2_score: float
    valid_fluid_cells: int
    total_cells: int
    active_fluid_fraction: float
    inference_latency_ms: float
    speedup_factor: Optional[float] = None
    speedup_notes: Optional[str] = None
    device: str = "cpu"
    data_provenance: str = "SURROGATE_PREDICTION"
    is_trained_checkpoint: bool = False
    prediction_type: str = "spatially_resolved_neural"  # or "scalar_tabular_baseline"
    stats_pred: Dict[str, float]
    stats_err: Dict[str, float]


class ProbeQueryRequest(BaseModel):
    """Query flow values at specific physical coordinates."""
    layout_id: str
    x_m: float
    y_m: float
    model_id: Optional[str] = "unet-v2-resblock4"


class ProbeQueryResponse(BaseModel):
    """Interpolated values at probe point."""
    layout_id: str
    x_m: float
    y_m: float
    is_inside_building: bool
    status_label: str = "Valid Fluid Cell"
    cfd_velocity_mps: float
    surrogate_velocity_mps: float
    abs_residual_mps: float
    relative_residual_pct: float
    inflow_speed_mps: float
    data_provenance: str = "CANONICAL_AERODYNAMIC_BENCHMARK"


class TransectQueryRequest(BaseModel):
    """Query 1D slice along a coordinate line."""
    layout_id: str
    axis: str = "y"  # "y" = horizontal line at fixed Y; "x" = vertical line at fixed X
    coordinate_m: float = 50.0  # Fixed coordinate
    num_samples: int = 200
    model_id: Optional[str] = "unet-v2-resblock4"


class TransectQueryResponse(BaseModel):
    """1D profiles along transect line."""
    layout_id: str
    axis: str
    fixed_coord_m: float
    coords_m: List[float]
    cfd_values: List[float]
    surrogate_values: List[float]
    building_intervals: List[Tuple[float, float]]  # [(x_start, x_end), ...]
    max_divergence_mps: float
    max_divergence_coord_m: float
    local_jet_match_pct: float
    data_provenance: str = "CANONICAL_AERODYNAMIC_BENCHMARK"


class ExperimentRecord(BaseModel):
    """Record for generalization benchmark table."""
    experiment_id: str
    architecture: str
    model_details: str
    dataset_split: str
    inflow_condition: str
    held_out_mae_mps: Optional[float] = None
    max_abs_error_mps: Optional[float] = None
    status: str  # "Completed", "Training", "Queued", "Baseline"
    training_progress_pct: Optional[float] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    is_implemented: bool = True
    provenance: str = "SYNTHETIC_TEST_FIXTURE"
    created_at: str


class ExperimentRunRequest(BaseModel):
    """Request to launch a new training or evaluation experiment."""
    architecture: str = "CompactUNet"  # "CompactUNet", "CompactCNN", "RandomForest"
    epochs: int = 50
    learning_rate: float = 1e-3
    holdout_family: str = "idealized_staggered"
    batch_size: int = 8
    seed: int = 42


# ---------------------------------------------------------------------------
# Inventory / Status Schemas (P1)
# ---------------------------------------------------------------------------

class DatasetInventoryItem(BaseModel):
    """Describes one entry in the dataset corpus."""
    layout_id: str
    layout_family: str
    data_provenance: str
    has_cfd_ground_truth: bool = False
    domain_size_m: Tuple[float, float, float]
    grid_shape: Tuple[int, int]
    inflow_speed_mps: float
    source_file: Optional[str] = None


class DatasetInventory(BaseModel):
    """Summary of all datasets available in-memory or on disk."""
    total_loaded: int
    synthetic_count: int
    canonical_benchmark_count: int
    cfd_solver_output_count: int
    items: List[DatasetInventoryItem]
    notes: str = ""


class CheckpointInfo(BaseModel):
    """Describes one model checkpoint on disk."""
    filename: str
    model_architecture: Optional[str] = None
    is_trained: bool
    epoch: Optional[int] = None
    val_mae_mps: Optional[float] = None
    val_nrmse_pct: Optional[float] = None
    file_size_kb: Optional[float] = None
    provenance: str = "UNKNOWN"
    notes: str = ""


class CheckpointInventory(BaseModel):
    """All checkpoint files found in the checkpoints directory."""
    checkpoint_dir: str
    total_found: int
    trained_count: int
    items: List[CheckpointInfo]


class CFDCaseInfo(BaseModel):
    """Describes one OpenFOAM case template or solver output."""
    case_id: str
    path: str
    case_type: str  # "template", "solver_output"
    solver_available: bool = False
    solver_name: Optional[str] = None
    has_reference_output: bool = False
    output_files: List[str] = []
    notes: str = ""


class CFDInventory(BaseModel):
    """OpenFOAM case templates and solver output status."""
    openfoam_available: bool
    solver_binary: Optional[str] = None
    total_cases: int
    cases: List[CFDCaseInfo]
    notes: str = ""


class ProvenanceInfo(BaseModel):
    """Nested provenance summary for overview banner and transparency."""
    solver_status: str
    active_cfd_case: str
    real_rans_count: int = 0
    synthetic_fixture_count: int = 24
    target_dataset_size: int = 538
    is_demonstration_fixture: bool = True


class SystemOverview(BaseModel):
    """High-level system metrics driving the dashboard header."""
    domain: str = "Computational Wind Engineering"
    active_dataset: str = "Synthetic Benchmark Corpus (In-Memory)"
    dataset_version: str = "v0.9.4-research"
    stack_info: str = "PyTorch • OpenFOAM Template"
    geometric_corpus_count: int
    corpus_target_total: int = 538
    corpus_provenance: str = "SYNTHETIC_PROCEDURAL_FIXTURE"
    corpus_status_desc: str = "24 Synthetic Benchmarks Loaded • Target UrbanTALES: 538 Layouts (Not Loaded)"
    topological_families_count: int
    ground_truth_solvers_count: int = 0
    solvers_status_desc: str = "0 Executed OpenFOAM Solutions • 24 Analytical Flow Benchmarks"
    total_solvers_target: int = 538
    mesh_resolution_desc: str = "16,384 Cartesian cells (128x128)"
    model_checkpoints_count: int
    model_architectures: List[str]
    zero_leakage_mae_mps: Optional[float] = None
    zero_leakage_change_pct: Optional[float] = None
    test_partition_desc: str = "Held-out Family D (Procedural)"
    selected_layout_id: str = "layout_042_staggered_canyon"
    backend_status: str = "Connected (Synthetic/Benchmark Mode)"
    device: str = "cpu"
    is_demonstration_fixture: bool = True
    data_provenance: Optional[ProvenanceInfo] = None


# ---------------------------------------------------------------------------
# Real UrbanTALES dataset inventory (UA0625_d00 geometry + vertical profile)
# ---------------------------------------------------------------------------

class UrbanTALESDatasetSummary(BaseModel):
    """Summary of one real UrbanTALES sample loaded from disk (UA0625_d00).

    This is geometry + a 1D vertical profile only -- no 2D wind field, and no
    assumed grid spacing. Fields mirror the honest provenance exposed by
    ``src.data.ua0625``.
    """
    dataset_id: str
    has_geometry: bool
    has_vertical_profile: bool
    has_2d_wind_field: bool = False
    has_geometry_image: bool = False
    topology_grid_shape: Tuple[int, int]
    plan_area_density: float
    building_height_values_m: List[float] = Field(default_factory=list)
    building_count: int = 0
    profile_num_levels: int
    profile_z_min_m: float
    profile_z_max_m: float
    resolution_m: Optional[Tuple[float, float, float]] = None
    domain_size_m: Optional[Tuple[float, float, float]] = None
    wind_direction_label_deg: Optional[float] = None
    limitations: List[str] = Field(default_factory=list)


class UrbanTALESInventory(BaseModel):
    """Inventory of real UrbanTALES samples available on disk (read-only)."""
    available: bool
    datasets: List[UrbanTALESDatasetSummary] = Field(default_factory=list)
    provenance_source: str = ""
    license: str = ""
    locally_simulated: bool = False
    notes: str = ""
