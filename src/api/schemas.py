"""Pydantic schemas for the Urban Microclimate Scientific API."""

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
    has_cfd_ground_truth: bool = True
    active_fluid_fraction: float = 0.85


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


class SurrogatePredictionRequest(BaseModel):
    """Request payload to run surrogate inference on a layout."""
    layout_id: str
    model_id: str = "unet-v2-resblock4"
    scalar_field: str = "velocity_magnitude"  # "velocity_magnitude", "velocity_u", "velocity_v", "tke"


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
    speedup_factor: float
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
    cfd_velocity_mps: float
    surrogate_velocity_mps: float
    abs_residual_mps: float
    relative_residual_pct: float
    inflow_speed_mps: float


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
    created_at: str


class ExperimentRunRequest(BaseModel):
    """Request to launch a new training or evaluation experiment."""
    architecture: str = "CompactUNet"  # "CompactUNet", "CompactCNN", "RandomForest"
    epochs: int = 50
    learning_rate: float = 1e-3
    holdout_family: str = "idealized_staggered"
    batch_size: int = 8
    seed: int = 42


class SystemOverview(BaseModel):
    """High-level system metrics driving the dashboard header."""
    domain: str = "Computational Wind Engineering"
    active_dataset: str = "UrbanFlow-2D v1.4"
    dataset_version: str = "v0.9.4-research"
    stack_info: str = "PyTorch 2.3 • OpenFOAM"
    geometric_corpus_count: int
    topological_families_count: int
    ground_truth_solvers_count: int
    total_solvers_target: int
    mesh_resolution_desc: str
    model_checkpoints_count: int
    model_architectures: List[str]
    zero_leakage_mae_mps: float
    zero_leakage_change_pct: float
    test_partition_desc: str
    selected_layout_id: str
    backend_status: str = "Connected"
    device: str = "cpu"
