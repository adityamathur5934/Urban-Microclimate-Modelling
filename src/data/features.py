"""Geometric summary feature extraction for traditional ML baselines.

Feature groups
--------------
1. Existing (11 features) — plan/frontal/wall densities, height mean/std/max,
   MacDonald z0 & d, inflow speed and direction components.
2. Height distribution quantiles (5 new) — min, p25, p75, IQR, CV.
3. Building count and footprint statistics (5 new) — count, mean/std/max
   footprint area, footprint CV.  Derived via scipy.ndimage.label; each
   connected component of building_mask is treated as one building.
4. Layout extent and domain summary (3 new) — domain aspect ratio,
   total built area, and the product mean_height * lambda_p.
5. Environmental boundary inputs (2 new) — reference_height_m and terrain
   roughness_z0_env_m (the *input* z0, distinct from the MacDonald-derived
   aerodynamic roughness length).

Deferred features (not implemented — data not available in raster schema)
------------------------------------------------------------------------
- Street / canyon orientation histograms: require vector street-network
  geometry not present in height_map + building_mask rasters.
- Per-building spacing / gap statistics: require polygon centroids or
  Euclidean distance transforms between labeled objects; deferred for a
  future iteration once building-polygon IDs are part of the schema.
- Sky view factor (SVF): requires 3D ray-casting from height_map.
- Building aspect ratio (H/W): requires per-building width from polygon.

See docs/geometric_features.md for full formulae, units, edge-case
handling, leakage audit, and interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.ndimage import label as ndimage_label

from src.data.contract import UrbanLayoutSample


# ---------------------------------------------------------------------------
# Helper: connected-component footprint statistics
# ---------------------------------------------------------------------------

def _building_component_stats(
    mask: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[int, float, float, float, float]:
    """Return (count, mean_area, std_area, max_area, cv_area) for building components.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array (Ny, Nx); True where building is present.
    dx, dy : float
        Cell spacing in x (columns) and y (rows) directions, metres.

    Returns
    -------
    count : int
        Number of connected building components (0 if no buildings).
    mean_area_m2 : float
        Mean component footprint area [m²]. 0.0 if count == 0.
    std_area_m2 : float
        Std-dev of component footprint areas [m²]. 0.0 if count <= 1.
    max_area_m2 : float
        Maximum component footprint area [m²]. 0.0 if count == 0.
    cv_area : float
        Coefficient of variation (std / mean) of footprint areas [-].
        0.0 if count <= 1 or mean_area == 0.
    """
    cell_area = dx * dy  # m²

    if not np.any(mask):
        return 0, 0.0, 0.0, 0.0, 0.0

    labeled, n_labels = ndimage_label(mask)
    if n_labels == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    # Count cells per label (labels are 1..n_labels; 0 = background)
    component_cell_counts = np.bincount(labeled.ravel())[1:]  # drop background
    component_areas = component_cell_counts.astype(np.float64) * cell_area

    mean_a = float(np.mean(component_areas))
    std_a = float(np.std(component_areas)) if n_labels > 1 else 0.0
    max_a = float(np.max(component_areas))
    cv_a = float(std_a / mean_a) if mean_a > 1e-9 else 0.0

    return n_labels, mean_a, std_a, max_a, cv_a


# ---------------------------------------------------------------------------
# GeometricFeatures dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeometricFeatures:
    """Summary morphometric and environmental features for one urban layout.

    layout_id and target_area_averaged_wind_speed are stored as metadata
    but are EXCLUDED from to_feature_vector() to prevent target leakage and
    accidental use of string IDs as numeric predictors.
    """
    # --- Metadata (not in feature vector) ---
    layout_id: str

    # --- Group 1: Existing plan/wall/frontal density features ---
    plan_area_density_lambda_p: float       # [-]   built footprint / total area
    frontal_area_density_lambda_f: float    # [-]   wind-projected frontal area / total area
    mean_building_height_m: float           # [m]   mean height over built cells
    std_building_height_m: float            # [m]   std-dev height over built cells
    max_building_height_m: float            # [m]   maximum height over built cells
    wall_area_density_lambda_w: float       # [-]   total facade area / total plan area
    roughness_length_macdonald_m: float     # [m]   aerodynamic z0 (MacDonald 1998)
    displacement_height_d_m: float          # [m]   displacement height d (MacDonald 1998)
    inflow_speed_mps: float                 # [m/s] reference inflow speed U_ref
    wind_direction_cos: float               # [-]   cos(wind_direction_deg)
    wind_direction_sin: float               # [-]   sin(wind_direction_deg)

    # --- Group 2: Height distribution quantiles (NEW) ---
    min_building_height_m: float            # [m]   minimum height over built cells
    p25_building_height_m: float            # [m]   25th percentile height over built cells
    p75_building_height_m: float            # [m]   75th percentile height over built cells
    height_iqr_m: float                     # [m]   interquartile range (p75 - p25)
    height_cv: float                        # [-]   coefficient of variation (std / mean)

    # --- Group 3: Building count and footprint statistics (NEW) ---
    building_count: float                   # [-]   number of connected building components
    mean_footprint_area_m2: float           # [m²]  mean per-building footprint area
    std_footprint_area_m2: float            # [m²]  std-dev per-building footprint area
    max_footprint_area_m2: float            # [m²]  largest single building footprint
    footprint_area_cv: float                # [-]   CV of footprint areas

    # --- Group 4: Layout extent and domain summary (NEW) ---
    domain_aspect_ratio: float              # [-]   Lx / Ly (domain width / depth)
    built_area_total_m2: float              # [m²]  total built footprint area
    mean_height_times_lambda_p: float       # [m]   H_mean * lambda_p (canopy volume proxy)

    # --- Group 5: Environmental boundary inputs (NEW) ---
    reference_height_m: float              # [m]   height at which U_ref is defined
    roughness_z0_env_m: float              # [m]   terrain surface roughness z0 (input BC)

    # --- Target scalar (metadata only — NOT in feature vector) ---
    target_area_averaged_wind_speed: float  # [m/s] outdoor area-averaged wind speed

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_feature_vector(self) -> np.ndarray:
        """Returns the predictive feature vector [X] as float32.

        Excludes: layout_id (string metadata) and
                  target_area_averaged_wind_speed (target — would be leakage).
        The order here MUST match feature_names() exactly.
        """
        return np.array([
            # Group 1
            self.plan_area_density_lambda_p,
            self.frontal_area_density_lambda_f,
            self.mean_building_height_m,
            self.std_building_height_m,
            self.max_building_height_m,
            self.wall_area_density_lambda_w,
            self.roughness_length_macdonald_m,
            self.displacement_height_d_m,
            self.inflow_speed_mps,
            self.wind_direction_cos,
            self.wind_direction_sin,
            # Group 2
            self.min_building_height_m,
            self.p25_building_height_m,
            self.p75_building_height_m,
            self.height_iqr_m,
            self.height_cv,
            # Group 3
            self.building_count,
            self.mean_footprint_area_m2,
            self.std_footprint_area_m2,
            self.max_footprint_area_m2,
            self.footprint_area_cv,
            # Group 4
            self.domain_aspect_ratio,
            self.built_area_total_m2,
            self.mean_height_times_lambda_p,
            # Group 5
            self.reference_height_m,
            self.roughness_z0_env_m,
        ], dtype=np.float32)

    @classmethod
    def feature_names(cls) -> List[str]:
        """Returns the ordered list of feature names matching to_feature_vector().

        layout_id and target_area_averaged_wind_speed are intentionally absent.
        """
        return [
            # Group 1 — existing
            "plan_area_density_lambda_p",
            "frontal_area_density_lambda_f",
            "mean_building_height_m",
            "std_building_height_m",
            "max_building_height_m",
            "wall_area_density_lambda_w",
            "roughness_length_macdonald_m",
            "displacement_height_d_m",
            "inflow_speed_mps",
            "wind_direction_cos",
            "wind_direction_sin",
            # Group 2 — height quantiles (new)
            "min_building_height_m",
            "p25_building_height_m",
            "p75_building_height_m",
            "height_iqr_m",
            "height_cv",
            # Group 3 — building count and footprint (new)
            "building_count",
            "mean_footprint_area_m2",
            "std_footprint_area_m2",
            "max_footprint_area_m2",
            "footprint_area_cv",
            # Group 4 — domain extent (new)
            "domain_aspect_ratio",
            "built_area_total_m2",
            "mean_height_times_lambda_p",
            # Group 5 — environmental boundary inputs (new)
            "reference_height_m",
            "roughness_z0_env_m",
        ]


# ---------------------------------------------------------------------------
# GeometricFeatureExtractor
# ---------------------------------------------------------------------------

class GeometricFeatureExtractor:
    """Extracts scalar morphometric features from layout geometry and inflow conditions.

    Parameters
    ----------
    cd_drag : float
        Drag coefficient for the MacDonald (1998) roughness parameterisation.
        Default 1.2 (bluff body buildings).
    von_karman_kappa : float
        von Kármán constant. Default 0.40.
    """

    def __init__(self, cd_drag: float = 1.2, von_karman_kappa: float = 0.40):
        self.cd_drag = cd_drag
        self.kappa = von_karman_kappa

    def extract(self, sample: UrbanLayoutSample) -> GeometricFeatures:
        """Compute all summary features for an UrbanLayoutSample.

        This method is deterministic: identical inputs always produce
        identical outputs.  No random state is used.

        Parameters
        ----------
        sample : UrbanLayoutSample
            A validated sample conforming to the data contract.

        Returns
        -------
        GeometricFeatures
            Frozen dataclass with all features populated.

        Notes
        -----
        - layout_id is propagated as metadata only.
        - target_area_averaged_wind_speed is derived from
          sample.targets (a target field), but stored separately
          and excluded from to_feature_vector() to prevent leakage.
        - All new features are derived exclusively from GeometryData
          (height_map, building_mask) and EnvironmentalConditions
          (inflow_speed_mps, wind_direction_deg, reference_height_m,
           roughness_length_z0_m), all of which are available at
          prediction time.
        """
        height_map = sample.geometry.height_map          # (Ny, Nx) float32
        mask = sample.geometry.building_mask             # (Ny, Nx) bool
        dx, dy, _ = sample.metadata.resolution_m        # cell spacings (m)
        lx, ly, _ = sample.metadata.domain_size_m       # domain extents (m)
        total_cells = height_map.size
        total_area = float(total_cells) * dx * dy        # m²

        # ----------------------------------------------------------------
        # GROUP 1 — Plan / wall / frontal area densities
        # ----------------------------------------------------------------

        # 1a. Plan area density λ_p = built_cells / total_cells
        building_cells = int(np.count_nonzero(mask))
        lambda_p = float(building_cells) / max(total_cells, 1)

        # 1b. Height statistics over built cells only
        built_heights = height_map[mask]
        if len(built_heights) > 0:
            mean_h  = float(np.mean(built_heights))
            std_h   = float(np.std(built_heights))
            max_h   = float(np.max(built_heights))
            min_h   = float(np.min(built_heights))
            p25_h   = float(np.percentile(built_heights, 25))
            p75_h   = float(np.percentile(built_heights, 75))
        else:
            mean_h = std_h = max_h = min_h = p25_h = p75_h = 0.0

        # 1c. Wind-direction components
        wind_deg = sample.environment.wind_direction_deg
        rad = np.deg2rad(wind_deg)
        cos_dir = float(np.cos(rad))
        sin_dir = float(np.sin(rad))

        # 1d. Frontal area density λ_f
        #   Step increases along x and y, projected onto wind direction.
        diff_x = np.diff(height_map, axis=1)
        frontal_x = float(np.sum(np.maximum(0.0, diff_x))) * dy
        frontal_x += float(np.sum(height_map[:, 0])) * dy   # left-boundary contribution

        diff_y = np.diff(height_map, axis=0)
        frontal_y = float(np.sum(np.maximum(0.0, diff_y))) * dx
        frontal_y += float(np.sum(height_map[0, :])) * dx   # bottom-boundary contribution

        eff_frontal_area = abs(cos_dir) * frontal_x + abs(sin_dir) * frontal_y
        lambda_f = eff_frontal_area / max(total_area, 1e-9)

        # 1e. Wall area density λ_w (total vertical facade / total plan area)
        wall_x = (float(np.sum(np.abs(diff_x))) * dy
                  + float(np.sum(height_map[:, 0] + height_map[:, -1])) * dy)
        wall_y = (float(np.sum(np.abs(diff_y))) * dx
                  + float(np.sum(height_map[0, :] + height_map[-1, :])) * dx)
        lambda_w = (wall_x + wall_y) / max(total_area, 1e-9)

        # ----------------------------------------------------------------
        # GROUP 1 — MacDonald (1998) canopy parameterisation
        # ----------------------------------------------------------------

        # Displacement height d
        #   d/H = 1 + 4^(-λ_p) * (λ_p - 1),  clamped to [0, 0.95H]
        if mean_h > 0.0 and lambda_p > 0.0:
            d_over_h = 1.0 + (4.0 ** (-lambda_p)) * (lambda_p - 1.0)
            d_val = mean_h * max(0.0, min(d_over_h, 0.95))
        else:
            d_val = 0.0

        # Aerodynamic roughness length z0 (MacDonald 1998)
        #   z0/H = (1 - d/H) * exp(-(0.5 * β * Cd/κ² * (1 - d/H) * λ_f)^(-0.5))
        if mean_h > 0.0 and lambda_f > 1e-4 and (mean_h - d_val) > 1e-4:
            term = (0.5 * 1.0 * (self.cd_drag / (self.kappa ** 2))
                    * (1.0 - d_val / mean_h) * lambda_f)
            z0_val = (float((mean_h - d_val) * np.exp(-1.0 / np.sqrt(term)))
                      if term > 1e-4
                      else float(sample.environment.roughness_length_z0_m))
        else:
            z0_val = float(sample.environment.roughness_length_z0_m)

        # ----------------------------------------------------------------
        # GROUP 2 — Height distribution quantiles (NEW)
        # ----------------------------------------------------------------
        iqr_h = p75_h - p25_h
        cv_h  = (std_h / mean_h) if mean_h > 1e-9 else 0.0

        # ----------------------------------------------------------------
        # GROUP 3 — Building count and footprint statistics (NEW)
        # ----------------------------------------------------------------
        n_buildings, mean_fp, std_fp, max_fp, cv_fp = _building_component_stats(
            mask, dx, dy
        )

        # ----------------------------------------------------------------
        # GROUP 4 — Layout extent and domain summary (NEW)
        # ----------------------------------------------------------------
        domain_ar = float(lx) / float(ly) if ly > 1e-9 else 1.0
        built_area_total = float(building_cells) * dx * dy
        mean_h_lambda_p  = mean_h * lambda_p

        # ----------------------------------------------------------------
        # GROUP 5 — Environmental boundary inputs (NEW)
        # ----------------------------------------------------------------
        ref_height  = float(sample.environment.reference_height_m)
        z0_env      = float(sample.environment.roughness_length_z0_m)

        # ----------------------------------------------------------------
        # Target scalar (metadata only — NOT used in to_feature_vector)
        # ----------------------------------------------------------------
        if sample.targets.area_averaged_wind_speed is not None:
            target_scalar = float(sample.targets.area_averaged_wind_speed)
        else:
            from src.data.area_averaging import compute_area_averaged_wind_speed
            target_scalar = compute_area_averaged_wind_speed(
                field=sample.targets.pedestrian_wind_speed,
                mask=mask,
                resolution_m=(dy, dx),
            )

        return GeometricFeatures(
            layout_id=sample.metadata.layout_id,
            # Group 1
            plan_area_density_lambda_p=round(lambda_p, 5),
            frontal_area_density_lambda_f=round(lambda_f, 5),
            mean_building_height_m=round(mean_h, 4),
            std_building_height_m=round(std_h, 4),
            max_building_height_m=round(max_h, 4),
            wall_area_density_lambda_w=round(lambda_w, 5),
            roughness_length_macdonald_m=round(z0_val, 5),
            displacement_height_d_m=round(d_val, 4),
            inflow_speed_mps=round(sample.environment.inflow_speed_mps, 4),
            wind_direction_cos=round(cos_dir, 4),
            wind_direction_sin=round(sin_dir, 4),
            # Group 2
            min_building_height_m=round(min_h, 4),
            p25_building_height_m=round(p25_h, 4),
            p75_building_height_m=round(p75_h, 4),
            height_iqr_m=round(iqr_h, 4),
            height_cv=round(cv_h, 5),
            # Group 3
            building_count=float(n_buildings),
            mean_footprint_area_m2=round(mean_fp, 4),
            std_footprint_area_m2=round(std_fp, 4),
            max_footprint_area_m2=round(max_fp, 4),
            footprint_area_cv=round(cv_fp, 5),
            # Group 4
            domain_aspect_ratio=round(domain_ar, 5),
            built_area_total_m2=round(built_area_total, 4),
            mean_height_times_lambda_p=round(mean_h_lambda_p, 5),
            # Group 5
            reference_height_m=round(ref_height, 4),
            roughness_z0_env_m=round(z0_env, 5),
            # Target metadata
            target_area_averaged_wind_speed=round(target_scalar, 4),
        )
