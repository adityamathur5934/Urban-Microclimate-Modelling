"""Geometric summary feature extraction for traditional ML baselines."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import numpy as np

from src.data.contract import UrbanLayoutSample, GeometryData, EnvironmentalConditions


@dataclass(frozen=True)
class GeometricFeatures:
    """Summary morphometric and environmental features for a layout."""
    layout_id: str
    plan_area_density_lambda_p: float
    frontal_area_density_lambda_f: float
    mean_building_height_m: float
    std_building_height_m: float
    max_building_height_m: float
    wall_area_density_lambda_w: float
    roughness_length_macdonald_m: float
    displacement_height_d_m: float
    inflow_speed_mps: float
    wind_direction_cos: float
    wind_direction_sin: float
    # Target scalar for baseline ML training
    target_area_averaged_wind_speed: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_feature_vector(self) -> np.ndarray:
        """Returns feature vector [X] excluding layout_id and target."""
        return np.array([
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
        ], dtype=np.float32)

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
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
        ]


class GeometricFeatureExtractor:
    """Extracts scalar morphometric features from layout geometry and inflow conditions."""

    def __init__(self, cd_drag: float = 1.2, von_karman_kappa: float = 0.40):
        self.cd_drag = cd_drag
        self.kappa = von_karman_kappa

    def extract(self, sample: UrbanLayoutSample) -> GeometricFeatures:
        """Computes all summary features for an UrbanLayoutSample."""
        height_map = sample.geometry.height_map
        mask = sample.geometry.building_mask
        dx, dy, _ = sample.metadata.resolution_m
        total_cells = height_map.size
        total_area = total_cells * dx * dy

        # 1. Plan area density lambda_p
        building_cells = np.count_nonzero(mask)
        lambda_p = float(building_cells / max(total_cells, 1))

        # Height statistics over built areas
        built_heights = height_map[mask]
        if len(built_heights) > 0:
            mean_h = float(np.mean(built_heights))
            std_h = float(np.std(built_heights))
            max_h = float(np.max(built_heights))
        else:
            mean_h = 0.0
            std_h = 0.0
            max_h = 0.0

        # 2. Frontal area density lambda_f
        # Calculate frontal area projected along wind direction
        wind_deg = sample.environment.wind_direction_deg
        rad = np.deg2rad(wind_deg)
        cos_dir = float(np.cos(rad))
        sin_dir = float(np.sin(rad))

        # Step changes along x (west-east) and y (south-north)
        diff_x = np.diff(height_map, axis=1)
        frontal_x = np.sum(np.maximum(0.0, diff_x)) * dy
        # Also include left boundary if building exists at edge
        frontal_x += np.sum(height_map[:, 0]) * dy

        diff_y = np.diff(height_map, axis=0)
        frontal_y = np.sum(np.maximum(0.0, diff_y)) * dx
        frontal_y += np.sum(height_map[0, :]) * dx

        # Effective projected frontal area for current wind direction
        eff_frontal_area = np.abs(cos_dir) * frontal_x + np.abs(sin_dir) * frontal_y
        lambda_f = float(eff_frontal_area / max(total_area, 1e-6))

        # 3. Wall area density lambda_w (vertical facade area / total lot area)
        wall_steps_x = np.sum(np.abs(diff_x)) * dy + np.sum(height_map[:, 0] + height_map[:, -1]) * dy
        wall_steps_y = np.sum(np.abs(diff_y)) * dx + np.sum(height_map[0, :] + height_map[-1, :]) * dx
        total_facade_area = wall_steps_x + wall_steps_y
        lambda_w = float(total_facade_area / max(total_area, 1e-6))

        # 4. Displacement height d (MacDonald et al., 1998)
        # d / H = 1 + alpha^(-lambda_p) * (lambda_p - 1), alpha = 4.0
        if mean_h > 0 and lambda_p > 0:
            d_over_h = 1.0 + (4.0 ** (-lambda_p)) * (lambda_p - 1.0)
            d_val = float(mean_h * max(0.0, min(d_over_h, 0.95)))
        else:
            d_val = 0.0

        # 5. Aerodynamic roughness length z0 (MacDonald et al., 1998 parameterization)
        # z0 / H = (1 - d/H) * exp( - (0.5 * beta * (Cd/kappa^2) * (1 - d/H) * lambda_f )^(-0.5) )
        # where beta approx 1.0
        if mean_h > 0 and lambda_f > 1e-4 and (mean_h - d_val) > 1e-4:
            term = 0.5 * 1.0 * (self.cd_drag / (self.kappa ** 2)) * (1.0 - d_val / mean_h) * lambda_f
            if term > 1e-4:
                z0_val = float((mean_h - d_val) * np.exp(-1.0 / np.sqrt(term)))
            else:
                z0_val = float(sample.environment.roughness_length_z0_m)
        else:
            z0_val = float(sample.environment.roughness_length_z0_m)

        # 6. Target area-averaged wind speed
        if sample.targets.area_averaged_wind_speed is not None:
            target_scalar = float(sample.targets.area_averaged_wind_speed)
        else:
            # Compute over pedestrian outdoor area (where mask is False)
            outdoor_mask = ~mask
            if np.any(outdoor_mask):
                target_scalar = float(np.mean(sample.targets.pedestrian_wind_speed[outdoor_mask]))
            else:
                target_scalar = float(np.mean(sample.targets.pedestrian_wind_speed))

        return GeometricFeatures(
            layout_id=sample.metadata.layout_id,
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
            target_area_averaged_wind_speed=round(target_scalar, 4),
        )
