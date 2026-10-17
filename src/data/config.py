"""Preprocessing configuration persistence and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.data.normalization import FieldNormalizer, FeatureNormalizer
from src.data.split import DatasetSplit


@dataclass
class PreprocessingConfig:
    """Encapsulates all preprocessing settings, fitted normalization stats, and split assignments."""
    config_version: str = "1.0.0"
    target_variable: str = "pedestrian_wind_speed_mps"
    include_distance_transform: bool = True
    include_inflow_channels: bool = True
    include_coordinates: bool = True
    zero_inside_buildings: bool = True
    input_field_normalizer: Optional[Dict[str, Any]] = None
    target_field_normalizer: Optional[Dict[str, Any]] = None
    feature_normalizer: Optional[Dict[str, Any]] = None
    dataset_split: Optional[Dict[str, Any]] = None

    def save_json(self, file_path: str | Path) -> None:
        """Saves configuration to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str | Path) -> PreprocessingConfig:
        """Loads configuration from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def get_input_field_normalizer(self) -> Optional[FieldNormalizer]:
        if self.input_field_normalizer is None:
            return None
        return FieldNormalizer.from_dict(self.input_field_normalizer)

    def get_target_field_normalizer(self) -> Optional[FieldNormalizer]:
        if self.target_field_normalizer is None:
            return None
        return FieldNormalizer.from_dict(self.target_field_normalizer)

    def get_feature_normalizer(self) -> Optional[FeatureNormalizer]:
        if self.feature_normalizer is None:
            return None
        return FeatureNormalizer.from_dict(self.feature_normalizer)

    def get_dataset_split(self) -> Optional[DatasetSplit]:
        if self.dataset_split is None:
            return None
        return DatasetSplit.from_dict(self.dataset_split)
