"""Unit tests for model interfaces and target configuration contracts."""

import pytest
from src.models.interfaces import (
    DEFAULT_SCALAR_WIND_TARGET,
    DEFAULT_WIND_FIELD_TARGET,
    OPTIONAL_TEMPERATURE_FIELD_TARGET,
    OPTIONAL_TKE_FIELD_TARGET,
    TargetConfig,
    TargetType,
)


def test_default_target_configs_valid():
    """Verify standard predefined target configurations pass validation."""
    DEFAULT_SCALAR_WIND_TARGET.validate()
    assert DEFAULT_SCALAR_WIND_TARGET.target_name == "area_averaged_wind_speed"
    assert DEFAULT_SCALAR_WIND_TARGET.target_type == TargetType.SCALAR
    assert DEFAULT_SCALAR_WIND_TARGET.unit == "m/s"

    DEFAULT_WIND_FIELD_TARGET.validate()
    assert DEFAULT_WIND_FIELD_TARGET.target_name == "pedestrian_wind_speed"
    assert DEFAULT_WIND_FIELD_TARGET.target_type == TargetType.FIELD_2D
    assert DEFAULT_WIND_FIELD_TARGET.unit == "m/s"
    assert DEFAULT_WIND_FIELD_TARGET.pedestrian_height_m == 1.75

    OPTIONAL_TEMPERATURE_FIELD_TARGET.validate()
    assert OPTIONAL_TEMPERATURE_FIELD_TARGET.unit == "K"

    OPTIONAL_TKE_FIELD_TARGET.validate()
    assert OPTIONAL_TKE_FIELD_TARGET.unit == "m^2/s^2"


def test_target_config_invalid_rejection():
    """Verify invalid configurations are rejected with descriptive errors."""
    with pytest.raises(ValueError, match="target_name cannot be empty"):
        TargetConfig(target_name="", unit="m/s").validate()

    with pytest.raises(ValueError, match="unit cannot be empty"):
        TargetConfig(target_name="temperature", unit="").validate()

    with pytest.raises(ValueError, match="pedestrian_height_m must be positive"):
        TargetConfig(target_name="wind", unit="m/s", pedestrian_height_m=-1.0).validate()
