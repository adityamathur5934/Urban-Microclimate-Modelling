"""Tests for spatial rasterization and channel alignment."""

import pytest
import numpy as np

from src.data.rasterizer import LayoutRasterizer
from src.data.contract import LayoutFamily
from tests.fixtures.synthetic_data import create_synthetic_sample


def test_rasterizer_channels_and_shapes():
    """Verify channel count, shape alignment, and channel names."""
    sample = create_synthetic_sample("rast_test_01", grid_shape=(32, 32))
    rasterizer = LayoutRasterizer()
    raster_data = rasterizer.rasterize(sample)

    assert raster_data.channels.shape == (7, 32, 32)
    assert raster_data.target_field.shape[1:] == (32, 32)
    assert "signed_distance_m" in raster_data.channel_names
    assert "pedestrian_wind_speed_mps" in raster_data.target_names


def test_signed_distance_field_properties():
    """Verify signed distance field has positive values outside and negative inside."""
    sample = create_synthetic_sample("sdf_test", grid_shape=(32, 32))
    rasterizer = LayoutRasterizer()
    raster_data = rasterizer.rasterize(sample)

    sdf_idx = raster_data.channel_names.index("signed_distance_m")
    sdf = raster_data.channels[sdf_idx]
    mask = raster_data.building_mask

    # Outside buildings: distance > 0 (except cells adjacent to boundaries)
    assert np.all(sdf[~mask] >= 0.0)
    # Deep inside buildings: distance < 0
    assert np.all(sdf[mask] <= 0.0)


def test_building_interior_zeroed_in_targets():
    """Verify target wind speeds are zeroed inside buildings when requested."""
    sample = create_synthetic_sample("zero_test", grid_shape=(32, 32))
    rasterizer = LayoutRasterizer(zero_inside_buildings=True)
    raster_data = rasterizer.rasterize(sample)

    target_wind = raster_data.target_field[0]
    mask = raster_data.building_mask

    assert np.all(target_wind[mask] == 0.0)
