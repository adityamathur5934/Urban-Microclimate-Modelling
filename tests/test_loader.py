"""Tests for sample serialization, loading, and corrupt file detection."""

import pytest
import numpy as np

from src.data.loader import LayoutDataLoader
from tests.fixtures.synthetic_data import create_synthetic_sample


def test_save_and_load_npz_roundtrip(tmp_path):
    """Verify sample saved to NPZ and loaded back has exact numerical match."""
    sample = create_synthetic_sample("save_load_01", building_height=24.5)
    file_path = tmp_path / "sample_01.npz"

    LayoutDataLoader.save_sample_npz(sample, file_path)
    assert file_path.exists()

    loaded = LayoutDataLoader.load_sample_npz(file_path)
    assert loaded.layout_id == sample.layout_id
    assert loaded.metadata.grid_shape == sample.metadata.grid_shape

    np.testing.assert_array_equal(loaded.geometry.height_map, sample.geometry.height_map)
    np.testing.assert_array_equal(loaded.geometry.building_mask, sample.geometry.building_mask)
    np.testing.assert_array_equal(loaded.targets.pedestrian_wind_speed, sample.targets.pedestrian_wind_speed)


def test_corrupt_file_raises_error(tmp_path):
    """Corrupted or missing metadata file should raise KeyError."""
    corrupt_file = tmp_path / "corrupt.npz"
    np.savez(corrupt_file, dummy=np.array([1, 2, 3]))

    with pytest.raises(KeyError, match="Corrupt or missing metadata"):
        LayoutDataLoader.load_sample_npz(corrupt_file)


def test_directory_batch_loading(tmp_path):
    """Verify loading directory with layout_id filtering."""
    s1 = create_synthetic_sample("layout_A")
    s2 = create_synthetic_sample("layout_B")
    s3 = create_synthetic_sample("layout_C")

    LayoutDataLoader.save_sample_npz(s1, tmp_path / "s1.npz")
    LayoutDataLoader.save_sample_npz(s2, tmp_path / "s2.npz")
    LayoutDataLoader.save_sample_npz(s3, tmp_path / "s3.npz")

    all_samples = LayoutDataLoader.load_directory(tmp_path)
    assert len(all_samples) == 3

    filtered = LayoutDataLoader.load_directory(tmp_path, layout_ids=["layout_A", "layout_C"])
    assert len(filtered) == 2
    assert {s.layout_id for s in filtered} == {"layout_A", "layout_C"}
