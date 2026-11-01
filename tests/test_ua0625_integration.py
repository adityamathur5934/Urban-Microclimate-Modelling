"""Focused tests for the real UrbanTALES UA0625_d00 dataset integration.

These tests exercise the loaders in ``src.data.ua0625`` against the real
shipped files (``UA0625_d00_topo.csv``, ``profile-UA0625_d00.csv``,
``UA0625_d00.png``) and assert the scientific-constraint guarantees:

* no 2D wind field is fabricated from the vertical profile;
* grid spacing is not assumed (resolution_m / domain_size_m are None);
* the filename lambda_p label is cross-checked against the raster;
* provenance is honest (LES data loaded, not locally simulated, no model trained).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import ua0625 as ua
from src.data.ua0625 import (
    UA0625Dataset,
    UA0625Topology,
    UA0625VerticalProfile,
    load_ua0625_dataset,
    load_ua0625_image,
    load_ua0625_profile,
    load_ua0625_topology,
    parse_ua0625_dataset_id,
)
from src.data.contract import (
    PhysicalConstraintValidationError,
    ShapeMismatchValidationError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPO_CSV = PROJECT_ROOT / "UA0625_d00_topo.csv"
PROFILE_CSV = PROJECT_ROOT / "profile-UA0625_d00.csv"
IMAGE_PNG = PROJECT_ROOT / "UA0625_d00.png"

requires_files = pytest.mark.skipif(
    not (TOPO_CSV.exists() and PROFILE_CSV.exists()),
    reason="UA0625_d00 source files not present in repo root",
)


# ---------------------------------------------------------------------------
# Dataset-id parsing
# ---------------------------------------------------------------------------

class TestDatasetIdParsing:
    def test_parse_known_id(self):
        out = parse_ua0625_dataset_id("UA0625_d00")
        assert out["parsed"] is True
        assert out["plan_area_density_label"] == pytest.approx(0.0625)
        assert out["wind_direction_label_deg"] == 0.0

    def test_parse_3digit_density(self):
        out = parse_ua0625_dataset_id("UA125_d45")
        assert out["parsed"] is True
        assert out["plan_area_density_label"] == pytest.approx(0.125)
        assert out["wind_direction_label_deg"] == 45.0

    def test_parse_invalid_id(self):
        out = parse_ua0625_dataset_id("not-a-ua-id")
        assert out["parsed"] is False
        assert out["plan_area_density_label"] is None
        assert out["wind_direction_label_deg"] is None


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

@requires_files
class TestTopology:
    def test_topology_shape_and_values(self):
        topo = load_ua0625_topology(TOPO_CSV)
        assert topo.grid_shape == (384, 640)
        assert topo.height_map.shape == (384, 640)
        assert topo.building_mask.shape == (384, 640)
        assert topo.height_map.dtype == np.float32
        # The real grid is binary: ground 0, building 16 (metres, interpreted).
        assert set(np.unique(topo.height_map).tolist()) == {0.0, 16.0}
        assert topo.building_height_values.tolist() == [16.0]

    def test_plan_area_density_matches_filename_label(self):
        topo = load_ua0625_topology(TOPO_CSV)
        # lambda_p computed from the raster must equal the filename label.
        assert topo.plan_area_density == pytest.approx(0.0625)

    def test_mask_consistency(self):
        topo = load_ua0625_topology(TOPO_CSV)
        # mask True iff height > 0 (two-way consistency).
        assert np.array_equal(topo.building_mask, topo.height_map > 1e-4)
        assert int(np.count_nonzero(topo.building_mask)) == 15360

    def test_geometry_summary_is_spacing_free(self):
        topo = load_ua0625_topology(TOPO_CSV)
        s = topo.geometry_summary()
        assert s["grid_shape"] == [384, 640]
        assert s["plan_area_density_lambda_p"] == pytest.approx(0.0625)
        assert s["building_height_values_m"] == [16.0]
        assert s["mean_building_height_m"] == 16.0
        assert s["std_building_height_m"] == 0.0
        # No spacing-dependent quantities are reported.
        assert s["spacing_available"] is False
        assert s["resolution_m"] is None
        assert s["domain_size_m"] is None
        # Building count is a spacing-independent connected-component count.
        assert s["building_count"] > 0

    def test_spacing_not_assumed(self):
        topo = load_ua0625_topology(TOPO_CSV)
        assert topo.resolution_m is None
        assert topo.domain_size_m is None

    def test_filename_label_mismatch_raises(self):
        # The real raster has lambda_p = 0.0625; claiming 0.125 must fail.
        with pytest.raises(PhysicalConstraintValidationError):
            load_ua0625_topology(TOPO_CSV, dataset_id="UA1250_d00")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ua0625_topology(tmp_path / "nope.csv")

    def test_negative_height_rejected(self):
        h = np.zeros((4, 4), dtype=np.float32)
        h[1, 1] = -2.0
        bad = UA0625Topology(
            dataset_id="UA0625_d00",
            height_map=h,
            building_mask=h > 1e-4,
            grid_shape=(4, 4),
            source_file="mem",
        )
        with pytest.raises(PhysicalConstraintValidationError):
            bad.validate()

    def test_mask_height_inconsistency_rejected(self):
        h = np.zeros((4, 4), dtype=np.float32)
        h[1, 1] = 16.0
        mask = np.zeros((4, 4), dtype=bool)  # height>0 but mask False -> inconsistent
        bad = UA0625Topology(
            dataset_id="UA0625_d00",
            height_map=h,
            building_mask=mask,
            grid_shape=(4, 4),
            source_file="mem",
        )
        with pytest.raises(PhysicalConstraintValidationError):
            bad.validate()

    def test_shape_mismatch_rejected(self):
        h = np.zeros((4, 4), dtype=np.float32)
        bad = UA0625Topology(
            dataset_id="UA0625_d00",
            height_map=h,
            building_mask=np.zeros((3, 4), dtype=bool),
            grid_shape=(4, 4),
            source_file="mem",
        )
        with pytest.raises(ShapeMismatchValidationError):
            bad.validate()


# ---------------------------------------------------------------------------
# Vertical profile
# ---------------------------------------------------------------------------

@requires_files
class TestProfile:
    def test_profile_structure(self):
        p = load_ua0625_profile(PROFILE_CSV)
        assert p.num_levels == 130
        assert p.z.shape == (130,)
        assert p.z_min == pytest.approx(0.0, abs=1e-9)
        assert p.z_max == pytest.approx(131.497, abs=0.01)
        # z must be strictly increasing.
        assert np.all(np.diff(p.z) > 0.0)

    def test_ground_level_is_nan_noslip(self):
        p = load_ua0625_profile(PROFILE_CSV)
        # z=0 is the no-slip ground; mean velocity/TKE undefined there.
        assert np.isnan(p.u[0])
        assert np.isnan(p.v[0])
        assert np.isnan(p.w[0])
        assert np.isnan(p.speed[0])
        assert np.isnan(p.tke[0])
        # All other levels finite for the velocity columns.
        assert np.all(np.isfinite(p.u[1:]))
        assert np.all(np.isfinite(p.v[1:]))
        assert np.all(np.isfinite(p.speed[1:]))

    def test_speed_not_recomputed_from_components(self):
        # The dataset-reported U must NOT equal sqrt(u^2+v^2+w^2); the loader
        # preserves the reported value rather than fabricating a relationship.
        p = load_ua0625_profile(PROFILE_CSV)
        finite = np.isfinite(p.u) & np.isfinite(p.v) & np.isfinite(p.w) & np.isfinite(p.speed)
        mag = np.sqrt(p.u[finite] ** 2 + p.v[finite] ** 2 + p.w[finite] ** 2)
        # They differ for this dataset (max relative diff ~18%).
        assert not np.allclose(p.speed[finite], mag)
        assert np.max(np.abs(p.speed[finite] - mag) / np.abs(p.speed[finite])) > 0.1

    def test_profile_summary(self):
        p = load_ua0625_profile(PROFILE_CSV)
        s = p.summary()
        assert s["num_levels"] == 130
        assert s["u"]["finite_count"] == 129
        assert s["speed_U"]["finite_count"] == 129

    def test_non_monotonic_z_rejected(self):
        z = np.array([0.0, 1.0, 0.5, 2.0])
        bad = UA0625VerticalProfile(
            dataset_id="UA0625_d00",
            z=z, u=z * 0.0, v=z * 0.0, w=z * 0.0, speed=z * 0.0, tke=z * 0.0,
            num_levels=4, column_names=["z", "u"], source_file="mem",
        )
        with pytest.raises(PhysicalConstraintValidationError):
            bad.validate()

    def test_wrong_length_rejected(self):
        z = np.array([0.0, 1.0, 2.0])
        bad = UA0625VerticalProfile(
            dataset_id="UA0625_d00",
            z=z, u=np.array([0.0, 1.0]), v=z * 0.0, w=z * 0.0, speed=z * 0.0, tke=z * 0.0,
            num_levels=3, column_names=["z", "u"], source_file="mem",
        )
        with pytest.raises(ShapeMismatchValidationError):
            bad.validate()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ua0625_profile(tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not IMAGE_PNG.exists(), reason="UA0625_d00.png not present")
class TestImage:
    def test_image_loads(self):
        img = load_ua0625_image(IMAGE_PNG)
        assert img.size == (5087, 1972)  # (width, height)
        assert img.mode == "RGBA"
        assert img.array.shape == (1972, 5087, 4)
        assert img.array.dtype == np.uint8

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ua0625_image(tmp_path / "nope.png")


# ---------------------------------------------------------------------------
# Full dataset container + provenance / scientific guards
# ---------------------------------------------------------------------------

@requires_files
class TestDataset:
    def test_load_full_dataset(self):
        ds = load_ua0625_dataset()
        assert isinstance(ds, UA0625Dataset)
        assert ds.dataset_id == "UA0625_d00"
        assert ds.topology.grid_shape == (384, 640)
        assert ds.profile.num_levels == 130
        assert ds.image is not None

    def test_provenance_is_honest(self):
        ds = load_ua0625_dataset()
        pr = ds.provenance
        assert pr.locally_simulated is False
        assert pr.has_geometry is True
        assert pr.has_vertical_profile is True
        assert pr.license == "CC BY-NC 4.0"
        assert "UrbanTALES" in pr.source

    def test_no_2d_wind_field_fabricated(self):
        """Scientific guard: a 2D pedestrian wind field is never fabricated."""
        ds = load_ua0625_dataset()
        assert ds.has_2d_wind_field is False
        pr = ds.provenance
        assert pr.has_2d_wind_field is False
        # The topology must expose geometry only -- no wind-speed attribute.
        assert not hasattr(ds.topology, "pedestrian_wind_speed")
        # The profile is strictly 1-D.
        for name in ("z", "u", "v", "w", "speed", "tke"):
            assert getattr(ds.profile, name).ndim == 1

    def test_spacing_not_assumed_at_dataset_level(self):
        ds = load_ua0625_dataset()
        assert ds.topology.resolution_m is None
        assert ds.topology.domain_size_m is None
        assert "spacing" in " ".join(ds.limitations).lower()

    def test_limitations_documented(self):
        ds = load_ua0625_dataset()
        assert len(ds.limitations) >= 5
        joined = " ".join(ds.limitations).lower()
        assert "2d" in joined or "2 d" in joined
        assert "spacing" in joined or "resolution" in joined

    def test_validate_passes(self):
        ds = load_ua0625_dataset()
        ds.validate()  # should not raise

    def test_load_without_image(self):
        ds = load_ua0625_dataset(load_image=False)
        assert ds.image is None
        assert ds.topology.grid_shape == (384, 640)
        assert ds.profile.num_levels == 130


# ---------------------------------------------------------------------------
# Package export sanity
# ---------------------------------------------------------------------------

def test_package_exports():
    import src.data as data

    for name in (
        "UA0625Dataset",
        "UA0625Topology",
        "UA0625VerticalProfile",
        "load_ua0625_dataset",
        "default_ua0625_paths",
    ):
        assert hasattr(data, name), f"src.data should export {name}"


# ---------------------------------------------------------------------------
# API integration (additive, read-only /api/inventory/urbantales endpoint)
# ---------------------------------------------------------------------------

@requires_files
class TestUrbanTALESAPIEndpoint:
    def test_inventory_urbantales_endpoint(self):
        from fastapi.testclient import TestClient

        from src.api.server import app

        client = TestClient(app)
        resp = client.get("/api/inventory/urbantales")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["locally_simulated"] is False
        assert len(data["datasets"]) == 1
        s = data["datasets"][0]
        assert s["dataset_id"] == "UA0625_d00"
        assert s["has_geometry"] is True
        assert s["has_vertical_profile"] is True
        # Crucial scientific guards surfaced through the API.
        assert s["has_2d_wind_field"] is False
        assert s["resolution_m"] is None
        assert s["domain_size_m"] is None
        assert s["topology_grid_shape"] == [384, 640]
        assert s["plan_area_density"] == pytest.approx(0.0625)
        assert s["profile_num_levels"] == 130
        assert s["building_height_values_m"] == [16.0]
        assert len(s["limitations"]) >= 5
