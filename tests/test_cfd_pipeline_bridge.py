import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.data.contract import UrbanLayoutSample, LayoutFamily
from src.data.loader import LayoutDataLoader
from src.data.rasterizer import LayoutRasterizer
from src.data.features import GeometricFeatureExtractor
from src.data.normalization import FieldNormalizer
from cfd.scripts.foam_to_contract import create_cfd_layout_sample
from cfd.scripts.generate_canonical_reference import generate_canonical_block_reference


def test_create_cfd_layout_sample_valid():
    """Verify CFD bridge constructs a fully valid sample conforming to data contract."""
    ny, nx = 32, 64
    height_map = np.zeros((ny, nx), dtype=np.float32)
    building_mask = np.zeros((ny, nx), dtype=bool)

    # Place building block
    height_map[10:22, 20:32] = 20.0
    building_mask[10:22, 20:32] = True

    ped_wind = np.full((ny, nx), 3.5, dtype=np.float32)
    ped_wind[building_mask] = 0.0

    sample = create_cfd_layout_sample(
        layout_id="test_cfd_block_01",
        grid_shape=(ny, nx),
        domain_size_m=(64.0, 32.0, 60.0),
        resolution_m=(1.0, 1.0, 0.5),
        height_map=height_map,
        building_mask=building_mask,
        pedestrian_wind_speed=ped_wind,
        solver_name="OpenFOAM-simpleFoam",
        turbulence_model="kOmegaSST",
    )

    sample.validate()
    assert sample.layout_id == "test_cfd_block_01"
    assert sample.geometry.building_mask.shape == (ny, nx)
    assert sample.targets.pedestrian_wind_speed.shape == (ny, nx)
    assert sample.targets.pedestrian_wind_speed[15, 25] == 0.0


def test_canonical_reference_npz_roundtrip(tmp_path: Path):
    """Verify canonical reference sample saves and reloads with 100% fidelity."""
    sample = generate_canonical_block_reference("test_canonical_01")
    sample.validate()

    npz_file = tmp_path / "canonical_test.npz"
    LayoutDataLoader.save_sample_npz(sample, npz_file)
    assert npz_file.exists()

    loaded = LayoutDataLoader.load_sample_npz(npz_file)
    loaded.validate()

    assert loaded.layout_id == "test_canonical_01"
    assert loaded.geometry.height_map.shape == sample.geometry.height_map.shape
    np.testing.assert_allclose(
        loaded.targets.pedestrian_wind_speed,
        sample.targets.pedestrian_wind_speed,
        rtol=1e-5,
    )


def test_cfd_sample_rasterizes_for_neural_surrogate():
    """Verify CFD sample transforms cleanly into 7-channel input tensor."""
    sample = generate_canonical_block_reference("test_raster_cfd")
    rasterizer = LayoutRasterizer()
    raster_data = rasterizer.rasterize(sample)

    # Check 7 channels: height_map, building_mask, signed_distance, inflow_u, inflow_v, coord_x, coord_y
    assert raster_data.channels.shape == (7, 64, 128)
    assert raster_data.target_field.shape[1:] == (64, 128)
    # Check that building interior is masked to zero in targets
    mask = raster_data.building_mask
    assert np.all(raster_data.target_field[0][mask] == 0.0)


def test_cfd_sample_tabular_features_for_ml_baseline():
    """Verify geometric feature extractor processes CFD sample without error."""
    sample = generate_canonical_block_reference("test_feat_cfd")
    extractor = GeometricFeatureExtractor()
    features = extractor.extract(sample)

    assert features.plan_area_density_lambda_p > 0.0
    assert features.plan_area_density_lambda_p < 1.0
    assert features.mean_building_height_m == 20.0
    assert features.target_area_averaged_wind_speed > 0.0

    vec = features.to_feature_vector()
    assert len(vec) == 26
    assert not np.any(np.isnan(vec))


def test_cfd_field_normalizer_invertibility():
    """Verify spatial field normalizer normalizes and exactly inverts CFD field."""
    sample = generate_canonical_block_reference("test_norm_cfd")
    rasterizer = LayoutRasterizer()
    r = rasterizer.rasterize(sample)

    normalizer = FieldNormalizer(method="standard", mask_buildings=True)
    normalizer.fit([r.target_field], masks=[r.building_mask])

    norm_field = normalizer.transform(r.target_field, mask=r.building_mask)
    inverted = normalizer.inverse_transform(norm_field, mask=r.building_mask)

    outdoor = ~r.building_mask
    np.testing.assert_allclose(
        inverted[0][outdoor],
        r.target_field[0][outdoor],
        rtol=1e-5,
    )


def test_interpolate_samples_to_cartesian_grid():
    """Verify interpolation from scattered sample points to regular 2D grid."""
    from cfd.scripts.foam_to_contract import interpolate_samples_to_cartesian_grid

    sample_dict = {
        "x": np.array([0.0, 100.0, 0.0, 100.0, 50.0]),
        "y": np.array([0.0, 0.0, 50.0, 50.0, 25.0]),
        "z": np.array([1.75, 1.75, 1.75, 1.75, 1.75]),
        "Ux": np.array([5.0, 3.0, 5.0, 4.0, 2.5]),
        "Uy": np.array([0.0, 0.5, 0.0, -0.2, 0.1]),
    }

    h_map, b_mask, fields = interpolate_samples_to_cartesian_grid(
        sample_data=sample_dict,
        grid_shape=(16, 32),
        domain_size_m=(100.0, 50.0, 30.0),
        building_bounds=(40.0, 60.0, 20.0, 30.0, 15.0),
    )

    assert h_map.shape == (16, 32)
    assert b_mask.shape == (16, 32)
    assert "Ux" in fields
    assert fields["Ux"].shape == (16, 32)
    # Check that solid building cells have velocity clamped to 0.0
    assert np.all(fields["Ux"][b_mask] == 0.0)


def test_foam_to_contract_cli_demo(tmp_path: Path):
    """Verify CLI demo mode generates and exports a valid NPZ reference sample."""
    import subprocess
    out_npz = tmp_path / "demo_block.npz"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "cfd" / "scripts" / "foam_to_contract.py"),
        "--layout-id", "demo_cli_test",
        "--generate-demo",
        "--output-npz", str(out_npz),
        "--nx", "32",
        "--ny", "16",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert out_npz.exists()

    loaded = LayoutDataLoader.load_sample_npz(out_npz)
    loaded.validate()
    assert loaded.layout_id == "demo_cli_test"
    assert loaded.metadata.grid_shape == (16, 32)

