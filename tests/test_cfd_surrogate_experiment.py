"""Focused tests for the CFD-to-surrogate experiment pipeline.

Covers:
  1. Analytical reference field loading and validation.
  2. Geometry/target grid alignment.
  3. Velocity normalisation (train-only, zero leakage).
  4. Layout-level split isolation.
  5. Model checkpoint save / load / inference.
  6. Inference output shapes.
  7. Metric calculations.
  8. Error-map generation.
  9. Uncertainty (ensemble predictive disagreement).
 10. Dataset wrapper masking correctness.
 11. CFD case generator writes valid files.
 12. Provenance and honest labelling.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from src.data.contract import LayoutFamily, UrbanLayoutSample
from src.data.rasterizer import LayoutRasterizer, verify_spatial_alignment
from src.data.normalization import FieldNormalizer
from src.data.split import LayoutSplitter
from src.evaluation.metrics import (
    compute_field_mae,
    compute_field_rmse,
    compute_max_error,
    compute_normalized_rmse,
    compute_r2_score,
    compute_spatial_error_map,
)
from src.models.neural_unet import CompactCNN, CompactUNet
from src.models.baseline_rf import RandomForestBaseline

from src.experiment.reference import (
    REFERENCE_PROVENANCE,
    analytical_pedestrian_wind_field,
    build_analytical_reference_sample,
    generate_reference_dataset,
)
from src.experiment.dataset import ExperimentDataset, fit_field_normalizers
from src.experiment.uncertainty import predictive_disagreement


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_samples() -> List[UrbanLayoutSample]:
    """8 analytical reference samples at 32×32 for fast tests."""
    return generate_reference_dataset(n_layouts=8, seed=0, grid_shape=(32, 32),
                                      domain_size_m=(64.0, 64.0, 40.0))


@pytest.fixture(scope="module")
def split_and_samples(tiny_samples):
    ids = [s.layout_id for s in tiny_samples]
    split = LayoutSplitter.split_by_ratio(ids, train_ratio=0.50, val_ratio=0.25,
                                           test_ratio=0.25, seed=42)
    by_id = {s.layout_id: s for s in tiny_samples}
    train = [by_id[i] for i in split.train_layout_ids]
    val   = [by_id[i] for i in split.val_layout_ids]
    test  = [by_id[i] for i in split.test_layout_ids]
    return split, train, val, test


@pytest.fixture(scope="module")
def normalizers_and_rasterizer(split_and_samples):
    _, train, _, _ = split_and_samples
    return fit_field_normalizers(train)


# ---------------------------------------------------------------------------
# 1. Analytical reference loading and validation
# ---------------------------------------------------------------------------

class TestAnalyticalReference:
    def test_single_sample_validates(self):
        s = build_analytical_reference_sample(
            "test_single", LayoutFamily.IDEALIZED_ALIGNED,
            grid_shape=(32, 32), domain_size_m=(64.0, 64.0, 40.0),
            buildings=[{"x": (20.0, 32.0), "y": (20.0, 32.0), "h": 12.0}],
            inflow_speed_mps=5.0, wind_direction_deg=0.0,
        )
        s.validate()  # must not raise

    def test_provenance_is_analytical_not_cfd(self):
        s = build_analytical_reference_sample(
            "prov_test", LayoutFamily.IDEALIZED_ALIGNED,
            grid_shape=(32, 32), domain_size_m=(64.0, 64.0, 40.0),
            buildings=[{"x": (10.0, 22.0), "y": (10.0, 22.0), "h": 20.0}],
            inflow_speed_mps=5.0, wind_direction_deg=0.0,
        )
        assert s.metadata.city_or_source == "ANALYTICAL_REFERENCE"
        assert "Analytical" in s.cfd_solver.solver_name
        assert s.cfd_solver.convergence_criteria == "analytical_exact"

    def test_generate_dataset_count(self, tiny_samples):
        assert len(tiny_samples) == 8

    def test_all_samples_validate(self, tiny_samples):
        for s in tiny_samples:
            s.validate()

    def test_field_finite_and_non_negative(self, tiny_samples):
        for s in tiny_samples:
            ped = s.targets.pedestrian_wind_speed
            assert np.all(np.isfinite(ped))
            assert np.all(ped >= 0.0)

    def test_no_pedestrian_field_in_buildings(self, tiny_samples):
        for s in tiny_samples:
            mask = s.geometry.building_mask
            if np.any(mask):
                assert np.allclose(s.targets.pedestrian_wind_speed[mask], 0.0)

    def test_wind_field_has_variation(self, tiny_samples):
        for s in tiny_samples:
            fluid = ~s.geometry.building_mask
            speeds = s.targets.pedestrian_wind_speed[fluid]
            assert float(np.std(speeds)) > 1e-4, "Pedestrian wind field is spatially uniform (no aerodynamic variation)"

    def test_dataset_families_all_labelled(self, tiny_samples):
        families = {s.metadata.layout_family for s in tiny_samples}
        assert len(families) >= 2

    def test_all_same_grid_shape(self, tiny_samples):
        shapes = {s.metadata.grid_shape for s in tiny_samples}
        assert len(shapes) == 1

    def test_analytical_field_function_returns_correct_shapes(self):
        x = np.linspace(0, 64, 32)
        y = np.linspace(0, 64, 32)
        X, Y = np.meshgrid(x, y)
        blds = [{"x": (20.0, 32.0), "y": (20.0, 32.0), "h": 15.0}]
        u_mag, u_x, v_y, tke = analytical_pedestrian_wind_field(X, Y, blds, 5.0, 0.0)
        for arr in (u_mag, u_x, v_y, tke):
            assert arr.shape == (32, 32)
            assert arr.dtype == np.float32
            assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# 2. Geometry / target alignment
# ---------------------------------------------------------------------------

class TestGeometryTargetAlignment:
    def test_rasterize_passes_verify(self, tiny_samples):
        rasterizer = LayoutRasterizer()
        for s in tiny_samples:
            r = rasterizer.rasterize(s)
            assert verify_spatial_alignment(r, s), f"Alignment failed for {s.layout_id}"

    def test_channel_shape(self, tiny_samples):
        rasterizer = LayoutRasterizer()
        for s in tiny_samples:
            r = rasterizer.rasterize(s)
            ny, nx = s.metadata.grid_shape
            assert r.channels.shape == (7, ny, nx)
            assert r.target_field.shape[1:] == (ny, nx)

    def test_building_mask_matches_height_map(self, tiny_samples):
        for s in tiny_samples:
            positive = s.geometry.height_map > 1e-4
            np.testing.assert_array_equal(positive, s.geometry.building_mask)

    def test_target_zero_inside_buildings(self, tiny_samples):
        rasterizer = LayoutRasterizer()
        for s in tiny_samples:
            r = rasterizer.rasterize(s)
            mask = r.building_mask
            if not np.any(mask):
                continue
            for c in range(r.target_field.shape[0]):
                assert np.allclose(r.target_field[c][mask], 0.0)


# ---------------------------------------------------------------------------
# 3. Velocity normalisation — train-only, zero leakage
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalizers_fitted(self, normalizers_and_rasterizer):
        inp_n, tgt_n, _ = normalizers_and_rasterizer
        assert inp_n.is_fitted
        assert tgt_n.is_fitted

    def test_normalizer_no_leakage(self, normalizers_and_rasterizer, split_and_samples):
        inp_n, tgt_n, _ = normalizers_and_rasterizer
        split, _, val, test = split_and_samples
        held_out = split.val_layout_ids + split.test_layout_ids
        inp_n.verify_no_leakage(held_out)   # must not raise
        tgt_n.verify_no_leakage(held_out)

    def test_transform_inverse_roundtrip(self, normalizers_and_rasterizer, split_and_samples):
        inp_n, tgt_n, rasterizer = normalizers_and_rasterizer
        _, train, _, _ = split_and_samples
        r = rasterizer.rasterize(train[0])
        t = r.target_field[0:1]
        tgt_norm = tgt_n.transform(t.copy(), mask=r.building_mask)
        t_back = tgt_n.inverse_transform(tgt_norm, mask=r.building_mask)
        np.testing.assert_allclose(t, t_back, atol=1e-4)

    def test_building_cells_zero_after_normalise(self, normalizers_and_rasterizer, split_and_samples):
        inp_n, tgt_n, rasterizer = normalizers_and_rasterizer
        _, train, _, _ = split_and_samples
        for s in train[:2]:
            r = rasterizer.rasterize(s)
            x_norm = inp_n.transform(r.channels.copy(), mask=r.building_mask)
            if np.any(r.building_mask):
                for c in range(x_norm.shape[0]):
                    assert np.allclose(x_norm[c][r.building_mask], 0.0)


# ---------------------------------------------------------------------------
# 4. Layout-level split isolation (zero leakage)
# ---------------------------------------------------------------------------

class TestSplitIsolation:
    def test_no_layout_id_overlap(self, split_and_samples):
        split, _, _, _ = split_and_samples
        split.validate()  # raises on overlap

    def test_correct_counts(self, split_and_samples, tiny_samples):
        split, train, val, test = split_and_samples
        assert len(train) + len(val) + len(test) == len(tiny_samples)

    def test_test_and_train_disjoint(self, split_and_samples):
        split, _, _, _ = split_and_samples
        train_set = set(split.train_layout_ids)
        test_set = set(split.test_layout_ids)
        assert train_set.isdisjoint(test_set)


# ---------------------------------------------------------------------------
# 5 & 6. Checkpoint save / load + inference output shapes
# ---------------------------------------------------------------------------

class TestCheckpointAndInference:
    def test_unet_save_load_predict(self, tiny_samples, tmp_path):
        in_ch = 7
        ny, nx = tiny_samples[0].metadata.grid_shape
        model = CompactUNet(in_channels=in_ch, out_channels=1, base_channels=8, depth=2)
        ckpt = tmp_path / "unet.pt"
        model.save_checkpoint(ckpt, epoch=1, val_metrics={"mae": 0.5})
        loaded, meta = CompactUNet.load_checkpoint(ckpt, device="cpu")
        assert meta["epoch"] == 1
        x = torch.zeros(1, in_ch, ny, nx)
        with torch.no_grad():
            out = loaded(x)
        assert out.shape == (1, 1, ny, nx)

    def test_cnn_save_load_predict(self, tiny_samples, tmp_path):
        in_ch = 7
        ny, nx = tiny_samples[0].metadata.grid_shape
        model = CompactCNN(in_channels=in_ch, out_channels=1, hidden_channels=8, num_layers=3)
        ckpt = tmp_path / "cnn.pt"
        model.save_checkpoint(ckpt, epoch=2, val_metrics={"mae": 0.4})
        loaded, meta = CompactCNN.load_checkpoint(ckpt, device="cpu")
        assert meta["epoch"] == 2
        x = torch.zeros(1, in_ch, ny, nx)
        with torch.no_grad():
            out = loaded(x)
        assert out.shape == (1, 1, ny, nx)

    def test_unet_zeros_inside_buildings(self, tiny_samples):
        s = tiny_samples[0]
        ny, nx = s.metadata.grid_shape
        model = CompactUNet(in_channels=7, out_channels=1, base_channels=8, depth=2)
        r = LayoutRasterizer().rasterize(s)
        x = torch.from_numpy(r.channels).unsqueeze(0).float()
        with torch.no_grad():
            out = model(x).squeeze(0).squeeze(0).numpy()
        if np.any(r.building_mask):
            assert np.allclose(out[r.building_mask], 0.0, atol=1e-6)

    def test_rf_save_load_predict(self, split_and_samples, tmp_path):
        _, train, _, test = split_and_samples
        rf = RandomForestBaseline(n_estimators=10, max_depth=5, random_state=42)
        rf.fit_from_samples(train)
        ckpt = tmp_path / "rf.joblib"
        rf.save(ckpt)
        rf2 = RandomForestBaseline.load(ckpt)
        assert rf2._is_fitted
        preds1 = rf.predict_from_samples(test)
        preds2 = rf2.predict_from_samples(test)
        np.testing.assert_allclose(preds1, preds2, rtol=1e-5)


# ---------------------------------------------------------------------------
# 7. Metric calculations
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.fixture
    def field_pair(self):
        rng = np.random.default_rng(0)
        ny, nx = 32, 32
        mask = np.zeros((ny, nx), dtype=bool)
        mask[10:20, 10:20] = True
        y_true = rng.uniform(1.0, 6.0, (ny, nx)).astype(np.float32)
        y_true[mask] = 0.0
        y_pred = y_true + rng.normal(0, 0.3, (ny, nx)).astype(np.float32)
        y_pred[mask] = 0.0
        y_pred = np.clip(y_pred, 0.0, None)
        return y_true, y_pred, mask

    def test_mae_positive(self, field_pair):
        y_true, y_pred, mask = field_pair
        mae = compute_field_mae(y_true, y_pred, building_mask=mask)
        assert mae > 0.0

    def test_rmse_ge_mae(self, field_pair):
        y_true, y_pred, mask = field_pair
        mae = compute_field_mae(y_true, y_pred, building_mask=mask)
        rmse = compute_field_rmse(y_true, y_pred, building_mask=mask)
        assert rmse >= mae - 1e-6

    def test_nrmse_range_between_0_and_100(self, field_pair):
        y_true, y_pred, mask = field_pair
        nrmse = compute_normalized_rmse(y_true, y_pred, building_mask=mask,
                                        normalization="range", as_percent=True)
        assert 0.0 < nrmse < 100.0

    def test_nrmse_inflow_uses_uref(self, field_pair):
        y_true, y_pred, mask = field_pair
        n1 = compute_normalized_rmse(y_true, y_pred, building_mask=mask,
                                     normalization="inflow", u_ref=5.0, as_percent=True)
        n2 = compute_normalized_rmse(y_true, y_pred, building_mask=mask,
                                     normalization="inflow", u_ref=10.0, as_percent=True)
        assert abs(n1 - 2.0 * n2) < 1e-4

    def test_r2_perfect(self, field_pair):
        y_true, _, mask = field_pair
        r2 = compute_r2_score(y_true, y_true, building_mask=mask)
        assert abs(r2 - 1.0) < 1e-5

    def test_max_error_non_negative(self, field_pair):
        y_true, y_pred, mask = field_pair
        assert compute_max_error(y_true, y_pred, building_mask=mask) >= 0.0

    def test_exact_zero_error(self, field_pair):
        y_true, _, mask = field_pair
        assert compute_field_mae(y_true, y_true, building_mask=mask) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. Error-map generation
# ---------------------------------------------------------------------------

class TestErrorMap:
    def test_error_map_shape_and_positivity(self):
        rng = np.random.default_rng(1)
        ny, nx = 32, 32
        mask = np.zeros((ny, nx), dtype=bool)
        mask[8:16, 8:16] = True
        y_true = rng.uniform(1.0, 5.0, (ny, nx)).astype(np.float32)
        y_true[mask] = 0.0
        y_pred = y_true * 0.9
        err = compute_spatial_error_map(y_true, y_pred, building_mask=mask)
        assert err.shape == (ny, nx)
        assert np.all(err >= 0.0)
        assert np.allclose(err[mask], 0.0)

    def test_error_map_zero_perfect_prediction(self):
        ny, nx = 16, 16
        mask = np.zeros((ny, nx), dtype=bool)
        y = np.ones((ny, nx), dtype=np.float32)
        err = compute_spatial_error_map(y, y, building_mask=mask)
        assert np.allclose(err, 0.0)


# ---------------------------------------------------------------------------
# 9. Uncertainty (ensemble predictive disagreement)
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_predictive_disagreement_zero_for_single_model(self):
        rng = np.random.default_rng(2)
        ny, nx = 16, 16
        mask = np.zeros((ny, nx), dtype=bool)
        field = rng.uniform(1.0, 5.0, (ny, nx)).astype(np.float32)
        preds = field[np.newaxis, ...]  # K=1
        result = predictive_disagreement(preds, mask)
        assert result["mean_pixel_std_mps"] == pytest.approx(0.0, abs=1e-6)

    def test_predictive_disagreement_positive_for_disagreeing_ensemble(self):
        rng = np.random.default_rng(3)
        ny, nx = 32, 32
        mask = np.zeros((ny, nx), dtype=bool)
        mask[8:16, 8:16] = True
        preds = rng.uniform(1.0, 5.0, (3, ny, nx)).astype(np.float32)
        preds[:, mask] = 0.0
        result = predictive_disagreement(preds, mask)
        assert result["mean_pixel_std_mps"] > 0.0
        assert result["max_pixel_std_mps"] >= result["p95_pixel_std_mps"]
        assert result["p95_pixel_std_mps"] >= result["mean_pixel_std_mps"]
        assert result["std_field"].shape == (ny, nx)
        assert np.allclose(result["std_field"][mask], 0.0)

    def test_predictive_disagreement_with_identical_models_zero_std(self):
        ny, nx = 16, 16
        mask = np.zeros((ny, nx), dtype=bool)
        f = np.full((ny, nx), 3.0, dtype=np.float32)
        preds = np.stack([f, f, f])  # identical
        result = predictive_disagreement(preds, mask)
        assert result["mean_pixel_std_mps"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. ExperimentDataset masking correctness
# ---------------------------------------------------------------------------

class TestExperimentDataset:
    def test_fluid_mask_is_correct(self, split_and_samples, normalizers_and_rasterizer):
        inp_n, tgt_n, rasterizer = normalizers_and_rasterizer
        _, _, _, test = split_and_samples
        ds = ExperimentDataset(test, rasterizer, inp_n, tgt_n)
        for i in range(len(ds)):
            x, y, fluid_mask_raw, lid = ds[i]
            building_mask = ds.building_mask(i)
            fluid_expected = (~building_mask).astype(np.float32)
            # fluid_mask_raw is (1, Ny, Nx) numpy float32
            if isinstance(fluid_mask_raw, torch.Tensor):
                fluid_got = fluid_mask_raw.numpy().squeeze()
            else:
                fluid_got = np.asarray(fluid_mask_raw).squeeze()
            np.testing.assert_array_equal(fluid_got, fluid_expected)

    def test_normalised_building_cells_zero(self, split_and_samples, normalizers_and_rasterizer):
        inp_n, tgt_n, rasterizer = normalizers_and_rasterizer
        _, _, _, test = split_and_samples
        ds = ExperimentDataset(test, rasterizer, inp_n, tgt_n)
        for i in range(len(ds)):
            x, y, _, _ = ds[i]
            building_mask = ds.building_mask(i)
            if not np.any(building_mask):
                continue
            for c in range(x.shape[0]):
                assert torch.allclose(x[c][torch.from_numpy(building_mask)], torch.zeros(1), atol=1e-6)
            assert torch.allclose(y[0][torch.from_numpy(building_mask)], torch.zeros(1), atol=1e-6)

    def test_raw_target_physical_units(self, split_and_samples, normalizers_and_rasterizer):
        inp_n, tgt_n, rasterizer = normalizers_and_rasterizer
        _, _, _, test = split_and_samples
        if not test:
            pytest.skip("No test samples")
        ds = ExperimentDataset(test, rasterizer, inp_n, tgt_n)
        raw = ds.raw_target(0)
        s = test[0]
        np.testing.assert_allclose(raw, s.targets.pedestrian_wind_speed, atol=1e-4)


# ---------------------------------------------------------------------------
# 11. CFD case generator writes valid files
# ---------------------------------------------------------------------------

class TestCFDCaseGenerator:
    def test_generate_cases_writes_files(self, tmp_path):
        from cfd.scripts.generate_cfd_cases import CaseSpec, write_case
        spec = CaseSpec("test_case_01", 20.0, 20.0, 5.0, 10.0, 0.05)
        case_dir = write_case(spec, tmp_path)
        assert case_dir.exists()
        required = [
            "system/blockMeshDict", "system/controlDict", "system/fvSchemes",
            "system/fvSolution", "0/U", "0/k", "0/omega", "0/p", "Allrun",
        ]
        for f in required:
            assert (case_dir / f).exists(), f"Missing expected file: {f}"

    def test_block_mesh_contains_building_height(self, tmp_path):
        from cfd.scripts.generate_cfd_cases import CaseSpec, block_mesh_dict
        spec = CaseSpec("test_h28", 28.0, 20.0, 5.0, 10.0, 0.05)
        bmd = block_mesh_dict(spec)
        assert "28.0" in bmd

    def test_abl_inlet_k_positive(self, tmp_path):
        from cfd.scripts.generate_cfd_cases import CaseSpec, _abl_k_omega
        spec = CaseSpec("test", 20.0, 20.0, 5.0, 10.0, 0.05)
        k, omega = _abl_k_omega(spec.inflow_speed_mps, spec.reference_height_m, spec.roughness_z0_m)
        assert k > 0.0
        assert omega > 0.0

    def test_allrun_checks_solver(self, tmp_path):
        from cfd.scripts.generate_cfd_cases import ALLRUN
        assert "simpleFoam" in ALLRUN
        assert "blockMesh" in ALLRUN
        assert "ERROR" in ALLRUN  # must emit an error if solver absent


# ---------------------------------------------------------------------------
# 12. Provenance and honest labelling
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_reference_provenance_constant(self):
        assert REFERENCE_PROVENANCE == "ANALYTICAL_REFERENCE"

    def test_samples_not_labelled_cfd(self, tiny_samples):
        for s in tiny_samples:
            assert s.metadata.city_or_source == "ANALYTICAL_REFERENCE"
            assert "CFD" not in s.cfd_solver.solver_name.upper() or "Analytical" in s.cfd_solver.solver_name
            # Crucially: NOT "OpenFOAM", "uDALES", "PALM"
            for real_solver in ("OpenFOAM", "uDALES", "PALM", "RANS-OpenFOAM"):
                assert real_solver not in s.cfd_solver.solver_name

    def test_runner_manifest_has_provenance(self, tmp_path):
        import json
        from src.experiment.runner import ExperimentConfig, run_experiment
        cfg = ExperimentConfig(
            n_layouts=6, grid_shape=(32, 32), epochs=1, batch_size=2,
            ensemble_size=2, patience=2, device="cpu",
            checkpoint_dir=str(tmp_path / "ckpt"),
            output_dir=str(tmp_path / "out"),
            manifest_path=str(tmp_path / "manifest.json"),
            generate_figures=False,
        )
        result = run_experiment(cfg)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest.get("provenance") == "ANALYTICAL_REFERENCE"

    def test_experiment_has_2d_wind_field_false_for_ua0625(self):
        from src.data.ua0625 import load_ua0625_dataset
        ds = load_ua0625_dataset()
        assert ds.has_2d_wind_field is False
