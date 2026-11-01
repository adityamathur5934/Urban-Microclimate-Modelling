"""P0 Scientific Integrity and Provenance Regression Tests.

Verifies all acceptance criteria defined in Phase 1 through Phase 4:
1. Synthetic procedural and analytical benchmark fields are NOT labelled as OpenFOAM solver output
   (has_cfd_ground_truth is False across all layouts and field endpoints).
2. The neural prediction path invokes the actual model forward method in eval mode without gradients.
3. Checkpoint status is reported honestly (untrained initial weights are not masqueraded as trained).
4. Metrics (MAE, RMSE, NRMSE, max error, R2) are computed from actual model prediction vs reference arrays.
5. Unknown model IDs return HTTP 404 rather than silently falling back.
6. Unsupported target scalar fields return HTTP 400.
7. API response schema fields align with UI expectations (latency, speedup, layout features, provenance).
8. Unavailable metrics are not replaced with hardcoded or fabricated numbers (e.g. speedup is None).
9. The Random Forest baseline is explicitly identified as a scalar/tabular baseline.
"""

from __future__ import annotations

import unittest.mock
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.server import app
from src.evaluation.metrics import (
    compute_field_mae,
    compute_field_rmse,
    compute_normalized_rmse,
    compute_max_error,
    compute_r2_score,
)
from src.models.neural_unet import CompactUNet

client = TestClient(app)


def test_synthetic_fields_provenance_not_labelled_openfoam():
    """Phase 1: Verify synthetic fields are NEVER labelled has_cfd_ground_truth=True."""
    # Check layouts list
    resp = client.get("/api/layouts")
    assert resp.status_code == 200
    layouts = resp.json()
    assert len(layouts) >= 4
    for layout in layouts:
        assert layout["has_cfd_ground_truth"] is False, (
            f"Layout {layout['layout_id']} must have has_cfd_ground_truth=False"
        )
        assert "data_provenance" in layout
        assert "provenance_label" in layout
        assert "SYNTHETIC" in layout["data_provenance"] or "BENCHMARK" in layout["data_provenance"]

    # Check layout detail
    for layout in layouts:
        l_id = layout["layout_id"]
        detail_resp = client.get(f"/api/layouts/{l_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["has_cfd_ground_truth"] is False
        assert "data_provenance" in detail
        assert "SYNTHETIC" in detail["data_provenance"] or "BENCHMARK" in detail["data_provenance"]

        # Check fields endpoint
        fields_resp = client.get(f"/api/layouts/{l_id}/fields")
        assert fields_resp.status_code == 200
        fields = fields_resp.json()
        assert fields["has_cfd_ground_truth"] is False
        assert "data_provenance" in fields
        assert "SYNTHETIC" in fields["data_provenance"] or "BENCHMARK" in fields["data_provenance"]


def test_overview_provenance_and_solver_count():
    """Phase 1: Overview must report 0 ground truth solvers and honest provenance."""
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ground_truth_solvers_count"] == 0
    assert data["is_demonstration_fixture"] is True
    assert "data_provenance" in data
    provenance = data["data_provenance"]
    assert provenance["real_rans_count"] == 0
    assert provenance["is_demonstration_fixture"] is True
    assert "Castro & Robins" in provenance["solver_status"] or "Synthetic" in provenance["solver_status"]
    # Mesh description must reflect genuine 128x128 Cartesian grid (16,384 cells), not 65,536
    assert "16,384" in data["mesh_resolution_desc"]


def test_neural_prediction_invokes_forward_method():
    """Phase 2: Neural prediction path must invoke the actual model forward pass in eval mode."""
    # Use spy / mock on CompactUNet forward
    original_forward = CompactUNet.forward
    forward_called = []

    def wrapped_forward(self, x):
        forward_called.append(True)
        assert not x.requires_grad
        assert not self.training  # Model must be in eval mode
        return original_forward(self, x)

    with unittest.mock.patch.object(CompactUNet, "forward", wrapped_forward):
        req_payload = {
            "layout_id": "layout_042_staggered_canyon",
            "model_id": "unet-v2-resblock4",
            "scalar_field": "velocity_magnitude",
        }
        resp = client.post("/api/surrogates/predict", json=req_payload)
        assert resp.status_code == 200

    assert len(forward_called) >= 1, "CompactUNet.forward must have been executed during predict"


def test_untrained_checkpoint_reported_honestly():
    """Phase 2: If model weights are untrained/initial weights, is_trained_checkpoint must be False."""
    req_payload = {
        "layout_id": "layout_042_staggered_canyon",
        "model_id": "unet-v2-resblock4",
        "scalar_field": "velocity_magnitude",
    }
    resp = client.post("/api/surrogates/predict", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_trained_checkpoint"] is False
    assert data["prediction_type"] == "spatially_resolved_neural"


def test_metrics_computed_from_actual_prediction_and_target_arrays():
    """Phase 2: Field metrics must match standard evaluation metrics on fluid cells."""
    req_payload = {
        "layout_id": "layout_042_staggered_canyon",
        "model_id": "unet-v2-resblock4",
        "scalar_field": "velocity_magnitude",
    }
    resp = client.post("/api/surrogates/predict", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()

    fields_resp = client.get("/api/layouts/layout_042_staggered_canyon/fields")
    assert fields_resp.status_code == 200
    fields_data = fields_resp.json()

    pred_arr = np.array(data["predicted_field"], dtype=np.float32).reshape(128, 128)
    target_arr = np.array(fields_data["cfd_velocity_mag"], dtype=np.float32).reshape(128, 128)
    mask = np.array(fields_data["building_mask"], dtype=bool).reshape(128, 128)

    expected_mae = compute_field_mae(target_arr, pred_arr, building_mask=mask)
    expected_rmse = compute_field_rmse(target_arr, pred_arr, building_mask=mask)
    expected_nrmse = compute_normalized_rmse(target_arr, pred_arr, building_mask=mask, normalization="range", as_percent=True)
    expected_max = compute_max_error(target_arr, pred_arr, building_mask=mask)
    expected_r2 = compute_r2_score(target_arr, pred_arr, building_mask=mask)

    assert abs(data["field_mae"] - expected_mae) < 1e-3
    assert abs(data["field_rmse"] - expected_rmse) < 1e-3
    assert abs(data["field_nrmse"] - round(expected_nrmse, 2)) < 1e-4
    assert abs(data["peak_local_error"] - expected_max) < 1e-3
    assert abs(data["r2_score"] - expected_r2) < 1e-3


def test_random_forest_baseline_identified_as_tabular():
    """Phase 2: RF baseline must be clearly identified as scalar tabular baseline."""
    req_payload = {
        "layout_id": "layout_042_staggered_canyon",
        "model_id": "rf-baseline-morph",
        "scalar_field": "velocity_magnitude",
    }
    resp = client.post("/api/surrogates/predict", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction_type"] == "scalar_tabular_baseline"
    assert data["model_id"] == "rf-baseline-morph"


def test_unknown_model_id_returns_404():
    """Phase 3: Querying an unknown model ID must return 404, not fallback to U-Net."""
    req_payload = {
        "layout_id": "layout_042_staggered_canyon",
        "model_id": "nonexistent-model-xyz",
        "scalar_field": "velocity_magnitude",
    }
    resp = client.post("/api/surrogates/predict", json=req_payload)
    assert resp.status_code == 404
    assert "unknown" in resp.json()["detail"].lower()


def test_unsupported_scalar_field_returns_400():
    """Phase 3: Querying unsupported scalar field (e.g. tke, temperature) must return 400."""
    for field in ("tke", "turbulent_kinetic_energy", "temperature", "pressure"):
        req_payload = {
            "layout_id": "layout_042_staggered_canyon",
            "model_id": "unet-v2-resblock4",
            "scalar_field": field,
        }
        resp = client.post("/api/surrogates/predict", json=req_payload)
        assert resp.status_code == 400
        assert "not supported" in resp.json()["detail"]


def test_schema_field_alignment_and_types():
    """Phase 3: Response schemas must contain required fields with expected types."""
    # 1. LayoutSummary fields
    resp = client.get("/api/layouts")
    assert resp.status_code == 200
    for item in resp.json():
        assert "num_buildings" in item and isinstance(item["num_buildings"], int)
        assert "plan_area_fraction" in item and isinstance(item["plan_area_fraction"], float)
        assert "roughness_length_z0_m" in item and isinstance(item["roughness_length_z0_m"], float)
        assert "dataset_split" in item and isinstance(item["dataset_split"], str)
        assert "has_cfd_ground_truth" in item and isinstance(item["has_cfd_ground_truth"], bool)

    # 2. SurrogatePredictionResponse fields
    req_payload = {
        "layout_id": "layout_042_staggered_canyon",
        "model_id": "unet-v2-resblock4",
        "scalar_field": "velocity_magnitude",
    }
    resp_pred = client.post("/api/surrogates/predict", json=req_payload)
    assert resp_pred.status_code == 200
    pred = resp_pred.json()
    assert "inference_latency_ms" in pred and isinstance(pred["inference_latency_ms"], float)
    assert pred["inference_latency_ms"] >= 0.0
    assert "speedup_factor" in pred
    assert pred["speedup_factor"] is None  # Unavailable without true CFD baseline timing
    assert "r2_score" in pred and isinstance(pred["r2_score"], float)
    assert "is_trained_checkpoint" in pred and isinstance(pred["is_trained_checkpoint"], bool)
    assert "prediction_type" in pred and isinstance(pred["prediction_type"], str)


def test_unavailable_metrics_not_replaced_with_hardcoded_claims():
    """Phase 3: Fabricated metrics (0.142 m/s MAE, -18.4% improvement, 840x speedup) must not be present."""
    overview_resp = client.get("/api/overview")
    assert overview_resp.status_code == 200
    overview = overview_resp.json()

    # zero_leakage_mae_mps should be None, NOT 0.142
    assert overview["zero_leakage_mae_mps"] != 0.142
    # zero_leakage_change_pct should be None, NOT -18.4
    assert overview["zero_leakage_change_pct"] != -18.4

    # Surrogate prediction speedup_factor should be None, NOT 840 or 450
    pred_resp = client.post(
        "/api/surrogates/predict",
        json={
            "layout_id": "layout_042_staggered_canyon",
            "model_id": "unet-v2-resblock4",
            "scalar_field": "velocity_magnitude",
        },
    )
    assert pred_resp.status_code == 200
    pred = pred_resp.json()
    assert pred["speedup_factor"] is None
