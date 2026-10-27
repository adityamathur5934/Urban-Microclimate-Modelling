"""Unit and integration tests for the Microclimate Research API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.server import app

client = TestClient(app)


def test_api_health():
    """Verify health endpoint returns status, compute device, and non-empty corpus."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert data["corpus_size"] > 0
    assert data["models_count"] > 0


def test_api_overview():
    """Verify high-level dashboard metrics match real corpus statistics."""
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["domain"] == "Computational Wind Engineering"
    assert data["geometric_corpus_count"] >= 20
    assert data["topological_families_count"] >= 3
    assert data["zero_leakage_mae_mps"] > 0.0
    assert "selected_layout_id" in data


def test_api_layouts_list_and_detail():
    """Verify layout enumeration and detail extraction."""
    resp = client.get("/api/layouts")
    assert resp.status_code == 200
    layouts = resp.json()
    assert len(layouts) >= 4

    layout_ids = [l["layout_id"] for l in layouts]
    assert "layout_042_staggered_canyon" in layout_ids

    # Detail query
    resp_detail = client.get("/api/layouts/layout_042_staggered_canyon")
    assert resp_detail.status_code == 200
    detail = resp_detail.json()
    assert detail["layout_id"] == "layout_042_staggered_canyon"
    assert "geometric_features" in detail
    assert "plan_area_density_lambda_p" in detail["geometric_features"]
    assert len(detail["building_heights"]) >= 2  # 12.0 and 45.0m buildings


def test_api_layout_fields():
    """Verify 2D spatial grid arrays and building masks are returned."""
    resp = client.get("/api/layouts/layout_042_staggered_canyon/fields")
    assert resp.status_code == 200
    data = resp.json()

    ny, nx = data["grid_shape"]
    assert ny == 128
    assert nx == 128
    assert len(data["height_map"]) == ny * nx
    assert len(data["building_mask"]) == ny * nx
    assert len(data["cfd_velocity_mag"]) == ny * nx
    assert data["stats"]["max"] > 0.0
    assert len(data["building_boxes"]) == 4  # B1, B2, B3, B4


def test_api_surrogate_predict():
    """Verify genuine surrogate inference produces predicted field, error map, and metrics."""
    req_payload = {
        "layout_id": "layout_042_staggered_canyon",
        "model_id": "unet-v2-resblock4",
        "scalar_field": "velocity_magnitude",
    }
    resp = client.post("/api/surrogates/predict", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["layout_id"] == "layout_042_staggered_canyon"
    assert data["field_mae"] > 0.0
    assert data["field_rmse"] >= data["field_mae"]
    assert data["field_nrmse"] > 0.0
    assert data["peak_local_error"] > 0.0
    assert data["r2_score"] > 0.8
    assert data["speedup_factor"] >= 100.0
    assert len(data["predicted_field"]) == 128 * 128
    assert len(data["abs_error_field"]) == 128 * 128


def test_api_probe_query():
    """Verify physical probe coordinates query."""
    resp = client.get("/api/layouts/layout_042_staggered_canyon/probe?x=52.4&y=48.1&model_id=unet-v2-resblock4")
    assert resp.status_code == 200
    data = resp.json()

    assert data["x_m"] == 52.4
    assert data["y_m"] == 48.1
    assert data["cfd_velocity_mps"] > 0.0
    assert data["surrogate_velocity_mps"] > 0.0
    assert data["abs_residual_mps"] >= 0.0


def test_api_transect_query():
    """Verify 1D slice query along canyon centerline (Y = 50.0m)."""
    resp = client.get("/api/layouts/layout_042_staggered_canyon/transect?axis=y&coord=50.0&samples=100")
    assert resp.status_code == 200
    data = resp.json()

    assert data["axis"] == "y"
    assert data["fixed_coord_m"] == 50.0
    assert len(data["coords_m"]) == 100
    assert len(data["cfd_values"]) == 100
    assert len(data["surrogate_values"]) == 100
    assert data["max_divergence_mps"] >= 0.0


def test_api_experiments_list_and_run():
    """Verify experiment benchmark records and background launching."""
    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    exps = resp.json()
    assert len(exps) >= 4

    # Run new experiment in background
    run_payload = {
        "architecture": "CompactUNet",
        "epochs": 10,
        "learning_rate": 0.001,
        "holdout_family": "idealized_staggered",
    }
    resp_run = client.post("/api/experiments/run", json=run_payload)
    assert resp_run.status_code == 200
    run_data = resp_run.json()
    assert run_data["status"] == "started"
    assert "experiment_id" in run_data


def test_api_export_triplet():
    """Verify publication triplet figure generation."""
    resp = client.get("/api/export/triplet/layout_042_staggered_canyon")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 1000


def test_ui_static_files_served():
    """Verify frontend static assets and index.html are served properly."""
    resp_index = client.get("/")
    assert resp_index.status_code == 200
    assert "Microclimate Lab" in resp_index.text

    resp_css = client.get("/static/styles.css")
    assert resp_css.status_code == 200
    assert "Microclimate Lab" in resp_css.text

    resp_js = client.get("/static/app.js")
    assert resp_js.status_code == 200
    assert "ScientificFieldRenderer" in resp_js.text
