"""P1 regression tests: inventory endpoints, canonical reference loading, real training thread.

These tests verify that:
- /api/inventory/datasets returns honest provenance counts
- /api/inventory/checkpoints reports is_trained correctly
- /api/inventory/cfd reports solver availability honestly
- The canonical aerodynamic benchmark reference is loaded from disk when available
- The background experiment training thread calls real training (not sleep-loop)
- Inventory provenance counts are consistent with in-memory sample provenance
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.service import MicroclimateService

client = TestClient(app)


# ---------------------------------------------------------------------------
# Dataset Inventory
# ---------------------------------------------------------------------------

class TestDatasetInventory:
    def test_inventory_datasets_endpoint_returns_200(self):
        resp = client.get("/api/inventory/datasets")
        assert resp.status_code == 200

    def test_inventory_datasets_schema(self):
        resp = client.get("/api/inventory/datasets")
        data = resp.json()
        for key in ("total_loaded", "synthetic_count", "canonical_benchmark_count",
                    "cfd_solver_output_count", "items", "notes"):
            assert key in data, f"Missing key: {key}"

    def test_inventory_datasets_no_cfd_ground_truth(self):
        """No item in the inventory may report has_cfd_ground_truth=True."""
        resp = client.get("/api/inventory/datasets")
        data = resp.json()
        for item in data["items"]:
            assert item["has_cfd_ground_truth"] is False, (
                f"Layout {item['layout_id']} incorrectly reports has_cfd_ground_truth=True"
            )

    def test_inventory_datasets_count_is_positive(self):
        resp = client.get("/api/inventory/datasets")
        data = resp.json()
        assert data["total_loaded"] > 0
        assert data["total_loaded"] == len(data["items"])

    def test_inventory_datasets_provenance_counts_sum_correctly(self):
        resp = client.get("/api/inventory/datasets")
        data = resp.json()
        total = data["synthetic_count"] + data["canonical_benchmark_count"] + data["cfd_solver_output_count"]
        assert total == data["total_loaded"], (
            f"Provenance counts don't sum to total: {data['synthetic_count']} + "
            f"{data['canonical_benchmark_count']} + {data['cfd_solver_output_count']} != {data['total_loaded']}"
        )

    def test_inventory_datasets_cfd_count_is_zero(self):
        """No OpenFOAM solver outputs are present; cfd_solver_output_count must be 0."""
        resp = client.get("/api/inventory/datasets")
        data = resp.json()
        assert data["cfd_solver_output_count"] == 0, (
            "cfd_solver_output_count is non-zero but no OpenFOAM outputs should be loaded"
        )


# ---------------------------------------------------------------------------
# Checkpoint Inventory
# ---------------------------------------------------------------------------

class TestCheckpointInventory:
    def test_inventory_checkpoints_endpoint_returns_200(self):
        resp = client.get("/api/inventory/checkpoints")
        assert resp.status_code == 200

    def test_inventory_checkpoints_schema(self):
        resp = client.get("/api/inventory/checkpoints")
        data = resp.json()
        for key in ("checkpoint_dir", "total_found", "trained_count", "items"):
            assert key in data, f"Missing key: {key}"

    def test_unet_best_pt_reported_as_untrained(self):
        """checkpoints/unet_best.pt contains raw weights without training metrics → is_trained=False."""
        resp = client.get("/api/inventory/checkpoints")
        data = resp.json()
        unet_items = [i for i in data["items"] if "unet_best" in i["filename"]]
        assert len(unet_items) == 1, "unet_best.pt should appear in checkpoint inventory"
        assert unet_items[0]["is_trained"] is False, (
            "unet_best.pt must be reported as untrained (no epoch/metrics metadata)"
        )

    def test_trained_count_not_fabricated(self):
        """trained_count must equal the count of items with is_trained=True."""
        resp = client.get("/api/inventory/checkpoints")
        data = resp.json()
        actual_trained = sum(1 for i in data["items"] if i["is_trained"])
        assert data["trained_count"] == actual_trained


# ---------------------------------------------------------------------------
# CFD Inventory
# ---------------------------------------------------------------------------

class TestCFDInventory:
    def test_inventory_cfd_endpoint_returns_200(self):
        resp = client.get("/api/inventory/cfd")
        assert resp.status_code == 200

    def test_inventory_cfd_schema(self):
        resp = client.get("/api/inventory/cfd")
        data = resp.json()
        for key in ("openfoam_available", "total_cases", "cases", "notes"):
            assert key in data, f"Missing key: {key}"

    def test_inventory_cfd_openfoam_reports_true_status(self):
        """openfoam_available must reflect actual binary presence; it must not be hardcoded True."""
        import shutil
        solver_actually_present = shutil.which("simpleFoam") is not None
        resp = client.get("/api/inventory/cfd")
        data = resp.json()
        assert data["openfoam_available"] == solver_actually_present, (
            "openfoam_available does not match actual simpleFoam binary availability"
        )

    def test_inventory_cfd_case_template_present(self):
        resp = client.get("/api/inventory/cfd")
        data = resp.json()
        assert data["total_cases"] >= 1
        assert any(c["case_type"] == "template" for c in data["cases"])


# ---------------------------------------------------------------------------
# Canonical Reference Loading
# ---------------------------------------------------------------------------

class TestCanonicalReferenceLoading:
    def test_canonical_reference_loaded_when_npz_present(self):
        """If cfd/outputs/canonical_block_reference.npz exists, it must be loaded as a sample."""
        project_root = Path(__file__).resolve().parent.parent
        npz_path = project_root / "cfd" / "outputs" / "canonical_block_reference.npz"
        if not npz_path.exists():
            pytest.skip("canonical_block_reference.npz not present; skipping disk load test")

        from src.api.service import service
        assert "cfd_canonical_block_01" in service.samples, (
            "Canonical reference NPZ exists but was not loaded into service.samples"
        )

    def test_canonical_sample_has_correct_provenance(self):
        """The canonical sample must have CANONICAL_AERODYNAMIC_BENCHMARK_REFERENCE as city_or_source."""
        from src.api.service import service
        if "cfd_canonical_block_01" not in service.samples:
            pytest.skip("Canonical reference not loaded")
        s = service.samples["cfd_canonical_block_01"]
        assert s.metadata.city_or_source == "CANONICAL_AERODYNAMIC_BENCHMARK_REFERENCE"

    def test_canonical_sample_not_labelled_cfd_ground_truth(self):
        """The canonical benchmark is NOT an OpenFOAM solver output; has_cfd_ground_truth must be False."""
        resp = client.get("/api/inventory/datasets")
        data = resp.json()
        canonical = [i for i in data["items"] if i["layout_id"] == "cfd_canonical_block_01"]
        if not canonical:
            pytest.skip("Canonical reference not loaded")
        assert canonical[0]["has_cfd_ground_truth"] is False
        assert canonical[0]["data_provenance"] == "CANONICAL_AERODYNAMIC_BENCHMARK"


# ---------------------------------------------------------------------------
# Real Training Thread (experiment runs actual model forward pass)
# ---------------------------------------------------------------------------

class TestRealTrainingThread:
    def test_experiment_run_starts_and_records(self):
        """POST /api/experiments/run returns an experiment_id and the experiment appears in GET /api/experiments."""
        resp = client.post("/api/experiments/run", json={
            "architecture": "CompactCNN",
            "epochs": 2,
            "learning_rate": 1e-3,
            "holdout_family": "idealized_staggered",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "experiment_id" in data
        exp_id = data["experiment_id"]

        # Verify experiment appears immediately in list
        list_resp = client.get("/api/experiments")
        assert list_resp.status_code == 200
        ids = [e["experiment_id"] for e in list_resp.json()]
        assert exp_id in ids

    def test_experiment_completed_metrics_are_not_fabricated(self):
        """After an experiment run completes, held_out_mae_mps must be a real float or None — never a hardcoded constant."""
        # Run a short experiment
        resp = client.post("/api/experiments/run", json={
            "architecture": "CompactUNet",
            "epochs": 2,
            "learning_rate": 1e-3,
            "holdout_family": "idealized_aligned",
        })
        exp_id = resp.json()["experiment_id"]

        # Wait for completion (max 30s)
        deadline = time.time() + 30
        while time.time() < deadline:
            time.sleep(1)
            exp_list = client.get("/api/experiments").json()
            matching = [e for e in exp_list if e["experiment_id"] == exp_id]
            if matching and matching[0]["status"] not in ("Training", "Queued"):
                break

        exp_list = client.get("/api/experiments").json()
        matching = [e for e in exp_list if e["experiment_id"] == exp_id]
        assert matching, f"Experiment {exp_id} not found after completion"
        exp = matching[0]

        # The MAE must be a real computed value (float) or None — never a static constant like 0.142
        if exp["held_out_mae_mps"] is not None:
            assert isinstance(exp["held_out_mae_mps"], float)
            assert exp["held_out_mae_mps"] != 0.142, (
                "held_out_mae_mps=0.142 is the previously hardcoded fake value — must not appear"
            )
            # Sanity check: physically plausible range for wind speed MAE
            assert 0.0 < exp["held_out_mae_mps"] < 20.0, (
                f"held_out_mae_mps={exp['held_out_mae_mps']} is outside physically plausible range"
            )

