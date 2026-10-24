"""Unit tests for neural surrogate models, loss functions, training loop, and evaluation."""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.models.unet import CompactUNet, SimpleCNNBaseline
from src.training.losses import MaskedL1Loss, MaskedMSELoss, CompositeSurrogateLoss
from src.training.dataset import UrbanSurrogateDataset
from src.training.trainer import SurrogateTrainer, TrainingConfig
from src.data.normalization import FieldNormalizer
from src.data.rasterizer import LayoutRasterizer
from src.evaluation.metrics import (
    compute_field_mae,
    compute_field_rmse,
    compute_normalized_rmse,
    compute_r2_score,
)
from src.evaluation.evaluator import SurrogateEvaluator
from src.evaluation.plotting import plot_prediction_comparison
from tests.fixtures.synthetic_data import create_synthetic_sample


# ============================================================================
# 1. Model Architecture Tests
# ============================================================================

def test_compact_unet_forward_pass_shape():
    """Verify CompactUNet processes (B, 7, H, W) to (B, 1, H, W)."""
    model = CompactUNet(in_channels=7, out_channels=1, base_channels=16)
    x = torch.randn(2, 7, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)
    assert not torch.isnan(out).any()


def test_simple_cnn_baseline_shape():
    """Verify SimpleCNNBaseline processes (B, 7, H, W) to (B, 1, H, W)."""
    model = SimpleCNNBaseline(in_channels=7, out_channels=1, hidden_channels=16)
    x = torch.randn(2, 7, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


# ============================================================================
# 2. Physics-Aware Masked Loss Tests
# ============================================================================

def test_masked_l1_ignores_building_interiors():
    """Errors inside building interiors must NOT contribute to loss."""
    pred = torch.tensor([[[[10.0, 5.0], [999.0, 999.0]]]])  # Bottom row is solid building
    target = torch.tensor([[[[10.0, 3.0], [0.0, 0.0]]]])
    mask = torch.tensor([[[[False, False], [True, True]]]])

    loss_fn = MaskedL1Loss()
    loss = loss_fn(pred, target, mask)

    # Fluid cells: |10-10| = 0, |5-3| = 2. Mean over 2 fluid cells = 1.0.
    # The 999.0 inside building is completely ignored.
    assert pytest.approx(loss.item(), abs=1e-5) == 1.0


def test_composite_loss_backward():
    """Verify gradients propagate through CompositeSurrogateLoss."""
    pred = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.randn(2, 1, 16, 16)
    mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    mask[:, :, 4:8, 4:8] = True  # Central building

    loss_fn = CompositeSurrogateLoss(alpha_l1=1.0, alpha_grad=0.2)
    loss = loss_fn(pred, target, mask)
    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


# ============================================================================
# 3. Dataset Adapter and Trainer Integration
# ============================================================================

def test_surrogate_dataset_and_training_smoke_test(tmp_path):
    """End-to-end smoke test: create synthetic layouts, fit normalizer, train 2 epochs, checkpoint."""
    s1 = create_synthetic_sample("layout_train_1", inflow_speed=5.0)
    s2 = create_synthetic_sample("layout_train_2", inflow_speed=7.0)
    s_val = create_synthetic_sample("layout_val", inflow_speed=6.0)

    rasterizer = LayoutRasterizer()
    r1 = rasterizer.rasterize(s1)
    r2 = rasterizer.rasterize(s2)

    # Fit target normalizer on train split only
    norm = FieldNormalizer(method="standard", mask_buildings=True)
    norm.fit([r1.target_field, r2.target_field], masks=[r1.building_mask, r2.building_mask])

    train_ds = UrbanSurrogateDataset([s1, s2], target_normalizer=norm)
    val_ds = UrbanSurrogateDataset([s_val], target_normalizer=norm)

    assert len(train_ds) == 2
    x, y, m, lid = train_ds[0]
    assert x.shape == (7, 32, 32)
    assert y.shape == (1, 32, 32)
    assert m.shape == (1, 32, 32)

    # Train compact model for 2 epochs
    model = CompactUNet(in_channels=7, out_channels=1, base_channels=8)
    config = TrainingConfig(
        epochs=2,
        batch_size=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )
    trainer = SurrogateTrainer(model, config)
    history = trainer.fit(train_ds, val_ds)

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert (tmp_path / "checkpoints" / "best_model.pt").exists()


# ============================================================================
# 4. Evaluation and Plotting Tests
# ============================================================================

def test_surrogate_evaluator_and_metrics():
    """Verify SurrogateEvaluator calculates physical metrics with inverse normalization."""
    s_test = create_synthetic_sample("test_eval_01", inflow_speed=5.0)
    rasterizer = LayoutRasterizer()
    r_test = rasterizer.rasterize(s_test)

    norm = FieldNormalizer(method="standard", mask_buildings=True)
    norm.fit([r_test.target_field], masks=[r_test.building_mask])

    test_ds = UrbanSurrogateDataset([s_test], target_normalizer=norm)
    model = CompactUNet(in_channels=7, out_channels=1, base_channels=8)

    evaluator = SurrogateEvaluator(target_normalizer=norm)
    summary = evaluator.evaluate(model, test_ds, device="cpu")

    assert summary.mean_mae_mps >= 0.0
    assert summary.mean_nrmse_percent >= 0.0
    assert summary.mean_latency_ms > 0.0
    assert len(summary.layout_results) == 1


def test_plot_prediction_comparison(tmp_path):
    """Verify 3-panel physical plot generation saves to disk."""
    s = create_synthetic_sample("plot_sample", inflow_speed=5.0)
    y_true = s.targets.pedestrian_wind_speed
    y_pred = y_true * 0.95 + 0.1
    mask = s.geometry.building_mask

    out_file = tmp_path / "prediction_comparison.png"
    saved_path = plot_prediction_comparison(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        layout_id=s.layout_id,
        output_path=out_file,
    )

    assert saved_path.exists()
    assert saved_path.stat().st_size > 1000  # Non-empty image file
