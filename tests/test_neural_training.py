"""Smoke tests for the neural surrogate training loop, reproducible seeding, and model selection."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.contract import LayoutFamily
from src.data.normalization import FieldNormalizer
from src.data.rasterizer import LayoutRasterizer
from src.models.neural_unet import CompactUNet
from src.training.trainer import (
    NeuralSurrogateTrainer,
    SurrogateDataset,
    TrainerConfig,
    set_seed,
)
from tests.fixtures.synthetic_data import create_synthetic_sample


@pytest.fixture
def tiny_synthetic_dataset():
    """Builds a tiny cohort of synthetic samples strictly for software smoke testing."""
    samples = []
    for i in range(4):
        s = create_synthetic_sample(
            layout_id=f"smoke_sample_{i}",
            layout_family=LayoutFamily.IDEALIZED_ALIGNED if i % 2 == 0 else LayoutFamily.IDEALIZED_STAGGERED,
            grid_shape=(32, 32),
            inflow_speed=4.0 + i,
        )
        samples.append(s)
    return samples


def test_surrogate_dataset_shapes_and_items(tiny_synthetic_dataset):
    """Verify SurrogateDataset yields correctly shaped 4D channel and target tensors."""
    dataset = SurrogateDataset(tiny_synthetic_dataset)
    assert len(dataset) == 4

    inputs, targets, fluid_mask, layout_id = dataset[0]
    assert inputs.shape == (7, 32, 32)
    assert targets.shape == (1, 32, 32)
    assert fluid_mask.shape == (1, 32, 32)
    assert layout_id == "smoke_sample_0"


def test_training_smoke_and_checkpoint_creation(tiny_synthetic_dataset, tmp_path: Path):
    """Smoke test running 3 epochs of training to verify convergence and checkpointing."""
    train_dataset = SurrogateDataset(tiny_synthetic_dataset[:3])
    val_dataset = SurrogateDataset(tiny_synthetic_dataset[3:])

    model = CompactUNet(in_channels=7, out_channels=1, base_channels=8, depth=2)

    config = TrainerConfig(
        epochs=3,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_dir=tmp_path / "ckpts",
        seed=123,
        device="cpu",
    )

    trainer = NeuralSurrogateTrainer(model=model, config=config)
    history = trainer.fit(train_dataset, val_dataset)

    assert len(history["train_loss"]) == 3
    assert len(history["val_nrmse"]) == 3
    assert (tmp_path / "ckpts" / "best_checkpoint.pt").exists()

    # Verify single sample prediction
    pred = trainer.predict_sample(tiny_synthetic_dataset[0])
    assert pred.shape == (1, 32, 32)
    # Check zero inside building
    mask = tiny_synthetic_dataset[0].geometry.building_mask
    np.testing.assert_allclose(pred[0, mask], 0.0)


def test_reproducible_random_seed():
    """Verify set_seed produces deterministic outputs for initialized models."""
    set_seed(42)
    m1 = CompactUNet(in_channels=7, out_channels=1, base_channels=8, depth=2)
    w1 = m1.inc.double_conv[0].weight.clone()

    set_seed(42)
    m2 = CompactUNet(in_channels=7, out_channels=1, base_channels=8, depth=2)
    w2 = m2.inc.double_conv[0].weight.clone()

    torch.testing.assert_close(w1, w2)
