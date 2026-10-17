"""Unit tests for neural surrogate network architectures and forward passes."""

from pathlib import Path
import pytest
import torch

from src.models.neural_unet import CompactCNN, CompactUNet


def test_compact_unet_shapes_and_forward_pass():
    """Verify CompactUNet processes multi-channel spatial inputs to expected output shape."""
    in_channels = 7
    out_channels = 1
    batch_size = 2
    ny, nx = 32, 32

    model = CompactUNet(in_channels=in_channels, out_channels=out_channels, base_channels=16, depth=3)
    x = torch.randn(batch_size, in_channels, ny, nx)

    out = model(x)
    assert out.shape == (batch_size, out_channels, ny, nx)
    assert not torch.isnan(out).any()


def test_compact_unet_building_mask_enforcement():
    """Verify output inside building mask is zeroed out to enforce physical constraints."""
    batch_size = 2
    in_channels = 7
    ny, nx = 32, 32

    model = CompactUNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=16,
        depth=2,
        enforce_zero_inside_buildings=True,
        building_mask_channel_idx=1,
    )

    x = torch.zeros(batch_size, in_channels, ny, nx)
    # Set building mask channel (idx 1) = 1 in central region
    x[:, 1, 10:20, 10:20] = 1.0

    out = model(x)
    # In building region, output must be exactly 0.0
    building_slice = out[:, 0, 10:20, 10:20]
    torch.testing.assert_close(building_slice, torch.zeros_like(building_slice))


def test_compact_cnn_shapes_and_forward_pass():
    """Verify CompactCNN multi-scale dilated architecture forward pass."""
    in_channels = 7
    out_channels = 1
    ny, nx = 32, 32

    model = CompactCNN(in_channels=in_channels, out_channels=out_channels, hidden_channels=16, num_layers=4)
    x = torch.randn(2, in_channels, ny, nx)

    out = model(x)
    assert out.shape == (2, out_channels, ny, nx)
    assert not torch.isnan(out).any()


def test_model_param_counts():
    """Verify parameter counts for compact lightweight models."""
    unet = CompactUNet(in_channels=7, out_channels=1, base_channels=16, depth=2)
    cnn = CompactCNN(in_channels=7, out_channels=1, hidden_channels=16, num_layers=3)

    unet_params = unet.get_param_count()
    cnn_params = cnn.get_param_count()

    assert unet_params > 1000
    assert cnn_params > 1000
    assert unet_params < 2_000_000  # Compact requirement (<2M params)
    assert cnn_params < 500_000


def test_compact_unet_checkpoint_roundtrip(tmp_path: Path):
    """Verify saving and loading weights and metadata preserves state exactly."""
    model = CompactUNet(in_channels=7, out_channels=1, base_channels=16, depth=2)
    x = torch.randn(1, 7, 32, 32)
    orig_out = model(x)

    ckpt_file = tmp_path / "unet_test.pt"
    model.save_checkpoint(ckpt_file, epoch=5, val_metrics={"field_nrmse": 0.08})
    assert ckpt_file.exists()

    loaded_model, meta = CompactUNet.load_checkpoint(ckpt_file)
    loaded_out = loaded_model(x)

    torch.testing.assert_close(orig_out, loaded_out)
    assert meta["epoch"] == 5
    assert meta["val_metrics"]["field_nrmse"] == 0.08


def test_compact_unet_invalid_channel_rejection():
    """Verify wrong channel dimension is rejected."""
    model = CompactUNet(in_channels=7, out_channels=1)
    wrong_input = torch.randn(1, 5, 32, 32)  # 5 instead of 7

    with pytest.raises(ValueError, match="Input channel count mismatch"):
        model(wrong_input)
