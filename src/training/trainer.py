"""Reproducible PyTorch training loop, validation-based model selection, and checkpointing.

Designed for neural surrogates operating on multi-channel rasterized CFD layouts.
Supports both NeuralSurrogateTrainer (Task 4/5 integration) and SurrogateTrainer (prototype pipeline).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.contract import UrbanLayoutSample
from src.data.normalization import FieldNormalizer
from src.data.rasterizer import LayoutRasterizer, RasterizedInput
from src.training.losses import (
    CombinedSurrogateLoss,
    CompositeSurrogateLoss,
    MaskedL1Loss,
    MaskedMSELoss,
)
from src.training.metrics import (
    compute_surrogate_metrics_summary,
    field_mae,
    field_nrmse,
)


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_default_device() -> torch.device:
    """Detects available compute hardware (MPS on Apple Silicon, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SurrogateDataset(Dataset):
    """PyTorch Dataset wrapping rasterized microclimate layouts.

    Returns:
      inputs: Tensor of shape (C_in, Ny, Nx)
      targets: Tensor of shape (C_out, Ny, Nx)
      fluid_mask: Tensor of shape (1, Ny, Nx) where 1=fluid, 0=building
      layout_id: str
    """

    def __init__(
        self,
        samples: Union[List[UrbanLayoutSample], List[RasterizedInput]],
        rasterizer: Optional[LayoutRasterizer] = None,
        field_normalizer: Optional[FieldNormalizer] = None,
        target_channel_idx: Optional[Union[int, List[int], Tuple[int, ...]]] = 0,
    ):
        self.rasterizer = rasterizer or LayoutRasterizer()
        self.field_normalizer = field_normalizer
        self.target_channel_idx = target_channel_idx
        self.raster_items: List[RasterizedInput] = []

        for item in samples:
            if isinstance(item, UrbanLayoutSample):
                self.raster_items.append(self.rasterizer.rasterize(item))
            elif isinstance(item, RasterizedInput):
                self.raster_items.append(item)
            else:
                raise TypeError(f"Unsupported item type in SurrogateDataset: {type(item)}")

    def __len__(self) -> int:
        return len(self.raster_items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        item = self.raster_items[idx]
        channels = item.channels.copy()          # (C_in, Ny, Nx)
        target = item.target_field.copy()        # (C_out_full, Ny, Nx)
        building_mask = item.building_mask       # (Ny, Nx) bool
        fluid_mask = (~building_mask).astype(np.float32)[None, :, :]  # (1, Ny, Nx)

        if self.target_channel_idx is not None:
            if isinstance(self.target_channel_idx, int):
                target = target[self.target_channel_idx : self.target_channel_idx + 1]
            else:
                target = target[list(self.target_channel_idx)]

        if self.field_normalizer is not None:
            # Normalize target using train-fitted normalizer
            target = self.field_normalizer.transform_field(target)

        inputs_t = torch.from_numpy(channels).float()
        targets_t = torch.from_numpy(target).float()
        fluid_mask_t = torch.from_numpy(fluid_mask).float()

        return inputs_t, targets_t, fluid_mask_t, item.layout_id


@dataclass
class TrainerConfig:
    """Configuration for reproducible neural surrogate training (teammate API)."""
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    loss_alpha: float = 0.7  # weight on L1 vs MSE in composite loss
    loss_grad_weight: float = 0.05
    checkpoint_dir: Union[str, Path] = "checkpoints"
    seed: int = 42
    device: Optional[str] = None  # None for auto-detection


class NeuralSurrogateTrainer:
    """Encapsulates training, validation, early stopping, and checkpoint management."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None,
    ):
        self.config = config or TrainerConfig()
        set_seed(self.config.seed)

        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = get_default_device()

        self.model = model.to(self.device)
        self.criterion = CombinedSurrogateLoss(
            alpha=self.config.loss_alpha,
            grad_weight=self.config.loss_grad_weight,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.epochs),
            eta_min=1e-5,
        )
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_nrmse": [],
        }
        self.best_val_nrmse: float = float("inf")
        self.best_checkpoint_path: Optional[Path] = None

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> Dict[str, List[float]]:
        """Executes training loop with validation-based model selection."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.config.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=False,
        )

        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(self.config.batch_size, len(val_dataset)),
                shuffle=False,
            )

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            # --- Training Phase ---
            self.model.train()
            train_losses = []

            for inputs, targets, fluid_masks, _ in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                fluid_masks = fluid_masks.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(inputs)
                loss = self.criterion(preds, targets, fluid_masks)
                loss.backward()

                # Gradient clipping for stable convergence
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            self.scheduler.step()
            mean_train_loss = float(np.mean(train_losses))
            self.history["train_loss"].append(mean_train_loss)

            # --- Validation Phase ---
            if val_loader is not None:
                val_metrics = self.evaluate_loader(val_loader)
                val_loss = val_metrics["loss"]
                val_mae = val_metrics["field_mae"]
                val_nrmse = val_metrics["field_nrmse_range"]

                self.history["val_loss"].append(val_loss)
                self.history["val_mae"].append(val_mae)
                self.history["val_nrmse"].append(val_nrmse)

                # Validation-based model selection (checkpointing on lowest NRMSE)
                if val_nrmse < self.best_val_nrmse:
                    self.best_val_nrmse = val_nrmse
                    epochs_without_improvement = 0
                    ckpt_path = checkpoint_dir / "best_checkpoint.pt"
                    self._save_checkpoint(ckpt_path, epoch, val_metrics)
                    self.best_checkpoint_path = ckpt_path
                else:
                    epochs_without_improvement += 1

                # Early stopping check
                if epochs_without_improvement >= self.config.patience:
                    break
            else:
                # If no validation set, checkpoint latest
                ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                self._save_checkpoint(ckpt_path, epoch, {"train_loss": mean_train_loss})
                self.best_checkpoint_path = ckpt_path

        return self.history

    def evaluate_loader(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluates model over DataLoader without gradients."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for inputs, targets, fluid_masks, _ in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                fluid_masks = fluid_masks.to(self.device)

                preds = self.model(inputs)
                loss = self.criterion(preds, targets, fluid_masks)
                total_loss += loss.item() * inputs.size(0)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_masks.append(fluid_masks.cpu())

        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        cat_masks = torch.cat(all_masks, dim=0)

        n_samples = cat_preds.size(0)
        mean_loss = total_loss / max(1, n_samples)

        metrics = compute_surrogate_metrics_summary(
            pred=cat_preds,
            target=cat_targets,
            fluid_mask=cat_masks,
        )
        metrics["loss"] = mean_loss
        return metrics

    def predict_sample(
        self,
        sample_or_raster: Union[UrbanLayoutSample, RasterizedInput],
        rasterizer: Optional[LayoutRasterizer] = None,
        field_normalizer: Optional[FieldNormalizer] = None,
    ) -> np.ndarray:
        """Generates spatial prediction field for a single layout sample."""
        if isinstance(sample_or_raster, UrbanLayoutSample):
            r = (rasterizer or LayoutRasterizer()).rasterize(sample_or_raster)
        else:
            r = sample_or_raster

        channels = torch.from_numpy(r.channels[None, ...]).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(channels).cpu().numpy()[0]  # (C_out, Ny, Nx)

        if field_normalizer is not None:
            pred = field_normalizer.inverse_transform_field(pred)

        # Enforce zero in solid buildings
        pred[:, r.building_mask] = 0.0
        return pred

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]) -> None:
        """Internal helper calling model's save_checkpoint interface."""
        if hasattr(self.model, "save_checkpoint"):
            self.model.save_checkpoint(
                path=path,
                epoch=epoch,
                val_metrics=metrics,
                optimizer_state=self.optimizer.state_dict(),
                extra_metadata={"config": self.config.__dict__},
            )
        else:
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "optimizer_state": self.optimizer.state_dict(),
                },
                path,
            )


# ============================================================================
# Prototype Pipeline Compatibility (SurrogateTrainer & TrainingConfig)
# ============================================================================

@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for surrogate training."""
    seed: int = 42
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    alpha_l1: float = 1.0
    alpha_grad: float = 0.2
    early_stopping_patience: int = 6
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"  # "auto", "cpu", "mps", "cuda"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SurrogateTrainer:
    """Orchestrates model training, validation, early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        self.config = config if config is not None else TrainingConfig()
        self.set_seed(self.config.seed)

        self.device = self._resolve_device(self.config.device)
        self.model = model.to(self.device)

        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = CompositeSurrogateLoss(
                alpha_l1=self.config.alpha_l1,
                alpha_mse=0.0,
                alpha_grad=self.config.alpha_grad,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    @staticmethod
    def set_seed(seed: int) -> None:
        """Sets deterministic random seeds across all libraries."""
        set_seed(seed)

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str != "auto":
            return torch.device(device_str)
        return get_default_device()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Runs one epoch of training over fluid cells."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y, mask, _ in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)

            # Ensure prediction is zeroed inside building before loss calculation
            pred = pred * (~mask).float()

            loss = self.loss_fn(pred, y, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self, dataloader: DataLoader) -> float:
        """Evaluates validation loss without computing gradients."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y, mask, _ in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)

                pred = self.model(x)
                pred = pred * (~mask).float()
                loss = self.loss_fn(pred, y, mask=mask)

                total_loss += float(loss.item())
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def fit(
        self,
        train_dataset: Any,
        val_dataset: Any,
    ) -> Dict[str, List[float]]:
        """Executes full training and validation cycles with checkpointing."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(checkpoint_dir / "best_model.pt", epoch=epoch, val_loss=val_loss)
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                break

        return self.history

    def save_checkpoint(self, path: Path | str, epoch: int, val_loss: float) -> None:
        """Saves weights and metadata."""
        state = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: Path | str) -> int:
        """Restores model weights from saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint.get("epoch", 0))
