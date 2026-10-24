"""Reproducible training loop, checkpointing, and validation for neural surrogates."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import CompositeSurrogateLoss
from src.training.dataset import UrbanSurrogateDataset


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
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str != "auto":
            return torch.device(device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

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

            loss = self.loss_fn(pred, y, mask)
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
                loss = self.loss_fn(pred, y, mask)

                total_loss += float(loss.item())
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def fit(
        self,
        train_dataset: UrbanSurrogateDataset,
        val_dataset: UrbanSurrogateDataset,
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
