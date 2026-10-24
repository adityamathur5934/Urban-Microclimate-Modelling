"""Normalization utilities fitted strictly on training data with inversion support."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class DataLeakageError(RuntimeError):
    """Raised when evaluation or test layouts contaminate normalizer fitting statistics."""
    pass


class FieldNormalizer:
    """Normalizes multi-channel spatial raster tensors or fields.

    Calculates mean and standard deviation (or min/max) channel-wise
    strictly across the training split. Supports masking to exclude
    building interiors from statistics computation.
    """

    def __init__(
        self,
        method: str = "standard",  # "standard" (z-score) or "minmax"
        mask_buildings: bool = True,
        eps: float = 1e-6,
    ):
        if method not in ("standard", "minmax"):
            raise ValueError(f"Unknown normalization method: {method}")
        self.method = method
        self.mask_buildings = mask_buildings
        self.eps = eps

        # Fitted statistics: arrays of shape (C,)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.channel_names_: Optional[Tuple[str, ...]] = None
        self.fitted_layout_ids_: List[str] = []

    @property
    def is_fitted(self) -> bool:
        return (self.mean_ is not None and self.std_ is not None) or (
            self.min_ is not None and self.max_ is not None
        )

    def fit(
        self,
        tensors: List[np.ndarray],
        masks: Optional[List[np.ndarray]] = None,
        channel_names: Optional[Tuple[str, ...]] = None,
        layout_ids: Optional[List[str]] = None,
    ) -> FieldNormalizer:
        """Fit normalization parameters across a collection of training tensors.

        Parameters
        ----------
        tensors : List[np.ndarray]
            List of 3D arrays of shape (C, Ny, Nx).
        masks : Optional[List[np.ndarray]]
            List of 2D boolean masks (Ny, Nx) where True indicates building.
            If mask_buildings=True, masked cells are ignored during fit.
        channel_names : Optional[Tuple[str, ...]]
            Names corresponding to each channel.
        layout_ids : Optional[List[str]]
            List of unique layout identifiers corresponding to the training samples.
            Used for leak-free verification against validation/test holdouts.
        """
        if not tensors:
            raise ValueError("Cannot fit on an empty list of tensors.")

        c_channels = tensors[0].shape[0]
        self.channel_names_ = channel_names

        means = []
        stds = []
        mins = []
        maxs = []

        for c in range(c_channels):
            channel_values = []
            for i, t in enumerate(tensors):
                c_slice = t[c]
                if self.mask_buildings and masks is not None:
                    # Fluid cells only
                    fluid_vals = c_slice[~masks[i]]
                    if len(fluid_vals) > 0:
                        channel_values.append(fluid_vals.flatten())
                    else:
                        channel_values.append(c_slice.flatten())
                else:
                    channel_values.append(c_slice.flatten())

            all_vals = np.concatenate(channel_values)
            m = float(np.mean(all_vals))
            s = float(np.std(all_vals))
            mn = float(np.min(all_vals))
            mx = float(np.max(all_vals))

            # Guard against zero variance
            if s < self.eps:
                s = 1.0
            if (mx - mn) < self.eps:
                mx = mn + 1.0

            means.append(m)
            stds.append(s)
            mins.append(mn)
            maxs.append(mx)

        self.mean_ = np.array(means, dtype=np.float32)
        self.std_ = np.array(stds, dtype=np.float32)
        self.min_ = np.array(mins, dtype=np.float32)
        self.max_ = np.array(maxs, dtype=np.float32)
        self.fitted_layout_ids_ = list(layout_ids) if layout_ids is not None else []

        return self

    def verify_no_leakage(self, test_layout_ids: List[str]) -> None:
        """Verifies that no test or evaluation layout IDs were included during fit().

        Parameters
        ----------
        test_layout_ids : List[str]
            List of layout identifiers assigned to the validation or test holdout split.

        Raises
        ------
        DataLeakageError
            If any test layout ID is found in the training fitted layout IDs.
        """
        if not self.is_fitted:
            raise RuntimeError("FieldNormalizer is not fitted yet.")
        if not self.fitted_layout_ids_:
            return

        overlap = set(self.fitted_layout_ids_).intersection(set(test_layout_ids))
        if overlap:
            sorted_overlap = sorted(list(overlap))
            raise DataLeakageError(
                f"Data leakage detected! FieldNormalizer was fitted on {len(self.fitted_layout_ids_)} layouts, "
                f"which includes layout(s) {sorted_overlap} belonging to the evaluation/test split."
            )

    def transform(self, tensor: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalize a tensor (C, Ny, Nx) using fitted statistics."""
        if not self.is_fitted:
            raise RuntimeError("FieldNormalizer is not fitted yet.")

        out = tensor.astype(np.float32).copy()
        c_channels = out.shape[0]

        if self.method == "standard":
            for c in range(c_channels):
                out[c] = (out[c] - self.mean_[c]) / self.std_[c]
        else:
            for c in range(c_channels):
                out[c] = (out[c] - self.min_[c]) / (self.max_[c] - self.min_[c])

        # If masking is requested, keep building interior strictly zero
        if self.mask_buildings and mask is not None:
            for c in range(c_channels):
                out[c][mask] = 0.0

        return out

    def inverse_transform(self, tensor: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Denormalize a tensor (C, Ny, Nx) back to original physical units."""
        if not self.is_fitted:
            raise RuntimeError("FieldNormalizer is not fitted yet.")

        out = tensor.astype(np.float32).copy()
        c_channels = out.shape[0]

        if self.method == "standard":
            for c in range(c_channels):
                out[c] = (out[c] * self.std_[c]) + self.mean_[c]
        else:
            for c in range(c_channels):
                out[c] = out[c] * (self.max_[c] - self.min_[c]) + self.min_[c]

        if self.mask_buildings and mask is not None:
            for c in range(c_channels):
                out[c][mask] = 0.0

        return out

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters for storage."""
        if not self.is_fitted:
            raise RuntimeError("Cannot export unfitted normalizer.")
        return {
            "method": self.method,
            "mask_buildings": self.mask_buildings,
            "eps": self.eps,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "min": self.min_.tolist() if self.min_ is not None else None,
            "max": self.max_.tolist() if self.max_ is not None else None,
            "channel_names": list(self.channel_names_) if self.channel_names_ else None,
            "fitted_layout_ids": list(self.fitted_layout_ids_),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FieldNormalizer:
        """Instantiate and load pre-fitted statistics."""
        norm = cls(
            method=data["method"],
            mask_buildings=data.get("mask_buildings", True),
            eps=data.get("eps", 1e-6),
        )
        if data.get("mean") is not None:
            norm.mean_ = np.array(data["mean"], dtype=np.float32)
        if data.get("std") is not None:
            norm.std_ = np.array(data["std"], dtype=np.float32)
        if data.get("min") is not None:
            norm.min_ = np.array(data["min"], dtype=np.float32)
        if data.get("max") is not None:
            norm.max_ = np.array(data["max"], dtype=np.float32)
        if data.get("channel_names") is not None:
            norm.channel_names_ = tuple(data["channel_names"])
        if data.get("fitted_layout_ids") is not None:
            norm.fitted_layout_ids_ = list(data["fitted_layout_ids"])
        return norm


class FeatureNormalizer:
    """Normalizes tabular feature matrices (N_samples, D_features) for the ML baseline."""

    def __init__(self, method: str = "standard", eps: float = 1e-6):
        if method not in ("standard", "minmax"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.fitted_layout_ids_: List[str] = []

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None or self.min_ is not None

    def fit(self, X: np.ndarray, layout_ids: Optional[List[str]] = None) -> FeatureNormalizer:
        """Fit on 2D array (N, D).

        Parameters
        ----------
        X : np.ndarray
            Tabular feature matrix (N_samples, D_features).
        layout_ids : Optional[List[str]]
            List of layout identifiers corresponding to rows of X for leakage checks.
        """
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < self.eps] = 1.0

        mn = np.min(X, axis=0)
        mx = np.max(X, axis=0)
        diff = mx - mn
        diff[diff < self.eps] = 1.0

        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        self.min_ = mn.astype(np.float32)
        self.max_ = mx.astype(np.float32)
        self.fitted_layout_ids_ = list(layout_ids) if layout_ids is not None else []
        return self

    def verify_no_leakage(self, test_layout_ids: List[str]) -> None:
        """Verifies that no test or evaluation layout IDs were included during fit()."""
        if not self.is_fitted:
            raise RuntimeError("FeatureNormalizer is not fitted yet.")
        if not self.fitted_layout_ids_:
            return

        overlap = set(self.fitted_layout_ids_).intersection(set(test_layout_ids))
        if overlap:
            sorted_overlap = sorted(list(overlap))
            raise DataLeakageError(
                f"Data leakage detected! FeatureNormalizer was fitted on {len(self.fitted_layout_ids_)} layouts, "
                f"which includes layout(s) {sorted_overlap} belonging to the evaluation/test split."
            )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureNormalizer is not fitted.")
        if self.method == "standard":
            return ((X - self.mean_) / self.std_).astype(np.float32)
        else:
            diff = self.max_ - self.min_
            diff[diff < self.eps] = 1.0
            return ((X - self.min_) / diff).astype(np.float32)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureNormalizer is not fitted.")
        if self.method == "standard":
            return (X * self.std_ + self.mean_).astype(np.float32)
        else:
            diff = self.max_ - self.min_
            return (X * diff + self.min_).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "eps": self.eps,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "min": self.min_.tolist() if self.min_ is not None else None,
            "max": self.max_.tolist() if self.max_ is not None else None,
            "fitted_layout_ids": list(self.fitted_layout_ids_),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureNormalizer:
        norm = cls(method=data["method"], eps=data.get("eps", 1e-6))
        if data.get("mean") is not None:
            norm.mean_ = np.array(data["mean"], dtype=np.float32)
        if data.get("std") is not None:
            norm.std_ = np.array(data["std"], dtype=np.float32)
        if data.get("min") is not None:
            norm.min_ = np.array(data["min"], dtype=np.float32)
        if data.get("max") is not None:
            norm.max_ = np.array(data["max"], dtype=np.float32)
        if data.get("fitted_layout_ids") is not None:
            norm.fitted_layout_ids_ = list(data["fitted_layout_ids"])
        return norm
