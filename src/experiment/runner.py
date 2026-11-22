"""End-to-end orchestration of the CFD-to-surrogate experiment.

Runs the full reproducible workflow:
  1. generate a fixed-grid analytical reference dataset (NOT CFD);
  2. split by complete layouts (zero leakage);
  3. fit input/target normalizers on the TRAIN split only;
  4. genuinely train Random Forest, a U-Net ensemble, and a dilated CNN;
  5. evaluate every model on held-out test layouts (field metrics + timing);
  6. estimate predictive uncertainty from the U-Net ensemble;
  7. save checkpoints, a results manifest, training curves, and figures.

Honesty contract
-----------------
* Reference fields are analytical (Castro & Robins 1977 formulation), labelled
  ``ANALYTICAL_REFERENCE`` — they are **not** CFD. Every metric and figure is
  "surrogate vs analytical reference", never claimed as "surrogate vs CFD".
* No checkpoint is reported as trained unless it is written to disk and
  reloaded successfully for the evaluation pass.
* Inference timing is a single-process per-sample forward pass on the recorded
  device; no CFD-vs-surrogate speedup is claimed (no CFD ran here).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.normalization import FieldNormalizer
from src.data.rasterizer import LayoutRasterizer
from src.data.split import LayoutSplitter
from src.evaluation.metrics import (
    compute_field_mae,
    compute_field_rmse,
    compute_max_error,
    compute_normalized_rmse,
    compute_r2_score,
    compute_regression_metrics,
    compute_spatial_error_map,
)
from src.evaluation.plotting import plot_field_triplet
from src.models.baseline_rf import RandomForestBaseline
from src.models.neural_unet import CompactCNN, CompactUNet
from src.training.trainer import NeuralSurrogateTrainer, TrainerConfig, get_default_device, set_seed
from src.training.visualize import plot_baseline_scatter, plot_training_curves

from src.experiment.dataset import ExperimentDataset, fit_field_normalizers
from src.experiment.reference import REFERENCE_PROVENANCE, generate_reference_dataset
from src.experiment.uncertainty import ensemble_predict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ExperimentConfig:
    """Reproducible experiment configuration."""
    seed: int = 42
    dataset_source: str = "reference"  # "urbantales" or "reference"
    n_layouts: int = 18
    grid_shape: Tuple[int, int] = (64, 64)
    domain_size_m: Tuple[float, float, float] = (128.0, 128.0, 60.0)
    train_ratio: float = 0.66
    val_ratio: float = 0.17
    test_ratio: float = 0.17
    # neural training
    epochs: int = 25
    batch_size: int = 4
    learning_rate: float = 1e-3
    unet_base_channels: int = 16
    unet_depth: int = 3
    cnn_hidden_channels: int = 16
    cnn_num_layers: int = 4
    ensemble_size: int = 3  # number of independently trained U-Nets
    patience: int = 10
    device: Optional[str] = None  # None => auto (MPS/CUDA/CPU)
    # outputs
    checkpoint_dir: str = "checkpoints/cfd_surrogate_experiment"
    output_dir: str = "outputs/cfd_surrogate_experiment"
    manifest_path: str = "configs/cfd_surrogate_experiment.json"
    generate_figures: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["grid_shape"] = list(self.grid_shape)
        d["domain_size_m"] = list(self.domain_size_m)
        return d


@dataclass
class ExperimentResult:
    config: Dict[str, Any]
    split: Dict[str, Any]
    models: Dict[str, Any]
    metrics: Dict[str, Any]
    uncertainty: Dict[str, Any]
    artifacts: Dict[str, Any]
    provenance: str = REFERENCE_PROVENANCE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _device(cfg: ExperimentConfig) -> torch.device:
    if cfg.device is not None:
        return torch.device(cfg.device)
    return get_default_device()


def _split_samples(samples: List, cfg: ExperimentConfig):
    # Partition by base geometry to guarantee zero geometry-level leakage
    # e.g., urbantales_UA0625_d00 and urbantales_UA0625_d45 share base geometry urbantales_UA0625
    layout_to_group = {
        s.layout_id: (s.layout_id.rsplit('_d', 1)[0] if '_d' in s.layout_id else s.layout_id)
        for s in samples
    }
    split = LayoutSplitter.split_by_group(
        layout_to_group=layout_to_group,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )
    split.assert_no_overlap()
    by_id = {s.layout_id: s for s in samples}
    train = [by_id[i] for i in split.train_layout_ids]
    val = [by_id[i] for i in split.val_layout_ids]
    test = [by_id[i] for i in split.test_layout_ids]
    return split, train, val, test


def _train_one_neural(
    kind: str,  # "unet" or "cnn"
    seed: int,
    train_ds: ExperimentDataset,
    val_ds: ExperimentDataset,
    ckpt_dir: Path,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, List[float]], Path]:
    """Genuinely train one neural model and return (model, history, ckpt_path)."""
    set_seed(seed)
    in_ch = train_ds[0][0].shape[0]
    if kind == "unet":
        model = CompactUNet(
            in_channels=in_ch, out_channels=1,
            base_channels=cfg.unet_base_channels, depth=cfg.unet_depth,
        )
        arch = "CompactUNet"
    else:
        model = CompactCNN(
            in_channels=in_ch, out_channels=1,
            hidden_channels=cfg.cnn_hidden_channels, num_layers=cfg.cnn_num_layers,
        )
        arch = "CompactCNN"

    sub = ckpt_dir / f"{arch.lower()}_seed{seed}"
    sub.mkdir(parents=True, exist_ok=True)
    config = TrainerConfig(
        epochs=cfg.epochs,
        batch_size=min(cfg.batch_size, len(train_ds)),
        learning_rate=cfg.learning_rate,
        patience=cfg.patience,
        checkpoint_dir=str(sub),
        seed=seed,
        device=str(device),
    )
    trainer = NeuralSurrogateTrainer(model=model, config=config)
    history = trainer.fit(train_ds, val_ds if len(val_ds) > 0 else None)

    ckpt_path = sub / "best_checkpoint.pt"
    if not ckpt_path.exists():
        # Fall back: trainer may have used checkpoint_epoch_* if no validation set.
        cands = sorted(sub.glob("checkpoint_epoch_*.pt"))
        ckpt_path = cands[-1] if cands else sub / "best_checkpoint.pt"
    # Reload weights from disk to prove the checkpoint is loadable for inference.
    if kind == "unet":
        model_loaded, _ = CompactUNet.load_checkpoint(ckpt_path, device=str(device))
        model = model_loaded
    else:
        model_loaded, _ = CompactCNN.load_checkpoint(ckpt_path, device=str(device))
        model = model_loaded
    return model, history, ckpt_path


@torch.no_grad()
def _predict_field(
    model: torch.nn.Module,
    sample,
    rasterizer: LayoutRasterizer,
    input_norm: FieldNormalizer,
    target_norm: FieldNormalizer,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    """Forward pass for one layout -> (pred_field (Ny,Nx) m/s, latency_ms)."""
    r = rasterizer.rasterize(sample)
    mask = r.building_mask
    x = r.channels.copy()
    if input_norm is not None and input_norm.is_fitted:
        x = input_norm.transform(x, mask=mask)
    x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)
    model.eval()
    model.to(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(x_t)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    out_np = out.squeeze(0).cpu().numpy()  # (1, Ny, Nx)
    if target_norm is not None and target_norm.is_fitted:
        out_np = target_norm.inverse_transform(out_np, mask=mask)
    pred = np.clip(out_np[0], 0.0, None)
    pred[mask] = 0.0
    return pred.astype(np.float32), float(latency_ms)


def _evaluate_neural(
    model, test_samples, rasterizer, input_norm, target_norm, device
) -> Dict[str, Any]:
    layout_results = []
    for s in test_samples:
        pred, lat = _predict_field(model, s, rasterizer, input_norm, target_norm, device)
        gt = s.targets.pedestrian_wind_speed
        mask = s.geometry.building_mask
        u_ref = float(s.environment.inflow_speed_mps)
        m = compute_regression_metrics(gt, pred, building_mask=mask, u_ref=u_ref)
        layout_results.append({
            "layout_id": s.layout_id,
            "mae_mps": _float(m["mae_mps"]),
            "rmse_mps": _float(m["rmse_mps"]),
            "nrmse_range_percent": _float(m["nrmse_range_percent"]),
            "nrmse_inflow_percent": _float(m.get("nrmse_inflow_percent", float("nan"))),
            "max_error_mps": _float(m["max_error_mps"]),
            "r2_score": _float(m["r2_score"]),
            "inference_ms": round(lat, 3),
            "u_ref_mps": u_ref,
        })
    agg = _aggregate(layout_results)
    return {"per_layout": layout_results, "aggregate": agg}


def _aggregate(layout_results: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = ["mae_mps", "rmse_mps", "nrmse_range_percent", "nrmse_inflow_percent", "max_error_mps", "r2_score", "inference_ms"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in layout_results if k in r and not math.isnan(r[k])]
        agg[f"mean_{k}"] = round(float(np.mean(vals)), 5) if vals else None
    return agg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(config: Optional[ExperimentConfig] = None) -> ExperimentResult:
    cfg = config or ExperimentConfig()
    set_seed(cfg.seed)
    device = _device(cfg)

    ckpt_dir = PROJECT_ROOT / cfg.checkpoint_dir
    out_dir = PROJECT_ROOT / cfg.output_dir
    manifest_path = PROJECT_ROOT / cfg.manifest_path
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (manifest_path.parent).mkdir(parents=True, exist_ok=True)

    # 1. Load dataset — real UrbanTALES LES or fallback analytical reference
    if getattr(cfg, "dataset_source", "reference") == "urbantales":
        from src.data.urbantales_loader import load_urbantales_dataset, MANIFEST_PATH
        if MANIFEST_PATH.exists():
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Loading real UrbanTALES LES dataset (20 cases, Nazarian et al. 2025)...")
            samples = load_urbantales_dataset()
            logger.info(f"Loaded {len(samples)} real UrbanTALES LES samples.")
            provenance = "REAL_URBANTALES_LES"
        else:
            raise FileNotFoundError("UrbanTALES manifest not found. Run the download script first.")
    else:
        # Fallback: analytical reference dataset
        samples = generate_reference_dataset(
            n_layouts=cfg.n_layouts, seed=cfg.seed,
            grid_shape=cfg.grid_shape, domain_size_m=cfg.domain_size_m,
        )
        provenance = REFERENCE_PROVENANCE

    # 2. Layout-level split (zero leakage)
    split, train_s, val_s, test_s = _split_samples(samples, cfg)
    split_dict = split.to_dict()

    # 3. Train-only normalization + leakage verification
    input_norm, target_norm, rasterizer = fit_field_normalizers(train_s)
    held_out_ids = split.val_layout_ids + split.test_layout_ids
    input_norm.verify_no_leakage(held_out_ids)
    target_norm.verify_no_leakage(held_out_ids)

    train_ds = ExperimentDataset(train_s, rasterizer, input_norm, target_norm)
    val_ds = ExperimentDataset(val_s, rasterizer, input_norm, target_norm)
    test_ds = ExperimentDataset(test_s, rasterizer, input_norm, target_norm)

    models_info: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {
        "checkpoint_dir": str(ckpt_dir),
        "output_dir": str(out_dir),
        "figures": [],
        "checkpoints": [],
    }

    # 4a. Random Forest (scalar area-averaged target)
    set_seed(cfg.seed)
    rf = RandomForestBaseline(n_estimators=100, max_depth=12, random_state=cfg.seed)
    rf.fit_from_samples(train_s)
    rf_path = ckpt_dir / "rf_baseline.joblib"
    rf.save(rf_path)
    rf = RandomForestBaseline.load(rf_path)  # reload to prove checkpoint loadable
    # scalar evaluation vs true area-averaged wind speed
    rf_scalar = rf.evaluate_from_samples(test_s)
    # broadcast spatial evaluation vs pedestrian field
    rf_layout = []
    for s in test_s:
        scalar_pred = float(rf.predict(extract_features(s).reshape(1, -1))[0])
        pred_field = np.where(s.geometry.building_mask, 0.0, scalar_pred).astype(np.float32)
        gt = s.targets.pedestrian_wind_speed
        m = compute_regression_metrics(gt, pred_field, building_mask=s.geometry.building_mask, u_ref=float(s.environment.inflow_speed_mps))
        rf_layout.append({"layout_id": s.layout_id, "scalar_pred_mps": round(scalar_pred, 4), "mae_mps": _float(m["mae_mps"]), "rmse_mps": _float(m["rmse_mps"]), "nrmse_inflow_percent": _float(m.get("nrmse_inflow_percent", float("nan"))), "max_error_mps": _float(m["max_error_mps"])})
    models_info["random_forest"] = {"checkpoint": str(rf_path), "is_fitted": True, "feature_dim": 26}
    metrics["random_forest"] = {"scalar": rf_scalar, "broadcast_spatial": {"per_layout": rf_layout, "aggregate": _aggregate(rf_layout)}}
    artifacts["checkpoints"].append(str(rf_path))

    # 4b. U-Net ensemble (independent seeds) + 4c. single CNN
    unet_models: List[torch.nn.Module] = []
    unet_histories: List[Dict[str, List[float]]] = []
    unet_paths: List[str] = []
    for k in range(cfg.ensemble_size):
        seed_k = cfg.seed + 100 + k
        model, history, ckpt = _train_one_neural("unet", seed_k, train_ds, val_ds, ckpt_dir, cfg, device)
        unet_models.append(model)
        unet_histories.append(history)
        unet_paths.append(str(ckpt))
        artifacts["checkpoints"].append(str(ckpt))
        if cfg.generate_figures:
            figp = out_dir / f"training_curves_unet_member{k}.png"
            plot_training_curves(history, save_path=figp)
            artifacts["figures"].append(str(figp))

    cnn_model, cnn_history, cnn_ckpt = _train_one_neural("cnn", cfg.seed + 200, train_ds, val_ds, ckpt_dir, cfg, device)
    models_info["cnn"] = {"checkpoint": str(cnn_ckpt), "is_trained": True, "params": int(cnn_model.get_param_count())}
    artifacts["checkpoints"].append(str(cnn_ckpt))
    if cfg.generate_figures:
        figp = out_dir / "training_curves_cnn.png"
        plot_training_curves(cnn_history, save_path=figp)
        artifacts["figures"].append(str(figp))

    # 5. Evaluation on held-out test layouts
    metrics["unet_ensemble_mean"] = _evaluate_neural(_mean_ensemble(unet_models), test_s, rasterizer, input_norm, target_norm, device)
    metrics["unet_member0"] = _evaluate_neural(unet_models[0], test_s, rasterizer, input_norm, target_norm, device)
    metrics["cnn"] = _evaluate_neural(cnn_model, test_s, rasterizer, input_norm, target_norm, device)
    models_info["unet_ensemble"] = {
        "size": cfg.ensemble_size, "checkpoints": unet_paths, "is_trained": True,
        "seeds": [cfg.seed + 100 + k for k in range(cfg.ensemble_size)],
    }

    # 6. Uncertainty: ensemble predictive disagreement over test layouts
    mean_pred, std_pred, preds_all, bmask_all = ensemble_predict(
        unet_models, test_s, rasterizer, input_norm, target_norm, device
    )
    unc_summary = []
    for i, s in enumerate(test_s):
        u = {
            "layout_id": s.layout_id,
            "mean_pixel_std_mps": round(float(np.mean(std_pred[i][~bmask_all[i]])), 5),
            "max_pixel_std_mps": round(float(np.max(std_pred[i][~bmask_all[i]])), 5),
            "p95_pixel_std_mps": round(float(np.percentile(std_pred[i][~bmask_all[i]], 95)), 5),
        }
        unc_summary.append(u)
    metrics["uncertainty"] = {"per_layout": unc_summary, "method": "ensemble_predictive_disagreement (U-Net std, NOT calibrated)"}

    # 7. Figures: per-model reference/prediction/error triplets + uncertainty + RF scatter
    if cfg.generate_figures and test_s:
        s0 = test_s[0]
        gt0 = s0.targets.pedestrian_wind_speed
        mask0 = s0.geometry.building_mask
        for name, mdl in [("unet_ensemble_mean", _mean_ensemble(unet_models)), ("unet_member0", unet_models[0]), ("cnn", cnn_model)]:
            pred0, _ = _predict_field(mdl, s0, rasterizer, input_norm, target_norm, device)
            figp = out_dir / f"triplet_{name}_{s0.layout_id}.png"
            plot_field_triplet(
                y_true=gt0, y_pred=pred0, mask=mask0, layout_id=s0.layout_id,
                output_path=figp, units="m/s", inflow_speed=float(s0.environment.inflow_speed_mps),
                wind_direction_deg=float(s0.environment.wind_direction_deg),
                provenance_tag=provenance,
            )
            artifacts["figures"].append(str(figp))
        # RF broadcast spatial triplet for first test layout
        rf_scalar_pred = float(rf.predict(extract_features(s0).reshape(1, -1))[0])
        rf_field0 = np.where(mask0, 0.0, rf_scalar_pred).astype(np.float32)
        figp = out_dir / f"triplet_rf_broadcast_{s0.layout_id}.png"
        plot_field_triplet(
            y_true=gt0, y_pred=rf_field0, mask=mask0, layout_id=s0.layout_id,
            output_path=figp, units="m/s", inflow_speed=float(s0.environment.inflow_speed_mps),
            wind_direction_deg=float(s0.environment.wind_direction_deg),
            provenance_tag=provenance,
        )
        artifacts["figures"].append(str(figp))
        # RF scalar scatter (true vs predicted area-averaged over test)
        y_true_avg = np.array([_true_area_avg(s) for s in test_s], dtype=np.float32)
        y_pred_avg = np.array([float(rf.predict(extract_features(s).reshape(1, -1))[0]) for s in test_s], dtype=np.float32)
        figp = out_dir / "scatter_rf_area_averaged.png"
        plot_baseline_scatter(y_true_avg, y_pred_avg, save_path=figp, target_name="Area-Averaged Wind Speed (m/s)")
        artifacts["figures"].append(str(figp))
        # Uncertainty map for first test layout (ensemble mean + std)
        figp = out_dir / f"uncertainty_{s0.layout_id}.png"
        _plot_uncertainty(mean_pred[0], std_pred[0], mask0, s0.layout_id, figp)
        artifacts["figures"].append(str(figp))

    # 8. Persist manifest (tracked)
    result = ExperimentResult(
        config=cfg.to_dict(),
        split=split_dict,
        models=models_info,
        metrics=metrics,
        uncertainty=metrics["uncertainty"],
        artifacts={**artifacts, "normalizer_fitted_layout_ids": input_norm.fitted_layout_ids_},
        provenance=provenance,
    )
    manifest_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Small utilities used above
# ---------------------------------------------------------------------------

def extract_features(sample) -> np.ndarray:
    from src.data.features import GeometricFeatureExtractor
    return GeometricFeatureExtractor().extract(sample).to_feature_vector()


def _true_area_avg(sample) -> float:
    mask = sample.geometry.building_mask
    return float(np.mean(sample.targets.pedestrian_wind_speed[~mask])) if np.any(~mask) else 0.0


class _MeanEnsemble(torch.nn.Module):
    """Wrapper that averages the outputs of several models (for eval only)."""
    def __init__(self, models: List[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
    def forward(self, x):
        outs = [m(x) for m in self.models]
        return torch.stack(outs, 0).mean(0)


def _mean_ensemble(models: List[torch.nn.Module]) -> _MeanEnsemble:
    return _MeanEnsemble(models)


def _plot_uncertainty(mean_field, std_field, mask, layout_id, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    mean_m = np.where(mask, np.nan, mean_field)
    std_m = np.where(mask, np.nan, std_field)
    im0 = axes[0].imshow(mean_m, origin="lower", cmap="viridis")
    axes[0].set_title(f"Ensemble mean prediction ({layout_id}) [m/s]")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(std_m, origin="lower", cmap="magma")
    axes[1].set_title("Ensemble std (predictive disagreement) [m/s]")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Uncertainty = per-pixel std across independently trained U-Nets (NOT calibrated)", fontsize=9, style="italic")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = ExperimentConfig(
        dataset_source="urbantales",
        checkpoint_dir="checkpoints/urbantales_experiment",
        output_dir="outputs/urbantales_experiment",
        manifest_path="configs/urbantales_experiment.json",
        epochs=35,
        batch_size=4,
        learning_rate=1e-3,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        patience=12,
        ensemble_size=3,
        device="cpu",
    )
    logging.info("Starting UrbanTALES DL surrogate experiment...")
    res = run_experiment(cfg)
    logging.info("Experiment completed successfully!")
    print("\n--- EXPERIMENT RESULTS SUMMARY ---")
    print("Provenance:", res.provenance)
    print("Models Evaluated:", list(res.models.keys()))
    print("Aggregate Metrics:")
    for m_name, m_data in res.metrics.items():
        if isinstance(m_data, dict) and "aggregate" in m_data:
            print(f"  {m_name}: {m_data['aggregate']}")

