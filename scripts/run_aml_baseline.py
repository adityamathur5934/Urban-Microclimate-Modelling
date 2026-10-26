#!/usr/bin/env python3
"""CLI entry point for training and evaluating the Traditional ML Baseline.

Predicts outdoor area-averaged pedestrian wind speed from 26 geometric
and environmental condition features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.contract import LayoutFamily, UrbanLayoutSample
from src.data.loader import LayoutDataLoader
from src.models.baseline_rf import BaselineConfig, run_baseline_experiment
from tests.fixtures.synthetic_data import create_synthetic_sample


def generate_synthetic_cohort(num_samples: int = 30) -> list[UrbanLayoutSample]:
    """Generates a reproducible synthetic layout cohort across diverse families."""
    samples = []
    families = [
        LayoutFamily.IDEALIZED_ALIGNED,
        LayoutFamily.IDEALIZED_STAGGERED,
        LayoutFamily.REALISTIC_NEIGHBORHOOD,
    ]
    speeds = [2.0, 3.5, 5.0, 7.5, 10.0]
    heights = [10.0, 15.0, 25.0, 35.0]

    idx = 0
    while len(samples) < num_samples:
        fam = families[idx % len(families)]
        speed = speeds[idx % len(speeds)]
        height = heights[idx % len(heights)]
        sid = f"synthetic_{fam.value}_{int(speed)}mps_h{int(height)}m_{idx:03d}"
        sample = create_synthetic_sample(
            layout_id=sid,
            layout_family=fam,
            grid_shape=(32, 32),
            inflow_speed=speed,
            building_height=height,
        )
        samples.append(sample)
        idx += 1

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate traditional ML baseline (Random Forest) for urban microclimate modelling."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file for BaselineConfig (e.g., configs/baseline_rf.json).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing verified UrbanLayoutSample NPZ files.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Run software smoke test using synthetic fixtures.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of synthetic samples to generate if --synthetic is specified (default: 30).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest regressor (default: 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=12,
        help="Maximum depth of decision trees (default: 12).",
    )
    parser.add_argument(
        "--split-method",
        choices=["ratio", "family"],
        default="ratio",
        help="Layout partitioning strategy: 'ratio' (random layout holdout) or 'family' (unseen family holdout).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of unique layouts allocated to training split (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of unique layouts allocated to validation split (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of unique layouts allocated to held-out test split (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data partitioning and Random Forest initialization (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline",
        help="Directory where trained model, split assignments, and metrics report are persisted (default: outputs/baseline).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("URBAN MICROCLIMATE MODELLING — TRADITIONAL ML BASELINE (RANDOM FOREST)")
    print("=" * 70)

    # Load or generate samples
    samples: list[UrbanLayoutSample] = []
    if args.data_dir:
        data_path = Path(args.data_dir)
        if data_path.exists() and data_path.is_dir():
            npz_files = list(data_path.glob("*.npz"))
            print(f"Found {len(npz_files)} NPZ archive(s) in {data_path}")
            for npz in npz_files:
                is_valid, issues = LayoutDataLoader.validate_npz_archive(npz)
                if is_valid:
                    sample = LayoutDataLoader.load_sample_npz(npz)
                    samples.append(sample)
                else:
                    print(f"  [WARN] Skipping invalid archive {npz.name}: {issues}")

    if not samples:
        print("[NOTICE] No valid local CFD datasets found.")
        print("         Executing pipeline with synthetic layout fixtures.")
        print("         >>> WARNING: SOFTWARE SMOKE TEST ONLY — NOT SCIENTIFIC PERFORMANCE <<<")
        samples = generate_synthetic_cohort(args.num_samples)
        print(f"Generated {len(samples)} synthetic layout samples across multiple families.")

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        config = BaselineConfig.from_dict(cfg_dict)
    else:
        config = BaselineConfig(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            split_method=args.split_method,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.seed,
            output_dir=args.output_dir,
        )

    print(f"\nConfiguration:")
    print(f"  Random Forest Trees : {config.n_estimators} (max_depth={config.max_depth})")
    print(f"  Split Strategy      : {config.split_method} (train={config.train_ratio}, val={config.val_ratio}, test={config.test_ratio})")
    print(f"  Random Seed         : {config.random_state}")
    print(f"  Output Directory    : {config.output_dir}")

    results = run_baseline_experiment(samples, config=config)

    print("\n" + "-" * 70)
    print("EXPERIMENT RESULTS")
    print("-" * 70)
    print(f"Data Provenance : {results['data_provenance']}")
    print(f"Dataset Splits  : Train={results['split_summary']['train_samples']} | "
          f"Val={results['split_summary']['val_samples']} | "
          f"Test={results['split_summary']['test_samples']}")

    print("\nModel Performance Comparison:")
    print(f"{'Split':<8} | {'Model':<16} | {'MAE [m/s]':<10} | {'RMSE [m/s]':<10} | {'R2':<8} | {'NRMSE(range)':<14} | {'NRMSE(mean)':<12}")
    print("-" * 90)

    for split_name in ["train", "val", "test"]:
        if split_name in results["rf_metrics"]:
            rf_m = results["rf_metrics"][split_name]
            nv_m = results["naive_metrics"][split_name]
            print(f"{split_name:<8} | {'Random Forest':<16} | {rf_m['mae']:<10.4f} | {rf_m['rmse']:<10.4f} | {rf_m['r2']:<8.4f} | {rf_m['nrmse_range']:<13.2f}% | {rf_m['nrmse_mean']:<11.2f}%")
            print(f"{'':<8} | {'Naive Baseline':<16} | {nv_m['mae']:<10.4f} | {nv_m['rmse']:<10.4f} | {nv_m['r2']:<8.4f} | {nv_m['nrmse_range']:<13.2f}% | {nv_m['nrmse_mean']:<11.2f}%")
            print("-" * 90)

    if results.get("improvement_over_naive_test"):
        imp = results["improvement_over_naive_test"]
        print(f"\nTest-Set Improvement over Naive Mean:")
        print(f"  MAE Reduction  : {imp['mae_reduction_percent']}%")
        print(f"  RMSE Reduction : {imp['rmse_reduction_percent']}%")
        print(f"  R^2 Delta      : +{imp['r2_delta']}")

    print(f"\nTop-5 Feature Importances:")
    for feat, imp_val in results["top_5_features"].items():
        print(f"  - {feat:<35}: {imp_val * 100:.2f}%")

    print(f"\nPersisted Artifacts:")
    print(f"  Model  : {results['artifacts']['model_path']}")
    print(f"  Splits : {results['artifacts']['split_path']}")
    print(f"  Report : {args.output_dir}/baseline_metrics.json")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
