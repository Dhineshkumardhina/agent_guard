"""Training and Evaluation CLI Script for Classical Machine Learning Baselines (Phase 8).

Trains and evaluates:
1. Logistic Regression
2. Random Forest
3. XGBoost
across prediction horizons K in {1, 3, 5, 10, 20}.

Usage:
    python scripts/train_classical_baselines.py --model all --horizon all --seed 42
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.storage import DatasetStorage
from ml.baselines.classical_ml.experiment import ClassicalMLExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and benchmark classical ML baselines (LR, RF, XGB) for cascading failure prediction."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/agentguard_dataset_v1",
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to train: 'logistic_regression', 'random_forest', 'xgboost', or 'all'.",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="all",
        help="Horizon to evaluate: '1', '3', '5', '10', '20', or 'all'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/baselines/classical_ml",
        help="Directory to persist machine-readable baseline results.",
    )
    return parser.parse_args()


def format_table_row(cols, widths):
    return " | ".join(f"{str(c):<{w}}" for c, w in zip(cols, widths))


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)

    # Resolve models
    if args.model.lower().strip() == "all":
        models_to_run = ["logistic_regression", "random_forest", "xgboost"]
    else:
        models_to_run = [m.strip() for m in args.model.split(",") if m.strip()]

    # Resolve horizons
    if args.horizon.lower().strip() == "all":
        horizons_to_eval = [1, 3, 5, 10, 20]
    else:
        horizons_to_eval = [int(h.strip()) for h in args.horizon.split(",") if h.strip()]

    print("=" * 85)
    print("AgentGuard: Classical Machine Learning Baselines (Phase 8)")
    print("=" * 85)
    print(f"Dataset Path : {dataset_path}")
    print(f"Models       : {models_to_run}")
    print(f"Horizons (K) : {horizons_to_eval}")
    print(f"Random Seed  : {args.seed}")
    print("-" * 85)

    storage = DatasetStorage(dataset_path.parent)
    train_file = dataset_path / "train.parquet"
    test_file = dataset_path / "test.parquet"

    if not train_file.exists() and not train_file.with_suffix(".jsonl").exists():
        print(f"Error: Training split not found at {train_file}", file=sys.stderr)
        sys.exit(1)
    if not test_file.exists() and not test_file.with_suffix(".jsonl").exists():
        print(f"Error: Testing split not found at {test_file}", file=sys.stderr)
        sys.exit(1)

    train_samples = storage.load_tabular(train_file)
    test_samples = storage.load_tabular(test_file)

    manifest_obj = storage.load_manifest(dataset_path) if (dataset_path / "manifest.json").exists() else None
    dataset_version = manifest_obj.dataset_version if manifest_obj else dataset_path.name

    print(f"Loaded {len(train_samples)} training samples, {len(test_samples)} testing samples.")

    runner = ClassicalMLExperimentRunner(
        base_results_dir=args.results_dir,
        random_seed=args.seed,
    )

    print("\nTraining and evaluating models across horizons...")
    experiment_results = runner.run_experiment(
        train_samples=train_samples,
        test_samples=test_samples,
        models_to_run=models_to_run,
        horizons=horizons_to_eval,
        dataset_version=dataset_version,
    )

    # Print results table
    print("\n" + "=" * 85)
    print("CLASSICAL ML BASELINE PERFORMANCE COMPARISON TABLE")
    print("=" * 85)
    headers = ["Model", "Horizon (k)", "Precision", "Recall", "F1 Score", "AUROC", "AUPRC", "FPR", "Lead Time (s)"]
    widths = [20, 11, 9, 9, 9, 9, 9, 8, 13]
    print(format_table_row(headers, widths))
    print("-" * 85)

    for model_name, h_dict in experiment_results["model_results"].items():
        for k, res in h_dict.items():
            m = res.get("metrics", {})
            lt = res.get("lead_time", {})
            row = [
                model_name,
                f"k={k}",
                f"{m.get('precision', 0.0):.4f}",
                f"{m.get('recall', 0.0):.4f}",
                f"{m.get('f1', 0.0):.4f}",
                f"{m.get('auroc', 0.0):.4f}",
                f"{m.get('auprc', 0.0):.4f}",
                f"{m.get('false_positive_rate', 0.0):.4f}",
                f"{lt.get('mean_lead_time', 0.0):.4f}",
            ]
            print(format_table_row(row, widths))
        print("-" * 85)

    print("\nExperiment completed successfully!")
    print(f"Results persisted under: {runner.base_results_dir}")
    print("=" * 85)


if __name__ == "__main__":
    main()
