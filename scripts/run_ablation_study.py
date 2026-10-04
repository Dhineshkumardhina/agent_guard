"""Automated Ablation Study Pipeline - Phase 13 (Section 12).

Executes end-to-end ablation experiments:
1. Loads dataset configuration and verified test split.
2. Selectively instantiates ablated Temporal GNN models.
3. Applies feature/structural masks without dataset corruption.
4. Trains models with validation threshold calibration.
5. Evaluates test performance and incident early warnings.
6. Calculates paired trajectory bootstrap difference tests against the full model.
7. Generates machine-readable tables, matrix, and metrics.
8. Produces all 8 publication-grade visualization plots.
9. Compiles research ablation report.

Supports CLI options:
  --ablation (default: all)
  --horizon (default: all [1, 3, 5, 10, 20])
  --seed (default: [42, 123, 456, 789, 2026])
  --epochs (default: 10)
  --dataset-dir
  --output-dir
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.ablation.runner import AblationStudyRunner
from ml.ablation.masking import ABLATION_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Run Phase 13 Research Ablation Study")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/agentguard_dataset_v1",
        help="Path to processed dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation",
        help="Target base directory for ablation outputs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="temporal_gnn",
        help="Model architecture for ablation study (default: temporal_gnn)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        nargs="+",
        default=None,
        help="Specific ablation keys to run (default: all registered ablations)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        nargs="+",
        default=None,
        help="Specific horizons K to evaluate (default: [1, 3, 5, 10, 20])",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds to evaluate (default: [42, 123, 456, 789, 2026])",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs per ablation run (default: 10)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("AgentGuard Phase 13: Research Ablation Study Framework")
    print("=" * 80)
    start_time = time.time()

    runner = AblationStudyRunner(
        dataset_dir=args.dataset_dir,
        base_results_dir=args.output_dir,
    )

    runner.load_dataset()
    print(f"[Dataset] Train samples: {len(runner.train_samples)} | Val: {len(runner.val_samples)} | Test: {len(runner.test_samples)}")

    ablations = args.ablation or list(ABLATION_REGISTRY.keys())
    horizons = args.horizon or [1, 3, 5, 10, 20]
    seeds = args.seed or [42, 123, 456, 789, 2026]

    print(f"[Configuration] Ablations ({len(ablations)}): {ablations}")
    print(f"[Configuration] Horizons: {horizons}")
    print(f"[Configuration] Seeds: {seeds}")
    print(f"[Configuration] Epochs: {args.epochs}")
    print("-" * 80)

    runner.run_study(
        ablations=ablations,
        horizons=horizons,
        seeds=seeds,
        epochs=args.epochs,
    )

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Phase 13 Ablation Study completed in {elapsed:.2f} seconds.")
    print(f"Results report: {os.path.join(args.output_dir, 'ablation_report.md')}")
    print(f"Tables:         {os.path.join(args.output_dir, 'tables')}")
    print(f"Plots:          {os.path.join(args.output_dir, 'plots')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
