"""Automated Generalization and Robustness Evaluation CLI - Phase 14 (Section 16).

Executes end-to-end out-of-distribution transfer experiments:
1. Loads dataset and graph sequences.
2. Selects generalization experiments (G1-G5, G3b, G4b).
3. Partitions runs with strict run-level isolation.
4. Trains models and freezes thresholds on in-distribution validation split.
5. Evaluates on both In-Distribution and Out-of-Distribution test sets.
6. Computes Generalization Gaps and trajectory block bootstrap 95% CIs.
7. Produces publication-grade visualization plots under results/generalization/plots/.
8. Saves summary tables and research report.

Usage:
    python scripts/run_generalization_experiments.py
    python scripts/run_generalization_experiments.py --dimension agent_count topology
    python scripts/run_generalization_experiments.py --model temporal_gnn logistic_regression --horizon 1 3 5
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.generalization.runner import GeneralizationExperimentRunner
from ml.generalization.splits import GENERALIZATION_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Run Phase 14 Generalization and Robustness Evaluation")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/agentguard_generalization_v1",
        help="Path to processed dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/generalization",
        help="Target base directory for generalization outputs",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiment IDs to execute (e.g. G1, G2, G3, G4, G5). Default: all.",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        nargs="+",
        default=None,
        help="Filter experiments by dimension: agent_count, topology, task, failure_type.",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        help="Model families to evaluate: temporal_gnn, logistic_regression, random_forest, xgboost, lstm, gru.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        nargs="+",
        default=None,
        help="Forecasting horizons K: 1, 3, 5, 10, 20. Default: [1, 3, 5, 10].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds to evaluate: 42, 123, 456. Default: [42, 123, 456].",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of training epochs per neural network run (default: 6).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("AgentGuard Phase 14: Generalization and Robustness Evaluation")
    print("=" * 80)
    start_time = time.time()

    runner = GeneralizationExperimentRunner(
        dataset_dir=args.dataset_dir,
        base_results_dir=args.output_dir,
    )
    runner.load_dataset()
    print(f"[Dataset] Source: {runner.dataset_dir} | Total Samples: {len(runner.all_samples)} | Total Graphs: {len(runner.all_graphs)}")

    # Resolve experiments
    exp_keys = args.experiment or list(GENERALIZATION_REGISTRY.keys())
    if args.dimension:
        dims = set(args.dimension)
        exp_keys = [k for k in exp_keys if GENERALIZATION_REGISTRY[k].dimension in dims]

    models = args.model or ["temporal_gnn", "logistic_regression", "random_forest", "xgboost"]
    horizons = args.horizon or [1, 3, 5, 10]
    seeds = args.seed or [42, 123, 456]

    print(f"[Config] Experiments ({len(exp_keys)}): {exp_keys}")
    print(f"[Config] Model Families: {models}")
    print(f"[Config] Horizons: {horizons}")
    print(f"[Config] Seeds: {seeds}")
    print(f"[Config] Epochs: {args.epochs}")
    print("-" * 80)

    runner.run_all_experiments(
        experiments=exp_keys,
        models=models,
        horizons=horizons,
        seeds=seeds,
        epochs=args.epochs,
    )

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Phase 14 Generalization Study completed in {elapsed:.2f} seconds.")
    print(f"Report: {os.path.join(args.output_dir, 'generalization_report.md')}")
    print(f"Tables: {os.path.join(args.output_dir, 'tables')}")
    print(f"Plots:  {os.path.join(args.output_dir, 'plots')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
