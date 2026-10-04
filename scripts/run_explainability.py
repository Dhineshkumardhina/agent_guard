"""CLI Automation Script for AgentGuard Phase 15 Explainability Studies.

Usage:
    python scripts/run_explainability.py
    python scripts/run_explainability.py --model temporal_gnn --horizon 1 --seed 42
    python scripts/run_explainability.py --run-id run_0000_agentguard_generalization_v1 --horizon 1
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.explainability.runner import ExplainabilityRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentGuard Phase 15: Explainability and Failure-Risk Interpretation")
    parser.add_argument("--dataset-dir", type=str, default="data/processed/agentguard_generalization_v1", help="Path to processed dataset")
    parser.add_argument("--output-dir", type=str, default="results/explainability", help="Output directory for reports, plots, and tables")
    parser.add_argument("--model", type=str, default="temporal_gnn", choices=["temporal_gnn", "random_forest", "logistic_regression", "xgboost", "all"], help="Target model family")
    parser.add_argument("--run-id", type=str, default=None, help="Target simulation run ID for case explanation")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 3, 5, 10, 20], help="Prediction horizon K")
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "permutation", "masking"], help="Attribution method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs for Temporal GNN")

    args = parser.parse_args()

    print("=" * 80)
    print("AgentGuard Phase 15: Explainability and Failure-Risk Interpretation Framework")
    print("=" * 80)
    print(f"[Config] Dataset:  {args.dataset_dir}")
    print(f"[Config] Output:   {args.output_dir}")
    print(f"[Config] Model:    {args.model}")
    print(f"[Config] Horizon:  K={args.horizon}")
    print(f"[Config] Seed:     {args.seed}")
    print(f"[Config] Epochs:   {args.epochs}")
    print("-" * 80)

    t0 = time.time()
    runner = ExplainabilityRunner(
        dataset_dir=Path(args.dataset_dir),
        results_dir=Path(args.output_dir),
    )

    # 1. Load data
    print("[1/6] Loading dataset and temporal graph sequences...")
    runner.load_dataset()
    print(f"      Loaded {len(runner.all_samples)} samples and {len(runner.all_graphs)} graph snapshots.")

    # 2. Train models
    print("[2/6] Training / Initializing Temporal GNN and classical baselines...")
    runner.train_models(horizon=args.horizon, seed=args.seed, epochs=args.epochs)
    print(f"      Calibrated operating threshold on validation split: theta* = {runner.threshold:.2f}")

    # 3. Global importance reports
    print("[3/6] Generating Level 1-5 Global Importance Rankings...")
    runner.generate_global_importance(horizon=args.horizon, seed=args.seed)
    print(f"      Global feature, agent role, and communication channel rankings computed.")

    # 4. Case studies & perturbations
    print("[4/6] Selecting representative case study archetypes & running perturbation analysis...")
    runner.select_case_studies(horizon=args.horizon, seed=args.seed)
    print(f"      Selected and explained {len(runner.case_studies)} case studies with counterfactual sensitivity analysis.")

    # 5. Stability evaluation
    print("[5/6] Evaluating explanation consistency across seeds and horizons...")
    runner.evaluate_stability(seeds=[42, 123, 456], horizons=[1, 3, 5])
    print(f"      Stability evaluated: Spearman rho = {runner.stability_report.feature_rank_correlation_seeds:.3f}, Top-5 Jaccard = {runner.stability_report.feature_top_k_jaccard_seeds:.3f}")

    # 6. Save artifacts, plots, and report
    print("[6/6] Generating publication plots, dashboard JSON data, and final markdown report...")
    runner.save_artifacts()

    elapsed = time.time() - t0
    print("=" * 80)
    print(f"Phase 15 Explainability Framework completed in {elapsed:.2f} seconds.")
    print(f"Report: {runner.results_dir / 'explainability_report.md'}")
    print(f"Plots:  {runner.dirs['plots']}")
    print(f"Tables: {runner.dirs['tables']}")
    print(f"Dashboard Data: {runner.dirs['dashboard']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
