"""Automated Research Evaluation Pipeline - Phase 12 (Section 25).

Executes end-to-end evaluation:
1. Discover compatible experiment results across all 5 paradigm families.
2. Validate schemas and test population compatibility.
3. Calculate prediction-level and failure-level metrics.
4. Compute lead-time distributions and early warning coverage.
5. Generate ROC and PR curves.
6. Generate calibration and reliability analysis.
7. Generate trajectory-level block bootstrap uncertainty estimates.
8. Perform paired difference statistical tests.
9. Execute subgroup analysis across failure levels, topologies, and tasks.
10. Extract representative error cases (FP and FN).
11. Save machine-readable Tables A through I.
12. Generate publication-quality plots.
13. Produce comprehensive research evaluation report.
"""

import sys
import os
import argparse
import time

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.evaluation.engine import ComprehensiveEvaluationEngine


def main():
    parser = argparse.ArgumentParser(description="Run Phase 12 Comprehensive Research Evaluation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Target base directory for all evaluation artifacts",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/agentguard_dataset_v1",
        help="Path to processed dataset directory containing test.parquet",
    )
    parser.add_argument(
        "--baselines-dir",
        type=str,
        default="results/baselines",
        help="Path to baseline experiment results",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("AgentGuard Phase 12: Comprehensive Research Evaluation Framework")
    print("=" * 80)
    start_time = time.time()

    engine = ComprehensiveEvaluationEngine(
        output_base_dir=args.output_dir,
        dataset_path=args.dataset_dir,
        baselines_dir=args.baselines_dir,
    )

    print(f"[1/5] Loading ground truth dataset from {args.dataset_dir}...")
    engine.load_ground_truth()
    print(f"      Loaded {len(engine.ground_truth_test_rows)} test split records.")

    print(f"[2/5] Discovering baseline experiments in {args.baselines_dir}...")
    inv = engine.discover_experiments()
    print("      Found experiments across:")
    print(f"      - Rule-Based: {len(inv.get('Rule-Based', []))} experiment(s)")
    print(f"      - Classical ML: {list(inv.get('Classical ML', {}).keys())}")
    print(f"      - Temporal Sequence: {list(inv.get('Temporal Sequence', {}).keys())}")
    print(f"      - Static GNN: {list(inv.get('Static GNN', {}).keys())}")
    print(f"      - Temporal GNN: {len(inv.get('Temporal GNN', {}))} horizon checkpoint(s)")

    print("[3/5] Evaluating all models, horizons, calibrations, lead times, and subgroups...")
    engine.evaluate_all()
    print(f"      Computed {len(engine.unified_records)} unified evaluation records.")
    print(f"      Conducted {len(engine.pairwise_records)} pairwise hypothesis tests.")
    print(f"      Evaluated {len(engine.subgroup_records)} stratified subgroup slices.")

    print(f"[4/5] Persisting machine-readable artifacts, tables, and publication plots to {args.output_dir}...")
    engine.save_all_artifacts()

    elapsed = time.time() - start_time
    print(f"[5/5] Comprehensive evaluation complete in {elapsed:.2f} seconds.")
    print("=" * 80)
    print(f"Evaluation report: {os.path.join(args.output_dir, 'research_evaluation_report.md')}")
    print(f"Plots directory:   {os.path.join(args.output_dir, 'plots')}")
    print(f"Tables directory:  {os.path.join(args.output_dir, 'reports')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
