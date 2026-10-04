"""Evaluation Script for Rule-Based Early Warning Baseline (Phase 7).

Evaluates the rule-based early warning baseline on the held-out test split:
1. Loads test partition from saved dataset
2. Applies causal rule-based detector
3. Evaluates all prediction horizons k in {1, 3, 5, 10, 20}
4. Computes early warning lead times across trajectories
5. Performs full rule ablation study (individual rules vs combined)
6. Saves machine-readable results without overwriting previous runs

Usage:
    python scripts/evaluate_rule_baseline.py --dataset-dir data/processed/agentguard_dataset_v1 --split test
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import json
from uuid import uuid4

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.storage import DatasetStorage, HAS_PYARROW
from ml.baselines.rule_based.config import RuleBaselineConfig, RuleWeights, RuleThresholds
from ml.baselines.rule_based.detector import RuleBasedEarlyWarningDetector
from ml.baselines.rule_based.evaluator import (
    evaluate_detector,
    evaluate_ablations,
)

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate rule-based early warning detector baseline."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/agentguard_dataset_v1",
        help="Path to dataset directory containing parquet splits and manifest.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate on (test strictly for final evaluation).",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,3,5,10,20",
        help="Comma-separated prediction horizons k.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Decision threshold cutoff for binary warnings.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/baselines/rule_based",
        help="Base directory for saving evaluation results.",
    )
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        default=True,
        help="Run rule ablation benchmarks.",
    )
    return parser.parse_args()


def format_table_row(cols, widths):
    return " | ".join(f"{str(c):<{w}}" for c, w in zip(cols, widths))


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_dir)
    split_name = args.split
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    print("=" * 80)
    print("AgentGuard: Rule-Based Early Warning Baseline Evaluation (Phase 7)")
    print("=" * 80)
    print(f"Dataset Path      : {dataset_path}")
    print(f"Evaluation Split  : {split_name}")
    print(f"Decision Threshold: {args.threshold}")
    print(f"Horizons (k)      : {horizons}")
    print("-" * 80)

    # 1. Load dataset split
    storage = DatasetStorage(dataset_path.parent)
    split_file = dataset_path / f"{split_name}.parquet"
    if not split_file.exists() and not split_file.with_suffix(".jsonl").exists():
        print(f"Error: Split file not found: {split_file}", file=sys.stderr)
        sys.exit(1)

    samples = storage.load_tabular(split_file)
    print(f"Loaded {len(samples)} samples from '{split_file.name}'")

    if not samples:
        print("Error: No samples found in split.", file=sys.stderr)
        sys.exit(1)

    # 2. Configure detector
    thresholds = RuleThresholds(decision_threshold=args.threshold)
    config = RuleBaselineConfig(
        thresholds=thresholds,
        prediction_horizons=horizons,
    )
    detector = RuleBasedEarlyWarningDetector(config=config)

    # 3. Generate predictions
    print("\nRunning causal rule evaluations across prediction points...")
    predictions = detector.predict_batch(samples)

    # 4. Evaluate across horizons and lead times
    eval_results = evaluate_detector(samples, detector, horizons=horizons)
    overall = eval_results["overall"]
    lead_time = eval_results["lead_time"]
    horizon_results = eval_results["horizons"]

    # 5. Display Horizon Results Table
    print("\n" + "=" * 80)
    print("CLASSIFICATION PERFORMANCE BY PREDICTION HORIZON (k)")
    print("=" * 80)
    headers = ["Horizon (k)", "Precision", "Recall", "F1 Score", "AUROC", "AUPRC", "FPR", "FAR", "TP/FP/TN/FN"]
    widths = [11, 9, 9, 9, 9, 9, 8, 8, 14]
    print(format_table_row(headers, widths))
    print("-" * 80)

    for k in horizons:
        if k in horizon_results:
            hr = horizon_results[k]
            cm = hr["confusion_matrix"]
            cm_str = f"{cm['tp']}/{cm['fp']}/{cm['tn']}/{cm['fn']}"
            row = [
                f"k={k}",
                f"{hr['precision']:.4f}",
                f"{hr['recall']:.4f}",
                f"{hr['f1']:.4f}",
                f"{hr['auroc']:.4f}",
                f"{hr['auprc']:.4f}",
                f"{hr['false_positive_rate']:.4f}",
                f"{hr['false_alarm_rate']:.4f}",
                cm_str,
            ]
            print(format_table_row(row, widths))

    print("-" * 80)
    cm_ov = overall["confusion_matrix"]
    cm_ov_str = f"{cm_ov['tp']}/{cm_ov['fp']}/{cm_ov['tn']}/{cm_ov['fn']}"
    ov_row = [
        "OVERALL",
        f"{overall['precision']:.4f}",
        f"{overall['recall']:.4f}",
        f"{overall['f1']:.4f}",
        f"{overall['auroc']:.4f}",
        f"{overall['auprc']:.4f}",
        f"{overall['false_positive_rate']:.4f}",
        f"{overall['false_alarm_rate']:.4f}",
        cm_ov_str,
    ]
    print(format_table_row(ov_row, widths))
    print("=" * 80)

    # 6. Display Early Warning Lead-Time Report
    print("\n" + "=" * 80)
    print("EARLY WARNING LEAD-TIME METRICS")
    print("=" * 80)
    print(f"Mean Lead Time                 : {lead_time['mean_lead_time']} s")
    print(f"Median Lead Time               : {lead_time['median_lead_time']} s")
    print(f"Min / Max Lead Time            : {lead_time['min_lead_time']} s / {lead_time['max_lead_time']} s")
    print(f"Successful Early Warnings      : {lead_time['successful_early_warnings']} / {lead_time['total_failed_trajectories']}")
    print(f"Early Warning Detection Rate   : {lead_time['early_warning_detection_rate'] * 100:.1f} %")
    print(f"Warnings Emitted per Trajectory: {lead_time['warnings_per_trajectory']}")
    print("=" * 80)

    # 7. Rule Ablation Study
    ablation_results = {}
    if args.run_ablations:
        print("\nRunning rule ablation benchmarks...")
        ablation_results = evaluate_ablations(samples, config, horizons=horizons)
        print("\n" + "=" * 80)
        print("RULE ABLATION SUMMARY (OVERALL F1 & AUROC)")
        print("=" * 80)
        abl_headers = ["Ablation Variant", "Precision", "Recall", "F1 Score", "AUROC", "AUPRC", "Mean Lead (s)"]
        abl_widths = [26, 10, 10, 10, 10, 10, 13]
        print(format_table_row(abl_headers, abl_widths))
        print("-" * 80)
        for rule_name, rep in ablation_results.items():
            ov = rep["overall"]
            lt = rep["lead_time"]
            abl_row = [
                rule_name,
                f"{ov['precision']:.4f}",
                f"{ov['recall']:.4f}",
                f"{ov['f1']:.4f}",
                f"{ov['auroc']:.4f}",
                f"{ov['auprc']:.4f}",
                f"{lt['mean_lead_time']:.4f}",
            ]
            print(format_table_row(abl_row, abl_widths))
        print("=" * 80)

    # 8. Persist machine-readable results
    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp_rule_baseline_{timestamp_slug}_{uuid4().hex[:6]}"
    exp_dir = Path(args.results_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # Save ablations
    with open(exp_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # Save configuration
    with open(exp_dir / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Save summary metadata
    summary = {
        "experiment_id": exp_id,
        "model_name": config.model_name,
        "dataset_path": str(dataset_path),
        "split": split_name,
        "sample_count": len(samples),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_f1": overall["f1"],
        "overall_auroc": overall["auroc"],
        "mean_lead_time": lead_time["mean_lead_time"],
        "early_warning_detection_rate": lead_time["early_warning_detection_rate"],
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save predictions
    flat_preds = []
    for p in predictions:
        row = {
            "sample_id": p["sample_id"],
            "run_id": p["run_id"],
            "step_idx": p["step_idx"],
            "timestamp": p["timestamp"],
            "prediction_horizon": p["prediction_horizon"],
            "ground_truth": p["ground_truth"],
            "risk_score": p["risk_score"],
            "prediction": p["prediction"],
            "warning_level": p["warning_level"],
        }
        for k, v in p["indicators"].items():
            row[f"ind_{k}"] = v
        flat_preds.append(row)

    if HAS_PYARROW:
        pred_table = pa.Table.from_pylist(flat_preds)
        pq.write_table(pred_table, exp_dir / "predictions.parquet")
    else:
        with open(exp_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for row in flat_preds:
                f.write(json.dumps(row) + "\n")

    print(f"\nMachine-readable results successfully persisted to:\n  {exp_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
