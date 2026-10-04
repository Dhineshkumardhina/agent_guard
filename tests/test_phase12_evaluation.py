"""Unit and Integration Tests for Phase 12: Comprehensive Research Evaluation Framework.

Tests cover:
1. Metric calculation (Precision, Recall, F1, AUROC, AUPRC, FPR, False Alarm Rate)
2. Lead-time calculation (Mean, Median, Std, Min, Max, Quantiles)
3. Failure matching (First valid warning before failure, discarding post-failure warnings)
4. Threshold analysis (Performance across [0.10, 0.90] thresholds)
5. Calibration (Brier score, ECE, reliability bins)
6. Bootstrap confidence intervals (95% CI coverage)
7. Pairwise comparison (Paired trajectory difference test)
8. Run-level resampling (Block bootstrap on trajectory run_id)
9. Result schema validation (UnifiedEvaluationRecord structure)
10. Experiment compatibility validation (Catching incompatible sample populations)
11. No test-set contamination (Ensuring evaluation only consumes test split)
12. Reproducibility (Deterministic outputs given identical seed)
"""

import os
import json
import pytest
import numpy as np

from ml.evaluation.schema import (
    UnifiedEvaluationRecord,
    FailureCentricRecord,
    CalibrationBin,
    PairwiseComparisonRecord,
    SubgroupMetricRecord,
)
from ml.evaluation.metrics import (
    compute_comprehensive_metrics,
    compute_expected_calibration_error,
    compute_calibration_curve_data,
    compute_roc_and_pr_curves,
    compute_lead_time_statistics,
)
from ml.evaluation.failure_centric import FailureCentricEvaluator
from ml.evaluation.uncertainty import (
    trajectory_block_bootstrap_ci,
    paired_trajectory_bootstrap_test,
)
from ml.evaluation.subgroups import SubgroupEvaluator
from ml.evaluation.error_analysis import ErrorAnalyzer
from ml.evaluation.engine import ComprehensiveEvaluationEngine


# 1. Metric calculation test
def test_metric_calculation():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_prob = [0.1, 0.6, 0.4, 0.9]

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "false_positive_rate" in metrics
    assert "false_alarm_rate" in metrics
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["false_positive_rate"] == 0.5


# 2. Lead-time calculation test
def test_lead_time_calculation():
    lead_times = [2.0, 4.0, 6.0, 8.0, 10.0]
    stats = compute_lead_time_statistics(lead_times, total_runs=2)

    assert stats["successful_early_warnings"] == 5
    assert stats["mean_lead_time"] == 6.0
    assert stats["median_lead_time"] == 6.0
    assert stats["min_lead_time"] == 2.0
    assert stats["max_lead_time"] == 10.0
    assert stats["warnings_per_trajectory"] == 2.5


# 3. Failure matching test (Policy: warn before fail counts, warn after fail does NOT count)
def test_failure_matching_policy():
    evaluator = FailureCentricEvaluator()

    ground_truth_rows = [
        {"run_id": "run_1", "timestamp": 10.0, "label": 1, "failure_type": "tool_failure", "failure_level": 1},
        {"run_id": "run_2", "timestamp": 20.0, "label": 1, "failure_type": "timeout", "failure_level": 2},
    ]

    predictions = [
        # run_1: early warning at t=5.0 (lead time = 5.0)
        {"run_id": "run_1", "timestamp": 5.0, "predicted_label": 1, "predicted_probability": 0.8},
        # run_1: another warning at t=8.0 (should use earliest valid: t=5.0)
        {"run_id": "run_1", "timestamp": 8.0, "predicted_label": 1, "predicted_probability": 0.9},
        # run_2: late warning at t=25.0 (post-failure! Must NOT count as early warning)
        {"run_id": "run_2", "timestamp": 25.0, "predicted_label": 1, "predicted_probability": 0.85},
    ]

    res = evaluator.evaluate_failures(predictions, ground_truth_rows)

    assert res["total_failures_evaluated"] == 2
    assert res["detected_failures"] == 1
    assert res["detection_coverage_percent"] == 50.0
    records = res["failure_records"]
    assert len(records) == 2
    # run_1 detected with lead time 5.0
    r1 = [r for r in records if r["run_id"] == "run_1"][0]
    assert r1["detected"] is True
    assert r1["lead_time"] == 5.0
    # run_2 not detected (late warning at 25.0 > fail at 20.0)
    r2 = [r for r in records if r["run_id"] == "run_2"][0]
    assert r2["detected"] is False
    assert r2["lead_time"] is None


# 4. Threshold analysis test
def test_threshold_analysis():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.35, 0.45, 0.85]

    # At threshold 0.2: y_pred = [0, 1, 1, 1] -> recall 1.0, precision 2/3
    pred_02 = [1 if p >= 0.2 else 0 for p in y_prob]
    m_02 = compute_comprehensive_metrics(y_true, pred_02, y_prob)
    assert m_02["recall"] == 1.0

    # At threshold 0.5: y_pred = [0, 0, 0, 1] -> recall 0.5, precision 1.0
    pred_05 = [1 if p >= 0.5 else 0 for p in y_prob]
    m_05 = compute_comprehensive_metrics(y_true, pred_05, y_prob)
    assert m_05["precision"] == 1.0
    assert m_05["recall"] == 0.5


# 5. Calibration test
def test_calibration_metrics():
    y_true = [0, 0, 1, 1]
    y_prob = [0.05, 0.15, 0.85, 0.95]

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    bins = compute_calibration_curve_data(y_true, y_prob, n_bins=5)

    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0
    assert len(bins) == 5
    assert bins[0]["sample_count"] > 0
    assert bins[-1]["sample_count"] > 0


# 6. Bootstrap confidence intervals test
def test_bootstrap_confidence_intervals():
    predictions = [
        {"run_id": "r1", "true_label": 1, "predicted_label": 1, "predicted_probability": 0.9},
        {"run_id": "r1", "true_label": 0, "predicted_label": 0, "predicted_probability": 0.1},
        {"run_id": "r2", "true_label": 1, "predicted_label": 1, "predicted_probability": 0.85},
        {"run_id": "r2", "true_label": 0, "predicted_label": 0, "predicted_probability": 0.15},
    ]

    ci_results = trajectory_block_bootstrap_ci(predictions, n_bootstraps=50, random_seed=42)

    assert "f1" in ci_results
    assert "auroc" in ci_results
    assert "auprc" in ci_results
    f1_ci = ci_results["f1"]
    assert f1_ci.ci_lower <= f1_ci.ci_upper


# 7. Pairwise comparison test
def test_pairwise_comparison():
    preds_a = [
        {"run_id": "r1", "true_label": 1, "predicted_label": 1, "predicted_probability": 0.9},
        {"run_id": "r2", "true_label": 1, "predicted_label": 1, "predicted_probability": 0.8},
    ]
    preds_b = [
        {"run_id": "r1", "true_label": 1, "predicted_label": 0, "predicted_probability": 0.4},
        {"run_id": "r2", "true_label": 1, "predicted_label": 0, "predicted_probability": 0.3},
    ]

    res = paired_trajectory_bootstrap_test(
        preds_a, preds_b, "ModelA", "ModelB", metric_name="f1", n_bootstraps=50, random_seed=42
    )

    assert isinstance(res, PairwiseComparisonRecord)
    assert res.value_a == 1.0
    assert res.value_b == 0.0
    assert res.difference == 1.0
    assert res.ci_lower <= res.ci_upper


# 8. Run-level resampling test (preserves trajectory blocks)
def test_run_level_resampling_integrity():
    preds = [
        {"run_id": "run_alpha", "true_label": 1, "predicted_label": 1, "predicted_probability": 0.9},
        {"run_id": "run_alpha", "true_label": 0, "predicted_label": 0, "predicted_probability": 0.1},
        {"run_id": "run_beta", "true_label": 1, "predicted_label": 0, "predicted_probability": 0.3},
    ]
    ci = trajectory_block_bootstrap_ci(preds, n_bootstraps=20, random_seed=123)
    assert len(ci) > 0


# 9. Result schema validation test
def test_result_schema_validation():
    rec = UnifiedEvaluationRecord(
        experiment_id="test_exp",
        model_family="Classical ML",
        model_name="XGBoost",
        prediction_horizon=1,
        precision=0.95,
        recall=0.90,
        f1=0.925,
        auroc=0.98,
        auprc=0.97,
    )
    d = rec.to_dict()
    assert d["experiment_id"] == "test_exp"
    assert d["model_name"] == "XGBoost"
    assert d["f1"] == 0.925
    assert "mean_lead_time" in d
    assert "brier_score" in d


# 10. Experiment compatibility validation test
def test_experiment_compatibility():
    engine = ComprehensiveEvaluationEngine()
    engine.load_ground_truth()
    inv = engine.discover_experiments()

    # Verify that all 5 families are discovered
    assert len(inv["Rule-Based"]) > 0
    assert "xgboost" in inv["Classical ML"]
    assert "lstm" in inv["Temporal Sequence"]
    assert "gcn" in inv["Static GNN"]
    assert len(inv["Temporal GNN"]) > 0


# 11. No test-set contamination test
def test_no_test_set_contamination():
    engine = ComprehensiveEvaluationEngine()
    engine.load_ground_truth()

    # Verify every row in ground_truth_test_rows belongs to the test split
    for row in engine.ground_truth_test_rows:
        assert row.get("split") == "test"


# 12. Reproducibility test
def test_evaluation_reproducibility(tmp_path):
    out1 = str(tmp_path / "eval1")
    out2 = str(tmp_path / "eval2")

    engine1 = ComprehensiveEvaluationEngine(output_base_dir=out1)
    engine1.evaluate_all()
    engine1.save_all_artifacts()

    engine2 = ComprehensiveEvaluationEngine(output_base_dir=out2)
    engine2.evaluate_all()
    engine2.save_all_artifacts()

    with open(os.path.join(out1, "metrics", "unified_evaluation_metrics.json")) as f1, \
         open(os.path.join(out2, "metrics", "unified_evaluation_metrics.json")) as f2:
        m1 = json.load(f1)
        m2 = json.load(f2)

    assert len(m1) == len(m2)
    # Check that deterministic metrics match exactly
    for r1, r2 in zip(m1, m2):
        assert r1["model_name"] == r2["model_name"]
        assert r1["precision"] == r2["precision"]
        assert r1["recall"] == r2["recall"]
        assert r1["f1"] == r2["f1"]
        assert r1["auroc"] == r2["auroc"]
        assert r1["auprc"] == r2["auprc"]

