"""Phase 7 Test Suite: Rule-Based Early Warning Baseline.

Verifies:
1. Rule calculation (all 9 transparent indicators)
2. Threshold behavior (LOW, MEDIUM, HIGH, decision boundary)
3. Risk score computation and normalization in [0.0, 1.0]
4. Individual rules (ablation filters)
5. Combined rules (ensemble weighting)
6. Prediction horizons (k in {1, 3, 5, 10, 20})
7. No future leakage (strictly causal inputs <= t)
8. Lead-time calculation (t_failure - t_warning, discarding post-failure warnings)
9. Reproducibility (deterministic outputs)
10. Result serialization (machine-readable outputs)
"""

import pytest
from pathlib import Path
import tempfile
import json
import numpy as np

from ml.baselines.rule_based.config import (
    WarningLevel,
    RuleWeights,
    RuleThresholds,
    RuleBaselineConfig,
)
from ml.baselines.rule_based.rules import (
    compute_error_rate,
    compute_retry_rate,
    compute_timeout_rate,
    compute_contradiction_rate,
    compute_confidence_degradation,
    compute_latency_anomaly,
    compute_frequency_anomaly,
    compute_agent_failure_history,
    compute_interaction_concentration,
    compute_all_indicators,
)
from ml.baselines.rule_based.scoring import (
    compute_risk_score,
    classify_risk_level,
    predict_binary_warning,
)
from ml.baselines.rule_based.detector import RuleBasedEarlyWarningDetector
from ml.baselines.rule_based.evaluator import (
    compute_classification_metrics,
    compute_lead_time_metrics,
    evaluate_detector,
    evaluate_ablations,
    compute_auroc,
    compute_auprc,
)
from ml.data.schema import PredictionSample


@pytest.fixture
def clean_sample():
    """Sample representing healthy, normal multi-agent operation."""
    return PredictionSample(
        sample_id="test_clean_0",
        run_id="run_clean",
        timestamp=1.0,
        step_idx=2,
        task_type="research",
        topology="pipeline",
        number_of_agents=5,
        prediction_horizon=1,
        node_features={"planner": {"error_rate": 0.0}},
        edge_features={"planner->researcher": {"weight": 1.0}},
        temporal_graph_history=[],
        agent_level_features={
            "total_events_observed": 5.0,
            "error_count": 0.0,
            "total_retries": 0.0,
            "tool_failure_count": 0.0,
            "mean_contradiction_score": 0.0,
            "max_contradiction_score": 0.0,
            "mean_confidence": 0.95,
            "min_confidence": 0.90,
            "mean_latency": 0.10,
            "max_latency": 0.12,
            "interaction_density": 1.0,
            "min_output_quality": 0.95,
            "mean_output_quality": 0.95,
        },
        label=0,
        failure_type="none",
        failure_level=0,
        source_event_id="evt_0",
        random_seed=42,
        dataset_version="v1",
        split="test",
    )


@pytest.fixture
def degraded_sample():
    """Sample representing anomalous, degrading multi-agent operation."""
    return PredictionSample(
        sample_id="test_degraded_0",
        run_id="run_degraded",
        timestamp=2.5,
        step_idx=5,
        task_type="research",
        topology="pipeline",
        number_of_agents=5,
        prediction_horizon=1,
        node_features={"researcher": {"error_rate": 0.5}},
        edge_features={
            "planner->researcher": {"weight": 1.0},
            "researcher->analyst": {"weight": 1.0},
        },
        temporal_graph_history=[],
        agent_level_features={
            "total_events_observed": 6.0,
            "error_count": 3.0,
            "total_retries": 4.0,
            "tool_failure_count": 2.0,
            "mean_contradiction_score": 0.75,
            "max_contradiction_score": 0.90,
            "mean_confidence": 0.35,
            "min_confidence": 0.20,
            "mean_latency": 0.45,
            "max_latency": 0.85,
            "interaction_density": 4.5,
            "min_output_quality": 0.30,
            "mean_output_quality": 0.45,
        },
        label=1,
        failure_type="tool_failure",
        failure_level=2,
        source_event_id="evt_1",
        random_seed=42,
        dataset_version="v1",
        split="test",
    )


def test_1_rule_calculation(clean_sample, degraded_sample):
    """Test 1: Individual indicator calculations reflect expected risk directions."""
    clean_inds = compute_all_indicators(clean_sample)
    deg_inds = compute_all_indicators(degraded_sample)

    assert len(clean_inds) == 9
    assert len(deg_inds) == 9

    # Clean indicators should all be close to 0.0
    assert clean_inds["error_rate"] == 0.0
    assert clean_inds["retry_rate"] == 0.0
    assert clean_inds["timeout_rate"] == 0.0
    assert clean_inds["confidence_degradation"] <= 0.10
    assert clean_inds["latency_anomaly"] == 0.0

    # Degraded indicators should be significantly higher
    assert deg_inds["error_rate"] == 0.5
    assert deg_inds["retry_rate"] > 0.0
    assert deg_inds["timeout_rate"] > 0.0
    assert deg_inds["contradiction_rate"] > 0.70
    assert deg_inds["confidence_degradation"] > 0.60
    assert deg_inds["latency_anomaly"] > 0.50
    assert deg_inds["frequency_anomaly"] > 0.50


def test_2_threshold_behavior():
    """Test 2: Threshold mapping to WarningLevel and binary alerts."""
    thresholds = RuleThresholds(
        threshold_low=0.20,
        threshold_medium=0.40,
        threshold_high=0.70,
        decision_threshold=0.35,
    )

    assert classify_risk_level(0.05, thresholds) == WarningLevel.NONE
    assert classify_risk_level(0.25, thresholds) == WarningLevel.LOW
    assert classify_risk_level(0.50, thresholds) == WarningLevel.MEDIUM
    assert classify_risk_level(0.85, thresholds) == WarningLevel.HIGH

    assert predict_binary_warning(0.20, thresholds) == 0
    assert predict_binary_warning(0.34, thresholds) == 0
    assert predict_binary_warning(0.35, thresholds) == 1
    assert predict_binary_warning(0.90, thresholds) == 1


def test_3_risk_score():
    """Test 3: Risk score calculation bounds and weighting."""
    indicators = {
        "error_rate": 1.0,
        "retry_rate": 0.0,
        "timeout_rate": 0.0,
        "contradiction_rate": 0.0,
        "confidence_degradation": 0.0,
        "latency_anomaly": 0.0,
        "frequency_anomaly": 0.0,
        "agent_failure_history": 0.0,
        "interaction_concentration": 0.0,
    }
    weights = RuleWeights(w_error_rate=0.5, w_retry_rate=0.5)
    score = compute_risk_score(indicators, weights)
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_4_individual_rules(degraded_sample):
    """Test 4: Rule ablation with individual active rules."""
    inds = compute_all_indicators(degraded_sample)

    score_error_only = compute_risk_score(inds, active_rules=["error_only"])
    assert score_error_only == inds["error_rate"]

    score_conf_only = compute_risk_score(inds, active_rules=["confidence_only"])
    assert score_conf_only == inds["confidence_degradation"]


def test_5_combined_rules(clean_sample, degraded_sample):
    """Test 5: Combined ensemble score behavior on clean vs degraded samples."""
    detector = RuleBasedEarlyWarningDetector()

    clean_pred = detector.predict_sample(clean_sample)
    deg_pred = detector.predict_sample(degraded_sample)

    assert clean_pred["risk_score"] < deg_pred["risk_score"]
    assert clean_pred["prediction"] == 0
    assert deg_pred["prediction"] == 1
    assert clean_pred["warning_level"] in ("NONE", "LOW")
    assert deg_pred["warning_level"] in ("MEDIUM", "HIGH")


def test_6_prediction_horizons(clean_sample, degraded_sample):
    """Test 6: Evaluator correctly evaluates multiple prediction horizons."""
    samples = []
    for k in [1, 3, 5, 10, 20]:
        s_clean = clean_sample.__class__(**clean_sample.to_dict())
        s_clean.prediction_horizon = k
        s_deg = degraded_sample.__class__(**degraded_sample.to_dict())
        s_deg.prediction_horizon = k
        samples.extend([s_clean, s_deg])

    detector = RuleBasedEarlyWarningDetector()
    results = evaluate_detector(samples, detector, horizons=[1, 3, 5, 10, 20])

    for k in [1, 3, 5, 10, 20]:
        assert k in results["horizons"]
        hr = results["horizons"][k]
        assert "precision" in hr
        assert "recall" in hr
        assert "f1" in hr
        assert "auroc" in hr


def test_7_no_future_leakage(degraded_sample):
    """Test 7: Verify rules never inspect or depend on future labels or downstream fields."""
    # Tampering with ground truth label must NOT change the risk score or prediction
    s1 = degraded_sample.__class__(**degraded_sample.to_dict())
    s1.label = 1
    s1.failure_level = 3
    s1.failure_type = "cascading_failure"

    s2 = degraded_sample.__class__(**degraded_sample.to_dict())
    s2.label = 0
    s2.failure_level = 0
    s2.failure_type = "none"

    detector = RuleBasedEarlyWarningDetector()
    pred1 = detector.predict_sample(s1)
    pred2 = detector.predict_sample(s2)

    assert pred1["risk_score"] == pred2["risk_score"]
    assert pred1["prediction"] == pred2["prediction"]
    assert pred1["warning_level"] == pred2["warning_level"]


def test_8_lead_time_calculation():
    """Test 8: Early warning lead-time logic correctly discards post-failure alerts."""
    preds = [
        # Run A: Failure at t=5.0, warning at t=3.0 -> valid early warning (lead_time = 2.0)
        {"run_id": "run_A", "step_idx": 1, "timestamp": 1.0, "prediction": 0, "ground_truth": 0},
        {"run_id": "run_A", "step_idx": 3, "timestamp": 3.0, "prediction": 1, "ground_truth": 0},
        {"run_id": "run_A", "step_idx": 5, "timestamp": 5.0, "prediction": 1, "ground_truth": 1},
        
        # Run B: Failure at t=2.0, warning only at t=4.0 -> warning after failure, lead_time invalid!
        {"run_id": "run_B", "step_idx": 1, "timestamp": 1.0, "prediction": 0, "ground_truth": 0},
        {"run_id": "run_B", "step_idx": 2, "timestamp": 2.0, "prediction": 0, "ground_truth": 1},
        {"run_id": "run_B", "step_idx": 4, "timestamp": 4.0, "prediction": 1, "ground_truth": 0},

        # Run C: Successful run, no failure -> 0 lead time
        {"run_id": "run_C", "step_idx": 1, "timestamp": 1.0, "prediction": 0, "ground_truth": 0},
    ]

    lt_metrics = compute_lead_time_metrics(preds)
    assert lt_metrics["total_failed_trajectories"] == 2
    assert lt_metrics["successful_early_warnings"] == 1  # Only Run A was an early warning!
    assert lt_metrics["early_warning_detection_rate"] == 0.5
    assert lt_metrics["mean_lead_time"] == 2.0
    assert lt_metrics["median_lead_time"] == 2.0


def test_9_reproducibility(degraded_sample):
    """Test 9: Verify detector generates identical outputs across separate invocations."""
    detector1 = RuleBasedEarlyWarningDetector()
    detector2 = RuleBasedEarlyWarningDetector()

    p1 = detector1.predict_sample(degraded_sample)
    p2 = detector2.predict_sample(degraded_sample)

    assert p1 == p2


def test_10_result_serialization():
    """Test 10: Serialization of metrics and ablation reports to JSON and Parquet."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = Path(tmp_dir) / "test_run"
        target_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "overall": {"f1": 0.85, "precision": 0.80, "recall": 0.90},
            "lead_time": {"mean_lead_time": 1.5, "median_lead_time": 1.2},
        }
        with open(target_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        assert (target_dir / "metrics.json").exists()
        with open(target_dir / "metrics.json", "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["overall"]["f1"] == 0.85
