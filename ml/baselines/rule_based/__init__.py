"""Rule-Based Early Warning Baseline Package.

Provides transparent, non-ML baseline indicators, risk scoring, thresholds,
detection engine, and evaluation infrastructure for multi-agent failure prediction.
"""

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
)

__all__ = [
    "WarningLevel",
    "RuleWeights",
    "RuleThresholds",
    "RuleBaselineConfig",
    "compute_error_rate",
    "compute_retry_rate",
    "compute_timeout_rate",
    "compute_contradiction_rate",
    "compute_confidence_degradation",
    "compute_latency_anomaly",
    "compute_frequency_anomaly",
    "compute_agent_failure_history",
    "compute_interaction_concentration",
    "compute_all_indicators",
    "compute_risk_score",
    "classify_risk_level",
    "predict_binary_warning",
    "RuleBasedEarlyWarningDetector",
    "compute_classification_metrics",
    "compute_lead_time_metrics",
    "evaluate_detector",
    "evaluate_ablations",
]
