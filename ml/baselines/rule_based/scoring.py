"""Transparent Risk Scoring and Classification Logic.

Calculates weighted risk scores from indicators and maps them into:
- Continuous risk score in [0.0, 1.0]
- WarningLevel (NONE, LOW, MEDIUM, HIGH)
- Binary warning prediction (0 or 1)
"""

from typing import Dict, List, Optional
import numpy as np

from ml.baselines.rule_based.config import RuleWeights, RuleThresholds, WarningLevel


# Map ablation rule names to indicator keys
RULE_ALIAS_MAP = {
    "error_only": ["error_rate"],
    "retry_only": ["retry_rate"],
    "timeout_only": ["timeout_rate"],
    "contradiction_only": ["contradiction_rate"],
    "confidence_only": ["confidence_degradation"],
    "latency_only": ["latency_anomaly"],
    "frequency_anomaly_only": ["frequency_anomaly"],
    "concentration_only": ["interaction_concentration"],
    "agent_failure_only": ["agent_failure_history"],
    "combined_rules": None,  # All rules
}


def compute_risk_score(
    indicators: Dict[str, float],
    weights: Optional[RuleWeights] = None,
    active_rules: Optional[List[str]] = None,
) -> float:
    """Compute normalized composite risk score in [0.0, 1.0].
    
    Args:
        indicators: Dictionary of indicator values in [0.0, 1.0].
        weights: Optional RuleWeights configuration (default weights used if omitted).
        active_rules: Optional filter of active rule/indicator names for ablation.
        
    Returns:
        Weighted risk score float in [0.0, 1.0].
    """
    rule_weights = weights or RuleWeights()
    w_map = {
        "error_rate": rule_weights.w_error_rate,
        "retry_rate": rule_weights.w_retry_rate,
        "timeout_rate": rule_weights.w_timeout_rate,
        "contradiction_rate": rule_weights.w_contradiction_rate,
        "confidence_degradation": rule_weights.w_confidence_degradation,
        "latency_anomaly": rule_weights.w_latency_anomaly,
        "frequency_anomaly": rule_weights.w_frequency_anomaly,
        "agent_failure_history": 0.05,
        "interaction_concentration": rule_weights.w_interaction_concentration,
    }

    # Resolve active indicators from rule list / aliases
    active_keys = None
    if active_rules:
        resolved = []
        for r in active_rules:
            if r in RULE_ALIAS_MAP:
                target = RULE_ALIAS_MAP[r]
                if target is None:
                    resolved = list(w_map.keys())
                    break
                else:
                    resolved.extend(target)
            elif r in w_map:
                resolved.append(r)
        active_keys = set(resolved)

    total_weight = 0.0
    accumulated_score = 0.0

    for key, weight in w_map.items():
        if active_keys is not None and key not in active_keys:
            continue
        val = indicators.get(key, 0.0)
        accumulated_score += weight * val
        total_weight += weight

    if total_weight <= 0.0:
        return 0.0

    # Normalize by sum of active weights
    final_score = accumulated_score / total_weight
    return float(np.clip(final_score, 0.0, 1.0))


def classify_risk_level(
    risk_score: float,
    thresholds: Optional[RuleThresholds] = None,
) -> WarningLevel:
    """Map continuous risk score to categorical WarningLevel."""
    t = thresholds or RuleThresholds()
    if risk_score >= t.threshold_high:
        return WarningLevel.HIGH
    elif risk_score >= t.threshold_medium:
        return WarningLevel.MEDIUM
    elif risk_score >= t.threshold_low:
        return WarningLevel.LOW
    return WarningLevel.NONE


def predict_binary_warning(
    risk_score: float,
    thresholds: Optional[RuleThresholds] = None,
) -> int:
    """Map continuous risk score to binary early warning alert (0 or 1)."""
    t = thresholds or RuleThresholds()
    return 1 if risk_score >= t.decision_threshold else 0
