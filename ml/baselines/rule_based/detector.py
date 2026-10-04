"""Rule-Based Early Warning Detector.

Applies causal behavioral rules to emit transparent early warning alerts for multi-agent systems.
"""

from typing import Dict, Any, List, Union, Optional

from ml.baselines.rule_based.config import RuleBaselineConfig, WarningLevel
from ml.baselines.rule_based.rules import compute_all_indicators
from ml.baselines.rule_based.scoring import compute_risk_score, classify_risk_level, predict_binary_warning
from ml.data.schema import PredictionSample


class RuleBasedEarlyWarningDetector:
    """Early warning detector operating purely on transparent, non-ML behavioral heuristics."""

    def __init__(self, config: Optional[RuleBaselineConfig] = None) -> None:
        self.config = config or RuleBaselineConfig()

    def predict_sample(
        self,
        sample: Union[PredictionSample, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate early warning risk score and prediction for an individual sample.
        
        Args:
            sample: PredictionSample instance or tabular dictionary row.
            
        Returns:
            Dictionary containing prediction outputs, risk scores, warning level, and indicators.
        """
        # 1. Compute causal indicators strictly from <= t information
        indicators = compute_all_indicators(sample)

        # 2. Compute composite risk score
        risk_score = compute_risk_score(
            indicators=indicators,
            weights=self.config.weights,
            active_rules=self.config.active_rules,
        )

        # 3. Classify warning level & binary alert
        level = classify_risk_level(risk_score, self.config.thresholds)
        binary_pred = predict_binary_warning(risk_score, self.config.thresholds)

        # Extract metadata
        sample_id = getattr(sample, "sample_id", None) or (sample.get("sample_id") if isinstance(sample, dict) else "")
        run_id = getattr(sample, "run_id", None) or (sample.get("run_id") if isinstance(sample, dict) else "")
        step_idx = getattr(sample, "step_idx", None) if hasattr(sample, "step_idx") else sample.get("step_idx")
        timestamp = getattr(sample, "timestamp", None) if hasattr(sample, "timestamp") else sample.get("timestamp")
        horizon = getattr(sample, "prediction_horizon", None) if hasattr(sample, "prediction_horizon") else sample.get("prediction_horizon")
        ground_truth = getattr(sample, "label", None) if hasattr(sample, "label") else sample.get("label")

        return {
            "sample_id": sample_id,
            "run_id": run_id,
            "step_idx": step_idx,
            "timestamp": timestamp,
            "prediction_horizon": horizon,
            "ground_truth": ground_truth,
            "risk_score": round(risk_score, 4),
            "prediction": binary_pred,
            "warning_level": level.value,
            "indicators": {k: round(v, 4) for k, v in indicators.items()},
        }

    def predict_batch(
        self,
        samples: List[Union[PredictionSample, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Generate predictions for a batch of samples."""
        return [self.predict_sample(s) for s in samples]
