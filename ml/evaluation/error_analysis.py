"""Error Analysis Module - Phase 12 (Section 22).

Identifies, extracts, and organizes representative prediction errors:
1. False Positives (y_true = 0, y_pred = 1):
   Model predicted impending failure, but no failure occurred within horizon.
2. False Negatives (y_true = 1, y_pred = 0):
   Failure occurred, but model failed to issue an early warning.

Stores representative cases with trajectory ID, timestamp, model, predicted probability,
ground truth, and contextual features.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class ErrorCase:
    """Detailed record of an individual classification error."""
    error_type: str  # 'False Positive' or 'False Negative'
    sample_id: str
    run_id: str
    timestamp: float
    horizon: int
    model_name: str
    predicted_probability: float
    threshold: float
    true_label: int
    predicted_label: int
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ErrorAnalyzer:
    """Analyzes and collects false positives and false negatives for model diagnostics."""

    def analyze_errors(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth_metadata: Dict[str, Dict[str, Any]],
        model_name: str,
        horizon: int,
        threshold: float = 0.50,
        max_cases_per_type: int = 10,
    ) -> Dict[str, Any]:
        """Extract representative false positive and false negative error cases.
        
        Args:
            predictions: List of test predictions.
            ground_truth_metadata: Mapping sample_id -> ground truth dictionary.
            model_name: Name of evaluated model.
            horizon: Evaluated horizon.
            threshold: Decision threshold.
            max_cases_per_type: Maximum error examples to retain per category.
            
        Returns:
            Dictionary with 'false_positives' and 'false_negatives' lists.
        """
        false_positives: List[ErrorCase] = []
        false_negatives: List[ErrorCase] = []

        for pred in predictions:
            sid = pred.get("sample_id", "")
            rid = pred.get("run_id", "")
            t = float(pred.get("timestamp", 0.0))
            yt = int(pred.get("true_label", pred.get("ground_truth", 0)))
            yp = int(pred.get("predicted_label", pred.get("prediction", 0)))
            prob = float(pred.get("predicted_probability", pred.get("prediction", 0.0)))

            meta = ground_truth_metadata.get(sid, {})
            context = {
                "topology": meta.get("topology", "unknown"),
                "task_type": meta.get("task_type", "unknown"),
                "failure_type": meta.get("failure_type", "none"),
                "failure_level": meta.get("failure_level", 0),
                "number_of_agents": meta.get("number_of_agents", 0),
            }

            if yt == 0 and yp == 1:
                # False positive
                if len(false_positives) < max_cases_per_type:
                    false_positives.append(
                        ErrorCase(
                            error_type="False Positive",
                            sample_id=sid,
                            run_id=rid,
                            timestamp=t,
                            horizon=horizon,
                            model_name=model_name,
                            predicted_probability=round(prob, 4),
                            threshold=threshold,
                            true_label=yt,
                            predicted_label=yp,
                            context=context,
                        )
                    )
            elif yt == 1 and yp == 0:
                # False negative
                if len(false_negatives) < max_cases_per_type:
                    false_negatives.append(
                        ErrorCase(
                            error_type="False Negative",
                            sample_id=sid,
                            run_id=rid,
                            timestamp=t,
                            horizon=horizon,
                            model_name=model_name,
                            predicted_probability=round(prob, 4),
                            threshold=threshold,
                            true_label=yt,
                            predicted_label=yp,
                            context=context,
                        )
                    )

        return {
            "false_positives": [c.to_dict() for c in false_positives],
            "false_negatives": [c.to_dict() for c in false_negatives],
            "total_false_positives": sum(
                1 for p in predictions
                if int(p.get("true_label", p.get("ground_truth", 0))) == 0
                and int(p.get("predicted_label", p.get("prediction", 0))) == 1
            ),
            "total_false_negatives": sum(
                1 for p in predictions
                if int(p.get("true_label", p.get("ground_truth", 0))) == 1
                and int(p.get("predicted_label", p.get("prediction", 0))) == 0
            ),
        }
