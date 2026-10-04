"""Evaluation Metrics and Lead-Time Computation for Early Warning Baselines.

Implements:
- Classification metrics: Precision, Recall, F1, AUROC, AUPRC, FPR, FAR, Confusion Matrix
- Early warning lead-time metrics:
    lead_time = t_failure - t_warning (only when t_warning <= t_failure)
    * Mean lead time
    * Median lead time
    * Successful early warnings count
    * Warnings per trajectory
- Ablation testing across individual and combined rules
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from ml.baselines.rule_based.config import RuleBaselineConfig
from ml.baselines.rule_based.detector import RuleBasedEarlyWarningDetector


def compute_auroc(y_true: List[int], y_scores: List[float]) -> float:
    """Compute exact Area Under the ROC Curve via Mann-Whitney U statistic."""
    pos_scores = [s for yt, s in zip(y_true, y_scores) if yt == 1]
    neg_scores = [s for yt, s in zip(y_true, y_scores) if yt == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined / uninformative baseline

    # Mann-Whitney U calculation
    u_stat = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                u_stat += 1.0
            elif ps == ns:
                u_stat += 0.5

    return float(u_stat / (n_pos * n_neg))


def compute_auprc(y_true: List[int], y_scores: List[float]) -> float:
    """Compute Area Under the Precision-Recall Curve (Average Precision)."""
    n_pos = sum(1 for yt in y_true if yt == 1)
    if n_pos == 0 or len(y_true) == 0:
        return 0.0

    # Sort pairs by score descending
    pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
    
    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for _, yt in pairs:
        if yt == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)

    # Average precision (trapezoidal integration)
    ap = 0.0
    prev_rec = 0.0
    for p, r in zip(precisions, recalls):
        delta_r = r - prev_rec
        if delta_r > 0:
            ap += p * delta_r
            prev_rec = r

    return float(np.clip(ap, 0.0, 1.0))


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Calculate classification metrics and confusion matrix.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Binary model predictions (0 or 1).
        y_scores: Optional continuous risk scores for ROC/PR curves.
        
    Returns:
        Dictionary of precision, recall, f1, auroc, auprc, fpr, far, and confusion matrix.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} true vs {len(y_pred)} pred")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0

    auroc = compute_auroc(y_true, y_scores) if y_scores is not None else 0.5
    auprc = compute_auprc(y_true, y_scores) if y_scores is not None else precision

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_alarm_rate": round(far, 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total": len(y_true),
        },
    }


def compute_lead_time_metrics(
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate early warning lead times across simulation trajectories.
    
    Lead-Time Definition:
    For a warning emitted at time t_warning and failure occurring at time t_failure:
        lead_time = t_failure - t_warning
    Condition:
        Only warnings emitted strictly BEFORE or AT the failure (t_warning <= t_failure)
        are considered valid early warnings. Warnings emitted after failure are not early warnings.
        
    Args:
        predictions: List of sample prediction dicts (containing run_id, timestamp, prediction, ground_truth).
        
    Returns:
        Dictionary of mean, median, min, max lead times and early warning statistics.
    """
    # Group samples and predictions by run_id
    runs_map: Dict[str, List[Dict[str, Any]]] = {}
    for p in predictions:
        rid = p.get("run_id", "default")
        if rid not in runs_map:
            runs_map[rid] = []
        runs_map[rid].append(p)

    valid_lead_times: List[float] = []
    total_failed_runs = 0
    total_warnings_emitted = sum(1 for p in predictions if p.get("prediction", 0) == 1)
    successful_early_warnings = 0

    for rid, run_preds in runs_map.items():
        # Sort chronologically
        run_preds.sort(key=lambda x: (x.get("timestamp", 0.0) or 0.0, x.get("step_idx", 0) or 0))

        # Check if run experienced failure
        failed_points = [p for p in run_preds if p.get("ground_truth", 0) == 1]
        if not failed_points:
            continue

        total_failed_runs += 1
        t_failure = failed_points[0].get("timestamp", 0.0) or 0.0

        # Find earliest warning emitted in this run
        warning_points = [p for p in run_preds if p.get("prediction", 0) == 1]
        if not warning_points:
            continue

        earliest_warning = warning_points[0]
        t_warning = earliest_warning.get("timestamp", 0.0) or 0.0

        # Strict early warning check: t_warning <= t_failure
        if t_warning <= t_failure:
            lead = max(0.0, t_failure - t_warning)
            valid_lead_times.append(round(lead, 4))
            successful_early_warnings += 1

    num_runs = max(1, len(runs_map))
    warnings_per_trajectory = round(total_warnings_emitted / num_runs, 2)

    mean_lead = float(np.mean(valid_lead_times)) if valid_lead_times else 0.0
    median_lead = float(np.median(valid_lead_times)) if valid_lead_times else 0.0
    min_lead = float(np.min(valid_lead_times)) if valid_lead_times else 0.0
    max_lead = float(np.max(valid_lead_times)) if valid_lead_times else 0.0
    detection_rate = float(successful_early_warnings / total_failed_runs) if total_failed_runs > 0 else 0.0

    return {
        "mean_lead_time": round(mean_lead, 4),
        "median_lead_time": round(median_lead, 4),
        "min_lead_time": round(min_lead, 4),
        "max_lead_time": round(max_lead, 4),
        "successful_early_warnings": successful_early_warnings,
        "total_failed_trajectories": total_failed_runs,
        "early_warning_detection_rate": round(detection_rate, 4),
        "total_warnings_emitted": total_warnings_emitted,
        "warnings_per_trajectory": warnings_per_trajectory,
        "lead_times_sample": valid_lead_times[:20],
    }


def evaluate_detector(
    samples: List[Any],
    detector: RuleBasedEarlyWarningDetector,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Evaluate detector performance across prediction horizons and compute lead times.
    
    Args:
        samples: Sequence of PredictionSample instances or tabular dicts.
        detector: RuleBasedEarlyWarningDetector instance.
        horizons: List of horizons k to evaluate (default: [1, 3, 5, 10, 20]).
        
    Returns:
        Structured evaluation report with overall and per-horizon performance.
    """
    eval_horizons = horizons or [1, 3, 5, 10, 20]
    all_preds = detector.predict_batch(samples)

    # Lead time is evaluated across the trajectory sequence
    lead_time_stats = compute_lead_time_metrics(all_preds)

    horizon_results: Dict[int, Dict[str, Any]] = {}

    for k in eval_horizons:
        k_indices = [
            idx for idx, p in enumerate(all_preds)
            if p.get("prediction_horizon") == k
        ]
        if not k_indices:
            continue

        y_true = [int(all_preds[idx]["ground_truth"]) for idx in k_indices]
        y_pred = [int(all_preds[idx]["prediction"]) for idx in k_indices]
        y_scores = [float(all_preds[idx]["risk_score"]) for idx in k_indices]

        metrics = compute_classification_metrics(y_true, y_pred, y_scores)
        horizon_results[k] = metrics

    # Combined / overall metrics
    all_y_true = [int(p["ground_truth"]) for p in all_preds if p.get("ground_truth") is not None]
    all_y_pred = [int(p["prediction"]) for p in all_preds if p.get("ground_truth") is not None]
    all_y_scores = [float(p["risk_score"]) for p in all_preds if p.get("ground_truth") is not None]

    overall_metrics = compute_classification_metrics(all_y_true, all_y_pred, all_y_scores)

    return {
        "overall": overall_metrics,
        "lead_time": lead_time_stats,
        "horizons": horizon_results,
    }


def evaluate_ablations(
    samples: List[Any],
    base_config: Optional[RuleBaselineConfig] = None,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run rule ablation study across individual rules and the combined ensemble.
    
    Ablations:
    - combined_rules
    - error_only
    - retry_only
    - timeout_only
    - contradiction_only
    - confidence_only
    - latency_only
    - frequency_anomaly_only
    - concentration_only
    """
    config = base_config or RuleBaselineConfig()
    ablation_rules = [
        "combined_rules",
        "error_only",
        "retry_only",
        "timeout_only",
        "contradiction_only",
        "confidence_only",
        "latency_only",
        "frequency_anomaly_only",
        "concentration_only",
    ]

    ablation_reports: Dict[str, Dict[str, Any]] = {}

    for rule_name in ablation_rules:
        # Create ablation configuration
        ablation_cfg = config.model_copy(deep=True)
        ablation_cfg.active_rules = [rule_name]
        detector = RuleBasedEarlyWarningDetector(config=ablation_cfg)

        eval_result = evaluate_detector(samples, detector, horizons=horizons)
        ablation_reports[rule_name] = {
            "overall": eval_result["overall"],
            "lead_time": eval_result["lead_time"],
            "horizons": eval_result["horizons"],
        }

    return ablation_reports
