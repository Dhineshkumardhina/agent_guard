"""Scientific Evaluation Metrics Engine - Phase 12.

Calculates:
1. Classification Metrics: Precision, Recall, F1, AUROC, AUPRC, FPR, False Alarm Rate
2. Probabilistic Calibration: Brier Score, Expected Calibration Error (ECE), Reliability Bins
3. Early Warning Lead Time: Mean, Median, Std, Min, Max, Warning Rate
4. Curve Coordinates: ROC curve (FPR vs TPR), PR curve (Recall vs Precision)
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        roc_curve,
        precision_recall_curve,
        brier_score_loss,
        confusion_matrix,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_comprehensive_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute standard classification and calibration metrics.
    
    Args:
        y_true: Ground-truth binary labels in {0, 1}.
        y_pred: Predicted binary labels in {0, 1}.
        y_prob: Predicted failure probabilities in [0.0, 1.0].
        
    Returns:
        Dictionary of computed metric values.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)
    n_samples = len(y_t)

    if n_samples == 0:
        return {}

    pos_count = int(np.sum(y_t == 1))
    neg_count = int(np.sum(y_t == 0))

    # Basic confusion matrix values
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0

    auroc = 0.5
    auprc = float(pos_count / n_samples) if n_samples > 0 else 0.0
    brier = None
    ece = None

    if y_prob is not None and len(y_prob) == n_samples:
        probs = np.array(y_prob, dtype=float)
        brier = float(np.mean((probs - y_t) ** 2))
        ece = compute_expected_calibration_error(y_t, probs, n_bins=10)

        # AUROC & AUPRC (require at least 1 positive and 1 negative)
        if pos_count > 0 and neg_count > 0 and HAS_SKLEARN:
            try:
                auroc = float(roc_auc_score(y_t, probs))
            except Exception:
                auroc = 0.5
            try:
                auprc = float(average_precision_score(y_t, probs))
            except Exception:
                auprc = float(pos_count / n_samples)
        elif pos_count > 0 and neg_count == 0:
            # All positive ground truth
            auroc = 1.0 if precision == 1.0 else 0.5
            auprc = 1.0 if precision == 1.0 else float(pos_count / n_samples)

    return {
        "sample_count": n_samples,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_alarm_rate": round(far, 4),
        "brier_score": round(brier, 4) if brier is not None else None,
        "ece": round(ece, 4) if ece is not None else None,
    }


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across probability bins."""
    y_t = np.array(y_true, dtype=int)
    probs = np.array(y_prob, dtype=float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_t)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        bin_size = int(np.sum(mask))

        if bin_size > 0:
            acc = float(np.mean(y_t[mask]))
            conf = float(np.mean(probs[mask]))
            ece += (bin_size / n) * abs(acc - conf)

    return float(ece)


def compute_calibration_curve_data(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    """Compute reliability diagram data points."""
    y_t = np.array(y_true, dtype=int)
    probs = np.array(y_prob, dtype=float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []

    for i in range(n_bins):
        b_low = float(bin_boundaries[i])
        b_high = float(bin_boundaries[i + 1])
        mask = (probs >= b_low) & (probs < b_high if i < n_bins - 1 else probs <= b_high)
        cnt = int(np.sum(mask))

        if cnt > 0:
            obs = float(np.mean(y_t[mask]))
            pred_m = float(np.mean(probs[mask]))
        else:
            obs = 0.0
            pred_m = (b_low + b_high) / 2.0

        bins_data.append({
            "bin_idx": i,
            "bin_lower": round(b_low, 2),
            "bin_upper": round(b_high, 2),
            "predicted_prob_mean": round(pred_m, 4),
            "observed_positive_ratio": round(obs, 4),
            "sample_count": cnt,
        })
    return bins_data


def compute_roc_and_pr_curves(
    y_true: List[int],
    y_prob: List[float],
) -> Dict[str, Any]:
    """Compute ROC and PR curve coordinate arrays."""
    y_t = np.array(y_true, dtype=int)
    probs = np.array(y_prob, dtype=float)

    if not HAS_SKLEARN or len(np.unique(y_t)) < 2:
        # Fallback when single class present
        return {
            "roc": {
                "fpr": [0.0, 0.0, 1.0],
                "tpr": [0.0, 1.0, 1.0],
                "thresholds": [1.0, 0.5, 0.0],
                "auc": 1.0 if np.all(y_t == 1) else 0.5,
            },
            "pr": {
                "precision": [1.0, 1.0, 0.0],
                "recall": [0.0, 1.0, 1.0],
                "thresholds": [1.0, 0.5],
                "auc": 1.0 if np.all(y_t == 1) else 0.5,
            },
        }

    fpr, tpr, roc_thresh = roc_curve(y_t, probs)
    prec, rec, pr_thresh = precision_recall_curve(y_t, probs)

    try:
        roc_auc = float(roc_auc_score(y_t, probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_t, probs))
    except Exception:
        pr_auc = float(np.mean(y_t))

    return {
        "roc": {
            "fpr": [round(float(v), 4) for v in fpr],
            "tpr": [round(float(v), 4) for v in tpr],
            "thresholds": [round(float(v), 4) for v in roc_thresh],
            "auc": round(roc_auc, 4),
        },
        "pr": {
            "precision": [round(float(v), 4) for v in prec],
            "recall": [round(float(v), 4) for v in rec],
            "thresholds": [round(float(v), 4) for v in pr_thresh],
            "auc": round(pr_auc, 4),
        },
    }


def compute_lead_time_statistics(lead_times: List[float], total_runs: int = 1) -> Dict[str, Any]:
    """Compute lead-time summary metrics from list of positive lead times.
    
    Args:
        lead_times: Valid early warning lead times (t_fail - t_warn > 0).
        total_runs: Total number of evaluated simulation trajectories.
    """
    if not lead_times:
        return {
            "successful_early_warnings": 0,
            "mean_lead_time": 0.0,
            "median_lead_time": 0.0,
            "std_lead_time": 0.0,
            "min_lead_time": 0.0,
            "max_lead_time": 0.0,
            "q25_lead_time": 0.0,
            "q75_lead_time": 0.0,
            "warnings_per_trajectory": 0.0,
        }

    arr = np.array(lead_times, dtype=float)
    return {
        "successful_early_warnings": len(arr),
        "mean_lead_time": round(float(np.mean(arr)), 4),
        "median_lead_time": round(float(np.median(arr)), 4),
        "std_lead_time": round(float(np.std(arr)), 4),
        "min_lead_time": round(float(np.min(arr)), 4),
        "max_lead_time": round(float(np.max(arr)), 4),
        "q25_lead_time": round(float(np.percentile(arr, 25)), 4),
        "q75_lead_time": round(float(np.percentile(arr, 75)), 4),
        "warnings_per_trajectory": round(float(len(arr) / max(1, total_runs)), 4),
    }
