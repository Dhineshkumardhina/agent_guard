"""Uncertainty Estimation and Trajectory-Level Statistical Testing - Phase 12 (Sections 13 & 15).

Resampling Methodology:
To account for within-run temporal correlation between events, the resampling unit
is the simulation trajectory (run_id), rather than independent event samples.
Computes:
1. Trajectory-level block bootstrap confidence intervals (95% CI) for F1, AUROC, AUPRC
2. Paired trajectory bootstrap difference tests comparing Model A vs Model B
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from ml.evaluation.metrics import compute_comprehensive_metrics
from ml.evaluation.schema import BootstrapResult, PairwiseComparisonRecord


def trajectory_block_bootstrap_ci(
    predictions: List[Dict[str, Any]],
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, BootstrapResult]:
    """Compute bootstrap confidence intervals using run-level block resampling.
    
    Args:
        predictions: List of test prediction dictionaries containing run_id, ground_truth/true_label, predicted_label, predicted_probability.
        n_bootstraps: Number of bootstrap resamples.
        confidence_level: Target confidence coverage (default 0.95).
        random_seed: Seed for reproducible resampling.
        
    Returns:
        Mapping of metric_name -> BootstrapResult.
    """
    rng = np.random.RandomState(random_seed)

    # Group predictions by run_id
    preds_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        rid = p.get("run_id", "")
        preds_by_run[rid].append(p)

    run_ids = sorted(list(preds_by_run.keys()))
    n_runs = len(run_ids)
    if n_runs == 0:
        return {}

    f1_scores: List[float] = []
    auroc_scores: List[float] = []
    auprc_scores: List[float] = []

    for _ in range(n_bootstraps):
        sampled_runs = rng.choice(run_ids, size=n_runs, replace=True)
        sampled_preds = []
        for rid in sampled_runs:
            sampled_preds.extend(preds_by_run[rid])

        y_true = [int(p.get("true_label", p.get("ground_truth", 0))) for p in sampled_preds]
        y_pred = [int(p.get("predicted_label", p.get("prediction", 0))) for p in sampled_preds]
        y_prob = [float(p.get("predicted_probability", p.get("prediction", 0.0))) for p in sampled_preds]

        m = compute_comprehensive_metrics(y_true, y_pred, y_prob)
        f1_scores.append(m.get("f1", 0.0))
        auroc_scores.append(m.get("auroc", 0.5))
        auprc_scores.append(m.get("auprc", 0.0))

    alpha = (1.0 - confidence_level) / 2.0
    results: Dict[str, BootstrapResult] = {}

    for name, arr in [("f1", f1_scores), ("auroc", auroc_scores), ("auprc", auprc_scores)]:
        data = np.array(arr)
        results[name] = BootstrapResult(
            metric_name=name,
            mean=round(float(np.mean(data)), 4),
            median=round(float(np.median(data)), 4),
            ci_lower=round(float(np.percentile(data, alpha * 100)), 4),
            ci_upper=round(float(np.percentile(data, (1.0 - alpha) * 100)), 4),
            confidence_level=confidence_level,
        )

    return results


def paired_trajectory_bootstrap_test(
    preds_a: List[Dict[str, Any]],
    preds_b: List[Dict[str, Any]],
    model_a_name: str,
    model_b_name: str,
    metric_name: str = "f1",
    horizon: int = 1,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> PairwiseComparisonRecord:
    """Perform paired trajectory-level bootstrap test between Model A and Model B."""
    rng = np.random.RandomState(random_seed)

    # Group predictions by run_id
    a_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    b_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in preds_a:
        a_by_run[p.get("run_id", "")].append(p)
    for p in preds_b:
        b_by_run[p.get("run_id", "")].append(p)

    shared_runs = sorted(list(set(a_by_run.keys()) & set(b_by_run.keys())))
    n_runs = len(shared_runs)

    if n_runs == 0:
        return PairwiseComparisonRecord(
            model_a=model_a_name,
            model_b=model_b_name,
            horizon=horizon,
            metric=metric_name,
            value_a=0.0,
            value_b=0.0,
            difference=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            sample_count=0,
            statistically_significant=False,
        )

    # Calculate baseline point estimates
    def get_metric_for_preds(preds):
        yt = [int(p.get("true_label", p.get("ground_truth", 0))) for p in preds]
        yp = [int(p.get("predicted_label", p.get("prediction", 0))) for p in preds]
        ypr = [float(p.get("predicted_probability", p.get("prediction", 0.0))) for p in preds]
        m = compute_comprehensive_metrics(yt, yp, ypr)
        return m.get(metric_name, 0.0)

    val_a = get_metric_for_preds(preds_a)
    val_b = get_metric_for_preds(preds_b)
    diff_point = val_a - val_b

    # Paired block bootstrap differences
    diffs: List[float] = []
    for _ in range(n_bootstraps):
        sampled_runs = rng.choice(shared_runs, size=n_runs, replace=True)
        samp_a = [p for r in sampled_runs for p in a_by_run[r]]
        samp_b = [p for r in sampled_runs for p in b_by_run[r]]

        m_a = get_metric_for_preds(samp_a)
        m_b = get_metric_for_preds(samp_b)
        diffs.append(m_a - m_b)

    alpha = (1.0 - confidence_level) / 2.0
    ci_low = float(np.percentile(diffs, alpha * 100))
    ci_high = float(np.percentile(diffs, (1.0 - alpha) * 100))

    # Empirical two-sided p-value
    diff_arr = np.array(diffs)
    p_le_0 = np.mean(diff_arr <= 0.0)
    p_ge_0 = np.mean(diff_arr >= 0.0)
    p_val = float(2.0 * min(p_le_0, p_ge_0))
    p_val = min(1.0, max(0.0, p_val))

    sig = (ci_low > 0.0 or ci_high < 0.0) and (p_val < 0.05)

    return PairwiseComparisonRecord(
        model_a=model_a_name,
        model_b=model_b_name,
        horizon=horizon,
        metric=metric_name,
        value_a=round(val_a, 4),
        value_b=round(val_b, 4),
        difference=round(diff_point, 4),
        ci_lower=round(ci_low, 4),
        ci_upper=round(ci_high, 4),
        p_value=round(p_val, 4),
        sample_count=len(preds_a),
        statistically_significant=bool(sig),
    )
