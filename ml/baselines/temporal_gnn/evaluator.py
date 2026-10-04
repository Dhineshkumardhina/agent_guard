"""Evaluation Engine and Early Warning Lead-Time Tracking for Temporal GNNs.

Computes:
1. Classification metrics: Precision, Recall, F1, AUROC, AUPRC, FPR, FAR, Confusion Matrix
2. Early Warning Lead Time:
   lead_time = t_failure - t_warning (only when t_warning <= t_failure)
3. Structured prediction records conforming to Phase 11 / Section 21 schema
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch

from ml.baselines.rule_based.evaluator import (
    compute_classification_metrics,
    compute_lead_time_metrics,
)
from ml.baselines.temporal_gnn.dataset import TemporalRunTrajectory
from ml.baselines.temporal_gnn.models import TemporalGraphFailurePredictor


def evaluate_temporal_gnn(
    model: TemporalGraphFailurePredictor,
    test_trajectories: List[TemporalRunTrajectory],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Evaluate trained Temporal GNN model on held-out test trajectories.
    
    Args:
        model: Trained TemporalGraphFailurePredictor instance.
        test_trajectories: List of held-out test simulation run trajectories.
        threshold: Frozen decision cutoff calibrated exclusively on validation split.
        
    Returns:
        Dictionary containing classification metrics, lead times, and prediction records.
    """
    model.eval()

    all_probs: List[float] = []
    all_trues: List[int] = []
    all_metas: List[Dict[str, Any]] = []

    with torch.no_grad():
        for traj in test_trajectories:
            # Strictly reset memory between independent test trajectories
            model.reset_memory()

            inter_idx = 0
            total_inters = len(traj.interactions)

            for pp in traj.prediction_points:
                target_t = pp.timestamp

                # Process all interactions up to prediction cutoff
                while inter_idx < total_inters and traj.interactions[inter_idx].timestamp <= target_t:
                    ev = traj.interactions[inter_idx]
                    e_feat = torch.tensor(ev.features, dtype=torch.float32, device=model.device)
                    model.process_interaction(
                        source_agent=ev.source_agent,
                        target_agent=ev.target_agent,
                        timestamp=ev.timestamp,
                        edge_features=e_feat,
                    )
                    inter_idx += 1

                # Predict failure probability at target_t
                _, prob = model.predict_at_timestamp(
                    agent_ids=pp.active_agents,
                    node_features_dict=pp.node_features,
                    current_timestamp=target_t,
                )

                all_probs.append(prob)
                all_trues.append(int(pp.label))
                all_metas.append({
                    "sample_id": pp.sample_id,
                    "run_id": pp.run_id,
                    "timestamp": round(pp.timestamp, 4),
                    "prediction_horizon": pp.prediction_horizon,
                })

    if not all_trues:
        return {
            "metrics": {},
            "lead_time": {},
            "predictions": [],
        }

    # Binary predictions using frozen validation threshold
    all_preds = [1 if p >= threshold else 0 for p in all_probs]

    # Classification metrics
    metrics = compute_classification_metrics(all_trues, all_preds, all_probs)
    metrics["threshold_used"] = round(float(threshold), 4)

    # Determine run-level earliest failure and warning timestamps
    runs_failure_time: Dict[str, float] = {}
    for meta, yt in zip(all_metas, all_trues):
        rid = meta.get("run_id", "")
        t = float(meta.get("timestamp", 0.0))
        if yt == 1:
            if rid not in runs_failure_time or t < runs_failure_time[rid]:
                runs_failure_time[rid] = t

    runs_first_warning: Dict[str, float] = {}
    for meta, yp in zip(all_metas, all_preds):
        rid = meta.get("run_id", "")
        t = float(meta.get("timestamp", 0.0))
        if yp == 1:
            if rid not in runs_first_warning or t < runs_first_warning[rid]:
                runs_first_warning[rid] = t

    # Build Section 21 prediction records
    predictions: List[Dict[str, Any]] = []
    for idx, (meta, yt, yp, prob) in enumerate(zip(all_metas, all_trues, all_preds, all_probs)):
        rid = meta.get("run_id", "")
        curr_t = float(meta.get("timestamp", 0.0))
        t_fail = runs_failure_time.get(rid)
        t_warn = runs_first_warning.get(rid)

        # Lead time calculation: only valid when warning precedes failure
        lead = None
        if t_fail is not None and t_warn is not None and t_warn <= t_fail:
            lead = round(t_fail - t_warn, 4)

        rec = {
            "sample_id": meta.get("sample_id", f"s_{idx}"),
            "run_id": rid,
            "timestamp": curr_t,
            "horizon": int(meta.get("prediction_horizon", 1)),
            "model_name": getattr(model, "model_name", "temporal_gnn"),
            "true_label": int(yt),
            "predicted_probability": round(float(prob), 4),
            "predicted_label": int(yp),
            "warning_timestamp": round(t_warn, 4) if t_warn is not None else None,
            "failure_timestamp": round(t_fail, 4) if t_fail is not None else None,
            "lead_time": lead,
            # Helper keys for lead_time metric calculation
            "prediction": int(yp),
            "ground_truth": int(yt),
        }
        predictions.append(rec)

    # Lead-time statistics
    lead_time_stats = compute_lead_time_metrics(predictions)

    return {
        "metrics": metrics,
        "lead_time": lead_time_stats,
        "predictions": predictions,
    }
