"""Evaluation Engine and Early Warning Lead-Time Tracking for Temporal Sequence Baselines.

Computes:
1. Classification metrics: Precision, Recall, F1, AUROC, AUPRC, FPR, FAR, Confusion Matrix
2. Early Warning Lead Time:
   lead_time = t_failure - t_warning (only when t_warning <= t_failure)
3. Structured prediction records conforming to Section 18 schema
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ml.baselines.rule_based.evaluator import (
    compute_classification_metrics,
    compute_lead_time_metrics,
)


def evaluate_sequence_model(
    model: Any,
    test_loader: Any,
    threshold: float = 0.50,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate trained temporal sequence model on held-out test split.
    
    Args:
        model: Trained PyTorch sequence model (LSTM or GRU).
        test_loader: DataLoader yielding (sequences, labels, metadata).
        threshold: Frozen decision cutoff calibrated on validation split.
        device: Torch compute device.
        
    Returns:
        Dictionary containing classification metrics, lead times, and prediction records.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for sequence evaluation.")

    dev = device or next(model.parameters()).device
    model.eval()

    all_probs = []
    all_trues = []
    all_metas = []

    with torch.no_grad():
        for batch_seqs, batch_labels, batch_metas in test_loader:
            batch_seqs = batch_seqs.to(dev)
            logits = model(batch_seqs)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

            all_probs.extend(probs if isinstance(probs, list) else [probs])
            all_trues.extend(batch_labels.numpy().astype(int).tolist())

            # Batch metas from DataLoader comes as dict of lists
            batch_size = len(batch_labels)
            for b_idx in range(batch_size):
                m_item = {k: batch_metas[k][b_idx] for k in batch_metas.keys()}
                # Extract Python primitive values if tensors/numpy
                clean_m = {
                    k: (v.item() if hasattr(v, "item") else v)
                    for k, v in m_item.items()
                }
                all_metas.append(clean_m)

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

    # Compile prediction records and determine run-level failure/warning timestamps
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

    predictions: List[Dict[str, Any]] = []
    for idx, (meta, yt, yp, prob) in enumerate(zip(all_metas, all_trues, all_preds, all_probs)):
        rid = meta.get("run_id", "")
        curr_t = float(meta.get("timestamp", 0.0))
        t_fail = runs_failure_time.get(rid)
        t_warn = runs_first_warning.get(rid)

        # Lead time applies when run experiences failure and warning precedes failure
        lead = None
        if t_fail is not None and t_warn is not None and t_warn <= t_fail:
            lead = round(t_fail - t_warn, 4)

        rec = {
            "sample_id": meta.get("sample_id", f"s_{idx}"),
            "run_id": rid,
            "timestamp": curr_t,
            "horizon": int(meta.get("prediction_horizon", 1)),
            "sequence_length": int(meta.get("sequence_length", 10)),
            "model_name": getattr(model, "model_name", "sequence_baseline"),
            "true_label": int(yt),
            "predicted_probability": round(float(prob), 4),
            "predicted_label": int(yp),
            "warning_timestamp": round(t_warn, 4) if t_warn is not None else None,
            "failure_timestamp": round(t_fail, 4) if t_fail is not None else None,
            "lead_time": lead,
            # Helper fields for generic lead_time calculator
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
