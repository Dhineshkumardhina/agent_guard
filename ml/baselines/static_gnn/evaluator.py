"""Evaluation Engine and Early Warning Lead-Time Tracking for Static GNN Baselines.

Computes:
1. Classification metrics: Precision, Recall, F1, AUROC, AUPRC, FPR, FAR, Confusion Matrix
2. Early Warning Lead Time:
   lead_time = t_failure - t_warning (only when t_warning <= t_failure)
3. Structured prediction records conforming to Phase 10 / Section 17 schema
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch_geometric
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from ml.baselines.rule_based.evaluator import (
    compute_classification_metrics,
    compute_lead_time_metrics,
)


def evaluate_static_gnn(
    model: Any,
    test_loader: Any,
    threshold: float = 0.50,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate trained static GNN model on held-out test split.
    
    Args:
        model: Trained PyTorch Geometric GNN model (GCN or GAT).
        test_loader: PyG DataLoader yielding batched graph snapshots.
        threshold: Frozen decision cutoff calibrated exclusively on validation split.
        device: Torch compute device.
        
    Returns:
        Dictionary containing classification metrics, lead times, and prediction records.
    """
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric required for static GNN evaluation.")

    if device is not None:
        dev = device
    else:
        params = list(model.parameters())
        dev = params[0].device if params else torch.device("cpu")
    model.eval()

    all_probs: List[float] = []
    all_trues: List[int] = []
    all_metas: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(dev)
            edge_attr = getattr(batch, "edge_attr", None)
            edge_weight = getattr(batch, "edge_weight", None)

            logits = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                edge_attr=edge_attr,
                edge_weight=edge_weight,
            )
            labels = batch.y.squeeze() if batch.y.dim() > 1 else batch.y
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

            all_probs.extend(probs if isinstance(probs, list) else [probs])
            trues = labels.cpu().numpy().astype(int).tolist()
            all_trues.extend(trues if isinstance(trues, list) else [trues])

            batch_size = len(probs) if isinstance(probs, list) else 1
            for b_idx in range(batch_size):
                # Extract sample metadata cleanly
                s_id = batch.sample_id[b_idx] if hasattr(batch, "sample_id") and isinstance(batch.sample_id, (list, tuple)) else str(b_idx)
                r_id = batch.run_id[b_idx] if hasattr(batch, "run_id") and isinstance(batch.run_id, (list, tuple)) else "unknown"
                
                if hasattr(batch, "timestamp"):
                    ts_val = batch.timestamp[b_idx].item() if hasattr(batch.timestamp[b_idx], "item") else float(batch.timestamp[b_idx])
                else:
                    ts_val = 0.0

                if hasattr(batch, "prediction_horizon"):
                    h_val = int(batch.prediction_horizon[b_idx].item() if hasattr(batch.prediction_horizon[b_idx], "item") else batch.prediction_horizon[b_idx])
                else:
                    h_val = 1

                all_metas.append({
                    "sample_id": s_id,
                    "run_id": r_id,
                    "timestamp": round(ts_val, 4),
                    "prediction_horizon": h_val,
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
            "model": getattr(model, "model_name", "static_gnn"),
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
