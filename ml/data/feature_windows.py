"""Causal Feature Window Extraction for Multi-Agent Failure Prediction.

Extracts three comparative feature views for prediction point t:
1. Agent-level behavioral features (tabular baseline)
2. Static graph snapshot G(t) = (V(t), E(t), X(t))
3. Temporal graph history [G(t-n), ..., G(t)]

STRICT CAUSALITY GUARANTEE:
All features are computed exclusively using telemetry events where:
step_idx <= current_step_idx AND timestamp <= current_timestamp.
Future events are never accessed during feature construction.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from ml.telemetry.schemas import AgentTelemetryEvent
from ml.graph.graph_snapshot import GraphSnapshot
from ml.graph.graph_builder import TemporalGraphBuilder
from ml.utils.reproducibility import verify_no_future_leakage


def extract_agent_level_features(
    events: List[Any],
    current_step_idx: int,
    current_timestamp: float,
) -> Dict[str, float]:
    """Compute aggregate behavioral features up to prediction point t.
    
    Args:
        events: Historical events sequence up to current step.
        current_step_idx: Current step index cutoff.
        current_timestamp: Current time cutoff.
        
    Returns:
        Dictionary of numerical behavioral features.
    """
    # Strict causal filtering
    causal_events = [
        e for e in events
        if (getattr(e, "step_idx", None) or 0) <= current_step_idx
        and (getattr(e, "timestamp", None) or 0.0) <= current_timestamp
    ]

    # Verify no future leakage
    used_steps = [getattr(e, "step_idx", 0) or 0 for e in causal_events]
    used_times = [getattr(e, "timestamp", 0.0) or 0.0 for e in causal_events]
    verify_no_future_leakage(
        current_step=current_step_idx,
        current_timestamp=current_timestamp,
        used_event_steps=used_steps,
        used_event_timestamps=used_times,
    )

    if not causal_events:
        return {
            "total_events_observed": 0.0,
            "mean_output_quality": 1.0,
            "min_output_quality": 1.0,
            "mean_confidence": 1.0,
            "min_confidence": 1.0,
            "total_token_count": 0.0,
            "total_latency": 0.0,
            "mean_latency": 0.0,
            "max_latency": 0.0,
            "total_retries": 0.0,
            "mean_contradiction_score": 0.0,
            "max_contradiction_score": 0.0,
            "tool_call_count": 0.0,
            "tool_failure_count": 0.0,
            "agent_count_active": 0.0,
            "error_count": 0.0,
            "interaction_density": 0.0,
        }

    qualities = [getattr(e, "output_quality", 1.0) or 1.0 for e in causal_events]
    confidences = [getattr(e, "confidence", 1.0) or 1.0 for e in causal_events]
    latencies = [getattr(e, "latency", 0.0) or 0.0 for e in causal_events]
    contradictions = [getattr(e, "contradiction_score", 0.0) or 0.0 for e in causal_events]
    tokens = [getattr(e, "token_count", 0) or 0 for e in causal_events]
    retries = sum(getattr(e, "retry_count", 0) or 0 for e in causal_events)

    tool_calls = sum(1 for e in causal_events if getattr(e, "tool_used", None) or getattr(e, "event_type", "") == "tool_call")
    tool_failures = sum(1 for e in causal_events if getattr(e, "tool_error", False) or (getattr(e, "tool_used", None) and getattr(e, "tool_success", True) is False))
    errors = sum(1 for e in causal_events if getattr(e, "event_type", "") in ("error", "failure") or getattr(e, "error_type", None) is not None)

    active_agents = set()
    for e in causal_events:
        src = getattr(e, "source_agent", None)
        tgt = getattr(e, "target_agent", None)
        if src:
            active_agents.add(src)
        if tgt:
            active_agents.add(tgt)

    n_active = max(1, len(active_agents))
    interaction_density = len(causal_events) / float(n_active)

    return {
        "total_events_observed": float(len(causal_events)),
        "mean_output_quality": float(np.mean(qualities)),
        "min_output_quality": float(np.min(qualities)),
        "mean_confidence": float(np.mean(confidences)),
        "min_confidence": float(np.min(confidences)),
        "total_token_count": float(sum(tokens)),
        "total_latency": float(sum(latencies)),
        "mean_latency": float(np.mean(latencies)),
        "max_latency": float(np.max(latencies)),
        "total_retries": float(retries),
        "mean_contradiction_score": float(np.mean(contradictions)),
        "max_contradiction_score": float(np.max(contradictions)),
        "tool_call_count": float(tool_calls),
        "tool_failure_count": float(tool_failures),
        "agent_count_active": float(len(active_agents)),
        "error_count": float(errors),
        "interaction_density": float(interaction_density),
    }


def extract_graph_features(
    events: List[Any],
    current_step_idx: int,
    current_timestamp: float,
    run_id: str,
    builder: Optional[TemporalGraphBuilder] = None,
    history_length: int = 5,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """Extract causal static snapshot G(t) features and temporal history sequence.
    
    Args:
        events: Historical events up to t.
        current_step_idx: Step index cutoff.
        current_timestamp: Timestamp cutoff.
        run_id: Simulation run ID.
        builder: Optional TemporalGraphBuilder instance.
        history_length: Number of historical graph snapshots.
        
    Returns:
        Tuple of:
            - node_features: Dict[agent_id -> Dict[feat_name, val]]
            - edge_features: Dict["src->tgt" -> Dict[feat_name, val]]
            - temporal_graph_history: List of serialized snapshot dicts
    """
    b = builder or TemporalGraphBuilder()

    causal_events = [
        e for e in events
        if (getattr(e, "step_idx", None) or 0) <= current_step_idx
        and (getattr(e, "timestamp", None) or 0.0) <= current_timestamp
    ]

    snapshot = b.build_snapshot(
        events=causal_events,
        timestamp=current_timestamp,
        step_idx=current_step_idx,
        run_id=run_id,
    )

    # Edge features indexed as string "src->dst" for JSON/Parquet compatibility
    serialized_edges: Dict[str, Dict[str, float]] = {}
    for (src, dst), feats in snapshot.edges.items():
        edge_key = f"{src}->{dst}"
        serialized_edges[edge_key] = feats

    # Extract temporal graph history [G(t-n), ..., G(t)]
    history_snaps = b.build_temporal_history(
        events=causal_events,
        current_timestamp=current_timestamp,
        history_length=history_length,
    )
    serialized_history = [s.to_dict() for s in history_snaps]

    return snapshot.nodes, serialized_edges, serialized_history
