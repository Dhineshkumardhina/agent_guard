"""Transparent Causal Risk Indicators for Rule-Based Early Warning.

Computes 9 individual risk signals strictly from observations <= prediction time t:
1. error_rate: Proportion of observed interactions resulting in execution or tool errors
2. retry_rate: Normalized count of agent retry events
3. timeout_rate: Rate of tool timeouts or dropped interactions
4. contradiction_rate: Inter-agent factual divergence and contradiction score
5. confidence_degradation: Decline in agent self-reported confidence
6. latency_anomaly: Abnormal deviation from baseline execution latency
7. frequency_anomaly: Message density / communication loop thrashing
8. agent_failure_history: Past quality degradation across participating agents
9. interaction_concentration: Communication bottlenecking around a central agent

CRITICAL RESEARCH INTEGRITY:
All indicators strictly process information <= prediction time t.
No future events, future labels, or future outcomes are ever accessed.
"""

from typing import Dict, Any, Union, List, Optional
import numpy as np

from ml.data.schema import PredictionSample


def _get_val(source: Any, key: str, default: float = 0.0) -> float:
    """Helper to extract a numerical field from PredictionSample, dictionary, or flat row."""
    if hasattr(source, "agent_level_features") and isinstance(source.agent_level_features, dict):
        if key in source.agent_level_features:
            return float(source.agent_level_features[key])
    
    if isinstance(source, dict):
        if key in source:
            return float(source[key])
        flat_key = f"agent_feat_{key}"
        if flat_key in source:
            return float(source[flat_key])
        
    return default


def compute_error_rate(source: Any) -> float:
    """Compute recent error rate indicator in [0.0, 1.0]."""
    err_cnt = _get_val(source, "error_count", 0.0)
    total_ev = _get_val(source, "total_events_observed", 1.0)
    if total_ev <= 0.0:
        return 0.0
    return float(np.clip(err_cnt / total_ev, 0.0, 1.0))


def compute_retry_rate(source: Any) -> float:
    """Compute normalized retry rate indicator in [0.0, 1.0]."""
    retries = _get_val(source, "total_retries", 0.0)
    total_ev = _get_val(source, "total_events_observed", 1.0)
    if total_ev <= 0.0 or retries <= 0.0:
        return 0.0
    # Normalized: 1 retry per event is maximum anomaly (1.0)
    return float(np.clip(retries / total_ev, 0.0, 1.0))


def compute_timeout_rate(source: Any) -> float:
    """Compute tool timeout and failure rate indicator in [0.0, 1.0]."""
    timeouts = _get_val(source, "tool_failure_count", 0.0)
    total_ev = _get_val(source, "total_events_observed", 1.0)
    if total_ev <= 0.0:
        return 0.0
    return float(np.clip(timeouts / total_ev, 0.0, 1.0))


def compute_contradiction_rate(source: Any) -> float:
    """Compute inter-agent contradiction indicator in [0.0, 1.0]."""
    mean_c = _get_val(source, "mean_contradiction_score", 0.0)
    max_c = _get_val(source, "max_contradiction_score", 0.0)
    # Blend mean and peak contradiction
    score = 0.6 * max_c + 0.4 * mean_c
    return float(np.clip(score, 0.0, 1.0))


def compute_confidence_degradation(source: Any) -> float:
    """Compute confidence degradation indicator in [0.0, 1.0].
    
    1.0 means complete loss of confidence (confidence = 0.0).
    0.0 means perfect full confidence (confidence = 1.0).
    """
    mean_conf = _get_val(source, "mean_confidence", 1.0)
    min_conf = _get_val(source, "min_confidence", 1.0)
    degrade = max(0.0, 1.0 - (0.5 * mean_conf + 0.5 * min_conf))
    return float(np.clip(degrade, 0.0, 1.0))


def compute_latency_anomaly(source: Any) -> float:
    """Compute execution latency spike indicator in [0.0, 1.0].
    
    Baseline expected latency is ~0.10s. Latencies > 0.40s indicate severe congestion.
    """
    mean_lat = _get_val(source, "mean_latency", 0.10)
    max_lat = _get_val(source, "max_latency", 0.10)
    # Normalize relative to expected 0.10s baseline
    excess = max(0.0, mean_lat - 0.12) / 0.35 + max(0.0, max_lat - 0.20) / 0.50
    return float(np.clip(0.5 * excess, 0.0, 1.0))


def compute_frequency_anomaly(source: Any) -> float:
    """Compute message density / communication thrashing indicator in [0.0, 1.0]."""
    density = _get_val(source, "interaction_density", 1.0)
    # Typical density per agent is ~1-2. Density > 3 indicates thrashing / loops
    if density <= 1.5:
        return 0.0
    excess = (density - 1.5) / 3.0
    return float(np.clip(excess, 0.0, 1.0))


def compute_agent_failure_history(source: Any) -> float:
    """Compute historical agent output degradation indicator in [0.0, 1.0]."""
    min_qual = _get_val(source, "min_output_quality", 1.0)
    mean_qual = _get_val(source, "mean_output_quality", 1.0)
    # Quality drops below 0.8 indicate historical degradation
    loss = (1.0 - min_qual) * 0.7 + (1.0 - mean_qual) * 0.3
    return float(np.clip(loss, 0.0, 1.0))


def compute_interaction_concentration(source: Any) -> float:
    """Compute topological communication bottlenecking indicator in [0.0, 1.0]."""
    # Check if edge_features or graph summary available
    edge_feats = getattr(source, "edge_features", None)
    if isinstance(source, dict) and "edge_features" in source:
        edge_feats = source["edge_features"]

    if edge_feats and isinstance(edge_feats, dict) and len(edge_feats) > 1:
        agent_counts: Dict[str, int] = {}
        for edge_str in edge_feats.keys():
            if "->" in edge_str:
                src, dst = edge_str.split("->", 1)
                agent_counts[src] = agent_counts.get(src, 0) + 1
                agent_counts[dst] = agent_counts.get(dst, 0) + 1
        
        total_interactions = sum(agent_counts.values())
        if total_interactions > 0:
            max_share = max(agent_counts.values()) / float(total_interactions)
            # If a single agent handles > 40% of interaction endpoints
            if max_share > 0.35:
                return float(np.clip((max_share - 0.35) / 0.50, 0.0, 1.0))

    return 0.0


def compute_all_indicators(source: Any) -> Dict[str, float]:
    """Compute all 9 risk indicators for a prediction point.
    
    Args:
        source: PredictionSample instance, dictionary, or flat DataFrame row.
        
    Returns:
        Dictionary mapping indicator name -> normalized float value in [0.0, 1.0].
    """
    return {
        "error_rate": compute_error_rate(source),
        "retry_rate": compute_retry_rate(source),
        "timeout_rate": compute_timeout_rate(source),
        "contradiction_rate": compute_contradiction_rate(source),
        "confidence_degradation": compute_confidence_degradation(source),
        "latency_anomaly": compute_latency_anomaly(source),
        "frequency_anomaly": compute_frequency_anomaly(source),
        "agent_failure_history": compute_agent_failure_history(source),
        "interaction_concentration": compute_interaction_concentration(source),
    }
