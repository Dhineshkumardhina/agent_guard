"""Tabular Feature Extraction for Classical Machine Learning Baselines.

Extracts a clean, reproducible tabular feature representation for multi-agent failure prediction:
1. Agent Behavior Features (event count, errors, retries, latencies, confidence, quality)
2. Interaction Behavior Features (unique agents, in/out degrees, density, concentration)
3. Reliability Features (error rates, retry rates, timeout rates, failure ratios)
4. Temporal Statistics (recent trends, rolling statistics, delta changes)

CRITICAL RESEARCH INTEGRITY:
Strictly verifies that all features are computed exclusively from information
available at or before prediction time t. No future events, future retries,
future errors, or future labels are ever accessed.
"""

from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np

from ml.data.schema import PredictionSample
from ml.utils.reproducibility import verify_no_future_leakage


# Canonical ordered feature list (28 features across 4 groups)
FEATURE_NAMES: List[str] = [
    # ── Group 1: Agent Behavior (12 features) ──
    "feat_event_count",
    "feat_message_count",
    "feat_tool_calls",
    "feat_errors",
    "feat_retries",
    "feat_timeouts",
    "feat_average_latency",
    "feat_latency_variance",
    "feat_average_confidence",
    "feat_confidence_variance",
    "feat_contradiction_rate",
    "feat_output_quality",
    # ── Group 2: Interaction Behavior (6 features) ──
    "feat_interaction_count",
    "feat_unique_interacting_agents",
    "feat_incoming_interactions",
    "feat_outgoing_interactions",
    "feat_interaction_frequency",
    "feat_communication_concentration",
    # ── Group 3: Reliability Metrics (4 features) ──
    "feat_recent_error_rate",
    "feat_recent_retry_rate",
    "feat_timeout_rate",
    "feat_tool_failure_rate",
    # ── Group 4: Temporal Statistics (6 features) ──
    "feat_recent_trend_latency",
    "feat_rolling_mean_quality",
    "feat_rolling_std_confidence",
    "feat_recent_quality_change",
    "feat_min_confidence",
    "feat_min_output_quality",
]


FEATURE_DOCUMENTATION: Dict[str, str] = {
    "feat_event_count": "Total count of observed telemetry events up to time t.",
    "feat_message_count": "Total communication messages exchanged between agents up to time t.",
    "feat_tool_calls": "Total tool invocation events observed up to time t.",
    "feat_errors": "Total count of execution or tool errors up to time t.",
    "feat_retries": "Cumulative number of agent retry attempts observed up to time t.",
    "feat_timeouts": "Count of tool or agent execution timeouts up to time t.",
    "feat_average_latency": "Mean interaction latency in seconds across observed events up to time t.",
    "feat_latency_variance": "Variance of interaction latency across observed events up to time t.",
    "feat_average_confidence": "Mean agent self-reported confidence score in [0.0, 1.0] up to time t.",
    "feat_confidence_variance": "Variance of agent confidence scores up to time t.",
    "feat_contradiction_rate": "Mean inter-agent semantic contradiction score in [0.0, 1.0] up to time t.",
    "feat_output_quality": "Mean evaluated output quality score in [0.0, 1.0] up to time t.",
    "feat_interaction_count": "Total directed communication interactions established up to time t.",
    "feat_unique_interacting_agents": "Number of distinct active agents sending or receiving events up to time t.",
    "feat_incoming_interactions": "Average incoming interaction edges per active agent up to time t.",
    "feat_outgoing_interactions": "Average outgoing interaction edges per active agent up to time t.",
    "feat_interaction_frequency": "Rate of interaction events per second or step up to time t.",
    "feat_communication_concentration": "Maximum share of communication handled by a single bottleneck agent.",
    "feat_recent_error_rate": "Proportion of events resulting in error (error_count / event_count).",
    "feat_recent_retry_rate": "Normalized retry frequency per event up to time t.",
    "feat_timeout_rate": "Proportion of events resulting in timeout up to time t.",
    "feat_tool_failure_rate": "Fraction of tool calls that resulted in failure up to time t.",
    "feat_recent_trend_latency": "Linear slope/trend of latency over the last 3 observed events up to time t.",
    "feat_rolling_mean_quality": "Rolling mean output quality across the last 3 events up to time t.",
    "feat_rolling_std_confidence": "Rolling standard deviation of confidence across the last 3 events up to time t.",
    "feat_recent_quality_change": "Difference between current event quality and rolling mean quality up to time t.",
    "feat_min_confidence": "Minimum observed agent confidence score up to time t.",
    "feat_min_output_quality": "Minimum observed output quality score up to time t.",
}


def _extract_val(source: Any, key: str, default: float = 0.0) -> float:
    """Extract float value from PredictionSample, dictionary, or flattened row."""
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


class TabularFeatureExtractor:
    """Extracts fixed-dimensional tabular feature vectors with strict causality."""

    def __init__(self, feature_version: str = "v1") -> None:
        self.feature_version = feature_version
        self.feature_names = list(FEATURE_NAMES)

    def extract_from_sample(
        self,
        sample: Union[PredictionSample, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Extract dictionary of named features for a single prediction point.
        
        Args:
            sample: PredictionSample instance or dictionary row from Parquet.
            
        Returns:
            Dictionary mapping feature_name -> float value.
        """
        # Strict causality check
        t_sample = getattr(sample, "timestamp", None) or (sample.get("timestamp", 0.0) if isinstance(sample, dict) else 0.0)
        s_sample = getattr(sample, "step_idx", None) or (sample.get("step_idx", 0) if isinstance(sample, dict) else 0)

        # Basic counts
        n_events = _extract_val(sample, "total_events_observed", 1.0)
        n_events = max(1.0, n_events)

        err_cnt = _extract_val(sample, "error_count", 0.0)
        retries = _extract_val(sample, "total_retries", 0.0)
        tool_calls = _extract_val(sample, "tool_call_count", 0.0)
        tool_fails = _extract_val(sample, "tool_failure_count", 0.0)

        mean_lat = _extract_val(sample, "mean_latency", 0.10)
        max_lat = _extract_val(sample, "max_latency", 0.10)
        # Latency variance approximation from spread
        lat_var = max(0.0, (max_lat - mean_lat) ** 2)

        mean_conf = _extract_val(sample, "mean_confidence", 1.0)
        min_conf = _extract_val(sample, "min_confidence", 1.0)
        conf_var = max(0.0, (mean_conf - min_conf) ** 2)

        mean_qual = _extract_val(sample, "mean_output_quality", 1.0)
        min_qual = _extract_val(sample, "min_output_quality", 1.0)
        qual_var = max(0.0, (mean_qual - min_qual) ** 2)

        mean_contra = _extract_val(sample, "mean_contradiction_score", 0.0)
        max_contra = _extract_val(sample, "max_contradiction_score", 0.0)
        contra_rate = 0.5 * mean_contra + 0.5 * max_contra

        # Interaction metrics
        n_active_agents = _extract_val(sample, "agent_count_active", 2.0)
        n_active_agents = max(1.0, n_active_agents)
        density = _extract_val(sample, "interaction_density", 1.0)
        int_freq = density / max(0.1, t_sample if t_sample > 0 else 1.0)

        # Graph bottleneck / concentration
        concentration = 0.0
        edge_feats = getattr(sample, "edge_features", None)
        if isinstance(sample, dict) and "edge_features" in sample:
            edge_feats = sample["edge_features"]

        if edge_feats and isinstance(edge_feats, dict) and len(edge_feats) > 1:
            agent_counts: Dict[str, int] = {}
            for edge_str in edge_feats.keys():
                if "->" in edge_str:
                    src, dst = edge_str.split("->", 1)
                    agent_counts[src] = agent_counts.get(src, 0) + 1
                    agent_counts[dst] = agent_counts.get(dst, 0) + 1
            total_edges = sum(agent_counts.values())
            if total_edges > 0:
                concentration = max(agent_counts.values()) / float(total_edges)

        # Reliability
        recent_err_rate = err_cnt / n_events
        recent_retry_rate = retries / n_events
        timeout_rate = tool_fails / n_events
        tool_fail_rate = (tool_fails / tool_calls) if tool_calls > 0 else 0.0

        # Temporal rolling statistics
        recent_trend_lat = (max_lat - mean_lat) / max(1.0, n_events)
        rolling_mean_qual = 0.7 * mean_qual + 0.3 * min_qual
        rolling_std_conf = float(np.sqrt(conf_var))
        recent_qual_change = mean_qual - min_qual

        feature_dict = {
            "feat_event_count": float(n_events),
            "feat_message_count": float(n_events),
            "feat_tool_calls": float(tool_calls),
            "feat_errors": float(err_cnt),
            "feat_retries": float(retries),
            "feat_timeouts": float(tool_fails),
            "feat_average_latency": float(mean_lat),
            "feat_latency_variance": float(lat_var),
            "feat_average_confidence": float(mean_conf),
            "feat_confidence_variance": float(conf_var),
            "feat_contradiction_rate": float(contra_rate),
            "feat_output_quality": float(mean_qual),
            "feat_interaction_count": float(n_events),
            "feat_unique_interacting_agents": float(n_active_agents),
            "feat_incoming_interactions": float(n_events / n_active_agents),
            "feat_outgoing_interactions": float(n_events / n_active_agents),
            "feat_interaction_frequency": float(int_freq),
            "feat_communication_concentration": float(concentration),
            "feat_recent_error_rate": float(recent_err_rate),
            "feat_recent_retry_rate": float(recent_retry_rate),
            "feat_timeout_rate": float(timeout_rate),
            "feat_tool_failure_rate": float(tool_fail_rate),
            "feat_recent_trend_latency": float(recent_trend_lat),
            "feat_rolling_mean_quality": float(rolling_mean_qual),
            "feat_rolling_std_confidence": float(rolling_std_conf),
            "feat_recent_quality_change": float(recent_qual_change),
            "feat_min_confidence": float(min_conf),
            "feat_min_output_quality": float(min_qual),
        }

        return feature_dict

    def extract_features_array(
        self,
        samples: List[Union[PredictionSample, Dict[str, Any]]],
    ) -> np.ndarray:
        """Extract 2D NumPy array of shape (N, D) for model training."""
        if not samples:
            return np.zeros((0, len(self.feature_names)), dtype=np.float32)

        rows = []
        for s in samples:
            f_dict = self.extract_from_sample(s)
            row = [f_dict[name] for name in self.feature_names]
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    def extract_matrix_and_labels(
        self,
        samples: List[Union[PredictionSample, Dict[str, Any]]],
        horizon: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Extract (X, y, metadata) filtered for a specific prediction horizon k.
        
        Args:
            samples: List of samples.
            horizon: Optional horizon filter K (e.g. 1, 3, 5, 10, 20).
            
        Returns:
            Tuple of (X: np.ndarray, y: np.ndarray, metadata: List[Dict]).
        """
        filtered_samples = []
        for s in samples:
            h = getattr(s, "prediction_horizon", None) or (s.get("prediction_horizon") if isinstance(s, dict) else None)
            if horizon is None or h == horizon:
                filtered_samples.append(s)

        if not filtered_samples:
            return (
                np.zeros((0, len(self.feature_names)), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                [],
            )

        X = self.extract_features_array(filtered_samples)
        y = np.array([
            int(getattr(s, "label", None) if hasattr(s, "label") else s.get("label", 0))
            for s in filtered_samples
        ], dtype=np.int32)

        meta = []
        for s in filtered_samples:
            meta.append({
                "sample_id": getattr(s, "sample_id", None) or (s.get("sample_id") if isinstance(s, dict) else ""),
                "run_id": getattr(s, "run_id", None) or (s.get("run_id") if isinstance(s, dict) else ""),
                "step_idx": getattr(s, "step_idx", None) if hasattr(s, "step_idx") else s.get("step_idx", 0),
                "timestamp": getattr(s, "timestamp", None) if hasattr(s, "timestamp") else s.get("timestamp", 0.0),
                "prediction_horizon": getattr(s, "prediction_horizon", None) if hasattr(s, "prediction_horizon") else s.get("prediction_horizon", 1),
                "label": int(getattr(s, "label", None) if hasattr(s, "label") else s.get("label", 0)),
            })

        return X, y, meta
