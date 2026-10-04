"""Temporal Graph Dataset Builder, Interaction Streams, and Diagnostics - Phase 11.

Constructs chronological interaction event streams and temporal prediction points
grouped per simulation trajectory for TGN-style processing.

STRICT CAUSALITY GUARANTEE:
All interactions are ordered chronologically by timestamp.
For any prediction point at time t, only interactions observed <= t are processed.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import torch

from ml.baselines.static_gnn.dataset import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES


@dataclass
class TemporalInteraction:
    """Represents a single directed chronological interaction event u -> v at timestamp t."""
    source_agent: str
    target_agent: str
    timestamp: float
    step_idx: int
    features: List[float]  # 10-dim vector conforming to EDGE_FEATURE_NAMES


@dataclass
class TemporalPredictionPoint:
    """Represents an impending failure prediction evaluation point at time t."""
    sample_id: str
    run_id: str
    step_idx: int
    timestamp: float
    prediction_horizon: int
    label: float
    active_agents: List[str]
    node_features: Dict[str, List[float]]  # Mapping agent_id -> 14-dim node feature vector


@dataclass
class TemporalRunTrajectory:
    """Encapsulates all chronological interactions and causal prediction points for one run."""
    run_id: str
    topology: str
    interactions: List[TemporalInteraction] = field(default_factory=list)
    prediction_points: List[TemporalPredictionPoint] = field(default_factory=list)


def extract_node_feature_vector(node_attrs: Dict[str, Any], incoming_cnt: float = 0.0, outgoing_cnt: float = 0.0) -> List[float]:
    """Format dictionary into standard 14-dimensional node feature vector."""
    return [
        float(node_attrs.get("event_count", 0.0)),
        float(node_attrs.get("message_count", 0.0)),
        float(node_attrs.get("tool_call_count", 0.0)),
        float(node_attrs.get("error_count", 0.0)),
        float(node_attrs.get("retry_count", 0.0)),
        float(node_attrs.get("timeout_count", 0.0)),
        float(node_attrs.get("average_latency", 0.0)),
        float(node_attrs.get("average_confidence", 1.0)),
        float(node_attrs.get("average_output_quality", 1.0)),
        float(node_attrs.get("contradiction_rate", 0.0)),
        float(node_attrs.get("recent_failure_count", 0.0)),
        float(incoming_cnt),
        float(outgoing_cnt),
        1.0 if node_attrs.get("is_active", True) else 0.0,
    ]


def extract_edge_feature_vector(edge_attrs: Dict[str, Any]) -> List[float]:
    """Format dictionary into standard 10-dimensional edge feature vector."""
    return [
        float(edge_attrs.get("interaction_count", 1.0)),
        float(edge_attrs.get("message_count", 0.0)),
        float(edge_attrs.get("average_latency", 0.0)),
        float(edge_attrs.get("average_message_length", 0.0)),
        float(edge_attrs.get("average_confidence", 1.0)),
        float(edge_attrs.get("retry_count", 0.0)),
        float(edge_attrs.get("contradiction_rate", 0.0)),
        float(edge_attrs.get("error_count", 0.0)),
        float(edge_attrs.get("timeout_count", 0.0)),
        float(edge_attrs.get("interaction_frequency", 0.0)),
    ]


class TemporalDatasetBuilder:
    """Builds trajectory-grouped temporal event streams and prediction points."""

    def build_trajectories(
        self,
        tabular_samples: List[Dict[str, Any]],
        graph_sequences: Dict[str, List[Dict[str, Any]]],
        horizon_filter: Optional[int] = None,
    ) -> List[TemporalRunTrajectory]:
        """Convert dataset partitions into list of TemporalRunTrajectory instances.
        
        Args:
            tabular_samples: Tabular sample rows.
            graph_sequences: Mapping sample_id -> temporal graph snapshots history.
            horizon_filter: Optional integer horizon to filter prediction points (e.g. 1, 3, 5, 10, 20).
            
        Returns:
            List of TemporalRunTrajectory instances ordered by run_id.
        """
        # Group tabular samples by run_id
        samples_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in tabular_samples:
            if horizon_filter is not None and int(s.get("prediction_horizon", 1)) != horizon_filter:
                continue
            samples_by_run[s["run_id"]].append(s)

        trajectories: List[TemporalRunTrajectory] = []

        for run_id in sorted(samples_by_run.keys()):
            run_samples = samples_by_run[run_id]
            # Sort prediction points by (timestamp, step_idx)
            run_samples.sort(key=lambda s: (float(s.get("timestamp", 0.0)), int(s.get("step_idx", 0))))
            topology = run_samples[0].get("topology", "unknown")

            # Collect all distinct snapshots across samples of this run to extract chronological interactions
            # Use max-step sample to get complete available sequence
            max_step_sample = max(run_samples, key=lambda s: int(s.get("step_idx", 0)))
            snaps_history = graph_sequences.get(max_step_sample["sample_id"], [])

            # Extract distinct chronological interactions from the snapshot sequence
            interactions: List[TemporalInteraction] = []
            seen_interactions = set()

            for step_idx, snap in enumerate(snaps_history):
                snap_t = float(snap.get("timestamp", 0.0))
                edges = snap.get("edges", {})
                for edge_key, attrs in edges.items():
                    src = attrs.get("source_agent", "")
                    tgt = attrs.get("target_agent", "")
                    if not src and "->" in edge_key:
                        src, tgt = edge_key.split("->", 1)

                    # Deduplicate interaction occurrences by (src, tgt, timestamp)
                    interaction_id = (src, tgt, round(snap_t, 4))
                    if interaction_id not in seen_interactions and src and tgt:
                        seen_interactions.add(interaction_id)
                        feat_vec = extract_edge_feature_vector(attrs)
                        interactions.append(TemporalInteraction(
                            source_agent=src,
                            target_agent=tgt,
                            timestamp=snap_t,
                            step_idx=step_idx,
                            features=feat_vec,
                        ))

            # Ensure interactions are sorted strictly by timestamp
            interactions.sort(key=lambda ev: (ev.timestamp, ev.step_idx))

            # Build prediction points
            pred_points: List[TemporalPredictionPoint] = []
            for s in run_samples:
                s_id = s.get("sample_id", "")
                seq = graph_sequences.get(s_id, [])
                last_snap = seq[-1] if seq else {}

                snap_nodes = last_snap.get("nodes", {})
                active_agents = sorted(list(snap_nodes.keys()))
                if not active_agents:
                    active_agents = [f"agent_{i}" for i in range(int(s.get("number_of_agents", 3)))]

                # Compute incoming/outgoing counts
                in_cnts: Dict[str, float] = defaultdict(float)
                out_cnts: Dict[str, float] = defaultdict(float)
                for e_key, e_attrs in last_snap.get("edges", {}).items():
                    u = e_attrs.get("source_agent", "")
                    v = e_attrs.get("target_agent", "")
                    if not u and "->" in e_key:
                        u, v = e_key.split("->", 1)
                    cnt = float(e_attrs.get("interaction_count", 1.0))
                    out_cnts[u] += cnt
                    in_cnts[v] += cnt

                node_feats: Dict[str, List[float]] = {}
                for aid in active_agents:
                    attrs = snap_nodes.get(aid, {})
                    node_feats[aid] = extract_node_feature_vector(attrs, in_cnts[aid], out_cnts[aid])

                p_point = TemporalPredictionPoint(
                    sample_id=s_id,
                    run_id=run_id,
                    step_idx=int(s.get("step_idx", 0)),
                    timestamp=float(s.get("timestamp", 0.0)),
                    prediction_horizon=int(s.get("prediction_horizon", 1)),
                    label=float(s.get("label", 0.0)),
                    active_agents=active_agents,
                    node_features=node_feats,
                )
                pred_points.append(p_point)

            trajectories.append(TemporalRunTrajectory(
                run_id=run_id,
                topology=topology,
                interactions=interactions,
                prediction_points=pred_points,
            ))

        return trajectories


def compute_temporal_diagnostics(trajectories: List[TemporalRunTrajectory]) -> Dict[str, Any]:
    """Compute and format comprehensive temporal diagnostics across multi-agent runs."""
    total_runs = len(trajectories)
    if total_runs == 0:
        return {}

    total_events = 0
    all_agents = set()
    events_per_run = []
    deltas: List[float] = []
    total_prediction_points = 0
    pos_count = 0
    neg_count = 0

    for traj in trajectories:
        n_events = len(traj.interactions)
        total_events += n_events
        events_per_run.append(n_events)

        # Compute delta t between consecutive interactions
        prev_t = 0.0
        for idx, ev in enumerate(traj.interactions):
            all_agents.add(ev.source_agent)
            all_agents.add(ev.target_agent)
            if idx > 0:
                dt = max(0.0, ev.timestamp - prev_t)
                deltas.append(dt)
            prev_t = ev.timestamp

        for pp in traj.prediction_points:
            total_prediction_points += 1
            if pp.label == 1.0:
                pos_count += 1
            else:
                neg_count += 1

    return {
        "number_of_runs": total_runs,
        "total_interaction_events": total_events,
        "unique_agents_count": len(all_agents),
        "average_interactions_per_run": round(float(np.mean(events_per_run)), 4) if events_per_run else 0.0,
        "temporal_gap_distribution": {
            "mean_delta_t": round(float(np.mean(deltas)), 4) if deltas else 0.0,
            "median_delta_t": round(float(np.median(deltas)), 4) if deltas else 0.0,
            "min_delta_t": round(float(np.min(deltas)), 4) if deltas else 0.0,
            "max_delta_t": round(float(np.max(deltas)), 4) if deltas else 0.0,
        },
        "average_memory_updates_per_run": round(float(total_events * 2 / max(1, total_runs)), 4),  # both src & tgt update
        "total_prediction_points": total_prediction_points,
        "class_distribution": {
            "positive_samples": pos_count,
            "negative_samples": neg_count,
            "positive_ratio": round(pos_count / max(1, total_prediction_points), 4),
        },
    }


def debug_memory_trace(
    model: Any,
    trajectory: TemporalRunTrajectory,
    target_sample_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Produce chronological debugging trace of memory updates and temporal states for a run.
    
    Verifies that events are strictly processed chronologically.
    """
    model.eval()
    model.reset_memory()
    trace: List[Dict[str, Any]] = []

    # Process each interaction up to prediction point
    target_time = float("inf")
    if target_sample_id:
        for pp in trajectory.prediction_points:
            if pp.sample_id == target_sample_id:
                target_time = pp.timestamp
                break

    for ev in trajectory.interactions:
        if ev.timestamp > target_time:
            break

        src = ev.source_agent
        tgt = ev.target_agent
        last_src_t = model.node_memory.get_last_timestamp(src)
        dt = max(0.0, ev.timestamp - last_src_t)

        e_tensor = torch.tensor(ev.features, dtype=torch.float32, device=model.device)
        new_src_mem, new_tgt_mem = model.process_interaction(
            source_agent=src,
            target_agent=tgt,
            timestamp=ev.timestamp,
            edge_features=e_tensor,
        )

        trace.append({
            "timestamp": round(ev.timestamp, 4),
            "source_agent": src,
            "target_agent": tgt,
            "temporal_delta": round(float(dt), 4),
            "src_memory_norm": round(float(torch.norm(new_src_mem).item()), 4),
            "tgt_memory_norm": round(float(torch.norm(new_tgt_mem).item()), 4),
        })

    return trace
