"""Temporal Graph Neural Network Explainer - Phase 15.

Provides multi-level attribution for TemporalGraphFailurePredictor:
- Level 2: Feature Attribution (Node, Edge, and Temporal feature masking)
- Level 3: Node / Agent Importance (Multi-agent population attribution via node masking)
- Level 4: Edge / Interaction Importance (Communication channel attribution via edge masking)
- Level 5: Temporal Event Importance (Historical interaction ranking strictly <= t_pred)
- Level 6: Risk Trajectory generation (time -> predicted failure risk)
- Level 8: Signed positive and negative risk contributions

Adheres strictly to causal time ordering (no future leakage).
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import copy
import numpy as np
import torch

from ml.baselines.temporal_gnn.models import TemporalGraphFailurePredictor
from ml.baselines.temporal_gnn.dataset import (
    TemporalRunTrajectory,
    TemporalPredictionPoint,
    TemporalInteraction,
)
from ml.baselines.static_gnn.dataset import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES
from ml.explainability.schema import (
    FeatureAttribution,
    AgentAttribution,
    InteractionAttribution,
    TemporalEventAttribution,
    CAUSALITY_DISCLAIMER,
)


class TemporalGNNExplainer:
    """Interprets predictions of the TemporalGraphFailurePredictor model."""

    def __init__(
        self,
        model: TemporalGraphFailurePredictor,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or model.device or torch.device("cpu")
        self.model.eval()

    def _predict_point(
        self,
        trajectory: TemporalRunTrajectory,
        prediction_point: TemporalPredictionPoint,
        excluded_interactions: Optional[Set[int]] = None,  # Set of step_idx to skip
        masked_agent: Optional[str] = None,  # Agent ID to mask out
        feature_mask_node: Optional[int] = None,  # Node feature index to zero
        feature_mask_edge: Optional[int] = None,  # Edge feature index to zero
        feature_scale_node: Optional[Tuple[int, float]] = None,  # (index, factor)
    ) -> float:
        """Helper to re-run causal simulation stream up to t_pred under targeted perturbation."""
        self.model.reset_memory()
        cutoff_t = prediction_point.timestamp
        excluded = excluded_interactions or set()

        with torch.no_grad():
            for inter in trajectory.interactions:
                if inter.timestamp > cutoff_t:
                    break
                if inter.step_idx in excluded:
                    continue

                e_feat = torch.tensor(inter.features, dtype=torch.float32, device=self.device)
                if feature_mask_edge is not None and feature_mask_edge < len(e_feat):
                    e_feat[feature_mask_edge] = 0.0

                self.model.process_interaction(
                    source_agent=inter.source_agent,
                    target_agent=inter.target_agent,
                    timestamp=inter.timestamp,
                    edge_features=e_feat,
                )

            # Node features dict
            n_feats_dict = copy.deepcopy(prediction_point.node_features)
            active_agents = [a for a in prediction_point.active_agents]

            if masked_agent and masked_agent in active_agents:
                # Zero features and memory for masked agent
                if masked_agent in n_feats_dict:
                    n_feats_dict[masked_agent] = [0.0] * len(n_feats_dict[masked_agent])

            if feature_mask_node is not None:
                for aid, feat_list in n_feats_dict.items():
                    if feature_mask_node < len(feat_list):
                        feat_list[feature_mask_node] = 0.0

            if feature_scale_node is not None:
                idx, factor = feature_scale_node
                for aid, feat_list in n_feats_dict.items():
                    if idx < len(feat_list):
                        feat_list[idx] = feat_list[idx] * factor

            _, prob = self.model.predict_at_timestamp(
                agent_ids=active_agents,
                node_features_dict=n_feats_dict,
                current_timestamp=cutoff_t,
            )
        return float(prob)

    def explain_agents(
        self,
        trajectory: TemporalRunTrajectory,
        prediction_point: TemporalPredictionPoint,
        base_probability: Optional[float] = None,
    ) -> List[AgentAttribution]:
        """Compute Level 3 Agent Importance via node masking ablation."""
        if base_probability is None:
            base_prob = self._predict_point(trajectory, prediction_point)
        else:
            base_prob = base_probability

        active_agents = prediction_point.active_agents
        agent_deltas: Dict[str, float] = {}

        for aid in active_agents:
            prob_without = self._predict_point(trajectory, prediction_point, masked_agent=aid)
            # delta = base_prob - prob_without
            # If prob drops when agent is removed, agent was contributing positively to risk
            agent_deltas[aid] = base_prob - prob_without

        max_delta = max([abs(d) for d in agent_deltas.values()]) if agent_deltas else 1.0
        max_delta = max(max_delta, 1e-6)

        sorted_agents = sorted(agent_deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
        attributions: List[AgentAttribution] = []

        # Gather agent telemetry counts strictly up to cutoff
        agent_event_counts: Dict[str, int] = {aid: 0 for aid in active_agents}
        agent_contradictions: Dict[str, float] = {aid: 0.0 for aid in active_agents}
        agent_retries: Dict[str, int] = {aid: 0 for aid in active_agents}

        for inter in trajectory.interactions:
            if inter.timestamp > prediction_point.timestamp:
                break
            if inter.source_agent in agent_event_counts:
                agent_event_counts[inter.source_agent] += 1
            if inter.target_agent in agent_event_counts:
                agent_event_counts[inter.target_agent] += 1

        for aid in active_agents:
            n_feats = prediction_point.node_features.get(aid, [])
            if len(n_feats) >= 11:
                agent_retries[aid] = int(n_feats[4])  # retry_count
                agent_contradictions[aid] = float(n_feats[9])  # contradiction_rate

        for rank_idx, (aid, delta) in enumerate(sorted_agents, start=1):
            norm_score = float(abs(delta) / max_delta)
            # Infer role from name or ID
            role = aid.split("_")[0].lower() if "_" in aid else aid.lower()
            attributions.append(
                AgentAttribution(
                    agent_id=aid,
                    agent_role=role,
                    attribution_score=round(norm_score, 4),
                    delta_risk=round(delta, 4),
                    signed_contribution=round(delta, 4),
                    event_count=agent_event_counts.get(aid, 0),
                    message_count=agent_event_counts.get(aid, 0),
                    contradiction_rate=round(agent_contradictions.get(aid, 0.0), 4),
                    retry_count=agent_retries.get(aid, 0),
                    rank=rank_idx,
                )
            )

        return attributions

    def explain_interactions(
        self,
        trajectory: TemporalRunTrajectory,
        prediction_point: TemporalPredictionPoint,
        base_probability: Optional[float] = None,
        top_k: int = 6,
    ) -> List[InteractionAttribution]:
        """Compute Level 4 Interaction Importance via directed communication channel masking."""
        if base_probability is None:
            base_prob = self._predict_point(trajectory, prediction_point)
        else:
            base_prob = base_probability

        cutoff_t = prediction_point.timestamp
        # Find all active interaction channels u -> v up to t_pred
        channel_steps: Dict[Tuple[str, str], List[int]] = {}
        channel_telemetry: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for inter in trajectory.interactions:
            if inter.timestamp > cutoff_t:
                break
            pair = (inter.source_agent, inter.target_agent)
            channel_steps.setdefault(pair, []).append(inter.step_idx)
            tel = channel_telemetry.setdefault(pair, {"latencies": [], "contradictions": [], "retries": []})
            if len(inter.features) >= 7:
                tel["latencies"].append(inter.features[2])
                tel["retries"].append(inter.features[5])
                tel["contradictions"].append(inter.features[6])

        attributions: List[InteractionAttribution] = []
        if not channel_steps:
            return attributions

        channel_deltas: Dict[Tuple[str, str], float] = {}
        for pair, step_list in channel_steps.items():
            prob_without = self._predict_point(
                trajectory,
                prediction_point,
                excluded_interactions=set(step_list),
            )
            channel_deltas[pair] = base_prob - prob_without

        max_delta = max([abs(d) for d in channel_deltas.values()]) if channel_deltas else 1.0
        max_delta = max(max_delta, 1e-6)

        sorted_channels = sorted(channel_deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)

        for rank_idx, (pair, delta) in enumerate(sorted_channels[:top_k], start=1):
            norm_score = float(abs(delta) / max_delta)
            tel = channel_telemetry.get(pair, {})
            avg_lat = float(np.mean(tel.get("latencies", [0.0]))) if tel.get("latencies") else 0.0
            avg_cont = float(np.mean(tel.get("contradictions", [0.0]))) if tel.get("contradictions") else 0.0
            tot_ret = int(np.sum(tel.get("retries", [0]))) if tel.get("retries") else 0

            attributions.append(
                InteractionAttribution(
                    source_agent=pair[0],
                    target_agent=pair[1],
                    attribution_score=round(norm_score, 4),
                    delta_risk=round(delta, 4),
                    interaction_count=len(channel_steps[pair]),
                    average_latency=round(avg_lat, 4),
                    contradiction_rate=round(avg_cont, 4),
                    retry_count=tot_ret,
                    signed_contribution=round(delta, 4),
                    rank=rank_idx,
                )
            )

        return attributions

    def explain_temporal_events(
        self,
        trajectory: TemporalRunTrajectory,
        prediction_point: TemporalPredictionPoint,
        base_probability: Optional[float] = None,
        top_k: int = 6,
    ) -> List[TemporalEventAttribution]:
        """Compute Level 5 Temporal Event Importance via leave-one-out event masking."""
        if base_probability is None:
            base_prob = self._predict_point(trajectory, prediction_point)
        else:
            base_prob = base_probability

        cutoff_t = prediction_point.timestamp
        relevant_events = [inter for inter in trajectory.interactions if inter.timestamp <= cutoff_t]

        attributions: List[TemporalEventAttribution] = []
        if not relevant_events:
            return attributions

        event_deltas: List[Tuple[TemporalInteraction, float]] = []
        for inter in relevant_events:
            prob_without = self._predict_point(
                trajectory,
                prediction_point,
                excluded_interactions={inter.step_idx},
            )
            delta = base_prob - prob_without
            event_deltas.append((inter, delta))

        max_delta = max([abs(d) for _, d in event_deltas]) if event_deltas else 1.0
        max_delta = max(max_delta, 1e-6)

        sorted_events = sorted(event_deltas, key=lambda kv: abs(kv[1]), reverse=True)

        for rank_idx, (inter, delta) in enumerate(sorted_events[:top_k], start=1):
            norm_score = float(abs(delta) / max_delta)
            # Determine event type
            e_type = "message"
            if len(inter.features) >= 8:
                if inter.features[7] > 0:  # error_count
                    e_type = "error"
                elif inter.features[5] > 0:  # retry_count
                    e_type = "retry"
                elif inter.features[6] > 0:  # contradiction_rate
                    e_type = "contradiction"

            attributions.append(
                TemporalEventAttribution(
                    event_id=f"{trajectory.run_id}_step_{inter.step_idx}",
                    timestamp=round(float(inter.timestamp), 4),
                    time_before_prediction=round(float(cutoff_t - inter.timestamp), 4),
                    event_type=e_type,
                    source_agent=inter.source_agent,
                    target_agent=inter.target_agent,
                    importance_score=round(norm_score, 4),
                    signed_contribution=round(delta, 4),
                    details={
                        "step_idx": inter.step_idx,
                        "features": [round(float(f), 3) for f in inter.features],
                    },
                    rank=rank_idx,
                )
            )

        return attributions

    def explain_features(
        self,
        trajectory: TemporalRunTrajectory,
        prediction_point: TemporalPredictionPoint,
        base_probability: Optional[float] = None,
        top_k: int = 8,
    ) -> List[FeatureAttribution]:
        """Compute Level 2 Feature Importance for Temporal GNN via node and edge feature masking."""
        if base_probability is None:
            base_prob = self._predict_point(trajectory, prediction_point)
        else:
            base_prob = base_probability

        feature_deltas: List[Tuple[str, str, float, float]] = []

        # 1. Node features (14 dimensions)
        for i, fname in enumerate(NODE_FEATURE_NAMES):
            prob_without = self._predict_point(trajectory, prediction_point, feature_mask_node=i)
            delta = base_prob - prob_without
            feature_deltas.append((fname, "node", delta, 0.0))

        # 2. Edge features (10 dimensions)
        for j, efname in enumerate(EDGE_FEATURE_NAMES):
            prob_without = self._predict_point(trajectory, prediction_point, feature_mask_edge=j)
            delta = base_prob - prob_without
            feature_deltas.append((efname, "edge", delta, 0.0))

        max_delta = max([abs(d) for _, _, d, _ in feature_deltas]) if feature_deltas else 1.0
        max_delta = max(max_delta, 1e-6)

        sorted_feats = sorted(feature_deltas, key=lambda kv: abs(kv[2]), reverse=True)
        attributions: List[FeatureAttribution] = []

        for rank_idx, (fname, grp, delta, val) in enumerate(sorted_feats[:top_k], start=1):
            norm_score = float(abs(delta) / max_delta)
            attributions.append(
                FeatureAttribution(
                    feature_name=fname,
                    feature_group=grp,
                    importance_score=round(norm_score, 4),
                    signed_contribution=round(delta, 4),
                    baseline_value=val,
                    rank=rank_idx,
                )
            )

        return attributions

    def extract_signed_signals(
        self,
        feature_attributions: List[FeatureAttribution],
        agent_attributions: List[AgentAttribution],
        interaction_attributions: List[InteractionAttribution],
    ) -> Tuple[List[str], List[str]]:
        """Extract signed positive (risk increasing) and negative (risk mitigating) summary signals."""
        positive_factors: List[str] = []
        negative_factors: List[str] = []

        for f in feature_attributions:
            name_clean = f.feature_name.replace("_", " ").title()
            if f.signed_contribution > 0.01:
                positive_factors.append(f"Elevated {name_clean} (impact: +{f.signed_contribution:.3f})")
            elif f.signed_contribution < -0.01:
                negative_factors.append(f"Stable {name_clean} (mitigation: {f.signed_contribution:.3f})")

        for a in agent_attributions[:2]:
            if a.signed_contribution > 0.02:
                positive_factors.append(f"Anomalous telemetry from {a.agent_role.capitalize()} ({a.agent_id}) (+{a.signed_contribution:.3f})")
            elif a.signed_contribution < -0.02:
                negative_factors.append(f"High reliability from {a.agent_role.capitalize()} ({a.agent_id}) ({a.signed_contribution:.3f})")

        for i in interaction_attributions[:2]:
            if i.signed_contribution > 0.02:
                positive_factors.append(f"Communication friction on {i.source_agent} -> {i.target_agent} (+{i.signed_contribution:.3f})")

        if not positive_factors:
            positive_factors.append("No strong positive failure indicators detected.")
        if not negative_factors:
            negative_factors.append("No active risk-mitigating signals observed.")

        return positive_factors, negative_factors

    def compute_risk_trajectory(
        self,
        trajectory: TemporalRunTrajectory,
    ) -> List[Dict[str, Any]]:
        """Compute failure risk probability timeline t -> P(failure) across all trajectory prediction points."""
        timeline: List[Dict[str, Any]] = []
        for pp in trajectory.prediction_points:
            prob = self._predict_point(trajectory, pp)
            timeline.append({
                "timestamp": float(pp.timestamp),
                "step_idx": int(pp.step_idx),
                "sample_id": pp.sample_id,
                "horizon": int(pp.prediction_horizon),
                "predicted_probability": round(prob, 4),
                "true_label": int(pp.label),
            })
        return timeline
