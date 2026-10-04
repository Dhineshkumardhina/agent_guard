"""Ablation Feature and Structure Masker - Phase 13.

Applies deterministic, copy-on-write masks to temporal trajectories, node features,
and interaction edge features to isolate individual information sources.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import copy
import torch

from ml.baselines.temporal_gnn.dataset import (
    TemporalRunTrajectory,
    TemporalInteraction,
    TemporalPredictionPoint,
)
from ml.baselines.temporal_gnn.models import AblationConfig
from ml.ablation.schema import AblationMatrixEntry


# Feature Index Definitions
# Node Features (14-dim)
# 0: event_count, 1: message_count, 2: tool_call_count, 3: error_count, 4: retry_count,
# 5: timeout_count, 6: average_latency, 7: average_confidence, 8: average_output_quality,
# 9: contradiction_rate, 10: recent_failure_count, 11: incoming_cnt, 12: outgoing_cnt, 13: is_active
NODE_IDX_INTERACTION_FREQ = [0, 1, 11, 12]
NODE_IDX_CONTRADICTION = [9]
NODE_IDX_CONFIDENCE = [7]
NODE_IDX_FAILURE_HISTORY = [3, 4, 5, 10]

# Edge Features (10-dim)
# 0: interaction_count, 1: message_count, 2: average_latency, 3: average_message_length,
# 4: average_confidence, 5: retry_count, 6: contradiction_rate, 7: error_count,
# 8: timeout_count, 9: interaction_frequency
EDGE_IDX_INTERACTION_FREQ = [0, 1, 9]
EDGE_IDX_CONTRADICTION = [6]
EDGE_IDX_CONFIDENCE = [4]
EDGE_IDX_FAILURE_HISTORY = [5, 7, 8]


ABLATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "full_temporal_gnn": {
        "name": "Full Temporal GNN",
        "description": "Complete architecture with continuous time encoding, dynamic memory, temporal attention, and all features.",
        "removed_component": "None (Reference)",
        "ablation_config": AblationConfig(),
        "node_mask_indices": [],
        "edge_mask_indices": [],
    },
    "no_temporal_info": {
        "name": "No Temporal Information",
        "description": "Removes continuous Fourier time encodings and elapsed delta intervals.",
        "removed_component": "Continuous Time Encoding",
        "ablation_config": AblationConfig(enable_time_encoding=False),
        "node_mask_indices": [],
        "edge_mask_indices": [],
    },
    "no_graph_structure": {
        "name": "No Graph Structure",
        "description": "Disables interaction graph connectivity and neighborhood aggregation (isolated per-agent processing).",
        "removed_component": "Graph Topology",
        "ablation_config": AblationConfig(enable_graph_structure=False, enable_neighborhood=False),
        "node_mask_indices": [],
        "edge_mask_indices": [],
    },
    "no_node_features": {
        "name": "No Node Features",
        "description": "Zeros out all 14 node behavioral telemetry features while retaining graph interaction topology.",
        "removed_component": "Node Behavioral Telemetry",
        "ablation_config": AblationConfig(enable_node_features=False),
        "node_mask_indices": list(range(14)),
        "edge_mask_indices": [],
    },
    "no_edge_features": {
        "name": "No Edge Features",
        "description": "Zeros out all 10 edge interaction attribute features while retaining graph topology and timestamps.",
        "removed_component": "Edge Behavioral Attributes",
        "ablation_config": AblationConfig(enable_edge_features=False),
        "node_mask_indices": [],
        "edge_mask_indices": list(range(10)),
    },
    "no_temporal_memory": {
        "name": "No Temporal Memory",
        "description": "Disables persistent per-agent node memory updates across chronological interaction events.",
        "removed_component": "Persistent Node Memory (m_v)",
        "ablation_config": AblationConfig(enable_memory=False),
        "node_mask_indices": [],
        "edge_mask_indices": [],
    },
    "no_interaction_freq": {
        "name": "No Interaction Frequency",
        "description": "Removes communication intensity and message volume features from node and edge representations.",
        "removed_component": "Communication Frequency Metrics",
        "ablation_config": AblationConfig(enable_interaction_frequency=False),
        "node_mask_indices": NODE_IDX_INTERACTION_FREQ,
        "edge_mask_indices": EDGE_IDX_INTERACTION_FREQ,
    },
    "no_contradiction": {
        "name": "No Contradiction Information",
        "description": "Removes semantic contradiction and conflicting communication signals.",
        "removed_component": "Contradiction / Conflict Signals",
        "ablation_config": AblationConfig(enable_contradiction_features=False),
        "node_mask_indices": NODE_IDX_CONTRADICTION,
        "edge_mask_indices": EDGE_IDX_CONTRADICTION,
    },
    "no_confidence": {
        "name": "No Confidence Information",
        "description": "Removes agent confidence levels and confidence degradation indicators.",
        "removed_component": "Confidence Telemetry",
        "ablation_config": AblationConfig(enable_confidence_features=False),
        "node_mask_indices": NODE_IDX_CONFIDENCE,
        "edge_mask_indices": EDGE_IDX_CONFIDENCE,
    },
    "no_failure_history": {
        "name": "No Failure History",
        "description": "Removes recent failure, error, timeout, and retry counts to test proactive interaction forecasting.",
        "removed_component": "Recent Failure / Error History",
        "ablation_config": AblationConfig(enable_failure_history=False),
        "node_mask_indices": NODE_IDX_FAILURE_HISTORY,
        "edge_mask_indices": EDGE_IDX_FAILURE_HISTORY,
    },
}


class AblationMasker:
    """Transforms trajectories by applying feature masking without corrupting source data."""

    def __init__(self, ablation_key: str):
        if ablation_key not in ABLATION_REGISTRY:
            raise KeyError(f"Unknown ablation key: {ablation_key}. Available: {list(ABLATION_REGISTRY.keys())}")
        self.ablation_key = ablation_key
        self.spec = ABLATION_REGISTRY[ablation_key]
        self.node_mask_indices = set(self.spec.get("node_mask_indices", []))
        self.edge_mask_indices = set(self.spec.get("edge_mask_indices", []))
        self.ablation_config = self.spec.get("ablation_config", AblationConfig())

    def mask_node_features(self, feat_vec: List[float]) -> List[float]:
        """Apply zero-mask to configured node feature indices."""
        if not self.node_mask_indices:
            return list(feat_vec)
        out = list(feat_vec)
        for idx in self.node_mask_indices:
            if idx < len(out):
                out[idx] = 0.0
        return out

    def mask_edge_features(self, feat_vec: List[float]) -> List[float]:
        """Apply zero-mask to configured edge feature indices."""
        if not self.edge_mask_indices:
            return list(feat_vec)
        out = list(feat_vec)
        for idx in self.edge_mask_indices:
            if idx < len(out):
                out[idx] = 0.0
        return out

    def transform_trajectories(
        self,
        trajectories: List[TemporalRunTrajectory],
    ) -> List[TemporalRunTrajectory]:
        """Produce a transformed copy of trajectories with features masked."""
        transformed: List[TemporalRunTrajectory] = []

        for traj in trajectories:
            new_traj = TemporalRunTrajectory(
                run_id=traj.run_id,
                topology=traj.topology,
                interactions=[],
                prediction_points=[],
            )

            # Mask interactions
            for inter in traj.interactions:
                masked_edge_feats = self.mask_edge_features(inter.features)
                new_traj.interactions.append(
                    TemporalInteraction(
                        source_agent=inter.source_agent,
                        target_agent=inter.target_agent,
                        timestamp=inter.timestamp,
                        step_idx=inter.step_idx,
                        features=masked_edge_feats,
                    )
                )

            # Mask prediction points
            for pp in traj.prediction_points:
                masked_node_dict: Dict[str, List[float]] = {}
                for aid, feat in pp.node_features.items():
                    masked_node_dict[aid] = self.mask_node_features(feat)

                new_traj.prediction_points.append(
                    TemporalPredictionPoint(
                        sample_id=pp.sample_id,
                        run_id=pp.run_id,
                        step_idx=pp.step_idx,
                        timestamp=pp.timestamp,
                        prediction_horizon=pp.prediction_horizon,
                        label=pp.label,
                        active_agents=list(pp.active_agents),
                        node_features=masked_node_dict,
                    )
                )

            transformed.append(new_traj)

        return transformed

    @classmethod
    def get_matrix_entry(cls, ablation_key: str) -> AblationMatrixEntry:
        """Construct AblationMatrixEntry for reporting."""
        spec = ABLATION_REGISTRY.get(ablation_key, {})
        cfg: AblationConfig = spec.get("ablation_config", AblationConfig())
        node_masks = set(spec.get("node_mask_indices", []))
        edge_masks = set(spec.get("edge_mask_indices", []))

        return AblationMatrixEntry(
            experiment=ablation_key,
            temporal=cfg.enable_time_encoding,
            graph=cfg.enable_graph_structure,
            node=(len(node_masks) < 14) and cfg.enable_node_features,
            edge=(len(edge_masks) < 10) and cfg.enable_edge_features,
            memory=cfg.enable_memory,
            interaction_freq=cfg.enable_interaction_frequency and not (set(NODE_IDX_INTERACTION_FREQ).issubset(node_masks)),
            contradiction=cfg.enable_contradiction_features and not (set(NODE_IDX_CONTRADICTION).issubset(node_masks)),
            confidence=cfg.enable_confidence_features and not (set(NODE_IDX_CONFIDENCE).issubset(node_masks)),
            failure_history=cfg.enable_failure_history and not (set(NODE_IDX_FAILURE_HISTORY).issubset(node_masks)),
        )
