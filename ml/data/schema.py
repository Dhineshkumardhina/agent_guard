"""Scientific Dataset Schemas for AgentGuard Prediction Samples.

Defines schemas and models for:
- P(F(t+k) | observations up to t) prediction sample representations
- Three comparative feature views:
  1. Agent-level behavioral features
  2. Static graph representations
  3. Temporal graph representations
- Dataset version manifests and split summaries
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pydantic import BaseModel, Field


@dataclass
class PredictionSample:
    """Individual prediction point sample for failure forecasting.
    
    Attributes:
        sample_id: Unique deterministic sample identifier (e.g. {run_id}_s{step_idx}_k{horizon}).
        run_id: Simulation trajectory identifier.
        timestamp: Simulation timestamp t at prediction point.
        step_idx: Discrete event step index at prediction point t.
        task_type: Multi-agent task benchmark category (research, coding, analysis, planning).
        topology: Communication topology (pipeline, star, mesh, custom).
        number_of_agents: Agent population count in the simulation run.
        prediction_horizon: Forecast horizon k in interaction steps (1, 3, 5, 10, 20).
        node_features: Causal agent node features up to step t.
        edge_features: Causal pairwise interaction edge features up to step t.
        temporal_graph_history: Chronological sequence of graph snapshots [G(t-n), ..., G(t)].
        agent_level_features: Tabular behavioral metrics aggregated across agents up to step t.
        label: Binary ground truth (0 = no failure within horizon, 1 = failure within horizon).
        failure_type: Specific fault type if failure occurs in (t, t+k], else "none".
        failure_level: Severity level (0 = none, 1 = agent, 2 = interaction, 3 = cascading).
        source_event_id: Event ID corresponding to step t.
        random_seed: PRNG seed used for the run.
        dataset_version: Semantic dataset version string (e.g. agentguard_dataset_v1).
        split: Dataset split assignment ("train", "val", "test").
    """

    sample_id: str
    run_id: str
    timestamp: float
    step_idx: int
    task_type: str
    topology: str
    number_of_agents: int
    prediction_horizon: int
    node_features: Dict[str, Dict[str, float]]
    edge_features: Dict[str, Dict[str, float]]
    temporal_graph_history: List[Dict[str, Any]]
    agent_level_features: Dict[str, float]
    label: int
    failure_type: str
    failure_level: int
    source_event_id: Optional[str]
    random_seed: int
    dataset_version: str
    split: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to standard dictionary."""
        return asdict(self)

    def to_tabular_dict(self) -> Dict[str, Any]:
        """Convert sample into flat dictionary suitable for Parquet / DataFrame tabular storage.
        
        Extracts scalar metadata, labels, agent-level features, and graph summary stats,
        leaving detailed graph history sequences to structured graph sequence storage.
        """
        row: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "run_id": self.run_id,
            "timestamp": round(self.timestamp, 4),
            "step_idx": self.step_idx,
            "task_type": self.task_type,
            "topology": self.topology,
            "number_of_agents": self.number_of_agents,
            "prediction_horizon": self.prediction_horizon,
            "label": int(self.label),
            "failure_type": self.failure_type,
            "failure_level": int(self.failure_level),
            "source_event_id": self.source_event_id or "",
            "random_seed": self.random_seed,
            "dataset_version": self.dataset_version,
            "split": self.split or "unassigned",
            # Graph metadata summary
            "num_nodes": len(self.node_features),
            "num_edges": len(self.edge_features),
            "history_length": len(self.temporal_graph_history),
        }

        # Prefix and incorporate agent-level behavioral features
        for k, v in self.agent_level_features.items():
            row[f"agent_feat_{k}"] = float(v)

        return row

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionSample":
        """Reconstruct PredictionSample from dictionary."""
        return cls(
            sample_id=data["sample_id"],
            run_id=data["run_id"],
            timestamp=float(data["timestamp"]),
            step_idx=int(data["step_idx"]),
            task_type=data["task_type"],
            topology=data["topology"],
            number_of_agents=int(data["number_of_agents"]),
            prediction_horizon=int(data["prediction_horizon"]),
            node_features=data.get("node_features", {}),
            edge_features=data.get("edge_features", {}),
            temporal_graph_history=data.get("temporal_graph_history", []),
            agent_level_features=data.get("agent_level_features", {}),
            label=int(data["label"]),
            failure_type=data.get("failure_type", "none"),
            failure_level=int(data.get("failure_level", 0)),
            source_event_id=data.get("source_event_id"),
            random_seed=int(data.get("random_seed", 42)),
            dataset_version=data.get("dataset_version", "agentguard_dataset_v1"),
            split=data.get("split"),
        )


class DatasetManifest(BaseModel):
    """Metadata manifest detailing a generated scientific dataset."""

    dataset_version: str = Field(..., description="Unique dataset version identifier")
    creation_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp"
    )
    simulator_version: str = Field(default="0.6.0")
    configuration: Dict[str, Any] = Field(default_factory=dict)
    random_seed: int = Field(default=42)
    number_of_runs: int = Field(default=0)
    number_of_samples: int = Field(default=0)
    prediction_horizons: List[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    topologies: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    agent_counts: List[int] = Field(default_factory=list)
    class_distribution: Dict[str, Any] = Field(default_factory=dict)
    split_distribution: Dict[str, Any] = Field(default_factory=dict)
    feature_schema: Dict[str, Any] = Field(default_factory=dict)
    storage_format: str = Field(default="parquet")
    description: str = Field(
        default="Controlled experimental multi-agent dataset for cascading failure prediction"
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetManifest":
        """Deserialize manifest from JSON string."""
        return cls.model_validate_json(json_str)
