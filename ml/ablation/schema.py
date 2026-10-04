"""Standard Schema Definitions for Phase 13 Ablation Study Framework.

Defines unified data models for:
- AblationResultRecord: Metric outputs for each ablation run.
- AblationComparisonRecord: Pairwise comparison with Full Temporal GNN (differences, CIs, p-values).
- AblationMatrixEntry: System matrix showing enabled/disabled components across experiments.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field


@dataclass
class AblationResultRecord:
    """Unified schema for ablation experiment evaluation (Section 7)."""
    experiment_id: str
    parent_experiment_id: str
    ablation_name: str
    removed_component: str
    model: str = "temporal_gnn"
    dataset_version: str = "agentguard_dataset_v1"
    feature_version: str = "v1_temporal_gnn_14node_10edge"
    graph_version: str = "v1"
    horizon: int = 1
    seed: int = 42
    threshold: float = 0.50
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auroc: float = 0.50
    auprc: float = 0.0
    false_positive_rate: float = 0.0
    false_alarm_rate: float = 0.0
    mean_lead_time: float = 0.0
    median_lead_time: float = 0.0
    successful_warnings: int = 0
    warnings_per_trajectory: float = 0.0
    sample_count: int = 0
    positive_count: int = 0
    negative_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationComparisonRecord:
    """Pairwise comparison of an ablated model against the Full Temporal GNN."""
    ablation_name: str
    removed_component: str
    horizon: int
    metric: str
    full_value: float
    ablated_value: float
    difference: float  # ablated_value - full_value
    pct_change: float  # ((ablated - full) / full) * 100
    mean_diff_seeds: float = 0.0
    std_diff_seeds: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    p_value: float = 1.0
    statistically_significant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationMatrixEntry:
    """Ablation component matrix representation (Section 11)."""
    experiment: str
    temporal: bool = True
    graph: bool = True
    node: bool = True
    edge: bool = True
    memory: bool = True
    interaction_freq: bool = True
    contradiction: bool = True
    confidence: bool = True
    failure_history: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown_row(self) -> str:
        def sym(val: bool) -> str:
            return "✓" if val else "✗"
        return f"| `{self.experiment}` | {sym(self.temporal)} | {sym(self.graph)} | {sym(self.node)} | {sym(self.edge)} | {sym(self.memory)} | {sym(self.interaction_freq)} | {sym(self.contradiction)} | {sym(self.confidence)} | {sym(self.failure_history)} |"
