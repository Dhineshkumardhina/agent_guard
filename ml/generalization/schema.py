"""Generalization and Robustness Evaluation Schemas - Phase 14.

Defines data models for:
- Out-of-Distribution (OOD) experiment definitions (G1–G5)
- Generalization result records (In-Distribution vs OOD)
- Generalization gap metrics (ID - OOD transfer gaps with bootstrap CIs)
- Subgroup and failure-type breakdown records
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
import json


@dataclass
class GeneralizationExperimentConfig:
    """Specification for an out-of-distribution generalization experiment."""

    experiment_id: str
    dimension: str  # "agent_count", "topology", "task", "failure_type"
    name: str
    description: str
    train_filter: Dict[str, Any]  # Criteria for in-distribution runs
    test_ood_filter: Dict[str, Any]  # Criteria for out-of-distribution runs


@dataclass
class GeneralizationResultRecord:
    """Record storing performance for a specific model under ID or OOD conditions."""

    experiment_id: str
    dimension: str
    split_type: str  # "in_distribution" or "out_of_distribution"
    model: str
    dataset_version: str
    horizon: int
    seed: int
    threshold: float
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float
    false_positive_rate: float
    false_alarm_rate: float
    mean_lead_time: float
    median_lead_time: float
    successful_warnings: int
    warnings_per_trajectory: float
    sample_count: int
    positive_count: int
    negative_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneralizationGapRecord:
    """Quantifies the performance delta between in-distribution and OOD evaluation."""

    experiment_id: str
    dimension: str
    model: str
    horizon: int
    metric: str
    id_value: float
    ood_value: float
    gap: float  # ID - OOD (positive means performance dropped OOD)
    pct_change: float  # ((OOD - ID) / max(1e-4, ID)) * 100
    ci_lower: float
    ci_upper: float
    p_value: float
    statistically_significant: bool
    interpretation: str
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BreakdownRecord:
    """Performance breakdown for a single subgroup (topology, task, agent count, or failure type)."""

    dimension: str
    subgroup: str
    model: str
    horizon: int
    sample_count: int
    positive_count: int
    precision: float
    recall: float
    f1: float
    auprc: float
    mean_lead_time: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneralizationMatrixEntry:
    """Specification of experiment configuration for publication matrix."""

    experiment_id: str
    dimension: str
    train_configuration: str
    test_configuration: str
    models_tested: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown_row(self) -> str:
        return f"| `{self.experiment_id}` | {self.dimension} | {self.train_configuration} | {self.test_configuration} | {self.models_tested} |"
