"""Standard Unified Evaluation Schema - Phase 12.

Defines standardized data models for research-grade evaluation across all model families:
Family A: Classical ML (Logistic Regression, Random Forest, XGBoost)
Family B: Temporal Sequence (LSTM, GRU)
Family C: Static GNN (GCN, GAT)
Family D: Temporal GNN (TGN-style model)
Baseline: Rule-Based Early Warning Detector
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import json


@dataclass
class UnifiedEvaluationRecord:
    """Standardized metric record conforming to Section 3 of Phase 12 specification."""
    experiment_id: str = ""
    model_family: str = ""  # 'Rule-Based', 'Classical ML', 'Temporal Sequence', 'Static GNN', 'Temporal GNN'
    model_name: str = ""
    dataset_version: str = "v1"
    feature_version: str = "v1"
    graph_version: Optional[str] = None
    prediction_horizon: int = 1
    sequence_length: Optional[int] = None
    seed: int = 42
    threshold: float = 0.50
    sample_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auroc: float = 0.0
    auprc: float = 0.0
    false_positive_rate: float = 0.0
    false_alarm_rate: float = 0.0
    mean_lead_time: float = 0.0
    median_lead_time: float = 0.0
    successful_early_warnings: int = 0
    warnings_per_trajectory: float = 0.0
    brier_score: Optional[float] = None
    ece: Optional[float] = None  # Expected Calibration Error

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureCentricRecord:
    """Failure-level evaluation tracking whether impending failures were detected early."""
    failure_id: str
    run_id: str
    failure_timestamp: float
    first_valid_warning_timestamp: Optional[float] = None
    lead_time: Optional[float] = None
    warning_count: int = 0
    detected: bool = False
    failure_type: str = "unknown"
    failure_level: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationBin:
    """Calibration reliability bin."""
    bin_idx: int
    bin_lower: float
    bin_upper: float
    predicted_prob_mean: float
    observed_positive_ratio: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurvePoint:
    """Single coordinate point for ROC or PR curves."""
    threshold: float
    fpr: Optional[float] = None
    tpr: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result."""
    metric_name: str
    mean: float
    median: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairwiseComparisonRecord:
    """Pairwise model comparison with difference and bootstrap confidence bounds."""
    model_a: str
    model_b: str
    horizon: int
    metric: str
    value_a: float
    value_b: float
    difference: float  # value_a - value_b
    ci_lower: float
    ci_upper: float
    p_value: Optional[float] = None
    sample_count: int = 0
    statistically_significant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubgroupMetricRecord:
    """Metrics broken down by subgroup (topology, task, failure type, failure level, agent count)."""
    subgroup_category: str  # 'topology', 'task', 'failure_type', 'failure_level', 'agent_count'
    subgroup_value: str
    model_name: str
    horizon: int
    sample_count: int
    positive_count: int
    negative_count: int
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

