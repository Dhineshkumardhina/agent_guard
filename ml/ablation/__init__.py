"""Research Ablation Study Framework - Phase 13."""

from ml.ablation.schema import (
    AblationResultRecord,
    AblationComparisonRecord,
    AblationMatrixEntry,
)
from ml.ablation.masking import (
    AblationMasker,
    ABLATION_REGISTRY,
    NODE_IDX_INTERACTION_FREQ,
    NODE_IDX_CONTRADICTION,
    NODE_IDX_CONFIDENCE,
    NODE_IDX_FAILURE_HISTORY,
    EDGE_IDX_INTERACTION_FREQ,
    EDGE_IDX_CONTRADICTION,
    EDGE_IDX_CONFIDENCE,
    EDGE_IDX_FAILURE_HISTORY,
)
from ml.ablation.visualizer import AblationVisualizer
from ml.ablation.runner import AblationStudyRunner

__all__ = [
    "AblationResultRecord",
    "AblationComparisonRecord",
    "AblationMatrixEntry",
    "AblationMasker",
    "ABLATION_REGISTRY",
    "NODE_IDX_INTERACTION_FREQ",
    "NODE_IDX_CONTRADICTION",
    "NODE_IDX_CONFIDENCE",
    "NODE_IDX_FAILURE_HISTORY",
    "EDGE_IDX_INTERACTION_FREQ",
    "EDGE_IDX_CONTRADICTION",
    "EDGE_IDX_CONFIDENCE",
    "EDGE_IDX_FAILURE_HISTORY",
    "AblationVisualizer",
    "AblationStudyRunner",
]
