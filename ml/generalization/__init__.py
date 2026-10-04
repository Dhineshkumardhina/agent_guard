"""Generalization and Robustness Evaluation Framework - Phase 14."""

from ml.generalization.schema import (
    GeneralizationExperimentConfig,
    GeneralizationResultRecord,
    GeneralizationGapRecord,
    BreakdownRecord,
    GeneralizationMatrixEntry,
)
from ml.generalization.splits import (
    GeneralizationSplitter,
    GeneralizationSplitBundle,
    GENERALIZATION_REGISTRY,
    SEEN_FAILURE_MODES,
    UNSEEN_FAILURE_MODES,
)
from ml.generalization.models import GeneralizationModelRunner
from ml.generalization.visualizer import GeneralizationVisualizer
from ml.generalization.runner import GeneralizationExperimentRunner

__all__ = [
    "GeneralizationExperimentConfig",
    "GeneralizationResultRecord",
    "GeneralizationGapRecord",
    "BreakdownRecord",
    "GeneralizationMatrixEntry",
    "GeneralizationSplitter",
    "GeneralizationSplitBundle",
    "GENERALIZATION_REGISTRY",
    "SEEN_FAILURE_MODES",
    "UNSEEN_FAILURE_MODES",
    "GeneralizationModelRunner",
    "GeneralizationVisualizer",
    "GeneralizationExperimentRunner",
]
