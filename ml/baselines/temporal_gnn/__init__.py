"""Temporal Graph Neural Network (TGN-Style) Baseline Module - Phase 11.

Evaluates whether continuous temporal interaction dynamics and evolving agent memory
provide predictive value for cascading failure forecasting beyond agent-level and static graph baselines.
"""

from ml.baselines.temporal_gnn.time_encoding import TimeEncoder
from ml.baselines.temporal_gnn.memory import NodeMemory, MemoryUpdater
from ml.baselines.temporal_gnn.neighborhood import TemporalNeighborhoodTracker
from ml.baselines.temporal_gnn.models import (
    TemporalGraphFailurePredictor,
    MessageFunction,
    TemporalEmbedding,
    GraphReadout,
    FailurePredictionHead,
    AblationConfig,
    get_device,
)
from ml.baselines.temporal_gnn.dataset import (
    TemporalInteraction,
    TemporalPredictionPoint,
    TemporalRunTrajectory,
    TemporalDatasetBuilder,
    compute_temporal_diagnostics,
    debug_memory_trace,
)
from ml.baselines.temporal_gnn.trainer import TemporalGNNTrainer
from ml.baselines.temporal_gnn.evaluator import evaluate_temporal_gnn
from ml.baselines.temporal_gnn.experiment import TemporalGNNExperimentRunner

__all__ = [
    "TimeEncoder",
    "NodeMemory",
    "MemoryUpdater",
    "TemporalNeighborhoodTracker",
    "TemporalGraphFailurePredictor",
    "MessageFunction",
    "TemporalEmbedding",
    "GraphReadout",
    "FailurePredictionHead",
    "AblationConfig",
    "get_device",
    "TemporalInteraction",
    "TemporalPredictionPoint",
    "TemporalRunTrajectory",
    "TemporalDatasetBuilder",
    "compute_temporal_diagnostics",
    "debug_memory_trace",
    "TemporalGNNTrainer",
    "evaluate_temporal_gnn",
    "TemporalGNNExperimentRunner",
]
