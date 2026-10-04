"""Temporal Sequence Baselines Package (Phase 9).

Provides:
- MultiAgentSequenceDataset
- SequenceDatasetBuilder (configurable sequence lengths 5, 10, 20, 50)
- LSTMSequenceBaseline
- GRUSequenceBaseline
- SequenceTrainer (with validation threshold calibration and early stopping)
- evaluate_sequence_model
- SequenceExperimentRunner
- get_device
"""

from ml.baselines.sequence.dataset import (
    MultiAgentSequenceDataset,
    SequenceDatasetBuilder,
)
from ml.baselines.sequence.models import (
    LSTMSequenceBaseline,
    GRUSequenceBaseline,
    get_device,
    HAS_TORCH,
)
from ml.baselines.sequence.trainer import SequenceTrainer
from ml.baselines.sequence.evaluator import evaluate_sequence_model
from ml.baselines.sequence.experiment import SequenceExperimentRunner

__all__ = [
    "MultiAgentSequenceDataset",
    "SequenceDatasetBuilder",
    "LSTMSequenceBaseline",
    "GRUSequenceBaseline",
    "get_device",
    "HAS_TORCH",
    "SequenceTrainer",
    "evaluate_sequence_model",
    "SequenceExperimentRunner",
]
