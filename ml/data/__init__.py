"""AgentGuard Dataset Generation, Temporal Samples, and Failure Labeling (Phase 6).

Provides:
- PredictionSample, DatasetManifest schema
- compute_prediction_label with explicit failure level taxonomy
- extract_agent_level_features, extract_graph_features
- PredictionSampleGenerator
- split_trajectories, split_samples (trajectory-level split)
- DatasetStorage (Parquet & JSONL)
- DatasetValidator, ValidationResult
- DatasetBuilder
"""

from ml.data.schema import PredictionSample, DatasetManifest
from ml.data.labeling import compute_prediction_label
from ml.data.feature_windows import extract_agent_level_features, extract_graph_features
from ml.data.prediction_samples import PredictionSampleGenerator
from ml.data.split import split_trajectories, split_samples
from ml.data.storage import DatasetStorage
from ml.data.validation import DatasetValidator, ValidationResult
from ml.data.dataset_builder import DatasetBuilder

__all__ = [
    "PredictionSample",
    "DatasetManifest",
    "compute_prediction_label",
    "extract_agent_level_features",
    "extract_graph_features",
    "PredictionSampleGenerator",
    "split_trajectories",
    "split_samples",
    "DatasetStorage",
    "DatasetValidator",
    "ValidationResult",
    "DatasetBuilder",
]
