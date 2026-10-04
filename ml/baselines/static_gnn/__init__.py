"""Static Graph Neural Network (Static GNN) Baseline Module - Phase 10.

Evaluates whether representing multi-agent interactions as static graphs G(t) = (V, E, X)
provides predictive value beyond non-graph agent-level and temporal sequence baselines.
"""

from ml.baselines.static_gnn.models import (
    BaseStaticGNN,
    GCNBaseline,
    GATBaseline,
    get_device,
)
from ml.baselines.static_gnn.dataset import (
    StaticGraphDatasetBuilder,
    NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    compute_graph_diagnostics,
)
from ml.baselines.static_gnn.trainer import StaticGNNTrainer
from ml.baselines.static_gnn.evaluator import evaluate_static_gnn
from ml.baselines.static_gnn.experiment import StaticGNNExperimentRunner

__all__ = [
    "BaseStaticGNN",
    "GCNBaseline",
    "GATBaseline",
    "get_device",
    "StaticGraphDatasetBuilder",
    "NODE_FEATURE_NAMES",
    "EDGE_FEATURE_NAMES",
    "compute_graph_diagnostics",
    "StaticGNNTrainer",
    "evaluate_static_gnn",
    "StaticGNNExperimentRunner",
]
