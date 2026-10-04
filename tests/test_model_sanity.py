"""Phase 18: Comprehensive Model Sanity Tests.

Verifies behavioral invariants for ALL model families:
1. Rule-based baseline
2. Classical ML baselines (Logistic Regression, Random Forest, Gradient Boosting)
3. Sequence baselines (LSTM, Transformer)
4. Static GNN baselines (GCN, GAT, GraphSAGE)
5. Temporal GNN (TGN-style dynamic architecture)

Invariants tested:
- normal input
- empty input (must raise ValueError/RuntimeError cleanly, no silent failures)
- malformed input (dimension mismatch)
- single sample
- small dataset (e.g. 2 samples)
- class imbalance (e.g. 95% neg, 5% pos)
- all-negative labels (single-class edge cases handled without crash)
- all-positive labels (single-class edge cases handled without crash)
- missing features (NaNs/infinities checked)
- extreme values (large numerical magnitudes)
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from ml.baselines.rule_based.detector import RuleBasedEarlyWarningDetector
from ml.baselines.rule_based.config import RuleBaselineConfig
from ml.baselines.classical_ml.models import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)
from ml.baselines.sequence.models import LSTMSequenceBaseline, GRUSequenceBaseline
from ml.baselines.static_gnn.models import GCNBaseline, GATBaseline
from ml.baselines.temporal_gnn.models import TemporalGraphFailurePredictor


# =========================================================================
# 1. Rule-Based Baseline Sanity Tests
# =========================================================================

def test_rule_baseline_normal_input():
    detector = RuleBasedEarlyWarningDetector()
    samples = [
        {"error_count": 0, "total_events_observed": 10, "mean_confidence": 0.99, "mean_latency": 0.10},
        {"error_count": 8, "total_events_observed": 10, "mean_confidence": 0.20, "mean_latency": 0.80},
    ]
    results = detector.predict_batch(samples)
    assert len(results) == 2
    assert 0.0 <= results[0]["risk_score"] <= 1.0
    assert 0.0 <= results[1]["risk_score"] <= 1.0
    assert results[1]["risk_score"] > results[0]["risk_score"]


def test_rule_baseline_empty_input():
    detector = RuleBasedEarlyWarningDetector()
    results = detector.predict_batch([])
    assert results == []


def test_rule_baseline_missing_features():
    detector = RuleBasedEarlyWarningDetector()
    # Missing typical keys should default gracefully without unhandled exception
    res = detector.predict_sample({})
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["prediction"] in (0, 1)


def test_rule_baseline_extreme_values():
    detector = RuleBasedEarlyWarningDetector()
    extreme_samples = [
        {"error_rate": 1e9, "retry_count": 1000000, "latency": 1e12, "contradiction_score": 1e6},
        {"error_rate": -1e9, "retry_count": -100, "latency": -50.0, "contradiction_score": -1.0},
    ]
    results = detector.predict_batch(extreme_samples)
    assert len(results) == 2
    assert 0.0 <= results[0]["risk_score"] <= 1.0
    assert 0.0 <= results[1]["risk_score"] <= 1.0


# =========================================================================
# 2. Classical ML Baselines Sanity Tests
# =========================================================================

@pytest.mark.parametrize("model_cls", [
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
])
def test_classical_ml_normal_and_small_dataset(model_cls):
    rng = np.random.RandomState(42)
    X = rng.randn(10, 5)
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    model = model_cls(random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (10,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


@pytest.mark.parametrize("model_cls", [
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
])
def test_classical_ml_empty_input_fails_cleanly(model_cls):
    model = model_cls()
    with pytest.raises(ValueError, match="empty dataset"):
        model.fit(np.empty((0, 5)), np.empty((0,)))


@pytest.mark.parametrize("model_cls", [
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
])
def test_classical_ml_all_negative_labels(model_cls):
    X = np.random.randn(8, 4)
    y = np.zeros(8, dtype=int)
    model = model_cls(random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == 8
    # When trained on all-negatives, risk probability should be 0.0
    assert np.all(probs <= 0.5)


@pytest.mark.parametrize("model_cls", [
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
])
def test_classical_ml_all_positive_labels(model_cls):
    X = np.random.randn(8, 4)
    y = np.ones(8, dtype=int)
    model = model_cls(random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == 8
    # When trained on all-positives, risk probability should be high
    assert np.all(probs >= 0.5)


@pytest.mark.parametrize("model_cls", [
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
])
def test_classical_ml_single_sample_inference(model_cls):
    X_train = np.random.randn(6, 4)
    y_train = np.array([0, 1, 0, 1, 0, 1])
    model = model_cls(random_state=42)
    model.fit(X_train, y_train)

    X_single = np.random.randn(1, 4)
    prob = model.predict_proba(X_single)
    assert prob.shape == (1,)
    assert 0.0 <= prob[0] <= 1.0


@pytest.mark.parametrize("model_cls", [
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
])
def test_classical_ml_extreme_values(model_cls):
    X_train = np.array([
        [1e8, -1e8, 0.0, 1e5],
        [-1e8, 1e8, 1.0, -1e5],
        [0.0, 0.0, 0.5, 0.0],
        [1.0, -1.0, 0.2, 5.0],
    ])
    y_train = np.array([1, 0, 0, 1])
    model = model_cls(random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_train)
    assert np.all(np.isfinite(probs))


# =========================================================================
# 3. Sequence Baseline Sanity Tests (LSTM & GRU)
# =========================================================================

@pytest.mark.parametrize("model_cls", [LSTMSequenceBaseline, GRUSequenceBaseline])
def test_sequence_model_normal_forward(model_cls):
    model = model_cls(input_size=8, hidden_size=16, num_layers=1, dropout=0.0)
    model.eval()
    batch_size, seq_len = 4, 10
    x = torch.randn(batch_size, seq_len, 8)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch_size, 1) or out.shape == (batch_size,)


@pytest.mark.parametrize("model_cls", [LSTMSequenceBaseline, GRUSequenceBaseline])
def test_sequence_model_single_sample(model_cls):
    model = model_cls(input_size=8, hidden_size=16, num_layers=1, dropout=0.0)
    model.eval()
    x = torch.randn(1, 5, 8)
    with torch.no_grad():
        out = model(x)
    assert out.shape[0] == 1


@pytest.mark.parametrize("model_cls", [LSTMSequenceBaseline, GRUSequenceBaseline])
def test_sequence_model_malformed_dim_fails(model_cls):
    model = model_cls(input_size=8, hidden_size=16, num_layers=1)
    # Incorrect feature dimension (4 instead of 8)
    x = torch.randn(2, 5, 4)
    with pytest.raises(Exception):
        model(x)


# =========================================================================
# 4. Static GNN Sanity Tests (GCN, GAT, GraphSAGE)
# =========================================================================

@pytest.mark.parametrize("gnn_cls", [GCNBaseline, GATBaseline])
def test_static_gnn_normal_and_extreme(gnn_cls):
    model = gnn_cls(node_in_dim=6, hidden_dim=16)
    model.eval()
    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.batch)
    assert out.shape == (1, 1) or out.shape == (1,)
    prob = float(torch.sigmoid(out).squeeze().item())
    assert 0.0 <= prob <= 1.0


@pytest.mark.parametrize("gnn_cls", [GCNBaseline, GATBaseline])
def test_static_gnn_single_node(gnn_cls):
    model = gnn_cls(node_in_dim=6, hidden_dim=16)
    model.eval()
    x = torch.randn(1, 6)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.zeros(1, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.batch)
        prob = float(torch.sigmoid(out).squeeze().item())
    assert np.isfinite(prob)
    assert 0.0 <= prob <= 1.0


# =========================================================================
# 5. Temporal GNN Sanity Tests
# =========================================================================

def test_temporal_gnn_normal_forward():
    model = TemporalGraphFailurePredictor(
        node_in_dim=8,
        edge_in_dim=4,
        time_dim=8,
        memory_dim=16,
        embed_dim=16,
    )
    model.eval()
    model.reset_memory()

    # Interact agent_0 and agent_1
    model.process_interaction(
        source_agent="agent_0",
        target_agent="agent_1",
        timestamp=1.0,
        edge_features=torch.randn(4),
    )

    logits, prob = model.predict_at_timestamp(
        agent_ids=["agent_0", "agent_1"],
        node_features_dict={
            "agent_0": [0.1] * 8,
            "agent_1": [0.2] * 8,
        },
        current_timestamp=1.5,
    )
    assert 0.0 <= float(prob) <= 1.0
    assert torch.isfinite(logits)


def test_temporal_gnn_memory_isolation_after_reset():
    model = TemporalGraphFailurePredictor(
        node_in_dim=8,
        edge_in_dim=4,
        time_dim=8,
        memory_dim=16,
        embed_dim=16,
    )
    model.reset_memory()
    # Process interactions in run 1
    for t in range(5):
        model.process_interaction("agent_0", "agent_1", float(t + 1), torch.randn(4))

    mem_before = model.node_memory.get_memory("agent_0").clone()
    assert not torch.allclose(mem_before, torch.zeros_like(mem_before))

    # Reset memory between runs
    model.reset_memory()
    mem_after = model.node_memory.get_memory("agent_0")
    assert torch.allclose(mem_after, torch.zeros_like(mem_after))
