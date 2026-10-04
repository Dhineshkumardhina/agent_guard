"""Comprehensive Test Suite for Phase 10: Static Graph Neural Networks (GCN / GAT).

Covers all 17 mandatory verification checks:
1. Graph-to-tensor conversion
2. Node feature dimensions (14 features)
3. Edge dimensions (10 features)
4. Batch creation with PyG DataLoader
5. GCN forward pass
6. GAT forward pass
7. Graph pooling (mean, max, both)
8. Output shape validation
9. Probability range [0.0, 1.0]
10. Training step (loss backward and optimizer step)
11. Checkpoint save and load
12. Run-level dataset split isolation
13. Future leakage prevention in graph representations
14. Reproducibility test
15. CPU execution guarantee
16. Metric calculation
17. Lead-time calculation
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import json

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ml.graph.graph_snapshot import GraphSnapshot
from ml.baselines.static_gnn.dataset import (
    snapshot_to_pyg_data,
    compute_graph_diagnostics,
    StaticGraphDatasetBuilder,
    NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
)
from ml.baselines.static_gnn.models import (
    BaseStaticGNN,
    GCNBaseline,
    GATBaseline,
    get_device,
)
from ml.baselines.static_gnn.trainer import StaticGNNTrainer
from ml.baselines.static_gnn.evaluator import evaluate_static_gnn
from ml.data.storage import DatasetStorage


def create_synthetic_snapshot(timestamp: float = 2.0, run_id: str = "run_test_01") -> GraphSnapshot:
    """Helper to create a deterministic GraphSnapshot with 3 agents and 2 interactions."""
    nodes = {
        "agent_a": {
            "agent_id": "agent_a",
            "is_active": True,
            "event_count": 5,
            "message_count": 4,
            "tool_call_count": 2,
            "error_count": 1,
            "retry_count": 0,
            "timeout_count": 0,
            "average_latency": 0.12,
            "average_confidence": 0.88,
            "average_output_quality": 0.90,
            "contradiction_rate": 0.05,
            "recent_failure_count": 1,
        },
        "agent_b": {
            "agent_id": "agent_b",
            "is_active": True,
            "event_count": 4,
            "message_count": 3,
            "tool_call_count": 1,
            "error_count": 0,
            "retry_count": 0,
            "timeout_count": 0,
            "average_latency": 0.08,
            "average_confidence": 0.95,
            "average_output_quality": 0.98,
            "contradiction_rate": 0.0,
            "recent_failure_count": 0,
        },
        "agent_c": {
            "agent_id": "agent_c",
            "is_active": False,
            "event_count": 1,
            "message_count": 0,
            "tool_call_count": 0,
            "error_count": 0,
            "retry_count": 0,
            "timeout_count": 0,
            "average_latency": 0.0,
            "average_confidence": 1.0,
            "average_output_quality": 1.0,
            "contradiction_rate": 0.0,
            "recent_failure_count": 0,
        },
    }
    edges = {
        ("agent_a", "agent_b"): {
            "source_agent": "agent_a",
            "target_agent": "agent_b",
            "interaction_count": 3,
            "message_count": 3,
            "average_latency": 0.10,
            "average_message_length": 150.0,
            "average_confidence": 0.90,
            "retry_count": 0,
            "contradiction_rate": 0.0,
            "error_count": 0,
            "timeout_count": 0,
            "interaction_frequency": 1.5,
        },
        ("agent_b", "agent_c"): {
            "source_agent": "agent_b",
            "target_agent": "agent_c",
            "interaction_count": 1,
            "message_count": 1,
            "average_latency": 0.05,
            "average_message_length": 80.0,
            "average_confidence": 0.99,
            "retry_count": 0,
            "contradiction_rate": 0.0,
            "error_count": 0,
            "timeout_count": 0,
            "interaction_frequency": 0.5,
        },
    }
    return GraphSnapshot(
        timestamp=timestamp,
        run_id=run_id,
        step_idx=2,
        nodes=nodes,
        edges=edges,
    )


# ── Test 1: Graph-to-tensor conversion ──
def test_graph_to_tensor_conversion():
    snap = create_synthetic_snapshot()
    data = snapshot_to_pyg_data(snap, label=1.0, sample_id="s_test_1", horizon=3)

    assert isinstance(data, Data)
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)
    assert isinstance(data.edge_attr, torch.Tensor)
    assert isinstance(data.y, torch.Tensor)
    assert data.y.item() == 1.0
    assert data.sample_id == "s_test_1"
    assert data.prediction_horizon == 3


# ── Test 2: Node feature dimensions ──
def test_node_feature_dimensions():
    snap = create_synthetic_snapshot()
    data = snapshot_to_pyg_data(snap)

    assert data.x.shape == (3, 14)
    assert len(NODE_FEATURE_NAMES) == 14
    # Check incoming and outgoing interactions are included
    # agent_b has 1 incoming from a and 1 outgoing to c
    # node order: agent_a (0), agent_b (1), agent_c (2)
    incoming_idx = NODE_FEATURE_NAMES.index("incoming_interactions")
    outgoing_idx = NODE_FEATURE_NAMES.index("outgoing_interactions")
    assert data.x[1, incoming_idx].item() == 3.0
    assert data.x[1, outgoing_idx].item() == 1.0


# ── Test 3: Edge dimensions ──
def test_edge_dimensions():
    snap = create_synthetic_snapshot()
    data = snapshot_to_pyg_data(snap)

    assert data.edge_index.shape == (2, 2)
    assert data.edge_attr.shape == (2, 10)
    assert len(EDGE_FEATURE_NAMES) == 10
    assert data.edge_weight.shape == (2,)


# ── Test 4: Batch creation with PyG DataLoader ──
def test_batch_creation_dataloader():
    snap1 = create_synthetic_snapshot(timestamp=1.0, run_id="r1")
    snap2 = create_synthetic_snapshot(timestamp=2.0, run_id="r2")
    d1 = snapshot_to_pyg_data(snap1, label=1.0, sample_id="s1")
    d2 = snapshot_to_pyg_data(snap2, label=0.0, sample_id="s2")

    loader = DataLoader([d1, d2], batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert batch.num_graphs == 2
    assert batch.x.shape == (6, 14)
    assert batch.edge_index.shape == (2, 4)
    assert batch.y.shape == (2,)
    assert batch.sample_id == ["s1", "s2"]


# ── Test 5: GCN forward pass ──
def test_gcn_forward_pass():
    snap = create_synthetic_snapshot()
    d1 = snapshot_to_pyg_data(snap, label=1.0)
    d2 = snapshot_to_pyg_data(snap, label=0.0)
    loader = DataLoader([d1, d2], batch_size=2)
    batch = next(iter(loader))

    model = GCNBaseline(node_in_dim=14, hidden_dim=32, num_layers=2)
    logits = model(
        x=batch.x,
        edge_index=batch.edge_index,
        batch=batch.batch,
        edge_weight=batch.edge_weight,
    )
    assert logits.shape == (2,)
    assert not torch.isnan(logits).any()


# ── Test 6: GAT forward pass ──
def test_gat_forward_pass():
    snap = create_synthetic_snapshot()
    d1 = snapshot_to_pyg_data(snap, label=1.0)
    d2 = snapshot_to_pyg_data(snap, label=0.0)
    loader = DataLoader([d1, d2], batch_size=2)
    batch = next(iter(loader))

    model = GATBaseline(node_in_dim=14, edge_dim=10, hidden_dim=32, num_layers=2, heads=4)
    logits = model(
        x=batch.x,
        edge_index=batch.edge_index,
        batch=batch.batch,
        edge_attr=batch.edge_attr,
    )
    assert logits.shape == (2,)
    assert not torch.isnan(logits).any()


# ── Test 7: Graph pooling (mean, max, both) ──
def test_graph_pooling_modes():
    snap = create_synthetic_snapshot()
    d = snapshot_to_pyg_data(snap)
    loader = DataLoader([d], batch_size=1)
    batch = next(iter(loader))

    for pool_mode in ["mean", "max", "both"]:
        model = GCNBaseline(node_in_dim=14, hidden_dim=16, pooling=pool_mode)
        out = model(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (1,)
        assert not torch.isnan(out).any()


# ── Test 8: Output shape validation ──
def test_output_shape():
    graphs = [snapshot_to_pyg_data(create_synthetic_snapshot(), label=float(i % 2)) for i in range(5)]
    loader = DataLoader(graphs, batch_size=5)
    batch = next(iter(loader))

    for ModelClass in [GCNBaseline, GATBaseline]:
        model = ModelClass(node_in_dim=14, hidden_dim=16)
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=getattr(batch, "edge_attr", None))
        assert out.shape == (5,)


# ── Test 9: Probability range [0.0, 1.0] ──
def test_probability_range():
    snap = create_synthetic_snapshot()
    graphs = [snapshot_to_pyg_data(snap, label=0.0), snapshot_to_pyg_data(snap, label=1.0)]
    loader = DataLoader(graphs, batch_size=2)

    for ModelClass in [GCNBaseline, GATBaseline]:
        model = ModelClass(node_in_dim=14, hidden_dim=16)
        probs = model.predict_proba(loader)
        assert len(probs) == 2
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


# ── Test 10: Training step (loss backward and parameter update) ──
def test_training_step():
    torch.manual_seed(42)
    snap = create_synthetic_snapshot()
    graphs = [
        snapshot_to_pyg_data(snap, label=1.0),
        snapshot_to_pyg_data(snap, label=0.0),
        snapshot_to_pyg_data(snap, label=1.0),
    ]
    loader = DataLoader(graphs, batch_size=3)

    model = GCNBaseline(node_in_dim=14, hidden_dim=16)
    initial_param = next(model.parameters()).clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.BCEWithLogitsLoss()

    batch = next(iter(loader))
    logits = model(batch.x, batch.edge_index, batch.batch)
    loss = criterion(logits, batch.y)
    loss.backward()
    optimizer.step()

    updated_param = next(model.parameters())
    assert not torch.equal(initial_param, updated_param)
    assert loss.item() > 0.0


# ── Test 11: Checkpoint save and load ──
def test_checkpoint_save_and_load(tmp_path):
    model = GCNBaseline(node_in_dim=14, hidden_dim=32, num_layers=2)
    ckpt_file = tmp_path / "test_gcn_ckpt.pt"

    meta = {"horizon": 5, "seed": 42, "threshold": 0.45}
    saved_path = model.save_checkpoint(ckpt_file, metadata=meta)
    assert saved_path.exists()

    new_model = GCNBaseline(node_in_dim=14, hidden_dim=32, num_layers=2)
    new_model.load_checkpoint(saved_path)

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


# ── Test 12: Run-level dataset split isolation ──
def test_dataset_split_isolation():
    dataset_path = Path("data/processed/agentguard_dataset_v1")
    if not dataset_path.exists():
        pytest.skip("Processed dataset not found.")

    storage = DatasetStorage(dataset_path.parent)
    train_rows = storage.load_tabular(dataset_path / "train.parquet")
    val_rows = storage.load_tabular(dataset_path / "val.parquet")
    test_rows = storage.load_tabular(dataset_path / "test.parquet")

    train_runs = {r["run_id"] for r in train_rows}
    val_runs = {r["run_id"] for r in val_rows}
    test_runs = {r["run_id"] for r in test_rows}

    assert train_runs.isdisjoint(val_runs), "Leakage detected: train_runs & val_runs not disjoint!"
    assert train_runs.isdisjoint(test_runs), "Leakage detected: train_runs & test_runs not disjoint!"
    assert val_runs.isdisjoint(test_runs), "Leakage detected: val_runs & test_runs not disjoint!"


# ── Test 13: Future leakage prevention in graph representations ──
def test_future_leakage_assertion():
    """Verify that graph snapshot G(t) only includes events <= t."""
    from ml.telemetry.schemas import AgentTelemetryEvent
    from ml.graph.graph_builder import TemporalGraphBuilder

    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(event_id="e1", timestamp=1.0, step_idx=0, source_agent="p1", target_agent="r1", event_type="message"),
        AgentTelemetryEvent(event_id="e2", timestamp=2.0, step_idx=1, source_agent="r1", target_agent="c1", event_type="message"),
        # Future event with timestamp > 2.0
        AgentTelemetryEvent(event_id="e3", timestamp=3.0, step_idx=2, source_agent="c1", target_agent="v1", event_type="error", failure_label=1),
    ]

    snapshot = builder.build_snapshot(events=events, timestamp=2.0, step_idx=1, run_id="run_leakage_test")

    # The snapshot should NOT contain e3
    assert snapshot.timestamp == 2.0
    assert snapshot.step_idx == 1
    # Edge c1->v1 must not exist
    assert not snapshot.has_edge("c1", "v1")
    # c1 should not have error count 1 in snapshot
    if snapshot.has_node("c1"):
        assert snapshot.nodes["c1"].get("error_count", 0) == 0


# ── Test 14: Reproducibility test ──
def test_reproducibility():
    """Verify deterministic training produces identical losses with the same seed."""
    from ml.baselines.static_gnn.experiment import set_seed

    def run_trial(seed):
        set_seed(seed)
        snap = create_synthetic_snapshot()
        graphs = [
            snapshot_to_pyg_data(snap, label=1.0),
            snapshot_to_pyg_data(snap, label=0.0),
        ]
        loader = DataLoader(graphs, batch_size=2)
        model = GCNBaseline(node_in_dim=14, hidden_dim=16)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        crit = torch.nn.BCEWithLogitsLoss()

        batch = next(iter(loader))
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = crit(out, batch.y)
        loss.backward()
        opt.step()
        return loss.item()

    loss1 = run_trial(42)
    loss2 = run_trial(42)
    assert abs(loss1 - loss2) < 1e-6


# ── Test 15: CPU execution guarantee ──
def test_cpu_execution():
    device = get_device(verbose=False)
    # Even if CUDA were unavailable, device is CPU and execution works
    snap = create_synthetic_snapshot()
    data = snapshot_to_pyg_data(snap).to(torch.device("cpu"))
    model = GCNBaseline(node_in_dim=14, hidden_dim=16).to(torch.device("cpu"))
    logits = model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long))
    assert logits.device == torch.device("cpu")
    assert logits.shape == (1,)


# ── Test 16: Metric calculation ──
def test_metric_calculation():
    snap = create_synthetic_snapshot()
    graphs = [
        snapshot_to_pyg_data(snap, label=1.0, sample_id="s1"),
        snapshot_to_pyg_data(snap, label=0.0, sample_id="s2"),
        snapshot_to_pyg_data(snap, label=1.0, sample_id="s3"),
        snapshot_to_pyg_data(snap, label=0.0, sample_id="s4"),
    ]
    loader = DataLoader(graphs, batch_size=2)
    model = GCNBaseline(node_in_dim=14, hidden_dim=16)

    res = evaluate_static_gnn(model, loader, threshold=0.50)
    metrics = res["metrics"]

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "false_positive_rate" in metrics
    assert len(res["predictions"]) == 4


# ── Test 17: Lead-time calculation ──
def test_lead_time_calculation():
    snap1 = create_synthetic_snapshot(timestamp=1.0, run_id="r1")
    snap2 = create_synthetic_snapshot(timestamp=2.0, run_id="r1")
    snap3 = create_synthetic_snapshot(timestamp=3.0, run_id="r1")

    # t=1: prediction 1, ground truth 0 -> early warning at t=1.0
    # t=3: failure at t=3.0
    d1 = snapshot_to_pyg_data(snap1, label=0.0, sample_id="s1")
    d2 = snapshot_to_pyg_data(snap2, label=0.0, sample_id="s2")
    d3 = snapshot_to_pyg_data(snap3, label=1.0, sample_id="s3")

    loader = DataLoader([d1, d2, d3], batch_size=3)

    class MockModel(torch.nn.Module):
        model_name = "mock_gnn"
        def forward(self, *args, **kwargs):
            # Output high logit for s1 (warning), low for s2, high for s3
            return torch.tensor([5.0, -5.0, 5.0])

    res = evaluate_static_gnn(MockModel(), loader, threshold=0.50)
    lead_time_stats = res["lead_time"]

    # Warning at t=1.0, failure at t=3.0 -> lead_time = 3.0 - 1.0 = 2.0
    assert lead_time_stats["successful_early_warnings"] == 1
    assert lead_time_stats["mean_lead_time"] == 2.0
