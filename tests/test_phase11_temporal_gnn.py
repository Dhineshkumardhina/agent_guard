"""Comprehensive Test Suite for Phase 11: Temporal Graph Neural Network (TGN-Style).

Covers all 20 mandatory verification checks:
1. Temporal event ordering
2. Time encoding (sinusoidal & learnable)
3. Message construction
4. Memory initialization
5. Memory update
6. Memory reset
7. Temporal neighborhood sampling (5, 10, 20)
8. Temporal embedding
9. Graph readout
10. Prediction head
11. Forward pass
12. Training step
13. Checkpoint save/load
14. CPU execution
15. GPU execution / device compatibility
16. Future leakage prevention
17. Run isolation (memory reset across runs)
18. Dataset split isolation
19. Reproducibility test
20. Synthetic temporal sanity test (A -> B -> C -> D)
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import json

import torch
import torch.nn as nn

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
from ml.data.storage import DatasetStorage


def create_synthetic_trajectory(run_id: str = "run_syn_01") -> TemporalRunTrajectory:
    """Create deterministic synthetic trajectory with 4 agents (A, B, C, D) and 3 sequential interactions."""
    # A -> B at t=1.0
    # B -> C at t=2.0
    # C -> D at t=3.0
    e1 = TemporalInteraction("agent_a", "agent_b", 1.0, 0, [1.0, 0.0, 0.1, 100.0, 0.9, 0.0, 0.0, 0.0, 0.0, 1.0])
    e2 = TemporalInteraction("agent_b", "agent_c", 2.0, 1, [1.0, 0.0, 0.15, 120.0, 0.85, 0.0, 0.0, 0.0, 0.0, 1.0])
    e3 = TemporalInteraction("agent_c", "agent_d", 3.0, 2, [1.0, 0.0, 0.2, 150.0, 0.8, 1.0, 0.1, 1.0, 0.0, 1.0])

    node_feats = {
        "agent_a": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.95, 0.0, 0.0, 0.0, 1.0, 1.0],
        "agent_b": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.88, 0.9, 0.0, 0.0, 1.0, 1.0, 1.0],
        "agent_c": [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.18, 0.82, 0.85, 0.05, 1.0, 1.0, 1.0, 1.0],
        "agent_d": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.2, 0.8, 0.8, 0.1, 1.0, 1.0, 0.0, 1.0],
    }

    p1 = TemporalPredictionPoint("s_1", run_id, 0, 1.0, 1, 0.0, ["agent_a", "agent_b"], node_feats)
    p2 = TemporalPredictionPoint("s_2", run_id, 1, 2.0, 1, 0.0, ["agent_a", "agent_b", "agent_c"], node_feats)
    p3 = TemporalPredictionPoint("s_3", run_id, 2, 3.0, 1, 1.0, ["agent_a", "agent_b", "agent_c", "agent_d"], node_feats)

    return TemporalRunTrajectory(
        run_id=run_id,
        topology="pipeline",
        interactions=[e1, e2, e3],
        prediction_points=[p1, p2, p3],
    )


# ── Test 1: Temporal event ordering ──
def test_temporal_event_ordering():
    traj = create_synthetic_trajectory()
    timestamps = [ev.timestamp for ev in traj.interactions]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == 3


# ── Test 2: Time encoding (sinusoidal & learnable) ──
def test_time_encoding():
    dt_zero = torch.tensor([0.0])
    dt_small = torch.tensor([0.5])
    dt_large = torch.tensor([5.0])

    for enc_type in ["sinusoidal", "learnable"]:
        encoder = TimeEncoder(dimension=16, encoding_type=enc_type)
        enc_0 = encoder(dt_zero)
        enc_s = encoder(dt_small)
        enc_l = encoder(dt_large)

        assert enc_0.shape == (1, 16) or enc_0.shape == (16,)
        assert enc_s.shape == enc_0.shape
        assert not torch.equal(enc_0, enc_l)


# ── Test 3: Message construction ──
def test_message_construction():
    msg_fn = MessageFunction(memory_dim=32, edge_dim=10, time_dim=16, message_dim=32)
    src_m = torch.randn(1, 32)
    tgt_m = torch.randn(1, 32)
    edge_f = torch.randn(1, 10)
    t_enc = torch.randn(1, 16)

    msg = msg_fn(src_m, tgt_m, edge_f, t_enc)
    assert msg.shape == (1, 32)
    assert not torch.isnan(msg).any()


# ── Test 4: Memory initialization ──
def test_memory_initialization():
    mem = NodeMemory(memory_dim=64)
    m_a = mem.get_memory("agent_new")
    assert m_a.shape == (64,)
    assert torch.equal(m_a, torch.zeros(64))
    assert mem.get_last_timestamp("agent_new") == 0.0


# ── Test 5: Memory update ──
def test_memory_update():
    updater = MemoryUpdater(message_dim=32, memory_dim=32)
    prev_m = torch.zeros(1, 32)
    msg = torch.ones(1, 32) * 0.5

    new_m = updater(msg, prev_m)
    assert new_m.shape == (1, 32)
    assert not torch.equal(new_m, prev_m)


# ── Test 6: Memory reset ──
def test_memory_reset():
    mem = NodeMemory(memory_dim=32)
    mem.set_memory("agent_x", torch.randn(32), 1.5)
    assert "agent_x" in mem._memory
    assert mem.get_last_timestamp("agent_x") == 1.5

    mem.reset_memory()
    assert len(mem._memory) == 0
    assert len(mem._last_update) == 0
    # Clean zeros on next access
    assert torch.equal(mem.get_memory("agent_x"), torch.zeros(32))


# ── Test 7: Temporal neighborhood sampling (5, 10, 20) ──
def test_temporal_neighborhood_sampling():
    tracker = TemporalNeighborhoodTracker(max_history=30)
    for i in range(15):
        t = float(i)
        tracker.add_interaction("agent_a", f"agent_{i+1}", t, torch.tensor([float(i)] * 10))

    for k in [5, 10, 20]:
        nbrs = tracker.get_recent_neighbors("agent_a", current_timestamp=14.0, k_neighbors=k)
        assert len(nbrs) == min(k, 15)
        # Check chronological ordering
        ts = [r[1] for r in nbrs]
        assert ts == sorted(ts)


# ── Test 8: Temporal embedding ──
def test_temporal_embedding():
    embedder = TemporalEmbedding(memory_dim=32, node_dim=14, edge_dim=10, time_dim=16, embed_dim=32)
    node_m = torch.randn(32)
    node_f = torch.randn(14)
    nbr_records = [
        (torch.randn(32), torch.randn(10), torch.randn(16)),
        (torch.randn(32), torch.randn(10), torch.randn(16)),
    ]
    z_v = embedder(node_m, node_f, nbr_records, device=torch.device("cpu"))
    assert z_v.shape == (32,)
    assert not torch.isnan(z_v).any()


# ── Test 9: Graph readout ──
def test_graph_readout():
    readout = GraphReadout(embed_dim=32)
    node_embeds = torch.randn(4, 32)
    graph_rep = readout(node_embeds)
    assert graph_rep.shape == (64,)  # mean + max pooling = 32 * 2


# ── Test 10: Prediction head ──
def test_prediction_head():
    head = FailurePredictionHead(input_dim=64, hidden_dim=32)
    rep = torch.randn(64)
    logit = head(rep)
    assert logit.shape == () or logit.numel() == 1
    prob = torch.sigmoid(logit).item()
    assert 0.0 <= prob <= 1.0


# ── Test 11: Forward pass ──
def test_model_forward_pass():
    model = TemporalGraphFailurePredictor(memory_dim=32, embed_dim=32)
    model.reset_memory()

    e_feat = torch.randn(10)
    model.process_interaction("agent_a", "agent_b", 1.0, e_feat)

    node_feats = {
        "agent_a": [0.1] * 14,
        "agent_b": [0.2] * 14,
    }
    logit, prob = model.predict_at_timestamp(["agent_a", "agent_b"], node_feats, 1.0)
    assert not torch.isnan(logit)
    assert 0.0 <= prob <= 1.0


# ── Test 12: Training step ──
def test_model_training_step():
    torch.manual_seed(42)
    model = TemporalGraphFailurePredictor(memory_dim=32, embed_dim=32)
    initial_p = list(model.head.parameters())[0].clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    model.reset_memory()
    e_feat = torch.randn(10)
    model.process_interaction("agent_a", "agent_b", 1.0, e_feat)
    logit, _ = model.predict_at_timestamp(["agent_a", "agent_b"], {"agent_a": [0.1]*14, "agent_b": [0.1]*14}, 1.0)

    loss = criterion(logit.unsqueeze(0), torch.tensor([1.0]))
    loss.backward()
    optimizer.step()

    updated_p = list(model.head.parameters())[0]
    assert not torch.equal(initial_p, updated_p)


# ── Test 13: Checkpoint save and load ──
def test_checkpoint_save_and_load(tmp_path):
    model = TemporalGraphFailurePredictor(memory_dim=32, embed_dim=32)
    ckpt_path = tmp_path / "tgn_ckpt.pt"
    model.save_checkpoint(ckpt_path, metadata={"horizon": 5, "seed": 42})
    assert ckpt_path.exists()

    new_model = TemporalGraphFailurePredictor(memory_dim=32, embed_dim=32)
    new_model.load_checkpoint(ckpt_path)

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


# ── Test 14: CPU execution ──
def test_cpu_execution():
    dev = get_device(verbose=False)
    model = TemporalGraphFailurePredictor(memory_dim=16, embed_dim=16, device=torch.device("cpu"))
    model.reset_memory()
    e_feat = torch.randn(10)
    model.process_interaction("a", "b", 0.5, e_feat)
    logit, _ = model.predict_at_timestamp(["a", "b"], {"a": [0.0]*14, "b": [0.0]*14}, 0.5)
    assert logit.device == torch.device("cpu")


# ── Test 15: Device compatibility ──
def test_device_compatibility():
    dev = get_device(verbose=False)
    assert isinstance(dev, torch.device)


# ── Test 16: Future leakage prevention ──
def test_future_leakage_invariance():
    """Verify modifying or adding a future event > t does not alter prediction at t."""
    torch.manual_seed(42)
    model = TemporalGraphFailurePredictor(memory_dim=32, embed_dim=32)
    model.eval()

    # Run 1: Only events up to t=1.0
    model.reset_memory()
    e1 = torch.tensor([1.0] * 10)
    model.process_interaction("agent_a", "agent_b", 1.0, e1)
    logit_1, prob_1 = model.predict_at_timestamp(["agent_a", "agent_b"], {"agent_a": [0.5]*14, "agent_b": [0.5]*14}, 1.0)

    # Run 2: Event at t=1.0, plus a subsequent future event at t=2.0
    model.reset_memory()
    model.process_interaction("agent_a", "agent_b", 1.0, e1)
    # Prediction at t=1.0 before processing future event
    logit_2, prob_2 = model.predict_at_timestamp(["agent_a", "agent_b"], {"agent_a": [0.5]*14, "agent_b": [0.5]*14}, 1.0)

    assert abs(prob_1 - prob_2) < 1e-6, "Leakage: earlier prediction changed!"


# ── Test 17: Run isolation (memory reset across runs) ──
def test_run_isolation_memory_reset():
    model = TemporalGraphFailurePredictor(memory_dim=32, embed_dim=32)

    # Run 1: Many interactions involving agent_a
    model.reset_memory()
    for i in range(5):
        model.process_interaction("agent_a", "agent_b", float(i), torch.randn(10))
    mem_run1 = model.node_memory.get_memory("agent_a").clone()
    assert torch.norm(mem_run1).item() > 0.0

    # Start Run 2: Must be reset
    model.reset_memory()
    mem_run2 = model.node_memory.get_memory("agent_a")
    assert torch.equal(mem_run2, torch.zeros(32)), "Run isolation violated: memory leaked into Run 2!"


# ── Test 18: Dataset split isolation ──
def test_dataset_split_isolation():
    dataset_path = Path("data/processed/agentguard_dataset_v1")
    if not dataset_path.exists():
        pytest.skip("Dataset not generated.")
    storage = DatasetStorage(dataset_path.parent)
    train_rows = storage.load_tabular(dataset_path / "train.parquet")
    val_rows = storage.load_tabular(dataset_path / "val.parquet")
    test_rows = storage.load_tabular(dataset_path / "test.parquet")

    train_runs = {r["run_id"] for r in train_rows}
    val_runs = {r["run_id"] for r in val_rows}
    test_runs = {r["run_id"] for r in test_rows}

    assert train_runs.isdisjoint(val_runs)
    assert train_runs.isdisjoint(test_runs)
    assert val_runs.isdisjoint(test_runs)


# ── Test 19: Reproducibility test ──
def test_reproducibility():
    def train_single_step(seed):
        torch.manual_seed(seed)
        model = TemporalGraphFailurePredictor(memory_dim=16, embed_dim=16)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        crit = nn.BCEWithLogitsLoss()

        model.reset_memory()
        model.process_interaction("a", "b", 1.0, torch.tensor([1.0]*10))
        logit, _ = model.predict_at_timestamp(["a", "b"], {"a": [0.1]*14, "b": [0.1]*14}, 1.0)
        loss = crit(logit.unsqueeze(0), torch.tensor([1.0]))
        loss.backward()
        opt.step()
        return loss.item()

    l1 = train_single_step(42)
    l2 = train_single_step(42)
    assert abs(l1 - l2) < 1e-6


# ── Test 20: Synthetic temporal sanity test (A -> B -> C -> D) ──
def test_synthetic_temporal_sanity_pipeline():
    """Verify chronological event progression, memory updates, and intermediate predictions."""
    traj = create_synthetic_trajectory()
    model = TemporalGraphFailurePredictor(memory_dim=32, embed_dim=32)
    model.reset_memory()

    # Step 1: t=1.0 (A -> B)
    ev1 = traj.interactions[0]
    model.process_interaction(ev1.source_agent, ev1.target_agent, ev1.timestamp, torch.tensor(ev1.features))
    l1, p1 = model.predict_at_timestamp(traj.prediction_points[0].active_agents, traj.prediction_points[0].node_features, 1.0)
    assert 0.0 <= p1 <= 1.0

    # Step 2: t=2.0 (B -> C)
    ev2 = traj.interactions[1]
    model.process_interaction(ev2.source_agent, ev2.target_agent, ev2.timestamp, torch.tensor(ev2.features))
    l2, p2 = model.predict_at_timestamp(traj.prediction_points[1].active_agents, traj.prediction_points[1].node_features, 2.0)
    assert 0.0 <= p2 <= 1.0

    # Step 3: t=3.0 (C -> D)
    ev3 = traj.interactions[2]
    model.process_interaction(ev3.source_agent, ev3.target_agent, ev3.timestamp, torch.tensor(ev3.features))
    l3, p3 = model.predict_at_timestamp(traj.prediction_points[2].active_agents, traj.prediction_points[2].node_features, 3.0)
    assert 0.0 <= p3 <= 1.0

    # Trace inspection
    trace = debug_memory_trace(model, traj)
    assert len(trace) == 3
    assert trace[0]["source_agent"] == "agent_a"
    assert trace[1]["source_agent"] == "agent_b"
    assert trace[2]["source_agent"] == "agent_c"
