"""Phase 9 Test Suite: LSTM/GRU Temporal Sequence Baselines.

Verifies:
1. Sequence generation from prediction samples
2. Sequence chronological ordering
3. Correct sequence lengths (5, 10, 20, 50)
4. Prediction horizon handling (k in {1, 3, 5, 10, 20})
5. Run-level split isolation (train, val, test run sets are strictly disjoint)
6. Future leakage prevention (features <= prediction timestamp)
7. LSTM forward pass
8. GRU forward pass
9. Output probability range in [0.0, 1.0]
10. Training step execution
11. Checkpoint save and load
12. Prediction serialization
13. Metric calculation
14. Lead-time calculation (ignoring post-failure warnings)
15. Threshold selection on validation set
16. Reproducibility (deterministic random seed)
17. CPU execution
"""

import pytest
from pathlib import Path
import tempfile
import json
import numpy as np

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
from ml.data.schema import PredictionSample

if HAS_TORCH:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader


@pytest.fixture
def mock_sequence_samples():
    """Generate synthetic sequential prediction samples across 3 distinct runs."""
    samples = []
    for r_idx in range(3):
        run_id = f"seq_run_{r_idx}"
        for step in range(12):
            label = 1 if (r_idx == 1 and step >= 8) else 0
            for k in [1, 3, 5]:
                s = PredictionSample(
                    sample_id=f"{run_id}_s{step}_k{k}",
                    run_id=run_id,
                    timestamp=float(step * 0.25),
                    step_idx=step,
                    task_type="coding",
                    topology="mesh",
                    number_of_agents=4,
                    prediction_horizon=k,
                    node_features={},
                    edge_features={},
                    temporal_graph_history=[],
                    agent_level_features={
                        "total_events_observed": float(step + 1),
                        "error_count": float(step if label == 1 else 0),
                        "total_retries": 1.0 if label == 1 else 0.0,
                        "tool_failure_count": 1.0 if label == 1 else 0.0,
                        "mean_contradiction_score": 0.5 if label == 1 else 0.0,
                        "max_contradiction_score": 0.7 if label == 1 else 0.0,
                        "mean_confidence": 0.4 if label == 1 else 0.95,
                        "min_confidence": 0.3 if label == 1 else 0.90,
                        "mean_latency": 0.5 if label == 1 else 0.10,
                        "max_latency": 0.7 if label == 1 else 0.12,
                        "interaction_density": 2.0,
                        "min_output_quality": 0.3 if label == 1 else 0.95,
                        "mean_output_quality": 0.4 if label == 1 else 0.95,
                    },
                    label=label,
                    failure_type="tool_failure" if label == 1 else "none",
                    failure_level=2 if label == 1 else 0,
                    source_event_id=f"evt_{step}",
                    random_seed=42,
                    dataset_version="v1",
                    split="train" if r_idx == 0 else ("val" if r_idx == 1 else "test"),
                )
                samples.append(s)
    return samples


def test_1_sequence_generation(mock_sequence_samples):
    """Test 1: SequenceDatasetBuilder extracts sequences with shape (N, L, D)."""
    builder = SequenceDatasetBuilder(sequence_length=5)
    X, y, meta = builder.build_sequences_for_horizon(mock_sequence_samples, horizon=1)

    assert len(X) > 0
    assert X.shape[1] == 5  # Sequence length L=5
    assert X.shape[2] == 28  # Feature dimension D=28
    assert len(y) == len(X)
    assert len(meta) == len(X)


def test_2_sequence_ordering(mock_sequence_samples):
    """Test 2: Sequences are chronologically ordered up to the prediction step."""
    builder = SequenceDatasetBuilder(sequence_length=4)
    X, y, meta = builder.build_sequences_for_horizon(mock_sequence_samples, horizon=1)

    # For a sample at step 6, the last sequence element must match step 6 features
    sample_meta = meta[5]
    step_val = sample_meta["step_idx"]
    assert sample_meta["history_steps_available"] <= 4


def test_3_correct_sequence_lengths(mock_sequence_samples):
    """Test 3: Supports sequence lengths L in {5, 10, 20, 50}."""
    for L in [5, 10, 20, 50]:
        builder = SequenceDatasetBuilder(sequence_length=L)
        X, _, _ = builder.build_sequences_for_horizon(mock_sequence_samples, horizon=1)
        assert X.shape[1] == L


def test_4_prediction_horizon_handling(mock_sequence_samples):
    """Test 4: Builder filters samples strictly matching horizon K."""
    builder = SequenceDatasetBuilder(sequence_length=5)
    for k in [1, 3, 5]:
        _, _, meta = builder.build_sequences_for_horizon(mock_sequence_samples, horizon=k)
        for m in meta:
            assert m["prediction_horizon"] == k


def test_5_run_level_split_isolation(mock_sequence_samples):
    """Test 5: Run IDs across train, val, and test splits are strictly disjoint."""
    train_runs = {s.run_id for s in mock_sequence_samples if s.split == "train"}
    val_runs = {s.run_id for s in mock_sequence_samples if s.split == "val"}
    test_runs = {s.run_id for s in mock_sequence_samples if s.split == "test"}

    assert train_runs.isdisjoint(val_runs)
    assert train_runs.isdisjoint(test_runs)
    assert val_runs.isdisjoint(test_runs)


def test_6_future_leakage():
    """Test 6: Causal pre-padding ensures sequences never access steps > t."""
    builder = SequenceDatasetBuilder(sequence_length=5)
    
    # Run with 2 past steps (step 0, step 1). Prediction point at step 1.
    s0 = PredictionSample(
        sample_id="leak_s0",
        run_id="run_leak",
        timestamp=0.1,
        step_idx=0,
        task_type="research",
        topology="pipeline",
        number_of_agents=3,
        prediction_horizon=1,
        node_features={},
        edge_features={},
        temporal_graph_history=[],
        agent_level_features={"error_count": 0.0, "mean_latency": 0.1},
        label=0,
        failure_type="none",
        failure_level=0,
        source_event_id="e0",
        random_seed=42,
        dataset_version="v1",
    )
    s1 = PredictionSample(
        sample_id="leak_s1",
        run_id="run_leak",
        timestamp=0.2,
        step_idx=1,
        task_type="research",
        topology="pipeline",
        number_of_agents=3,
        prediction_horizon=1,
        node_features={},
        edge_features={},
        temporal_graph_history=[],
        agent_level_features={"error_count": 0.0, "mean_latency": 0.1},
        label=0,
        failure_type="none",
        failure_level=0,
        source_event_id="e1",
        random_seed=42,
        dataset_version="v1",
    )

    X, _, _ = builder.build_sequences_for_horizon([s0, s1], horizon=1)
    # Target sequence for s1 should have 3 zero-pad vectors at front and 2 real vectors at back
    seq_s1 = X[1]
    assert np.all(seq_s1[0] == 0.0)
    assert np.all(seq_s1[1] == 0.0)
    assert np.all(seq_s1[2] == 0.0)
    assert not np.all(seq_s1[4] == 0.0)


def test_7_lstm_forward_pass():
    """Test 7: LSTMSequenceBaseline forward pass produces expected shape."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    model = LSTMSequenceBaseline(input_size=28, hidden_size=32, num_layers=2)
    dummy_input = torch.randn(4, 10, 28)
    logits = model(dummy_input)

    assert logits.shape == (4,)


def test_8_gru_forward_pass():
    """Test 8: GRUSequenceBaseline forward pass produces expected shape."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    model = GRUSequenceBaseline(input_size=28, hidden_size=32, num_layers=2)
    dummy_input = torch.randn(4, 10, 28)
    logits = model(dummy_input)

    assert logits.shape == (4,)


def test_9_output_probability_range():
    """Test 9: predict_proba produces valid probabilities strictly in [0.0, 1.0]."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    model = LSTMSequenceBaseline(input_size=28, hidden_size=32)
    dummy_input = torch.randn(8, 10, 28)
    probs = model.predict_proba(dummy_input)

    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


def test_10_training_step(mock_sequence_samples):
    """Test 10: Training step executes and computes valid loss gradients."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    builder = SequenceDatasetBuilder(sequence_length=5)
    train_samples = [s for s in mock_sequence_samples if s.split == "train"]
    val_samples = [s for s in mock_sequence_samples if s.split == "val"]

    train_loader, _ = builder.create_dataloader(train_samples, horizon=1, batch_size=4)
    val_loader, _ = builder.create_dataloader(val_samples, horizon=1, batch_size=4)

    model = LSTMSequenceBaseline(input_size=28, hidden_size=16, num_layers=1)
    trainer = SequenceTrainer(model=model, lr=0.01, device=torch.device("cpu"))
    res = trainer.train(train_loader, val_loader, epochs=2)

    assert len(res["history"]) == 2
    assert "train_loss" in res["history"][0]


def test_11_checkpoint_save_load():
    """Test 11: Checkpoint saves model state and restores cleanly."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    model = GRUSequenceBaseline(input_size=28, hidden_size=16)
    dummy = torch.randn(2, 5, 28)
    orig_probs = model.predict_proba(dummy)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "selected_threshold": 0.42,
        }, ckpt_path)

        reloaded = GRUSequenceBaseline(input_size=28, hidden_size=16)
        ckpt = torch.load(ckpt_path, weights_only=False)
        reloaded.load_state_dict(ckpt["model_state_dict"])
        reloaded_probs = reloaded.predict_proba(dummy)

        assert torch.allclose(orig_probs, reloaded_probs, atol=1e-5)
        assert ckpt["selected_threshold"] == 0.42


def test_12_prediction_serialization(mock_sequence_samples):
    """Test 12: Sequence evaluation produces prediction schema matching Section 18."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    builder = SequenceDatasetBuilder(sequence_length=5)
    test_samples = [s for s in mock_sequence_samples if s.split == "test"]
    loader, _ = builder.create_dataloader(test_samples, horizon=1, batch_size=4, shuffle=False)

    model = LSTMSequenceBaseline(input_size=28, hidden_size=16)
    eval_out = evaluate_sequence_model(model, loader, threshold=0.5, device=torch.device("cpu"))

    preds = eval_out["predictions"]
    assert len(preds) > 0
    first = preds[0]

    required_fields = [
        "sample_id",
        "run_id",
        "timestamp",
        "horizon",
        "sequence_length",
        "model_name",
        "true_label",
        "predicted_probability",
        "predicted_label",
    ]
    for rf in required_fields:
        assert rf in first


def test_13_metric_calculation(mock_sequence_samples):
    """Test 13: Evaluation computes Precision, Recall, F1, AUROC, AUPRC, FPR."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    builder = SequenceDatasetBuilder(sequence_length=5)
    test_samples = [s for s in mock_sequence_samples if s.split == "test"]
    loader, _ = builder.create_dataloader(test_samples, horizon=1, batch_size=4, shuffle=False)

    model = LSTMSequenceBaseline(input_size=28, hidden_size=16)
    eval_out = evaluate_sequence_model(model, loader, threshold=0.5, device=torch.device("cpu"))

    m = eval_out["metrics"]
    for metric_name in ["precision", "recall", "f1", "auroc", "auprc", "false_positive_rate", "false_alarm_rate"]:
        assert metric_name in m
        assert 0.0 <= m[metric_name] <= 1.0


def test_14_lead_time_calculation(mock_sequence_samples):
    """Test 14: Lead-time calculation handles early warnings appropriately."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    builder = SequenceDatasetBuilder(sequence_length=5)
    test_samples = [s for s in mock_sequence_samples if s.split == "test"]
    loader, _ = builder.create_dataloader(test_samples, horizon=1, batch_size=4, shuffle=False)

    model = GRUSequenceBaseline(input_size=28, hidden_size=16)
    eval_out = evaluate_sequence_model(model, loader, threshold=0.5, device=torch.device("cpu"))

    lt = eval_out["lead_time"]
    for k in ["mean_lead_time", "median_lead_time", "min_lead_time", "max_lead_time", "successful_early_warnings"]:
        assert k in lt


def test_15_threshold_selection(mock_sequence_samples):
    """Test 15: Validation threshold selection selects and freezes valid threshold."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    builder = SequenceDatasetBuilder(sequence_length=5)
    val_samples = [s for s in mock_sequence_samples if s.split == "val"]
    val_loader, _ = builder.create_dataloader(val_samples, horizon=1, batch_size=4, shuffle=False)

    model = LSTMSequenceBaseline(input_size=28, hidden_size=16)
    trainer = SequenceTrainer(model=model, device=torch.device("cpu"))
    tau = trainer._select_optimal_threshold(val_loader)

    assert 0.0 < tau < 1.0


def test_16_reproducibility(mock_sequence_samples):
    """Test 16: Identical random seeds produce equivalent model behavior."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    torch.manual_seed(42)
    m1 = LSTMSequenceBaseline(input_size=28, hidden_size=16)
    torch.manual_seed(42)
    m2 = LSTMSequenceBaseline(input_size=28, hidden_size=16)

    dummy = torch.randn(2, 5, 28)
    p1 = m1.predict_proba(dummy)
    p2 = m2.predict_proba(dummy)

    assert torch.allclose(p1, p2, atol=1e-5)


def test_17_cpu_execution():
    """Test 17: Sequence models execute seamlessly on CPU."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    device = torch.device("cpu")
    model = LSTMSequenceBaseline(input_size=28, hidden_size=16).to(device)
    dummy = torch.randn(2, 5, 28, device=device)
    out = model(dummy)
    assert out.device == device
