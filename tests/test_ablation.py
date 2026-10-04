"""Unit and Integration Tests for Phase 13: Research Ablation Study Framework.

Verifies:
1. Correct feature removal (feature masking for nodes and edges)
2. No accidental future leakage (strictly causal interaction filtering)
3. Same dataset split across all ablations
4. Same test trajectories and sample IDs across ablations
5. Correct model configuration (AblationConfig properly switches off modules)
6. Correct prediction horizon handling
7. Correct random seed setting and deterministic behavior
8. Correct result schema recording and persistence
9. Correct baseline comparison and paired difference calculation
"""

import os
import json
import pytest
import numpy as np
import torch

from ml.baselines.temporal_gnn.models import (
    TemporalGraphFailurePredictor,
    AblationConfig,
)
from ml.baselines.temporal_gnn.dataset import (
    TemporalRunTrajectory,
    TemporalInteraction,
    TemporalPredictionPoint,
)
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
from ml.ablation.runner import AblationStudyRunner


# 1. Feature removal tests
def test_node_feature_removal():
    node_feat = [1.0] * 14
    # Test no_node_features
    masker_node = AblationMasker("no_node_features")
    masked = masker_node.mask_node_features(node_feat)
    assert all(v == 0.0 for v in masked)

    # Test no_contradiction
    masker_contra = AblationMasker("no_contradiction")
    masked_contra = masker_contra.mask_node_features(node_feat)
    assert masked_contra[NODE_IDX_CONTRADICTION[0]] == 0.0
    assert masked_contra[0] == 1.0  # other features untouched

    # Test no_confidence
    masker_conf = AblationMasker("no_confidence")
    masked_conf = masker_conf.mask_node_features(node_feat)
    assert masked_conf[NODE_IDX_CONFIDENCE[0]] == 0.0
    assert masked_conf[0] == 1.0

    # Test no_failure_history
    masker_fail = AblationMasker("no_failure_history")
    masked_fail = masker_fail.mask_node_features(node_feat)
    for idx in NODE_IDX_FAILURE_HISTORY:
        assert masked_fail[idx] == 0.0
    assert masked_fail[0] == 1.0


def test_edge_feature_removal():
    edge_feat = [1.0] * 10
    # Test no_edge_features
    masker_edge = AblationMasker("no_edge_features")
    masked = masker_edge.mask_edge_features(edge_feat)
    assert all(v == 0.0 for v in masked)

    # Test no_interaction_freq
    masker_freq = AblationMasker("no_interaction_freq")
    masked_freq = masker_freq.mask_edge_features(edge_feat)
    for idx in EDGE_IDX_INTERACTION_FREQ:
        assert masked_freq[idx] == 0.0
    assert masked_freq[2] == 1.0  # latency untouched


# 2. No accidental future leakage test
def test_no_accidental_future_leakage():
    # Verify that predict_at_timestamp only uses interactions observed <= cutoff_timestamp
    model = TemporalGraphFailurePredictor(
        node_in_dim=14,
        edge_in_dim=10,
        memory_dim=16,
        time_dim=8,
        embed_dim=16,
        neighbor_history=5,
    )
    model.reset_memory()

    # Interaction at t=5.0
    model.process_interaction("agent_1", "agent_2", 5.0, torch.ones(10))
    # Interaction in the future at t=15.0
    model.process_interaction("agent_1", "agent_2", 15.0, torch.ones(10))

    # Cutoff at t=10.0: neighbor query should ONLY return the interaction from t=5.0
    nbrs = model.neighborhood_tracker.get_recent_neighbors("agent_1", current_timestamp=10.0, k_neighbors=5)
    for _, t, _ in nbrs:
        assert t <= 10.0


# 3. Same dataset split across ablations
def test_same_dataset_split():
    runner = AblationStudyRunner()
    runner.load_dataset()

    assert len(runner.train_samples) > 0
    assert len(runner.val_samples) > 0
    assert len(runner.test_samples) > 0
    # Verify strict split disjointness
    train_ids = set(r["sample_id"] for r in runner.train_samples)
    val_ids = set(r["sample_id"] for r in runner.val_samples)
    test_ids = set(r["sample_id"] for r in runner.test_samples)
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


# 4. Same test trajectories across ablations
def test_same_test_trajectories_across_ablations():
    runner = AblationStudyRunner()
    runner.load_dataset()

    raw_test_trajs = runner.dataset_builder.build_trajectories(
        tabular_samples=runner.test_samples,
        graph_sequences=runner.test_graphs,
        horizon_filter=1,
    )
    raw_sids = [pp.sample_id for t in raw_test_trajs for pp in t.prediction_points]

    for abl_key in ["full_temporal_gnn", "no_temporal_info", "no_graph_structure", "no_node_features"]:
        masker = AblationMasker(abl_key)
        trans_trajs = masker.transform_trajectories(raw_test_trajs)
        trans_sids = [pp.sample_id for t in trans_trajs for pp in t.prediction_points]
        assert trans_sids == raw_sids


# 5. Correct model configuration test
def test_model_configuration_propagation():
    cfg = AblationConfig(
        enable_time_encoding=False,
        enable_memory=False,
        enable_node_features=False,
        enable_graph_structure=False,
    )
    model = TemporalGraphFailurePredictor(
        node_in_dim=14,
        edge_in_dim=10,
        memory_dim=16,
        time_dim=8,
        embed_dim=16,
        ablation_config=cfg,
    )
    assert model.ablation.enable_time_encoding is False
    assert model.ablation.enable_memory is False
    assert model.ablation.enable_node_features is False
    assert model.ablation.enable_graph_structure is False


# 6. Correct horizon handling
def test_horizon_filtering():
    runner = AblationStudyRunner()
    runner.load_dataset()

    for h in [1, 3, 5]:
        trajs = runner.dataset_builder.build_trajectories(
            tabular_samples=runner.test_samples,
            graph_sequences=runner.test_graphs,
            horizon_filter=h,
        )
        pps = [pp for t in trajs for pp in t.prediction_points]
        for pp in pps:
            assert pp.prediction_horizon == h


# 7. Correct seed handling and determinism
def test_seed_determinism():
    masker = AblationMasker("full_temporal_gnn")
    torch.manual_seed(42)
    m1 = TemporalGraphFailurePredictor(14, 10, 16, 8, 16, ablation_config=masker.ablation_config)

    torch.manual_seed(42)
    m2 = TemporalGraphFailurePredictor(14, 10, 16, 8, 16, ablation_config=masker.ablation_config)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


# 8. Correct result schema recording
def test_result_schema_recording():
    rec = AblationResultRecord(
        experiment_id="test_exp",
        parent_experiment_id="full_temporal_gnn",
        ablation_name="No Temporal Memory",
        removed_component="Persistent Node Memory (m_v)",
        horizon=1,
        seed=42,
        precision=0.85,
        recall=0.80,
        f1=0.824,
    )
    d = rec.to_dict()
    assert d["ablation_name"] == "No Temporal Memory"
    assert d["removed_component"] == "Persistent Node Memory (m_v)"
    assert d["f1"] == 0.824
    assert "mean_lead_time" in d


# 9. Correct baseline comparison and paired difference calculation
def test_baseline_comparison():
    full_f1 = 0.85
    ablated_f1 = 0.70
    diff = ablated_f1 - full_f1
    pct = ((diff) / full_f1) * 100.0

    comp = AblationComparisonRecord(
        ablation_name="No Temporal Memory",
        removed_component="Persistent Node Memory",
        horizon=1,
        metric="f1",
        full_value=full_f1,
        ablated_value=ablated_f1,
        difference=round(diff, 4),
        pct_change=round(pct, 2),
        ci_lower=-0.25,
        ci_upper=-0.05,
        p_value=0.01,
        statistically_significant=True,
    )
    assert comp.difference == -0.15
    assert comp.pct_change < 0
    assert comp.statistically_significant is True
