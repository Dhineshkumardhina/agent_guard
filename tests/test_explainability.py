"""Tests for Phase 15 Explainability and Attribution Framework.

Validates:
1. Explanation schema integrity and serialization
2. Feature attribution computation and normalization
3. Node / agent attribution via node masking
4. Edge / interaction attribution via channel masking
5. Temporal event attribution with strict causality (no future leakage)
6. Counterfactual perturbation / sensitivity analysis
7. Explanation stability metrics
8. End-to-end explainability pipeline execution
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from ml.explainability.schema import (
    FeatureAttribution,
    AgentAttribution,
    InteractionAttribution,
    TemporalEventAttribution,
    PerturbationResult,
    CaseStudyExplanation,
    GlobalImportanceReport,
    ExplanationStabilityReport,
    CAUSALITY_DISCLAIMER,
)
from ml.explainability.classical_explainer import ClassicalModelExplainer
from ml.explainability.temporal_gnn_explainer import TemporalGNNExplainer
from ml.explainability.perturbation import PerturbationAnalyzer
from ml.explainability.runner import ExplainabilityRunner
from ml.baselines.classical_ml.models import RandomForestBaseline, LogisticRegressionBaseline
from ml.baselines.temporal_gnn.models import TemporalGraphFailurePredictor
from ml.baselines.temporal_gnn.dataset import (
    TemporalRunTrajectory,
    TemporalPredictionPoint,
    TemporalInteraction,
)


@pytest.fixture
def mock_trajectory():
    """Create a synthetic chronological run trajectory for unit testing."""
    interactions = [
        TemporalInteraction(source_agent="planner_0", target_agent="researcher_1", timestamp=1.0, step_idx=0, features=[1.0, 1.0, 0.2, 50.0, 0.95, 0.0, 0.0, 0.0, 0.0, 1.0]),
        TemporalInteraction(source_agent="researcher_1", target_agent="analyst_2", timestamp=2.0, step_idx=1, features=[1.0, 1.0, 0.5, 80.0, 0.85, 1.0, 0.2, 0.0, 0.0, 1.0]),
        TemporalInteraction(source_agent="analyst_2", target_agent="coder_3", timestamp=3.0, step_idx=2, features=[1.0, 1.0, 0.8, 120.0, 0.70, 2.0, 0.5, 1.0, 0.0, 1.0]),
        # Future event after cutoff t=3.5
        TemporalInteraction(source_agent="coder_3", target_agent="planner_0", timestamp=4.5, step_idx=3, features=[1.0, 1.0, 1.2, 100.0, 0.60, 3.0, 0.8, 1.0, 1.0, 1.0]),
    ]
    node_feats = {
        "planner_0": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.95, 0.95, 0.0, 0.0, 1.0, 1.0, 1.0],
        "researcher_1": [2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.3, 0.88, 0.90, 0.1, 0.0, 1.0, 1.0, 1.0],
        "analyst_2": [3.0, 3.0, 2.0, 1.0, 2.0, 0.0, 0.6, 0.75, 0.70, 0.4, 1.0, 1.0, 1.0, 1.0],
        "coder_3": [2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 0.9, 0.65, 0.60, 0.6, 1.0, 1.0, 1.0, 1.0],
    }
    pred_pt = TemporalPredictionPoint(
        sample_id="test_run_s2_k1",
        run_id="test_run",
        step_idx=2,
        timestamp=3.5,
        prediction_horizon=1,
        label=1.0,
        active_agents=["planner_0", "researcher_1", "analyst_2", "coder_3"],
        node_features=node_feats,
    )
    return TemporalRunTrajectory(run_id="test_run", topology="pipeline", interactions=interactions, prediction_points=[pred_pt])


@pytest.fixture
def mock_tgnn_model():
    """Create a lightweight TemporalGraphFailurePredictor."""
    model = TemporalGraphFailurePredictor(
        node_in_dim=14,
        edge_in_dim=10,
        memory_dim=16,
        time_dim=8,
        embed_dim=16,
        neighbor_history=5,
        dropout=0.0,
        device=torch.device("cpu"),
    )
    model.eval()
    return model


def test_schema_serialization():
    """Test dataclass schema instantiation and JSON serialization."""
    fa = FeatureAttribution(feature_name="feat_retries", feature_group="reliability", importance_score=0.85, signed_contribution=0.32, rank=1)
    d = fa.to_dict()
    assert d["feature_name"] == "feat_retries"
    assert d["importance_score"] == 0.85

    cs = CaseStudyExplanation(
        explanation_id="exp_001",
        case_type="true_positive_early",
        sample_id="sample_01",
        run_id="run_01",
        topology="pipeline",
        task_type="research",
        prediction_timestamp=2.5,
        actual_failure_timestamp=4.0,
        prediction_horizon=1,
        predicted_probability=0.82,
        predicted_label=1,
        true_label=1,
        threshold=0.50,
        model_name="TemporalGNN",
        model_version="1.0.0",
        dataset_version="v1",
        seed=42,
    )
    cs_dict = cs.to_dict()
    assert cs_dict["case_type"] == "true_positive_early"
    formatted = cs.to_formatted_text()
    assert "EXPLANATION REPORT" in formatted
    assert CAUSALITY_DISCLAIMER in formatted


def test_classical_model_explainer():
    """Test global and local feature importance for classical models."""
    X = np.random.randn(50, 28)
    y = np.random.randint(0, 2, size=50)

    rf = RandomForestBaseline(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)

    expl = ClassicalModelExplainer(rf)
    global_attrs = expl.explain_global_importance(X_val=X, y_val=y, method="permutation")
    assert len(global_attrs) == 28
    assert all(isinstance(a, FeatureAttribution) for a in global_attrs)
    assert global_attrs[0].importance_score >= global_attrs[-1].importance_score

    # Instance-level explanation
    local_attrs = expl.explain_instance(X[0], top_k=5)
    assert len(local_attrs) == 5
    assert all(a.rank <= 5 for a in local_attrs)


def test_tgnn_agent_attribution(mock_trajectory, mock_tgnn_model):
    """Test Level 3 Agent Importance via node masking."""
    expl = TemporalGNNExplainer(model=mock_tgnn_model)
    pp = mock_trajectory.prediction_points[0]

    agent_attrs = expl.explain_agents(mock_trajectory, pp)
    assert len(agent_attrs) == len(pp.active_agents)
    assert all(isinstance(a, AgentAttribution) for a in agent_attrs)
    assert all(0.0 <= a.attribution_score <= 1.0 for a in agent_attrs)
    assert agent_attrs[0].rank == 1


def test_tgnn_interaction_attribution(mock_trajectory, mock_tgnn_model):
    """Test Level 4 Interaction Importance via communication channel masking."""
    expl = TemporalGNNExplainer(model=mock_tgnn_model)
    pp = mock_trajectory.prediction_points[0]

    inter_attrs = expl.explain_interactions(mock_trajectory, pp)
    assert len(inter_attrs) > 0
    assert all(isinstance(i, InteractionAttribution) for i in inter_attrs)
    assert inter_attrs[0].rank == 1


def test_tgnn_temporal_event_attribution_causality(mock_trajectory, mock_tgnn_model):
    """Test Level 5 Temporal Event Importance and verify strict causality (no future leakage)."""
    expl = TemporalGNNExplainer(model=mock_tgnn_model)
    pp = mock_trajectory.prediction_points[0]

    ev_attrs = expl.explain_temporal_events(mock_trajectory, pp)
    assert len(ev_attrs) > 0
    # Crucial causality check: all explained events must occur <= prediction cutoff (3.5s)
    # The 4th event at t=4.5s MUST NOT be included!
    for e in ev_attrs:
        assert e.timestamp <= pp.timestamp
        assert e.time_before_prediction >= 0.0


def test_tgnn_feature_attribution(mock_trajectory, mock_tgnn_model):
    """Test Level 2 Feature Importance for Temporal GNN."""
    expl = TemporalGNNExplainer(model=mock_tgnn_model)
    pp = mock_trajectory.prediction_points[0]

    feat_attrs = expl.explain_features(mock_trajectory, pp, top_k=6)
    assert len(feat_attrs) == 6
    assert all(isinstance(f, FeatureAttribution) for f in feat_attrs)
    assert all(f.feature_group in ("node", "edge") for f in feat_attrs)


def test_counterfactual_perturbations(mock_trajectory, mock_tgnn_model):
    """Test Level 10 Counterfactual Perturbation sensitivity analysis."""
    expl = TemporalGNNExplainer(model=mock_tgnn_model)
    analyzer = PerturbationAnalyzer(explainer=expl)
    pp = mock_trajectory.prediction_points[0]

    results = analyzer.run_all_perturbations(
        trajectory=mock_trajectory,
        prediction_point=pp,
        top_agent="analyst_2",
        top_edge=("researcher_1", "analyst_2"),
    )
    assert len(results) >= 5
    assert all(isinstance(r, PerturbationResult) for r in results)
    types = [r.perturbation_type for r in results]
    assert "reduce_contradiction_50pct" in types
    assert "reduce_retry_frequency_50pct" in types
    assert "remove_top_agent" in types


def test_signed_signals_extraction(mock_trajectory, mock_tgnn_model):
    """Test Level 8 extraction of positive (risk-increasing) and negative (mitigating) signals."""
    expl = TemporalGNNExplainer(model=mock_tgnn_model)
    pp = mock_trajectory.prediction_points[0]

    ag_attrs = expl.explain_agents(mock_trajectory, pp)
    inter_attrs = expl.explain_interactions(mock_trajectory, pp)
    feat_attrs = expl.explain_features(mock_trajectory, pp)

    pos, neg = expl.extract_signed_signals(feat_attrs, ag_attrs, inter_attrs)
    assert isinstance(pos, list) and len(pos) > 0
    assert isinstance(neg, list) and len(neg) > 0


def test_risk_trajectory_computation(mock_trajectory, mock_tgnn_model):
    """Test Level 6 Risk Trajectory generation t -> P(failure)."""
    expl = TemporalGNNExplainer(model=mock_tgnn_model)
    timeline = expl.compute_risk_trajectory(mock_trajectory)
    assert len(timeline) == len(mock_trajectory.prediction_points)
    assert "predicted_probability" in timeline[0]
    assert 0.0 <= timeline[0]["predicted_probability"] <= 1.0


def test_end_to_end_explainability_smoke(tmp_path):
    """Smoke test for full ExplainabilityRunner pipeline execution."""
    dataset_dir = Path("data/processed/agentguard_generalization_v1")
    if not dataset_dir.exists():
        pytest.skip("Dataset not available for full runner smoke test")

    runner = ExplainabilityRunner(dataset_dir=dataset_dir, results_dir=tmp_path)
    runner.load_dataset()
    assert len(runner.all_samples) > 0

    # Train for 1 epoch for smoke test
    runner.train_models(horizon=1, seed=42, epochs=1)
    assert runner.tgnn_model is not None

    # Global report
    rep = runner.generate_global_importance(horizon=1, seed=42)
    assert len(rep.feature_importance) > 0

    # Case studies
    cases = runner.select_case_studies(horizon=1, seed=42)
    assert len(cases) > 0

    # Stability
    stab = runner.evaluate_stability(seeds=[42, 123], horizons=[1])
    assert stab.feature_rank_correlation_seeds > 0.0

    # Save artifacts
    runner.save_artifacts()
    assert (tmp_path / "explainability_report.md").exists()
    assert (tmp_path / "dashboard_data" / "global_importance_report.json").exists()
    assert (tmp_path / "plots" / "global_feature_importance.png").exists()
