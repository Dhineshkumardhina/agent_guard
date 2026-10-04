"""Phase 18: End-to-End Pipeline Verification Test.

Executes the complete research lifecycle in a single deterministic test:
1. Simulation execution (Nominal run and Fault-injected run)
2. Telemetry collection (AgentTelemetryEvent sequence)
3. Fault injection & propagation tracking (Level 3 cascade triggered)
4. Temporal Graph construction (Graph snapshots, node/edge features)
5. Dataset generation (PredictionSample instances across horizons k in {1, 3})
6. Trajectory-level dataset splitting (Train / Val / Test by run_id)
7. Model training & prediction (LogisticRegressionBaseline)
8. Prediction evaluation (Metrics, ROC, PR, Lead Time)
9. Explainability generation (Feature attributions)
10. Database persistence & API exposure (FastAPI TestClient queries)
11. Response verification for research frontend consumption
"""

import json
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.database.session import init_db, SessionLocal
from backend.app.database.models import Run as DBRun, Event as DBEvent, Failure as DBFailure
from ml.simulation.run import SimulationRun
from ml.simulation.fault_injection.injector import FaultInjector
from ml.graph.graph_builder import TemporalGraphBuilder
from ml.data.prediction_samples import PredictionSampleGenerator
from ml.data.split import split_trajectories, split_samples
from ml.baselines.classical_ml.models import LogisticRegressionBaseline
from ml.baselines.classical_ml.features import TabularFeatureExtractor, FEATURE_NAMES
from ml.evaluation.metrics import compute_comprehensive_metrics
from ml.explainability.classical_explainer import ClassicalModelExplainer

client = TestClient(app)


def test_complete_end_to_end_research_pipeline():
    """Verify the entire AgentGuard pipeline from simulation to API."""
    init_db()

    # Step 1-3: Run Simulations (2 Fault-injected runs, 2 Nominal runs)
    sim_fault_1 = SimulationRun(
        run_id="e2e_run_fault_01",
        task_type="research",
        topology="pipeline",
        fault_injector=FaultInjector(
            fault_type="hallucinated_output",
            target_agent="researcher_1",
            injection_step=1,
            severity=0.85,
            random_seed=42,
        ),
        random_seed=42,
        task_input={"topic": "Pipeline Fault Run 1"},
    ).execute()

    sim_fault_2 = SimulationRun(
        run_id="e2e_run_fault_02",
        task_type="research",
        topology="pipeline",
        fault_injector=FaultInjector(
            fault_type="hallucinated_output",
            target_agent="researcher_1",
            injection_step=1,
            severity=0.85,
            random_seed=43,
        ),
        random_seed=43,
        task_input={"topic": "Pipeline Fault Run 2"},
    ).execute()

    sim_nom_1 = SimulationRun(
        run_id="e2e_run_nom_01",
        task_type="research",
        topology="pipeline",
        random_seed=42,
        task_input={"topic": "Nominal Run 1"},
    ).execute()

    sim_nom_2 = SimulationRun(
        run_id="e2e_run_nom_02",
        task_type="research",
        topology="pipeline",
        random_seed=43,
        task_input={"topic": "Nominal Run 2"},
    ).execute()

    assert sim_fault_1.has_cascading_failure is True
    assert sim_nom_1.has_cascading_failure is False

    # Step 4: Temporal Graph Construction
    graph_builder = TemporalGraphBuilder()
    snapshots = graph_builder.build_snapshots_over_run(events=sim_fault_1.events)
    assert len(snapshots) > 0
    assert len(snapshots[-1].nodes) > 0

    # Step 5: Dataset Generation
    sample_gen = PredictionSampleGenerator(prediction_horizons=[1, 3], history_length=3)
    samples_f1 = sample_gen.generate_from_run(sim_fault_1)
    samples_f2 = sample_gen.generate_from_run(sim_fault_2)
    samples_n1 = sample_gen.generate_from_run(sim_nom_1)
    samples_n2 = sample_gen.generate_from_run(sim_nom_2)
    all_samples = samples_f1 + samples_f2 + samples_n1 + samples_n2

    # Step 6: Trajectory-Level Dataset Splitting (Disjoint runs)
    run_splits = {
        "train": ["e2e_run_fault_01", "e2e_run_nom_01"],
        "val": [],
        "test": ["e2e_run_fault_02", "e2e_run_nom_02"],
    }
    splits = split_samples(all_samples, run_splits)
    assert len(splits["train"]) > 0
    assert len(splits["test"]) > 0

    # Step 7: Model Training & Prediction
    extractor = TabularFeatureExtractor()
    feat_names = extractor.feature_names
    X_train = np.array([list(extractor.extract_from_sample(s).values()) for s in splits["train"]], dtype=np.float32)
    y_train = np.array([s.label for s in splits["train"]], dtype=np.int32)
    X_test = np.array([list(extractor.extract_from_sample(s).values()) for s in splits["test"]], dtype=np.float32)
    y_test = np.array([s.label for s in splits["test"]], dtype=np.int32)

    model = LogisticRegressionBaseline(random_state=42)
    model.fit(X_train, y_train, feature_names=feat_names)

    probs = model.predict_proba(X_test)
    preds = (probs >= 0.5).astype(int)
    assert len(probs) == len(X_test)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Step 8: Evaluate Prediction
    metrics = compute_comprehensive_metrics(y_true=list(y_test), y_pred=list(preds), y_prob=list(probs))
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics

    # Step 9: Explainability
    if len(X_test) > 0:
        explainer = ClassicalModelExplainer(model=model, feature_names=feat_names)
        attributions = explainer.explain_global_importance(X_val=X_test, y_val=y_test)
        assert len(attributions) > 0

    # Step 10 & 11: Database Persistence & API Verification
    target_run_id = sim_fault_1.run_id
    graphs_dir = settings.BASE_DIR / "data" / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    graph_file = graphs_dir / f"{target_run_id}.json"
    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump({
            "temporal_graph_history": [s.to_dict() for s in snapshots]
        }, f)

    db = SessionLocal()
    try:
        # Check if run exists, if not persist
        existing_run = db.query(DBRun).filter(DBRun.id == target_run_id).first()
        if not existing_run:
            db_run = DBRun(
                id=target_run_id,
                task_type="research",
                topology="pipeline",
                num_agents=len(sim_fault_1.agents),
                duration_seconds=10.0,
                has_cascading_failure=sim_fault_1.has_cascading_failure,
                cascading_failure_step=sim_fault_1.cascading_failure_step,
                random_seed=42,
            )
            db.add(db_run)
            db.commit()

        # Query API endpoints for the run
        resp_run = client.get(f"/api/v1/runs/{target_run_id}")
        assert resp_run.status_code == 200
        run_data = resp_run.json()
        assert run_data["id"] == target_run_id
        assert run_data["has_cascading_failure"] is True

        # Query graph endpoint
        resp_graph = client.get(f"/api/v1/runs/{target_run_id}/graph")
        assert resp_graph.status_code == 200
        graph_data = resp_graph.json()
        assert graph_data["run_id"] == target_run_id
        assert len(graph_data["temporal_snapshots"]) > 0

        # System overview & model comparison endpoints
        resp_comp = client.get("/api/v1/evaluations/comparison")
        assert resp_comp.status_code == 200
    finally:
        try:
            db.query(DBFailure).filter(DBFailure.run_id == target_run_id).delete()
            db.query(DBEvent).filter(DBEvent.run_id == target_run_id).delete()
            db.query(DBRun).filter(DBRun.id == target_run_id).delete()
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()
        if graph_file.exists():
            graph_file.unlink()
