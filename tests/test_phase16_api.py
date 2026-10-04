"""Comprehensive test suite for Phase 16 FastAPI backend and research APIs."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


# ---------------------------------------------------------
# 1. Health and System Endpoints
# ---------------------------------------------------------

def test_health_check():
    """Verify /health returns service status and metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "agentguard"
    assert data["project"] == "AgentGuard"
    assert "version" in data


def test_root_metadata():
    """Verify / returns project overview and documentation URLs."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "AgentGuard" in data["project"]
    assert data["docs_url"] == "/docs"
    assert data["api_v1_prefix"] == "/api/v1"


def test_openapi_documentation():
    """Verify OpenAPI schema is successfully generated and contains all research routes."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    paths = schema["paths"]
    assert "/health" in paths
    assert "/api/v1/agents" in paths
    assert "/api/v1/runs" in paths
    assert "/api/v1/evaluations/comparison" in paths
    assert "/api/v1/ablations" in paths
    assert "/api/v1/generalization" in paths
    assert "/api/v1/explanations" in paths


# ---------------------------------------------------------
# 2. Agent APIs
# ---------------------------------------------------------

def test_list_agents():
    """Verify GET /api/v1/agents returns paginated agents."""
    response = client.get("/api/v1/agents?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert len(data["items"]) <= 10
    assert data["total"] > 0
    first = data["items"][0]
    assert "agent_id" in first
    assert "name" in first
    assert "role" in first
    assert "creation_time" in first


def test_list_agents_filtering():
    """Verify filtering agents by role."""
    response = client.get("/api/v1/agents?role=planner&limit=5")
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["role"].lower() == "planner"


def test_get_agent_by_id():
    """Verify GET /api/v1/agents/{agent_id} returns single agent metadata."""
    list_res = client.get("/api/v1/agents?limit=1")
    agent_id = list_res.json()["items"][0]["agent_id"]

    response = client.get(f"/api/v1/agents/{agent_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == agent_id
    assert "role" in data
    assert "creation_time" in data


def test_get_agent_not_found():
    """Verify GET /api/v1/agents/{agent_id} returns 404 for unknown agent."""
    response = client.get("/api/v1/agents/non_existent_agent_99999")
    assert response.status_code == 404
    error = response.json().get("error", {})
    assert error.get("code") == "AGENT_NOT_FOUND"


# ---------------------------------------------------------
# 3. Simulation Run APIs
# ---------------------------------------------------------

def test_list_runs():
    """Verify GET /api/v1/runs returns simulation trajectories."""
    response = client.get("/api/v1/runs?limit=20&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert data["total"] >= 70
    first = data["items"][0]
    assert "id" in first
    assert "task" in first
    assert "topology" in first
    assert "status" in first
    assert "agent_count" in first
    assert "has_cascading_failure" in first


def test_list_runs_filtering():
    """Verify filtering runs by task, topology, and status."""
    response = client.get("/api/v1/runs?task=research&topology=pipeline&status=failed")
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["task"].lower() == "research"
        assert item["topology"].lower() == "pipeline"
        assert item["status"] == "failed"


def test_get_run_by_id():
    """Verify GET /api/v1/runs/{run_id} returns detailed run information."""
    list_res = client.get("/api/v1/runs?limit=1")
    run_id = list_res.json()["items"][0]["id"]

    response = client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == run_id
    assert "agents" in data
    assert "failures" in data


def test_get_run_not_found():
    """Verify GET /api/v1/runs/{run_id} returns 404 for unknown run."""
    response = client.get("/api/v1/runs/non_existent_run_99999")
    assert response.status_code == 404
    error = response.json().get("error", {})
    assert error.get("code") == "RUN_NOT_FOUND"


def test_get_run_events():
    """Verify GET /api/v1/runs/{run_id}/events returns event stream."""
    response = client.get("/api/v1/runs/run_telemetry_demo_001/events")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    first = data["items"][0]
    assert first["run_id"] == "run_telemetry_demo_001"
    assert "source_agent" in first
    assert "target_agent" in first


def test_get_run_failures():
    """Verify GET /api/v1/runs/{run_id}/failures returns failures."""
    list_res = client.get("/api/v1/runs?status=failed&limit=1")
    failed_runs = list_res.json()["items"]
    if failed_runs:
        run_id = failed_runs[0]["id"]
        response = client.get(f"/api/v1/runs/{run_id}/failures")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) > 0


# ---------------------------------------------------------
# 4. Temporal Graph APIs
# ---------------------------------------------------------

def test_get_run_graph():
    """Verify GET /api/v1/runs/{run_id}/graph returns nodes, edges, snapshots."""
    run_id = "run_0005_agentguard_generalization_v1"
    response = client.get(f"/api/v1/runs/{run_id}/graph")
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == run_id
    assert "nodes" in data
    assert "edges" in data
    assert "timestamps" in data
    assert "temporal_snapshots" in data
    assert len(data["temporal_snapshots"]) > 0


def test_get_run_graph_window_filter():
    """Verify temporal window filtering on graph endpoint."""
    run_id = "run_0005_agentguard_generalization_v1"
    response = client.get(f"/api/v1/runs/{run_id}/graph?snapshot_idx=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data["temporal_snapshots"]) <= 1


def test_get_graph_alias():
    """Verify GET /api/v1/graphs/{run_id} returns graph data."""
    run_id = "run_0005_agentguard_generalization_v1"
    response = client.get(f"/api/v1/graphs/{run_id}")
    assert response.status_code == 200
    assert response.json()["run_id"] == run_id


def test_get_graph_not_found():
    """Verify GET /api/v1/runs/{run_id}/graph returns 404 for unknown run."""
    response = client.get("/api/v1/runs/non_existent_graph_run_9999/graph")
    assert response.status_code == 404
    error = response.json().get("error", {})
    assert error.get("code") == "GRAPH_NOT_FOUND"


# ---------------------------------------------------------
# 5. Prediction APIs
# ---------------------------------------------------------

def test_list_predictions():
    """Verify GET /api/v1/predictions returns prediction records."""
    response = client.get("/api/v1/predictions?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    first = data["items"][0]
    assert "prediction_id" in first
    assert "run_id" in first
    assert "model" in first
    assert "horizon" in first
    assert "predicted_probability" in first
    assert "predicted_label" in first


def test_get_prediction_by_id():
    """Verify GET /api/v1/predictions/{prediction_id} returns prediction."""
    list_res = client.get("/api/v1/predictions?limit=1")
    pred_id = list_res.json()["items"][0]["prediction_id"]

    response = client.get(f"/api/v1/predictions/{pred_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_id"] == pred_id


def test_get_prediction_not_found():
    """Verify GET /api/v1/predictions/{prediction_id} returns 404 for unknown prediction."""
    response = client.get("/api/v1/predictions/invalid_pred_99999")
    assert response.status_code == 404
    error = response.json().get("error", {})
    assert error.get("code") == "PREDICTION_NOT_FOUND"


def test_get_run_predictions():
    """Verify GET /api/v1/runs/{run_id}/predictions returns predictions for run."""
    list_res = client.get("/api/v1/predictions?limit=1")
    run_id = list_res.json()["items"][0]["run_id"]

    response = client.get(f"/api/v1/runs/{run_id}/predictions")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0


# ---------------------------------------------------------
# 6. Experiment APIs
# ---------------------------------------------------------

def test_list_experiments():
    """Verify GET /api/v1/experiments returns benchmark experiments."""
    response = client.get("/api/v1/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    first = data["items"][0]
    assert "experiment_id" in first
    assert "name" in first
    assert "status" in first


def test_get_experiment_by_id():
    """Verify GET /api/v1/experiments/{experiment_id} returns experiment details."""
    list_res = client.get("/api/v1/experiments?limit=1")
    exp_id = list_res.json()["items"][0]["experiment_id"]

    response = client.get(f"/api/v1/experiments/{exp_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == exp_id


def test_get_experiment_not_found():
    """Verify GET /api/v1/experiments/{experiment_id} returns 404 for unknown experiment."""
    response = client.get("/api/v1/experiments/invalid_exp_99999")
    assert response.status_code == 404
    error = response.json().get("error", {})
    assert error.get("code") == "EXPERIMENT_NOT_FOUND"


# ---------------------------------------------------------
# 7. Evaluation & Model Comparison APIs
# ---------------------------------------------------------

def test_list_evaluations():
    """Verify GET /api/v1/evaluations returns evaluation metrics."""
    response = client.get("/api/v1/evaluations")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    first = data["items"][0]
    assert "model" in first
    assert "horizon" in first
    assert "precision" in first
    assert "recall" in first
    assert "f1" in first
    assert "auroc" in first
    assert "auprc" in first
    assert "mean_lead_time" in first


def test_get_evaluation_by_id():
    """Verify GET /api/v1/evaluations/{experiment_id} returns evaluation."""
    response = client.get("/api/v1/evaluations/Rule-Based_k1")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == "Rule-Based_k1"
    assert data["model"] == "Rule-Based"


def test_model_comparison_endpoint():
    """Verify GET /api/v1/evaluations/comparison compares all 9 canonical models."""
    response = client.get("/api/v1/evaluations/comparison?horizon=1")
    assert response.status_code == 200
    data = response.json()
    assert data["horizon"] == 1
    assert "models" in data
    assert "best_model_by_f1" in data
    assert "best_model_by_auroc" in data
    assert "best_model_by_lead_time" in data

    # Verify that the 9 target models are evaluated
    model_names = [m["model_name"] for m in data["models"]]
    expected_models = [
        "Rule-Based",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LSTM",
        "GRU",
        "GCN",
        "GAT",
        "Temporal GNN",
    ]
    for em in expected_models:
        assert em in model_names, f"Expected {em} in comparison list, found {model_names}"

    # Verify normalized metrics presence
    for m in data["models"]:
        assert 0.0 <= m["normalized_f1"] <= 1.0
        assert 0.0 <= m["normalized_auroc"] <= 1.0


def test_model_comparison_alias():
    """Verify GET /api/v1/models/comparison alias works identically."""
    response = client.get("/api/v1/models/comparison?horizon=1")
    assert response.status_code == 200
    assert "models" in response.json()


# ---------------------------------------------------------
# 8. Ablation APIs
# ---------------------------------------------------------

def test_list_ablations():
    """Verify GET /api/v1/ablations returns Phase 13 ablation results."""
    response = client.get("/api/v1/ablations?limit=20")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    first = data["items"][0]
    assert "ablation_name" in first
    assert "removed_component" in first
    assert "baseline_model" in first
    assert "metrics" in first


def test_get_ablation_by_id():
    """Verify GET /api/v1/ablations/{experiment_id} returns single ablation condition."""
    response = client.get("/api/v1/ablations/abl_full_temporal_gnn_k1_s42")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == "abl_full_temporal_gnn_k1_s42"
    assert data["ablation_name"] == "Full Temporal GNN"


def test_get_ablation_not_found():
    """Verify GET /api/v1/ablations/{experiment_id} returns 404 for unknown ablation."""
    response = client.get("/api/v1/ablations/invalid_ablation_id_9999")
    assert response.status_code == 404
    error = response.json().get("error", {})
    assert error.get("code") == "ABLATION_NOT_FOUND"


# ---------------------------------------------------------
# 9. Generalization APIs
# ---------------------------------------------------------

def test_list_generalization():
    """Verify GET /api/v1/generalization returns Phase 14 generalization evaluations."""
    response = client.get("/api/v1/generalization?limit=20")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    first = data["items"][0]
    assert "dimension" in first
    assert "split_type" in first
    assert "model" in first
    assert "training_configuration" in first
    assert "testing_configuration" in first
    assert "metrics" in first


def test_get_generalization_by_id():
    """Verify GET /api/v1/generalization/{experiment_id} returns experiment."""
    response = client.get("/api/v1/generalization/G1")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == "G1"
    assert "dimension" in data


def test_get_generalization_not_found():
    """Verify GET /api/v1/generalization/{experiment_id} returns 404 for unknown."""
    response = client.get("/api/v1/generalization/invalid_gen_id_9999")
    assert response.status_code == 404
    error = response.json().get("error", {})
    assert error.get("code") == "GENERALIZATION_NOT_FOUND"


# ---------------------------------------------------------
# 10. Explainability APIs
# ---------------------------------------------------------

def test_list_explanations():
    """Verify GET /api/v1/explanations returns Phase 15 explainability cases."""
    response = client.get("/api/v1/explanations")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    first = data["items"][0]
    assert "explanation_id" in first
    assert "predicted_probability" in first
    assert "prediction_horizon" in first
    assert "important_features" in first
    assert "important_agents" in first
    assert "important_edges" in first
    assert "explanation_method" in first


def test_get_explanation_by_id():
    """Verify GET /api/v1/explanations/{explanation_id} returns full report."""
    list_res = client.get("/api/v1/explanations?limit=1")
    exp_id = list_res.json()["items"][0]["explanation_id"]

    response = client.get(f"/api/v1/explanations/{exp_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["explanation_id"] == exp_id
    assert "causality_disclaimer" in data


def test_get_explanation_not_found():
    """Verify GET /api/v1/explanations/{explanation_id} returns 404 for unknown."""
    response = client.get("/api/v1/explanations/invalid_exp_id_9999")
    assert response.status_code == 404
    error = response.json().get("error", {})
    assert error.get("code") == "EXPLANATION_NOT_FOUND"


# ---------------------------------------------------------
# 11. Security, Validation & Error Envelope Tests
# ---------------------------------------------------------

def test_validation_error_envelope():
    """Verify invalid query parameter returns 422 with standardized error envelope."""
    response = client.get("/api/v1/runs?limit=999999")
    assert response.status_code == 422
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "INVALID_PARAMETER"
    assert "details" in payload["error"]


def test_pagination_bounds():
    """Verify pagination limit and offset function accurately."""
    r1 = client.get("/api/v1/runs?limit=5&offset=0")
    r2 = client.get("/api/v1/runs?limit=5&offset=5")

    d1 = r1.json()
    d2 = r2.json()

    assert len(d1["items"]) == 5
    assert len(d2["items"]) == 5
    # Ensure items are disjoint
    ids1 = {item["id"] for item in d1["items"]}
    ids2 = {item["id"] for item in d2["items"]}
    assert ids1.isdisjoint(ids2)
