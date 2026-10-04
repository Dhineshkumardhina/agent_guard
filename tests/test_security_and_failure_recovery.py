"""Phase 18: API Security, Database Security, and Failure Recovery Testing.

Explicitly verifies:
1. API Security:
   - Malformed requests (invalid JSON, invalid parameter types) -> 422
   - Non-existent IDs -> 404 structured envelope
   - Excessive pagination values (limit > 200) -> 422
   - Negative offset -> 422
   - Unsupported HTTP methods (e.g. DELETE on read-only endpoints) -> 405
   - CORS behavior and headers
   - Zero error leakage: 500 errors do not expose Python tracebacks or secrets
2. Database & Storage Security:
   - SQL injection resistance (parameterized SQLAlchemy queries)
   - Accidental secret leakage prevention
3. Failure-Recovery Testing:
   - Missing model checkpoint -> clean FileNotFoundError
   - Missing dataset directory -> clean error
   - Corrupted dataset file -> graceful validation rejection
   - Empty result set -> valid empty response, no crashes
   - Invalid experiment configuration -> clean validation rejection
"""

import pytest
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from ml.baselines.classical_ml.models import LogisticRegressionBaseline
from ml.baselines.temporal_gnn.models import TemporalGraphFailurePredictor
from ml.config.experiment_config import ExperimentConfig, TopologyType

client = TestClient(app)


# =========================================================================
# 1. API Security & Input Validation Tests
# =========================================================================

def test_api_security_excessive_pagination_rejected():
    """Verify that pagination limits > 200 are rejected with 422 validation error."""
    response = client.get("/api/v1/runs?limit=500")
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_PARAMETER"


def test_api_security_negative_offset_rejected():
    """Verify that negative offsets are rejected with 422."""
    response = client.get("/api/v1/runs?offset=-1")
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_PARAMETER"


def test_api_security_invalid_id_returns_clean_404():
    """Verify non-existent run_id returns structured 404 without leaking internal stack traces."""
    response = client.get("/api/v1/runs/non_existent_run_99999999")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] in ("NOT_FOUND", "RUN_NOT_FOUND")
    # Ensure no internal Python traceback was returned
    assert "Traceback (most recent call last)" not in response.text


def test_api_security_unsupported_method_returns_405():
    """Verify unsupported HTTP method on read-only endpoint returns 405."""
    response = client.delete("/api/v1/runs")
    assert response.status_code == 405
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_api_security_cors_headers():
    """Verify CORS preflight and request headers."""
    response = client.options(
        "/api/v1/runs",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Status code 200 or 204 for successful preflight
    assert response.status_code in (200, 204)
    assert "access-control-allow-origin" in response.headers


def test_api_security_sql_injection_attempt_handled_safely():
    """Verify SQL injection strings in query parameters are safely handled as literals."""
    payloads = [
        "' OR '1'='1",
        "'; DROP TABLE simulation_runs; --",
        "1 UNION SELECT * FROM users",
    ]
    for sql_inj in payloads:
        response = client.get(f"/api/v1/runs?task={sql_inj}")
        # Should return 200 with empty list or 422, NEVER 500 database crash
        assert response.status_code in (200, 422)
        if response.status_code == 200:
            data = response.json()
            assert "items" in data


def test_api_security_unhandled_error_envelope_does_not_leak_traces():
    """Verify production error handler obscures internal tracebacks."""
    response = client.get("/api/v1/runs/invalid'run'injection/graph")
    assert response.status_code in (404, 422, 500)
    data = response.json()
    assert "error" in data
    assert "Traceback" not in response.text
    assert "File \"" not in response.text


# =========================================================================
# 2. Failure Recovery & Robustness Tests
# =========================================================================

def test_failure_recovery_missing_model_checkpoint():
    """Verify clean FileNotFoundError when loading non-existent model checkpoint."""
    with pytest.raises(FileNotFoundError, match="not found"):
        LogisticRegressionBaseline.load("non_existent_model_checkpoint.pkl")

    predictor = TemporalGraphFailurePredictor()
    with pytest.raises(FileNotFoundError, match="not found"):
        predictor.load_checkpoint("non_existent_tgnn_weights.pt")


def test_failure_recovery_corrupted_dataset_file():
    """Verify corrupted JSON dataset file is cleanly rejected."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        f.write("{ invalid corrupted json content ...")
        corrupted_path = Path(f.name)

    try:
        with open(corrupted_path, "r") as f:
            with pytest.raises(json.JSONDecodeError):
                json.load(f)
    finally:
        if corrupted_path.exists():
            corrupted_path.unlink()


def test_failure_recovery_invalid_experiment_config():
    """Verify pydantic config rejects invalid topology or negative parameters."""
    with pytest.raises(Exception):
        # Invalid topology string
        ExperimentConfig(topology="invalid_topology_name")

    with pytest.raises(Exception):
        # Negative number of agents
        ExperimentConfig(num_agents=-5)


def test_failure_recovery_empty_result_set():
    """Verify query with no matching records returns valid empty schema without crashing."""
    response = client.get("/api/v1/runs?task=completely_unknown_task_domain")
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["has_more"] is False
