# AgentGuard Research REST API Specification

## 1. Overview & Architecture

AgentGuard provides a production-grade FastAPI research service exposing simulation trajectories, telemetry events, temporal graph snapshots, baseline and temporal GNN models, cross-model benchmark evaluations, ablation study matrices, out-of-distribution generalization experiments, and explainability attributions.

### Backend Structure

```
backend/
└── app/
    ├── main.py                     # FastAPI application setup, lifespan, CORS, and global error handlers
    ├── api/
    │   ├── router.py               # Central versioned router (/api/v1)
    │   ├── dependencies.py         # DB session & pagination dependencies
    │   └── routes/
    │       ├── health.py           # Service status (/health)
    │       ├── agents.py           # Multi-agent participant directory (/api/v1/agents)
    │       ├── runs.py             # Trajectories, events, failures, graphs, predictions (/api/v1/runs)
    │       ├── events.py           # Telemetry communication streams (/api/v1/events)
    │       ├── graphs.py           # Temporal graph slices and snapshots (/api/v1/graphs)
    │       ├── predictions.py      # Early warning failure predictions (/api/v1/predictions)
    │       ├── experiments.py      # Benchmark suites and hyperparameters (/api/v1/experiments)
    │       ├── evaluations.py      # Empirical evaluations & 9-model comparison (/api/v1/evaluations)
    │       ├── ablations.py        # Phase 13 component attribution (/api/v1/ablations)
    │       ├── generalization.py   # Phase 14 shift dimensions & transfer gaps (/api/v1/generalization)
    │       └── explainability.py   # Phase 15 feature/agent/edge attributions (/api/v1/explanations)
    ├── schemas/                    # Strictly typed Pydantic models for requests and responses
    ├── services/                   # Business logic, streaming JSONL graph slice reader, DB access
    ├── models/                     # Re-exporting SQLAlchemy domain entities
    ├── database/                   # Relational models, session management, and tables
    ├── core/                       # Settings, logging, and error handling
    └── utils/                      # Pagination and streaming helpers
```

---

## 2. Configuration & Security

The service is configured using strongly-typed Pydantic settings (`Settings` in `backend/app/core/config.py`), reading from `.env` or system environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENV` | `development` | Environment mode (`development`, `testing`, `production`) |
| `DEBUG` | `True` | Debug flag |
| `LOG_LEVEL` | `INFO` | Standard Python logging level |
| `PORT` | `8000` | Port for uvicorn server |
| `HOST` | `127.0.0.1` | Network interface binding |
| `ALLOWED_ORIGINS` | `["http://localhost:5173", ...]` | CORS allowed origins (frontend dashboard) |
| `DATABASE_URL` | `sqlite:///./agentguard.db` | SQLAlchemy connection string |

### Production Security Practices
1. **No Sensitive Leaks**: Model outputs, internal stack traces, and database connection strings are never exposed in error responses.
2. **CORS Headers**: Configured strictly via `CORSMiddleware` based on environment variables.
3. **Input Validation**: All query parameters and route parameters are validated against strict Pydantic schemas. Query pagination bounds enforce `1 <= limit <= 200`.
4. **Clean Error Envelope**: All API exceptions adhere to a standardized REST error envelope.

---

## 3. Error Handling Specification

All error responses return a standardized JSON structure with machine-readable uppercase error codes:

```json
{
  "error": {
    "code": "RUN_NOT_FOUND",
    "message": "The requested simulation run 'run_invalid_001' was not found.",
    "details": null
  }
}
```

### Standard Error Codes

| HTTP Status | Error Code | Description |
| :--- | :--- | :--- |
| `400` | `INVALID_PARAMETER` | Malformed parameter or unsupported query value |
| `404` | `AGENT_NOT_FOUND` | Agent ID was not found |
| `404` | `RUN_NOT_FOUND` | Simulation run ID was not found |
| `404` | `GRAPH_NOT_FOUND` | Temporal graph data was not found for run |
| `404` | `PREDICTION_NOT_FOUND` | Prediction record ID was not found |
| `404` | `EXPERIMENT_NOT_FOUND` | Benchmark experiment was not found |
| `404` | `EVALUATION_NOT_FOUND` | Evaluation metric record was not found |
| `404` | `ABLATION_NOT_FOUND` | Ablation condition was not found |
| `404` | `GENERALIZATION_NOT_FOUND` | Generalization experiment was not found |
| `404` | `EXPLANATION_NOT_FOUND` | Explanation report was not found |
| `422` | `INVALID_PARAMETER` | Request schema validation failure |
| `500` | `INTERNAL_SERVER_ERROR` | Unexpected server fault (stack trace logged safely) |

---

## 4. API Endpoints Reference

### 4.1 System & Health

#### `GET /health`
Returns service readiness and project metadata.
- **Response**:
```json
{
  "status": "ok",
  "service": "agentguard",
  "project": "AgentGuard",
  "version": "0.1.0",
  "environment": "development"
}
```

#### `GET /`
Returns root documentation URLs and service prefix.

---

### 4.2 Agents

#### `GET /api/v1/agents`
- **Query Parameters**:
  - `run_id` (optional, string): Filter by simulation run.
  - `role` (optional, string): Filter by role (`planner`, `researcher`, `analyst`, `verifier`, `decision`, etc.).
  - `limit` (int, default=50, max=200): Pagination limit.
  - `offset` (int, default=0): Pagination offset.
- **Response**: `PaginatedResponse[AgentResponse]`

#### `GET /api/v1/agents/{agent_id}`
- **Path Parameter**: `agent_id` (string).
- **Response**: `AgentResponse` containing `agent_id`, `name`, `role`, `description`, `creation_time`, `run_id`, `status`.

---

### 4.3 Simulation Runs

#### `GET /api/v1/runs`
- **Query Parameters**:
  - `task` (string): Filter by task domain (`research`, `coding`, `analysis`, `planning`).
  - `topology` (string): Filter by topology (`pipeline`, `star`, `mesh`, `custom`).
  - `status` (string): Filter by outcome (`completed`, `failed`).
  - `agent_count` (int): Number of participating agents (`3`, `5`, `8`, `12`).
  - `start_date` / `end_date` (ISO datetime): Date range filtering.
  - `limit` / `offset`: Pagination controls.
- **Response**: `RunListResponse`

#### `GET /api/v1/runs/{run_id}`
- **Path Parameter**: `run_id` (string).
- **Response**: `RunDetailResponse` including participating agents and failure occurrences.

#### `GET /api/v1/runs/{run_id}/events`
- **Path Parameter**: `run_id` (string).
- **Query Parameters**: `limit`, `offset`.
- **Response**: `EventListResponse` with chronological communication telemetry.

#### `GET /api/v1/runs/{run_id}/failures`
- **Path Parameter**: `run_id` (string).
- **Query Parameters**: `limit`, `offset`.
- **Response**: `RunFailureListResponse` with failure level, originating agent, affected agents, and taxonomy.

---

### 4.4 Temporal Graph

#### `GET /api/v1/runs/{run_id}/graph`
Streams and slices temporal graphs without loading full multi-megabyte datasets into memory.
- **Path Parameter**: `run_id` (string).
- **Query Parameters**:
  - `start_time` (float): Minimum timestamp window bound.
  - `end_time` (float): Maximum timestamp window bound.
  - `step_idx` (int): Filter by discrete simulation step.
  - `snapshot_idx` (int): Filter by exact snapshot index in sequence.
- **Response**: `RunGraphResponse` containing:
  - `run_id`
  - `nodes`: List of unique active agent nodes with behavioral feature vectors.
  - `edges`: List of directed communication edges with interaction frequencies and contradictions.
  - `timestamps`: List of snapshot timestamps.
  - `node_features`: Dictionary mapping `agent_id` to feature attributes.
  - `edge_features`: Dictionary mapping `src->tgt` to interaction attributes.
  - `temporal_snapshots`: Chronological snapshot sequence.
  - `window`: Applied filter bounds.

#### `GET /api/v1/graphs/{run_id}`
Direct alias route for temporal graph retrieval.

---

### 4.5 Predictions

#### `GET /api/v1/predictions`
- **Query Parameters**:
  - `run_id` (string): Filter by run ID.
  - `model` (string): Filter by model architecture (`temporal_gnn`, `gat`, `gcn`, etc.).
  - `horizon` (int): Filter by prediction horizon $k \in \{1, 3, 5, 10, 20\}$.
  - `limit` / `offset`: Pagination controls.
- **Response**: `PredictionListResponse` containing `prediction_id`, `run_id`, `model`, `timestamp`, `horizon`, `predicted_probability`, `predicted_label`, `threshold`, `actual_outcome`, `risk_level`, `lead_time`.

#### `GET /api/v1/predictions/{prediction_id}`
- **Path Parameter**: `prediction_id` (string).
- **Response**: `PredictionResponse`.

#### `GET /api/v1/runs/{run_id}/predictions`
- **Path Parameter**: `run_id` (string).
- **Query Parameters**: `model`, `horizon`, `limit`, `offset`.
- **Response**: `PredictionListResponse`.

---

### 4.6 Experiments

#### `GET /api/v1/experiments`
- **Query Parameters**: `status`, `model`, `limit`, `offset`.
- **Response**: `ExperimentListResponse`.

#### `GET /api/v1/experiments/{experiment_id}`
- **Path Parameter**: `experiment_id` (string).
- **Response**: `ExperimentResponse` containing hyperparameters, seed, dataset version, and lifecycle status.

---

### 4.7 Evaluations & Model Comparison

#### `GET /api/v1/evaluations`
- **Query Parameters**:
  - `model` (string): Filter by evaluated model.
  - `horizon` (int): Filter by prediction horizon.
  - `experiment` (string): Filter by experiment identifier.
  - `dataset_version` (string): Filter by dataset version.
  - `limit` / `offset`: Pagination controls.
- **Response**: `EvaluationListResponse` returning Precision, Recall, F1, AUROC, AUPRC, FPR, False Alarm Rate, Mean Lead Time, Median Lead Time, Successful Warnings, and Warnings Per Trajectory.

#### `GET /api/v1/evaluations/comparison` (and `GET /api/v1/models/comparison`)
Returns normalized comparative benchmark metrics across the 9 canonical architectures:
1. **Rule-Based**
2. **Logistic Regression**
3. **Random Forest**
4. **XGBoost**
5. **LSTM**
6. **GRU**
7. **GCN**
8. **GAT**
9. **Temporal GNN**

- **Query Parameters**:
  - `horizon` (int, default=1): Prediction horizon $k$.
- **Response**: `ModelComparisonResponse` returning:
  - `horizon`
  - `total_models` (9 canonical models)
  - `models`: List of models with raw metrics and relative normalized metrics (`normalized_f1`, `normalized_auroc`, `normalized_lead_time`).
  - `best_model_by_f1`
  - `best_model_by_auroc`
  - `best_model_by_lead_time`

#### `GET /api/v1/evaluations/{experiment_id}`
- **Path Parameter**: `experiment_id` (string).
- **Response**: `EvaluationResponse`.

---

### 4.8 Ablations

#### `GET /api/v1/ablations`
Exposes the 120 Phase 13 component attribution and architectural ablation conditions.
- **Query Parameters**:
  - `model` (string): Base model (`temporal_gnn`, etc.).
  - `horizon` (int): Prediction horizon $k$.
  - `seed` (int): Random seed (`42`, `123`, `456`).
  - `limit` / `offset`: Pagination controls.
- **Response**: `AblationListResponse` returning `ablation_name`, `removed_component`, `baseline_model`, `metrics`, `horizon`, `seed`, `dataset_version`.

#### `GET /api/v1/ablations/{experiment_id}`
- **Path Parameter**: `experiment_id` (string, e.g. `abl_full_temporal_gnn_k1_s42`).
- **Response**: `AblationResponse`.

---

### 4.9 Generalization

#### `GET /api/v1/generalization`
Exposes the 210 Phase 14 out-of-distribution evaluation results across agent count, topology, task type, and compound shifts.
- **Query Parameters**:
  - `dimension` (string): Shift dimension (`agent_count`, `topology`, `task_type`, `compound`).
  - `model` (string): Evaluated model.
  - `horizon` (int): Horizon $k$.
  - `split_type` (string): `in_distribution` or `out_of_distribution`.
  - `limit` / `offset`: Pagination controls.
- **Response**: `GeneralizationListResponse` returning `dimension`, `training_configuration`, `testing_configuration`, `model`, `metrics`, `generalization_gap`, `gaps`.

#### `GET /api/v1/generalization/{experiment_id}`
- **Path Parameter**: `experiment_id` (string, e.g. `G1`, `G2`, `G3`, `G4`).
- **Response**: `GeneralizationMetricResponse`.

---

### 4.10 Explainability

#### `GET /api/v1/explanations`
Exposes Phase 15 explainability reports and case studies.
- **Query Parameters**:
  - `run_id` (string): Filter by simulation run.
  - `model` (string): Filter by model.
  - `limit` / `offset`: Pagination controls.
- **Response**: `ExplanationListResponse` returning:
  - `explanation_id`
  - `run_id`
  - `predicted_probability`
  - `prediction_horizon`
  - `important_features` (ranked feature saliency)
  - `important_agents` (ranked node attribution)
  - `important_edges` (ranked interaction attribution)
  - `important_events` (salient chronological events)
  - `explanation_method`
  - `model_version`
  - `dataset_version`
  - `high_level_summary`
  - `causality_disclaimer`

#### `GET /api/v1/explanations/{explanation_id}`
- **Path Parameter**: `explanation_id` (string).
- **Response**: `ExplanationResponse`.

#### `GET /api/v1/runs/{run_id}/explanations`
- **Path Parameter**: `run_id` (string).
- **Response**: `ExplanationListResponse`.

---

## 5. Local Startup & Execution

From the project root directory:

```bash
# Start backend server with auto-reload
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive documentation is available at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc UI**: `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema**: `http://127.0.0.1:8000/openapi.json`

---

## 6. Verification & Test Suite

The test suite runs with `pytest`:

```bash
# Run all backend and research API tests
pytest tests/test_phase16_api.py -v

# Run full project regression suite
pytest tests/ -v
```

### Performance Observations (Smoke Test)
Measured on local development workstation:

| Endpoint | Average Latency | Status |
| :--- | :--- | :--- |
| `GET /health` | ~1.5 ms | 200 OK |
| `GET /api/v1/agents?limit=50` | ~35 ms | 200 OK |
| `GET /api/v1/runs?limit=50` | ~38 ms | 200 OK |
| `GET /api/v1/runs/{run_id}` | ~18 ms | 200 OK |
| `GET /api/v1/runs/{run_id}/graph` | ~45 ms | 200 OK |
| `GET /api/v1/runs/{run_id}/events` | ~54 ms | 200 OK |
| `GET /api/v1/evaluations/comparison?horizon=1` | ~20 ms | 200 OK |
| `GET /api/v1/ablations?limit=50` | ~33 ms | 200 OK |
| `GET /api/v1/generalization?limit=50` | ~59 ms | 200 OK |
| `GET /api/v1/explanations` | ~26 ms | 200 OK |
