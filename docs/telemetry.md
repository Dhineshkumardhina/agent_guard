# Structured Telemetry Documentation

## Overview

The AgentGuard Telemetry System (`ml/telemetry/`) provides structured, high-resolution instrumentation for multi-agent interactions. It captures fine-grained behavioral signals, communication dynamics, tool invocations, and failure annotations into typed schemas while enforcing strict credential sanitization at ingestion.

---

## 1. Core Event Schema: `AgentTelemetryEvent`

Every inter-agent communication, tool invocation, or error event is represented as a validated Pydantic model (`AgentTelemetryEvent` in `ml/telemetry/schemas.py`).

### Schema Field Definitions

| Field | Type | Description | Invariant / Constraints |
|---|---|---|---|
| `event_id` | `UUID` / `str` | Unique identifier for the event record. | Generated via `uuid4()`. |
| `run_id` | `str` | Trajectory execution identifier. | Links to parent simulation run. |
| `step_idx` | `int` | Monotonically increasing step counter within the run. | $\ge 0$. |
| `timestamp` | `float` | Simulated continuous timestamp (seconds from run start). | Monotonically non-decreasing within run. |
| `source_agent` | `str` | Agent ID or role initiating the interaction. | Non-empty string. |
| `target_agent` | `str` | Agent ID or role receiving the message/delegation. | Non-empty string. |
| `event_type` | `str` | Category of interaction event (see Event Types below). | Validated against supported set. |
| `message` | `str` | Sanitized communication payload text or summary. | PII/credentials scrubbed. |
| `message_length`| `int` | Character length of the sanitized message. | $\ge 0$. |
| `token_count` | `int` | Estimated or exact token consumption for the step. | $\ge 0$. |
| `latency` | `float` | Response execution time in simulated seconds. | $\ge 0.0$. |
| `confidence` | `float` | Agent self-reported or calibrated confidence score. | $\in [0.0, 1.0]$. |
| `output_quality`| `float` | Automated output quality rating score. | $\in [0.0, 1.0]$. |
| `contradiction_score` | `float` | Semantic contradiction metric against upstream context. | $\in [0.0, 1.0]$. |
| `tool_used` | `Optional[str]` | Identifier of tool invoked during step (if any). | Nullable. |
| `tool_success` | `Optional[bool]`| Success status of tool execution. | Nullable. |
| `tool_error` | `bool` | Flag indicating tool exception or abnormal exit code. | Default `False`. |
| `retry_count` | `int` | Number of consecutive retries executed for this step. | $\ge 0$. |
| `injected_fault`| `Optional[str]`| Identifier of synthetic fault injected at this step. | Nullable. |
| `error_type` | `Optional[str]`| Classification code of error (e.g., `TIMEOUT`, `PARSING`).| Nullable. |
| `failure_label`| `int` | Ground-truth failure hierarchy: 0=normal, 1=agent, 2=interaction, 3=cascade. | $\in \{0, 1, 2, 3\}$. |
| `downstream_failure` | `bool` | Whether this event eventually causes a Level 3 cascade. | Ground-truth causal indicator. |
| `metadata_json`| `Dict[str, Any]`| Supplemental execution attributes and diagnostics. | JSON-serializable dictionary. |

---

## 2. Telemetry Event Categories

Defined in `TelemetryEventType`:

1. `AGENT_START`: Initialization of an agent for a subtask.
2. `AGENT_END`: Completion or deactivation of an agent.
3. `MESSAGE`: Standard message exchange or task delegation between two agents.
4. `TOOL_CALL`: Invocation of an external tool or computational routine.
5. `TOOL_RESULT`: Return value or output payload from a tool.
6. `RETRY`: Re-execution of an agent operation following a failure.
7. `ERROR`: Local error condition or exception caught.
8. `TIMEOUT`: Execution deadline exceeded.
9. `VALIDATION`: Verifier or Critic schema/factuality validation result.
10. `DELEGATION`: Explicit transfer of task ownership from agent $u$ to agent $v$.
11. `FAILURE`: Terminal or cascading failure event declaration.

---

## 3. Telemetry Collector Engine (`ml/telemetry/collector.py`)

The `TelemetryCollector` singleton coordinates in-memory buffering, validation, and storage:
- **Stream Ingestion:** Validates incoming dictionaries or event objects against Pydantic models.
- **Sanitization Pipeline:** Applies regex masks to redact simulated API keys, bearer tokens, passwords, and private identifiers prior to persistence.
- **In-Memory Buffering:** Maintains bounded circular buffers per active trajectory for low-latency streaming to the frontend WebSocket/REST endpoints.
- **Batch Persistence:** Flushes structured events into SQLite or PostgreSQL via SQLAlchemy `bulk_insert_mappings` to maintain sub-millisecond per-event logging latency.

---

## 4. Derived Telemetry Features (`ml/telemetry/features.py`)

From raw event sequences, the telemetry feature extractor generates rolling statistical summaries:
- `rolling_latency_mean` / `rolling_latency_std`: Temporal variance in agent execution latency.
- `retry_velocity`: First derivative of retry counts over a sliding 5-event window.
- `contradiction_acceleration`: Rate of increase in contradiction scores between interacting agent pairs.
- `token_intensity`: Token consumption per simulated unit time.
- `reciprocity_ratio`: Proportion of bidirectional message exchanges relative to unidirectional broadcasts.
