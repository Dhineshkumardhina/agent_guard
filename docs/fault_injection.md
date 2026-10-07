# Fault Injection and Failure Taxonomy

## Overview

AgentGuard implements a controlled fault injection engine (`ml/simulation/fault_injection/`) designed to perturb multi-agent workflows reproducibly. The framework defines a three-level failure hierarchy and 12 canonical fault modes grounded in observed multi-agent failure patterns.

---

## 1. Three-Level Failure Hierarchy

AgentGuard formalizes multi-agent failures into three distinct severity levels:

```
[Level 0: Nominal Execution]
       │
       ▼ (Local perturbation introduced)
[Level 1: Agent-Level Failure]
       │
       ▼ (Inter-agent communication corrupted)
[Level 2: Interaction-Level Failure]
       │
       ▼ (Multi-hop error propagation across dependencies)
[Level 3: Cascading System-Level Failure]
```

### Level 0: Nominal Execution
* **Definition:** All agents operate within expected behavioral parameters.
* **Manifestation:** Tasks complete successfully without unhandled exceptions, contradiction spikes, or excessive retries.

### Level 1: Agent-Level Failure
* **Definition:** An isolated anomaly occurring within the boundary of a single agent.
* **Manifestation:** A tool invocation throws an unhandled exception, output JSON fails schema validation, or the agent reports an ungrounded hallucination.
* **Containment:** If downstream agents discard or correct the invalid output, the failure remains contained at Level 1 and does not bring down the entire system.

### Level 2: Interaction-Level Failure
* **Definition:** A coordination or communication breakdown between two or more communicating agents.
* **Manifestation:** Pairwise contradiction (e.g., Verifier rejects Analyst's premise without explanation), infinite ping-pong delegation between two agents, or transmission of unparsable payloads across channels.
* **Containment:** The failure affects pairwise message exchange but has not yet caused total workflow collapse or system abandonment.

### Level 3: Cascading / System-Level Failure
* **Definition:** An unrecoverable, multi-hop failure that propagates through the multi-agent dependency graph, resulting in terminal task failure.
* **Manifestation:** Execution budget exhausted, deadlock across mutual dependencies, infinite communication loop involving multiple agents, or terminal output that fundamentally violates task constraints.
* **Target Objective:** **AgentGuard's primary prediction objective is early warning of impending Level 3 cascading failures.**

---

## 2. Implemented Fault Modes (The 12 Canonical Faults)

The `FaultInjector` class (`ml/simulation/fault_injection/injector.py`) implements exactly 12 reproducible fault modes:

### 1. `hallucinated_output`
* **Mechanism:** The target agent injects fabricated facts, imaginary citations, or non-existent entity references into its output message.
* **Telemetry Signature:** Moderate drop in calibrated confidence, normal latency, normal message length. Downstream verification steps encounter contradiction spikes.

### 2. `incorrect_information`
* **Mechanism:** The target agent outputs mathematically or logically erroneous intermediate values while maintaining valid formatting.
* **Telemetry Signature:** High self-reported confidence (overconfidence failure), normal execution latency, zero local tool errors.

### 3. `tool_failure`
* **Mechanism:** An external tool or API called by the agent raises an unhandled runtime error or non-zero exit code.
* **Telemetry Signature:** `tool_error = True`, immediate retry attempt, output quality degradation, local retry velocity spike.

### 4. `tool_timeout`
* **Mechanism:** A tool invocation blocks indefinitely until the agent's timeout threshold is exceeded.
* **Telemetry Signature:** Latency spike to maximal threshold ($t \ge t_{\text{timeout}}$), `tool_error = True`, step execution delay.

### 5. `delayed_response`
* **Mechanism:** Agent reasoning or token generation experiences severe latency degradation without throwing an explicit error.
* **Telemetry Signature:** Response latency increases by $300\% - 1000\%$, disrupting synchronous delegation schedules.

### 6. `malformed_output`
* **Mechanism:** Agent generates unparsable syntax (e.g., truncated JSON, unescaped quotes, broken Markdown fence tags).
* **Telemetry Signature:** Downstream parsing exception, immediate retry trigger, low output quality score.

### 7. `low_confidence_output`
* **Mechanism:** Agent detects internal ambiguity or epistemic uncertainty and emits an extremely low confidence score ($\le 0.20$).
* **Telemetry Signature:** Self-reported `confidence` drops sharply, triggering hesitation and repeated verifier cycles.

### 8. `contradictory_output`
* **Mechanism:** Agent generates claims that directly contradict verified assertions established by upstream agents in the same session.
* **Telemetry Signature:** Immediate spike in `contradiction_score` ($\ge 0.70$) on the directed edge $(u \to v)$.

### 9. `communication_loop`
* **Mechanism:** Two or more agents enter an unproductive delegation cycle, repeatedly passing the same task back and forth without progressing the task state.
* **Telemetry Signature:** Rapid surge in message frequency along cyclic edge pairs ($(u \to v)$ and $(v \to u)$), zero output delta, accumulating token expenditure.

### 10. `incorrect_delegation`
* **Mechanism:** The planner or coordinating agent assigns a subtask to an inappropriate role (e.g., delegating Python implementation to the literature Researcher).
* **Telemetry Signature:** Target agent encounters multiple failed tool attempts, high retry count, followed by task handoff rejection.

### 11. `stale_context`
* **Mechanism:** Agent discards recent conversational updates and operates on obsolete initial premises.
* **Telemetry Signature:** Semantic inconsistency with recent history, moderate contradiction score, repeated redundant tool executions.

### 12. `agent_dropout`
* **Mechanism:** Target agent crashes completely and becomes unreachable, failing to respond to incoming messages.
* **Telemetry Signature:** Repeated unanswered message events, target agent marked `status = "failed"`, upstream callers encountering timeouts.

---

## 3. Propagation Engine (`ml/simulation/fault_injection/propagation.py`)

When a fault is injected into agent $u$ at step $k$, the `PropagationEngine` determines whether and how the error spreads:
- **Resilience Probability:** Based on receiving agent role capabilities (e.g., Verifier agents have higher probability of catching and halting malformed outputs).
- **Cascading Transition:** If receiving agent $v$ accepts corrupted input, $v$'s state transitions to degraded, propagating Level 2 inconsistencies downstream.
- **Terminal Cascade Trigger:** When corrupted messages cross $\ge 2$ intermediate hops and exceed retry limits, the session transitions to Level 3 (Cascading Failure), marking the terminal failure step.
