# Table 2: Multi-Level Failure Hierarchy and Canonical Fault Taxonomy

| Hierarchy Level | Fault Identifier | Triggering Operational Mechanism | Observable Telemetry Signature | Propagation Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1**<br>*(Agent-Level)* | `hallucinated_output` | Agent emits fabricated facts or ungrounded entity citations. | Calibrated confidence drops; normal latency; no tool errors. | Local output corrupted; downstream verifier triggers check. |
| | `incorrect_information` | Agent generates mathematically or logically false intermediate values. | High self-reported confidence; normal latency; zero tool errors. | Subtly corrupts downstream task premises. |
| | `tool_failure` | External tool/API returns runtime exception or abnormal exit code. | `tool_error = True`; immediate retry attempt; quality drops. | Handled via retry; escalates if retries fail. |
| | `tool_timeout` | Tool blocks indefinitely exceeding agent deadline. | Latency spikes to max threshold; step blocked. | Execution schedule delay; potential starvation. |
| | `delayed_response` | Excessive inference latency during token generation. | Step latency increases $300\% - 1000\%$. | Causes coordination skew across synchronous channels. |
| | `malformed_output` | Output violates schema syntax (e.g. truncated JSON, invalid tags). | Parsing exception; retry trigger; quality score $= 0.0$. | Downstream agent rejects payload. |
| | `low_confidence_output` | Agent signals extreme epistemic uncertainty ($\le 0.20$). | Confidence drops sharply; hesitation cycles. | Triggers repeated critic review cycles. |
| **Level 2**<br>*(Interaction-Level)* | `contradictory_output` | Output directly contradicts assertions verified by upstream agents. | Semantic contradiction score spikes ($\ge 0.70$). | Generates deadlocks between Analyst and Verifier. |
| | `communication_loop` | Cyclic task ping-pong between agents without task progress. | Rapid surge in message frequency along cyclic pair. | Consumes token budget; blocks forward milestone. |
| | `incorrect_delegation` | Task delegated to an agent role lacking required tool access. | Target agent encounters failed tool attempts. | Task handoff rejected; workflow stalled. |
| | `stale_context` | Agent ignores recent updates and re-evaluates obsolete premises. | Semantic inconsistency with recent history. | Redundant work; uncoordinated revisions. |
| | `agent_dropout` | Agent crashes completely and ceases message responses. | Repeated unanswered events; `status = "failed"`. | Dependent agents time out awaiting reply. |
| **Level 3**<br>*(Systemic Cascade)* | **Cascading Breakdown** | Propagated multi-hop inconsistencies exhaust execution budget or cause deadlock. | Simultaneous contradiction surges, retry loops, and latency spikes. | **Terminal workflow collapse / task failure.** |
