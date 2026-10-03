"""Demo Script: Telemetry Collection & Export (Phase 3).

Executes a multi-agent simulation, collects telemetry events via TelemetryCollector,
and exports the raw event stream to data/raw/<run_id>.jsonl.
Computes and verifies rolling features with temporal leakage protection.
"""

import sys
import json
import logging
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Silence SQL engine logs for clean output
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

from ml.simulation.run import SimulationRun
from ml.telemetry.collector import TelemetryCollector
from ml.telemetry.features import compute_rolling_features


def main():
    print("=" * 70)
    print(" AgentGuard -- Telemetry Collection & Export (Phase 3)")
    print("=" * 70)

    # 1. Execute a controlled simulation run
    run_id = "run_telemetry_demo_001"
    print(f"\n[1] Executing simulation trajectory: {run_id}...")
    sim = SimulationRun(
        run_id=run_id,
        task_type="research",
        topology="pipeline",
        random_seed=42,
        task_input={"topic": "Graph-based Cascading Failure Early Warning"},
    ).execute()

    print(f"    Completed in {sim.duration_seconds:.4f}s with status: {sim.final_status}")
    print(f"    Generated {len(sim.events)} execution events.")

    # 2. Ingest into TelemetryCollector
    print("\n[2] Ingesting events into TelemetryCollector...")
    collector = TelemetryCollector(active_run_id=run_id)
    ingested_events = collector.receive_batch(sim.events, run_id=run_id)
    print(f"    Validated and buffered {len(ingested_events)} events.")

    # 3. Export to data/raw/<run_id>.jsonl
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = raw_dir / f"{run_id}.jsonl"

    print(f"\n[3] Exporting raw event stream to: {jsonl_path.relative_to(_PROJECT_ROOT)}...")
    collector.export(jsonl_path, format="jsonl", run_id=run_id)

    # Also export CSV for format validation
    csv_path = raw_dir / f"{run_id}.csv"
    collector.export(csv_path, format="csv", run_id=run_id)
    print(f"    Also exported CSV format to: {csv_path.relative_to(_PROJECT_ROOT)}")

    # 4. Verify exported JSONL content integrity
    print("\n[4] Verifying exported JSONL integrity...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    print(f"    File exists: {jsonl_path.exists()} (size: {jsonl_path.stat().st_size} bytes)")
    print(f"    Total lines written: {len(lines)}")
    assert len(lines) == len(sim.events), "Mismatch between simulation events and exported lines!"

    for i, line in enumerate(lines):
        print(f"    Line {i}: step={line['step_idx']} | t={line['timestamp']}s | "
              f"{line['source_agent']} -> {line['target_agent']} [{line['event_type']}]")

    # 5. Compute Rolling Features with Data Leakage Protection
    print("\n[5] Computing causal rolling features (window_size=3)...")
    feature_records = compute_rolling_features(ingested_events, window_size=3)
    for rec in feature_records:
        print(f"    Step {rec['step_idx']} (t={rec['timestamp']:.3f}s): "
              f"err_rate={rec['rolling_error_rate']:.2f}, "
              f"latency={rec['rolling_latency']:.3f}s, "
              f"conf={rec['rolling_confidence']:.3f}, "
              f"msg_freq={rec['message_frequency']:.2f}/s, "
              f"contradiction={rec['contradiction_rate']:.3f}")

    print("\n" + "=" * 70)
    print(" Telemetry Export and Verification Complete: SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()
