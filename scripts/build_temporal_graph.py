"""Demo Script: Temporal Graph Snapshot Construction & Verification (Phase 5).

Generates a multi-agent simulation trajectory, converts chronological events
into temporal interaction graph snapshots G(t) = (V(t), E(t), X(t)),
serializes the snapshots to disk, and verifies exact round-trip reconstruction.
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Silence SQL engine logs for clean terminal output
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

from ml.simulation.run import SimulationRun
from ml.simulation.fault_injection.injector import FaultInjector
from ml.graph.graph_builder import TemporalGraphBuilder
from ml.graph.serializers import save_snapshot, load_snapshot
from ml.graph.visualization import visualize_snapshot


def main():
    print("=" * 75)
    print(" AgentGuard -- Temporal Graph Builder & Snapshot Verification (Phase 5)")
    print("=" * 75)

    # 1. Generate a controlled simulation run with fault injection to study failure graphs
    run_id = "run_temporal_graph_demo_01"
    print(f"\n[1] Generating simulation run with deterministic seed (run_id: {run_id})...")
    injector = FaultInjector(
        fault_type="tool_failure",
        target_agent="researcher_1",
        injection_step=1,
        probability=1.0,
        severity=0.8,
        random_seed=42,
    )
    sim = SimulationRun(
        run_id=run_id,
        task_type="research",
        topology="pipeline",
        random_seed=42,
        fault_injector=injector,
        task_input={"topic": "Graph Neural Network Cascading Failure Dynamics"},
    ).execute()

    print(f"    Execution completed: status={sim.final_status}, duration={sim.duration_seconds:.4f}s")
    print(f"    Run ID:               {sim.run_id}")
    print(f"    Number of Agents:     {len(sim.agents)}")
    print(f"    Number of Events:     {len(sim.events)}")

    # 2. Build temporal graph snapshots across the trajectory
    print("\n[2] Converting telemetry event stream into temporal graph snapshots G(t)...")
    builder = TemporalGraphBuilder()
    agent_roles = {a.agent_id: (a.role.value if hasattr(a.role, "value") else str(a.role)) for a in sim.agents}
    known_agents = [a.agent_id for a in sim.agents]

    snapshots = builder.build_snapshots_over_run(
        events=sim.events,
        known_agents=known_agents,
        agent_roles=agent_roles,
    )
    # Attach future prediction targets (K=1, 3, 5)
    builder.attach_prediction_targets(snapshots, sim.events, horizons=[1, 3, 5])

    print(f"    Total Snapshots Built: {len(snapshots)}")

    # 3. Print details per snapshot
    print("\n--- SNAPSHOT TRAJECTORY EVOLUTION ---")
    for i, snap in enumerate(snapshots):
        print(f"  Snapshot {i} | t={snap.timestamp:.3f}s (step={snap.step_idx}): "
              f"Nodes={snap.num_nodes}, Edges={snap.num_edges}, Targets={snap.metadata.get('targets')}")

    # 4. Print sample node and edge features from the final snapshot
    final_snap = snapshots[-1]
    print(f"\n--- SAMPLE NODE FEATURES (at t={final_snap.timestamp:.3f}s) ---")
    sample_agent_id = "researcher_1"
    sample_node_feats = final_snap.get_node_features(sample_agent_id)
    if sample_node_feats:
        print(f"  Agent [{sample_agent_id}]:")
        for k, v in sample_node_feats.items():
            print(f"    * {k:<25}: {v}")

    print(f"\n--- SAMPLE EDGE FEATURES (at t={final_snap.timestamp:.3f}s) ---")
    sample_edge_key = ("researcher_1", "analyst_1")
    sample_edge_feats = final_snap.get_edge_features(*sample_edge_key)
    if sample_edge_feats:
        print(f"  Directed Edge [{sample_edge_key[0]} -> {sample_edge_key[1]}]:")
        for k, v in sample_edge_feats.items():
            print(f"    * {k:<25}: {v}")

    # 5. Visual representation
    print("\n--- VISUAL GRAPH SNAPSHOT ---")
    visualize_snapshot(final_snap, print_ascii=True)

    # 6. Save snapshot representation to disk
    output_dir = _PROJECT_ROOT / "data" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = output_dir / f"{run_id}_snapshot_final.json"
    print(f"\n[5] Saving graph snapshot to disk: {snapshot_file.relative_to(_PROJECT_ROOT)}...")
    save_snapshot(final_snap, snapshot_file)
    print(f"    File saved successfully: size={snapshot_file.stat().st_size} bytes")

    # 7. Confirm successful round-trip reconstruction
    print("\n[6] Testing round-trip reconstruction from saved file...")
    reconstructed = load_snapshot(snapshot_file)
    assert reconstructed.run_id == final_snap.run_id
    assert reconstructed.timestamp == final_snap.timestamp
    assert reconstructed.num_nodes == final_snap.num_nodes
    assert reconstructed.num_edges == final_snap.num_edges
    assert reconstructed.nodes == final_snap.nodes
    assert reconstructed.edges == final_snap.edges
    print("    [OK] Exact match: All nodes, edges, and features verified without data loss.")

    print("\n" + "=" * 75)
    print(" Temporal Graph Building and Verification Complete: SUCCESS")
    print("=" * 75)


if __name__ == "__main__":
    main()
