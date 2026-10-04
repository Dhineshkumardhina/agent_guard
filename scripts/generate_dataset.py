"""Dataset Generation CLI Script for AgentGuard (Phase 6).

Executes reproducible simulation, temporal graph extraction, causal prediction labeling,
and dataset artifact persistence.

Usage:
    python scripts/generate_dataset.py --runs 50 --seed 42 --version agentguard_dataset_v1
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset_builder import DatasetBuilder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scientific dataset for cascading failure prediction in multi-agent systems."
    )
    parser.add_argument("--runs", type=int, default=50, help="Number of simulation runs to execute.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="research,coding,analysis,planning",
        help="Comma-separated task types to simulate.",
    )
    parser.add_argument(
        "--topologies",
        type=str,
        default="pipeline,star,mesh,custom",
        help="Comma-separated topologies to simulate.",
    )
    parser.add_argument(
        "--agent-counts",
        type=str,
        default="3,5,8,12",
        help="Comma-separated agent counts to simulate.",
    )
    parser.add_argument(
        "--fault-prob",
        type=float,
        default=0.40,
        help="Probability of injecting a fault into a run.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,3,5,10,20",
        help="Comma-separated future prediction horizons k.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="agentguard_dataset_v1",
        help="Semantic dataset version string.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory for dataset artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    topo_list = [t.strip() for t in args.topologies.split(",") if t.strip()]
    agent_counts = [int(c.strip()) for c in args.agent_counts.split(",") if c.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    print("=" * 70)
    print("AgentGuard Dataset Generator (Phase 6)")
    print("=" * 70)
    print(f"Dataset Version   : {args.version}")
    print(f"Number of Runs    : {args.runs}")
    print(f"Random Seed       : {args.seed}")
    print(f"Tasks             : {task_list}")
    print(f"Topologies        : {topo_list}")
    print(f"Agent Counts      : {agent_counts}")
    print(f"Horizons (k)      : {horizons}")
    print(f"Fault Probability : {args.fault_prob}")
    print("-" * 70)

    builder = DatasetBuilder(
        num_runs=args.runs,
        random_seed=args.seed,
        task_types=task_list,
        topologies=topo_list,
        agent_counts=agent_counts,
        fault_probability=args.fault_prob,
        prediction_horizons=horizons,
        dataset_version=args.version,
        output_dir=args.output_dir,
    )

    print("Generating simulation trajectories and causal samples...")
    saved_path, manifest, val_result = builder.build_dataset()

    print("\nDataset generation completed successfully!")
    print(f"Output Directory  : {saved_path}")
    print(f"Total Trajectories: {manifest.number_of_runs}")
    print(f"Total Samples     : {manifest.number_of_samples}")
    print(f"Train Samples     : {manifest.split_distribution.get('train_samples', 0)}")
    print(f"Val Samples       : {manifest.split_distribution.get('val_samples', 0)}")
    print(f"Test Samples      : {manifest.split_distribution.get('test_samples', 0)}")
    print(f"Validation Status : {'PASSED (Zero Leakage)' if val_result.is_valid else 'FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
