"""Dataset Summary CLI Script for AgentGuard.

Outputs:
- Dataset version
- Total runs and total samples
- Positive and negative samples & class ratio
- Failure modes and failure level breakdown (Level 0, 1, 2, 3)
- Topology distribution
- Task benchmark distribution
- Agent count distribution (3, 5, 8, 12)
- Prediction horizon distribution (k = 1, 3, 5, 10, 20)
- Train / Val / Test split breakdown

Usage:
    python scripts/dataset_summary.py --path data/processed/agentguard_dataset_v1
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.storage import DatasetStorage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print comprehensive statistical summary of an AgentGuard dataset."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="data/processed/agentguard_dataset_v1",
        help="Path to dataset directory.",
    )
    return parser.parse_args()


def print_dict_table(title: str, d: dict):
    print(f"\n--- {title} ---")
    if not d:
        print("  (None)")
        return
    for k, v in sorted(d.items(), key=lambda x: str(x[0])):
        print(f"  {str(k):<30} : {v}")


def main():
    args = parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Dataset directory does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    storage = DatasetStorage(path.parent)
    try:
        manifest = storage.load_manifest(path)
    except Exception as e:
        print(f"Error reading manifest: {e}", file=sys.stderr)
        sys.exit(1)

    cd = manifest.class_distribution or {}

    print("=" * 70)
    print("AgentGuard Dataset Summary Report")
    print("=" * 70)
    print(f"Dataset Version       : {manifest.dataset_version}")
    print(f"Creation Timestamp    : {manifest.creation_timestamp}")
    print(f"Simulator Version     : {manifest.simulator_version}")
    print(f"Random Seed           : {manifest.random_seed}")
    print(f"Total Trajectories    : {manifest.number_of_runs}")
    print(f"Total Samples         : {manifest.number_of_samples}")
    print(f"Positive Samples      : {cd.get('positive_samples', 0)}")
    print(f"Negative Samples      : {cd.get('negative_samples', 0)}")
    print(f"Positive Class Ratio  : {cd.get('positive_class_ratio', 0.0):.4f}")

    print_dict_table("Splits Distribution", manifest.split_distribution)
    print_dict_table("Failure Levels (0=None, 1=Agent, 2=Interaction, 3=Cascading)", cd.get("failure_levels", {}))
    print_dict_table("Failure Types", cd.get("failure_types", {}))
    print_dict_table("Topologies Distribution", cd.get("topologies", {}))
    print_dict_table("Task Benchmark Distribution", cd.get("tasks", {}))
    print_dict_table("Horizons (k-steps)", cd.get("horizons", {}))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
