"""Dataset Validation CLI Script for AgentGuard.

Verifies:
- Schema and column types
- Missing values and NaNs
- Valid, non-negative timestamps
- Duplicate sample detection
- Label validity (binary label in {0,1}, failure_level in {0,1,2,3})
- Strict future leakage prevention
- Trajectory split contamination (zero overlap between train, val, and test)
- Class distribution metrics
- Graph reference integrity
- Broken run references

Usage:
    python scripts/validate_dataset.py --path data/processed/agentguard_dataset_v1
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.validation import DatasetValidator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate an AgentGuard dataset directory for schema compliance and leakage."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="data/processed/agentguard_dataset_v1",
        help="Path to dataset directory containing manifest.json and parquet splits.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.path)

    print("=" * 70)
    print("AgentGuard Dataset Validator")
    print(f"Target Directory: {dataset_path}")
    print("=" * 70)

    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    validator = DatasetValidator(strict_mode=True)
    result = validator.validate_dataset_directory(dataset_path)

    print(f"Total Checks Executed : {result.checks_run}")
    print(f"Status                : {'PASSED' if result.is_valid else 'FAILED'}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  [WARN] {w}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):", file=sys.stderr)
        for e in result.errors:
            print(f"  [ERROR] {e}", file=sys.stderr)
        print("\nDataset validation FAILED!", file=sys.stderr)
        sys.exit(1)

    print("\nDataset Summary Statistics:")
    for k, v in result.statistics.items():
        print(f"  - {k}: {v}")

    print("\nAll validation checks passed with ZERO data leakage.")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
