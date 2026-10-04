"""Trajectory-Level Dataset Splitting for AgentGuard.

Implements leak-free trajectory / run-level partitioning:
- 70% Train
- 15% Validation
- 15% Test

CRITICAL DESIGN REQUIREMENT:
All prediction points originating from the same simulation trajectory (run_id)
are strictly assigned to the SAME partition. Samples from a single run are never
split across train and test sets, completely eliminating trajectory-level and
temporal leakage.
"""

from typing import List, Dict, Set, Tuple
import random

from ml.data.schema import PredictionSample


def split_trajectories(
    run_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, List[str]]:
    """Partition unique run_ids deterministically into train, validation, and test splits.
    
    Args:
        run_ids: Sequence of run identifiers.
        train_ratio: Fraction of runs allocated to training (default 0.70).
        val_ratio: Fraction of runs allocated to validation (default 0.15).
        test_ratio: Fraction of runs allocated to testing (default 0.15).
        random_seed: PRNG seed for deterministic assignment.
        
    Returns:
        Dictionary mapping "train", "val", "test" to lists of run_ids.
    """
    unique_runs = sorted(list(set(run_ids)))
    n_runs = len(unique_runs)
    if n_runs == 0:
        return {"train": [], "val": [], "test": []}

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    rng = random.Random(random_seed)
    shuffled_runs = list(unique_runs)
    rng.shuffle(shuffled_runs)

    if n_runs < 3:
        # Edge case for very small run counts
        return {
            "train": shuffled_runs,
            "val": [],
            "test": [],
        }

    n_val = max(1, int(round(n_runs * val_ratio)))
    n_test = max(1, int(round(n_runs * test_ratio)))
    n_train = n_runs - n_val - n_test

    if n_train < 1:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    train_runs = sorted(shuffled_runs[:n_train])
    val_runs = sorted(shuffled_runs[n_train: n_train + n_val])
    test_runs = sorted(shuffled_runs[n_train + n_val:])

    # Strict disjointness verification
    train_set = set(train_runs)
    val_set = set(val_runs)
    test_set = set(test_runs)

    assert train_set.isdisjoint(val_set), "Train and validation runs overlap!"
    assert train_set.isdisjoint(test_set), "Train and test runs overlap!"
    assert val_set.isdisjoint(test_set), "Validation and test runs overlap!"
    assert len(train_set | val_set | test_set) == n_runs, "Missing runs after splitting!"

    return {
        "train": train_runs,
        "val": val_runs,
        "test": test_runs,
    }


def split_samples(
    samples: List[PredictionSample],
    run_splits: Dict[str, List[str]],
) -> Dict[str, List[PredictionSample]]:
    """Assign prediction samples to train, val, and test splits based on their run_id.
    
    Args:
        samples: List of PredictionSample instances.
        run_splits: Dictionary of run_ids per split.
        
    Returns:
        Dictionary mapping "train", "val", "test" to lists of PredictionSample instances.
    """
    train_runs = set(run_splits.get("train", []))
    val_runs = set(run_splits.get("val", []))
    test_runs = set(run_splits.get("test", []))

    split_samples_map: Dict[str, List[PredictionSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for s in samples:
        if s.run_id in train_runs:
            s.split = "train"
            split_samples_map["train"].append(s)
        elif s.run_id in val_runs:
            s.split = "val"
            split_samples_map["val"].append(s)
        elif s.run_id in test_runs:
            s.split = "test"
            split_samples_map["test"].append(s)
        else:
            raise ValueError(f"Sample {s.sample_id} references unassigned run_id '{s.run_id}'")

    return split_samples_map
