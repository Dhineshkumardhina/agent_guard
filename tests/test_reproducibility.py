"""Tests for scientific reproducibility, seeding, and leakage prevention."""

import random
import numpy as np
import pytest

from ml.utils.reproducibility import (
    seed_everything,
    get_system_provenance,
    verify_no_future_leakage,
)
from ml.utils.hashing import compute_config_hash, compute_data_fingerprint


def test_seed_everything_determinism():
    """Verify that seed_everything produces identical random sequences."""
    seed_everything(12345)
    seq1_rand = [random.random() for _ in range(5)]
    seq1_np = np.random.rand(5).tolist()

    seed_everything(12345)
    seq2_rand = [random.random() for _ in range(5)]
    seq2_np = np.random.rand(5).tolist()

    assert seq1_rand == seq2_rand
    assert seq1_np == seq2_np


def test_system_provenance_structure():
    """Verify provenance metadata extraction contains expected fields."""
    prov = get_system_provenance()
    assert "python_version" in prov
    assert "platform" in prov
    assert "numpy_version" in prov


def test_compute_config_hash_consistency():
    """Verify deterministic hash for identical configs regardless of key order."""
    cfg1 = {"num_agents": 5, "topology": "pipeline", "seed": 42}
    cfg2 = {"seed": 42, "topology": "pipeline", "num_agents": 5}
    assert compute_config_hash(cfg1) == compute_config_hash(cfg2)


def test_compute_data_fingerprint():
    """Verify data fingerprint generation across event lists."""
    records1 = [{"step": 0, "val": 1.0}, {"step": 1, "val": 2.0}]
    records2 = [{"step": 0, "val": 1.0}, {"step": 1, "val": 2.0}]
    assert compute_data_fingerprint(records1) == compute_data_fingerprint(records2)


def test_verify_no_future_leakage_passes():
    """Verify leakage check passes when all steps and timestamps are in the past."""
    assert verify_no_future_leakage(
        current_step=5,
        current_timestamp=5.2,
        used_event_steps=[0, 1, 2, 3, 4, 5],
        used_event_timestamps=[0.5, 1.2, 2.3, 3.4, 4.5, 5.2],
    )


def test_verify_no_future_leakage_detects_step_leak():
    """Verify error raised if a future step enters feature calculation."""
    with pytest.raises(ValueError, match="Data Leakage Detected"):
        verify_no_future_leakage(
            current_step=5,
            current_timestamp=5.2,
            used_event_steps=[0, 1, 6],  # Step 6 is future!
            used_event_timestamps=[0.5, 1.2, 6.0],
        )


def test_verify_no_future_leakage_detects_timestamp_leak():
    """Verify error raised if a future timestamp enters feature calculation."""
    with pytest.raises(ValueError, match="Temporal Leakage Detected"):
        verify_no_future_leakage(
            current_step=5,
            current_timestamp=5.0,
            used_event_steps=[0, 1, 5],
            used_event_timestamps=[0.5, 1.2, 5.5],  # 5.5 > 5.0!
        )
