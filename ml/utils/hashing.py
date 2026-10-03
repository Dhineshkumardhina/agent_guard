"""Hashing and fingerprinting utilities for configs and datasets.

Enables cryptographic verification of experiment configurations to ensure
identical reproducibility and detect silent parameter drift.
"""

import json
import hashlib
from typing import Any, Dict


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Generate a deterministic SHA-256 fingerprint from a configuration dictionary."""
    # Ensure recursive key sorting and standard JSON serialization
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def compute_data_fingerprint(records: list[Dict[str, Any]]) -> str:
    """Compute a deterministic hash across a list of raw event or trajectory dictionaries."""
    hasher = hashlib.sha256()
    for rec in records:
        chunk = json.dumps(rec, sort_keys=True, default=str).encode("utf-8")
        hasher.update(chunk)
    return hasher.hexdigest()
