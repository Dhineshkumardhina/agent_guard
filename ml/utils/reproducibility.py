"""Scientific Reproducibility Utilities for AgentGuard.

Provides deterministic random seed setting, environment provenance recording,
and audit trail verification for all ML and simulation experiments.
"""

import os
import random
import sys
import platform
import subprocess
from typing import Dict, Any, Optional
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set seeds across Python's random, numpy, and environment for reproducible execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Conditionally seed PyTorch if installed
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_commit_hash() -> Optional[str]:
    """Retrieve current Git commit hash if in a Git repository."""
    import shutil
    git_exec = shutil.which("git")
    if not git_exec:
        return None
    try:
        commit = subprocess.check_output(  # nosec
            [git_exec, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return None


def get_system_provenance() -> Dict[str, Any]:
    """Capture full runtime environment provenance for research paper reproducibility."""
    provenance = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "git_commit": get_git_commit_hash(),
    }
    
    try:
        import torch
        provenance["torch_version"] = torch.__version__
        provenance["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        provenance["torch_version"] = None
        provenance["cuda_available"] = False
        
    return provenance


def verify_no_future_leakage(
    current_step: int,
    current_timestamp: float,
    used_event_steps: list[int],
    used_event_timestamps: list[float],
) -> bool:
    """Strictly assert that no future events or timestamps contaminated current features.

    Raises:
        ValueError: If any utilized step or timestamp exceeds current time boundaries.
    """
    for step in used_event_steps:
        if step > current_step:
            raise ValueError(
                f"Data Leakage Detected: Step {step} used to compute features at step {current_step}!"
            )
            
    for ts in used_event_timestamps:
        if ts > current_timestamp:
            raise ValueError(
                f"Temporal Leakage Detected: Timestamp {ts} exceeds current timestamp {current_timestamp}!"
            )
            
    return True
