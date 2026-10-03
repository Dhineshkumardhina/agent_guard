# Scientific Reproducibility Protocol

## 1. Principles
1. **Deterministic Execution:** Fixed pseudo-random seeds across random, numpy, and any DL backends.
2. **Configuration Fingerprinting:** Every experiment records an immutable SHA-256 hash of its parameters.
3. **Hardware & Environment Provenance:** Automatic capture of OS, Python version, platform architecture, dependency lockfiles, and git commit hash.
4. **No Cherry-Picking:** All runs, including negative or statistically insignificant outcomes, are serialized and retained in `results/`.
5. **No Synthetic Result Fabrication:** Evaluation tables are populated solely from executable evaluation scripts.

---

## 2. Reproduction Steps

### Environment Initialization
```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r backend/requirements.txt
```

### Verification & Test Suite
```bash
pytest tests/ -v
```

### Automated Configuration Verification
```python
from ml.utils.hashing import compute_config_hash
from ml.config.experiment_config import ExperimentConfig

config = ExperimentConfig(experiment_id="exp_001")
print("Fingerprint:", compute_config_hash(config.model_dump()))
```
