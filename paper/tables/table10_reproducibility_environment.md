# Table 10: Experimental Environment, Software Provenance, and System Specifications

| Provenance Dimension | Evaluated Specification | Verification Method |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Enterprise (Version 10.0.26300 SP0) x86_64 | Python `platform.platform()` |
| **Processor Architecture** | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel | Python `platform.processor()` |
| **Python Runtime** | Python 3.14.7 (`tags/v3.14.7:823f032`) | `sys.version` |
| **Deep Learning Framework** | PyTorch `2.14.1+cpu` (CPU Compute Backend) | `torch.__version__`, `torch.cuda.is_available()` |
| **Graph Neural Network Framework** | PyTorch Geometric `2.8.0.post1` | `torch_geometric.__version__` |
| **Machine Learning Suite** | Scikit-Learn `1.9.1`, XGBoost `3.4.1`, NumPy `2.5.3` | Package `__version__` inspection |
| **Backend Framework** | FastAPI `0.142.2`, Uvicorn `0.54.0`, SQLAlchemy `2.1.3` | Package `__version__` inspection |
| **Frontend Runtime** | Node.js `v24.21.0`, npm `11.19.0`, Vite `5.0.3`, React `18.3.1` | `node --version`, `npm.cmd --version` |
| **Static Security Scanning** | Bandit `1.9.4` AST Security Analyzer | `bandit --version` (0 Med / 0 High) |
| **Git Version Control Commit** | `f2d56bcb09894ba1ef5585bcead77e9387b53a81` | `git rev-parse HEAD` resolution |
| **Master Pseudo-Random Seed** | Fixed seed `42` across all generators and trainers | Manifest configuration records |
| **Evaluation Resampling** | Trajectory Block Bootstrap ($B=500$, run-level resampling) | `ml/evaluation/uncertainty.py` |
