# Table 1: Experimental Dataset Configurations and Partition Statistics

| Dataset Characteristic | Benchmark Dataset (`agentguard_dataset_v1`) | Generalization Dataset (`agentguard_generalization_v1`) |
| :--- | :---: | :---: |
| **Primary Evaluation Purpose** | Baseline Comparison, Horizons & Ablations | Out-of-Distribution Scaling, Topology, Task & Fault Transfer |
| **Total Simulation Trajectories** | 20 runs | 72 runs |
| **Total Prediction Samples** | 305 samples | 1,098 samples |
| **Class Distribution (Positive / Negative)** | 118 positive (38.69%) / 187 negative (61.31%) | 769 positive (70.04%) / 329 negative (29.96%) |
| **Training Partition** | 14 runs (70%) / 224 samples (73.4%) | 50 runs (69.4%) / 746 samples (67.9%) |
| **Validation Partition** | 3 runs (15%) / 46 samples (15.1%) | 11 runs (15.3%) / 138 samples (12.6%) |
| **Held-Out Test Partition** | 3 runs (15%) / 35 samples (11.5%) | 11 runs (15.3%) / 214 samples (19.5%) |
| **Prediction Horizons $K$ (Steps Ahead)** | $\{1, 3, 5, 10, 20\}$ | $\{1, 3, 5, 10, 20\}$ |
| **Agent Counts Tested** | $\{3, 5, 8, 12\}$ | $\{3, 5, 8, 12\}$ |
| **Communication Topologies** | Pipeline, Star, Mesh, Custom | Pipeline, Star, Mesh, Custom |
| **Benchmark Task Categories** | Research, Coding, Analysis, Planning | Research, Coding, Analysis, Planning |
| **Active Fault Modes** | 3 modes (`none`, `hallucinated_output`, `delayed_response`) | All 12 canonical fault modes |
| **Storage Formats** | Apache Parquet (`.parquet`) + JSON Lines (`.jsonl`) | Apache Parquet (`.parquet`) + JSON Lines (`.jsonl`) |
| **Master Pseudo-Random Seed** | 42 | 42 |
