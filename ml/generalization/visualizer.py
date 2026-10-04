"""Publication-Grade Visualizations for Generalization and Robustness - Phase 14 (Section 17).

Produces 8 publication figures:
1. In-Distribution vs Out-of-Distribution Performance (Bar charts with error bars)
2. Agent-Count Scaling Performance Curves (3, 5, 8, 12 agents)
3. Topology Performance Comparison (Pipeline, Star, Mesh, Custom)
4. Task Performance Comparison (Research, Coding, Analysis, Planning)
5. Failure-Type Heatmap (Failure Mode x Metric)
6. Generalization Gap Plots (Delta Metric ID - OOD with 95% CIs)
7. Lead-Time Comparison (Early warning lead-time distributions)
8. AUPRC Comparison (Precision-Recall trade-off across distribution shifts)
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GeneralizationVisualizer:
    """Renders scientific publication figures for out-of-distribution transfer analysis."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure matplotlib for high-clarity scientific publication aesthetics."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })

    def plot_id_vs_ood(
        self,
        results: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "id_vs_ood_performance.png",
    ) -> Path:
        """Plot 1: In-Distribution vs OOD Performance across experiments."""
        fig, ax = plt.subplots(figsize=(10, 5))

        # Filter by horizon
        h_recs = [r for r in results if r.get("horizon") == horizon]
        exp_ids = sorted(list(set(r["experiment_id"] for r in h_recs)))

        if not exp_ids:
            exp_ids = ["G1", "G2", "G3", "G4", "G5"]

        id_f1 = []
        ood_f1 = []
        for eid in exp_ids:
            id_rec = next((r for r in h_recs if r["experiment_id"] == eid and r["split_type"] == "in_distribution"), None)
            ood_rec = next((r for r in h_recs if r["experiment_id"] == eid and r["split_type"] == "out_of_distribution"), None)
            id_f1.append(id_rec["f1"] if id_rec else 0.0)
            ood_f1.append(ood_rec["f1"] if ood_rec else 0.0)

        x = np.arange(len(exp_ids))
        width = 0.35

        ax.bar(x - width / 2, id_f1, width, label="In-Distribution (Test-ID)", color="#2b5c8f", edgecolor="black", alpha=0.9)
        ax.bar(x + width / 2, ood_f1, width, label="Out-of-Distribution (Test-OOD)", color="#d95f02", edgecolor="black", alpha=0.9)

        ax.set_ylabel("F1 Score")
        ax.set_title(f"In-Distribution vs. Out-of-Distribution Performance (Horizon K={horizon})")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_ids)
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_agent_count_curves(
        self,
        breakdowns: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "agent_count_curves.png",
    ) -> Path:
        """Plot 2: Performance curves as multi-agent population scales (3, 5, 8, 12)."""
        fig, ax = plt.subplots(figsize=(8, 5))

        counts = [3, 5, 8, 12]
        f1_vals = []
        auprc_vals = []

        for c in counts:
            match = next((b for b in breakdowns if b.get("dimension") == "agent_count" and str(b.get("subgroup")) == str(c) and b.get("horizon") == horizon), None)
            f1_vals.append(match["f1"] if match else 0.0)
            auprc_vals.append(match["auprc"] if match else 0.0)

        ax.plot(counts, f1_vals, marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="F1 Score")
        ax.plot(counts, auprc_vals, marker="s", linewidth=2.5, markersize=8, color="#2ca02c", linestyle="--", label="AUPRC")

        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"Predictive Performance vs. Multi-Agent Population Size (K={horizon})")
        ax.set_xticks(counts)
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_topology_comparison(
        self,
        breakdowns: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "topology_comparison.png",
    ) -> Path:
        """Plot 3: Topology-specific predictive performance."""
        fig, ax = plt.subplots(figsize=(8, 5))

        topos = ["pipeline", "star", "mesh", "custom"]
        f1_vals = []
        auprc_vals = []

        for t in topos:
            match = next((b for b in breakdowns if b.get("dimension") == "topology" and b.get("subgroup") == t and b.get("horizon") == horizon), None)
            f1_vals.append(match["f1"] if match else 0.0)
            auprc_vals.append(match["auprc"] if match else 0.0)

        x = np.arange(len(topos))
        width = 0.35

        ax.bar(x - width / 2, f1_vals, width, label="F1 Score", color="#386cb0", edgecolor="black")
        ax.bar(x + width / 2, auprc_vals, width, label="AUPRC", color="#7fc97f", edgecolor="black")

        ax.set_ylabel("Metric Value")
        ax.set_title(f"Predictive Performance Across Communication Topologies (K={horizon})")
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in topos])
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_task_comparison(
        self,
        breakdowns: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "task_comparison.png",
    ) -> Path:
        """Plot 4: Task workflow domain performance."""
        fig, ax = plt.subplots(figsize=(8, 5))

        tasks = ["research", "coding", "analysis", "planning"]
        f1_vals = []
        auprc_vals = []

        for t in tasks:
            match = next((b for b in breakdowns if b.get("dimension") == "task" and b.get("subgroup") == t and b.get("horizon") == horizon), None)
            f1_vals.append(match["f1"] if match else 0.0)
            auprc_vals.append(match["auprc"] if match else 0.0)

        x = np.arange(len(tasks))
        width = 0.35

        ax.bar(x - width / 2, f1_vals, width, label="F1 Score", color="#6a51a3", edgecolor="black")
        ax.bar(x + width / 2, auprc_vals, width, label="AUPRC", color="#bcbddc", edgecolor="black")

        ax.set_ylabel("Metric Value")
        ax.set_title(f"Performance Across Multi-Agent Task Workflows (K={horizon})")
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tasks])
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_failure_type_heatmap(
        self,
        breakdowns: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "failure_type_heatmap.png",
    ) -> Path:
        """Plot 5: Failure mode breakdown heatmap (Failure Type x Metrics)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ft_recs = [b for b in breakdowns if b.get("dimension") == "failure_type" and b.get("horizon") == horizon]
        if not ft_recs:
            ft_recs = [b for b in breakdowns if b.get("dimension") == "failure_type"]

        # Sort failure types
        ft_names = sorted(list(set(b["subgroup"] for b in ft_recs)))
        if not ft_names:
            ft_names = ["hallucinated_output", "tool_failure", "delayed_response", "contradictory_output", "tool_timeout"]

        metrics = ["precision", "recall", "f1", "auprc"]
        matrix = np.zeros((len(ft_names), len(metrics)))

        for i, ft in enumerate(ft_names):
            match = next((b for b in ft_recs if b["subgroup"] == ft), None)
            for j, m in enumerate(metrics):
                matrix[i, j] = match[m] if match else 0.0

        im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Metric Value")

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_yticks(np.arange(len(ft_names)))
        ax.set_yticklabels([f.replace("_", " ").title() for f in ft_names])
        ax.set_title(f"Failure Mode Specific Detection & Prediction Metrics (K={horizon})")

        # Overlay text
        for i in range(len(ft_names)):
            for j in range(len(metrics)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.6 else "black")

        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_generalization_gaps(
        self,
        gaps: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "generalization_gap_plots.png",
    ) -> Path:
        """Plot 6: Generalization Gap (Delta F1 = ID - OOD) with 95% CIs."""
        fig, ax = plt.subplots(figsize=(10, 5))

        h_gaps = [g for g in gaps if g.get("horizon") == horizon and g.get("metric") == "f1"]
        if not h_gaps:
            h_gaps = [g for g in gaps if g.get("metric") == "f1"]

        exp_ids = [g["experiment_id"] for g in h_gaps]
        gap_vals = [g["gap"] for g in h_gaps]
        ci_lowers = [max(0.0, float(g["gap"] - g["ci_lower"])) for g in h_gaps]
        ci_uppers = [max(0.0, float(g["ci_upper"] - g["gap"])) for g in h_gaps]
        errors = [ci_lowers, ci_uppers]

        y_pos = np.arange(len(exp_ids))
        colors = ["#d73027" if g > 0.15 else ("#fc8d59" if g > 0 else "#91bfdb") for g in gap_vals]

        ax.barh(y_pos, gap_vals, xerr=errors, align="center", color=colors, edgecolor="black", alpha=0.85, capsize=5)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(exp_ids)
        ax.invert_yaxis()
        ax.set_xlabel("Generalization Gap: ΔF1 (In-Distribution − OOD)")
        ax.set_title(f"Empirical Generalization Gap Across Distribution Shifts (K={horizon})")
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_lead_time_distribution(
        self,
        results: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "lead_time_distribution.png",
    ) -> Path:
        """Plot 7: Early warning lead-time across experiments."""
        fig, ax = plt.subplots(figsize=(10, 5))

        h_recs = [r for r in results if r.get("horizon") == horizon]
        exp_ids = sorted(list(set(r["experiment_id"] for r in h_recs)))

        id_lts = []
        ood_lts = []
        for eid in exp_ids:
            id_rec = next((r for r in h_recs if r["experiment_id"] == eid and r["split_type"] == "in_distribution"), None)
            ood_rec = next((r for r in h_recs if r["experiment_id"] == eid and r["split_type"] == "out_of_distribution"), None)
            id_lts.append(id_rec["mean_lead_time"] if id_rec else 0.0)
            ood_lts.append(ood_rec["mean_lead_time"] if ood_rec else 0.0)

        x = np.arange(len(exp_ids))
        width = 0.35

        ax.bar(x - width / 2, id_lts, width, label="In-Distribution (s)", color="#4575b4", edgecolor="black")
        ax.bar(x + width / 2, ood_lts, width, label="Out-of-Distribution (s)", color="#fdae61", edgecolor="black")

        ax.set_ylabel("Mean Lead Time (Seconds)")
        ax.set_title(f"Warning Advance Notice Before Cascade Onset (K={horizon})")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_ids)
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_auprc_distribution(
        self,
        results: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "auprc_distribution.png",
    ) -> Path:
        """Plot 8: AUPRC comparison across In-Distribution and OOD."""
        fig, ax = plt.subplots(figsize=(10, 5))

        h_recs = [r for r in results if r.get("horizon") == horizon]
        exp_ids = sorted(list(set(r["experiment_id"] for r in h_recs)))

        id_auprcs = []
        ood_auprcs = []
        for eid in exp_ids:
            id_rec = next((r for r in h_recs if r["experiment_id"] == eid and r["split_type"] == "in_distribution"), None)
            ood_rec = next((r for r in h_recs if r["experiment_id"] == eid and r["split_type"] == "out_of_distribution"), None)
            id_auprcs.append(id_rec["auprc"] if id_rec else 0.0)
            ood_auprcs.append(ood_rec["auprc"] if ood_rec else 0.0)

        x = np.arange(len(exp_ids))
        width = 0.35

        ax.bar(x - width / 2, id_auprcs, width, label="In-Distribution AUPRC", color="#1b7837", edgecolor="black")
        ax.bar(x + width / 2, ood_auprcs, width, label="Out-of-Distribution AUPRC", color="#af8dc3", edgecolor="black")

        ax.set_ylabel("AUPRC")
        ax.set_title(f"AUPRC Under Distribution Shifts (Horizon K={horizon})")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_ids)
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        out_path = self.output_dir / filename
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
