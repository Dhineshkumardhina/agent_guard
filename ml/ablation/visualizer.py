"""Ablation Study Visualizer - Phase 13 (Section 10).

Generates publication-quality figures:
1. Ablation performance bar charts (F1 across ablations)
2. Metric degradation plots (Delta F1 / Delta AUROC vs Full model)
3. Horizon-wise ablation curves (F1 vs K)
4. Seed variability plots (mean +/- std across random seeds)
5. Lead-time comparison across ablations
6. Precision-Recall curves across ablations
7. AUPRC comparison ranking
8. Component contribution summary (ordered impact on predictive utility)
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.ablation.schema import AblationResultRecord, AblationComparisonRecord


class AblationVisualizer:
    """Generates publication-grade ablation visualizations."""

    def __init__(self, output_dir: str = "results/ablation/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "grid.alpha": 0.3,
            "axes.grid": True,
        })
        self.palette = [
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
            "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5"
        ]

    def plot_performance_bars(
        self,
        records: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "ablation_performance_bars.png",
    ) -> str:
        """Plot comparative F1 bar chart for all ablations at horizon K."""
        sub = [r for r in records if r.get("horizon") == horizon]
        if not sub:
            return ""

        # Aggregate across seeds if multiple exist
        grouped: Dict[str, List[float]] = {}
        for r in sub:
            name = r.get("ablation_name", "")
            grouped.setdefault(name, []).append(r.get("f1", 0.0))

        names = list(grouped.keys())
        means = [float(np.mean(grouped[n])) for n in names]
        stds = [float(np.std(grouped[n])) if len(grouped[n]) > 1 else 0.0 for n in names]

        # Sort so full model is first, then ascending by mean F1
        full_name = "Full Temporal GNN"
        if full_name in names:
            idx = names.index(full_name)
            names_rest = [n for i, n in enumerate(names) if i != idx]
            names_rest.sort(key=lambda n: np.mean(grouped[n]))
            ordered_names = [full_name] + names_rest
        else:
            ordered_names = names

        ord_means = [float(np.mean(grouped[n])) for n in ordered_names]
        ord_stds = [float(np.std(grouped[n])) if len(grouped[n]) > 1 else 0.0 for n in ordered_names]

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(ordered_names))
        colors = ["#2ca02c" if "Full" in n else "#1f77b4" for n in ordered_names]

        ax.barh(y_pos, ord_means, xerr=ord_stds, align="center", color=colors, alpha=0.85, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ordered_names)
        ax.invert_yaxis()
        ax.set_xlabel("F1 Score")
        ax.set_xlim([0.0, 1.05])
        ax.set_title(f"Ablation Performance Comparison (Horizon K={horizon})")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_metric_degradation(
        self,
        comparisons: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "metric_degradation.png",
    ) -> str:
        """Plot Delta F1 (Ablated - Full) showing performance drop when components are removed."""
        sub = [c for c in comparisons if c.get("horizon") == horizon and c.get("metric") == "f1"]
        if not sub:
            return ""

        sub.sort(key=lambda x: x.get("difference", 0.0))
        names = [c.get("removed_component", c.get("ablation_name", "")) for c in sub]
        diffs = [c.get("difference", 0.0) for c in sub]
        ci_lows = [abs(c.get("difference", 0.0) - c.get("ci_lower", 0.0)) for c in sub]
        ci_highs = [abs(c.get("ci_upper", 0.0) - c.get("difference", 0.0)) for c in sub]
        errs = [ci_lows, ci_highs]

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(names))
        colors = ["#d62728" if d < 0 else "#2ca02c" for d in diffs]

        ax.barh(y_pos, diffs, xerr=errs, align="center", color=colors, alpha=0.85, capsize=3)
        ax.axvline(0, color="black", lw=1.0, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Delta F1 Score (Ablated - Full)")
        ax.set_title(f"Impact of Component Removal on F1 Score (Horizon K={horizon})")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_horizon_wise_ablation(
        self,
        records: List[Dict[str, Any]],
        metric: str = "f1",
        filename: str = "horizon_wise_ablation.png",
    ) -> str:
        """Plot F1 or AUPRC across prediction horizons K for each ablation."""
        fig, ax = plt.subplots(figsize=(9, 5))

        grouped: Dict[str, Dict[int, List[float]]] = {}
        for r in records:
            name = r.get("ablation_name", "")
            h = r.get("horizon", 1)
            grouped.setdefault(name, {}).setdefault(h, []).append(r.get(metric, 0.0))

        # Focus on key ablations to keep plot readable
        key_names = [
            "Full Temporal GNN",
            "No Temporal Information",
            "No Graph Structure",
            "No Temporal Memory",
            "No Failure History",
        ]
        color_map = {
            "Full Temporal GNN": "#2ca02c",
            "No Temporal Information": "#d62728",
            "No Graph Structure": "#9467bd",
            "No Temporal Memory": "#ff7f0e",
            "No Failure History": "#1f77b4",
        }

        for name in key_names:
            if name not in grouped:
                continue
            h_dict = grouped[name]
            hs = sorted(h_dict.keys())
            vals = [float(np.mean(h_dict[h])) for h in hs]

            ax.plot(
                hs,
                vals,
                marker="o",
                lw=2,
                label=name,
                color=color_map.get(name, "#7f7f7f"),
                alpha=0.9,
            )

        ax.set_xlabel("Prediction Horizon K (Steps Ahead)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs Prediction Horizon Across Key Ablations")
        ax.set_xticks([1, 3, 5, 10, 20])
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc="best", framealpha=0.9)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_seed_variability(
        self,
        records: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "seed_variability.png",
    ) -> str:
        """Plot seed variation boxplots / distributions for major ablations."""
        sub = [r for r in records if r.get("horizon") == horizon]
        grouped: Dict[str, List[float]] = {}
        for r in sub:
            name = r.get("ablation_name", "")
            grouped.setdefault(name, []).append(r.get("f1", 0.0))

        # Filter to those with multiple seeds or all
        names = list(grouped.keys())
        data = [grouped[n] for n in names]

        fig, ax = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True, showmeans=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#aec7e8")
            patch.set_alpha(0.7)

        ax.set_ylabel("F1 Score")
        ax.set_title(f"Seed Variability Across Ablations (Horizon K={horizon})")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_lead_time_comparison(
        self,
        records: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "lead_time_comparison.png",
    ) -> str:
        """Plot mean and median early warning lead times across ablations."""
        sub = [r for r in records if r.get("horizon") == horizon]
        grouped: Dict[str, List[float]] = {}
        for r in sub:
            name = r.get("ablation_name", "")
            grouped.setdefault(name, []).append(r.get("mean_lead_time", 0.0))

        names = list(grouped.keys())
        means = [float(np.mean(grouped[n])) for n in names]

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, means, align="center", color="#ff7f0e", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Lead Time (Seconds)")
        ax.set_title(f"Early Warning Lead Time by Ablation (Horizon K={horizon})")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_pr_curves_ablation(
        self,
        curve_data_by_ablation: Dict[str, Dict[str, Any]],
        horizon: int = 1,
        filename: str = "pr_curves_ablation.png",
    ) -> str:
        """Plot comparative Precision-Recall curves across ablations."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for idx, (name, c_data) in enumerate(curve_data_by_ablation.items()):
            pr_data = c_data.get("pr", {})
            precision = pr_data.get("precision", [])
            recall = pr_data.get("recall", [])
            auc = pr_data.get("auc", 0.0)

            if precision and recall:
                color = self.palette[idx % len(self.palette)]
                ax.plot(
                    recall,
                    precision,
                    label=f"{name} (AUPRC = {auc:.3f})",
                    color=color,
                    lw=2,
                    alpha=0.85,
                )

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curves Across Ablations (Horizon K={horizon})")
        ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_auprc_comparison(
        self,
        records: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "auprc_comparison.png",
    ) -> str:
        """Plot comparative AUPRC bar chart across ablations."""
        sub = [r for r in records if r.get("horizon") == horizon]
        grouped: Dict[str, List[float]] = {}
        for r in sub:
            name = r.get("ablation_name", "")
            grouped.setdefault(name, []).append(r.get("auprc", 0.0))

        names = sorted(grouped.keys(), key=lambda n: np.mean(grouped[n]), reverse=True)
        means = [float(np.mean(grouped[n])) for n in names]

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, means, align="center", color="#9467bd", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("AUPRC")
        ax.set_xlim([0.0, 1.05])
        ax.set_title(f"Area Under Precision-Recall Curve (AUPRC) by Ablation (K={horizon})")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_component_contribution_summary(
        self,
        comparisons: List[Dict[str, Any]],
        horizon: int = 1,
        filename: str = "component_contribution_summary.png",
    ) -> str:
        """Plot ranking of component contribution magnitude based on F1 drop."""
        sub = [c for c in comparisons if c.get("horizon") == horizon and c.get("metric") == "f1"]
        if not sub:
            return ""

        # Contribution magnitude = |Delta F1|
        components = []
        for c in sub:
            comp = c.get("removed_component", c.get("ablation_name", ""))
            drop = abs(min(0.0, c.get("difference", 0.0)))
            components.append((comp, drop, c.get("difference", 0.0)))

        components.sort(key=lambda x: x[1], reverse=True)
        names = [x[0] for x in components]
        drops = [x[1] for x in components]

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, drops, align="center", color="#2ca02c", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Contribution Magnitude (|Delta F1|)")
        ax.set_title(f"Ranked Component Contribution to Predictive Performance (K={horizon})")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path
