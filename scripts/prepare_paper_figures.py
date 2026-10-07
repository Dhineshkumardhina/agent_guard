"""Publication Figure Generator and Organizer for AgentGuard Research Paper (Phase 20).

Maps existing experimental plots and generates architectural schematics for
Figures 1 through 10 in paper/figures/.
"""

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_figure1_architecture():
    """Generate publication-ready Figure 1: System Architecture Schematic."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis("off")

    stages = [
        ("1. Multi-Agent\nSimulation", "#E8F0FE", "#1A73E8"),
        ("2. Fault\nInjection", "#FCE8E6", "#D93025"),
        ("3. Telemetry\nCollector", "#E6F4EA", "#1E8E3E"),
        ("4. Dynamic Graph\nBuilder G(t)", "#FEF7E0", "#F9AB00"),
        ("5. Baseline & TGN\nModels", "#F3E8FD", "#9334E6"),
        ("6. Early Warning &\nEvaluation", "#E0F2F1", "#00897B"),
        ("7. Explainability &\nDashboard", "#EFEBE9", "#6D4C41"),
    ]

    for i, (label, bg_color, border_color) in enumerate(stages):
        x = 0.5 + i * 1.55
        y = 3.0
        rect = patches.FancyBboxPatch(
            (x - 0.65, y - 0.7), 1.3, 1.4,
            boxstyle="round,pad=0.1",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold", color="#202124")

        if i < len(stages) - 1:
            ax.annotate(
                "", xy=(x + 0.9, y), xytext=(x + 0.65, y),
                arrowprops=dict(arrowstyle="->", lw=2, color="#5F6368"),
            )

    ax.text(5.5, 5.0, "AgentGuard End-to-End System Architecture", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(5.5, 4.4, "Continuous-Time Temporal Graph Formulation for Multi-Agent Failure Prediction", ha="center", va="center", fontsize=10, style="italic", color="#5F6368")

    # Invariants box at bottom
    inv_rect = patches.FancyBboxPatch(
        (0.5, 0.6), 10.0, 1.2,
        boxstyle="round,pad=0.08",
        facecolor="#F8F9FA",
        edgecolor="#BDC1C6",
        linewidth=1,
    )
    ax.add_patch(inv_rect)
    ax.text(
        5.5, 1.2,
        "Zero-Future-Leakage Invariants: Event History t_event <= t_eval  |  Prediction Target Horizon (t, t+k]  |  Run-Level Partitioning",
        ha="center", va="center", fontsize=8.5, fontweight="medium", color="#3C4043"
    )

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    out_path = FIGURES_DIR / "fig1_system_architecture.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Generated: {out_path}")


def generate_figure2_temporal_interaction():
    """Generate Figure 2: Multi-Agent Temporal Interaction Graph Example."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis("off")

    nodes = {
        "Planner": (2.0, 3.5),
        "Researcher": (5.0, 4.2),
        "Analyst": (5.0, 2.5),
        "Coder": (8.0, 3.8),
        "Verifier": (8.0, 2.0),
    }

    colors = {
        "Planner": "#1A73E8",
        "Researcher": "#00897B",
        "Analyst": "#F9AB00",
        "Coder": "#9334E6",
        "Verifier": "#E53935",
    }

    for name, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.55, facecolor="#F8F9FA", edgecolor=colors[name], linewidth=2.5)
        ax.add_patch(circle)
        ax.text(x, y, name, ha="center", va="center", fontsize=9, fontweight="bold")

    interactions = [
        ("Planner", "Researcher", "e1: Task Spec\n(t=0.0s, Δt=0.0)"),
        ("Researcher", "Analyst", "e2: Context Data\n(t=1.4s, Δt=1.4)"),
        ("Analyst", "Coder", "e3: Feature Logic\n(t=2.8s, Δt=1.4)"),
        ("Coder", "Verifier", "e4: Script Unit\n(t=4.1s, Δt=1.3)"),
        ("Verifier", "Planner", "e5: Validation Reject\n(t=5.9s, Δt=1.8, Conflict)"),
    ]

    for src, tgt, label in interactions:
        x1, y1 = nodes[src]
        x2, y2 = nodes[tgt]
        dx = x2 - x1
        dy = y2 - y1
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + (0.35 if src == "Verifier" else 0.15)
        ax.annotate(
            "", xy=(x2 - 0.5 * (dx/((dx**2+dy**2)**0.5)), y2 - 0.5 * (dy/((dx**2+dy**2)**0.5))),
            xytext=(x1 + 0.5 * (dx/((dx**2+dy**2)**0.5)), y1 + 0.5 * (dy/((dx**2+dy**2)**0.5))),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#D93025" if "Conflict" in label else "#1A73E8"),
        )
        ax.text(mid_x, mid_y, label, ha="center", va="center", fontsize=7.5, color="#202124",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#DADCE0", alpha=0.9))

    ax.text(5.0, 5.2, "Continuous-Time Interaction Flow with Fourier Encoded Intervals Δt",
            ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(1.0, 5.6)
    out_path = FIGURES_DIR / "fig2_temporal_interaction_example.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Generated: {out_path}")


def generate_figure3_failure_propagation():
    """Generate Figure 3: Failure Propagation Schematic (L1 -> L2 -> L3)."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.axis("off")

    levels = [
        ("Level 1: Local Agent Failure", "Initial trigger: Tool timeout or local hallucination\ncontained in Coder agent", (2.0, 2.5), "#E8F0FE", "#1A73E8"),
        ("Level 2: Interaction Failure", "Propagation: Coder sends malformed payload to Verifier;\nVerifier rejects; retry ping-pong initiated", (5.5, 2.5), "#FEF7E0", "#F9AB00"),
        ("Level 3: Systemic Cascade", "Catastrophe: Planner budget exhausted; workflow deadlocks;\nmulti-hop coordination failure", (9.0, 2.5), "#FCE8E6", "#D93025"),
    ]

    for title, desc, (x, y), bg_col, edge_col in levels:
        box = patches.FancyBboxPatch(
            (x - 1.4, y - 1.1), 2.8, 2.2,
            boxstyle="round,pad=0.15",
            facecolor=bg_col,
            edgecolor=edge_col,
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.5, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#202124")
        ax.text(x, y - 0.2, desc, ha="center", va="center", fontsize=7.5, color="#3C4043")

    ax.annotate("", xy=(3.8, 2.5), xytext=(3.45, 2.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#5F6368"))
    ax.annotate("", xy=(7.3, 2.5), xytext=(6.95, 2.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#5F6368"))

    ax.text(5.5, 4.2, "Three-Level Failure Escalation Hierarchy in Multi-Agent Workflows",
            ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0.2, 10.8)
    ax.set_ylim(0.8, 4.6)
    out_path = FIGURES_DIR / "fig3_failure_propagation_example.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Generated: {out_path}")


def copy_experimental_figures():
    """Copy experimentally generated figures from results/ directories into paper/figures/."""
    copies = [
        # Figure 4: Model comparison (ROC and PR curves)
        (PROJECT_ROOT / "results" / "evaluation" / "plots" / "roc_curves.png", FIGURES_DIR / "fig4_model_comparison_roc.png"),
        (PROJECT_ROOT / "results" / "evaluation" / "plots" / "pr_curves.png", FIGURES_DIR / "fig4_model_comparison_pr.png"),
        # Figure 5: Horizon performance
        (PROJECT_ROOT / "results" / "evaluation" / "plots" / "f1_vs_horizon.png", FIGURES_DIR / "fig5_horizon_performance.png"),
        # Figure 6: Early-warning lead time
        (PROJECT_ROOT / "results" / "evaluation" / "plots" / "lead_time_distributions.png", FIGURES_DIR / "fig6_early_warning_lead_time.png"),
        # Figure 7: Ablation results
        (PROJECT_ROOT / "results" / "ablation" / "plots" / "component_contribution_summary.png", FIGURES_DIR / "fig7_ablation_contributions.png"),
        # Figure 8: Generalization results
        (PROJECT_ROOT / "results" / "generalization" / "plots" / "generalization_gap_plots.png", FIGURES_DIR / "fig8_generalization_gaps.png"),
        # Figure 9: Explainability risk trajectories
        (PROJECT_ROOT / "results" / "explainability" / "plots" / "risk_trajectory_cases.png", FIGURES_DIR / "fig9_risk_trajectories.png"),
        # Figure 10: Interaction edge importance / case study
        (PROJECT_ROOT / "results" / "explainability" / "plots" / "interaction_edge_importance.png", FIGURES_DIR / "fig10_interaction_edge_importance.png"),
    ]

    for src, dst in copies:
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"[OK] Copied {src.name} -> {dst.name}")
        else:
            print(f"[WARNING] Missing source: {src}")


def main():
    print("Preparing publication figures for AgentGuard paper...")
    generate_figure1_architecture()
    generate_figure2_temporal_interaction()
    generate_figure3_failure_propagation()
    copy_experimental_figures()
    print("All paper figures compiled successfully in paper/figures/")


if __name__ == "__main__":
    main()
