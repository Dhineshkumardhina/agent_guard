"""Lightweight Graph Visualization Utility for Research and Debugging.

Provides text/ASCII and optional matplotlib/graphviz visualizations of dynamic interaction graphs:
- Agent nodes with roles, activity, and error status
- Directed communication channels (A -> B)
- Edge interaction counts, latencies, and contradiction scores
"""

from typing import Optional, Union
from pathlib import Path

from ml.graph.graph_snapshot import GraphSnapshot


def render_ascii_graph(snapshot: GraphSnapshot) -> str:
    """Generate a clean ASCII textual representation of the interaction graph."""
    lines = [
        "=" * 60,
        f" Graph Snapshot: t={snapshot.timestamp}s (step={snapshot.step_idx})",
        f" Run ID: {snapshot.run_id} | Nodes: {snapshot.num_nodes} | Edges: {snapshot.num_edges}",
        "=" * 60,
        "\n--- NODES (Agents) ---",
    ]

    for aid, attrs in sorted(snapshot.nodes.items()):
        role = attrs.get("role", "agent")
        errs = attrs.get("error_count", 0)
        conf = attrs.get("average_confidence", 1.0)
        qual = attrs.get("average_output_quality", 1.0)
        status_flag = "[FAIL]" if errs > 0 else "[OK]"
        lines.append(
            f"  {status_flag} {aid:<16} role={role:<12} msgs={attrs.get('message_count', 0):<3} "
            f"errs={errs:<2} conf={conf:.2f} qual={qual:.2f}"
        )

    lines.append("\n--- DIRECTED EDGES (Interactions) ---")
    if not snapshot.edges:
        lines.append("  (No direct interaction edges in this window)")
    else:
        for (u, v), attrs in sorted(snapshot.edges.items()):
            cnt = attrs.get("interaction_count", 0)
            lat = attrs.get("average_latency", 0.0)
            contra = attrs.get("contradiction_rate", 0.0)
            err = attrs.get("error_count", 0)
            err_flag = " [!] ERR" if err > 0 else ""
            lines.append(
                f"  {u} ---> {v} (count={cnt}, latency={lat:.3f}s, contradiction={contra:.2f}){err_flag}"
            )

    targets = snapshot.metadata.get("targets", {})
    if targets:
        lines.append("\n--- PREDICTION HORIZON TARGETS ---")
        for k, v in targets.items():
            lines.append(f"  {k}: {v}")

    lines.append("=" * 60)
    return "\n".join(lines)


def visualize_snapshot(
    snapshot: GraphSnapshot,
    output_path: Optional[Union[str, Path]] = None,
    print_ascii: bool = True,
) -> str:
    """Visualize a graph snapshot using text output and optional matplotlib rendering.
    
    Args:
        snapshot: Target GraphSnapshot to visualize.
        output_path: Optional file path to save visualization (PNG/TXT).
        print_ascii: If True, prints ASCII graph summary to stdout.
        
    Returns:
        ASCII string representation of the graph.
    """
    ascii_repr = render_ascii_graph(snapshot)
    if print_ascii:
        print(ascii_repr)

    # If matplotlib is available and output_path ends in image extension
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() in (".png", ".jpg", ".svg", ".pdf"):
            try:
                import matplotlib.pyplot as plt
                import networkx as nx

                plt.figure(figsize=(8, 6))
                g = snapshot.to_networkx()
                pos = nx.spring_layout(g, seed=42)

                node_colors = []
                for n in g.nodes():
                    attrs = snapshot.nodes.get(n, {})
                    if attrs.get("error_count", 0) > 0:
                        node_colors.append("#ff6b6b")  # Red for error
                    else:
                        node_colors.append("#4dabf7")  # Blue for normal

                nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=1200, alpha=0.9)
                nx.draw_networkx_labels(g, pos, font_size=10, font_weight="bold")
                nx.draw_networkx_edges(g, pos, arrowstyle="->", arrowsize=20, edge_color="#495057", width=1.5)

                edge_labels = {
                    (u, v): f"{snapshot.edges.get((u, v), {}).get('interaction_count', '')}"
                    for u, v in g.edges()
                }
                nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)

                plt.title(f"AgentGuard Dynamic Interaction Graph G(t={snapshot.timestamp})", fontsize=12)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out, dpi=150)
                plt.close()
            except ImportError:
                # Fallback to saving text representation
                text_out = out.with_suffix(".txt")
                with open(text_out, "w", encoding="utf-8") as f:
                    f.write(ascii_repr)
        else:
            with open(out, "w", encoding="utf-8") as f:
                f.write(ascii_repr)

    return ascii_repr
