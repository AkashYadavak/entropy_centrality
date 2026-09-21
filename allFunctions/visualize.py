"""Plotting: draw the network coloured by entropy centrality.

Figures are shown on screen only -- nothing is written to disk.

matplotlib is imported inside the function so the rest of the project still
runs on a machine without it; plotting is a nice-to-have, not part of the
measure.
"""

from __future__ import annotations

from typing import Dict, Hashable

import networkx as nx

__all__ = ["show_plots"]

# Past this size a spring layout is slow and the picture is an unreadable
# hairball, so we say so rather than pretending it is useful.
_READABLE_LIMIT = 300


def show_plots(
    G: nx.Graph, measures: Dict[str, Dict[Hashable, float]], seed: int = 42
) -> None:
    """Open two figures: the network coloured by entropy, and entropy vs degree.

    The scatter is often the more useful of the two on bigger networks, since
    the node-link diagram becomes unreadable well before 456 nodes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping plots (pip install matplotlib)")
        return

    entropy = measures["entropy"]
    degree = measures["degree"]
    nodes = list(G.nodes())
    values = [entropy[v] for v in nodes]

    if G.number_of_nodes() > _READABLE_LIMIT:
        print(f"note: {G.number_of_nodes()} nodes is past the point where a "
              "node-link diagram is readable; treat it as decorative.")

    # --- figure 1: the network itself -------------------------------------
    # A fixed seed keeps the layout reproducible, which matters if the figure
    # ends up in a report next to specific node labels.
    layout = nx.spring_layout(G, seed=seed)
    low, high = min(values), max(values)
    spread = (high - low) or 1.0  # avoid divide-by-zero on a symmetric graph

    figure, axis = plt.subplots(figsize=(11, 8))
    drawn = nx.draw_networkx_nodes(
        G, layout, ax=axis,
        node_color=values, cmap="viridis",
        node_size=[120 + 900 * (v - low) / spread for v in values],
        edgecolors="white", linewidths=0.4,
    )
    nx.draw_networkx_edges(G, layout, ax=axis, alpha=0.25, width=0.6)
    if G.number_of_nodes() <= 80:
        nx.draw_networkx_labels(G, layout, ax=axis, font_size=7)

    figure.colorbar(drawn, ax=axis, label="entropy centrality")
    axis.set_title(f"{G.graph.get('name', 'Network')} - entropy centrality")
    axis.axis("off")
    figure.tight_layout()

    # --- figure 2: entropy against degree ---------------------------------
    # Points far off the diagonal trend are the nodes worth discussing.
    figure2, axis2 = plt.subplots(figsize=(7, 6))
    axis2.scatter([degree[v] for v in nodes], values,
                  s=28, alpha=0.75, edgecolors="white", linewidths=0.4)
    axis2.set_xlabel("degree centrality")
    axis2.set_ylabel("entropy centrality")
    axis2.set_title("entropy vs degree")
    axis2.grid(alpha=0.25)
    figure2.tight_layout()

    print("showing plots -- close the figure windows to continue")
    plt.show()
