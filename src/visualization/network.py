from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import textwrap
from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.core.constants import BIG_M


def plot_network(
    problem: TransportationProblem,
    result: AlgorithmResult,
    figsize: tuple[float, float] = (12, 7),
    show_labels: bool = True,
) -> plt.Figure:
    def _wrap(label: str, width: int = 11) -> str:
        if len(label) <= width:
            return label
        return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))

    G = nx.DiGraph()
    pm, pn = problem.m, problem.n
    alloc = result.allocation[:pm, :pn]
    cost = problem.cost[:pm, :pn].astype(float)

    src_nodes = [f"Nguồn\n{_wrap(s)}" for s in problem.sources[:pm]]
    dst_nodes = [f"Đích\n{_wrap(d)}" for d in problem.destinations[:pn]]
    G.add_nodes_from(src_nodes, bipartite=0)
    G.add_nodes_from(dst_nodes, bipartite=1)

    edges = []
    widths = []
    labels = {}
    max_alloc = alloc.max() if alloc.max() > 0 else 1
    for i in range(pm):
        for j in range(pn):
            x = alloc[i, j]
            if x > 0:
                u = src_nodes[i]
                v = dst_nodes[j]
                G.add_edge(u, v, weight=x)
                edges.append((u, v))
                widths.append(1 + 5 * x / max_alloc)
                c_label = "∞" if cost[i, j] >= BIG_M / 2 else f"{cost[i,j]:.0f}"
                labels[(u, v)] = f"{x:.0f} @ {c_label}"

    pos = {}
    src_y = np.linspace(max(pm, pn) - 1, 0, pm) if pm > 1 else np.array([0.5])
    dst_y = np.linspace(max(pm, pn) - 1, 0, pn) if pn > 1 else np.array([0.5])
    for k, node in enumerate(src_nodes):
        pos[node] = (0, float(src_y[k]))
    for k, node in enumerate(dst_nodes):
        pos[node] = (3, float(dst_y[k]))

    width = max(figsize[0], 0.9 * pn + 4.0)
    height = max(figsize[1], 0.55 * max(pm, pn) + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    compact = (pm + pn) > 12 or len(edges) > 14
    nx.draw_networkx_nodes(G, pos, nodelist=src_nodes, node_color="lightblue",
                           node_size=900 if compact else 1100, edgecolors="#2563eb", linewidths=1.2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=dst_nodes, node_color="lightgreen",
                           node_size=900 if compact else 1100, edgecolors="#059669", linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7 if compact else 8, font_weight="bold")
    if edges:
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths,
                               edge_color="#2563eb", arrows=True,
                               arrowsize=12 if compact else 15, ax=ax,
                               connectionstyle="arc3,rad=0.08")
        if show_labels:
            if compact:
                short_labels = {edge: label.split(" @ ")[0] for edge, label in labels.items()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=short_labels, ax=ax,
                                             font_size=6, label_pos=0.5,
                                             bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#cbd5e1", "alpha": 0.85})
            else:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax,
                                             font_size=7, label_pos=0.5,
                                             bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#cbd5e1", "alpha": 0.9})
    else:
        ax.text(0.5, 0.5, "Chưa có tuyến phân bổ", ha="center", va="center",
                transform=ax.transAxes, color="#64748b", fontsize=11)

    # Legend
    ax.legend(handles=[
        mpatches.Patch(color="lightblue", label="Nguồn"),
        mpatches.Patch(color="lightgreen", label="Đích"),
    ], loc="upper center")

    ax.set_title(f"Network: {result.algorithm_name}", fontsize=13, fontweight="bold", color="#1e293b")
    ax.axis("off")
    fig.tight_layout()
    return fig
