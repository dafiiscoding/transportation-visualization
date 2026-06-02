from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from src.models.problem import TransportationProblem
from src.models.step import AlgorithmStep
from src.core.constants import BIG_M, EPSILON


def plot_cost_heatmap(
    problem: TransportationProblem,
    selected_cell: tuple[int, int] | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    cost = problem.cost.astype(float).copy()
    forbidden = problem.forbidden

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    display = cost.copy()
    if forbidden is not None:
        display[forbidden] = np.nan

    masked = np.ma.array(display, mask=np.isnan(display))
    im = ax.imshow(masked, cmap="YlOrRd", aspect="auto", vmin=np.nanmin(display) if not np.all(np.isnan(display)) else 0)
    cbar = plt.colorbar(im, ax=ax, label="Chi phí")
    cbar.ax.tick_params(labelsize=8)

    for i in range(problem.m):
        for j in range(problem.n):
            if forbidden is not None and forbidden[i, j]:
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#dc2626", alpha=0.7, zorder=2))
                ax.text(j, i, "∞", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
            else:
                val = cost[i, j]
                label = f"{val:.0f}" if val < BIG_M / 2 else "M"
                text_color = "white" if val > (np.nanmax(display) * 0.7 if not np.all(np.isnan(display)) else 1) else "#1e293b"
                ax.text(j, i, label, ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")

    if selected_cell is not None:
        si, sj = selected_cell
        ax.add_patch(mpatches.Rectangle((sj - 0.5, si - 0.5), 1, 1, fill=False,
                                         edgecolor="#2563eb", linewidth=3, zorder=3))

    ax.set_xticks(range(problem.n))
    ax.set_xticklabels(problem.destinations, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(problem.m))
    ax.set_yticklabels(problem.sources, fontsize=9)
    ax.set_title("Bảng chi phí", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax.spines[:].set_color("#cbd5e1")
    fig.tight_layout()
    return fig


def plot_allocation_heatmap(
    problem: TransportationProblem,
    allocation: np.ndarray,
    step: AlgorithmStep | None = None,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f0f9ff")

    pm, pn = problem.m, problem.n
    alloc = allocation[:pm, :pn]

    im = ax.imshow(alloc, cmap="Blues", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="Số lượng")
    cbar.ax.tick_params(labelsize=8)

    for i in range(pm):
        for j in range(pn):
            val = alloc[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="#1e3a5f")

    if step and step.selected_cell:
        si, sj = step.selected_cell
        if si < pm and sj < pn:
            ax.add_patch(mpatches.Rectangle((sj - 0.5, si - 0.5), 1, 1, fill=False,
                                             edgecolor="#059669", linewidth=3, zorder=3))

    ax.set_xticks(range(pn))
    ax.set_xticklabels(problem.destinations[:pn], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(pm))
    ax.set_yticklabels(problem.sources[:pm], fontsize=9)
    ax.set_title("Phân bổ hàng hoá", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax.spines[:].set_color("#cbd5e1")
    fig.tight_layout()
    return fig


def plot_delta_heatmap(
    problem: TransportationProblem,
    step: AlgorithmStep,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    pm, pn = problem.m, problem.n
    deltas = step.deltas[:pm, :pn] if step.deltas is not None else np.full((pm, pn), np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    finite = deltas[~np.isnan(deltas)]
    if finite.size:
        max_abs = max(abs(float(finite.min())), abs(float(finite.max())), 1.0)
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        masked = np.ma.array(deltas, mask=np.isnan(deltas))
        im = ax.imshow(masked, cmap="RdYlGn_r", norm=norm, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, label="Delta")
        cbar.ax.tick_params(labelsize=8)
    else:
        ax.imshow(np.zeros((pm, pn)), cmap="Greys", aspect="auto", vmin=0, vmax=1)

    for i in range(pm):
        for j in range(pn):
            value = deltas[i, j]
            if np.isnan(value):
                label = "co so"
                color = "#64748b"
            else:
                label = f"{value:+.0f}"
                color = "white" if abs(value) > 0.6 * (max_abs if finite.size else 1) else "#0f172a"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, fontweight="bold", color=color)

    if step.selected_cell:
        si, sj = step.selected_cell
        if si < pm and sj < pn:
            ax.add_patch(mpatches.Rectangle((sj - 0.5, si - 0.5), 1, 1, fill=False,
                                             edgecolor="#2563eb", linewidth=3, zorder=3))

    ax.set_xticks(range(pn))
    ax.set_xticklabels(problem.destinations[:pn], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(pm))
    ax.set_yticklabels(problem.sources[:pm], fontsize=9)
    ax.set_title("Heatmap Delta MODI", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax.spines[:].set_color("#cbd5e1")
    fig.tight_layout()
    return fig


def build_transportation_tableau(
    problem: TransportationProblem,
    allocation: np.ndarray,
    step: AlgorithmStep | None = None,
) -> pd.DataFrame:
    """
    Creates a textbook-style transportation tableau.
    Each cell contains: [Cost] | Allocation | (Delta if non-basic)
    """
    pm, pn = problem.m, problem.n
    alloc = allocation[:pm, :pn]
    costs = problem.cost[:pm, :pn]
    
    # Delta and potentials for MODI steps
    deltas = step.deltas[:pm, :pn] if (step and step.deltas is not None) else None
    
    cycle_signs = {}
    if step and step.cycle:
        cycle_signs = {(r, c): sign for r, c, sign in step.cycle}

    rows: list[list[str]] = []
    for i in range(pm):
        row: list[str] = []
        for j in range(pn):
            c_val = costs[i, j]
            x_val = alloc[i, j]
            
            cost_str = "∞" if c_val >= BIG_M / 2 else f"{c_val:.0f}"
            
            # Format: "[Cost] Allocation"
            cell_parts = [f"[{cost_str}]"]
            
            if x_val > EPSILON:
                cell_parts.append(f" {x_val:.0f}")
            elif (i, j) in cycle_signs or (step and step.selected_cell == (i, j)):
                # Keep it empty but add sign later
                pass
            else:
                # Basic zero (degeneracy)
                is_basic = False
                if step and step.potentials:
                    # In MODI, basis is defined. We could check basis list but it's not in Step.
                    # As a heuristic, if it's not in basic_cells list if we had one.
                    # For now, just show allocation.
                    pass

            if (i, j) in cycle_signs:
                cell_parts.append(f" ({cycle_signs[(i, j)]})")
            elif step and step.selected_cell == (i, j):
                cell_parts.append(" ⬅")

            if deltas is not None and x_val <= EPSILON and (i, j) not in cycle_signs:
                d_val = deltas[i, j]
                if not np.isnan(d_val):
                    cell_parts.append(f" Δ:{d_val:+.0f}")

            row.append("".join(cell_parts))
            
        row.append(f"{problem.supply[i]:.0f}")
        rows.append(row)

    demand_row = [f"{problem.demand[j]:.0f}" for j in range(pn)] + [""]
    df = pd.DataFrame(
        rows + [demand_row],
        index=problem.sources[:pm] + ["Cầu (Demand)"],
        columns=problem.destinations[:pn] + ["Cung (Supply)"],
    )
    return df
