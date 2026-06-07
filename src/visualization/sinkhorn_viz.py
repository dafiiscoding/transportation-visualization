"""Biểu đồ cho demo MODI vs Sinkhorn (Optimal Transport)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from src.models.problem import TransportationProblem


def plot_convergence(
    rows: list[dict], optimum: float, ylabel: str = "Tổng chi phí vận chuyển",
) -> plt.Figure:
    """Giá trị mục tiêu Sinkhorn theo ε (giảm dần) hội tụ về mốc tối ưu LP/MODI."""
    eps = [r["eps"] for r in rows]
    cost = [r["cost"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    ax.plot(eps, cost, "o-", color="#7c3aed", linewidth=2, markersize=7,
            zorder=3, label="Sinkhorn")
    ax.axhline(optimum, color="#dc2626", linestyle="--", linewidth=2,
               zorder=2, label=f"Tối ưu LP/MODI: {optimum:,.0f}")

    for e, c in zip(eps, cost):
        ax.annotate(f"{c:,.0f}", (e, c), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#1e293b")

    ax.set_xscale("log")
    ax.invert_xaxis()  # ε giảm dần từ trái sang phải → tiến tới tối ưu
    ax.set_xlabel("ε (chính quy hoá entropy) — giảm dần →", fontsize=10, color="#475569")
    ax.set_ylabel(ylabel, fontsize=10, color="#475569")
    ax.set_title("ε → 0: Sinkhorn hội tụ về nghiệm tối ưu", fontsize=12,
                 fontweight="bold", color="#1e293b", pad=10)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, color="#e2e8f0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#475569")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    fig.tight_layout()
    return fig


def _draw_plan(ax, problem: TransportationProblem, P: np.ndarray, title: str):
    masked = np.ma.array(P, mask=(P <= 1e-9))
    ax.imshow(masked, cmap="Blues", aspect="auto",
              vmin=0, vmax=float(P.max()) if P.max() > 0 else 1)
    for i in range(problem.m):
        for j in range(problem.n):
            v = P[i, j]
            if v > 1e-6:
                shade = "white" if v > 0.6 * P.max() else "#1e293b"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color=shade, fontweight="bold")
    ax.set_xticks(range(problem.n))
    ax.set_xticklabels(problem.destinations, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(problem.m))
    ax.set_yticklabels(problem.sources, fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1e293b", pad=8)
    ax.spines[:].set_color("#cbd5e1")


def plot_plan_comparison(
    problem: TransportationProblem,
    sparse_plan: np.ndarray,
    sparse_label: str,
    blur_plan: np.ndarray,
    blur_eps: float,
    sharp_plan: np.ndarray,
    sharp_eps: float,
) -> plt.Figure:
    """3 bảng cạnh nhau: nghiệm thưa (MODI/LP) ↔ Sinkhorn ε lớn (mờ) ↔ ε nhỏ (sắc)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    fig.patch.set_facecolor("#f8fafc")
    _draw_plan(axes[0], problem, sparse_plan, f"{sparse_label}\n(thưa — nghiệm góc)")
    _draw_plan(axes[1], problem, blur_plan, f"Sinkhorn ε lớn ({blur_eps:.3g})\n(mờ — entropy cao)")
    _draw_plan(axes[2], problem, sharp_plan, f"Sinkhorn ε nhỏ ({sharp_eps:.3g})\n(sắc dần về nghiệm góc)")
    fig.tight_layout()
    return fig
