from __future__ import annotations
import matplotlib.pyplot as plt
from src.core.transform import compute_real_cost
from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult

ALGO_COLORS = {
    "Góc Tây Bắc (NW)": "#94a3b8",
    "Chi phí nhỏ nhất (LCM)": "#60a5fa",
    "Xấp xỉ Vogel (VAM)": "#34d399",
    "MODI": "#2563eb",
    "LP": "#dc2626",
    "Assignment": "#f59e0b",
}


def objective_label(problem: TransportationProblem) -> str:
    return "Lợi nhuận" if problem.problem_type == "max" else "Chi phí"


def objective_value(problem: TransportationProblem, result: AlgorithmResult) -> float:
    return compute_real_cost(result.allocation, problem.cost, problem.forbidden)


def allocation_objective_value(problem: TransportationProblem, allocation) -> float:
    return compute_real_cost(allocation, problem.cost, problem.forbidden)


def gap_to_optimum_text(problem: TransportationProblem, value: float, optimum: float | None) -> str:
    if optimum is None or optimum == 0:
        return "Chưa có mốc LP"

    if problem.problem_type == "max":
        gap = optimum - value
    else:
        gap = value - optimum

    if abs(gap) < 1e-7:
        return "0 - đạt mốc tối ưu"

    pct = gap / abs(optimum) * 100
    return f"{gap:,.0f} ({pct:.2f}%)"


def improvement_text(problem: TransportationProblem, before: float, after: float) -> str:
    improvement = after - before if problem.problem_type == "max" else before - after
    if abs(improvement) < 1e-7:
        return "Không đổi"
    return f"{improvement:,.0f}"


def plot_phase_comparison(
    problem: TransportationProblem,
    initial_results: dict[str, AlgorithmResult],
    modi_results: dict[str, AlgorithmResult],
    lp_result: AlgorithmResult | None,
    figsize: tuple[float, float] = (10, 4.5),
) -> plt.Figure:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    for name, result in initial_results.items():
        labels.append(name.replace(" ", "\n", 1))
        values.append(objective_value(problem, result))
        colors.append(ALGO_COLORS.get(name, "#64748b"))

    for name, result in modi_results.items():
        labels.append(f"MODI từ\n{name.split('(')[-1].replace(')', '')}")
        values.append(objective_value(problem, result))
        colors.append(ALGO_COLORS["MODI"])

    lp_value = objective_value(problem, lp_result) if lp_result is not None else None

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    if values:
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5, zorder=3, width=0.62)
        max_value = max(values + ([lp_value] if lp_value is not None else []))
        min_value = min(values + ([lp_value] if lp_value is not None else []))
        span = max(max_value - min_value, 1.0)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02 * span,
                f"{value:,.0f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e293b",
            )

    if lp_value is not None:
        ax.axhline(
            lp_value, color=ALGO_COLORS["LP"], linestyle="--", linewidth=2,
            label=f"LP tối ưu: {lp_value:,.0f}", zorder=4,
        )
        ax.legend(framealpha=0.9)

    label = objective_label(problem)
    ax.set_ylabel(label, fontsize=10, color="#475569")
    ax.set_title(f"{label} và khoảng cách tới mốc tối ưu", fontsize=13, fontweight="bold", color="#1e293b", pad=12)
    ax.tick_params(colors="#475569")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.yaxis.grid(True, color="#e2e8f0", zorder=0)
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, ha="right", fontsize=8.5)
    fig.tight_layout()
    return fig
