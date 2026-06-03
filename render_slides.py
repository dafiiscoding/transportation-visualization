from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.models.problem import TransportationProblem
from src.core.transform import transform_problem
from src.algorithms.northwest import northwest_corner
from src.algorithms.least_cost import least_cost_method
from src.algorithms.vogel import vogel
from src.algorithms.modi import modi
from src.algorithms.lp_solver import lp_solver
from src.visualization.comparison import plot_phase_comparison
from src.visualization.heatmap import plot_allocation_heatmap, plot_cost_heatmap
from src.visualization.network import plot_network

OUT = os.path.join(os.path.dirname(__file__), "..", "slide", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

INITIAL = {
    "Góc Tây Bắc (NW)": northwest_corner,
    "Chi phí nhỏ nhất (LCM)": least_cost_method,
    "Xấp xỉ Vogel (VAM)": vogel,
}


def save(fig, name, dpi=150):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("saved", path)


def build_3x4():
    return TransportationProblem(
        name="Ví dụ 3x4",
        problem_type="min",
        sources=["A1", "A2", "A3"],
        destinations=["B1", "B2", "B3", "B4"],
        supply=np.array([50, 70, 80], dtype=float),
        demand=np.array([60, 30, 40, 70], dtype=float),
        cost=np.array([[2, 4, 5, 1], [3, 6, 4, 8], [1, 2, 5, 3]], dtype=float),
    )


def solve(problem, do_modi=True):
    t = transform_problem(problem)
    initial, modi_res = {}, {}
    for name, fn in INITIAL.items():
        r = fn(t)
        r.algorithm_name = name
        initial[name] = r
        if do_modi:
            o = modi(t, r)
            o.algorithm_name = f"MODI từ {name}"
            modi_res[name] = o
    lp = lp_solver(t)
    lp.algorithm_name = "LP tối ưu"
    return t, initial, modi_res, lp


def main():
    p = build_3x4()
    t, initial, modi_res, lp = solve(p, do_modi=True)
    print("LP value:", lp.allocation, sep="\n")

    # 1) Bar chart so sánh khởi tạo vs MODI vs LP
    fig = plot_phase_comparison(t, initial, modi_res, lp, figsize=(10, 4.2))
    save(fig, "viz_comparison.png")

    # 2) Allocation heatmap (nghiệm tối ưu LP)
    fig = plot_allocation_heatmap(t, lp.allocation, figsize=(8, 3.6))
    save(fig, "viz_alloc.png")

    # 3) Network graph nghiệm tối ưu 3x4 (bỏ nhãn cho gọn)
    fig = plot_network(t, lp, figsize=(9, 4.4), show_labels=False)
    save(fig, "viz_network.png")

    # 4) Cost heatmap 3x4
    fig = plot_cost_heatmap(t, figsize=(8, 3.4))
    save(fig, "viz_cost.png")

    # 5) Logistics network (nếu có ví dụ 5x12)
    try:
        from src.io.examples import list_examples, load_example
        cand = [e for e in list_examples() if "logistic" in e.lower()]
        if cand:
            lp_problem = load_example(cand[0])
            tl = transform_problem(lp_problem)
            lpl = lp_solver(tl); lpl.algorithm_name = "LP tối ưu"
            big = (tl.m + tl.n) >= 12
            fig = plot_network(tl, lpl, figsize=(13, 7), show_labels=not big)
            save(fig, "viz_network_logistics.png")
            print("logistics size:", tl.m, "x", tl.n)
    except Exception as e:
        print("logistics skip:", e)


if __name__ == "__main__":
    main()
