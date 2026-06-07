import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.models.problem import TransportationProblem
from src.core.transform import transform_problem
from src.algorithms.lp_solver import lp_solver
from src.algorithms.sinkhorn import (
    sinkhorn, sinkhorn_sweep, cost_scale, plan_entropy, count_support,
)


def _basic():
    return TransportationProblem(
        "t", "min", ["A1", "A2", "A3"], ["B1", "B2", "B3", "B4"],
        np.array([30.0, 40.0, 50.0]), np.array([20.0, 30.0, 25.0, 45.0]),
        np.array([[8.0, 6.0, 10.0, 9.0], [9.0, 12.0, 13.0, 7.0], [14.0, 9.0, 16.0, 5.0]]),
    )


def test_marginals_respected():
    p = transform_problem(_basic())
    r = sinkhorn(p, epsilon=0.5 * cost_scale(p.cost))
    np.testing.assert_allclose(r.allocation.sum(axis=1), p.supply, atol=1e-4)
    np.testing.assert_allclose(r.allocation.sum(axis=0), p.demand, atol=1e-4)


def test_converges_to_lp_as_eps_shrinks():
    p = transform_problem(_basic())
    opt = lp_solver(p).total_cost
    scale = cost_scale(p.cost)
    big = sinkhorn(p, epsilon=0.5 * scale).total_cost
    small = sinkhorn(p, epsilon=0.02 * scale).total_cost
    # ε nhỏ phải gần tối ưu hơn ε lớn, và không thấp hơn tối ưu (LP là cận dưới).
    assert small >= opt - 1e-6
    assert abs(small - opt) < abs(big - opt)
    assert abs(small - opt) / opt < 0.02


def test_larger_eps_blurrier_plan():
    p = transform_problem(_basic())
    scale = cost_scale(p.cost)
    blur = sinkhorn(p, epsilon=0.5 * scale).allocation
    sharp = sinkhorn(p, epsilon=0.02 * scale).allocation
    # ε lớn → entropy cao hơn và nhiều ô mang khối lượng hơn (nghiệm "mờ").
    assert plan_entropy(blur) > plan_entropy(sharp)
    assert count_support(blur) >= count_support(sharp)


def test_sweep_monotone_gap():
    p = transform_problem(_basic())
    opt = lp_solver(p).total_cost
    scale = cost_scale(p.cost)
    rows = sinkhorn_sweep(p, [0.5 * scale, 0.1 * scale, 0.02 * scale], optimum=opt)
    gaps = [r["gap"] for r in rows]
    assert all(g >= -1e-6 for g in gaps)
    assert gaps[0] > gaps[-1]  # ε giảm → gap giảm


def test_real_objective_convergence_min_and_max():
    """Logic của panel (C1): quy về mục tiêu THẬT bằng compute_real_cost, hội tụ
    về tối ưu LP cho cả bài min lẫn max; |khoảng cách| luôn ≥ 0 và giảm dần."""
    from src.core.transform import compute_real_cost

    def run(orig):
        t = transform_problem(orig)
        opt = compute_real_cost(lp_solver(t).allocation, orig.cost, orig.forbidden)
        scale = cost_scale(t.cost)
        sweep = sinkhorn_sweep(t, [0.5 * scale, 0.1 * scale, 0.02 * scale])
        gaps = [abs(compute_real_cost(s["allocation"], orig.cost, orig.forbidden) - opt)
                for s in sweep]
        return opt, gaps

    # min
    _, gmin = run(_basic())
    assert all(g >= -1e-9 for g in gmin) and gmin[0] > gmin[-1]
    # max
    pmax = TransportationProblem(
        "m", "max", ["A1", "A2"], ["B1", "B2", "B3"],
        np.array([40.0, 60.0]), np.array([30.0, 30.0, 40.0]),
        np.array([[12.0, 4.0, 7.0], [3.0, 9.0, 11.0]]),
    )
    opt, gmax = run(pmax)
    assert all(g >= -1e-9 for g in gmax) and gmax[0] > gmax[-1]


def test_viz_functions_return_figures():
    import matplotlib
    matplotlib.use("Agg")
    from src.visualization.sinkhorn_viz import plot_convergence, plot_plan_comparison
    p = transform_problem(_basic())
    opt = lp_solver(p).total_cost
    scale = cost_scale(p.cost)
    rows = sinkhorn_sweep(p, [0.5 * scale, 0.05 * scale], optimum=opt)
    assert plot_convergence(rows, opt, ylabel="Chi phí") is not None
    fig = plot_plan_comparison(
        p, lp_solver(p).allocation, "LP",
        rows[0]["allocation"], rows[0]["eps"],
        rows[-1]["allocation"], rows[-1]["eps"],
    )
    assert fig is not None
