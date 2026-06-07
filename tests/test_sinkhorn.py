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
