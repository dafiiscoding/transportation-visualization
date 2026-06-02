from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.models.step import AlgorithmStep


def solve_assignment(problem: TransportationProblem) -> AlgorithmResult:
    cost = problem.cost.astype(float)
    row_ind, col_ind = linear_sum_assignment(cost)
    allocation = np.zeros_like(cost)
    for r, c in zip(row_ind, col_ind):
        allocation[r, c] = 1.0
    tc = float(cost[row_ind, col_ind].sum())

    pairs = ", ".join(
        f"{problem.sources[r]}→{problem.destinations[c]} (c={cost[r,c]:.0f})"
        for r, c in zip(row_ind, col_ind)
    )
    step = AlgorithmStep(
        title="Assignment Optimal Solution",
        description=f"Gán tối ưu: {pairs}. Tổng chi phí = {tc:.0f}.",
        allocation=allocation,
        remaining_supply=np.zeros(problem.m),
        remaining_demand=np.zeros(problem.n),
        cost=tc,
    )
    return AlgorithmResult(
        algorithm_name="Assignment (Hungarian)",
        allocation=allocation,
        total_cost=tc,
        is_optimal=True,
        steps=[step],
        warnings=[],
    )
