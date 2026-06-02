from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.models.step import AlgorithmStep
from src.core.constants import BIG_M


def lp_solver(problem: TransportationProblem) -> AlgorithmResult:
    m, n = problem.m, problem.n
    cost = problem.cost.astype(float)

    # Variables: x[i,j] = x[i*n + j]
    c = cost.flatten()

    # Equality constraints: supply and demand
    A_eq = []
    b_eq = []

    # Supply constraints: sum_j x[i,j] = supply[i]
    for i in range(m):
        row = [0.0] * (m * n)
        for j in range(n):
            row[i * n + j] = 1.0
        A_eq.append(row)
        b_eq.append(float(problem.supply[i]))

    # Demand constraints: sum_i x[i,j] = demand[j]
    for j in range(n):
        row = [0.0] * (m * n)
        for i in range(m):
            row[i * n + j] = 1.0
        A_eq.append(row)
        b_eq.append(float(problem.demand[j]))

    bounds = [(0, None)] * (m * n)

    result = linprog(
        c,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return AlgorithmResult(
            algorithm_name="LP Solver",
            allocation=np.zeros((m, n)),
            total_cost=float("inf"),
            is_optimal=False,
            steps=[],
            warnings=[f"LP solver failed: {result.message}"],
        )

    allocation = result.x.reshape(m, n)
    # Round tiny values to zero
    allocation[allocation < 1e-8] = 0.0
    tc = float((allocation * cost).sum())

    step = AlgorithmStep(
        title="LP Optimal Solution",
        description="Giải bài toán vận tải bằng lập trình tuyến tính (HiGHS solver).",
        allocation=allocation.copy(),
        remaining_supply=np.zeros(m),
        remaining_demand=np.zeros(n),
        cost=tc,
    )

    return AlgorithmResult(
        algorithm_name="LP Solver",
        allocation=allocation,
        total_cost=tc,
        is_optimal=True,
        steps=[step],
        warnings=[],
    )
