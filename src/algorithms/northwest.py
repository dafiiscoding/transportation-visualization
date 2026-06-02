from __future__ import annotations
import numpy as np
from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.models.step import AlgorithmStep
from src.core.cost import total_cost


def northwest_corner(problem: TransportationProblem) -> AlgorithmResult:
    m, n = problem.m, problem.n
    supply = problem.supply.astype(float).copy()
    demand = problem.demand.astype(float).copy()
    cost = problem.cost.astype(float)
    allocation = np.zeros((m, n))
    steps: list[AlgorithmStep] = []

    i, j = 0, 0
    while i < m and j < n:
        si = supply[i]
        dj = demand[j]
        x = min(si, dj)
        allocation[i, j] = x
        supply[i] -= x
        demand[j] -= x

        src = problem.sources[i]
        dst = problem.destinations[j]
        desc = (
            f"Xét ô phía Tây Bắc nhất còn lại là {src}→{dst}. "
            f"Lượng có thể phân bổ là min(cung {src}, cầu {dst}) = min({si:.0f}, {dj:.0f}) = {x:.0f}."
        )

        sel = (i, j)
        if supply[i] == 0 and demand[j] == 0:
            desc += f" Hàng {src} và cột {dst} đều thỏa mãn."
            i += 1
            j += 1
        elif supply[i] == 0:
            desc += f" Hàng {src} đã hết cung, chuyển xuống hàng tiếp theo."
            i += 1
        else:
            desc += f" Cột {dst} đã đủ cầu, chuyển sang cột tiếp theo."
            j += 1

        steps.append(AlgorithmStep(
            title=f"Bước {len(steps)+1}: Phân bổ {x:.0f} vào {src}→{dst}",
            description=desc,
            allocation=allocation.copy(),
            remaining_supply=supply.copy(),
            remaining_demand=demand.copy(),
            selected_cell=sel,
            cost=total_cost(allocation, cost),
        ))

    tc = total_cost(allocation, cost)
    return AlgorithmResult(
        algorithm_name="Northwest Corner",
        allocation=allocation,
        total_cost=tc,
        is_optimal=None,
        steps=steps,
        warnings=[],
    )
