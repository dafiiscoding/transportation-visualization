from __future__ import annotations
import numpy as np
from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.models.step import AlgorithmStep
from src.core.cost import total_cost
from src.core.constants import BIG_M


def least_cost_method(problem: TransportationProblem) -> AlgorithmResult:
    m, n = problem.m, problem.n
    supply = problem.supply.astype(float).copy()
    demand = problem.demand.astype(float).copy()
    cost = problem.cost.astype(float)
    allocation = np.zeros((m, n))
    steps: list[AlgorithmStep] = []

    done_rows = set()
    done_cols = set()

    step_num = 0
    while len(done_rows) < m and len(done_cols) < n:
        best = None  # (cost, -potential, row, col)
        for i in range(m):
            if i in done_rows:
                continue
            for j in range(n):
                if j in done_cols:
                    continue
                potential = min(supply[i], demand[j])
                c = cost[i, j]
                key = (c, -potential, i, j)
                if best is None or key < best:
                    best = key

        if best is None:
            break

        c_val, neg_pot, i, j = best
        x = min(supply[i], demand[j])
        allocation[i, j] = x
        supply[i] -= x
        demand[j] -= x

        src = problem.sources[i]
        dst = problem.destinations[j]
        step_num += 1
        desc = (
            f"Ô rẻ nhất hiện tại là {src}→{dst} với chi phí {c_val:.0f}. "
            f"Phân bổ min(cung {src}, cầu {dst}) = min({supply[i]+x:.0f}, {demand[j]+x:.0f}) = {x:.0f}."
        )

        if supply[i] == 0 and demand[j] == 0:
            desc += f" Hàng {src} và cột {dst} đều thỏa mãn."
            done_rows.add(i)
            # eliminate row first, keep col active
            if len(done_rows) < m:
                done_rows.add(i)
            else:
                done_cols.add(j)
        elif supply[i] == 0:
            desc += f" Hàng {src} đã hết cung."
            done_rows.add(i)
        else:
            desc += f" Cột {dst} đã đủ cầu."
            done_cols.add(j)

        steps.append(AlgorithmStep(
            title=f"Bước {step_num}: Phân bổ {x:.0f} vào {src}→{dst} (c={c_val:.0f})",
            description=desc,
            allocation=allocation.copy(),
            remaining_supply=supply.copy(),
            remaining_demand=demand.copy(),
            selected_cell=(i, j),
            cost=total_cost(allocation, cost),
        ))

    tc = total_cost(allocation, cost)
    return AlgorithmResult(
        algorithm_name="Least Cost Method",
        allocation=allocation,
        total_cost=tc,
        is_optimal=None,
        steps=steps,
        warnings=[],
    )
