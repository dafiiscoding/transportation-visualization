from __future__ import annotations
import numpy as np
from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.models.step import AlgorithmStep
from src.core.cost import total_cost
from src.core.constants import BIG_M


def _row_penalty(cost_row: np.ndarray, active_cols: list[int]) -> float:
    vals = sorted(cost_row[c] for c in active_cols)
    if len(vals) < 2:
        return 0.0
    return vals[1] - vals[0]


def _col_penalty(cost: np.ndarray, col: int, active_rows: list[int]) -> float:
    vals = sorted(cost[r, col] for r in active_rows)
    if len(vals) < 2:
        return 0.0
    return vals[1] - vals[0]


def vogel(problem: TransportationProblem) -> AlgorithmResult:
    m, n = problem.m, problem.n
    supply = problem.supply.astype(float).copy()
    demand = problem.demand.astype(float).copy()
    cost = problem.cost.astype(float)
    allocation = np.zeros((m, n))
    steps: list[AlgorithmStep] = []

    active_rows = list(range(m))
    active_cols = list(range(n))

    step_num = 0
    while active_rows and active_cols:
        row_penalties = {i: _row_penalty(cost[i], active_cols) for i in active_rows}
        col_penalties = {j: _col_penalty(cost, j, active_rows) for j in active_cols}

        penalty_table = {
            "row": {problem.sources[i]: row_penalties[i] for i in active_rows},
            "col": {problem.destinations[j]: col_penalties[j] for j in active_cols},
        }

        # Find max penalty (row or col)
        max_row_pen = max(row_penalties.values()) if row_penalties else -1
        max_col_pen = max(col_penalties.values()) if col_penalties else -1

        if max_row_pen > max_col_pen:
            # pick row with max penalty (tie: smallest index)
            best_i = min(
                (i for i in active_rows if row_penalties[i] == max_row_pen),
                key=lambda x: x,
            )
            # pick cheapest col in that row (tie: smallest potential then col index)
            min_c = min(cost[best_i, j] for j in active_cols)
            best_j = min(
                (j for j in active_cols if cost[best_i, j] == min_c),
                key=lambda jj: (-min(supply[best_i], demand[jj]), jj),
            )
            kind = f"hàng {problem.sources[best_i]}"
            pen_val = max_row_pen
        else:
            # pick col with max penalty (tie: smallest index)
            best_j = min(
                (j for j in active_cols if col_penalties[j] == max_col_pen),
                key=lambda x: x,
            )
            # pick cheapest row in that col
            min_c = min(cost[i, best_j] for i in active_rows)
            best_i = min(
                (i for i in active_rows if cost[i, best_j] == min_c),
                key=lambda ii: (-min(supply[ii], demand[best_j]), ii),
            )
            kind = f"cột {problem.destinations[best_j]}"
            pen_val = max_col_pen

        i, j = best_i, best_j
        x = min(supply[i], demand[j])
        si, dj = supply[i], demand[j]
        allocation[i, j] = x
        supply[i] -= x
        demand[j] -= x

        src = problem.sources[i]
        dst = problem.destinations[j]
        step_num += 1
        desc = (
            f"Penalty lớn nhất hiện tại nằm ở {kind} với penalty={pen_val:.0f}. "
            f"Trong {kind}, ô rẻ nhất là {src}→{dst} với chi phí {cost[i,j]:.0f}. "
            f"Phân bổ min({si:.0f}, {dj:.0f}) = {x:.0f} vào ô này."
        )

        if supply[i] == 0 and demand[j] == 0:
            desc += f" Hàng {src} và cột {dst} đều thỏa mãn."
            active_rows.remove(i)
            if active_rows:
                pass  # keep col active only if rows remain — already removed row
            else:
                active_cols.remove(j)
        elif supply[i] == 0:
            desc += f" Hàng {src} đã hết cung."
            active_rows.remove(i)
        else:
            desc += f" Cột {dst} đã đủ cầu."
            active_cols.remove(j)

        steps.append(AlgorithmStep(
            title=f"Bước {step_num}: Phân bổ {x:.0f} vào {src}→{dst}",
            description=desc,
            allocation=allocation.copy(),
            remaining_supply=supply.copy(),
            remaining_demand=demand.copy(),
            selected_cell=(i, j),
            penalty_table=penalty_table,
            cost=total_cost(allocation, cost),
        ))

    tc = total_cost(allocation, cost)
    return AlgorithmResult(
        algorithm_name="Vogel's Approximation",
        allocation=allocation,
        total_cost=tc,
        is_optimal=None,
        steps=steps,
        warnings=[],
    )
