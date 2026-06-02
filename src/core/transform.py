from __future__ import annotations
import numpy as np
from src.models.problem import TransportationProblem
from src.core.constants import BIG_M


def transform_problem(p: TransportationProblem) -> TransportationProblem:
    """Apply all transforms: max→min, forbidden→Big-M, unbalanced→dummy."""
    cost = p.cost.astype(float).copy()
    supply = p.supply.astype(float).copy()
    demand = p.demand.astype(float).copy()
    sources = list(p.sources)
    destinations = list(p.destinations)
    forbidden = p.forbidden.copy() if p.forbidden is not None else np.zeros((p.m, p.n), dtype=bool)

    # Max → Min
    if p.problem_type == "max":
        c_max = cost.max()
        cost = c_max - cost

    # Forbidden → Big-M
    cost[forbidden] = BIG_M

    # Unbalanced → dummy
    total_s = supply.sum()
    total_d = demand.sum()
    if not np.isclose(total_s, total_d):
        if total_s > total_d:
            diff = total_s - total_d
            destinations.append("Dummy")
            demand = np.append(demand, diff)
            new_col = np.zeros((len(sources), 1))
            cost = np.hstack([cost, new_col])
            forbidden = np.hstack([forbidden, np.zeros((len(sources), 1), dtype=bool)])
        else:
            diff = total_d - total_s
            sources.append("Dummy")
            supply = np.append(supply, diff)
            new_row = np.zeros((1, len(destinations)))
            cost = np.vstack([cost, new_row])
            forbidden = np.vstack([forbidden, np.zeros((1, len(destinations)), dtype=bool)])

    return TransportationProblem(
        name=p.name,
        problem_type="min",
        sources=sources,
        destinations=destinations,
        supply=supply,
        demand=demand,
        cost=cost,
        forbidden=forbidden if forbidden.any() else None,
        metadata=p.metadata,
    )


def compute_real_cost(
    allocation: np.ndarray,
    original_cost: np.ndarray,
    forbidden: np.ndarray | None = None,
) -> float:
    """Compute cost ignoring dummy rows/cols and Big-M cells."""
    m_orig, n_orig = original_cost.shape
    alloc_real = allocation[:m_orig, :n_orig]
    cost_real = original_cost.copy().astype(float)
    if forbidden is not None:
        cost_real[forbidden] = 0.0
    return float((alloc_real * cost_real).sum())
