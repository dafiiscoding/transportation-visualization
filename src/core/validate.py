from __future__ import annotations
import numpy as np
from src.models.problem import TransportationProblem


def validate_problem(p: TransportationProblem) -> list[str]:
    """Return list of error strings; empty list means valid."""
    errors: list[str] = []

    if p.m < 1 or p.n < 1:
        errors.append("Problem must have at least 1 source and 1 destination.")

    if np.any(p.supply < 0):
        errors.append("All supply values must be non-negative.")

    if np.any(p.demand < 0):
        errors.append("All demand values must be non-negative.")

    if p.cost.shape != (p.m, p.n):
        errors.append(f"Cost matrix shape {p.cost.shape} does not match ({p.m}, {p.n}).")

    if p.supply.sum() == 0:
        errors.append("Total supply cannot be zero.")

    if p.demand.sum() == 0:
        errors.append("Total demand cannot be zero.")

    if p.forbidden is not None and p.forbidden.shape != (p.m, p.n):
        errors.append("Forbidden matrix shape must match cost matrix shape.")

    return errors


def get_warnings(p: TransportationProblem) -> list[str]:
    warnings: list[str] = []

    if not p.is_balanced:
        diff = p.total_supply - p.total_demand
        if diff > 0:
            warnings.append(
                f"Unbalanced: supply exceeds demand by {diff:.0f}. "
                "A dummy destination will be added."
            )
        else:
            warnings.append(
                f"Unbalanced: demand exceeds supply by {-diff:.0f}. "
                "A dummy source will be added."
            )

    if p.forbidden is not None and p.forbidden.any():
        count = int(p.forbidden.sum())
        warnings.append(f"{count} forbidden cell(s) detected. Big-M method will be applied.")

    if p.problem_type == "max":
        warnings.append("Maximization problem detected. Will transform to minimization.")

    return warnings
