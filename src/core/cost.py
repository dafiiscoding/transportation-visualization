from __future__ import annotations
import numpy as np


def total_cost(allocation: np.ndarray, cost: np.ndarray) -> float:
    return float((allocation * cost).sum())
