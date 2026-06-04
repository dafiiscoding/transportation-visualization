from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AlgorithmStep:
    title: str
    description: str
    allocation: np.ndarray
    remaining_supply: np.ndarray
    remaining_demand: np.ndarray
    selected_cell: tuple[int, int] | None = None
    penalty_table: dict | None = None
    row_penalties: dict | None = None
    col_penalties: dict | None = None
    potentials: dict | None = None
    deltas: np.ndarray | None = None
    cycle: list[tuple[int, int, str]] | None = None
    cost: float | None = None
    theta: float | None = None
    leaving_cell: tuple[int, int] | None = None
