from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass
class TransportationProblem:
    name: str
    problem_type: Literal["min", "max"]
    sources: list[str]
    destinations: list[str]
    supply: np.ndarray
    demand: np.ndarray
    cost: np.ndarray
    description: str = ""
    insight: str = ""
    key_highlights: list[dict] = field(default_factory=list)
    forbidden: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return len(self.sources)

    @property
    def n(self) -> int:
        return len(self.destinations)

    @property
    def total_supply(self) -> float:
        return float(self.supply.sum())

    @property
    def total_demand(self) -> float:
        return float(self.demand.sum())

    @property
    def is_balanced(self) -> bool:
        return np.isclose(self.total_supply, self.total_demand)
