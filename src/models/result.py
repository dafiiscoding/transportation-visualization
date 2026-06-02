from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .step import AlgorithmStep
from .problem import TransportationProblem


@dataclass
class AlgorithmResult:
    algorithm_name: str
    allocation: np.ndarray
    total_cost: float
    is_optimal: bool | None
    steps: list[AlgorithmStep]
    warnings: list[str]
    transformed_problem: TransportationProblem | None = None
