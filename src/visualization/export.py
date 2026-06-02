from __future__ import annotations
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.result import AlgorithmResult
from src.models.problem import TransportationProblem


def fig_to_bytes(fig: plt.Figure, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.getvalue()


def allocation_to_csv(problem: TransportationProblem, result: AlgorithmResult) -> str:
    pm, pn = problem.m, problem.n
    alloc = result.allocation[:pm, :pn]
    df = pd.DataFrame(alloc, index=problem.sources[:pm], columns=problem.destinations[:pn])
    df.index.name = "Nguon \\ Dich"
    return df.to_csv()


def result_to_json(problem: TransportationProblem, result: AlgorithmResult) -> str:
    pm, pn = problem.m, problem.n
    data = {
        "algorithm": result.algorithm_name,
        "total_cost": result.total_cost,
        "is_optimal": result.is_optimal,
        "sources": problem.sources[:pm],
        "destinations": problem.destinations[:pn],
        "allocation": result.allocation[:pm, :pn].tolist(),
        "warnings": result.warnings,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)
