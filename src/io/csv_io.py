from __future__ import annotations
import io
import numpy as np
import pandas as pd
from src.models.problem import TransportationProblem


def parse_csv(file_content: bytes | str) -> TransportationProblem:
    """
    CSV format:
    Row 0: header row — first cell empty, then destination names
    Rows 1..m: source name, costs...
    Row m+1: "Supply", supply values...
    Last row: "Demand", demand values...
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")
    df = pd.read_csv(io.StringIO(file_content), header=None)
    destinations = list(df.iloc[0, 1:].astype(str))
    sources = []
    cost_rows = []
    supply = []
    demand = None

    for _, row in df.iloc[1:].iterrows():
        label = str(row.iloc[0]).strip()
        vals = list(row.iloc[1:])
        if label.lower() == "demand":
            demand = [float(v) for v in vals if str(v).strip() != ""]
        elif label.lower() == "supply":
            supply = [float(v) for v in vals if str(v).strip() != ""]
        else:
            sources.append(label)
            cost_rows.append([float(v) for v in vals])

    if demand is None:
        raise ValueError("CSV missing 'Demand' row")

    return TransportationProblem(
        name="CSV Import",
        problem_type="min",
        sources=sources,
        destinations=destinations,
        supply=np.array(supply, dtype=float),
        demand=np.array(demand, dtype=float),
        cost=np.array(cost_rows, dtype=float),
    )
