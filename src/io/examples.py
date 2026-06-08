from __future__ import annotations
import json
import os
import numpy as np
from src.models.problem import TransportationProblem

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "examples")

EXAMPLE_LABELS = {
    "01_co_ban": "01. Ví dụ cơ bản 3×4 (xuyên suốt)",
    "02_so_sanh_khoi_tao": "02. So sánh khởi tạo: NW / LCM / Vogel",
    "03_da_nghiem": "03. Bài toán đa nghiệm",
    "04_cung_cau_lech": "04. Cung khác cầu (thêm điểm giả)",
    "05_o_cam": "05. Ô cấm (gán chi phí M)",
    "06_cuc_dai": "06. Bài toán cực đại (MAX)",
    "07_phan_viec": "07. Bài toán phân việc",
    "08_logistics": "08. Logistics 5×12 (Z* = 120 000)",
}


def list_examples() -> list[str]:
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    found = {f[:-5] for f in os.listdir(EXAMPLES_DIR) if f.endswith(".json")}
    # Sort by the prefix number if possible
    def sort_key(name):
        parts = name.split('_')
        if parts[0].isdigit():
            return int(parts[0])
        return 999
    
    return sorted(list(found), key=sort_key)


def example_label(name: str) -> str:
    return EXAMPLE_LABELS.get(name, name)


def load_example(name: str) -> TransportationProblem:
    path = os.path.join(EXAMPLES_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return _parse_problem(data)


def _parse_problem(data: dict) -> TransportationProblem:
    cost = np.array(data["cost"], dtype=float)
    forbidden = None
    if "forbidden" in data and data["forbidden"]:
        fb = np.array(data["forbidden"], dtype=bool)
        if fb.any():
            forbidden = fb
            cost[forbidden] = 1e6

    return TransportationProblem(
        name=data["name"],
        problem_type=data.get("problem_type", "min"),
        sources=data["sources"],
        destinations=data["destinations"],
        supply=np.array(data["supply"], dtype=float),
        demand=np.array(data["demand"], dtype=float),
        cost=cost,
        description=data.get("description", ""),
        insight=data.get("insight", ""),
        key_highlights=data.get("key_highlights", []),
        forbidden=forbidden,
        metadata=data.get("metadata", {}),
    )
