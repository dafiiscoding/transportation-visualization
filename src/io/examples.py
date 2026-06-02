from __future__ import annotations
import json
import os
import numpy as np
from src.models.problem import TransportationProblem

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "examples")

EXAMPLE_LABELS = {
    "01_intro_battle": "01. NW vs VAM: Sức mạnh của xấp xỉ",
    "02_greedy_trap": "02. LCM: Cái bẫy của sự tham lam",
    "03_initial_degeneracy": "03. Suy biến: Điểm thắt nút toán học",
    "04_multiple_optimal": "04. Đa nghiệm: Lựa chọn của nhà quản trị",
    "05_broken_route": "05. Ràng buộc: Xử lý tuyến đường hỏng",
    "06_inventory_insight": "06. Cung > Cầu: Chiến lược tồn kho",
    "07_penalty_insight": "07. Cầu > Cung: Ưu tiên khách hàng",
    "08_modi_climb": "08. MODI: Hành trình tối ưu hóa",
    "09_assignment": "09. Phân công: Khi mỗi người một việc",
    "10_vietnam_logistics": "10. Thực tế: Logistics 3 miền Việt Nam",
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
