from __future__ import annotations
import os, sys, time, gc
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from src.models.problem import TransportationProblem
from src.algorithms.northwest import northwest_corner
from src.algorithms.modi import modi
from src.algorithms.lp_solver import lp_solver

rng = np.random.default_rng(20237354)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "slide", "images"))
os.makedirs(OUT, exist_ok=True)


def make_balanced(m, n):
    """Sinh bài toán cân bằng ngẫu nhiên cỡ m x n."""
    supply = rng.integers(20, 120, size=m).astype(float)
    total = supply.sum()
    w = rng.random(n)
    demand = np.floor(w / w.sum() * total).astype(float)
    demand[-1] += total - demand.sum()        # chỉnh khớp tổng
    cost = rng.integers(1, 40, size=(m, n)).astype(float)
    return TransportationProblem(
        name=f"rand {m}x{n}", problem_type="min",
        sources=[f"A{i}" for i in range(m)],
        destinations=[f"B{j}" for j in range(n)],
        supply=supply, demand=demand, cost=cost,
    )


def time_call(fn, repeat):
    best = float("inf")
    val = None
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        val = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best * 1000.0, val   # ms


# (m, n, rep_lp, run_modi)
SIZES = [
    (3, 4, 7, True), (5, 5, 7, True), (6, 6, 7, True), (8, 8, 5, True),
    (10, 10, 5, True), (12, 15, 5, True), (15, 20, 3, True), (20, 20, 3, True),
    (30, 30, 3, False), (50, 50, 3, False), (80, 80, 2, False), (100, 120, 2, False),
]

rows = []
print(f"{'cỡ':>9} {'biến':>7} {'MODI(ms)':>12} {'LP(ms)':>10} {'khớp':>6}")
for m, n, rep, run_modi in SIZES:
    p = make_balanced(m, n)
    lp_ms, lp_res = time_call(lambda: lp_solver(p), rep)

    modi_ms, ok = None, ""
    if run_modi:
        def _hand():
            init = northwest_corner(p)
            return modi(p, init)
        modi_ms, modi_res = time_call(_hand, 1)
        if modi_res.total_cost < float("inf") and lp_res.total_cost < float("inf"):
            ok = "✓" if abs(modi_res.total_cost - lp_res.total_cost) < 1e-6 else "≈"

    rows.append((m, n, m * n, modi_ms, lp_ms, ok))
    print(f"{m:>4}x{n:<4} {m*n:>7} "
          f"{(f'{modi_ms:10.1f}' if modi_ms is not None else '         —'):>12} "
          f"{lp_ms:10.2f} {ok:>6}")

# ---- Biểu đồ ----
xs = [r[2] for r in rows]
lp = [r[4] for r in rows]
modi_x = [r[2] for r in rows if r[3] is not None]
modi_y = [r[3] for r in rows if r[3] is not None]

fig, ax = plt.subplots(figsize=(9, 4.6))
fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
ax.plot(modi_x, modi_y, "o-", color="#dc2626", lw=2, ms=6, label="NW + MODI (thủ công, code hoá)")
ax.plot(xs, lp, "s-", color="#2563eb", lw=2, ms=6, label="LP — HiGHS (solver)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Số biến  $m\\times n$", fontsize=11, color="#334155")
ax.set_ylabel("Thời gian chạy (ms, log)", fontsize=11, color="#334155")
ax.set_title("Thời gian giải thực tế: MODI vs LP (HiGHS)", fontsize=13, fontweight="bold", color="#1e293b", pad=10)
ax.grid(True, which="both", color="#e2e8f0")
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.95)
ax.spines[["top", "right"]].set_color("#cbd5e1")
fig.tight_layout()
path = os.path.join(OUT, "viz_benchmark.png")
fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
print("saved", path)

# Lưu bảng ra CSV để tham khảo
import csv
csvp = os.path.join(os.path.dirname(__file__), "bench_results.csv")
with open(csvp, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["m", "n", "vars", "modi_ms", "lp_ms", "match"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], f"{r[3]:.1f}" if r[3] is not None else "", f"{r[4]:.2f}", r[5]])
print("saved", csvp)
