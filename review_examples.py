from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from src.io.examples import list_examples, load_example, example_label
from src.core.transform import transform_problem, compute_real_cost
from src.algorithms.lp_solver import lp_solver
from src.algorithms.northwest import northwest_corner
from src.algorithms.vogel import vogel
from src.algorithms.least_cost import least_cost_method
from src.algorithms.modi import modi

def multiopt(t, lp):
    """Đếm ô ngoài cơ sở có Δ≈0 ở nghiệm LP (dấu hiệu đa nghiệm) — bằng MODI trên nghiệm LP."""
    try:
        r = modi(t, lp)
        if not r.steps: return "?"
        d = r.steps[-1].deltas
        if d is None: return "?"
        a = r.allocation
        cnt = 0
        for i in range(t.m):
            for j in range(t.n):
                if a[i,j] <= 1e-6 and d[i,j] is not None and not np.isnan(d[i,j]) and abs(d[i,j])<1e-6:
                    cnt += 1
        return cnt
    except Exception as e:
        return f"err"

for name in list_examples():
    p = load_example(name)
    t = transform_problem(p)
    ss, sd = p.supply.sum(), p.demand.sum()
    bal = "cân bằng" if abs(ss-sd)<1e-9 else f"LỆCH({ss:.0f}/{sd:.0f})"
    lp = lp_solver(t)
    real = compute_real_cost(lp.allocation, p.cost, p.forbidden)
    fb = "ô cấm" if p.forbidden is not None else "-"
    mo = multiopt(t, lp)
    print(f"{name:32} {p.m}x{p.n}  {p.problem_type:3}  {bal:16} {fb:6} "
          f"LP_real={real:>8.0f}  status={'OK' if lp.is_optimal else 'FAIL':4}  Δ0_ngoài_cs={mo}")
