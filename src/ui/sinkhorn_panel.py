"""Panel demo: MODI/LP (vận tải cổ điển) vs Sinkhorn (Optimal Transport).

Minh hoạ cây cầu giữa bài toán vận tải trong môn học và Optimal Transport hiện
đại (chương 9 slide): cùng một bài, Sinkhorn giải bản *chính quy hoá entropy* và
  - khi ε → 0 thì chi phí hội tụ về đúng nghiệm tối ưu LP/MODI;
  - khi ε lớn thì nghiệm "mờ" (entropy cao) thay vì thưa như nghiệm góc của MODI.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.core.constants import SINKHORN_EPS_FACTORS
from src.core.transform import compute_real_cost
from src.algorithms.sinkhorn import (
    sinkhorn, sinkhorn_sweep, cost_scale, plan_entropy, count_support,
)
from src.visualization.comparison import objective_label
from src.visualization.sinkhorn_viz import plot_convergence, plot_plan_comparison


def render_sinkhorn_panel(
    problem: TransportationProblem,
    transformed: TransportationProblem,
    lp_result: AlgorithmResult | None,
    modi_results: dict[str, AlgorithmResult],
) -> None:
    st.subheader("🌀 MODI vs Sinkhorn — cầu nối tới Optimal Transport")
    st.caption(
        "Sinkhorn giải bản *chính quy hoá entropy* của chính bài toán vận tải này. "
        "Xem nó hội tụ về nghiệm MODI/LP khi ε → 0, và 'mờ' đi khi ε lớn."
    )

    # Cần một nghiệm thưa (LP/MODI) làm mốc đối chiếu.
    sparse_plan = None
    sparse_label = ""
    if lp_result is not None and lp_result.is_optimal:
        sparse_plan = lp_result.allocation
        sparse_label = "LP tối ưu"
    elif modi_results:
        name, r = next(iter(modi_results.items()))
        sparse_plan = r.allocation
        sparse_label = f"MODI ({name})"

    if sparse_plan is None:
        st.info("Hãy bật **LP tối ưu** hoặc **MODI** ở sidebar để có mốc đối chiếu cho Sinkhorn.")
        return

    if not st.checkbox("▶ Chạy demo Sinkhorn", value=False, key="run_sinkhorn"):
        st.caption("Tích vào ô trên để chạy (Sinkhorn lặp nhiều vòng, tính khi cần).")
        return

    # Mọi giá trị quy về MỤC TIÊU THẬT (chi phí thuần / lợi nhuận) như phần còn lại
    # của app — compute_real_cost bỏ qua ô Dummy & Big-M, dùng chi phí gốc. Nhờ đó
    # bài MAX hiển thị đúng lợi nhuận thay vì giá trị min đã transform.
    obj_name = objective_label(problem)  # "Lợi nhuận" (max) / "Chi phí" (min)

    def real_obj(alloc):
        return compute_real_cost(alloc, problem.cost, problem.forbidden)

    optimum = real_obj(sparse_plan)

    scale = cost_scale(transformed.cost)
    eps_values = [f * scale for f in SINKHORN_EPS_FACTORS]

    # Cache theo chữ ký bài toán + dãy ε để khỏi giải lại 5 lần mỗi lần rerun.
    sig = (transformed.cost.tobytes(), tuple(round(e, 9) for e in eps_values))
    cached = st.session_state.get("_sinkhorn_sweep")
    if cached and cached[0] == sig:
        sweep = cached[1]
    else:
        with st.spinner("⏳ Đang chạy Sinkhorn theo dãy ε..."):
            sweep = sinkhorn_sweep(transformed, eps_values)
        st.session_state["_sinkhorn_sweep"] = (sig, sweep)

    # Quy đổi từng nghiệm Sinkhorn sang mục tiêu thật. Dùng khoảng cách |·| tới tối
    # ưu: luôn ≥ 0 cho cả min/max, và tránh dấu âm khó hiểu khi bài có dummy_costs
    # (chi phí thật cắt cột Dummy ≠ mục tiêu LP tối ưu thực sự).
    rows = []
    for s in sweep:
        val = real_obj(s["allocation"])
        rows.append({**s, "cost": val, "gap": abs(val - optimum)})

    # ── 1. Hội tụ mục tiêu theo ε ─────────────────────────────────────────────
    st.markdown(f"##### 1️⃣ ε → 0: {obj_name.lower()} Sinkhorn hội tụ về nghiệm tối ưu")
    st.pyplot(plot_convergence(rows, optimum, ylabel=obj_name), width="stretch")

    table = pd.DataFrame([
        {
            "ε": f"{r['eps']:.3g}",
            f"{obj_name} Sinkhorn": f"{r['cost']:,.1f}",
            "Khoảng cách tới tối ưu": (
                f"{r['gap']:,.2f} ({r['gap']/abs(optimum)*100:.2f}%)" if optimum else f"{r['gap']:,.2f}"
            ),
            "Entropy (độ mờ)": f"{r['entropy']:,.2f}",
            "Số ô > 0": r["support"],
            "Số vòng lặp": r["iters"],
        }
        for r in rows
    ])
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        f"Nghiệm thưa **{sparse_label}** chỉ dùng **{count_support(sparse_plan)}** ô "
        f"(≈ m+n−1). ε càng nhỏ, số ô > 0 của Sinkhorn càng giảm về mức đó."
    )

    # ── 2. Độ mờ của nghiệm (entropy) ─────────────────────────────────────────
    st.markdown("##### 2️⃣ Tác dụng của entropy: nghiệm 'mờ' ↔ nghiệm góc thưa")
    blur = rows[0]            # ε lớn nhất
    sharp = rows[-1]          # ε nhỏ nhất
    fig = plot_plan_comparison(
        transformed,
        sparse_plan, sparse_label,
        blur["allocation"], blur["eps"],
        sharp["allocation"], sharp["eps"],
    )
    st.pyplot(fig, width="stretch")
    st.info(
        "MODI/LP cho **nghiệm góc** nên thưa, chỉ ~m+n−1 ô khác 0. Entropy phạt sự "
        "tập trung: ε lớn trải hàng ra nhiều ô nên 'mờ', ε nhỏ thì nghiệm sắc lại "
        "về đúng nghiệm góc."
    )
