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
from src.algorithms.sinkhorn import (
    sinkhorn, sinkhorn_sweep, cost_scale, plan_entropy, count_support,
)
from src.visualization.sinkhorn_viz import plot_convergence, plot_plan_comparison


def render_sinkhorn_panel(
    transformed: TransportationProblem,
    lp_result: AlgorithmResult | None,
    modi_results: dict[str, AlgorithmResult],
) -> None:
    st.subheader("🌀 MODI vs Sinkhorn — cầu nối tới Optimal Transport")
    st.caption(
        "Sinkhorn giải bản *chính quy hoá entropy* của chính bài toán vận tải này. "
        "Xem nó hội tụ về nghiệm MODI/LP khi ε → 0, và 'mờ' đi khi ε lớn."
    )

    # Cần một mốc tối ưu + nghiệm thưa để đối chiếu.
    sparse_plan = None
    sparse_label = ""
    optimum = None
    if lp_result is not None and lp_result.is_optimal:
        sparse_plan = lp_result.allocation
        sparse_label = "LP tối ưu"
        optimum = lp_result.total_cost
    elif modi_results:
        name, r = next(iter(modi_results.items()))
        sparse_plan = r.allocation
        sparse_label = f"MODI ({name})"
        optimum = r.total_cost

    if sparse_plan is None or optimum is None:
        st.info("Hãy bật **LP tối ưu** hoặc **MODI** ở sidebar để có mốc đối chiếu cho Sinkhorn.")
        return

    if not st.checkbox("▶ Chạy demo Sinkhorn", value=False, key="run_sinkhorn"):
        st.caption("Tích vào ô trên để chạy (Sinkhorn lặp nhiều vòng, tính khi cần).")
        return

    scale = cost_scale(transformed.cost)
    eps_values = [f * scale for f in SINKHORN_EPS_FACTORS]

    with st.spinner("⏳ Đang chạy Sinkhorn theo dãy ε..."):
        rows = sinkhorn_sweep(transformed, eps_values, optimum=optimum)

    # ── 1. Hội tụ chi phí theo ε ──────────────────────────────────────────────
    st.markdown("##### 1️⃣ ε → 0: chi phí Sinkhorn hội tụ về nghiệm tối ưu")
    st.pyplot(plot_convergence(rows, optimum), use_container_width=True)

    table = pd.DataFrame([
        {
            "ε": f"{r['eps']:.3g}",
            "Chi phí Sinkhorn": f"{r['cost']:,.1f}",
            "Chênh lệch vs tối ưu": f"{r['gap']:,.2f} ({r['gap']/optimum*100:.2f}%)",
            "Entropy (độ mờ)": f"{r['entropy']:,.2f}",
            "Số ô > 0": r["support"],
            "Số vòng lặp": r["iters"],
        }
        for r in rows
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
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
    st.pyplot(fig, use_container_width=True)
    st.info(
        "**Vì sao khác nhau?** MODI/LP cho **nghiệm góc** (đỉnh của đa diện ràng buộc) "
        "nên thưa — chỉ ~m+n−1 ô khác 0. Entropy trong Sinkhorn 'phạt' sự tập trung, "
        "nên ε lớn trải khối lượng ra nhiều ô (mờ); ε nhỏ thì hình phạt yếu dần và "
        "nghiệm sắc lại, tiệm cận nghiệm góc. Đây chính là đánh đổi **tốc độ ↔ độ "
        "chính xác** của Sinkhorn dùng rộng rãi trong học máy (Wasserstein, OT)."
    )
