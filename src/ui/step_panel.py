from __future__ import annotations
import pandas as pd
import streamlit as st
from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.models.step import AlgorithmStep
from src.visualization.heatmap import (
    build_transportation_tableau,
    plot_allocation_heatmap,
    plot_delta_heatmap,
)


def render_step_panel(
    problem: TransportationProblem,
    initial_results: dict[str, AlgorithmResult],
    modi_results: dict[str, AlgorithmResult] | None = None,
    lp_result: AlgorithmResult | None = None,
    assignment_result: AlgorithmResult | None = None,
) -> None:
    st.subheader("🔍 Từng bước theo giai đoạn")

    # --- Strategic Highlights Section ---
    if problem.key_highlights:
        st.info("💡 **Tình huống then chốt:** Nhảy nhanh đến các bước quan trọng để phân tích.")
        cols = st.columns(len(problem.key_highlights))
        for i, highlight in enumerate(problem.key_highlights):
            if cols[i].button(f"📌 {highlight['label']}", key=f"jump_{i}", use_container_width=True):
                # Set stage
                st.session_state["step_stage"] = highlight["stage"]
                # Set method
                if "Tìm điểm xuất phát" in highlight["stage"]:
                    st.session_state["step_initial_method"] = highlight["method"]
                    st.session_state[f"step_idx_initial_{highlight['method']}"] = highlight["step"]
                elif "MODI" in highlight["stage"]:
                    st.session_state["step_modi_method"] = highlight["method"]
                    st.session_state[f"step_idx_modi_{highlight['method']}"] = highlight["step"]
                elif "LP" in highlight["stage"]:
                    st.session_state["step_idx_lp_steps"] = highlight["step"]
                st.rerun()

    modi_results = modi_results or {}

    if assignment_result is not None:
        _render_step_sequence(
            problem,
            assignment_result,
            assignment_result.steps,
            state_key="assignment_steps",
            heading="Bài toán phân công - nghiệm tối ưu Hungarian",
        )
        return

    if not initial_results and not modi_results and lp_result is None:
        st.info("Chưa có kết quả.")
        return

    stage_options = []
    if initial_results:
        stage_options.append("1. Tìm điểm xuất phát")
    if modi_results:
        stage_options.append("2. MODI tối ưu hóa")
    if lp_result is not None:
        stage_options.append("3. LP tối ưu")

    stage = st.radio("Giai đoạn", stage_options, horizontal=True, key="step_stage")

    if stage.startswith("1."):
        method = st.selectbox("Phương pháp điểm xuất phát", list(initial_results.keys()), key="step_initial_method")
        result = initial_results[method]
        _render_step_sequence(
            problem,
            result,
            result.steps,
            state_key=f"initial_{method}",
            heading=f"Giai đoạn 1 - {method}: tạo phương án cơ sở ban đầu",
        )
        return

    if stage.startswith("2."):
        method = st.selectbox("MODI bắt đầu từ", list(modi_results.keys()), key="step_modi_method")
        result = modi_results[method]
        initial_result = initial_results.get(method)
        initial_step_count = len(initial_result.steps) if initial_result is not None else 0
        modi_steps = result.steps[initial_step_count:]
        path_costs = [step.cost for step in modi_steps if step.cost is not None]
        st.caption(
            f"Bước gốc: {initial_step_count} · Vòng MODI: {len(modi_steps)} · Tổng: {len(result.steps)}"
        )
        if path_costs:
            st.caption(
                " → ".join(f"{cost:,.0f}" for cost in path_costs)
            )
        _render_step_sequence(
            problem,
            result,
            modi_steps,
            state_key=f"modi_{method}",
            heading=f"Giai đoạn 2 - MODI từ {method}: kiểm tra và tối ưu phương án",
        )
        return

    if lp_result is not None:
        _render_step_sequence(
            problem,
            lp_result,
            lp_result.steps,
            state_key="lp_steps",
            heading="Giai đoạn kiểm chứng - LP là mốc tối ưu toàn cục",
        )


def _render_step_sequence(
    problem: TransportationProblem,
    result: AlgorithmResult,
    steps: list[AlgorithmStep],
    state_key: str,
    heading: str,
) -> None:
    if not steps:
        st.info("Không có bước nào để hiển thị.")
        return

    visible_steps = len(steps)

    # Navigation: slider + prev/next buttons
    if visible_steps == 1:
        step_num = 1
        st.caption("Bước 1/1")
    else:
        # Clamp session state to valid range when switching algos
        slider_key = f"step_idx_{state_key}"
        slider_kwargs = {"key": slider_key}
        if slider_key in st.session_state:
            clamped = max(1, min(int(st.session_state[slider_key]), visible_steps))
            if int(st.session_state[slider_key]) != clamped:
                st.session_state[slider_key] = clamped
        else:
            slider_kwargs["value"] = 1
        col_prev, col_slider, col_next = st.columns([1, 8, 1])
        with col_slider:
            step_num = st.slider("Bước", 1, visible_steps, **slider_kwargs)
        with col_prev:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("◀", key=f"prev_step_{state_key}", disabled=(step_num <= 1)):
                st.session_state[slider_key] = step_num - 1
                st.rerun()
        with col_next:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("▶", key=f"next_step_{state_key}", disabled=(step_num >= visible_steps)):
                st.session_state[slider_key] = step_num + 1
                st.rerun()

    step_idx = step_num - 1
    step = steps[step_idx]

    st.markdown(f"### {heading}")
    st.markdown(f"<div style='font-size: 1.2rem; font-weight: bold; color: #1e40af; margin-bottom: 1rem'>Bước {step_num}/{visible_steps}: {step.title}</div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown(f"<div style='background: #eff6ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #2563eb; margin-bottom: 1.5rem'>{step.description}</div>", unsafe_allow_html=True)
        
        if step.cost is not None:
            st.metric("💰 Tổng chi phí tại bước này", f"{step.cost:,.0f}")

        # Cycle - High visibility for MODI
        if step.cycle:
            cycle_str = " → ".join(
                f"<span style='background:#fef08a; padding: 2px 6px; border-radius: 4px; font-weight:bold'>{problem.sources[r]}→{problem.destinations[c]} ({sign})</span>"
                for r, c, sign in step.cycle
            )
            st.markdown(f"**🔄 Chu trình điều chỉnh:**<br>{cycle_str}", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1.5rem'></div>", unsafe_allow_html=True)

        st.markdown("**📋 Bảng làm bài (Tableau):**")
        tableau = build_transportation_tableau(problem, step.allocation, step)
        
        # Style the tableau
        def style_tableau(val):
            if "⬅" in val: return "background-color: #dbeafe; font-weight: bold"
            if "(+)" in val: return "background-color: #d1fae5; font-weight: bold"
            if "(-)" in val: return "background-color: #fee2e2; font-weight: bold"
            if "Δ:+" in val: return "color: #b91c1c; font-weight: bold"
            return ""

        st.dataframe(tableau.style.map(style_tableau), width="stretch")
        st.caption("Ký hiệu: `[Chi phí]` | `Phân bổ` | `Δ: Cơ hội tối ưu` | `(±) Chu trình` | `⬅ Ô đang xét` ")

        if step.row_penalties is not None and step.col_penalties is not None:
            st.markdown("---")
            st.markdown("#### 🎯 Bảng phân tích Penalty (Chi phí cơ hội)")
            st.caption("Penalty = Hiệu số giữa 2 chi phí nhỏ nhất trong cùng một hàng hoặc cột. Thuật toán chọn hàng/cột có Penalty lớn nhất để ưu tiên phân bổ trước, nhằm tránh bị ép vào các ô có chi phí 'cắt cổ'.")
            c1, c2 = st.columns(2)
            with c1:
                r_df = pd.DataFrame(step.row_penalties.items(), columns=["Nguồn", "Penalty Hàng"])
                # Highlight the max penalty
                max_r = r_df["Penalty Hàng"].max() if not r_df.empty else 0
                st.dataframe(r_df.style.apply(lambda s: ['background-color: #fef08a; font-weight:bold' if v == max_r and v > 0 else '' for v in s], subset=['Penalty Hàng']), hide_index=True)
            with c2:
                c_df = pd.DataFrame(step.col_penalties.items(), columns=["Đích", "Penalty Cột"])
                max_c = c_df["Penalty Cột"].max() if not c_df.empty else 0
                st.dataframe(c_df.style.apply(lambda s: ['background-color: #fef08a; font-weight:bold' if v == max_c and v > 0 else '' for v in s], subset=['Penalty Cột']), hide_index=True)

    with right_col:
        fig = plot_allocation_heatmap(problem, step.allocation, step)
        st.pyplot(fig, width="stretch")

        if step.deltas is not None:
            fig_delta = plot_delta_heatmap(problem, step)
            st.pyplot(fig_delta, width="stretch")

    # Potentials and deltas - More prominent for MODI
    if step.potentials or step.deltas is not None:
        st.markdown("---")
        st.markdown("#### 🔢 Phân tích Thế vị & Delta")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            if step.potentials:
                st.markdown("**Thế vị nguồn ($u_i$):**")
                u_df = pd.DataFrame(step.potentials["u"].items(), columns=["Nguồn", "u_i"])
                st.dataframe(u_df, hide_index=True)
                
                st.markdown("**Thế vị đích ($v_j$):**")
                v_df = pd.DataFrame(step.potentials["v"].items(), columns=["Đích", "v_j"])
                st.dataframe(v_df, hide_index=True)

        with c2:
            if step.deltas is not None:
                st.markdown("**Ma trận Delta ($\Delta_{ij} = u_i + v_j - c_{ij}$):**")
                pm2, pn2 = problem.m, problem.n
                delta_display = step.deltas[:pm2, :pn2].copy()
                df_d = pd.DataFrame(
                    delta_display,
                    index=problem.sources[:pm2],
                    columns=problem.destinations[:pn2],
                )
                
                def color_delta(v):
                    if pd.isna(v): return "background-color: #f1f5f9; color: #94a3b8" # basic
                    if v > 0: return "background-color: #fee2e2; color: #991b1b; font-weight: bold"
                    return "background-color: #d1fae5; color: #065f46"

                st.dataframe(df_d.style.map(color_delta), width="stretch")
                st.caption("Ô màu đỏ ($\Delta > 0$): còn có thể giảm chi phí.")
