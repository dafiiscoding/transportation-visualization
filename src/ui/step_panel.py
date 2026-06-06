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
    plot_modi_tableau,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dựng các "hành trình" liền mạch: mỗi thuật toán = 1 chuỗi bước duy nhất chạy
# xuyên suốt Khởi tạo → MODI → Tối ưu (modi_results[m].steps đã nối sẵn 2 phần).
# ──────────────────────────────────────────────────────────────────────────────
def _build_journeys(
    initial_results: dict[str, AlgorithmResult],
    modi_results: dict[str, AlgorithmResult],
    lp_result: AlgorithmResult | None,
) -> dict[str, dict]:
    journeys: dict[str, dict] = {}
    for method, initial in initial_results.items():
        if method in modi_results:
            seq = modi_results[method].steps
            boundary = len(initial.steps)
            label = f"{method} → MODI"
        else:
            seq = initial.steps
            boundary = len(seq)
            label = method
        journeys[label] = {
            "seq": seq, "boundary": boundary, "kind": "method", "method": method,
        }
    if lp_result is not None:
        journeys["⚡ LP tối ưu"] = {
            "seq": lp_result.steps, "boundary": len(lp_result.steps),
            "kind": "lp", "method": None, "result": lp_result,
        }
    return journeys


def render_step_panel(
    problem: TransportationProblem,
    initial_results: dict[str, AlgorithmResult],
    modi_results: dict[str, AlgorithmResult] | None = None,
    lp_result: AlgorithmResult | None = None,
    assignment_result: AlgorithmResult | None = None,
) -> None:
    st.subheader("🔍 Hành trình từng bước")

    modi_results = modi_results or {}

    # Bài toán phân công: 1 chuỗi Hungarian liền mạch (không có hành trình/giai đoạn).
    if assignment_result is not None:
        _render_journey(
            problem, assignment_result.steps, len(assignment_result.steps),
            label="assignment", kind="assignment", method_label="Phân công (Hungarian)",
        )
        return

    journeys = _build_journeys(initial_results, modi_results, lp_result)
    if not journeys:
        st.info("Chưa có kết quả.")
        return

    labels = list(journeys.keys())

    # --- Nút "Tình huống then chốt": nhảy thẳng tới đúng hành trình + đúng bước ---
    if problem.key_highlights:
        st.info("💡 **Tình huống then chốt:** Nhảy nhanh đến các bước quan trọng để phân tích.")
        cols = st.columns(len(problem.key_highlights))
        for i, highlight in enumerate(problem.key_highlights):
            if cols[i].button(f"📌 {highlight['label']}", key=f"jump_{i}", use_container_width=True):
                _apply_highlight_jump(highlight, journeys, initial_results)
                st.rerun()

    # --- Bộ chọn hành trình (1 lớp, thay cho radio giai đoạn + selectbox phương pháp) ---
    if st.session_state.get("journey") not in labels:
        st.session_state["journey"] = labels[0]
    journey_label = st.radio("Hành trình", labels, horizontal=True, key="journey")
    J = journeys[journey_label]

    if J["kind"] == "lp":
        _render_lp_header(problem, J["result"])

    _render_journey(
        problem, J["seq"], J["boundary"], journey_label,
        kind=J["kind"], method_label=J["method"],
    )


def _apply_highlight_jump(highlight: dict, journeys: dict[str, dict], initial_results: dict) -> None:
    """Quy nút then chốt (stage, method, step) → hành trình + chỉ số bước toàn cục."""
    stage = highlight.get("stage", "")
    method = highlight.get("method")
    step = int(highlight.get("step", 1))

    if "LP" in stage:
        label = "⚡ LP tối ưu"
        gidx = step
    else:
        label = f"{method} → MODI" if f"{method} → MODI" in journeys else method
        if label not in journeys:
            return
        if "MODI" in stage:
            boundary = len(initial_results[method].steps) if method in initial_results else 0
            gidx = boundary + step
        else:
            gidx = step

    if label not in journeys:
        return
    seq_len = len(journeys[label]["seq"])
    st.session_state["journey"] = label
    st.session_state[f"step_idx_journey_{label}"] = max(1, min(gidx, seq_len))


def _render_lp_header(problem: TransportationProblem, lp_result: AlgorithmResult) -> None:
    st.markdown("#### ⚙️ Cơ chế LP (HiGHS) — đầu vào & đầu ra")
    ci = st.columns(4)
    ci[0].metric("Số biến (xᵢⱼ)", f"{problem.m * problem.n}")
    ci[1].metric("Số ràng buộc", f"{problem.m + problem.n}")
    ci[2].metric("Solver", "HiGHS (SciPy)")
    ci[3].metric("Trạng thái", "Tối ưu" if lp_result.is_optimal else "—")
    st.markdown("**📥 Đầu vào gửi cho solver:** ma trận chi phí $C$ (m×n), vector cung $a$, vector cầu $b$ — đóng gói thành bài LP chuẩn:")
    st.latex(r"\min \sum_{i,j} c_{ij}x_{ij}\quad\text{với}\quad \sum_j x_{ij}=a_i,\ \ \sum_i x_{ij}=b_j,\ \ x_{ij}\ge 0")
    st.markdown(f"**📤 Đầu ra solver trả về:** trạng thái = *optimal*, tổng chi phí $Z^* = {lp_result.total_cost:,.0f}$, và ma trận phân bổ $x^*$ (bảng bên dưới).")
    st.caption("Khác MODI: HiGHS nạp toàn bộ bài toán một lần và trả thẳng nghiệm tối ưu — không đi từng vòng, nên chỉ có 1 'bước'.")
    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# Điều hướng + dải giai đoạn + vẽ 1 bước
# ──────────────────────────────────────────────────────────────────────────────
def _step_navigator(visible_steps: int, state_key: str) -> int:
    """Slider + nút ◀▶ cho 1 hành trình; trả về số bước đang xem (1-based)."""
    if visible_steps == 1:
        st.caption("Bước 1/1")
        return 1

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
        if st.button("◀", key=f"prev_{state_key}", disabled=(step_num <= 1)):
            st.session_state[slider_key] = step_num - 1
            st.rerun()
    with col_next:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶", key=f"next_{state_key}", disabled=(step_num >= visible_steps)):
            st.session_state[slider_key] = step_num + 1
            st.rerun()
    return step_num


def _render_phase_strip(idx: int, boundary: int, seq_len: int) -> str | None:
    """Dải tiến trình giai đoạn; in-đậm giai đoạn hiện tại. Trả về nhãn để gắn badge."""
    if boundary >= seq_len:
        return None  # hành trình 1 giai đoạn (không có MODI) → không cần dải
    in_init = idx < boundary
    p1 = f"① Khởi tạo (b1–b{boundary})"
    p2 = f"② MODI (b{boundary + 1}–b{seq_len})"
    p1 = f"**{p1}**" if in_init else p1
    p2 = p2 if in_init else f"**{p2}**"
    st.markdown(f"🧭 Giai đoạn: {p1} &nbsp;·&nbsp; {p2}")
    return "① Khởi tạo" if in_init else "② MODI"


def _render_journey(
    problem: TransportationProblem,
    seq: list[AlgorithmStep],
    boundary: int,
    label: str,
    kind: str,
    method_label: str | None,
) -> None:
    if not seq:
        st.info("Không có bước nào để hiển thị.")
        return

    state_key = f"journey_{label}"
    seq_len = len(seq)

    if kind == "method" and boundary < seq_len:
        st.caption(f"Bước gốc: {boundary} · Vòng MODI: {seq_len - boundary} · Tổng: {seq_len}")
        st.info(
            "👀 **Cách đọc mỗi vòng:** xem **thẻ KPI** (chi phí giảm bao nhiêu · ô vào · ô ra · θ) "
            "và **heatmap phân bổ** bên phải trước; bảng số chỉ để tra chi tiết khi cần."
        )

    step_num = _step_navigator(seq_len, state_key)
    idx = step_num - 1

    phase = _render_phase_strip(idx, boundary, seq_len) if kind == "method" else None

    if kind == "lp":
        heading = "⚡ Phương án tối ưu do LP trả về"
    elif kind == "assignment":
        heading = "Bài toán phân công — nghiệm tối ưu Hungarian"
    else:
        heading = f"🚚 Hành trình {method_label}"

    _render_one_step(problem, seq, idx, heading, phase_label=phase)


def _render_one_step(
    problem: TransportationProblem,
    steps: list[AlgorithmStep],
    idx: int,
    heading: str,
    phase_label: str | None = None,
) -> None:
    step = steps[idx]
    visible_steps = len(steps)

    st.markdown(f"### {heading}")
    badge = ""
    if phase_label:
        badge = (
            f" &nbsp;<span style='background:#e0e7ff; color:#3730a3; padding:2px 8px; "
            f"border-radius:999px; font-size:0.85rem; font-weight:600'>{phase_label}</span>"
        )
    st.markdown(
        f"<div style='font-size: 1.2rem; font-weight: bold; color: #1e40af; margin-bottom: 1rem'>"
        f"Bước {idx + 1}/{visible_steps}: {step.title}{badge}</div>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown(f"<div style='background: #eff6ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #2563eb; margin-bottom: 1.5rem'>{step.description}</div>", unsafe_allow_html=True)

        # --- Thẻ KPI: nhìn vài số là hiểu cả vòng, khỏi dò bảng (delta tính xuyên giai đoạn) ---
        prev_cost = steps[idx - 1].cost if idx > 0 else None
        kpi = st.columns(4)
        if step.cost is not None:
            delta = (step.cost - prev_cost) if prev_cost is not None else None
            kpi[0].metric(
                "💰 Chi phí", f"{step.cost:,.0f}",
                f"{delta:+,.0f}" if delta not in (None, 0) else None,
                delta_color="inverse",
            )
        if step.selected_cell is not None:
            r, c = step.selected_cell
            kpi[1].metric("➕ Ô vào", f"{problem.sources[r]}→{problem.destinations[c]}")
        if step.leaving_cell is not None:
            r, c = step.leaving_cell
            kpi[2].metric("➖ Ô ra", f"{problem.sources[r]}→{problem.destinations[c]}")
        if step.theta is not None:
            kpi[3].metric("θ (lượng dịch)", f"{step.theta:,.0f}")

        # Cycle - High visibility for MODI
        if step.cycle:
            cycle_str = " → ".join(
                f"<span style='background:#fef08a; padding: 2px 6px; border-radius: 4px; font-weight:bold'>{problem.sources[r]}→{problem.destinations[c]} ({sign})</span>"
                for r, c, sign in step.cycle
            )
            st.markdown(f"**🔄 Chu trình điều chỉnh:**<br>{cycle_str}", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1.5rem'></div>", unsafe_allow_html=True)

        with st.expander("📋 Bảng làm bài chi tiết (Tableau)"):
            tableau = build_transportation_tableau(problem, step.allocation, step)

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
                max_r = r_df["Penalty Hàng"].max() if not r_df.empty else 0
                st.dataframe(r_df.style.apply(lambda s: ['background-color: #fef08a; font-weight:bold' if v == max_r and v > 0 else '' for v in s], subset=['Penalty Hàng']), hide_index=True)
            with c2:
                c_df = pd.DataFrame(step.col_penalties.items(), columns=["Đích", "Penalty Cột"])
                max_c = c_df["Penalty Cột"].max() if not c_df.empty else 0
                st.dataframe(c_df.style.apply(lambda s: ['background-color: #fef08a; font-weight:bold' if v == max_c and v > 0 else '' for v in s], subset=['Penalty Cột']), hide_index=True)

    with right_col:
        fig_tab = plot_modi_tableau(problem, step, title="Bảng vận tải (kiểu báo cáo)")
        st.pyplot(fig_tab, width="stretch")
        st.caption(
            "Chi phí ở góc trên–trái, **lượng phân** đậm góc dưới–phải; "
            "$u_i$/$v_j$ ở lề, $\\Delta_{ij}$ có cung; chu trình $+/-$ nối nét đứt đỏ "
            "(ô vào: viền lục, ô ra: viền đỏ đứt)."
        )
        with st.expander("🌡️ Heatmap Δ (xem nhanh ô dương)"):
            if step.deltas is not None:
                st.pyplot(plot_delta_heatmap(problem, step), width="stretch")
            else:
                st.caption("Bước này chưa tính Δ.")

    # --- Bảng vận tải TRƯỚC → SAU bước này (dễ theo dõi thay đổi) ---
    if idx > 0:
        prev_step = steps[idx - 1]
        with st.expander("📊 Bảng vận tải: TRƯỚC → SAU bước này", expanded=bool(step.cycle)):
            bcol, acol = st.columns(2)
            with bcol:
                st.markdown("**Trước bước**")
                st.pyplot(
                    plot_modi_tableau(problem, prev_step, title=f"Trước (bước {idx}/{visible_steps})"),
                    width="stretch",
                )
            with acol:
                st.markdown("**Sau bước** (ô đổi tô vàng)")
                st.pyplot(
                    plot_modi_tableau(problem, step, prev_allocation=prev_step.allocation,
                                      title=f"Sau (bước {idx + 1}/{visible_steps})"),
                    width="stretch",
                )

    # Thế vị & Delta — gập vào expander, nêu rõ mục đích
    if step.potentials or step.deltas is not None:
        with st.expander("🔢 Chi tiết Thế vị & Δ — vì sao chọn ô vào"):
            st.caption("Δ tính cho ô NGOÀI cơ sở; **Δ dương lớn nhất ⇒ ô được đưa vào cơ sở** vòng này. Ô đỏ (Δ>0) là cơ hội giảm chi phí.")
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
                    st.markdown(r"**Ma trận Delta ($\Delta_{ij} = u_i + v_j - c_{ij}$):**")
                    pm2, pn2 = problem.m, problem.n
                    delta_display = step.deltas[:pm2, :pn2].copy()
                    df_d = pd.DataFrame(
                        delta_display,
                        index=problem.sources[:pm2],
                        columns=problem.destinations[:pn2],
                    )

                    def color_delta(v):
                        if pd.isna(v): return "background-color: #f1f5f9; color: #94a3b8"
                        if v > 0: return "background-color: #fee2e2; color: #991b1b; font-weight: bold"
                        return "background-color: #d1fae5; color: #065f46"

                    st.dataframe(df_d.style.map(color_delta), width="stretch")
