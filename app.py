from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import streamlit as st

from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.core.validate import validate_problem, get_warnings
from src.core.transform import transform_problem, compute_real_cost
from src.algorithms.northwest import northwest_corner
from src.algorithms.least_cost import least_cost_method
from src.algorithms.vogel import vogel
from src.algorithms.modi import modi
from src.algorithms.lp_solver import lp_solver
from src.algorithms.assignment import solve_assignment
from src.io.examples import load_example
from src.io.csv_io import parse_csv
from src.io.templates import TEMPLATE_CSV
from src.ui.sidebar import render_sidebar
from src.ui.input_panel import render_manual_input, render_problem_table
from src.ui.result_panel import render_result_overview
from src.ui.step_panel import render_step_panel
from src.ui.explanation import nw_explanation, lcm_explanation, vogel_explanation, modi_explanation, lp_explanation
from src.ui.guide import render_guide
from src.visualization.heatmap import build_transportation_tableau, plot_cost_heatmap, plot_allocation_heatmap
from src.visualization.network import plot_network
from src.visualization.export import fig_to_bytes, allocation_to_csv, result_to_json


INITIAL_METHODS = {
    "Góc Tây Bắc (NW)": northwest_corner,
    "Chi phí nhỏ nhất (LCM)": least_cost_method,
    "Xấp xỉ Vogel (VAM)": vogel,
}

st.set_page_config(
    page_title="Vận tải tối ưu",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #1e40af 100%);
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #e0f0ff !important;
}
[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
[data-testid="stSidebar"] hr { border-color: #3b5c8a; }

/* Primary button in sidebar */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #2563eb;
    border: none;
    color: white;
    font-weight: 700;
    border-radius: 8px;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #1d4ed8;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.15);
    color: white;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #f1f5f9;
    padding: 4px 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 0.9rem;
    color: #475569;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: white !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px 16px;
    border-left: 4px solid #2563eb;
}

/* Divider */
hr { border-color: #e2e8f0; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}

/* Info / Success / Warning */
[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def run_algorithms(
    problem: TransportationProblem,
    selected_initial_methods: list[str],
    apply_modi: bool,
    run_lp: bool,
    is_assignment: bool,
) -> dict:
    transformed = transform_problem(problem)
    results = {
        "initial": {},
        "modi": {},
        "lp": None,
        "assignment": None,
        "transformed": transformed,
    }

    if is_assignment:
        assignment = solve_assignment(transformed)
        assignment.algorithm_name = "Hungarian - tối ưu phân công"
        results["assignment"] = assignment
        return results

    for method in selected_initial_methods:
        solver = INITIAL_METHODS.get(method)
        if solver is None:
            continue

        initial = solver(transformed)
        initial.algorithm_name = method
        results["initial"][method] = initial

        if apply_modi:
            optimized = modi(transformed, initial)
            optimized.algorithm_name = f"MODI từ {method}"
            results["modi"][method] = optimized

    if run_lp:
        lp = lp_solver(transformed)
        lp.algorithm_name = "LP tối ưu"
        results["lp"] = lp

    return results


def _has_results(results: dict) -> bool:
    return bool(
        results.get("initial")
        or results.get("modi")
        or results.get("lp") is not None
        or results.get("assignment") is not None
    )


def _result_choices(results: dict) -> dict[str, AlgorithmResult]:
    choices: dict[str, AlgorithmResult] = {}
    assignment = results.get("assignment")
    if assignment is not None:
        choices["Hungarian - tối ưu phân công"] = assignment
        return choices

    for name, result in results.get("initial", {}).items():
        choices[f"Điểm xuất phát - {name}"] = result
    for name, result in results.get("modi", {}).items():
        choices[f"MODI tối ưu từ {name}"] = result
    lp = results.get("lp")
    if lp is not None:
        choices["LP tối ưu"] = lp
    return choices


def main() -> None:
    st.set_page_config(
        page_title="Transportation Algorithm Visualizer",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Onboarding First Load ---
    if "first_visit" not in st.session_state:
        st.toast("👋 Chào mừng bạn đến với Transportation Visualizer! Nhấn 'Chạy & Giải' để bắt đầu.", icon="🎉")
        st.session_state["first_visit"] = False

    st.title("🚚 Transportation Algorithm Visualizer")
    st.markdown("Giải quyết và trực quan hóa bài toán vận tải bằng các thuật toán quy hoạch tuyến tính.")

    cfg = render_sidebar()

    if cfg["reset"]:
        for key in list(st.session_state.keys()):
            if key not in ("input_mode", "problem_type", "initial_methods", "first_visit"):
                del st.session_state[key]
        st.rerun()

    # ── Load problem ──────────────────────────────────────────────────────────
    problem: TransportationProblem | None = None
    is_assignment = False

    if cfg["mode"] == "Ví dụ có sẵn" and cfg["example_name"]:
        try:
            problem = load_example(cfg["example_name"])
            if "assignment" in cfg["example_name"].lower():
                is_assignment = True
            if cfg["problem_type"] == "max" and problem.problem_type != "max":
                problem = TransportationProblem(
                    name=problem.name, problem_type="max",
                    sources=problem.sources, destinations=problem.destinations,
                    supply=problem.supply, demand=problem.demand,
                    cost=problem.cost, forbidden=problem.forbidden,
                )
        except Exception as e:
            st.error(f"Lỗi tải ví dụ: {e}")

    elif cfg["mode"] == "Nhập tay":
        problem = render_manual_input(cfg["problem_type"])

    elif cfg["mode"] == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.download_button("⬇ Tải template CSV", TEMPLATE_CSV, "template.csv", "text/csv")
        if uploaded:
            try:
                problem = parse_csv(uploaded.read())
            except Exception as e:
                st.error(f"Lỗi đọc CSV: {e}")

    # ── Tabs (always rendered) ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab_guide = st.tabs([
        "📋 Dữ liệu", "📊 Kết quả & Network", "🔍 Từng bước giải", "💾 Export", "📖 Hướng dẫn"
    ])

    with tab_guide:
        render_guide()

    # ── Validate ──────────────────────────────────────────────────────────────
    if problem is None:
        with tab1:
            st.info("👈 Chọn ví dụ hoặc nhập dữ liệu ở sidebar để bắt đầu.")
        return

    errors = validate_problem(problem)
    if errors:
        with tab1:
            for e in errors:
                st.error(e)
        return

    warnings = get_warnings(problem)

    with tab1:
        if problem.description:
            st.markdown(f"""
            <div style='background:#f1f5f9; border-left:4px solid #1e40af; padding:16px; border-radius:8px; margin-bottom:20px'>
                <div style='color:#475569; font-weight:700; text-transform:uppercase; font-size:0.8rem; margin-bottom:8px'>🔍 Phân tích bối cảnh</div>
                <div style='color:#1e293b; font-size:1.1rem; line-height:1.5'>{problem.description}</div>
            </div>
            """, unsafe_allow_html=True)
            
        if problem.insight:
            st.markdown(f"""
            <div style='background:#fff7ed; border-left:4px solid #ea580c; padding:16px; border-radius:8px; margin-bottom:20px'>
                <div style='color:#9a3412; font-weight:700; text-transform:uppercase; font-size:0.8rem; margin-bottom:8px'>💡 Bài học rút ra (Insight)</div>
                <div style='color:#7c2d12; font-size:1rem; font-style: italic'>{problem.insight}</div>
            </div>
            """, unsafe_allow_html=True)
        
        render_problem_table(problem)
        for w in warnings:
            st.warning(w)

        transformed_preview = transform_problem(problem)
        if (
            transformed_preview.m != problem.m
            or transformed_preview.n != problem.n
            or problem.problem_type == "max"
            or problem.forbidden is not None
        ):
            with st.expander("Bảng sau khi chuẩn hóa để giải (Dummy / Phạt)"):
                render_problem_table(transformed_preview)

        st.divider()
        with st.expander("📖 Giải thích các thuật toán"):
            st.info(nw_explanation())
            st.info(lcm_explanation())
            st.info(vogel_explanation())
            st.info(modi_explanation())
            st.info(lp_explanation())

    # Handle run trigger from tabs
    run_clicked = cfg["run"] or st.session_state.get("run_triggered_from_tab", False)
    if run_clicked:
        st.session_state["run_triggered_from_tab"] = False

    # Run when button pressed
    if run_clicked or "results" in st.session_state:
        if run_clicked:
            with st.spinner("⏳ Đang tính toán..."):
                results = run_algorithms(
                    problem,
                    cfg["initial_methods"],
                    cfg["apply_modi"],
                    cfg["run_lp"],
                    is_assignment,
                )
            st.session_state["results"] = results
            st.session_state["problem"] = problem
            st.session_state["transformed"] = results["transformed"]
        else:
            results = st.session_state.get("results", {})
            problem = st.session_state.get("problem", problem)

        if not _has_results(results):
            with tab2:
                st.info("Dữ liệu đã sẵn sàng. Hãy bấm nút **Chạy & Giải** để bắt đầu.")
                if st.button("🚀 Chạy & Giải ngay", type="primary", key="run_tab2"):
                    st.session_state["run_triggered_from_tab"] = True
                    st.rerun()
            with tab3:
                st.info("Vui lòng chạy thuật toán để xem chi tiết từng bước giải.")
                if st.button("🚀 Chạy & Giải ngay", type="primary", key="run_tab3"):
                    st.session_state["run_triggered_from_tab"] = True
                    st.rerun()
            return

        transformed = results.get("transformed", transform_problem(problem))
        initial_results = results.get("initial", {})
        modi_results = results.get("modi", {})
        lp_result = results.get("lp")
        assignment_result = results.get("assignment")

        with tab2:
            render_result_overview(problem, initial_results, modi_results, lp_result, assignment_result)
            
            st.markdown("---")
            st.subheader("🗺 Sơ đồ mạng lưới vận tải tối ưu (Network Graph)")
            
            # Draw network for LP result (or best result)
            best_r = lp_result if lp_result else list(_result_choices(results).values())[0] if _result_choices(results) else None
            if best_r:
                is_large = transformed.m + transformed.n >= 12
                if is_large:
                    st.info("🌟 **Góc nhìn vĩ mô (Macro View):** Với mạng lưới phức tạp (nhiều điểm cung/cầu), biểu đồ sẽ tự động ẩn các con số chi phí để tập trung hiển thị **luồng phân phối chính** (các đường nối đậm). Đây chính là cách LP Solver bao quát toàn cục hệ thống thay vì sa đà vào tính toán cục bộ.")
                else:
                    st.caption("Trực quan hóa cấu trúc phân bổ hàng hóa giữa các điểm Phát và Thu. Đường nối càng đậm, lượng phân bổ càng lớn.")

                show_labels = st.checkbox("Hiển thị nhãn chi phí trên đường nối", value=not is_large)
                fig_net = plot_network(transformed, best_r, show_labels=show_labels, figsize=(14, 8))
                st.pyplot(fig_net, use_container_width=True)

        with tab3:
            render_step_panel(transformed, initial_results, modi_results, lp_result, assignment_result)

        with tab4:
            st.subheader("💾 Export kết quả")
            choices = _result_choices(results)
            if choices:
                exp_label = st.selectbox("Chọn kết quả để export", list(choices.keys()), key="exp_result")
                r_exp = choices[exp_label]

                c1, c2, c3 = st.columns(3)
                fig_c = plot_cost_heatmap(transformed)
                c1.download_button("🖼 Heatmap chi phí (PNG)", fig_to_bytes(fig_c), "cost_heatmap.png", "image/png", width="stretch")

                fig_a = plot_allocation_heatmap(transformed, r_exp.allocation)
                c2.download_button("🖼 Heatmap phân bổ (PNG)", fig_to_bytes(fig_a), "allocation_heatmap.png", "image/png", width="stretch")

                fig_n = plot_network(transformed, r_exp)
                c3.download_button("🖼 Network graph (PNG)", fig_to_bytes(fig_n), "network.png", "image/png", width="stretch")

                c4, c5 = st.columns(2)
                csv_data = allocation_to_csv(transformed, r_exp)
                c4.download_button("📄 Phân bổ (CSV)", csv_data, "allocation.csv", "text/csv", width="stretch")

                json_data = result_to_json(transformed, r_exp)
                c5.download_button("📄 Kết quả (JSON)", json_data, "result.json", "application/json", width="stretch")
            else:
                st.info("Chưa có kết quả để export.")
    else:
        with tab2:
            st.info("👈 Bấm **▶ Chạy & Giải** trong thanh bên trái để bắt đầu tính toán.")


if __name__ == "__main__":
    main()
