from __future__ import annotations
import streamlit as st
from src.io.examples import example_label, list_examples


def render_sidebar() -> dict:
    st.sidebar.markdown("## 🚚 Vận tải tối ưu")
    st.sidebar.divider()

    st.sidebar.markdown("**📥 Chế độ nhập liệu**")
    mode = st.sidebar.radio(
        "Chế độ nhập liệu",
        ["Ví dụ có sẵn", "Nhập tay", "Upload CSV"],
        key="input_mode",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("**🎯 Loại bài toán**")
    problem_type = st.sidebar.radio(
        "Loại bài toán",
        ["min", "max"],
        format_func=lambda x: "⬇ Tối thiểu hoá chi phí" if x == "min" else "⬆ Tối đa hoá lợi nhuận",
        key="problem_type",
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    st.sidebar.markdown("**1️⃣ Tìm điểm xuất phát**")
    initial_methods = st.sidebar.multiselect(
        "Chọn phương pháp điểm xuất phát",
        ["Góc Tây Bắc (NW)", "Chi phí nhỏ nhất (LCM)", "Xấp xỉ Vogel (VAM)"],
        default=["Góc Tây Bắc (NW)", "Chi phí nhỏ nhất (LCM)", "Xấp xỉ Vogel (VAM)"],
        key="initial_methods",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("**2️⃣ Tối ưu hóa**")
    apply_modi = st.sidebar.checkbox("Chạy MODI sau các điểm xuất phát", value=True)
    run_lp = st.sidebar.checkbox("Tính LP làm mốc tối ưu", value=True)

    st.sidebar.divider()

    example_name = None
    if mode == "Ví dụ có sẵn":
        st.sidebar.markdown("**📂 Lộ trình phân tích**")
        examples = list_examples()
        example_name = st.sidebar.selectbox(
            "Chọn ví dụ",
            examples,
            format_func=example_label,
            key="example_name",
            label_visibility="collapsed",
        )
        st.sidebar.divider()

    col1, col2 = st.sidebar.columns(2)
    run = col1.button("▶ Chạy", type="primary", width="stretch")
    reset = col2.button("↺ Đặt lại", width="stretch")

    return {
        "mode": mode,
        "problem_type": problem_type,
        "initial_methods": initial_methods,
        "apply_modi": apply_modi,
        "run_lp": run_lp,
        "example_name": example_name,
        "run": run,
        "reset": reset,
    }
