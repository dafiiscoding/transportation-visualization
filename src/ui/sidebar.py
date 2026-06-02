from __future__ import annotations
import streamlit as st
from src.io.examples import example_label, list_examples

def render_sidebar() -> dict:
    st.sidebar.markdown("## 🚚 Vận tải tối ưu")
    st.sidebar.divider()

    st.sidebar.markdown("**📥 Chọn hoặc Nhập liệu**")
    mode = st.sidebar.radio(
        "Chế độ nhập liệu",
        ["Ví dụ có sẵn", "Nhập tay", "Upload CSV"],
        key="input_mode",
        label_visibility="collapsed",
    )

    example_name = None
    if mode == "Ví dụ có sẵn":
        examples = list_examples()
        example_name = st.sidebar.selectbox(
            "Chọn ví dụ",
            examples,
            format_func=example_label,
            key="example_name",
            label_visibility="collapsed",
        )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("**🎯 Loại bài toán**")
    problem_type = st.sidebar.radio(
        "Loại bài toán",
        ["min", "max"],
        format_func=lambda x: "📉 Cực tiểu chi phí (Min)" if x == "min" else "📈 Tối đa lợi nhuận (Max)",
        key="problem_type",
        label_visibility="collapsed",
    )
    
    st.sidebar.divider()

    col1, col2 = st.sidebar.columns(2)
    run = col1.button("▶ Chạy & Giải", type="primary", use_container_width=True)
    reset = col2.button("🔄 Khởi tạo lại", use_container_width=True)

    return {
        "mode": mode,
        "problem_type": problem_type,
        "initial_methods": ["Xấp xỉ Vogel (VAM)", "Góc Tây Bắc (NW)", "Chi phí nhỏ nhất (LCM)"],
        "apply_modi": True,
        "run_lp": True,
        "example_name": example_name,
        "run": run,
        "reset": reset,
    }