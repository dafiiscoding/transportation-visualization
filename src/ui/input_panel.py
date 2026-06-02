from __future__ import annotations
import numpy as np
import streamlit as st
from src.models.problem import TransportationProblem


def render_manual_input(problem_type: str) -> TransportationProblem | None:
    st.subheader("✏️ Nhập dữ liệu thủ công")
    col1, col2 = st.columns(2)
    with col1:
        m = st.number_input("Số nguồn (m)", min_value=1, max_value=10, value=3, step=1)
    with col2:
        n = st.number_input("Số đích (n)", min_value=1, max_value=12, value=4, step=1)

    m, n = int(m), int(n)

    st.markdown("**Tên nguồn:**")
    src_cols = st.columns(m)
    sources = [src_cols[i].text_input(f"Nguồn {i+1}", value=f"A{i+1}", key=f"src_{i}") for i in range(m)]

    st.markdown("**Tên đích:**")
    dst_cols = st.columns(n)
    destinations = [dst_cols[j].text_input(f"Đích {j+1}", value=f"B{j+1}", key=f"dst_{j}") for j in range(n)]

    st.markdown("**Cung (Supply):**")
    sup_cols = st.columns(m)
    supply = np.array([sup_cols[i].number_input(sources[i], value=50.0, key=f"sup_{i}") for i in range(m)])

    st.markdown("**Cầu (Demand):**")
    dem_cols = st.columns(n)
    demand = np.array([dem_cols[j].number_input(destinations[j], value=30.0, key=f"dem_{j}") for j in range(n)])

    st.markdown("**Ma trận chi phí:**")
    cost = np.zeros((m, n))
    for i in range(m):
        row_cols = st.columns(n + 1)
        row_cols[0].markdown(f"**{sources[i]}**")
        for j in range(n):
            cost[i, j] = row_cols[j + 1].number_input(
                destinations[j], value=1.0, min_value=0.0, key=f"cost_{i}_{j}"
            )

    has_forbidden = st.checkbox("Có ô cấm?", value=False)
    forbidden = None
    if has_forbidden:
        st.markdown("**Ma trận ô cấm:**")
        forbidden = np.zeros((m, n), dtype=bool)
        for i in range(m):
            fb_cols = st.columns(n + 1)
            fb_cols[0].markdown(f"{sources[i]}")
            for j in range(n):
                if fb_cols[j + 1].checkbox(destinations[j], key=f"fb_{i}_{j}"):
                    forbidden[i, j] = True
        if not forbidden.any():
            forbidden = None

    return TransportationProblem(
        name="Nhập tay",
        problem_type=problem_type,
        sources=sources,
        destinations=destinations,
        supply=supply,
        demand=demand,
        cost=cost,
        forbidden=forbidden,
    )


def render_problem_table(problem: TransportationProblem) -> None:
    import pandas as pd
    from src.core.constants import BIG_M

    st.subheader(f"📋 Bài toán: {problem.name}")

    cost_display = problem.cost.copy().astype(object)
    if problem.forbidden is not None:
        for i in range(problem.m):
            for j in range(problem.n):
                if problem.forbidden[i, j]:
                    cost_display[i, j] = "∞ (cấm)"

    df = pd.DataFrame(
        cost_display,
        index=problem.sources,
        columns=problem.destinations,
    )
    df["Supply"] = problem.supply
    demand_row = list(problem.demand) + [None]
    df.loc["Demand"] = demand_row

    st.dataframe(df, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Tổng cung", f"{problem.total_supply:.0f}")
    col2.metric("🛒 Tổng cầu", f"{problem.total_demand:.0f}")
    balanced = problem.is_balanced
    col3.metric(
        "⚖️ Cân bằng?",
        "✅ Có" if balanced else "❌ Không",
        delta=f"Chênh lệch: {abs(problem.total_supply - problem.total_demand):.0f}" if not balanced else None,
    )
