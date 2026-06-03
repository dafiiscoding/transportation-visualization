from __future__ import annotations
import pandas as pd
import streamlit as st
from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.visualization.comparison import (
    allocation_objective_value,
    gap_to_optimum_text,
    improvement_text,
    objective_label,
    objective_value,
    plot_phase_comparison,
)


def render_result_overview(
    problem: TransportationProblem,
    initial_results: dict[str, AlgorithmResult],
    modi_results: dict[str, AlgorithmResult],
    lp_result: AlgorithmResult | None,
    assignment_result: AlgorithmResult | None = None,
) -> None:
    st.subheader("📊 Tổng hợp kết quả")
    
    if problem.problem_type == "max":
        st.info("💡 **Ghi chú bài toán MAX (Tối đa lợi nhuận):** Các thuật toán giải (NW, LCM, VAM, MODI) bản chất là đi tìm cực tiểu (MIN). Do đó, hệ thống đã ngầm chuyển đổi ma trận lợi nhuận thành ma trận chi phí bằng cách: `C_new = M - C_old` (với M là giá trị lớn nhất trong ma trận). Thuật toán giải trên `C_new` để tìm cực tiểu, kết quả cuối cùng tự động được tính ngược lại thành cực đại lợi nhuận cho bạn.")

    # Check for multiple optimal solutions
    alt_cells = []
    if lp_result and lp_result.steps and lp_result.steps[-1].deltas is not None:
        final_deltas = lp_result.steps[-1].deltas
        final_alloc = lp_result.allocation
        m, n = problem.m, problem.n
        for i in range(m):
            for j in range(n):
                if final_alloc[i, j] <= 1e-6 and abs(final_deltas[i, j]) < 1e-6:
                    if problem.forbidden is None or not problem.forbidden[i,j]:
                        alt_cells.append((problem.sources[i], problem.destinations[j]))
            
    if alt_cells:
        st.warning("⚠️ **Phát hiện đa nghiệm tối ưu (Multiple Optimal Solutions)**")
        st.markdown(f"Hệ thống phát hiện **{len(alt_cells)} tuyến đường** hiện đang trống (không được phân bổ) nhưng có chi phí cơ hội (Delta) bằng 0:")
        alt_routes_str = ", ".join([f"**{s} → {d}**" for s, d in alt_cells])
        st.info(f"📍 **Các tuyến:** {alt_routes_str}\n\n👉 **Ý nghĩa thực tế:** Người điều phối có thể tùy ý điều hướng một phần hàng hóa vào các tuyến này (tạo thành các chu trình mới) để tạo ra các phương án phân bổ khác nhau mà **không làm thay đổi tổng chi phí tối ưu**. Điều này cung cấp sự linh hoạt cực lớn khi gặp sự cố đứt gãy chuỗi cung ứng. Tập hợp các phương án này tạo thành một mặt phẳng đa chiều (Hyperplane) trong không gian nghiệm.")

    if assignment_result is not None:
        value = objective_value(problem, assignment_result)
        st.metric("Hungarian - nghiệm tối ưu", f"{value:,.0f}")
        st.dataframe(
            pd.DataFrame([{
                "Phương pháp": "Hungarian",
                objective_label(problem): f"{value:,.0f}",
                "Số bước": len(assignment_result.steps),
                "Kết luận": "Nghiệm tối ưu cho bài toán phân công",
            }]),
            width="stretch",
            hide_index=True,
        )
        return

    lp_value = objective_value(problem, lp_result) if lp_result is not None else None
    label = objective_label(problem)

    summary_rows = []
    for name, initial in initial_results.items():
        modi = modi_results.get(name)
        final_result = modi if modi is not None else initial
        initial_steps = len(initial.steps)
        modi_steps = len(final_result.steps) - initial_steps
        final_value = objective_value(problem, final_result)
        
        path_values = []
        if initial is not None:
            initial_value = objective_value(problem, initial)
            path_values.append(initial_value)
        if modi is not None:
            modi_steps_list = final_result.steps[initial_steps:]
            path_values.extend(allocation_objective_value(problem, step.allocation) for step in modi_steps_list if step.cost is not None)
            
        summary_rows.append({
            "Phương pháp": name,
            "Bước gốc": initial_steps,
            "Vòng MODI": max(modi_steps, 0),
            "Tổng bước": len(final_result.steps),
            "Đường đi": " → ".join(f"{v:,.0f}" for v in path_values) if len(path_values) > 1 else "—",
            label: f"{final_value:,.0f}",
            "Cách mốc tối ưu": gap_to_optimum_text(problem, final_value, lp_value),
        })

    metric_cols = st.columns(3)
    if initial_results:
        best_initial_name, best_initial = _best_result(problem, initial_results)
        metric_cols[0].metric(
            f"Điểm xuất phát tốt nhất - {best_initial_name}",
            f"{objective_value(problem, best_initial):,.0f}",
        )
    else:
        metric_cols[0].metric("Điểm xuất phát", "Chưa chạy")

    if modi_results:
        best_modi_name, best_modi = _best_result(problem, modi_results)
        metric_cols[1].metric(
            f"MODI tốt nhất - từ {best_modi_name}",
            f"{objective_value(problem, best_modi):,.0f}",
        )
    else:
        metric_cols[1].metric("MODI", "Chưa chạy")

    metric_cols[2].metric(
        "LP tối ưu",
        f"{lp_value:,.0f}" if lp_value is not None else "Chưa chạy",
    )

    if lp_value is not None:
        st.success(f"LP là mốc tối ưu toàn cục: **{label} = {lp_value:,.0f}**.")
    else:
        st.info("Chưa có mốc LP nên bảng chỉ thể hiện giá trị từng phương pháp.")

    if summary_rows:
        st.markdown("### 🏆 Bảng so sánh tổng hợp")
        df_summary = pd.DataFrame(summary_rows)
        
        # Style to highlight the best (lowest cost for min, highest for max)
        def highlight_best(s):
            is_max = problem.problem_type == "max"
            try:
                # Extract numeric value from string (handling commas)
                vals = s.apply(lambda x: float(x.replace(',', '')))
                is_best = vals == (vals.max() if is_max else vals.min())
                return ['background-color: #dcfce7; font-weight: bold' if v else '' for v in is_best]
            except:
                return ['' for _ in s]

        st.dataframe(
            df_summary.style.apply(highlight_best, subset=[label]),
            width="stretch",
            hide_index=True
        )
        st.caption("💡 Hàng tô màu xanh là phương án có kết quả tốt nhất hiện tại.")

    st.divider()
    fig = plot_phase_comparison(problem, initial_results, modi_results, lp_result)
    st.pyplot(fig, width="stretch")


def _best_result(
    problem: TransportationProblem,
    results: dict[str, AlgorithmResult],
) -> tuple[str, AlgorithmResult]:
    if problem.problem_type == "max":
        return max(results.items(), key=lambda item: objective_value(problem, item[1]))
    return min(results.items(), key=lambda item: objective_value(problem, item[1]))
