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


def _detect_alt_optima(problem, lp_result):
    """Tìm tuyến Δ=0 ngoài cơ sở (đa nghiệm) trên nghiệm LP và dựng 1 nghiệm tối ưu thay thế."""
    import numpy as np
    from src.core.transform import transform_problem
    from src.algorithms.modi import (
        _complete_basis, _compute_potentials, _compute_deltas, _find_cycle,
    )
    t = transform_problem(problem)
    alloc = lp_result.allocation
    m, n = t.m, t.n
    if alloc.shape != (m, n):
        return [], None
    cost = t.cost.astype(float)
    basis = _complete_basis(alloc, cost, m, n)
    if len(basis) < m + n - 1:
        return [], None
    u, v = _compute_potentials(basis, cost, m, n)
    if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        return [], None
    deltas = _compute_deltas(basis, cost, u, v, m, n)

    routes, alt_alloc = [], None
    for i in range(problem.m):
        for j in range(problem.n):
            if alloc[i, j] > 1e-6 or np.isnan(deltas[i, j]) or abs(deltas[i, j]) >= 1e-6:
                continue
            if problem.forbidden is not None and problem.forbidden[i, j]:
                continue
            # Đa nghiệm THẬT chỉ khi pivot dịch được hàng (theta > 0). Nếu theta = 0
            # thì đây là ô suy biến (degenerate): đẩy hàng vào cho lại đúng phương án cũ,
            # không sinh ra đỉnh tối ưu mới -> không tính là đa nghiệm.
            cyc = _find_cycle((i, j), basis)
            if not cyc:
                continue
            signs = ['+' if k % 2 == 0 else '-' for k in range(len(cyc))]
            minus = [cyc[k] for k in range(len(cyc)) if signs[k] == '-']
            theta = min(alloc[r, c] for r, c in minus)
            if theta <= 1e-6:
                continue
            routes.append((t.sources[i], t.destinations[j]))
            if alt_alloc is None:
                a2 = alloc.copy()
                for k, (r, c) in enumerate(cyc):
                    a2[r, c] += theta if signs[k] == '+' else -theta
                a2[a2 < 1e-9] = 0.0
                alt_alloc = a2
    return routes, alt_alloc


def _enumerate_optimal_face(problem, lp_result, max_vertices: int = 40):
    """Liệt kê toàn bộ ĐỈNH (cực biên) của diện tối ưu và số chiều của diện.

    BFS trên 1-skeleton của diện: từ đỉnh LP, pivot qua mọi ô ngoài cơ sở có
    Δ=0 và θ>0 để sang đỉnh tối ưu kề; lặp tới khi không còn đỉnh mới.
    Trả về (danh sách allocation các đỉnh, số chiều affine của diện).
    """
    import numpy as np
    from src.core.transform import transform_problem
    from src.algorithms.modi import (
        _complete_basis, _compute_potentials, _compute_deltas, _find_cycle,
    )
    t = transform_problem(problem)
    m, n = t.m, t.n
    if lp_result.allocation.shape != (m, n):
        return [], 0
    cost = t.cost.astype(float)
    forb = problem.forbidden

    def key(a):
        return tuple(np.round(a.ravel(), 6))

    start = lp_result.allocation.astype(float).copy()
    start[start < 1e-9] = 0.0
    vertices = {key(start): start}
    queue = [start]

    while queue and len(vertices) < max_vertices:
        a = queue.pop(0)
        basis = _complete_basis(a, cost, m, n)
        if len(basis) < m + n - 1:
            continue
        u, v = _compute_potentials(basis, cost, m, n)
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            continue
        deltas = _compute_deltas(basis, cost, u, v, m, n)
        basis_set = set(basis)
        for i in range(m):
            for j in range(n):
                if (i, j) in basis_set:
                    continue
                # Chỉ xét đa nghiệm trên tuyến THẬT (bỏ hàng/cột giả & ô cấm),
                # nhất quán với _detect_alt_optima.
                if i >= problem.m or j >= problem.n:
                    continue
                if np.isnan(deltas[i, j]) or abs(deltas[i, j]) >= 1e-6:
                    continue
                if forb is not None and forb[i, j]:
                    continue
                cyc = _find_cycle((i, j), basis)
                if not cyc:
                    continue
                signs = ['+' if kk % 2 == 0 else '-' for kk in range(len(cyc))]
                minus = [cyc[kk] for kk in range(len(cyc)) if signs[kk] == '-']
                theta = min(a[r, c] for r, c in minus)
                if theta <= 1e-6:
                    continue
                a2 = a.copy()
                for kk, (r, c) in enumerate(cyc):
                    a2[r, c] += theta if signs[kk] == '+' else -theta
                a2[a2 < 1e-9] = 0.0
                kx = key(a2)
                if kx not in vertices:
                    vertices[kx] = a2
                    queue.append(a2)

    verts = list(vertices.values())
    if len(verts) <= 1:
        return verts, 0
    base = verts[0].ravel()
    mat = np.array([vv.ravel() - base for vv in verts[1:]])
    dim = int(np.linalg.matrix_rank(mat, tol=1e-6))
    return verts, dim


def _face_shape_name(dim: int, k: int) -> str:
    """Tên hình học của diện tối ưu theo số chiều (dim) và số đỉnh (k)."""
    if dim <= 0:
        return "một điểm (nghiệm thực chất là duy nhất)"
    if dim == 1:
        return "một đoạn thẳng (cạnh) nối 2 đỉnh"
    if dim == 2:
        if k == 3:
            return "một tam giác (diện phẳng 2 chiều, 3 đỉnh)"
        if k == 4:
            return "một tứ giác (diện phẳng 2 chiều, 4 đỉnh)"
        return f"một đa giác phẳng (diện 2 chiều, {k} đỉnh)"
    if dim == 3:
        if k == 4:
            return "một tứ diện (khối 3 chiều, 4 đỉnh)"
        return f"một khối đa diện (diện 3 chiều, {k} đỉnh)"
    return f"một đa diện {dim} chiều ({k} đỉnh)"


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

    # Bài có chi phí cột/hàng giả (metadata.dummy_costs, vd phí lưu kho): tách rõ
    # phần chi phí ở tuyến giả — vốn bị bảng tổng hợp (chi phí vận chuyển THUẦN) bỏ
    # qua vì compute_real_cost cắt cột/hàng Dummy. Nhờ đó câu chuyện "kho rẻ" hiện đủ.
    if lp_result is not None and problem.metadata.get("dummy_costs"):
        real = objective_value(problem, lp_result)
        storage = lp_result.total_cost - real  # chênh = chi phí dồn ở tuyến giả
        if storage > 1e-6:
            st.info(
                f"🏬 **Tách chi phí (bài có tuyến giả tính phí):** vận chuyển thuần "
                f"**{real:,.0f}** + lưu kho/phạt ở tuyến giả **{storage:,.0f}** = tổng "
                f"**{real + storage:,.0f}**. Bảng tổng hợp bên dưới hiển thị *chi phí vận chuyển thuần*; "
                f"phần lưu kho quyết định hàng dư nằm ở đâu (xem sơ đồ phân bổ)."
            )

    # Đa nghiệm tối ưu: liệt kê toàn bộ ĐỈNH của diện tối ưu (tính trên nghiệm LP)
    MAX_FACE_VERTS = 40
    routes, verts, face_dim = [], [], 0
    if lp_result is not None:
        routes, _ = _detect_alt_optima(problem, lp_result)
        verts, face_dim = _enumerate_optimal_face(problem, lp_result, MAX_FACE_VERTS)

    if len(verts) >= 2:
        pm, pn = problem.m, problem.n
        k = len(verts)
        capped = k >= MAX_FACE_VERTS
        k_txt = f"≥ {k}" if capped else f"{k}"
        shape = _face_shape_name(face_dim, k)
        if capped:
            shape = f"một đa diện {face_dim} chiều (rất nhiều đỉnh)"

        st.warning(
            f"⚠️ **Kết luận: bài toán có ĐA NGHIỆM — tập nghiệm tối ưu là một DIỆN.** "
            f"Về mặt hình học, diện nghiệm tối ưu là **{shape}** (gồm **{k_txt} đỉnh** cực biên, "
            f"tất cả cùng chi phí tối ưu Z\\*)."
        )
        if routes:
            st.caption("Các tuyến Δ=0 (ô trống có thể đẩy hàng vào để đổi sang đỉnh khác): "
                       + ", ".join(f"{s}→{d}" for s, d in routes))

        st.markdown("**Phương trình (tham số) của diện nghiệm tối ưu — tập mọi nghiệm:**")
        if k == 2 and not capped:
            st.latex(r"x^* = \lambda\,x_1 + (1-\lambda)\,x_2,\qquad 0 \le \lambda \le 1")
        else:
            kk = "k" if capped else str(k)
            st.latex(
                r"x^* = \sum_{i=1}^{%s} \lambda_i\,x_i,\qquad "
                r"\sum_{i=1}^{%s}\lambda_i = 1,\ \ \lambda_i \ge 0" % (kk, kk)
            )
        st.caption(
            f"Mọi tổ hợp lồi của các đỉnh $x_i$ bên dưới đều là nghiệm tối ưu → **vô số phương án** "
            f"(diện {face_dim} chiều), không chỉ riêng các đỉnh."
        )

        show = verts[:6]
        title = (f"🔀 Xem {len(show)} trong số {k_txt} đỉnh tối ưu của diện"
                 if k > len(show) else f"🔀 Xem {k} đỉnh tối ưu của diện")
        with st.expander(title):
            for idx, vv in enumerate(show, start=1):
                st.markdown(f"**Đỉnh $x_{{{idx}}}$**")
                df_v = pd.DataFrame(
                    vv[:pm, :pn].astype(int),
                    index=problem.sources[:pm], columns=problem.destinations[:pn],
                )
                st.dataframe(df_v, width="stretch")
            if k > len(show):
                st.caption(f"… và {('nhiều' if capped else k - len(show))} đỉnh khác (đã ẩn bớt cho gọn).")

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
            label: f"{final_value:,.0f}",
        })

    if lp_value is not None:
        st.metric(f"🎯 Nghiệm tối ưu (LP) — {label}", f"{lp_value:,.0f}")
    elif modi_results:
        _bn, _br = _best_result(problem, modi_results)
        st.metric(f"Kết quả MODI tốt nhất — {label}", f"{objective_value(problem, _br):,.0f}")

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
