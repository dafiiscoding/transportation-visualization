"""Thuật toán Sinkhorn — Optimal Transport có chính quy hoá entropy.

Thay vì giải bài toán vận tải tuyến tính chính xác (LP/MODI), Sinkhorn giải
bài toán *chính quy hoá*:

    min_P  <P, C>  - ε · H(P)        với H(P) = -Σ P_ij (log P_ij - 1)

trong đó ε > 0 là hệ số chính quy hoá entropy. Nghiệm có dạng
``P = diag(u) · exp(-C/ε) · diag(v)`` và được tìm bằng cách lặp chuẩn hoá
hàng/cột (thuật toán Sinkhorn–Knopp).

Hai tính chất minh hoạ trong app:
  1. ε → 0:  chi phí Sinkhorn hội tụ về nghiệm tối ưu LP/MODI.
  2. ε lớn:  nghiệm "mờ" (entropy cao, nhiều ô > 0) thay vì thưa như MODI.

Triển khai ở miền log (log-domain / log-sum-exp) để ổn định số học cả khi ε
rất nhỏ — tránh tràn số ``exp(-C/ε)``.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp

from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.models.step import AlgorithmStep
from src.core.constants import BIG_M, SINKHORN_MAX_ITER, SINKHORN_TOL


def cost_scale(cost: np.ndarray) -> float:
    """Thang chi phí tiêu biểu của bài (bỏ qua các ô Big-M/ô cấm).

    Dùng để quy ε tương đối → ε tuyệt đối, giúp demo hoạt động trên mọi ví dụ
    bất kể chi phí lớn nhỏ.
    """
    finite = cost[cost < BIG_M / 2]
    if finite.size == 0:
        return 1.0
    scale = float(np.median(np.abs(finite)))
    return scale if scale > 1e-9 else 1.0


def _record_iters(max_iter: int) -> set[int]:
    """Các vòng lặp sẽ ghi lại làm bước minh hoạ (thưa dần)."""
    picks = {1, 2, 3, 5, 10, 20, 50, 100}
    return {k for k in picks if k <= max_iter}


def _sinkhorn_core(
    problem: TransportationProblem,
    epsilon: float,
    max_iter: int,
    tol: float,
    record_steps: bool,
) -> tuple[np.ndarray, float, int, list[AlgorithmStep], list[str]]:
    """Lõi lặp Sinkhorn ở miền log. Trả về (P, cost, số_vòng_lặp, steps, warnings)."""
    m, n = problem.m, problem.n
    cost = problem.cost.astype(float)
    a = problem.supply.astype(float)
    b = problem.demand.astype(float)

    warnings: list[str] = []
    if not np.isclose(a.sum(), b.sum()):
        warnings.append("Cung ≠ Cầu — Sinkhorn cần bài cân bằng (hãy transform trước).")

    # Chuẩn hoá khối lượng về phân phối xác suất để ổn định, rồi nhân lại.
    total = a.sum()
    log_a = np.log(np.where(a > 0, a / total, 1e-300))
    log_b = np.log(np.where(b > 0, b / total, 1e-300))

    # Thế năng đối ngẫu f (hàng), g (cột) ở miền log.
    f = np.zeros(m)
    g = np.zeros(n)

    def plan(f_, g_):
        # P_ij = exp((f_i + g_j - C_ij) / ε), đã chuẩn hoá → nhân total để về khối lượng gốc.
        return np.exp((f_[:, None] + g_[None, :] - cost) / epsilon) * total

    steps: list[AlgorithmStep] = []
    record_at = _record_iters(max_iter) if record_steps else set()

    iters_done = 0
    for it in range(1, max_iter + 1):
        iters_done = it
        # Cập nhật xen kẽ ở miền log (chiếu Sinkhorn).
        f = epsilon * (log_a - logsumexp((g[None, :] - cost) / epsilon, axis=1))
        g = epsilon * (log_b - logsumexp((f[:, None] - cost) / epsilon, axis=0))

        # Sai số marginal hàng (marginal cột đã chính xác sau cập nhật g).
        P = plan(f, g)
        row_err = float(np.max(np.abs(P.sum(axis=1) - a)))

        if record_steps and (it in record_at or row_err < tol):
            tc_it = float((P * cost).sum())
            steps.append(AlgorithmStep(
                title=f"Vòng lặp {it}",
                description=f"Sai số marginal hàng = {row_err:.2e}; chi phí hiện tại = {tc_it:,.2f}.",
                allocation=P.copy(),
                remaining_supply=np.abs(P.sum(axis=1) - a),
                remaining_demand=np.abs(P.sum(axis=0) - b),
                cost=tc_it,
            ))

        if row_err < tol:
            break

    P = plan(f, g)
    P[P < 1e-12] = 0.0
    tc = float((P * cost).sum())
    if iters_done >= max_iter:
        warnings.append(f"Đạt giới hạn {max_iter} vòng lặp mà chưa hội tụ (ε nhỏ làm hội tụ chậm).")
    return P, tc, iters_done, steps, warnings


def sinkhorn(
    problem: TransportationProblem,
    epsilon: float,
    max_iter: int = SINKHORN_MAX_ITER,
    tol: float = SINKHORN_TOL,
    record_steps: bool = True,
) -> AlgorithmResult:
    """Giải bài toán vận tải (đã cân bằng) bằng Sinkhorn với hệ số ε cho trước.

    Yêu cầu bài toán đã ``transform_problem`` (cân bằng, max→min, ô cấm→Big-M).
    ``epsilon`` là giá trị tuyệt đối; dùng :func:`cost_scale` để quy đổi nếu cần.
    """
    if epsilon <= 0:
        epsilon = 1e-6
    P, tc, _, steps, warnings = _sinkhorn_core(problem, epsilon, max_iter, tol, record_steps)
    return AlgorithmResult(
        algorithm_name=f"Sinkhorn (ε={epsilon:.3g})",
        allocation=P,
        total_cost=tc,
        is_optimal=False,  # nghiệm xấp xỉ chính quy hoá, không phải đỉnh tối ưu
        steps=steps,
        warnings=warnings,
        transformed_problem=problem,
    )


def plan_entropy(P: np.ndarray) -> float:
    """Shannon entropy H = -Σ q log q của nghiệm đã chuẩn hoá thành xác suất.

    Đo độ 'mờ' không phụ thuộc thang khối lượng: 0 khi toàn bộ dồn vào 1 ô,
    lớn nhất ``log(số ô)`` khi trải đều. ε lớn → entropy cao.
    """
    total = P.sum()
    if total <= 0:
        return 0.0
    q = P[P > 1e-12] / total
    return float(-(q * np.log(q)).sum())


def count_support(P: np.ndarray, frac: float = 1e-4) -> int:
    """Số ô mang khối lượng đáng kể (≥ frac · tổng) — đo độ thưa của nghiệm."""
    thr = frac * P.sum()
    return int((P > thr).sum())


def sinkhorn_sweep(
    problem: TransportationProblem,
    eps_values: list[float],
    optimum: float | None = None,
    max_iter: int = SINKHORN_MAX_ITER,
    tol: float = SINKHORN_TOL,
) -> list[dict]:
    """Chạy Sinkhorn cho dãy ε, trả về số liệu so sánh để vẽ hội tụ.

    Mỗi phần tử: ``{eps, cost, gap, entropy, support, iters, allocation}``.
    """
    rows: list[dict] = []
    for eps in eps_values:
        eps = eps if eps > 0 else 1e-6
        P, tc, iters, _, _ = _sinkhorn_core(problem, eps, max_iter, tol, record_steps=False)
        gap = (tc - optimum) if optimum is not None else None
        rows.append({
            "eps": eps,
            "cost": tc,
            "gap": gap,
            "entropy": plan_entropy(P),
            "support": count_support(P),
            "iters": iters,
            "allocation": P,
        })
    return rows
