from __future__ import annotations
import numpy as np
from src.models.problem import TransportationProblem
from src.models.result import AlgorithmResult
from src.models.step import AlgorithmStep
from src.core.cost import total_cost
from src.core.constants import BIG_M, MAX_MODI_ITERATIONS, EPSILON


def _compute_potentials(
    basis: list[tuple[int, int]], cost: np.ndarray, m: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    u[0] = 0.0
    changed = True
    while changed:
        changed = False
        for i, j in basis:
            if not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = cost[i, j] - u[i]
                changed = True
            elif not np.isnan(v[j]) and np.isnan(u[i]):
                u[i] = cost[i, j] - v[j]
                changed = True
    return u, v


def _compute_deltas(
    basis: list[tuple[int, int]], cost: np.ndarray, u: np.ndarray, v: np.ndarray, m: int, n: int
) -> np.ndarray:
    # Delta_ij = u_i + v_j - c_ij  (positive means improving)
    deltas = np.full((m, n), np.nan)
    basis_set = set(basis)
    for i in range(m):
        for j in range(n):
            if (i, j) not in basis_set:
                deltas[i, j] = u[i] + v[j] - cost[i, j]
    return deltas


class _DisjointSet:
    def __init__(self, nodes: list[tuple[str, int]]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: tuple[str, int]) -> tuple[str, int]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: tuple[str, int], b: tuple[str, int]) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _complete_basis(
    allocation: np.ndarray,
    cost: np.ndarray,
    m: int,
    n: int,
) -> list[tuple[int, int]]:
    """Build a spanning-tree basis and keep zero-allocation basics for degeneracy."""
    basis = [(i, j) for i in range(m) for j in range(n) if allocation[i, j] > EPSILON]
    nodes = [("r", i) for i in range(m)] + [("c", j) for j in range(n)]
    ds = _DisjointSet(nodes)

    tree_basis: list[tuple[int, int]] = []
    for i, j in basis:
        if ds.union(("r", i), ("c", j)):
            tree_basis.append((i, j))

    basis_set = set(tree_basis)
    candidates = [
        (cost[i, j] >= BIG_M / 2, float(cost[i, j]), i, j)
        for i in range(m)
        for j in range(n)
        if (i, j) not in basis_set
    ]

    for _, _, i, j in sorted(candidates):
        if len(tree_basis) >= m + n - 1:
            break
        if ds.union(("r", i), ("c", j)):
            tree_basis.append((i, j))
            basis_set.add((i, j))

    return tree_basis


def _find_cycle(
    entering: tuple[int, int], basis: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    """Return the unique alternating cycle formed by adding entering to the basis."""
    er, ec = entering
    start = ("r", er)
    target = ("c", ec)

    adjacency: dict[tuple[str, int], list[tuple[tuple[str, int], tuple[int, int]]]] = {}
    for i, j in basis:
        r_node = ("r", i)
        c_node = ("c", j)
        adjacency.setdefault(r_node, []).append((c_node, (i, j)))
        adjacency.setdefault(c_node, []).append((r_node, (i, j)))

    queue: list[tuple[tuple[str, int], list[tuple[int, int]]]] = [(start, [])]
    seen = {start}
    while queue:
        node, path_cells = queue.pop(0)
        if node == target:
            return [entering] + path_cells
        for next_node, cell in adjacency.get(node, []):
            if next_node in seen:
                continue
            seen.add(next_node)
            queue.append((next_node, path_cells + [cell]))
    return None


def modi(problem: TransportationProblem, initial_result: AlgorithmResult) -> AlgorithmResult:
    m, n = problem.m, problem.n
    cost = problem.cost.astype(float)
    allocation = initial_result.allocation.astype(float).copy()
    steps: list[AlgorithmStep] = []
    warnings = list(initial_result.warnings)
    basis = _complete_basis(allocation, cost, m, n)

    iteration = 0
    while iteration < MAX_MODI_ITERATIONS:
        if len(basis) < m + n - 1:
            warnings.append("Khong du o co so de tinh MODI.")
            break

        u, v = _compute_potentials(basis, cost, m, n)

        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            warnings.append("Could not compute all potentials — basis may be degenerate.")
            break

        deltas = _compute_deltas(basis, cost, u, v, m, n)

        potentials_dict = {
            "u": {problem.sources[i]: float(u[i]) for i in range(m)},
            "v": {problem.destinations[j]: float(v[j]) for j in range(n)},
        }

        # Check optimality: all deltas <= 0
        non_nan = deltas[~np.isnan(deltas)]
        if len(non_nan) == 0 or non_nan.max() <= EPSILON:
            tc = total_cost(allocation, cost)
            steps.append(AlgorithmStep(
                title=f"Bước {len(steps)+1}: Tối ưu",
                description=(
                    f"Mọi Delta_ij ≤ 0, phương án đã tối ưu. "
                    f"u = {[round(float(x),2) for x in u]}, "
                    f"v = {[round(float(x),2) for x in v]}. "
                    f"Tổng chi phí = {tc:.0f}."
                ),
                allocation=allocation.copy(),
                remaining_supply=np.zeros(m),
                remaining_demand=np.zeros(n),
                potentials=potentials_dict,
                deltas=deltas.copy(),
                cost=tc,
            ))
            return AlgorithmResult(
                algorithm_name=f"{initial_result.algorithm_name} + MODI",
                allocation=allocation,
                total_cost=tc,
                is_optimal=True,
                steps=initial_result.steps + steps,
                warnings=warnings,
            )

        # Find entering cell: max positive delta
        max_delta = float(non_nan.max())
        candidates = [
            (i, j) for i in range(m) for j in range(n)
            if not np.isnan(deltas[i, j]) and abs(deltas[i, j] - max_delta) < EPSILON
        ]
        entering = min(candidates)  # smallest (i,j) on tie

        cycle = _find_cycle(entering, basis)
        if cycle is None:
            warnings.append(f"Could not find cycle for entering cell {entering}.")
            break

        # Assign +/- signs: entering=+, alternating
        signs = []
        for k, cell in enumerate(cycle):
            signs.append('+' if k % 2 == 0 else '-')

        minus_cells = [cycle[k] for k in range(len(cycle)) if signs[k] == '-']
        theta = min(allocation[r, c] for r, c in minus_cells)
        leaving = min(
            minus_cells,
            key=lambda cell: (allocation[cell[0], cell[1]], cell[0], cell[1]),
        )

        old_cost = total_cost(allocation, cost)

        # Update allocation
        for k, (r, c) in enumerate(cycle):
            if signs[k] == '+':
                allocation[r, c] += theta
            else:
                allocation[r, c] -= theta
        allocation[allocation < EPSILON] = 0.0
        basis.append(entering)
        basis.remove(leaving)

        new_cost = total_cost(allocation, cost)

        cycle_desc = " → ".join(
            f"{problem.sources[r]}→{problem.destinations[c]}({s})"
            for (r, c), s in zip(cycle, signs)
        )
        src_e = problem.sources[entering[0]]
        dst_e = problem.destinations[entering[1]]
        desc = (
            f"u = {[round(float(x),2) for x in u]}, v = {[round(float(x),2) for x in v]}. "
            f"Ô vào cơ sở: {src_e}→{dst_e} với Delta={max_delta:.2f} > 0. "
            f"Chu trình: {cycle_desc}. "
            f"θ = {theta:.0f}; ô rời cơ sở: {problem.sources[leaving[0]]}→{problem.destinations[leaving[1]]}. "
            f"Chi phí giảm {old_cost - new_cost:.0f}, còn {new_cost:.0f}."
        )

        cycle_with_sign = [(r, c, s) for (r, c), s in zip(cycle, signs)]

        steps.append(AlgorithmStep(
            title=f"MODI Vòng {iteration+1}: đưa {src_e}→{dst_e} vào cơ sở, θ={theta:.0f}, cost→{new_cost:.0f}",
            description=desc,
            allocation=allocation.copy(),
            remaining_supply=np.zeros(m),
            remaining_demand=np.zeros(n),
            selected_cell=entering,
            potentials=potentials_dict,
            deltas=deltas.copy(),
            cycle=cycle_with_sign,
            cost=new_cost,
        ))

        iteration += 1

    tc = total_cost(allocation, cost)
    warnings.append(f"MODI stopped after {iteration} iterations.")
    return AlgorithmResult(
        algorithm_name=f"{initial_result.algorithm_name} + MODI",
        allocation=allocation,
        total_cost=tc,
        is_optimal=None,
        steps=initial_result.steps + steps,
        warnings=warnings,
    )
