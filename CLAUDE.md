# Transportation Algorithm Visualizer — CLAUDE.md

## Conventions
- Delta_ij = u_i + v_j - c_ij  (NOT c_ij - u_i - v_j)
- Min problem is optimal when all Delta_ij <= 0
- u[0] = 0 when computing potentials
- Tie-breaking Least Cost: (1) largest potential allocation, (2) smaller row index, (3) smaller col index
- MODI guaranteed only for balanced min problems, size <= 6x6; larger problems use LP Solver

## Test Results
- Basic 3x4: NW=690, LCM=530, VAM=450, LP=450
- Logistics 5x12: LP=120000
- MODI trace from NW: 690→640→540→450
