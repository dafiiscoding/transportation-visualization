import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.models.problem import TransportationProblem
from src.algorithms.northwest import northwest_corner
from src.algorithms.least_cost import least_cost_method
from src.algorithms.vogel import vogel
from src.algorithms.modi import modi

SUPPLY = np.array([50, 70, 80], dtype=float)
DEMAND = np.array([60, 30, 40, 70], dtype=float)
COST = np.array([[2,4,5,1],[3,6,4,8],[1,2,5,3]], dtype=float)


def make_problem():
    return TransportationProblem("test","min",["A1","A2","A3"],["B1","B2","B3","B4"],SUPPLY,DEMAND,COST)


def test_modi_from_nw_reaches_optimal():
    p = make_problem()
    r = modi(p, northwest_corner(p))
    assert r.total_cost == pytest.approx(450.0)
    assert r.is_optimal is True


def test_modi_trace_costs():
    p = make_problem()
    nw = northwest_corner(p)
    r = modi(p, nw)
    modi_steps = r.steps[len(nw.steps):]
    costs = [s.cost for s in modi_steps if s.selected_cell is not None]
    # Expected: 690 -> 640 -> 540 -> 450
    assert costs[0] == pytest.approx(640.0)
    assert costs[1] == pytest.approx(540.0)
    assert costs[2] == pytest.approx(450.0)


def test_modi_from_lcm():
    p = make_problem()
    r = modi(p, least_cost_method(p))
    assert r.total_cost == pytest.approx(450.0)


def test_modi_from_vogel():
    p = make_problem()
    r = modi(p, vogel(p))
    assert r.total_cost == pytest.approx(450.0)
    assert r.is_optimal is True
