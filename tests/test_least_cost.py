import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.models.problem import TransportationProblem
from src.algorithms.least_cost import least_cost_method

SUPPLY = np.array([50, 70, 80], dtype=float)
DEMAND = np.array([60, 30, 40, 70], dtype=float)
COST = np.array([[2,4,5,1],[3,6,4,8],[1,2,5,3]], dtype=float)


def make_problem():
    return TransportationProblem("test","min",["A1","A2","A3"],["B1","B2","B3","B4"],SUPPLY,DEMAND,COST)


def test_lcm_cost():
    r = least_cost_method(make_problem())
    assert r.total_cost == pytest.approx(530.0)


def test_lcm_allocation():
    r = least_cost_method(make_problem())
    expected = np.array([[0,0,0,50],[0,10,40,20],[60,20,0,0]], dtype=float)
    np.testing.assert_array_almost_equal(r.allocation, expected)


def test_lcm_supply_demand():
    r = least_cost_method(make_problem())
    np.testing.assert_array_almost_equal(r.allocation.sum(axis=1), SUPPLY)
    np.testing.assert_array_almost_equal(r.allocation.sum(axis=0), DEMAND)
