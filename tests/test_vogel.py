import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.models.problem import TransportationProblem
from src.algorithms.vogel import vogel

SUPPLY = np.array([50, 70, 80], dtype=float)
DEMAND = np.array([60, 30, 40, 70], dtype=float)
COST = np.array([[2,4,5,1],[3,6,4,8],[1,2,5,3]], dtype=float)


def make_problem():
    return TransportationProblem("test","min",["A1","A2","A3"],["B1","B2","B3","B4"],SUPPLY,DEMAND,COST)


def test_vogel_cost():
    r = vogel(make_problem())
    assert r.total_cost == pytest.approx(450.0)


def test_vogel_allocation():
    r = vogel(make_problem())
    expected = np.array([[0,0,0,50],[30,0,40,0],[30,30,0,20]], dtype=float)
    np.testing.assert_array_almost_equal(r.allocation, expected)


def test_vogel_supply_demand():
    r = vogel(make_problem())
    np.testing.assert_array_almost_equal(r.allocation.sum(axis=1), SUPPLY)
    np.testing.assert_array_almost_equal(r.allocation.sum(axis=0), DEMAND)
