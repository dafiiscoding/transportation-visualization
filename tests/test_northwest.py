import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.models.problem import TransportationProblem
from src.algorithms.northwest import northwest_corner

SUPPLY = np.array([50, 70, 80], dtype=float)
DEMAND = np.array([60, 30, 40, 70], dtype=float)
COST = np.array([[2,4,5,1],[3,6,4,8],[1,2,5,3]], dtype=float)
SOURCES = ["A1","A2","A3"]
DESTS = ["B1","B2","B3","B4"]


def make_problem():
    return TransportationProblem("test","min",SOURCES,DESTS,SUPPLY,DEMAND,COST)


def test_nw_cost():
    r = northwest_corner(make_problem())
    assert r.total_cost == pytest.approx(690.0)


def test_nw_allocation():
    r = northwest_corner(make_problem())
    expected = np.array([[50,0,0,0],[10,30,30,0],[0,0,10,70]], dtype=float)
    np.testing.assert_array_almost_equal(r.allocation, expected)


def test_nw_supply_satisfied():
    r = northwest_corner(make_problem())
    np.testing.assert_array_almost_equal(r.allocation.sum(axis=1), SUPPLY)


def test_nw_demand_satisfied():
    r = northwest_corner(make_problem())
    np.testing.assert_array_almost_equal(r.allocation.sum(axis=0), DEMAND)
