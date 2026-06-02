import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.models.problem import TransportationProblem
from src.core.transform import transform_problem
from src.core.constants import BIG_M


def test_balanced_no_change():
    p = TransportationProblem("t","min",["A"],["B"],np.array([10.0]),np.array([10.0]),np.array([[5.0]]))
    tp = transform_problem(p)
    assert tp.m == 1 and tp.n == 1


def test_unbalanced_supply_adds_dummy_dest():
    p = TransportationProblem("t","min",["A"],["B"],np.array([20.0]),np.array([10.0]),np.array([[5.0]]))
    tp = transform_problem(p)
    assert tp.n == 2
    assert tp.destinations[-1] == "Dummy"
    assert tp.demand[-1] == 10.0


def test_unbalanced_demand_adds_dummy_src():
    p = TransportationProblem("t","min",["A"],["B"],np.array([10.0]),np.array([20.0]),np.array([[5.0]]))
    tp = transform_problem(p)
    assert tp.m == 2
    assert tp.sources[-1] == "Dummy"
    assert tp.supply[-1] == 10.0


def test_forbidden_becomes_bigm():
    forbidden = np.array([[True, False]])
    p = TransportationProblem("t","min",["A"],["B1","B2"],np.array([10.0]),np.array([5.0,5.0]),
                               np.array([[3.0,2.0]]),forbidden=forbidden)
    tp = transform_problem(p)
    assert tp.cost[0, 0] == pytest.approx(BIG_M)


def test_max_transforms_to_min():
    cost = np.array([[3.0, 5.0],[4.0, 2.0]])
    p = TransportationProblem("t","max",["A1","A2"],["B1","B2"],
                               np.array([10.0,10.0]),np.array([10.0,10.0]),cost)
    tp = transform_problem(p)
    c_max = 5.0
    expected = c_max - cost
    np.testing.assert_array_almost_equal(tp.cost, expected)
