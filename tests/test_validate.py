import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.models.problem import TransportationProblem
from src.core.validate import validate_problem, get_warnings


def make_basic():
    return TransportationProblem(
        "test","min",["A1"],["B1"],
        np.array([10.0]),np.array([10.0]),np.array([[5.0]])
    )


def test_valid_problem():
    assert validate_problem(make_basic()) == []


def test_negative_supply():
    p = make_basic()
    p.supply[0] = -1
    errs = validate_problem(p)
    assert any("supply" in e.lower() for e in errs)


def test_zero_total_supply():
    p = TransportationProblem("t","min",["A"],["B"],np.array([0.0]),np.array([10.0]),np.array([[1.0]]))
    errs = validate_problem(p)
    assert any("supply" in e.lower() for e in errs)


def test_unbalanced_warning():
    p = TransportationProblem("t","min",["A"],["B"],np.array([20.0]),np.array([10.0]),np.array([[1.0]]))
    w = get_warnings(p)
    assert any("unbalanced" in ww.lower() or "supply" in ww.lower() for ww in w)


def test_max_warning():
    p = TransportationProblem("t","max",["A"],["B"],np.array([10.0]),np.array([10.0]),np.array([[5.0]]))
    w = get_warnings(p)
    assert any("max" in ww.lower() for ww in w)
