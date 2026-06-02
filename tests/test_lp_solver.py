import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.models.problem import TransportationProblem
from src.algorithms.lp_solver import lp_solver


def test_lp_example1():
    p = TransportationProblem(
        "test","min",["A1","A2","A3"],["B1","B2","B3","B4"],
        np.array([50,70,80],dtype=float),
        np.array([60,30,40,70],dtype=float),
        np.array([[2,4,5,1],[3,6,4,8],[1,2,5,3]],dtype=float),
    )
    r = lp_solver(p)
    assert r.total_cost == pytest.approx(450.0)
    assert r.is_optimal is True


def test_lp_example2():
    supply = np.array([800,500,1000,400,800],dtype=float)
    demand = np.array([200,250,200,150,300,200,400,150,200,250,300,900],dtype=float)
    cost = np.array([
        [10,15,30,60,120,150,160,170,175,180,190,200],
        [70,80,40,10,50,80,90,100,110,120,130,150],
        [160,170,140,100,40,10,5,15,20,30,40,60],
        [200,210,180,140,80,50,40,50,30,15,10,20],
        [150,160,130,90,35,15,10,10,25,35,45,65],
    ],dtype=float)
    p = TransportationProblem("logistics","min",
        ["HN","DN","HCM","CT","BD"],
        ["HP","QN","Vinh","Hue","NT","VT","DNai","TN","LA","TG","AG","CM"],
        supply,demand,cost)
    r = lp_solver(p)
    assert r.total_cost == pytest.approx(120000.0)


def test_lp_unbalanced_supply_gt_demand():
    from src.core.transform import transform_problem
    p = TransportationProblem(
        "test","min",["A1","A2","A3"],["B1","B2","B3","B4"],
        np.array([100,160,140],dtype=float),
        np.array([80,70,100,90],dtype=float),
        np.array([[6,5,3,1],[9,7,5,8],[2,9,4,6]],dtype=float),
    )
    tp = transform_problem(p)
    r = lp_solver(tp)
    assert r.total_cost == pytest.approx(1160.0)


def test_lp_unbalanced_demand_gt_supply():
    from src.core.transform import transform_problem
    p = TransportationProblem(
        "test","min",["A1","A2"],["B1","B2","B3"],
        np.array([40,60],dtype=float),
        np.array([30,50,70],dtype=float),
        np.array([[2,3,5],[4,1,2]],dtype=float),
    )
    tp = transform_problem(p)
    r = lp_solver(tp)
    assert r.total_cost == pytest.approx(170.0)
