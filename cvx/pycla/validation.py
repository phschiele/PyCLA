from __future__ import annotations

from typing import Union

import cvxpy as cvx
import numpy as np

from cvx.pycla.pycla import PyCLA
from cvx.pycla.pycla import SemiPyCLA


def validate_frontier(pycla: PyCLA) -> None:
    validate_max_return(pycla)
    validate_min_volatility(pycla)


def validate_semivariance_frontier(semipycla: SemiPyCLA) -> None:
    validate_max_return(semipycla)
    validate_min_semideviation(semipycla)


def validate_max_return(pycla: Union[PyCLA, SemiPyCLA]) -> None:
    mu = pycla.mu
    ub = pycla.ub
    ub[ub == np.inf] = 1e30
    lb = pycla.lb
    A = pycla.A
    b = pycla.b

    x = cvx.Variable(pycla.n)
    max_mu = cvx.Problem(cvx.Maximize(mu @ x), [lb <= x, x <= ub, A @ x == b])
    max_mu.solve(solver=cvx.OSQP)
    assert max_mu.status == "optimal"

    assert np.allclose(max_mu.value, pycla.output[0]["E"], atol=1e-4), max_mu.value - pycla.output[0]["E"]


def validate_min_volatility(pycla: PyCLA) -> None:
    mu = pycla.mu
    C = pycla.C
    ub = pycla.ub
    ub[ub == np.inf] = 1e30
    lb = pycla.lb
    A = pycla.A
    b = pycla.b

    x = cvx.Variable(pycla.n)
    MVP = cvx.Problem(cvx.Minimize(cvx.QuadForm(x, C)), [lb <= x, x <= ub, A @ x == b])
    MVP.solve(solver=cvx.OSQP)
    assert MVP.status == "optimal"

    assert np.allclose(x.value @ mu, pycla.output[-1]["E"], atol=1e-4), x.value @ mu - pycla.output[-1]["E"]
    assert np.allclose(np.sqrt(MVP.value), pycla.output[-1]["std"], atol=1e-4), np.sqrt(MVP.value) - pycla.output[-1]["std"]


def validate_min_semideviation(pycla: SemiPyCLA) -> None:
    # MTXY method

    mu = pycla.mu
    ub = pycla.ub
    ub[ub == np.inf] = 1e30
    lb = pycla.lb
    A = pycla.A
    b = pycla.b

    assert pycla.traced_frontier
    x = cvx.Variable(pycla.n)
    p = cvx.Variable(pycla.t, nonneg=True)
    n = cvx.Variable(pycla.t, nonneg=True)
    objective = cvx.Minimize(cvx.sum(cvx.square(n)))

    B = (pycla.excess_returns) / np.sqrt(pycla.t)
    constraints = [B @ x - p + n == 0, lb <= x, x <= ub, A @ x == b]

    MSP = cvx.Problem(objective, constraints)
    MSP.solve(solver=cvx.OSQP)
    assert MSP.status == "optimal"

    assert np.allclose(x.value @ mu, pycla.output[-1]["E"], atol=1e-4), x.value @ mu - pycla.output[-1]["E"]
    assert np.allclose(np.sqrt(MSP.value), pycla.output[-1]["semideviation"], atol=1e-4), np.sqrt(MSP.value) - pycla.output[-1]["semideviation"]
