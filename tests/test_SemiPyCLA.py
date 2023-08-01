from __future__ import annotations

import numpy as np
import pytest as pytest

from cvx.pycla import SemiPyCLA
from cvx.pycla.validation import validate_semivariance_frontier


@pytest.mark.parametrize("lp_method", ["SCIPY", "TWO_STAGE_SIMPLEX"])
def test_markowitz_et_al(resource_dir, lp_method: str, update_tests: bool) -> None:
    # Example taken from Avoiding the Downside: A Practical Review of the Critical Line Algorithm for
    # Mean-Semivariance Portfolio Optimization (Markowitz et. al 2019)
    # https://www.hudsonbaycapital.com/documents/FG/hudsonbay/research/599440_paper.pdf

    historic_returns = np.loadtxt(resource_dir / "CLA_test_markowitz_et_al_data.csv", delimiter=",")
    mu = np.mean(historic_returns, axis=0)

    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([10.0, 10.0, 10.0])

    A = np.array([[1, 1, 1]])
    b = np.array([1])

    A_in = None
    b_in = None

    pycla = SemiPyCLA(historic_returns, mu, A, b, A_in, b_in, lb, ub, lp_method=lp_method)
    pycla.trace_frontier()
    validate_semivariance_frontier(pycla)

    weights = np.vstack([i["X"] for i in pycla.output])
    file_name = resource_dir / "CLA_test_markowitz_et_al_semi.csv"
    if update_tests:
        np.savetxt(file_name, weights, delimiter=",")
    expected_weights = np.loadtxt(file_name, delimiter=",")
    assert np.allclose(weights, expected_weights)
