import numpy as np
import pytest
from pycla import PyCLA
from pycla.helpers import Frontier, make_symmetric
from pycla.plotting import plot_efficient_frontiers
from pycla.validation import validate_frontier
from sklearn.datasets import make_spd_matrix


@pytest.mark.parametrize("lp_method", ["SCIPY", "TWO_STAGE_SIMPLEX"])
def test_sharpe(resource_dir, lp_method: str, update_tests: bool) -> None:
    # Example taken from http://web.stanford.edu/~wfsharpe/mia/opt/mia_opt3.htm

    n_sec = 3

    mu = np.array([2.8000, 6.3000, 10.8000])
    cc = np.array([[1.0000, 0.4000, 0.1500], [0.4000, 1.0000, 0.3500], [0.1500, 0.3500, 1.0000]])
    sd = np.array([1.0000, 7.4000, 15.4000]).reshape([n_sec, 1])
    C = (sd @ sd.T) * cc

    lb = 0.2 * np.ones(n_sec)
    ub = 0.5 * np.ones(n_sec)

    A = np.ones([1, n_sec])
    b = np.array([1.0])

    A_in = None
    b_in = None

    pycla = PyCLA(mu, C, A, b, A_in, b_in, lb, ub, lp_method=lp_method)
    pycla.trace_frontier()
    validate_frontier(pycla)

    weights = np.vstack([i["X"] for i in pycla.output])
    file_name = resource_dir / "CLA_test_sharpe.csv"
    if update_tests:
        np.savetxt(file_name, weights, delimiter=",")
    expected_weights = np.loadtxt(file_name, delimiter=",")

    mus_cla = [i["E"] for i in pycla.output]
    sigmas_cla = [i["std"] for i in pycla.output]

    cla_frontier = Frontier(mus_cla, sigmas_cla, "cla")

    sharpe_weights = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.2, 0.5, 0.3],
            [0.2, 0.5, 0.3],
            [0.2218, 0.5, 0.2782],
            [0.4519, 0.3481, 0.2],
            [0.5, 0.3, 0.2],
        ]
    )

    sharpe_mus = pycla.mu @ sharpe_weights.T
    sharpe_sigmas = [np.sqrt(sharpe_weights[i, :] @ pycla.C @ sharpe_weights[i, :].T) for i in range(sharpe_weights.shape[0])]
    sharpe_frontier = Frontier(sharpe_mus, sharpe_sigmas, "sharpe")

    plot_frontier = False
    if plot_frontier:
        plot_efficient_frontiers([cla_frontier, sharpe_frontier])

    assert np.allclose(weights, expected_weights)
    assert np.allclose(weights[:-1, :], sharpe_weights, atol=1e-4)


@pytest.mark.parametrize("lp_method", ["SCIPY", "TWO_STAGE_SIMPLEX"])
def test_many_constraints(resource_dir, lp_method: str, update_tests: bool) -> None:
    mu = np.loadtxt(resource_dir / "CLA_test_many_constraints_mu.csv", delimiter=",")
    C = np.loadtxt(resource_dir / "CLA_test_many_constraints_C.csv", delimiter=",")
    n_sec = len(mu)

    lb = np.zeros(n_sec)
    ub = np.loadtxt(resource_dir / "CLA_test_many_constraints_ub.csv", delimiter=",")

    A = np.ones([1, n_sec])
    b = np.array([1.0])

    A_in = np.loadtxt(resource_dir / "CLA_test_many_constraints_A_in.csv", delimiter=",")
    b_in = np.loadtxt(resource_dir / "CLA_test_many_constraints_b_in.csv", delimiter=",")

    if lp_method == "SCIPY":
        m = 12
        pycla = PyCLA(mu, C, A, b, A_in[:m], b_in[:m], lb, ub, lp_method=lp_method)

    elif lp_method == "TWO_STAGE_SIMPLEX":
        pycla = PyCLA(mu, C, A, b, A_in, b_in, lb, ub, lp_method=lp_method)
    else:
        raise Exception(f"Unknown LP method {lp_method}.")

    pycla.trace_frontier()
    validate_frontier(pycla)
    weights = np.vstack([i["X"] for i in pycla.output])
    file_name = resource_dir / f"CLA_test_many_constraints_{lp_method}.csv"
    if update_tests:
        np.savetxt(file_name, weights, delimiter=",")
    expected_weights = np.loadtxt(file_name, delimiter=",")

    assert np.allclose(weights, expected_weights)


@pytest.mark.parametrize("lp_method", ["SCIPY", "TWO_STAGE_SIMPLEX"])
def test_markowitz_todd(resource_dir, lp_method: str, update_tests: bool) -> None:
    # Example taken from Mean-Variance Analysis in Portfolio Choice and Capital Markets (Markowitz and Todd 2000)

    mu = np.array([1.175, 1.19, 0.396, 1.12, 0.346, 0.679, 0.089, 0.73, 0.481, 1.08])
    C = np.loadtxt(resource_dir / "CLA_test_markowitz_todd_C.csv", delimiter=",")

    lb = np.array([0.0999, 0, 0, 0, 0.0999, 0, 0, 0, 0, 0])
    ub = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

    A = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    b = np.array([1])

    A_in = np.array([[-1, -1, -1, -0.5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0.5, 1, 1, 1, 0, 0, 0]])
    b_in = np.array([-0.2, 0.5])

    pycla = PyCLA(mu, C, A, b, A_in, b_in, lb, ub, lp_method=lp_method)
    pycla.trace_frontier()
    validate_frontier(pycla)

    weights = np.vstack([i["X"] for i in pycla.output])
    file_name = resource_dir / "CLA_test_markowitz_todd.csv"
    if update_tests:
        np.savetxt(file_name, weights, delimiter=",")
    expected_weights = np.loadtxt(file_name, delimiter=",")
    assert np.allclose(weights, expected_weights)


@pytest.mark.parametrize("lp_method", ["SCIPY", "TWO_STAGE_SIMPLEX"])
def test_random(resource_dir, lp_method: str, update_tests: bool) -> None:
    for n_sec in range(10, 101, 10):
        np.random.seed(n_sec)
        mu = np.random.random(n_sec)
        C = make_spd_matrix(n_sec, random_state=n_sec)
        C = make_symmetric(C)

        lb = np.zeros(n_sec)
        ub = np.ones(n_sec)

        A = np.ones((1, n_sec))
        b = np.array([1])

        A_in = None
        b_in = None

        ub[np.argmax(mu)] -= 0.000001  # Required for Scipy

        pycla = PyCLA(mu, C, A, b, A_in, b_in, lb, ub, lp_method=lp_method)
        pycla.trace_frontier()
        validate_frontier(pycla)

        weights = np.vstack([i["X"] for i in pycla.output])
        file_name = resource_dir / "CLA_test_random_{n_sec}.csv"
        if update_tests:
            np.savetxt(file_name, weights, delimiter=",")
        expected_weights = np.loadtxt(resource_dir / f"CLA_test_random_{n_sec}.csv", delimiter=",")
        assert np.allclose(weights, expected_weights)


@pytest.mark.parametrize("lp_method", ["SCIPY", "TWO_STAGE_SIMPLEX"])
def test_markowitz_et_al(resource_dir, lp_method: str, update_tests: bool) -> None:
    # Example taken from Avoiding the Downside: A Practical Review of the Critical Line Algorithm for
    # Mean-Semivariance Portfolio Optimization (Markowitz et. al 2019)
    # https://www.hudsonbaycapital.com/documents/FG/hudsonbay/research/599440_paper.pdf

    historic_returns = np.loadtxt(resource_dir / "CLA_test_markowitz_et_al_data.csv", delimiter=",")
    mu = np.mean(historic_returns, axis=0)
    C = np.cov(historic_returns.T)

    lb = np.array([0.1, 0.1, 0.1])
    ub = np.array([0.5, 0.5, 0.5])

    A = np.array([[1, 1, 1]])
    b = np.array([1])

    A_in = None
    b_in = None

    pycla = PyCLA(mu, C, A, b, A_in, b_in, lb, ub, lp_method=lp_method)
    pycla.trace_frontier()
    validate_frontier(pycla)

    weights = np.vstack([i["X"] for i in pycla.output])
    file_name = resource_dir / "CLA_test_markowitz_et_al.csv"
    if update_tests:
        np.savetxt(file_name, weights, delimiter=",")
    expected_weights = np.loadtxt(file_name, delimiter=",")
    assert np.allclose(weights, expected_weights)
