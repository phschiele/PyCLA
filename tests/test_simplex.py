import numpy as np
import pytest
from cvx.pycla.simplex import Simplex


def test_lp1() -> None:
    mu = np.array([4.0, 5.0, 0.0, 0.0])
    A = np.array([[2.0, 1.0, 1.0, 0.0], [1.0, 2.0, 0.0, 1.0]])
    b = np.array([3.0, 3.0])
    lb = np.array([0.0, 0.0, 0.0, 0.0])
    ub = np.array([np.inf, np.inf, np.inf, np.inf])
    simp = Simplex(A, b, lb, ub, mu, tol=1e-8)
    X = simp.solve()[0]
    assert np.allclose(X, np.array([1, 1, 0, 0]))


def test_alter_mu() -> None:
    mu = np.array([5.0, 5.0, 5.0, 5.0])
    A = np.array([[1.0, 1.0, 1.0, 1.0]])
    b = np.array([1.0])
    lb = np.array([0.0, 0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0, 1.0])
    simp = Simplex(A, b, lb, ub, mu, tol=1e-8)
    X, altered_mus, _, _, _ = simp.solve()
    assert set(X) == {0, 1}
    assert set(altered_mus) == {5.0 + 1e-6, 5.0, 5.0 - 1e-6}


def test_unbounded() -> None:
    mu = np.array([4, 5, 0])
    A = np.array([[-2, -1, 1]])
    b = np.array([3])
    lb = np.array([0, 0, 0])
    ub = np.array([np.inf, np.inf, np.inf])
    simp = Simplex(A, b, lb, ub, mu, tol=1e-8)
    with pytest.raises(Exception, match="Unbounded E"):
        simp.solve()


def test_infeasible() -> None:
    mu = np.array([4, 5, 0])
    A = np.array([[-2, -1, -1]])
    b = np.array([3])
    lb = np.array([0, 0, 0])
    ub = np.array([np.inf, np.inf, np.inf])
    simp = Simplex(A, b, lb, ub, mu, tol=1e-8)
    with pytest.raises(Exception, match="Infeasible"):
        simp.solve()
