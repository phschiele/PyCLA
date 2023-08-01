from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from cvx.pycla.simplex import Simplex


@dataclass
class Frontier:
    mus: np.array
    sigmas: np.array
    name: str


def validate_and_fix_asymmetry(C: np.array, tol: float = 1e-8) -> np.array:
    max_sim_vio = max_symmetry_violation(C)
    assert max_sim_vio < tol
    return make_symmetric(C)


def make_symmetric(C: np.array) -> np.array:
    i_lower = np.tril_indices(C.shape[0], -1)
    C[i_lower] = C.T[i_lower]
    return C


def max_symmetry_violation(A: np.array) -> float:
    return np.max(np.abs(A - A.T))


def is_psd(A: np.array, tol: float = 1e-8) -> bool:
    eig = np.linalg.eigvalsh(A)
    return np.all(eig > -tol)


def transform_ineq_to_eq(
    mu: np.array,
    C: Optional[np.array],
    A: np.array,
    b: np.array,
    A_in: np.array,
    b_in: np.array,
    lb: np.array,
    ub: np.array,
) -> Tuple[np.array, Optional[np.array], np.array, np.array, np.array, np.array]:
    if A_in is None:
        return mu, C, A, b, lb, ub
    else:
        nineq = len(b_in)
        neq = len(b)
        ns = len(mu)

        mu_adj = np.append(mu, np.zeros(nineq))

        b_adj = np.append(b, b_in)
        lb_adj = np.append(lb, np.zeros(nineq))
        ub_adj = np.append(ub, np.ones(nineq) * np.inf)

        A_11 = A
        A_21 = A_in
        A_12 = np.zeros((neq, nineq))
        A_22 = np.identity(nineq)
        A_adj = np.block([[A_11, A_12], [A_21, A_22]])

        if C is not None:
            C_adj = np.zeros((ns + nineq, ns + nineq))
            C_adj[:ns, :ns] = C
        else:
            C_adj = None

        return mu_adj, C_adj, A_adj, b_adj, lb_adj, ub_adj


def find_max_E(mu: np.array, A: np.array, b: np.array, lb: np.array, ub: np.array, tol: float, method: str) -> Tuple[np.array, set, set, Optional[np.array]]:
    n = len(mu)
    m = A.shape[0]

    if method == "TWO_STAGE_SIMPLEX":
        simp = Simplex(A, b, lb, ub, mu, tol)
        X, altered_mus, in_vars, out_vars, Ai = simp.solve()
    elif method == "SCIPY":
        res = linprog(
            -mu,
            A_eq=A,
            b_eq=b,
            bounds=[(lower_bound, upper_bound) for lower_bound, upper_bound in zip(lb, ub)],
            method="revised simplex",
        )
        assert res.success
        X = res.x
        out_vars = set(np.logical_or((abs(X - ub) < tol), (abs(X - lb) < tol)).nonzero()[0])
        in_vars = set(range(n)) - out_vars
        Ai = None
    else:
        raise Exception(f"Unknown LP solve method {method}.")

    assert np.all(np.abs((A @ X) - b) < tol)
    assert np.all(np.logical_and(lb <= X + tol, X <= ub + tol))
    assert out_vars & in_vars == set()
    assert out_vars | in_vars == set(range(n))
    assert len(in_vars) == m

    return X, in_vars, out_vars, Ai
