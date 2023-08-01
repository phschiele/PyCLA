from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from pycla.constants import Direction, VariableState
from pycla.helpers import find_max_E, is_psd, transform_ineq_to_eq, validate_and_fix_asymmetry


class PyCLABase(ABC):
    def __init__(
        self,
        mu: np.array,
        A: np.array,
        b: np.array,
        lb: np.array,
        ub: np.array,
        n_sec: int,
        n: int,
        m: int,
        X: np.array,
        in_vars: set,
        out_vars: set,
        tol: float,
        MAX_CP: int,
        verbose: bool,
    ) -> None:

        self.mu = mu
        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub
        self.n_sec = n_sec
        self.n = n
        self.m = m
        self.tol = tol

        self.X = X
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.State = self.get_initial_state()

        self.MAX_CP = MAX_CP
        self.LambdaE = np.inf
        self.output: List[dict] = []
        self.verbose = verbose
        self.traced_frontier = False

    def go_in(self, j_in: int) -> None:
        self.out_vars.remove(j_in)
        self.in_vars.add(j_in)
        self.State[j_in] = VariableState.IN

    def go_out(self, j_out: int, out_direction: Direction) -> None:
        assert j_out < self.n
        self.in_vars.remove(j_out)
        self.out_vars.add(j_out)
        self.State[j_out] = VariableState.UP if out_direction == Direction.HIGHER else VariableState.LOW

    @property
    def _I(self) -> List[int]:
        return sorted(self.in_vars)

    @property
    def _O(self) -> List[int]:
        return sorted(self.out_vars)

    @property
    def _in_sec(self) -> List[int]:
        return [i for i in self._I if i < self.n]

    def get_initial_state(self) -> np.array:
        state = np.zeros(self.n)
        state[self._in_sec] = VariableState.IN
        up_diff = np.abs(self.X - self.ub)
        state[self._O] = np.where(up_diff[self._O] < self.tol, VariableState.UP, VariableState.LOW)
        return state

    def trace_frontier(self) -> None:
        for iteration in range(self.MAX_CP):
            try:
                self.iteration()
                if self.verbose:
                    print(iteration, self.LambdaE)
            except StopIteration:
                self.traced_frontier = True
                break
            if iteration == self.MAX_CP - 1:
                raise Exception(f"Frontier not fully traced after {self.MAX_CP} iterations.")

    @abstractmethod
    def iteration(self) -> None:
        pass


class PyCLA(PyCLABase):
    def __init__(
        self,
        mu: np.array,
        C: np.array,
        A: np.array,
        b: np.array,
        A_in: np.array,
        b_in: np.array,
        lb: np.array,
        ub: np.array,
        tol: float = 1e-8,
        lp_method: str = "TWO_STAGE_SIMPLEX",
        validate_inputs: bool = True,
        verbose: bool = False,
    ):

        if validate_inputs:
            C = validate_and_fix_asymmetry(C, tol)
            self.validate_inputs(mu, C, A, b, A_in, b_in, lb, ub, tol)

        n_sec = len(mu)
        mu, C, A, b, lb, ub = transform_ineq_to_eq(mu, C, A, b, A_in, b_in, lb, ub)
        self.C = C

        n = len(mu)
        m = A.shape[0]

        X, in_vars, out_vars, Ai = find_max_E(mu, A, b, lb, ub, tol, lp_method)
        in_vars = in_vars | set(range(n, n + m))
        out_vars = out_vars

        self.V = 0
        MAX_CP = 5 * (n + m)

        super().__init__(mu, A, b, lb, ub, n_sec, n, m, X, in_vars, out_vars, tol, MAX_CP, verbose)

        if Ai is None:
            Ai = np.linalg.inv(self.A[:, self._in_sec])

        self._alpha = np.zeros(n + m)
        self._beta = np.zeros(n + m)
        self._bbar = np.zeros(n + m)
        self._Mi = np.zeros((n + m, n + m))

        self._MMat = np.block([[self.C, self.A.T], [self.A, np.zeros((m, m))]])
        self._alpha[self._O] = self.X[self._O]
        self._bbar[self._I] = np.append(np.zeros(len(self.in_vars) - m), self.b) - (self._M_io @ self._X_o)

        C_ii = self.C[np.ix_(self._in_sec, self._in_sec)]
        Mi_small = np.block([[np.zeros((m, m)), Ai], [Ai.T, -Ai.T @ C_ii @ Ai]])
        self._Mi[np.ix_(self._I, self._I)] = Mi_small

    def iteration(self) -> None:
        in_vars = self._I
        out_vars = self._O

        Mi_ii = self._Mi_ii
        self._alpha[in_vars] = Mi_ii @ self._bbar[in_vars]
        mu_i_zero_extended = np.append(self.mu[self._in_sec], np.zeros(len(self.in_vars) - len(self._in_sec)))
        self._beta[in_vars] = Mi_ii @ mu_i_zero_extended

        lambda_in_goes_out = -np.ones(self.n) * np.inf

        lower_idx = np.where(self._beta[: self.n] > self.tol)[0]
        lambda_in_goes_out[lower_idx] = (self.lb[lower_idx] - self._alpha[lower_idx]) / self._beta[lower_idx]
        higher_idx = np.where(self._beta[: self.n] < -self.tol)[0]
        lambda_in_goes_out[higher_idx] = (self.ub[higher_idx] - self._alpha[higher_idx]) / self._beta[higher_idx]

        j_in_goes_out = int(np.argmax(lambda_in_goes_out))
        lambdaA = lambda_in_goes_out[j_in_goes_out]
        out_direction = Direction.LOWER if self._beta[j_in_goes_out] > 0 else Direction.HIGHER

        gamma = np.zeros(self.n)
        delta = np.zeros(self.n)
        lambda_out_goes_in = -np.ones(self.n) * np.inf

        MMat_o = self._MMat[out_vars, :]

        gamma[out_vars] = MMat_o @ self._alpha
        delta[out_vars] = MMat_o @ self._beta - self.mu[out_vars]

        delta[np.where(np.logical_and(self.State == VariableState.UP, delta > 0))] = 0
        delta[np.where(np.logical_and(self.State == VariableState.LOW, delta < 0))] = 0
        idx = np.where(np.abs(delta) > self.tol)
        lambda_out_goes_in[idx] = -gamma[idx] / delta[idx]

        j_out_goes_in = int(np.argmax(lambda_out_goes_in))
        lambdaB = lambda_out_goes_in[j_out_goes_in]

        new_LambdaE = max([lambdaA, lambdaB, 0])
        if self.LambdaE < np.inf:
            assert new_LambdaE - self.LambdaE < self.tol, f"Difference: {new_LambdaE - self.LambdaE}"
        old_lambda_E = self.LambdaE
        self.LambdaE = new_LambdaE
        self.calc_corner_pf(old_lambda_E)

        if self.LambdaE < self.tol:
            raise StopIteration

        if lambdaA > lambdaB:
            self.delete_var(j_in_goes_out, out_direction)
        else:
            self.add_var(j_out_goes_in)

    def calc_corner_pf(self, old_lambda_E: float) -> None:
        in_sec = self._in_sec
        if old_lambda_E == np.inf:
            assert np.allclose(self.X[in_sec], self._alpha[in_sec] + self._beta[in_sec] * self.LambdaE)
        self.X[in_sec] = self._alpha[in_sec] + self._beta[in_sec] * self.LambdaE
        dE_dLambda = self._beta[in_sec] @ self.mu[in_sec]

        if dE_dLambda < self.tol:
            self.E = self.mu @ self.X
            self.V = self.X @ self.C @ self.X

        else:
            assert old_lambda_E < np.inf
            a2 = 1 / dE_dLambda
            a1 = 2 * (old_lambda_E - a2 * self.E)
            a0 = self.V - a1 * self.E - a2 * self.E ** 2

            self.E += (self.LambdaE - old_lambda_E) * dE_dLambda
            self.V = a0 + a1 * self.E + a2 * self.E ** 2

        self.output.append(
            {
                "E": self.E,
                "std": np.sqrt(self.V),
                "X": self.X.copy(),
            }
        )

    def add_var(self, jAdd: int) -> None:
        Mi_ii = self._Mi_ii
        in_vars = self._I

        M_ij = self._MMat[np.ix_(in_vars, [jAdd])]
        M_jj = self._MMat[jAdd, jAdd]

        xi = Mi_ii @ M_ij
        xi_j_squared = M_jj - xi.T @ M_ij

        off_diag = (-xi / xi_j_squared).flatten()

        self._Mi[np.ix_(in_vars, in_vars)] = Mi_ii + np.outer(xi, xi) / xi_j_squared
        self._Mi[jAdd, jAdd] = 1 / xi_j_squared
        self._Mi[in_vars, jAdd] = off_diag
        self._Mi[jAdd, in_vars] = off_diag

        self._bbar[in_vars] += self._MMat[in_vars, jAdd] * self.X[jAdd]

        self.go_in(jAdd)
        self._bbar[jAdd] = -np.sum(self._MMat[jAdd, self._O] * self.X[self._O])

    def delete_var(self, jDel: int, out_direction: Direction) -> None:
        self._alpha[jDel] = self.X[jDel]
        self._beta[jDel] = 0

        self.go_out(jDel, out_direction)

        in_vars = self._I
        self._Mi[np.ix_(in_vars, in_vars)] -= (self._Mi[in_vars, jDel].reshape(-1, 1) @ self._Mi[jDel, in_vars].reshape(1, -1)) / self._Mi[jDel, jDel]
        self._bbar[in_vars] -= self._MMat[in_vars, jDel] * self.X[jDel]

    @property
    def _M_io(self) -> np.array:
        return self._MMat[np.ix_(self._I, self._O)]

    @property
    def _X_o(self) -> np.array:
        return self.X[self._O]

    @property
    def _Mi_ii(self) -> np.array:
        return self._Mi[np.ix_(self._I, self._I)]

    @staticmethod
    def validate_inputs(
        mu: np.array,
        C: np.array,
        A: np.array,
        b: np.array,
        A_in: Optional[np.array],
        b_in: Optional[np.array],
        lb: np.array,
        ub: np.array,
        tol: float,
    ) -> None:
        assert all(isinstance(arg, np.ndarray) for arg in [mu, C, A, b, lb, ub])
        assert mu.ndim == b.ndim == lb.ndim == ub.ndim == 1

        assert is_psd(C, tol)

        if A_in is not None or b_in is not None:
            assert A_in is not None and b_in is not None
            assert isinstance(A_in, np.ndarray)
            assert isinstance(b_in, np.ndarray)
            assert A_in.shape[0] == b_in.shape[0]
            assert b_in.ndim == 1

        assert A.shape[0] == b.shape[0]
        assert mu.shape[0] == C.shape[0] == C.shape[1] == lb.shape[0] == ub.shape[0]

        assert np.all(ub - lb > tol)


class SemiPyCLA(PyCLABase):
    def __init__(
        self,
        historic_returns: np.array,
        mu: np.array,
        A: np.array,
        b: np.array,
        A_in: np.array,
        b_in: np.array,
        lb: np.array,
        ub: np.array,
        reference_return: float = 0.0,
        tol: float = 1e-8,
        lp_method: str = "TWO_STAGE_SIMPLEX",
        verbose: bool = True,
    ):

        self.historic_returns = historic_returns
        self.reference_return = reference_return
        self.t = historic_returns.shape[0]

        n_sec = len(mu)
        mu, _, A, b, lb, ub = transform_ineq_to_eq(mu, None, A, b, A_in, b_in, lb, ub)
        n = len(mu)
        m = A.shape[0]

        X, in_vars, out_vars, Ai = find_max_E(mu, A, b, lb, ub, tol, lp_method)

        self.excess_returns = historic_returns - reference_return
        y = self.excess_returns @ X
        in_obs = y < tol
        assert np.any(in_obs)

        MAX_CP = 5 * (n + m + self.t)

        super().__init__(mu, A, b, lb, ub, n_sec, n, m, X, in_vars, out_vars, tol, MAX_CP, verbose)

        self._local_semicovariance = self.excess_returns[in_obs].T @ self.excess_returns[in_obs] / self.t
        self._Ploc = np.hstack([self._local_semicovariance, self.A.T])

    def iteration(self) -> None:
        old_lambda_E = self.LambdaE
        Sbar = self.bar(self._local_semicovariance, self._O)
        Abar = self.zero_cols(self.A, self._O)
        Mbar = np.block([[Sbar, Abar.T], [Abar, np.zeros((self.m, self.m))]])
        k = np.zeros(self.n)
        k[self.State == VariableState.UP] = self.ub[self.State == VariableState.UP]
        k[self.State == VariableState.LOW] = self.lb[self.State == VariableState.LOW]
        rhsa = np.zeros(self.m + self.n)
        rhsa[-self.m :] = self.b - self.A @ k
        rhsb = np.zeros(self.m + self.n)
        rhsb[: self.n] = self.zero_rows(self.mu, self._O)

        alpha = np.linalg.solve(Mbar, rhsa)
        beta = np.linalg.solve(Mbar, rhsb)
        gamma = self._Ploc @ alpha
        delta = self._Ploc @ beta - self.mu

        lambda_in_goes_out = -np.ones(self.n) * np.inf
        lower_idx = np.where(np.logical_and(beta[: self.n] > self.tol, self.State == VariableState.IN))[0]
        lambda_in_goes_out[lower_idx] = (self.lb[lower_idx] - alpha[lower_idx]) / beta[lower_idx]
        higher_idx = np.where(np.logical_and(beta[: self.n] < -self.tol, self.State == VariableState.IN))[0]
        lambda_in_goes_out[higher_idx] = (self.ub[higher_idx] - alpha[higher_idx]) / beta[higher_idx]

        j_in_goes_out = int(np.argmax(lambda_in_goes_out))
        lambdaA = lambda_in_goes_out[j_in_goes_out]
        out_direction = Direction.LOWER if beta[j_in_goes_out] > 0 else Direction.HIGHER

        lambda_out_goes_in = -np.ones(self.n) * np.inf
        delta[np.where(np.logical_and(self.State == VariableState.UP, delta > 0))] = 0
        delta[np.where(np.logical_and(self.State == VariableState.LOW, delta < 0))] = 0
        idx = np.where(np.logical_and(~(self.State == VariableState.IN), np.abs(delta) > self.tol))
        lambda_out_goes_in[idx] = -gamma[idx] / delta[idx]

        j_out_goes_in = int(np.argmax(lambda_out_goes_in))
        lambdaB = lambda_out_goes_in[j_out_goes_in]

        lambda_sec = max([lambdaA, lambdaB, 0])

        # Check if local semi-variance matrix needs to change
        numerator = self.excess_returns @ alpha[: self.n]
        denominator = self.excess_returns @ beta[: self.n]
        rat = -np.ones(self.t) * np.inf
        valid = np.abs(denominator) > self.tol
        rat[valid] = -numerator[valid] / denominator[valid]
        lambda_obs = -np.inf
        i = rat - old_lambda_E < -self.tol
        if np.any(i):
            lambda_obs = np.max(rat[i])

        if lambda_obs > lambda_sec:
            self.LambdaE = lambda_obs
            self.X = alpha[: self.n] + self.LambdaE * beta[: self.n]
            y = self.excess_returns @ self.X
            in_obs = y < -self.tol
            assert np.any(in_obs)
            self._local_semicovariance = self.excess_returns[in_obs].T @ self.excess_returns[in_obs] / self.t
            self._Ploc = np.hstack([self._local_semicovariance, self.A.T])

        else:
            self.LambdaE = lambda_sec
            self.X = alpha[: self.n] + self.LambdaE * beta[: self.n]

            if lambdaA > lambdaB:
                print("in goes out", j_in_goes_out, out_direction)
                self.go_out(j_in_goes_out, out_direction)
            else:
                print("out goes in", j_out_goes_in)
                self.go_in(j_out_goes_in)

        self.output.append(
            {
                "E": self.X @ self.mu,
                "semideviation": np.sqrt(self.X @ self._local_semicovariance @ self.X),
                "X": self.X.copy(),
            }
        )

        if self.LambdaE < self.tol:
            raise StopIteration

    @staticmethod
    def zero_rows(x: np.array, j: List[int]) -> np.array:
        y = x.copy()
        for k in j:
            y[k] = 0
        return y

    @staticmethod
    def zero_cols(x: np.array, j: List[int]) -> np.array:
        y = x.copy()
        for k in j:
            y[:, k] = 0
        return y

    @staticmethod
    def bar(x: np.array, j: List[int]) -> np.array:
        y = x.copy()
        for k in j:
            y[k] = 0
            y[k, k] = 1
        return y
