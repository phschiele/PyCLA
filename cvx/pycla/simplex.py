from __future__ import annotations

import copy as cp
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from cvx.pycla.constants import Direction
from cvx.pycla.constants import SimplexPhase
from cvx.pycla.constants import SimplexReturn
from cvx.pycla.constants import VariableState


class Simplex:
    def __init__(self, A: np.array, b: np.array, lb: np.array, ub: np.array, mu: np.array, tol: float):
        self.m = A.shape[0]
        self.mu = mu
        self.tol = tol
        assert len(lb) == len(ub) == A.shape[1]
        self.n = len(lb)

        z = np.zeros(self.n + self.m)
        z[self.n :] = -1
        self.z = z
        self.price = np.zeros(self.m)
        self.profit = np.zeros(self.n)
        self.adj_rate = np.zeros(self.m)

        state = np.zeros(self.n + self.m)
        state[: self.n] = VariableState.LOW
        state[self.n :] = VariableState.IN
        self.State = state

        self.out_vars = set(range(self.n))
        self.in_vars = set(range(self.n, self.n + self.m))

        A_lb = A @ lb
        B = np.zeros((self.m, self.m))
        np.fill_diagonal(B, np.where(b >= A_lb, 1, -1))
        self.A = np.hstack([A, B])
        self.lb = np.hstack([lb, np.zeros(self.m)])
        self.ub = np.hstack([ub, np.ones(self.m) * np.inf])
        Y = np.abs(b - A_lb)
        X = np.zeros(self.n)
        X[: self.n] = lb[: self.n]
        self.X = np.hstack([X, Y])

        self.Ai = cp.deepcopy(B)
        self.nIABV = self.m

    def solve(self) -> Tuple[np.array, np.array, set, set, np.array]:
        return_code = self.simplex_phase(SimplexPhase.ONE)

        if return_code == SimplexReturn.OPTIMAL:
            assert self.nIABV == 0
            self.lb = self.lb[: self.n]
            self.ub = self.ub[: self.n]
            self.A = self.A[:, : self.n]
            self.X = self.X[: self.n]
            self.State = self.State[: self.n]

        elif return_code == SimplexReturn.DEGENERATE:
            # Potentially recover
            raise NotImplementedError

        if return_code == SimplexReturn.OPTIMAL:
            self.z[: self.n] = self.mu
            return_code = self.simplex_phase(SimplexPhase.TWO)

            if return_code == SimplexReturn.OPTIMAL:
                return self.X, self.alter_mus(), self.in_vars, self.out_vars, self.Ai

        if return_code == SimplexReturn.OPTIMAL:
            raise Exception("Optimal state should have returned the function")
        elif return_code == return_code.DEGENERATE:
            raise Exception("Degenerate problem, should have been handled above.")
        else:
            raise Exception(f"Unknown return code: {return_code}")

    def simplex_phase(self, phase: SimplexPhase) -> SimplexReturn:
        while True:
            self.price = -self.Ai.T @ self.z[self._I]

            self.profit[self._O] = (self.z[self._O] + self.price @ self.A[:, self._O]) * np.where(self.is_up(self._O), -1, 1)
            out_max = np.argmax(self.profit[self._O])
            j_in = self._O[out_max]
            profit_max = self.profit[j_in]

            if profit_max < self.tol:
                if phase == SimplexPhase.ONE:
                    if np.any(self.X[self._I[-self.nIABV :]] > self.tol):
                        raise Exception("Infeasible problem. Check constraints.")
                    else:
                        return SimplexReturn.DEGENERATE
                else:
                    return SimplexReturn.OPTIMAL

            if self.is_up(j_in):
                in_direction = Direction.LOWER
                adj_sign = 1
            else:
                in_direction = Direction.HIGHER
                adj_sign = -1

            self.adj_rate = adj_sign * self.Ai @ self.A[:, j_in]

            theta = self.ub[j_in] - self.lb[j_in]

            theta_candidate = np.ones(self.m) * np.inf
            ind_adj_rate_low = np.where(self.adj_rate < -self.tol)[0]
            ind_adj_rate_high = np.where(self.adj_rate > self.tol)[0]
            theta_candidate[ind_adj_rate_low] = (self.lb[self._I][ind_adj_rate_low] - self.X[self._I][ind_adj_rate_low]) / self.adj_rate[ind_adj_rate_low]
            theta_candidate[ind_adj_rate_high] = (self.ub[self._I][ind_adj_rate_high] - self.X[self._I][ind_adj_rate_high]) / self.adj_rate[ind_adj_rate_high]
            if np.min(theta_candidate) < theta:
                i_out = np.argmin(theta_candidate)
                j_out = self._I[i_out]
                theta = theta_candidate[i_out]
                out_direction = Direction.LOWER if self.adj_rate[i_out] < 0 else Direction.HIGHER
            else:
                i_out = None
                j_out = j_in
                out_direction = in_direction

            if theta == np.inf:
                raise Exception("Unbounded E. Make sure you have a valid budget constraint.")

            self.X[self._I] += theta * self.adj_rate
            self.X[j_in] = self.X[j_in] + theta if in_direction == Direction.HIGHER else self.X[j_in] - theta

            self.go_in(j_in)
            self.go_out(j_out, out_direction)

            if j_in != j_out:
                i_in = self._I.index(j_in)
                self.update_ai(i_out, i_in, in_direction)  # type: ignore

            if phase == SimplexPhase.ONE and j_out >= self.n:
                self.nIABV = self.nIABV - 1
                if self.nIABV == 0:
                    return SimplexReturn.OPTIMAL

    @property
    def _I(self) -> List[int]:
        return sorted(self.in_vars)

    @property
    def _O(self) -> List[int]:
        return sorted(self.out_vars)

    def is_up(self, j: Union[int, List[int]]) -> Union[bool, np.array]:
        return self.State[j] == VariableState.UP

    def is_low(self, j: Union[int, List[int]]) -> Union[bool, np.array]:
        return self.State[j] == VariableState.LOW

    def go_in(self, j_in: int) -> None:
        self.out_vars.remove(j_in)
        self.in_vars.add(j_in)
        self.State[j_in] = VariableState.IN

    def go_out(self, j_out: int, out_direction: Direction) -> None:
        self.in_vars.remove(j_out)
        if j_out < self.n:
            self.out_vars.add(j_out)

        self.State[j_out] = VariableState.UP if out_direction == Direction.HIGHER else VariableState.LOW

    def update_ai(self, i_out: int, i_in: int, in_direction: Direction) -> None:
        ind_minus_i_out = list(range(self.m))
        ind_minus_i_out.remove(i_out)
        temp = self.adj_rate[ind_minus_i_out] / self.adj_rate[i_out]
        self.Ai[ind_minus_i_out, :] -= np.outer(temp, self.Ai[i_out, :])

        adj_rate_i_out_signed = -self.adj_rate[i_out] if in_direction == Direction.HIGHER else self.adj_rate[i_out]
        self.Ai[i_out, :] = self.Ai[i_out, :] / adj_rate_i_out_signed

        self.reorder_ai_rows(i_out, i_in)

    def reorder_ai_rows(self, del_row: int, add_row: int) -> None:
        rows = list(range(self.m))
        rows.pop(del_row)
        rows.insert(add_row, del_row)
        self.Ai = self.Ai[rows, :]

    def alter_mus(self, eps: float = 1e-6) -> np.array:
        altered_mus = self.mu.copy()
        nonneg_profit = self.profit[self._O] > -eps
        mu_increase = np.where(np.logical_and(nonneg_profit, self.is_up(self._O)), eps, 0)
        mu_decrease = np.where(np.logical_and(nonneg_profit, self.is_low(self._O)), -eps, 0)
        altered_mus[self._O] += mu_increase + mu_decrease
        return altered_mus
