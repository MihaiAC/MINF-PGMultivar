import sys
import sympy
import cvxopt
import qpsolvers
import numpy as np
import scipy
from typing import Optional
from functools import partial


class ProxOperator:
    def __init__(self):
        pass

    def prox(self, objective: np.ndarray, reg_parameter: float, node: int, data_points: np.ndarray,
             keep_diag_zero: bool) -> np.ndarray:
        pass


class SoftThreshold(ProxOperator):
    def __init__(self):
        super().__init__()
        self.constraint_solver = 'soft'

    def prox(self, objective: np.ndarray, reg_parameter: float, node: int, data_points: np.ndarray,
             keep_diag_zero: bool) -> np.ndarray:
        obj_plus = objective - reg_parameter
        obj_minus = objective + reg_parameter
        return obj_plus * (obj_plus > 0) + obj_minus * (obj_minus < 0)

class QPGM_SoftThreshold(ProxOperator):
    def __init__(self, qpgm_qtc):
        super().__init__()
        self.qpgm_qtc = qpgm_qtc
        self.constraint_solver = 'qpgm_soft'

    def prox(self, objective:np.ndarray, reg_parameter:float, node:int, data_points:np.ndarray,
             keep_diag_zero:bool) -> np.ndarray:
        quad_param = objective[-1]
        obj_plus = objective - reg_parameter
        obj_minus = objective + reg_parameter
        soft_thresholded_obj = obj_plus * (obj_plus > 0) + obj_minus * (obj_minus < 0)
        if quad_param > -1/self.qpgm_qtc:
            soft_thresholded_obj[-1] = -1/self.qpgm_qtc
        else:
            soft_thresholded_obj[-1] = quad_param
        return soft_thresholded_obj

class QuadProgOperator(ProxOperator):
    def __init__(self, constraint_solver: str):
        super().__init__()
        self.constraint_solver = constraint_solver

    @staticmethod
    def remove_dependent_rows(A: np.ndarray) -> np.ndarray:
        _, ind_rows = sympy.Matrix(A).T.rref()
        B = A[list(ind_rows)]
        return B

    def prox(self, objective: np.ndarray, reg_parameter: float, node: int, data_points: np.ndarray,
             keep_diag_zero: bool) -> np.ndarray:
        n = len(objective)

        v = objective.reshape((n, 1))
        alpha = reg_parameter
        q = alpha * np.ones((2 * n, 1)) - np.vstack([v, -v])
        q = q.reshape((2*n, ))

        # The order of operations is: C -> -C -> (C, -C) -> add one row to it, ensuring positivity of beta aka a
        # row of -1s; d is 2n x 1 of 0s.
        C = np.copy(data_points)
        if keep_diag_zero:
            C[:, node] = 0
        else:
            C[:, node] = 1
        xC, yC = C.shape
        C = -C  # Since we want our ineq to be greater than or equal to zero.
        G = np.hstack([C, -C])

        # The following line doesn't work. Appending ones will sum up beta+, beta-.
        # While we want to enforce every element of beta+ and beta- being greater than 0.
        # We must append the identity matrix, not an array of 1s.

        # Wrong:
        # G = np.vstack([G, -np.ones((1, 2 * yC))])

        # Correct:
        G = np.vstack([G, -np.identity(2*yC)])

        # Removing dependent rows means that we would lose some useful constraints.
        # G = QuadProgOperator.remove_dependent_rows(G)

        h = np.zeros((G.shape[0], ))

        P = np.zeros((2 * n, 2 * n))
        P[0:n, 0:n] = np.identity(n)
        P[n:2 * n, 0:n] = -np.identity(n)
        P[0:n, n:2 * n] = -np.identity(n)
        P[n:2 * n, n:2 * n] = np.identity(n)

        # if self.constraint_solver == 'cvxopt':
        #     P = cvxopt.matrix(P, tc='d')
        #     q = cvxopt.matrix(q, tc='d')
        #     G = cvxopt.matrix(G, tc='d')
        #     h = cvxopt.matrix(h, tc='d')
        #     solution = cvxopt.solvers.qp(P, q, G, h, kktsolver='ldl', options={'kktreg': 1e-9, 'show_progress': False,
        #                                                                        'maxiters': 500})
        #
        #     beta_2n = np.array(solution['x']).reshape((2 * n,))
        #     beta = beta_2n[:n] - beta_2n[n:]
        #
        #     return beta

        if self.constraint_solver == 'qpoases':
            solution = qpsolvers.qpoases_solve_qp(P, q, G, h)

            beta_2n = solution.reshape((2 * n,))
            beta = beta_2n[:n] - beta_2n[n:]

            return beta

        elif self.constraint_solver == 'osqp':
            solution = qpsolvers.osqp_solve_qp(P, q, G, h)

            beta_2n = solution.reshape((2 * n,))
            beta = beta_2n[:n] - beta_2n[n:]

            return beta

        else:
            print("Unrecognised constraint solver: " + str(self.constraint_solver))
            sys.exit()


class AdmmOperator(ProxOperator):
    def __init__(self, tau: Optional[float] = 0.1, min_iter: Optional[int] = 100,
                 max_iter: Optional[int] = 1000):
        super().__init__()
        self.tau = tau
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.constraint_solver = 'admm'

    @staticmethod
    def l2norm(x: np.ndarray, objective: np.ndarray, reg_param: float) -> float:
        sq_norm = (x - objective) ** 2
        return np.sum(sq_norm) * 1 / (2 * reg_param)

    @staticmethod
    def jac_l2norm(x: np.ndarray, objective: np.ndarray, reg_param: float) -> np.ndarray:
        return (x - objective) / reg_param

    @staticmethod
    def construct_constraints(data: np.ndarray, node: int, keep_diag_zero: bool):
        A = np.copy(data)
        if keep_diag_zero:
            A[:, node] = 0
        else:
            A[:, node] = 1
        constraints = scipy.optimize.LinearConstraint(A, 0, np.inf, keep_feasible=False)
        return constraints

    @staticmethod
    def check_convergence(x: np.ndarray, z: np.ndarray, threshold: Optional[float] = 1e-3):
        converged_params = (x - z) ** 2 <= (threshold ** 2) * (x ** 2)
        all_converged = all(converged_params)
        return all_converged

    def prox(self, objective: np.ndarray, reg_parameter: float, node: int, data_points: np.ndarray,
             keep_diag_zero: bool) -> np.ndarray:
        n = len(objective)
        u = np.zeros((n,))
        z = np.copy(objective)
        x = np.copy(objective)

        new_reg_parameter = reg_parameter / (1 + self.tau)
        soft_thresh_op = SoftThreshold()

        f_operator = partial(soft_thresh_op.prox,
                             reg_parameter=new_reg_parameter,
                             node=node,
                             data_points=data_points,
                             keep_diag_zero=keep_diag_zero)

        g_operator = partial(scipy.optimize.minimize,
                             method="SLSQP",
                             constraints=AdmmOperator.construct_constraints(data_points, node, keep_diag_zero))

        # Found bug: this flavor of ADMM makes x converge to -z.
        # Since z is the result of g_operator, keeping constraints true for x, means.
        # multiplying them by -1 to keep them true for z (if it makes sense).

        iteration = 0
        while (iteration <= self.min_iter or not AdmmOperator.check_convergence(x, z)) and iteration <= self.max_iter:
            prior_lasso_objective = (objective + self.tau * (z - u)) / (1 + self.tau)

            x = f_operator(prior_lasso_objective)

            fun = partial(AdmmOperator.l2norm, objective=x + u, reg_param=1.0)
            jac = partial(AdmmOperator.jac_l2norm, objective=x + u, reg_param=1.0)
            z = g_operator(x0=(x + u), fun=fun, jac=jac)['x']

            u = u + x - z

            iteration += 1

        return x
