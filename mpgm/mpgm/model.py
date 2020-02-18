import numpy as np
import sys


class Model():
    def __init__(self, *params):
        pass

    def calculate_nll_and_grad_nll_datapoint(self, node, datapoint, theta_curr):
        pass

    def calculate_ll_datapoint(self, node, datapoint, theta_curr):
        pass

    def calculate_nll_and_grad_nll(self, node, data, theta_curr):
        N = data.shape[0]

        nll = 0
        grad_nll = np.zeros((len(theta_curr), ))

        for ii in range(N):
            nll_ii, grad_nll_ii = self.calculate_nll_and_grad_nll_datapoint(node, data[ii, :], theta_curr)
            nll += nll_ii
            grad_nll += grad_nll_ii

        nll = nll/N
        grad_nll = grad_nll/N

        return nll, grad_nll

    def calculate_nll(self, node, data, theta_curr):
        N = data.shape[0]

        nll = 0

        for ii in range(N):
            nll += -self.calculate_ll_datapoint(node, data[ii, :], theta_curr)[0]

        nll = nll/N

        return nll

    @staticmethod
    def provide_args(nr_nodes, other_args):
        for ii in range(nr_nodes):
            yield [ii] + other_args
        return

    @staticmethod
    def fit_prox_grad(node, model, data, alpha, accelerated=False, max_iter=5000, max_line_search_iter=50,
                      line_search_rel_tol=1e-4, lambda_p=1.0, beta=0.5, rel_tol=1e-3, abs_tol=1e-6):
        """
        Proximal gradient descent for solving the l1-regularized node-wise regressions required to fit some models in this
            package.
        :param model: Model to be fit by prox_grad. Must implement "calculate_nll_and_grad_nll", "calculate_nll" and a
            condition function to be evaluated after each iteration (necessary for QPGM, SPGM) as per the Model "interface".
        :param data: Data used to fit the model.
        :param alpha: L1 regularization parameter.
        :param node: The node we're doing regression on.
        :param max_iter: Maximum iterations allowed for the algorithm.
        :param max_line_search_iter: Maximum iterations allowed for the line search performed at every iteration step.
        :param lambda_p: Initial step-size.
        :param beta: Line search parameter.
        :param rel_tol: Relative tolerance value for early stopping.
        :param abs_tol: Absolute tolerance value for early stopping. Should be 1e-3 * rel_tol.
        :return: (parameters, likelihood_values, converged) - tuple containing parameters and the NLL value for each
            iteration;
        """

        f = model.calculate_nll
        f_and_grad_f = model.calculate_nll_and_grad_nll
        theta_init = model.theta[node, :]

        likelihoods = np.zeros((max_iter,))

        conditions = None
        if model.condition is not None:
            conditions = np.zeros((max_iter,))

        theta_k_2 = np.array(theta_init)
        theta_k_1 = np.array(theta_init)
        lambda_k = lambda_p

        converged = False

        epsilon = sys.float_info.epsilon

        z = np.zeros(np.size(theta_init))
        f_z = 0

        for k in range(1, max_iter + 1):
            if (accelerated):
                w_k = k / (k + 3)
                y_k = theta_k_1 + w_k * (theta_k_1 - theta_k_2)
            else:
                y_k = theta_k_1
            f_y_k, grad_y_k = f_and_grad_f(node, data, y_k)

            sw = False
            for _ in range(max_line_search_iter):
                z = Model.prox_operator(y_k - lambda_k * grad_y_k, threshold=lambda_k * alpha)
                f_tilde = f_y_k + np.dot(grad_y_k, z - y_k) + (1 / (2 * lambda_k)) * np.sum((z - y_k) ** 2)
                f_z = f(node, data, z)  # NLL at current step.
                if f_z < f_tilde or np.isclose(f_z, f_tilde, rtol=1e-4):
                    sw = True
                    break
                else:
                    lambda_k = lambda_k * beta
            if sw:
                theta_k_2 = theta_k_1
                theta_k_1 = z

                likelihoods[k-1] = f_z

                if model.condition is not None:
                    conditions[k - 1] = model.condition(node, z, data)
                    if (not conditions[k-1]):
                        print('FALSE!')
                    if (not conditions[k-1] and model.break_fit_if_condition_broken == True):
                        break

                # if (k > 3 and (np.abs(f_z - likelihoods[k - 2]) < abs_tol or
                #               (np.abs(f_z - likelihoods[k - 2]) / min(np.abs(f_z),
                #                                                       np.abs(likelihoods[k - 2]))) < rel_tol)):
                #    converged = True
                #    break
            else:
                break
        return theta_k_1, likelihoods, conditions, converged

    @staticmethod
    def prox_operator(x, threshold):
        """
        Applies the soft thresholding operation to vector x with the specified threshold.

        :param x:
        :param threshold:
        :return:
        """
        x_plus = x - threshold
        x_minus = -x - threshold
        return x_plus * (x_plus > 0) - x_minus * (x_minus > 0)

    @classmethod
    def call_prox_grad_wrapper(cls, packed_params):
        # model_params = packed_params[0]
        # prox_grad_params = packed_params[1]
        # node = packed_params[2]

        model = cls(*packed_params[0])
        return packed_params[2], cls.fit_prox_grad(packed_params[2], model, *packed_params[1])