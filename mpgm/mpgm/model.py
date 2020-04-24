import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import trange

# All model subclasses that permit sampling, must have a parameter named theta.
class Model():
    def __init__(self, *params):
        pass

    def generate_node_sample(self, node, nodes_values):
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

    def calculate_joint_nll(self, data):
        nr_nodes = data.shape[1]

        joint_nll = 0
        for node in range(nr_nodes):
            joint_nll += self.calculate_nll(node, data, self.theta[node, :])

        return joint_nll

    @staticmethod
    def provide_args(nr_nodes, other_args):
        for ii in range(nr_nodes):
            yield [ii] + other_args
        return

    @staticmethod
    def fit_prox_grad(node, model, data, alpha, accelerated=True, max_iter=5000, max_line_search_iter=50,
                      line_search_rel_tol=1e-4, lambda_p=1.0, beta=0.5, rel_tol=1e-3, abs_tol=1e-6,
                      early_stop_criterion='weight'):
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
        :param early_stop_criterion: If equal to 'weight', training stops when the weight values have converged.
            If equal to 'likelihood', training stops when the value of the NLL has converged.
        :return: (parameters, likelihood_values, converged) - tuple containing parameters and the NLL value for each
            iteration;
        """

        f = model.calculate_nll
        f_and_grad_f = model.calculate_nll_and_grad_nll
        theta_init = model.theta[node, :]

        likelihoods = []
        conditions = []

        theta_k_2 = np.array(theta_init)
        theta_k_1 = np.array(theta_init)
        lambda_k = lambda_p

        converged = False

        nr_variables = len(theta_init)

        z = np.zeros(np.size(theta_init))
        f_z = 0

        for k in range(1, max_iter + 1):
            if accelerated:
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
                if f_z < f_tilde or np.isclose(f_z, f_tilde, rtol=line_search_rel_tol):
                    sw = True
                    break
                else:
                    lambda_k = lambda_k * beta
            if sw:
                theta_k_2 = theta_k_1
                theta_k_1 = z

                likelihoods.append(f_z)

                # Debug code for checking conditions.
                if model.condition is not None:
                    conditions.append(model.condition(node, z, data))
                    if (not conditions[k-1]):
                        print('FALSE!')
                    if (not conditions[k-1] and model.break_fit_if_condition_broken == True):
                        break

                # Convergence criterion for parameters (weights).
                if early_stop_criterion == 'weight':
                    converged_params = (theta_k_1 - theta_k_2) ** 2 <= (rel_tol ** 2) * (theta_k_1 ** 2)
                    if all(converged_params):
                        converged = True
                        print('\nParameters for node ' + str(node) + ' converged in ' + str(k) + ' iterations.')
                        break
                elif early_stop_criterion == 'likelihood':
                    # Convergence criterion for likelihoods.
                    if (k > 3 and (f_z > likelihoods[k-2] or np.isclose(f_z, likelihoods[k-2], rel_tol, abs_tol))):
                        converged = True
                        print('\nParameters for node ' + str(node) + ' converged in ' + str(k) + ' iterations.')
                        break
            else:
                converged = False
                print('\nProx grad failed to converge for node ' + str(node))
                break

        return theta_k_1, np.array(likelihoods), conditions, converged

    @staticmethod
    def prox_operator(x, threshold):
        """
        Applies the soft thresholding operation to vector x with the specified threshold.

        :return:
        """
        x_plus = x - threshold
        x_minus = -x - threshold
        return x_plus * (x_plus > 0) - x_minus * (x_minus > 0)

    @classmethod
    def call_prox_grad_wrapper(cls, packed_params):
        # model_params = packed_params[0]
        # prox_grad_params = packed_params[1], dictionary.
        # node = packed_params[2]

        model = cls(*packed_params[0])
        return packed_params[2], cls.fit_prox_grad(packed_params[2], model, **packed_params[1])

    def generate_samples_gibbs(self, init, nr_samples=50, burn_in=700, thinning_nr=100):
        """
        Gibbs sampler for this implementation of the TPGM distribution.

        :param model: model to draw samples from, must implement "generate_sample_cond_prob(node, data)"
        :param init: (N, ) np array = starting value for the sampling process;
        :param nr_samples: number of samples we want to generate.
        :param burn_in: number of initial samples to be discarded.
        :param thinning_nr: we select every thinning_nr samples after the burn-in period;

        :return: (nr_samples X N) matrix containing one sample per row.
        """
        nr_variables = len(init)
        samples = np.zeros((nr_samples, nr_variables))
        nodes_values = np.array(init)

        for sample_nr in trange(burn_in + (nr_samples - 1) * thinning_nr + 1):
            # Generate one sample.
            for node in range(nr_variables):
                node_sample = self.generate_node_sample(node, nodes_values)
                nodes_values[node] = node_sample

            # Check if we should keep this sample.
            if sample_nr >= burn_in and (sample_nr - burn_in) % thinning_nr == 0:
                samples[(sample_nr - burn_in) // thinning_nr, :] = nodes_values

        return samples