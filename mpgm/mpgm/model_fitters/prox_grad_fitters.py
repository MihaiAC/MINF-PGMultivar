import numpy as np
import multiprocessing
from typing import Callable, Optional, Iterator, Any, Tuple, List

class Prox_Grad_Fitter():
    def __init__(self, alpha, accelerated=True, max_iter=5000, max_line_search_iter=50,
                 line_search_rel_tol=1e-4, init_step_size=1.0, beta=0.5, rel_tol=1e-3,
                 abs_tol=1e-6, early_stop_criterion='weight'):
        """
        Proximal gradient descent for solving the l1-regularized node-wise regressions required to fit some models in this
            package.
        :param alpha: L1 regularization parameter.
        :param accelerated: Whether to use the accelerated prox grad descent or not.
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
        self.alpha = alpha
        self.accelerated = accelerated
        self.max_iter = max_iter
        self.max_line_search_iter = max_line_search_iter
        self.line_search_rel_tol = line_search_rel_tol
        self.init_step_size = init_step_size
        self.beta = beta
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.early_stop_criterion = early_stop_criterion

    # Prox operators -> perhaps make them a class of their own if needed.
    # For now, it's not necessary, since we have been using only soft thresholding

    @staticmethod
    def soft_thresholding_prox_operator(x, threshold):
        x_plus = x - threshold
        x_minus = x - threshold

        return x_plus * (x_plus > 0) - x_minus * (x_minus > 0)

    def update_fit_params(self, iteration, prev_params, penultimate_params):
        if self.accelerated:
            w_k = iteration / (iteration + 3)
            params = prev_params + w_k * (prev_params - penultimate_params)
        else:
            params = prev_params
        return params

    def check_line_search_condition_simple(self, f_z, f_tilde):
        return f_z <= f_tilde

    def check_line_search_condition_closeto(self, f_z, f_tilde):
        return f_z <= f_tilde or np.isclose(f_z, f_tilde, rtol=self.line_search_rel_tol)

    def fit_node_parameter_generator(self, nr_nodes:int, nll:Callable[[int, np.ndarray, np.ndarray], float],
                      grad_nll:Callable[[int, np.array, np.array], np.array],
                      data_points:np.array, theta_init:np.array) -> Iterator[Any]:
        for node in range(nr_nodes):
            yield (node, nll, grad_nll, data_points, theta_init)

    def call_fit_node(self, nll:Callable[[int, np.array, np.array], float],
                      grad_nll:Callable[[int, np.array, np.array], np.array],
                      data_points:np.array,
                      theta_init:np.array,
                      parallelize:Optional[bool]=True) -> Tuple[np.ndarray, List[np.ndarray], List[bool], List[Any]]:

        nr_nodes = data_points.shape[1]
        theta_fit = np.zeros(theta_init.shape)
        likelihoods = []
        converged = []
        conditions = []

        if parallelize == False:
            for node in nr_nodes:
                theta_fit_node, likelihoods_node, converged_node, conditions_node = \
                    self.fit_node(node, nll, grad_nll, data_points, theta_init)
                theta_fit[node, :] = theta_fit_node
                likelihoods.append(likelihoods_node)
                converged.append(converged_node)
                conditions.append(conditions_node)

        else:
            nr_cpus = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=nr_cpus-1) as pool:
                results = pool.starmap(self.fit_node, self.fit_node_parameter_generator(nr_nodes, nll, grad_nll, data_points, theta_init))
            for node, node_result in enumerate(results):
                theta_fit[node, :] = node_result[0]
                likelihoods.append(node_result[1])
                converged.append(node_result[2])
                conditions.append(conditions[3])

        return (theta_fit, likelihoods, converged, conditions)

    def fit_node(self, node, nll, grad_nll, data_points, theta_init):
        """

        :param node: Index of the node to fit.
        :param nll: calculates negative ll of node value given the other data_points.
        :param grad_nll: calculates gradient of negative ll.
        :param data_points: N x m matrix, N = number of points and m = number of nodes in the graph.
        :param theta_init: m x m matrix; theta_init[node, :] must contain the initial guesses for the parameters fit here.
        :return: (parameters which resulted from the method, list of lists of line search likelihoods,
        bool which indicated if the method converged, not sure what this was)
        """

        likelihoods = []
        conditions = []
        converged = False

        theta_node_init = theta_init[node, :]
        theta_k_2 = np.array(theta_node_init)
        theta_k_1 = np.array(theta_node_init)
        step_size_k = self.init_step_size

        z = np.zeros(np.size(theta_node_init))
        f_z = 0

        for k in range(1, self.max_iter+1):
            theta_k = self.update_fit_params(k, theta_k_1, theta_k_2)
            f_theta_k = nll(node, data_points, theta_k)
            grad_f_theta_k = grad_nll(node, data_points, theta_k)

            found_step_size = False

            for _ in range(self.max_line_search_iter):
                candidate_new_theta_k = theta_k - step_size_k * grad_f_theta_k
                threshold = step_size_k * self.alpha
                z = Prox_Grad_Fitter.soft_thresholding_prox_operator(candidate_new_theta_k, threshold)

                f_tilde = f_theta_k + np.dot(grad_f_theta_k, z-theta_k) + (1/(2 * step_size_k)) * np.sum((z-theta_k) ** 2)
                f_z = nll(node, data_points, z)

                if self.check_line_search_condition_simple(f_z, f_tilde):
                    found_step_size = True
                    break
                else:
                    step_size_k = step_size_k * self.beta

            if found_step_size:
                theta_k_2 = theta_k_1
                theta_k_1 = z

                likelihoods.append(f_z)

                # Model-specific condition checking? Not sure.

                if self.early_stop_criterion == 'weight':
                    converged_params = (theta_k_1 - theta_k_2) ** 2 <= (self.rel_tol ** 2) * (theta_k_1 ** 2)
                    if all(converged_params):
                        converged = True
                        print('\nParameters for node ' + str(node) + ' converged in ' + str(k) + ' iterations.')
                        break

                elif self.early_stop_criterion == 'likelihood':
                    if (k > 3 and (f_z > likelihoods[k-2] or np.isclose(f_z, likelihoods[k-2], self.rel_tol, self.abs_tol))):
                        converged = True
                        print('\nParameters for node ' + str(node) + ' converged in ' + str(k) + ' iterations.')
                        break

            else:
                converged = False
                print('\nProx grad failed to converge for node ' + str(node))
                break

        return theta_k_1, np.array(likelihoods), converged, np.array(conditions)
