import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize
from mpgm.mpgm.models.Model import Model
from functools import partial

from typing import Optional, List, Tuple


class SPGM(Model):
    def __init__(self, theta:Optional[np.ndarray]=None, R:Optional[int]=10, R0:Optional[int]=5,
                 partition_atol:Optional[float]=1e-20, partition_max_iter:Optional[int]=10000):
        super().__init__(theta)
        self.R = R
        self.R0 = R0

        self.partition_atol = partition_atol
        self.partition_max_iter = partition_max_iter

        self.condition = None

    def __setattr__(self, key, value):
        if key == 'R':
            assert type(value) is int and value > 0, "R must be a positive integer"

        if key == 'R0':
            assert type(value) is int and value > 0 and value < self.R, 'R0 must be a positive integer, smaller than R'

        super(SPGM, self).__setattr__(key, value)

    def sufficient_statistics(self, node_value):
        if(node_value <= self.R0):
            return node_value
        elif(node_value <= self.R):
            return (-0.5 * (node_value ** 2) + self.R * node_value - 0.5 * (self.R0 ** 2)) / (self.R - self.R0)
        else:
            return 0.5 * (self.R + self.R0)

    def estimate_partition_exponents(self, node:int, node_values_suffst:np.ndarray) -> List[float]:
        '''
        Estimates partition by summing terms until it converges or runs out of alotted iterations.
        '''
        # TODO: This missed the first exponent, which is always 0.
        first_exponent = 0
        # first_exponent = (self.theta[node, node] + np.dot(self.theta[node, :], node_values_suffst) - \
        #             self.theta[node, node] * node_values_suffst[node]) * node_values_suffst[1] - gammaln(2)

        # TODO: I no longer agree with the statement below. Revisit after experimentation?
        # atol = min(np.exp(first_exponent) * self.partition_atol, self.partition_atol)
        atol = self.partition_atol
        max_iterations = self.partition_max_iter

        iterations = 0
        exponents = []
        exponent = first_exponent
        while (not np.isclose(np.exp(exponent), 0, atol=atol) and iterations < max_iterations):
            exponents.append(exponent)
            iterations += 1
            node_value = iterations

            exponent = (self.theta[node, node] + np.dot(self.theta[node, :], node_values_suffst) - \
                    self.theta[node, node] * node_values_suffst[node]) * self.sufficient_statistics(node_value) - \
                    gammaln(node_value+1)

            # print(exponent)

        return exponents

    def node_cond_prob(self, node:int, node_value:int, node_values_suffst:np.ndarray, dot_product:Optional[float]=None,
                       log_partition:Optional[float]=None) -> Tuple[float, float, float]:
        if dot_product is None:
            dot_product = np.dot(self.theta[node, :], node_values_suffst) - self.theta[node, node] * \
                          node_values_suffst[node] + self.theta[node, node]

        if log_partition is None:
            partition_exponents = self.estimate_partition_exponents(node, node_values_suffst)
            log_partition = logsumexp(partition_exponents)
            # # TODO: remove debug.
            # print("Node: " + str(node) + " Node_value: " + str(node_value) + " Log partition: " + str(log_partition))

        cond_prob = np.exp(dot_product * self.sufficient_statistics(node_value) - gammaln(node_value+1) - log_partition)
        return cond_prob, dot_product, log_partition


# This is enough to test sampling from SPGM.
# TODO: Refactor the rest for fitting.
    def calculate_ll_datapoint(self, node, datapoint, theta_curr):
        """
        :return: returns (nll_datapoint, log_partition)
        """
        datapoint_sf = np.zeros((len(datapoint), ))
        for ii in range(len(datapoint)):
            datapoint_sf[ii] = self.sufficient_statistics(datapoint[ii])

        dot_product = np.dot(datapoint_sf, theta_curr) - theta_curr[node] * datapoint_sf[node] + theta_curr[node]

        log_partition = np.exp(dot_product)

        return dot_product * datapoint_sf[node] - gammaln(datapoint[node]) - log_partition, log_partition, datapoint_sf

    def calculate_grad_ll_datapoint(self, node, datapoint, theta_curr, log_partition, datapoint_sf):
        grad = np.zeros(datapoint.shape)

        grad[node] = datapoint_sf[node] - log_partition
        for ii in range(datapoint.shape[0]):
            if ii != node:
                grad[ii] = grad[node] * datapoint_sf[node]

        return grad

    def calculate_nll_and_grad_nll_datapoint(self, node, datapoint, theta_curr):
        ll, log_partition, datapoint_sf = self.calculate_ll_datapoint(node, datapoint, theta_curr)
        grad_ll = self.calculate_grad_ll_datapoint(node, datapoint, theta_curr, log_partition, datapoint_sf)

        return -ll, -grad_ll

    # def condition(self, node, theta, data):
    #     conditions = 0
    #     for kk in range(data.shape[0]):
    #         datapoint = data[kk, :]
    #         datapoint_sf = np.zeros((len(datapoint), ))
    #         for ii, node_value in enumerate(datapoint):
    #             datapoint_sf[ii] = self.sufficient_statistics(node_value)
    #         dot_product = np.dot(theta, datapoint_sf) - theta[node] * datapoint_sf[node] + theta[node]
    #         if(dot_product >= 0):
    #             conditions += 1
    #     return conditions

    @staticmethod
    def fit_prox_grad(node, model, data, alpha, method='SLSQP', accelerated=True, max_iter=5000, max_line_search_iter=50,
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
        :param method: Method to be used in finding the closest parameter values to the objective which satisfy the
            inequalities we are interested in.
        :return: (parameters, likelihood_values, converged) - tuple containing parameters and the NLL value for each
            iteration;
        """

        f = model.calculate_nll
        f_and_grad_f = model.calculate_nll_and_grad_nll
        theta_init = model.theta[node, :]

        likelihoods = []

        # Debug code to check if conditions that must be met are respected while training.
        conditions = []

        theta_k_2 = np.array(theta_init)
        theta_k_1 = np.array(theta_init)
        lambda_k = lambda_p

        converged = False

        nr_variables = len(theta_init)
        #parameters_regularization_path = np.array(theta_init).reshape((1, nr_variables))

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
                optimize_result = SPGM.satisfy_constraints(x, node, data, method)

                if not optimize_result['success']:
                    converged = False
                    print('\nProx grad failed to converge for node ' + str(node))
                    break
                else:
                    z = optimize_result['x']

                theta_k_2 = theta_k_1
                theta_k_1 = z

                #parameters_regularization_path = np.concatenate((parameters_regularization_path, theta_k_1.reshape((1, nr_variables))), axis=0)

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
    def l2norm(x, v):
        """
        :param x: Parameter values to be fit.
        :param v: Objective-parameters.
        :return: Returns the L2 norm of vector x-v.
        """
        return np.sum((x-v)**2)

    @staticmethod
    def l2norm_partial(v):
        """
        Partially apply variable v in l2norm.
        """
        return partial(SPGM.l2norm, v=v)

    @staticmethod
    def l2norm_jac(x, v):
        """
        :param x: Parameter values to be fit.
        :param v: Objective-parameters.
        :return: Returns the jacobian of the l2 norm wrt parameters x.
        """
        return 2*(x-v)

    @staticmethod
    def l2norm_jac_partial(v):
        """
        Partially apply variable v in l2norm_jac.
        """
        return partial(SPGM.l2norm_jac, v=v)

    @staticmethod
    def constraint(x, data_point, node):
        """
        Checks if the inequality corresponding to parameters x, data point "data_point" and node "node" holds.
        :param x: Current parameter values.
        :param data_point: Data point.
        :param node: Node whose parameters we are currently fitting.
        :return: True if the inequality holds, false otherwise.
        """
        inequality_lh = np.dot(x, data_point) - x[node] * data_point[node] + x[node]
        return inequality_lh >= 0

    @staticmethod
    def constraint_jac(x, data_point, node):
        """
        :param x: Current parameter values.
        :param data_point: Current data_point.
        :param node: Node we are currently fitting the parameters for.
        :return: Returns the Jacobian of inequality_lh defined in "constraint".
        """
        jac = np.array(data_point)
        jac[node] = 1
        return jac

    @staticmethod
    def construct_constraints(data, node):
        """
        Construct a list of dictionaries of constraints, as required by scipy.optimize.minimize.
        """
        dict_seq = []
        for ii in range(data.shape[0]):
            data_point = data[ii, :]
            constraint_dict = dict()
            constraint_dict['type'] = 'ineq'
            constraint_dict['fun'] = partial(SPGM.constraint, data_point=data_point, node=node)
            constraint_dict['jac'] = partial(SPGM.constraint_jac, data_point=data_point, node=node)
            dict_seq.append(constraint_dict)
        return dict_seq

    @staticmethod
    def satisfy_constraints(x, node, data, method):
        """
        Returns an OptimizeResult object.
        """
        objective = x

        # Construct parameters for scipy.optimize.minimize.
        func_to_optimize = SPGM.l2norm_partial(objective)
        initial_guess = np.array(objective)
        jac_func_to_optimize = SPGM.l2norm_jac_partial(objective)
        constraint_dict_list = SPGM.construct_constraints(data, node)
        tol = 1e-3

        return minimize(func_to_optimize, x0=initial_guess, method=method, jac=jac_func_to_optimize,
                        constraints=constraint_dict_list, tol=tol)

    def fit(self, **prox_grad_params):
        model_params = [self.theta, self.R, self.R0]

        nr_nodes = prox_grad_params['data'].shape[1]

        ordered_results = [()] * nr_nodes
        with Pool(processes=4) as pool:
            for result in tqdm(pool.imap_unordered(SPGM.call_prox_grad_wrapper,
                                                   iter(((model_params, prox_grad_params, x) for x in range(nr_nodes))),
                                                   chunksize=1), total=nr_nodes):
                ordered_results[result[0]] = result[1]

        return ordered_results
