import numpy as np
import sys
from multiprocessing import Pool
from mpgm.mpgm.model import Model
from tqdm import tqdm


class QPGM(Model):
    """
    Class for generating samples from and fitting QPGM distributions.

    :param theta: N x N matrix; theta[s][t] = weight of the edge between nodes (dimensions) s and t.
    :param theta_quad: 1 x N matrix; theta_quad[ss] = parameter of the quadratic term;
    """

    def __init__(self, theta=None, theta_quad=None):
        self.theta = theta
        self.theta_quad = theta_quad
        self.break_fit_if_condition_broken = True

    def generate_node_sample(self, node, nodes_values):
        uu = np.random.uniform(0, 1, 1)[0]

        prob1, partition_max_exp, partition_reduced, dot_product = self.node_cond_prob(node, 0, nodes_values)
        cdf = prob1

        current_node_value = 1
        while uu > cdf:
            cdf += self.node_cond_prob(node, current_node_value, nodes_values, dot_product, partition_max_exp,
                                       partition_reduced)[0]
            current_node_value += 1

        return current_node_value - 1

    def node_cond_prob(self, node, current_node_value, nodes_values, dot_product=None, partition_max_exp=None,
                       partition_reduced=None):
        """
        Calculates the probability of node having the provided value given the other nodes.

        :param node: Integer representing the node we're interested in.
        :param nodes_values: Values of the other nodes.
        :param partition_max_exp: The partition is a sum of exponentials. partition_max_exp is the maximum exponent in
            this sum.
        :param partition_reduced: partition divided by exp(partition_max_exp).
        :return: A tuple containing the likelihood, the value of the partition function and the dot_product in this order.
        """
        if(dot_product == None):
            dot_product = np.dot(self.theta[node, :], nodes_values) - self.theta[node][node] * nodes_values[node] + \
                self.theta[node][node]

        if partition_reduced is None:
            partition_exponents = []
            partition_max_exp = -np.inf
            kk = 0
            curr_exp = 1
            #TODO: good stopping condition?
            while len(partition_exponents) < 200 or (len(partition_exponents) <= 500 and np.exp(curr_exp)/np.exp(partition_max_exp) > 1e-10):
                curr_exp = dot_product * kk + self.theta_quad[node] * (kk ** 2)
                partition_exponents.append(curr_exp)
                if partition_max_exp < curr_exp:
                    partition_max_exp = curr_exp
                kk += 1

            partition_reduced = 0
            for exponent in partition_exponents:
                partition_reduced += np.exp(exponent - partition_max_exp)

        cond_prob = np.exp(dot_product * current_node_value + self.theta_quad[node] * (current_node_value ** 2) - partition_max_exp)/partition_reduced

        return cond_prob, partition_max_exp, partition_reduced, dot_product

    def calculate_ll_datapoint(self, node, datapoint, theta_curr):
        """
        :param theta[-1] must be the parameter of the quadratic term;
               theta[:-1][t] must be the parameter theta_node_t
               theta[node] must be the parameter theta_node
        :return: returns ll
        """
        theta_normal = theta_curr[:-1]
        theta_quad = theta_curr[-1]
        ll = 0
        dot_product = np.dot(datapoint, theta_normal) - theta_normal[node] * datapoint[node] + theta_normal[node]
        ll += dot_product
        ll *= datapoint[node]
        ll += theta_quad * datapoint[node] ** 2
        if(theta_quad > 0):
            print(str(theta_quad) + ' ' + str(self.condition(node, theta_curr, datapoint)))
        ll = ll - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(-2 * theta_quad) + 1/(4 * theta_quad) * (dot_product ** 2)

        return (ll, )

    def calculate_grad_ll_datapoint(self, node, datapoint, theta_curr):
        grad = np.zeros(theta_curr.shape)

        theta_normal = theta_curr[:-1]
        theta_quad = theta_curr[-1]

        dot_product = np.dot(datapoint, theta_normal) - theta_normal[node] * datapoint[node] + theta_normal[node]

        for ii in range(len(theta_curr)):
            if ii == len(theta_curr) - 1:
                grad[ii] = datapoint[node] ** 2 + 1/(2 * theta_quad) - 1/(4 * theta_quad ** 2) * dot_product ** 2
            elif ii == node:
                grad[ii] = datapoint[node] + 1/(2 * theta_quad) * dot_product
            else:
                grad[ii] = datapoint[node] * datapoint[ii] + 1/(2 * theta_quad) * dot_product * datapoint[ii]

        return grad

    def calculate_nll_and_grad_nll_datapoint(self, node, datapoint, theta_curr):
        ll = self.calculate_ll_datapoint(node, datapoint, theta_curr)[0]
        grad_ll = self.calculate_grad_ll_datapoint(node, datapoint, theta_curr)

        return -ll, -grad_ll

    def condition(self, node, theta_curr, data):
        return theta_curr[-1] < 0

    @staticmethod
    def fit_prox_grad(node, model, data, alpha, qtp_c=1e4, accelerated=True, max_iter=5000, max_line_search_iter=50,
                      line_search_rel_tol=1e-4, lambda_p=1.0, beta=0.5, rel_tol=1e-3, abs_tol=1e-6,
                      early_stop_criterion='weight'):
        """
            Proximal gradient descent for solving the l1-regularized node-wise regressions required to fit some models in this
                package.
            :param model: Model to be fit by prox_grad. Must implement "calculate_nll_and_grad_nll", "calculate_nll" and a
                condition function to be evaluated after each iteration (necessary for QPGM, SPGM) as per the Model "interface".
            :param data: Data used to fit the model.
            :param alpha: L1 regularization parameter.
            :param qtp_c: Parameters which would become positive get clipped to -1/qtp_c instead in order to preserve
                the upper bound used for fitting the model.
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

        likelihoods = np.zeros((max_iter,))

        conditions = []
        if model.condition is not None:
            conditions = np.zeros((max_iter, ))

        theta_k_2 = np.array(theta_init)
        theta_k_1 = np.array(theta_init)
        lambda_k = lambda_p

        converged = False

        epsilon = sys.float_info.epsilon

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
            # TODO: prox_operator | thresholding should be outside of the line search (I'm pretty sure, can compare).
            # Does it make that much of a difference, though?
            for _ in range(max_line_search_iter):
                z = QPGM.prox_operator(y_k - lambda_k * grad_y_k, threshold=lambda_k * alpha, qtp_c=qtp_c)
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

                likelihoods[k - 1] = f_z

                if model.condition is not None:
                    conditions[k - 1] = model.condition(node, z, data)
                    if (not conditions[k - 1]):
                        print('FALSE!')
                    if (not conditions[k - 1] and model.break_fit_if_condition_broken == True):
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
                    if (k > 3 and (
                            f_z > likelihoods[k - 2] or np.isclose(f_z, likelihoods[k - 2], rel_tol, abs_tol))):
                        converged = True
                        print('\nParameters for node ' + str(node) + ' converged in ' + str(k) + ' iterations.')
                        break
            else:
                converged = False
                print('\nProx grad failed to converge for node ' + str(node))
                break
        return theta_k_1, likelihoods, conditions, converged

    @staticmethod
    def prox_operator(x, threshold, qtp_c=1e4):
        ret_vector = np.zeros((len(x), ))
        ret_vector[:-1] = Model.prox_operator(x[:-1], threshold)

        # We have the constraint that theta_ss = x[-1] should remain negative, below a threshold -1/qtp_c.
        # So, we need to find the minimum of (theta_ss - x[-1]) ** 2 subject to theta_ss <= -1/qtp_c.

        ret_vector[-1] = x[-1] - max(0, x[-1] + 1/qtp_c)
        return ret_vector

    def fit(self, **prox_grad_params):
        nr_nodes = prox_grad_params['data'].shape[1]

        self.theta = np.hstack((self.theta, np.reshape(self.theta_quad, (nr_nodes, 1))))
        model_params = [self.theta, self.theta_quad]

        ordered_results = [()] * nr_nodes

        with Pool(processes=4) as pool:
            for result in tqdm(pool.imap_unordered(QPGM.call_prox_grad_wrapper,
                                                   iter(((model_params, prox_grad_params, x) for x in range(nr_nodes))),
                                                   chunksize=1), total=nr_nodes):
                ordered_results[result[0]] = result[1]
        return ordered_results

