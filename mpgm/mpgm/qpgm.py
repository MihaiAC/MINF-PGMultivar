import numpy as np
from decimal import Decimal
from multiprocessing import Pool
from mpgm.mpgm.solvers import prox_grad


class Qpgm(object):
    """
    Class for generating samples from and fitting QPGM distributions.

    :param theta: N x N matrix; theta[s][t] = weight of the edge between nodes (dimensions) s and t.
    :param theta_quad: 1 x N matrix; theta_quad[ss] = parameter of the quadratic term;
    :param data: Data to fit the model with.
    """

    def __init__(self, theta=None, theta_quad=None):
        self.theta = theta
        self.theta_quad = theta_quad

    def generate_node_sample(self, node, nodes_values):
        uu = np.random.uniform(0, 1, 1)[0]
        uu = Decimal(uu)

        prob1, partition_max_exp, partition_reduced, dot_product = self.node_cond_prob(node, 0, nodes_values)
        cdf = prob1

        current_node_value = 1
        while uu.compare(Decimal(cdf)) == Decimal('1'):
            cdf += self.node_cond_prob(node, current_node_value, nodes_values, dot_product, partition_max_exp, partition_reduced)[0]
            current_node_value += 1

        return current_node_value - 1

    def node_cond_prob(self, node, current_node_value, nodes_values, partition_max_exp=None, partition_reduced=None):
        """
        Calculates the probability of node having the provided value given the other nodes.

        :param node: Integer representing the node we're interested in.
        :param nodes_values: Values of the other nodes.
        :param partition_max_exp: The partition is a sum of exponentials. partition_max_exp is the maximum exponent in
            this sum.
        :param partition_reduced: partition divided by exp(partition_max_exp).
        :return: A tuple containing the likelihood, the value of the partition function and the dot_product in this order.
        """
        dot_product = np.dot(self.theta[node, :], nodes_values) - self.theta[node][node] * nodes_values[node] + \
                      self.theta[node][node]

        if partition_reduced is None:
            partition_exponents = []
            partition_max_exp = -np.inf
            kk = 0
            curr_exp = 1
            #TODO: good stopping condition?
            while curr_exp > np.log(1e-15):
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

    @staticmethod
    def calculate_ll_datapoint(node, datapoint, theta):
        """
        :param theta[-1] must be the parameter of the quadratic term;
               theta[:-1][t] must be the parameter theta_node_t
               theta[node] must be the parameter theta_node
        :return: returns ll
        """
        theta_normal = theta[:-1]
        theta_quad = theta[-1]
        ll = 0
        dot_product = np.dot(datapoint, theta_normal) - theta_normal[node] * datapoint[node] + theta_normal[node]
        ll += dot_product
        ll *= datapoint[node]
        ll += theta_quad * datapoint[node] ** 2
        ll = ll - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(-2 * theta_quad) + 1/(4 * theta_quad) * (dot_product ** 2)

        return ll

    @staticmethod
    def calculate_grad_ll_datapoint(node, datapoint, theta):
        grad = np.zeros(datapoint.shape)

        theta_normal = theta[:-1]
        theta_quad = theta[-1]

        dot_product = np.dot(datapoint, theta_normal) - theta_normal[node] * datapoint[node] + theta_normal[node]

        for ii in range(len(theta)):
            if ii == len(theta) - 1:
                grad[ii] = datapoint[node] ** 2 + 1/(2 * theta_quad) - 1/(4 * theta_quad ** 2) * dot_product ** 2
            elif ii == node:
                grad[ii] = datapoint[node] + 1/(2 * theta_quad) * dot_product
            else:
                grad[ii] = datapoint[node] * datapoint[ii] + 1/(2 * theta_quad) * dot_product * datapoint[ii]

        return grad

    @staticmethod
    def calculate_nll_and_grad_nll_datapoint(node, datapoint, theta, R):
        ll = Qpgm.calculate_ll_datapoint(node, datapoint, theta)
        grad_ll = Qpgm.calculate_grad_ll_datapoint(node, datapoint, theta)

        return -ll, -grad_ll

    @staticmethod
    def calculate_nll_and_grad_nll(node, data, theta, R):
        N = data.shape[0]

        nll = 0
        grad_nll = np.zeros((data.shape[1], ))

        for ii in range(N):
            nll_ii, grad_nll_ii = Qpgm.calculate_nll_and_grad_nll_datapoint(node, data[ii, :], theta, R)
            nll += nll_ii
            grad_nll += grad_nll_ii

        nll = nll/N
        grad_nll = grad_nll/N

        return nll, grad_nll

    @staticmethod
    def calculate_nll(node, data, theta, R):
        N = data.shape[0]

        nll = 0

        for ii in range(N):
            nll -= Qpgm.calculate_ll_datapoint(node, data[ii, :], theta, R)[0]

        nll = nll/N

        return nll

    @staticmethod
    def condition(node, theta, data, *args):
        return theta[-1] < 0

    @staticmethod
    def provide_args(nr_nodes, other_args):
        for ii in range(nr_nodes):
            yield [ii] + other_args
        return

    @staticmethod
    def fit(data, theta_init, alpha, R, condition, max_iter=5000, max_line_search_iter=50, lambda_p=1.0, beta=0.5,
            rel_tol=1e-3, abs_tol=1e-6):
        """
        :param data: N X P matrix; each row is a datapoint;
        :param theta_init: starting parameter values;
        :param alpha: regularization parameter;
        :param max_iter:
        :param max_line_search_iter:
        :param lambda_p: step size for the proximal gradient descent method;
        :param beta:
        :param rel_tol:
        :param abs_tol:
        :return:
        """

        f = Qpgm.calculate_nll
        f_and_grad_f = Qpgm.calculate_nll_and_grad_nll

        tail_args = [theta_init, alpha, data, f, f_and_grad_f, [R], condition, max_iter, max_line_search_iter, lambda_p, beta,
                     rel_tol, abs_tol]
        nr_nodes = data.shape[1]

        with Pool(processes=4) as pool:
            return pool.starmap(prox_grad, Qpgm.provide_args(nr_nodes, tail_args))