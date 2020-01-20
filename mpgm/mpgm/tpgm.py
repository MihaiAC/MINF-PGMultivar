import numpy as np
from decimal import Decimal
from multiprocessing import Pool
from math import factorial
from mpgm.mpgm.solvers import prox_grad


class Tpgm(object):
    """
    Class for generating samples from and fitting TPGM distributions.

    :param N: dimensionality of our random variables;
    :param theta: N x N matrix; theta[s][t] = weight of the edge between nodes (dimensions) s and t.
    :param R: maximum count (truncation value).
    :param log_factorials: Log factorials of integers up to R.
    :param data: Data to fit the model with.
    """

    def __init__(self, theta=None, R=100):
        """
        Constructor for TPGM class.

        :param theta: N x N matrix, given when we want to generate samples from an existing distribution.
        :param R: maximum count value, should be an integer.
        """
        self.R = R
        self.theta = np.array(theta)

    def __setattr__(self, key, value):
        if key == 'R':
            assert type(value) is int and value > 0, "R must be a positive integer"

            #TODO: Test the performance of the TPGM model for different R values
            assert value <= 100, "The model has not been tested with R values higher than 100"
        super(Tpgm, self).__setattr__(key, value)

    def generate_node_sample(self, node, nodes_values):
        """
        Generate a sample for node from its node-conditional probability.

        :param node: Node to generate sample for.
        :param data: Current values of the nodes (ordered).
        :return: A sample from the node-conditional probability of our node.
        """

        uu = np.random.uniform(0, 1, 1)[0]
        uu = Decimal(uu)

        prob1, partition_max_exp, partition_reduced, dot_product = self.node_cond_prob(node, 0, nodes_values)
        cdf = prob1

        for node_value in range(1, self.R + 1):
            if uu.compare(Decimal(cdf)) == Decimal('-1'):
                return node_value-1
            cdf += self.node_cond_prob(node, node_value, nodes_values, dot_product, partition_max_exp, partition_reduced)[0]

        return self.R

    def node_cond_prob(self, node, node_value, data, dot_product=None, partition_max_exp=None, partition_reduced=None):
        """
        Calculates the probability of node having the provided value given the other nodes.

        :param node: Integer representing the node we're interested in.
        :param data: 1 X P array containing the values of the nodes (including our node).
        :param node_value: Value of the node we're interested in.
        :param partition: Optional argument, equal to the value of the partition function. Relevant only for sampling.
        :param dot_product: Inner dot product present in the partition function and denominator of the conditional
            probability.
        :return: A tuple containing the likelihood, the value of the partition function and the dot_product in this order.
        """
        if dot_product is None:
            dot_product = np.dot(self.theta[node, :], data) - self.theta[node, node] * data[node] + self.theta[node, node]

        if partition_reduced is None:

            partition_exponents = np.zeros((self.R+1, ))
            for kk in range(self.R+1):
                partition_exponents[kk] = dot_product * kk - np.log(factorial(kk))

            partition_max_exp = max(partition_exponents)
            partition_reduced = 0
            for kk in range(self.R+1):
                partition_reduced += np.exp(partition_exponents[kk] - partition_max_exp)

        cond_prob = np.exp(dot_product * node_value - np.log(factorial(node_value)) - partition_max_exp)/partition_reduced

        return cond_prob, partition_max_exp, partition_reduced, dot_product

    @staticmethod
    def calculate_ll_datapoint(node, datapoint, theta, R):
        """
        :return: returns (nll_datapoint, log_partition)
        """

        log_partition = 0

        exponents = []
        dot_product = np.dot(datapoint, theta) - theta[node] * datapoint[node] + theta[node]
        for kk in range(R+1):
            curr_exponent = dot_product * kk - np.log(float(factorial(kk)))
            exponents.append(curr_exponent)

        max_exponent = max(exponents)
        log_partition += max_exponent

        sum_of_rest = 0
        for kk in range(R+1):
            sum_of_rest += np.exp(exponents[kk] - max_exponent)

        log_partition += np.log(sum_of_rest)

        return dot_product * datapoint[node] - np.log(float(factorial(datapoint[node]))) - log_partition, log_partition

    @staticmethod
    def calculate_grad_ll_datapoint(node, datapoint, theta, R, log_partition):
        grad = np.zeros(datapoint.shape)

        dot_product = np.dot(datapoint, theta) - theta[node] * datapoint[node] + theta[node]

        log_partition_derivative_term = 0
        for kk in range(1, R+1):
            log_partition_derivative_term += np.exp(dot_product * kk - np.log(float(factorial(kk))) + np.log(kk) - log_partition)

        grad[node] = datapoint[node] - log_partition_derivative_term
        for ii in range(datapoint.shape[0]):
            if ii != node:
                grad[ii] = datapoint[ii] * grad[node]

        return grad

    @staticmethod
    def calculate_nll_and_grad_nll_datapoint(node, datapoint, theta, R):
        ll, log_partition = Tpgm.calculate_ll_datapoint(node, datapoint, theta, R)
        grad_ll = Tpgm.calculate_grad_ll_datapoint(node, datapoint, theta, R, log_partition)

        return -ll, -grad_ll

    @staticmethod
    def calculate_nll_and_grad_nll(node, data, theta, R):
        N = data.shape[0]

        nll = 0
        grad_nll = np.zeros((data.shape[1], ))

        for ii in range(N):
            nll_ii, grad_nll_ii = Tpgm.calculate_nll_and_grad_nll_datapoint(node, data[ii, :], theta, R)
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
            nll -= Tpgm.calculate_ll_datapoint(node, data[ii, :], theta, R)[0]

        nll = nll/N

        return nll

    @staticmethod
    def provide_args(nr_nodes, other_args):
        for ii in range(nr_nodes):
            yield [ii] + other_args
        return

    # TODO: Add docstring + smart kwargs; parallelize for each node with multiprocessing.
    @staticmethod
    def fit(data, theta_init, alpha, R, condition=None, max_iter=5000, max_line_search_iter=50, lambda_p=1.0, beta=0.5,
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

        f = Tpgm.calculate_nll
        f_and_grad_f = Tpgm.calculate_nll_and_grad_nll

        tail_args = [theta_init, alpha, data, f, f_and_grad_f, [R], condition, max_iter, max_line_search_iter, lambda_p, beta,
                     rel_tol, abs_tol]
        nr_nodes = data.shape[1]

        with Pool(processes=4) as pool:
            return pool.starmap(prox_grad, Tpgm.provide_args(nr_nodes, tail_args))