import numpy as np
from decimal import Decimal
from multiprocessing import Pool
from mpgm.mpgm.solvers import prox_grad
from mpgm.mpgm.model import Model
from scipy.special import gammaln


class TPGM(Model):
    """
    Class for generating samples from and fitting TPGM distributions.

    :param theta: N x N matrix; theta[s][t] = weight of the edge between nodes (dimensions) s and t.
    :param R: maximum count (truncation value).
    :param data: Data to fit the model with.
    """

    def __init__(self, theta=None, R=100):
        """
        Constructor for TPGM class.

        :param theta: N x N matrix, given when we want to generate samples from an existing distribution.
        :param R: maximum count value, should be an integer.
        """
        self.R = R
        self.theta = theta
        self.condition = None # Condition to be checked after each iteration of prox grad.

    def __setattr__(self, key, value):
        if key == 'R':
            assert type(value) is int and value > 0, "R must be a positive integer"
            assert value <= 100, "The model has not been tested with R values higher than 100"
        super(TPGM, self).__setattr__(key, value)

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
                partition_exponents[kk] = dot_product * kk - gammaln(kk+1)

            partition_max_exp = max(partition_exponents)
            partition_reduced = 0
            for kk in range(self.R+1):
                partition_reduced += np.exp(partition_exponents[kk] - partition_max_exp)

        cond_prob = np.exp(dot_product * node_value - gammaln(node_value+1) - partition_max_exp)/partition_reduced

        return cond_prob, partition_max_exp, partition_reduced, dot_product

    def calculate_ll_datapoint(self, node, datapoint, theta_curr):
        """
        :return: returns (nll_datapoint, log_partition)
        """

        log_partition = 0

        exponents = []
        dot_product = np.dot(datapoint, theta_curr) - theta_curr[node] * datapoint[node] + theta_curr[node]
        for kk in range(self.R+1):
            curr_exponent = dot_product * kk - gammaln(kk+1)
            exponents.append(curr_exponent)

        max_exponent = max(exponents)
        log_partition += max_exponent

        sum_of_rest = 0
        for kk in range(self.R+1):
            sum_of_rest += np.exp(exponents[kk] - max_exponent)

        log_partition += np.log(sum_of_rest)

        return dot_product * datapoint[node] - gammaln(datapoint[node] + 1) - log_partition, log_partition

    def calculate_grad_ll_datapoint(self, node, datapoint, theta_curr, log_partition):
        grad = np.zeros(datapoint.shape)

        dot_product = np.dot(datapoint, theta_curr) - theta_curr[node] * datapoint[node] + theta_curr[node]

        log_partition_derivative_term = 0
        for kk in range(1, self.R+1):
            log_partition_derivative_term += np.exp(dot_product * kk - gammaln(kk+1) + np.log(kk) - log_partition)

        grad[node] = datapoint[node] - log_partition_derivative_term
        for ii in range(datapoint.shape[0]):
            if ii != node:
                grad[ii] = datapoint[ii] * grad[node]

        return grad

    def calculate_nll_and_grad_nll_datapoint(self, node, datapoint, theta_curr):
        ll, log_partition = self.calculate_ll_datapoint(node, datapoint, theta_curr)
        grad_ll = self.calculate_grad_ll_datapoint(node, datapoint, theta_curr, log_partition)

        return -ll, -grad_ll

    # TODO: Need theta_init, R as initial model parameters.
    def fit(self, data, alpha, max_iter=5000, max_line_search_iter=50, prox_grad_lambda_p=1.0, prox_grad_beta=0.5,
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

        tail_args = [self, data, alpha, max_iter, max_line_search_iter, prox_grad_lambda_p, prox_grad_beta, rel_tol,
                     abs_tol]
        nr_nodes = data.shape[1]

        with Pool(processes=4) as pool:
            return pool.starmap(prox_grad, Model.provide_args(nr_nodes, tail_args))