import numpy as np
from multiprocessing import Pool
from scipy.special import gammaln
from mpgm.mpgm.solvers import prox_grad


class Spgm(object):
    """
    Class for generating samples from and fitting SPGM distributions.
    :param R: model parameter
    :param R0: model parameter
    :param data: Data to fit the model with.
    """

    def __init__(self, theta=None, R=100, R0=50):
        """
        Constructor for SPGM class.

        :param theta: N x N matrix, given when we want to generate samples from an existing distribution.
        :param R: maximum count value, should be an integer.
        """
        self.R = R
        self.R0 = R0
        self.theta = np.array(theta)

    def __setattr__(self, key, value):
        if key == 'R':
            assert type(value) is int and value > 0, "R must be a positive integer"

            #TODO: Test the performance of the TPGM model for different R values
            assert value <= 100, "The model has not been tested with R values higher than 100"

        if key == 'R0':
            assert type(value) is int and value > 0 and value < self.R, 'R0 must be a positive integer, smaller than R'

        super(Spgm, self).__setattr__(key, value)

    @staticmethod
    def sufficient_statistics(node_value, R, R0):
        if(node_value <= R0):
            return node_value
        elif(node_value <= R):
            return (-0.5 * (node_value ** 2) + R * node_value - 0.5 * (R0 ** 2)) / (R - R0)
        else:
            return 0.5 * (R + R0)

    def generate_node_sample(self, node, nodes_values):
        """
        Generate a sample for node from its node-conditional probability.

        :param node: Node to generate sample for.
        :param data: Current values of the nodes (ordered).
        :return: A sample from the node-conditional probability of our node.
        """
        nodes_values_suffst = []
        for node_value in nodes_values:
            nodes_values_suffst.append(Spgm.sufficient_statistics(node_value, self.R, self.R0))

        uu = np.random.uniform(0, 1, 1)[0]

        prob1, partition_max_exp, partition_reduced, dot_product = self.node_cond_prob(node, 0, nodes_values_suffst)
        cdf = prob1

        node_value = 1
        while(True):
            if uu < cdf:
                return node_value-1
            cdf += self.node_cond_prob(node, node_value, nodes_values, dot_product, partition_max_exp, partition_reduced)[0]
            node_value += 1

    def calculate_partition(self, node, node_values_suffst):
        first_exponent = (node_values_suffst[node] + np.dot(self.theta[node, :], node_values_suffst) - \
                    self.theta[node, node] * node_values_suffst[node]) * node_values_suffst[1] - gammaln(2)

        atol = np.exp(first_exponent) * 1e-20
        max_iterations = 1000

        iterations = 0
        exponents = []
        exponent = first_exponent
        max_exponent = -np.inf
        while (not np.isclose(np.exp(exponent), 0, atol=atol) and iterations < max_iterations):
            exponents.append(exponent)
            iterations += 1

            exponent = (node_values_suffst[node] + np.dot(self.theta[node, :], node_values_suffst) - \
                    self.theta[node, node] * node_values_suffst[node]) * node_values_suffst[iterations+1] - \
                    gammaln(iterations+2)

            if(exponent > max_exponent):
                max_exponent = exponent

        return exponents, max_exponent

    def node_cond_prob(self, node, node_value, node_values_suffst, dot_product=None, partition_max_exp=None, partition_reduced=None):
        """
        Calculates the probability of node having the provided value given the other nodes.

        :param node: Integer representing the node we're interested in.
        :param node_values_suffst: 1 X P array containing the values of the sufficient statistics of the other nodes;
        :param node_value: Value of the node we're interested in.
        :param partition: Optional argument, equal to the value of the partition function. Relevant only for sampling.
        :param dot_product: Inner dot product present in the partition function and denominator of the conditional
            probability.
        :return: A tuple containing the likelihood, the value of the partition function and the dot_product in this order.
        """
        if dot_product is None:
            dot_product = np.dot(self.theta[node, :], node_values_suffst) - self.theta[node, node] * \
                          node_values_suffst[node] + self.theta[node, node]

        if partition_reduced is None:
            partition_exponents, partition_max_exp = self.calculate_partition(node, node_values_suffst)
            partition_reduced = 0
            for kk in range(len(partition_exponents)):
                partition_reduced += np.exp(partition_exponents[kk] - partition_max_exp)

        cond_prob = np.exp(dot_product * node_value - gammaln(node_value) - partition_max_exp)/partition_reduced

        return cond_prob, partition_max_exp, partition_reduced, dot_product

    @staticmethod
    def calculate_ll_datapoint(node, datapoint, theta, R, R0):
        """
        :return: returns (nll_datapoint, log_partition)
        """
        datapoint_sf = np.zeros((len(datapoint), ))
        for ii in range(len(datapoint)):
            datapoint_sf[ii] = Spgm.sufficient_statistics(datapoint[ii], R, R0)

        dot_product = np.dot(datapoint_sf, theta) - theta[node] * datapoint_sf[node] + theta[node]

        log_partition = np.exp(dot_product)

        return dot_product * datapoint_sf[node] - gammaln(datapoint[node]) - log_partition, log_partition, datapoint_sf

    @staticmethod
    def calculate_grad_ll_datapoint(node, datapoint, theta, R, R0, log_partition, datapoint_sf):
        grad = np.zeros(datapoint.shape)

        grad[node] = datapoint_sf[node] - log_partition
        for ii in range(datapoint.shape[0]):
            if ii != node:
                grad[ii] = grad[node] * datapoint_sf[node]

        return grad

    @staticmethod
    def calculate_nll_and_grad_nll_datapoint(node, datapoint, theta, R, R0):
        ll, log_partition, datapoint_sf = Spgm.calculate_ll_datapoint(node, datapoint, theta, R, R0)
        grad_ll = Spgm.calculate_grad_ll_datapoint(node, datapoint, theta, R, log_partition, datapoint_sf)

        return -ll, -grad_ll

    @staticmethod
    def calculate_nll_and_grad_nll(node, data, theta, R, R0):
        N = data.shape[0]

        nll = 0
        grad_nll = np.zeros((data.shape[1], ))

        for ii in range(N):
            nll_ii, grad_nll_ii = Spgm.calculate_nll_and_grad_nll_datapoint(node, data[ii, :], theta, R, R0)
            nll += nll_ii
            grad_nll += grad_nll_ii

        nll = nll/N
        grad_nll = grad_nll/N

        return nll, grad_nll

    @staticmethod
    def calculate_nll(node, data, theta, R, R0):
        N = data.shape[0]

        nll = 0

        for ii in range(N):
            nll -= Spgm.calculate_ll_datapoint(node, data[ii, :], theta, R, R0)[0]

        nll = nll/N

        return nll

    @staticmethod
    def provide_args(nr_nodes, other_args):
        for ii in range(nr_nodes):
            yield [ii] + other_args
        return

    @staticmethod
    def condition(node, theta, data, R, R0):
        conditions = 0
        for kk in range(data.shape[0]):
            datapoint = data[kk, :]
            datapoint_sf = np.zeros((len(datapoint), ))
            for ii, node_value in enumerate(datapoint):
                datapoint_sf[ii] = Spgm.sufficient_statistics(node_value, R, R0)
            dot_product = np.dot(theta, datapoint_sf) - theta[node] * datapoint_sf[node] + theta[node]
            if(dot_product >= 0):
                conditions += 1
        return conditions

    # TODO: Add docstring + smart kwargs; parallelize for each node with multiprocessing.
    @staticmethod
    def fit(data, theta_init, alpha, R, R0, condition, max_iter=5000, max_line_search_iter=50, lambda_p=1.0, beta=0.5,
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

        f = Spgm.calculate_nll
        f_and_grad_f = Spgm.calculate_nll_and_grad_nll

        tail_args = [theta_init, alpha, data, f, f_and_grad_f, [R, R0], condition, max_iter, max_line_search_iter, lambda_p, beta,
                     rel_tol, abs_tol]
        nr_nodes = data.shape[1]

        with Pool(processes=4) as pool:
            return pool.starmap(prox_grad, Spgm.provide_args(nr_nodes, tail_args))