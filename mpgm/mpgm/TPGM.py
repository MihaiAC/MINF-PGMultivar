import numpy as np
from multiprocessing import Pool
from mpgm.mpgm.model import Model
from scipy.special import gammaln
from tqdm import tqdm


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
                      N x N array, representing the starting theta values for when we want to fit the model.
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

        uu = np.random.uniform(0, 1)

        prob1, partition_max_exp, partition_reduced, dot_product = self.node_cond_prob(node, 0, nodes_values)
        cdf = prob1

        for node_value in range(1, self.R + 1):
            if uu < cdf:
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

        min_exp = dot_product * node_value - gammaln(node_value+1)
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

        # TODO: remove debug statement.
        ll = dot_product * datapoint[node] - gammaln(datapoint[node] + 1) - log_partition
        return ll, log_partition

    def calculate_grad_ll_datapoint(self, node, datapoint, theta_curr, log_partition):
        grad = np.zeros(datapoint.shape)

        dot_product = np.dot(datapoint, theta_curr) - theta_curr[node] * datapoint[node] + theta_curr[node]

        #log_partition_derivative_term = 0
        #for kk in range(1, self.R+1):
        #    log_partition_derivative_term += np.exp(dot_product * kk - gammaln(kk+1) + np.log(kk) - log_partition)
        exponents_numerator = []
        exponents_denominator = []
        exponents_denominator.append(-gammaln(1))
        for kk in range(1, self.R+1):
            exponents_numerator.append(dot_product * kk - gammaln(kk+1) + np.log(kk))
            exponents_denominator.append(dot_product * kk - gammaln(kk+1))

        max_exponent_numerator = max(exponents_numerator)
        max_exponent_denominator = max(exponents_denominator)

        sum_numerator = 0
        sum_denominator = 0
        for ii in range(self.R):
            sum_numerator += np.exp(exponents_numerator[ii] - max_exponent_numerator)
            sum_denominator += np.exp(exponents_denominator[ii] - max_exponent_denominator)
        sum_denominator += np.exp(exponents_denominator[self.R] - max_exponent_denominator)

        log_partition_derivative_term = np.exp(max_exponent_numerator - max_exponent_denominator) * \
                                        (sum_numerator/sum_denominator)

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
    def fit(self, prox_grad_params):
        model_params = [self.theta, self.R]

        nr_nodes = prox_grad_params['data'].shape[1]

        ordered_results = [()] * nr_nodes
        with Pool(processes=4) as pool:
            for result in tqdm(pool.imap_unordered(TPGM.call_prox_grad_wrapper,
                                                   iter(((model_params, prox_grad_params, x) for x in range(nr_nodes))),
                                                   chunksize=1), total=nr_nodes):
                ordered_results[result[0]] = result[1]

        return ordered_results

        #with Pool(processes=4) as pool:
        #    args = list(Model.provide_args(nr_nodes, tail_args))
        #    return pool.starmap(prox_grad, args)
