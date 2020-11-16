import numpy as np
from mpgm.mpgm.models.model import Model
from scipy.special import gammaln
from typing import Tuple


class TPGM(Model):
    """
    Class for generating samples from and fitting TPGM distributions.

    :param theta: N x N matrix; theta[s][t] = weight of the edge between nodes (dimensions) s and t.
    :param R: maximum count (truncation value).
    :param data: Data to fit the model with.
    """

    def __init__(self, theta:np.array=None, R:int=100):
        """
        Constructor for TPGM class.

        :param theta: N x N matrix, given when we want to generate samples from an existing distribution.
                      N x N array, representing the starting theta values for when we want to fit the model.
        :param R: maximum count value, should be an integer.
        """
        super().__init__(theta)
        self.R = R
        self.theta = theta

    def __setattr__(self, key, value):
        if key == 'R':
            assert type(value) is int and value > 0, "R must be a positive integer"
            assert value <= 100, "The model has not been tested with R values higher than 100"
        super(TPGM, self).__setattr__(key, value)

    def node_cond_prob(self, node:int, node_value:int, data:np.array, dot_product:float=None,
                       partition_max_exp:float=None, partition_reduced:float=None) -> Tuple[float, float, float, float]:
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

    def calculate_ll_datapoint(self, node:int, datapoint:np.array, node_theta_curr:np.array) -> float:
        """
        :return: returns (nll_datapoint, log_partition)
        """

        log_partition = 0

        exponents = []
        dot_product = np.dot(datapoint, node_theta_curr) - node_theta_curr[node] * datapoint[node] + node_theta_curr[node]
        for kk in range(self.R+1):
            curr_exponent = dot_product * kk - gammaln(kk+1)
            exponents.append(curr_exponent)

        max_exponent = max(exponents)
        log_partition += max_exponent

        sum_of_rest = 0
        for kk in range(self.R+1):
            sum_of_rest += np.exp(exponents[kk] - max_exponent)

        log_partition += np.log(sum_of_rest)

        ll = dot_product * datapoint[node] - gammaln(datapoint[node]+1) - log_partition
        return ll

    def calculate_grad_ll_datapoint(self, node:int, datapoint:np.array, theta_curr:np.array) -> np.array:
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
