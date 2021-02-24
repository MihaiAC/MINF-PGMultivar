import numpy as np
from mpgm.mpgm.models.Model import Model
from scipy.special import gammaln, logsumexp
from typing import Tuple, Optional


class TPGM(Model):
    """
    Class for generating samples from and fitting TPGM distributions.

    :param theta: N x N matrix; theta[s][t] = weight of the edge between nodes (dimensions) s and t.
    :param R: maximum count (truncation value).
    :param data: Data to fit the model with.
    """

    def __init__(self, theta:Optional[np.ndarray]=None, R:Optional[int]=10):
        """
        Constructor for TPGM class.

        :param theta: N x N matrix, given when we want to generate samples from an existing distribution.
                      N x N array, representing the starting theta values for when we want to fit the model.
        :param R: maximum count value, should be an integer.
        """
        super().__init__(theta)
        self.R = R
        # TODO: See if commenting the line below broke anything.
        # self.theta = theta

    def __setattr__(self, key, value):
        if key == 'R':
            assert type(value) is int and value > 0, "R must be a positive integer"
        super(TPGM, self).__setattr__(key, value)

    def node_cond_prob(self, node:int, node_value:int, data:np.ndarray, dot_product:Optional[float]=None,
                       partition:Optional[float]=None) -> \
            Tuple[float, float, float]:
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

        if partition is None:
            partition_exponents = np.zeros((self.R+1, ))
            for kk in range(self.R+1):
                partition_exponents[kk] = dot_product * kk - gammaln(kk+1)

            partition = logsumexp(partition_exponents)

        cond_prob = np.exp(dot_product * node_value - gammaln(node_value+1) - partition)
        return cond_prob, dot_product, partition

    def log_node_cond_prob(self, node:int, node_value:int, data:np.ndarray, dot_product:Optional[float]=None,
                       partition:Optional[float]=None) -> \
            Tuple[float, float, float]:
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

        if partition is None:
            partition_exponents = np.zeros((self.R+1, ))
            for kk in range(self.R+1):
                partition_exponents[kk] = dot_product * kk - gammaln(kk+1)

            partition = logsumexp(partition_exponents)

        cond_prob = dot_product * node_value - gammaln(node_value+1) - partition
        return cond_prob, dot_product, partition

    def calculate_ll_datapoint(self, node:int, datapoint:np.ndarray, node_theta_curr:np.ndarray) -> float:
        """
        :return: returns (nll_datapoint, log_partition)
        """
        dot_product = np.dot(datapoint, node_theta_curr) - node_theta_curr[node] * datapoint[node] + node_theta_curr[node]

        partition_exponents = np.zeros((self.R+1, ))
        for kk in range(self.R+1):
            partition_exponents[kk] = dot_product * kk - gammaln(kk+1)

        log_partition = logsumexp(partition_exponents)
        ll = dot_product * datapoint[node] - gammaln(datapoint[node]+1) - log_partition

        return ll

    def calculate_grad_ll_datapoint(self, node:int, datapoint:np.ndarray, node_theta_curr:np.ndarray) -> np.ndarray:
        grad = np.zeros(datapoint.shape)

        dot_product = np.dot(datapoint, node_theta_curr) - node_theta_curr[node] * datapoint[node] + node_theta_curr[node]

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

        # TODO: Does this work as expected (reference-wise?)
        grad[node] = datapoint[node] - log_partition_derivative_term
        for ii in range(datapoint.shape[0]):
            if ii != node:
                grad[ii] = datapoint[ii] * grad[node]

        return grad
