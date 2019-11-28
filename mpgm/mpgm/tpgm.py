import numpy as np
from decimal import Decimal


class Tpgm(object):
    """
    Class for generating samples from and fitting TPGM distributions.

    :param N: dimensionality of our random variables;
    :param theta: N x N matrix; theta[s][t] = weight of the edge between nodes (dimensions) s and t.
    :param R: maximum count (truncation value).
    :param log_factorials: Log factorials of integers up to R.
    :param data: Data to fit the model with.
    :param seed: Seed used for np.random.
    """

    def __init__(self, theta=None, data=None, R=100, seed=None):
        """
        Constructor for TPGM class.

        :param theta: N x N matrix, given when we want to generate samples from an existing distribution.
        :param data: M x N matrix, one row is equivalent to one sample of the distribution to be fitted.
        :param R: maximum count value, should be an integer.
        """

        self.R = R
        self.log_factorials = Tpgm.calculate_log_factorials_upto(R)

        if theta is not None:
            self.N = len(theta[0])
            self.theta = np.array(theta)

        if data is not None:
            self.data = np.array(data)
            self.data[self.data >= self.R] = self.R
            self.N = len(data[0])

        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

    def __setattr__(self, key, value):
        if key == 'R':
            assert type(value) is int and value > 0, "R must be a positive integer"

            #TODO: Test the performance of the TPGM model for different R values
            assert value <= 100, "The model has not been tested with R values higher than 100"
        super(Tpgm, self).__setattr__(key, value)

    def generate_sample_cond_prob(self, node, data):
        """
        Generate a sample for node from its node-conditional probability.

        :param node: Node to generate sample for.
        :param data: Current values of the nodes (ordered).
        :param seed: Optional parameter, sets the seed of np.random.
        :return: A sample from the node-conditional probability of our node.
        """

        uu = np.random.uniform(0, 1, 1)[0]
        uu = Decimal(uu)

        prob1, partition, dot_product = self.node_cond_prob(node, 0, data)
        cdf = prob1

        for node_value in range(1, self.R + 1):
            #print(type(uu))
            #print(type(cdf))
            if uu.compare(Decimal(cdf)) == Decimal('-1'):
                return node_value-1
            cdf += self.node_cond_prob(node, node_value, data, partition, dot_product)[0]

        return self.R

    def node_cond_prob(self, node, node_value, data, partition=None, dot_product=None):
        """
        Calculates the probability of node having the provided value given the other nodes.

        :param node: Integer representing the node we're interested in.
        :param data: 1 X N array containing the values of the nodes (including our node).
        :param node_value: Value of the node we're interested in.
        :param partition: Optional argument, equal to the value of the partition function. Relevant only for sampling.
        :param dot_product: Inner dot product present in the partition function and denominator of the conditional
            probability.
        :return: A tuple containing the likelihood, the value of the partition function and the dot_product in this order.
        """
        if dot_product is None:
            dot_product = np.dot(self.theta[node, :], data)

        if partition is None:
            partition = 0
            for kk in range(self.R+1):
                partition += np.exp(dot_product * kk - self.log_factorials[kk])

        cond_prob = np.exp(dot_product * node_value - self.log_factorials[node_value])/partition

        return cond_prob, partition, dot_product

    @staticmethod
    def calculate_log_factorials_upto(R):
        # TODO: May need to remove this function for large R.
        """
        Self-explanatory.

        :param R: Positive integer value up to which to calculate log factorials.
        :return: np.array, containing log[i!] at index i.
        """

        log_factorials = np.zeros(R+1)
        log_factorials[0] = 0.0
        log_sum = 0

        for ii in range(1, R+1):
            log_sum += np.log(ii)
            log_factorials[ii] = log_sum

        return log_factorials

