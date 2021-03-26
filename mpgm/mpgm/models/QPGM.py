import numpy as np
from mpgm.mpgm.models.Model import Model
from typing import Optional


class QPGM(Model):
    """
    Class for generating samples from and fitting QPGM distributions.

    :param theta: N x N matrix; theta[s][t] = weight of the edge between nodes (dimensions) s and t.
    :param theta_quad: 1 x N matrix; theta_quad[ss] = parameter of the quadratic term;
    """

    def __init__(self, theta:Optional[np.ndarray]=None):
        super().__init__(theta)

    # Generating samples from QPGM: undefined in the original paper.
    # def generate_node_sample(self, node, nodes_values):
    #     uu = np.random.uniform(0, 1, 1)[0]
    #
    #     prob1, partition_max_exp, partition_reduced, dot_product = self.node_cond_prob(node, 0, nodes_values)
    #     cdf = prob1
    #
    #     current_node_value = 1
    #     while uu > cdf:
    #         cdf += self.node_cond_prob(node, current_node_value, nodes_values, dot_product, partition_max_exp,
    #                                    partition_reduced)[0]
    #         current_node_value += 1
    #
    #     return current_node_value - 1
    #
    # def node_cond_prob(self, node, current_node_value, nodes_values, dot_product=None, partition_max_exp=None,
    #                    partition_reduced=None):
    #     """
    #     Calculates the probability of node having the provided value given the other nodes.
    #
    #     :param node: Integer representing the node we're interested in.
    #     :param nodes_values: Values of the other nodes.
    #     :param partition_max_exp: The partition is a sum of exponentials. partition_max_exp is the maximum exponent in
    #         this sum.
    #     :param partition_reduced: partition divided by exp(partition_max_exp).
    #     :return: A tuple containing the likelihood, the value of the partition function and the dot_product in this order.
    #     """
    #     if(dot_product == None):
    #         dot_product = np.dot(self.theta[node, :], nodes_values) - self.theta[node][node] * nodes_values[node] + \
    #             self.theta[node][node]
    #
    #     if partition_reduced is None:
    #         partition_exponents = []
    #         partition_max_exp = -np.inf
    #         kk = 0
    #         curr_exp = 1
    #
    #         while len(partition_exponents) < 200 or (len(partition_exponents) <= 500 and np.exp(curr_exp)/np.exp(partition_max_exp) > 1e-10):
    #             curr_exp = dot_product * kk + self.theta_quad[node] * (kk ** 2)
    #             partition_exponents.append(curr_exp)
    #             if partition_max_exp < curr_exp:
    #                 partition_max_exp = curr_exp
    #             kk += 1
    #
    #         partition_reduced = 0
    #         for exponent in partition_exponents:
    #             partition_reduced += np.exp(exponent - partition_max_exp)
    #
    #     cond_prob = np.exp(dot_product * current_node_value + self.theta_quad[node] * (current_node_value ** 2) - partition_max_exp)/partition_reduced
    #
    #     return cond_prob, partition_max_exp, partition_reduced, dot_product

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
        ll = ll - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(-2 * theta_quad) + 1/(4 * theta_quad) * (dot_product ** 2)

        return ll

    def calculate_grad_ll_datapoint(self, node:int, datapoint:np.ndarray, theta_curr:np.ndarray):
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
