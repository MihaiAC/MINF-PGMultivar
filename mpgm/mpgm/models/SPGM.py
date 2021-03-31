import numpy as np
from scipy.special import gammaln, logsumexp
from mpgm.mpgm.models.Model import Model

from typing import Optional, List, Tuple


class SPGM(Model):
    def __init__(self, theta:Optional[np.ndarray]=None, R:Optional[int]=10, R0:Optional[int]=5,
                 partition_atol:Optional[float]=1e-20, partition_max_iter:Optional[int]=10000):
        super().__init__(theta)
        self.R = R
        self.R0 = R0

        self.partition_atol = partition_atol
        self.partition_max_iter = partition_max_iter

        self.condition = None

    def __setattr__(self, key, value):
        if key == 'R':
            assert type(value) is int and value > 0, "R must be a positive integer"

        if key == 'R0':
            assert type(value) is int and value > 0 and value <= self.R, 'R0 must be an integer which satisfies 0 < R0 < R'

        super(SPGM, self).__setattr__(key, value)

    def sufficient_statistics(self, node_value):
        if(node_value <= self.R0):
            return node_value
        elif(node_value <= self.R):
            return (-0.5 * (node_value ** 2) + self.R * node_value - 0.5 * (self.R0 ** 2)) / (self.R - self.R0)
        else:
            return 0.5 * (self.R + self.R0)

    def estimate_partition_exponents(self, node:int, node_values_suffst:np.ndarray) -> List[float]:
        '''
        Estimates partition by summing terms until it converges or runs out of alotted iterations.
        '''
        first_exponent = 0
        atol = self.partition_atol
        max_iterations = self.partition_max_iter

        iterations = 0
        exponents = []
        exponent = first_exponent
        while (not np.isclose(np.exp(exponent), 0, atol=atol) and iterations < max_iterations):
            exponents.append(exponent)
            iterations += 1
            node_value = iterations

            exponent = (self.theta[node, node] + np.dot(self.theta[node, :], node_values_suffst) - \
                    self.theta[node, node] * node_values_suffst[node]) * self.sufficient_statistics(node_value) - \
                    gammaln(node_value+1)
        return exponents

    def node_cond_prob(self, node:int, node_value:int, node_values_suffst:np.ndarray, dot_product:Optional[float]=None,
                       log_partition:Optional[float]=None) -> Tuple[float, float, float]:
        if dot_product is None:
            dot_product = np.dot(self.theta[node, :], node_values_suffst) - self.theta[node, node] * \
                          node_values_suffst[node] + self.theta[node, node]

        if log_partition is None:
            partition_exponents = self.estimate_partition_exponents(node, node_values_suffst)
            log_partition = logsumexp(partition_exponents)

        cond_prob = np.exp(dot_product * self.sufficient_statistics(node_value) - gammaln(node_value+1) - log_partition)
        return cond_prob, dot_product, log_partition


    def calculate_ll_datapoint(self, node:int, datapoint:np.ndarray, node_theta_curr:np.ndarray) -> float:
        datapoint_sf = np.zeros((len(datapoint),))
        for ii in range(len(datapoint)):
            datapoint_sf[ii] = self.sufficient_statistics(datapoint[ii])

        dot_product = np.dot(datapoint_sf, node_theta_curr) - node_theta_curr[node] * datapoint_sf[node] + node_theta_curr[node]
        # if dot_product > 100:
        #     print(node)
        #     print(datapoint)
        #     print(node_theta_curr)
        #     print(dot_product)
        #     print('')
        log_partition = np.exp(dot_product)
        ll = dot_product * datapoint_sf[node] - gammaln(datapoint[node]+1) - log_partition

        return ll

    def calculate_grad_ll_datapoint(self, node:int, datapoint:np.ndarray, node_theta_curr:np.ndarray) -> np.ndarray:
        grad = np.zeros(datapoint.shape)

        datapoint_sf = np.zeros((len(datapoint), ))
        for ii in range(len(datapoint)):
            datapoint_sf[ii] = self.sufficient_statistics(datapoint[ii])

        dot_product = np.dot(datapoint_sf, node_theta_curr) - node_theta_curr[node] * datapoint_sf[node] + node_theta_curr[node]
        log_partition = np.exp(dot_product)

        grad[node] = datapoint_sf[node] - log_partition
        for ii in range(datapoint.shape[0]):
            if ii != node:
                grad[ii] = grad[node] * datapoint_sf[node]
        return grad
