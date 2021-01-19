import numpy as np
from typing import Tuple

# All model subclasses that permit sampling, must have a parameter named theta.
class Model():
    def __init__(self, theta:np.ndarray):
        self.theta = theta

    def node_cond_prob(self, node:int, node_value:int, data=np.ndarray) -> Tuple[float, float, float]:
        pass

    def calculate_nll_and_grad_nll_datapoint(self, node, datapoint, theta_curr):
        pass

    def calculate_ll_datapoint(self, node, datapoint, theta_curr):
        pass

    def calculate_grad_ll_datapoint(self, node:int, datapoint:np.ndarray, theta_curr:np.ndarray) -> np.ndarray:
        pass

    def calculate_nll_and_grad_nll(self, node, data, theta_curr):
        N = data.shape[0]

        nll = 0
        grad_nll = np.zeros((len(theta_curr), ))

        for ii in range(N):
            nll_ii, grad_nll_ii = self.calculate_nll_and_grad_nll_datapoint(node, data[ii, :], theta_curr)
            nll += nll_ii
            grad_nll += grad_nll_ii

        nll = nll/N
        grad_nll = grad_nll/N

        return nll, grad_nll

    def calculate_grad_nll(self, node, data, theta_curr):
        N = data.shape[0]
        grad_nll = np.zeros((len(theta_curr), ))

        for ii in range(N):
            grad_nll_ii = (-1) * self.calculate_grad_ll_datapoint(node, data[ii, :], theta_curr)
            grad_nll += grad_nll_ii

        grad_nll = grad_nll/N

        return grad_nll

    def calculate_nll(self, node, data, theta_curr):
        N = data.shape[0]

        nll = 0

        for ii in range(N):
            nll += (-1) * self.calculate_ll_datapoint(node, data[ii, :], theta_curr)

        nll = nll/N

        return nll

    def calculate_joint_nll(self, data):
        nr_nodes = data.shape[1]

        joint_nll = 0
        for node in range(nr_nodes):
            joint_nll += self.calculate_nll(node, data, self.theta[node, :])

        return joint_nll

    @staticmethod
    def provide_args(nr_nodes, other_args):
        for ii in range(nr_nodes):
            yield [ii] + other_args
        return

    @staticmethod
    def prox_operator(x, threshold):
        """
        Applies the soft thresholding operation to vector x with the specified threshold.

        :return:
        """
        x_plus = x - threshold
        x_minus = -x - threshold
        return x_plus * (x_plus > 0) - x_minus * (x_minus > 0)

    @classmethod
    def call_prox_grad_wrapper(cls, packed_params):
        # model_params = packed_params[0]
        # prox_grad_params = packed_params[1], dictionary.
        # node = packed_params[2]

        model = cls(*packed_params[0])
        return packed_params[2], cls.fit_prox_grad(packed_params[2], model, **packed_params[1])
