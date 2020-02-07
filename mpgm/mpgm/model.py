import numpy as np


class Model():
    def calculate_nll_and_grad_nll_datapoint(self, node, datapoint, theta_curr):
        pass

    def calculate_ll_datapoint(self, node, datapoint, theta_curr):
        pass

    def calculate_nll_and_grad_nll(self, node, data, theta_curr):
        N = data.shape[0]

        nll = 0
        grad_nll = np.zeros((data.shape[1], ))

        for ii in range(N):
            nll_ii, grad_nll_ii = self.calculate_nll_and_grad_nll_datapoint(node, data[ii, :], theta_curr)
            nll += nll_ii
            grad_nll += grad_nll_ii

        nll = nll/N
        grad_nll = grad_nll/N

        return nll, grad_nll

    def calculate_nll(self, node, data, theta_curr):
        N = data.shape[0]

        nll = 0

        for ii in range(N):
            nll += -self.calculate_ll_datapoint(node, data[ii, :], theta_curr)[0]

        nll = nll/N

        return nll

    @staticmethod
    def provide_args(nr_nodes, other_args):
        for ii in range(nr_nodes):
            yield [ii] + other_args
        return
