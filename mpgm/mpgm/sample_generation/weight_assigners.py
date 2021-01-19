import numpy as np
import random
from typing import List, Optional, Tuple
from math import floor, ceil

class Weight_Assigner():
    def __init__(self):
        pass

    def assign_weights(self, graph):
        return graph

class Bimodal_Distr_Weight_Assigner(Weight_Assigner):
    def __init__(self, neg_mean:Optional[float]=-0.04, pos_mean:Optional[float]=0.04,
                 threshold:Optional[float]=0.5, std:Optional[float]=0.02):
        super().__init__()
        self.neg_mean = neg_mean
        self.pos_mean = pos_mean
        self.threshold = threshold
        self.std = std

    def assign_weights(self, graph:np.ndarray):
        nr_variables = graph.shape[0]

        for ii in range(1, nr_variables):
            for jj in range(ii):
                if (graph[ii][jj] != 0):
                    uu = np.random.uniform(0, 1)
                    if (uu < self.threshold):
                        weight = np.random.normal(self.neg_mean, self.std)
                    else:
                        weight = np.random.normal(self.pos_mean, self.std)
                    graph[ii][jj] = weight
                    graph[jj][ii] = weight

class Bimodal_Gaussian_Weight_Assigner(Weight_Assigner):
    def __init__(self, mean_1:float, std_1:float, mean_2:float, std_2:float, split:float):
        '''
        :param mean_1, std_1: Mean and std of the first Gaussian.
        :param mean_2, std+2: Mean and std of the second Gaussian.
        :param split: Proportion of edges whose weights will be drawn from the first Gaussian.
        '''
        super().__init__()
        self.mean_1 = mean_1
        self.std_1 = std_1
        self.mean_2 = mean_2
        self.std_2 = std_2
        self.split = split

    def __setattr__(self, key, value):
        if key == 'split':
            assert value >= 0 and value <= 1, "Split must be a number in [0,1]."
        super(Bimodal_Gaussian_Weight_Assigner, self).__setattr__(key, value)

    @staticmethod
    def get_nonzero_edges(graph:np.ndarray) -> List[Tuple[int, int]]:
        nr_variables = graph.shape[0]
        edges = []
        for ii in range(nr_variables):
            for jj in range(ii+1, nr_variables):
                if graph[ii][jj] != 0:
                    edges.append((ii, jj))
        return edges

    def assign_weights(self, graph:np.ndarray):
        edges = Bimodal_Gaussian_Weight_Assigner.get_nonzero_edges(graph)
        nr_edges = len(edges)

        nr_edges_1 = floor(self.split * nr_edges)
        edges_1 = set(random.sample(edges, nr_edges_1))
        for (ii, jj) in edges:
            if (ii, jj) in edges_1:
                weight = np.random.normal(self.mean_1, self.std_1)
            else:
                weight = np.random.normal(self.mean_2, self.std_2)
            graph[ii][jj] = weight
            graph[jj][ii] = weight


class Constant_Weight_Assigner(Weight_Assigner):
    def __init__(self, ct_weight:float):
        super().__init__()
        self.ct_weight = ct_weight

    def assign_weights(self, graph:np.ndarray):
        nr_variables = graph.shape[0]

        for ii in range(1, nr_variables):
            for jj in range(ii):
                if (graph[ii][jj] != 0):
                    graph[ii][jj] = self.ct_weight
                    graph[jj][ii] = self.ct_weight


class Dummy_Weight_Assigner(Weight_Assigner):
    """
    Leaves the graph as is; used for compatibility reasons.
    """
    def __init__(self):
        super().__init__()

    def assign_weights(self, graph:np.ndarray):
        pass