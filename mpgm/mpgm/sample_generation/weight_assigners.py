import numpy as np

class Weight_Assigner():
    def __init__(self):
        pass

    def assign_weights(self, graph):
        return graph

class Bimodal_Distr_Weight_Assigner(Weight_Assigner):
    def __init__(self, neg_mean=-0.04, pos_mean=0.04, threshold=0.5, std=0.02):
        super().__init__()
        self.neg_mean = neg_mean
        self.pos_mean = pos_mean
        self.threshold = threshold
        self.std = std

    def assign_weights(self, graph):
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

class Dummy_Weight_Assigner(Weight_Assigner):
    """
    Leaves the graph as is; used for compatibility reasons.
    """
    def __init__(self):
        super().__init__()

    def assign_weights(self, graph):
        pass