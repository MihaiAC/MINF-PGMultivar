import numpy as np

class weight_assigner():
    def __init__(self):
        pass

    def assign_weights(self, graph):
        return graph

class bimodal_distr_weight_assigner(weight_assigner):
    def __init__(self, neg_mean=-0.04, pos_mean=0.04, neg_threshold=0.5, std=0.03):
        super().__init__()
        self.neg_mean = neg_mean
        self.pos_mean = pos_mean
        self.neg_threshold = neg_threshold
        self.std = std

    def assign_weights(self, graph):
        nr_variables = graph.shape[0]

        for ii in range(1, nr_variables):
            for jj in range(ii):
                if (graph[ii][jj] != 0):
                    uu = np.random.uniform(0, 1)
                    if (uu < self.neg_threshold):
                        weight = np.random.normal(self.neg_mean, self.std)
                    else:
                        weight = np.random.normal(self.pos_mean, self.std)
                    graph[ii][jj] = weight
                    graph[jj][ii] = weight


