import numpy as np

def assign_weights_bimodal_distr(graph, neg_mean=-0.04, pos_mean=0.04, neg_threshold=0.5, std=0.03):
    nr_variables = graph.shape[0]

    for ii in range(1, nr_variables):
        for jj in range(ii):
            if(graph[ii][jj] != 0):
                uu = np.random.uniform(0, 1)
                if(uu < neg_threshold):
                    weight = np.random.normal(neg_mean, std)
                else:
                    weight = np.random.normal(pos_mean, std)
                graph[ii][jj] = weight
                graph[jj][ii] = weight