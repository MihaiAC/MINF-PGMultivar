import networkx as nx
import numpy as np
from abc import ABC

# Every graph generator must have nr_variables as first argument, other arguments must be keyword arguments.
def generate_scale_free_graph(nr_variables, neg_percentage=0.5, pos_mean=0.04, neg_mean=-0.04, std=0.03, alpha=0.1,
                              beta=0.8, gamma=0.1):
    # alpha: add new node + edge to existing node
    # beta: add new edge between existing node
    # gamma: add new node + edge from existing node
    assert np.isclose(alpha + beta + gamma, 1), "alpha + beta + gamma must be equal to 1"

    G = nx.scale_free_graph(nr_variables, alpha, beta, gamma, seed=np.random)

    # Remove self-loop edges.
    sle = list(nx.selfloop_edges(G))
    for u, v in sle:
        G.remove_edge(u, v)

    # Remove multiedges (convert to a simple undirected graph).
    G = nx.Graph(G)

    for u, v in list(G.edges()):
        uniform_rv = np.random.uniform(0, 1, 1)[0]
        weight = 0
        if uniform_rv < neg_percentage:
            weight = np.random.normal(neg_mean, std)
        else:
            weight = np.random.normal(pos_mean, std)
        G[u][v]['weight'] = weight

    return nx.to_numpy_array(G)

def generate_lattice_graph(nr_variables, lattice_edge_weight=0.04, sparsity_level=0):

    # Get all divisors of nr_variables.
    divisors = []
    for ii in range(1, nr_variables+1):
        if(nr_variables % ii == 0):
            divisors.append(ii)

    # A sparsity level of 0 corresponds to selecting the two divisors m, n of nr_variables such that the m x n grid
    # graph contains the maximum number of edges, subject to m * n = nr_variables.
    # A sparsity level of 1 = selecting the next two divisors m, n of nr_variables such that m x n grid contains the
    # next possible highest number of edges (lower than the number of edges for sparsity_level = 0).

    if(len(divisors) % 2 == 1):
        start_index = (len(divisors) - 1) // 2
    else:
        start_index = (len(divisors) // 2) - 1

    for ii in range(sparsity_level):
        start_index -= 1
        if(start_index == 0):
            break

    m = divisors[start_index]
    n = divisors[len(divisors)-1-start_index]
    G = nx.grid_2d_graph(m, n)

    arr = nx.to_numpy_array(G)
    arr = arr * lattice_edge_weight

    return arr

if __name__ == '__main__':
    G = nx.grid_2d_graph(2, 5)
    for u, v in list(G.edges()):
        print(str(u) + ' ' + str(v))
    arr = nx.to_numpy_array(G)
    print(arr.shape)

