import networkx as nx
import numpy as np


def generate_scale_free_graph(nr_variables, weak_pos_mean, weak_neg_mean, weak_std, alpha, beta, gamma):
    # alpha: add new node + edge to existing node
    # beta: add new edge between existing node
    # gamma: add new node + edge from existing node

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
        if uniform_rv < 0.5:
            weight = np.random.normal(weak_neg_mean, weak_std)
        else:
            weight = np.random.normal(weak_pos_mean, weak_std)
        G[u][v]['weight'] = weight

    return nx.to_numpy_array(G)

