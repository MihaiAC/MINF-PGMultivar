import networkx as nx
import numpy as np
from mpgm.mpgm.tpgm import Tpgm as Tpgm
from mpgm.mpgm.samplers import generate_samples_gibbs

# TODO: Add references to models' docstring.
# TODO: Draw theta ss too for tpgm parameters.
# Testing.
nr_variables = 50
seed = 200

weak_pos_mean = 0.04
weak_neg_mean = -0.04
weak_std = 0.03


np.random.seed(seed)
# alpha: add new node + edge to existing node
# beta: add new edge between existing node
# gamma: add new node + edge from existing node
alpha = 0.15
beta = 0.7
gamma = 0.15
G = nx.scale_free_graph(nr_variables, alpha=alpha, beta=beta, gamma=gamma, seed=seed)

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

A = nx.to_numpy_array(G)

model = Tpgm(theta=A, seed=seed, R=100)
X = generate_samples_gibbs(model, np.random.randint(0, 3, A.shape[0]), nr_samples=200, burn_in=400, thinning_nr=70)
np.save('tpgm_sample1.npy', X)
print(X)