import networkx as nx
import numpy as np

# Every graph generator must have nr_variables as first argument, other arguments must be keyword arguments.
# Every method must return a square, symmetric numpy array, having entry (i,j) nonzero if and only if the graph
# has an edge from node i to node j.
def generate_scale_free_graph(nr_variables, alpha=0.1, beta=0.8, gamma=0.1):
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

    return nx.to_numpy_array(G)

def generate_lattice_graph(nr_variables, sparsity_level=0):

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

    return arr

def generate_hub_graph(nr_variables, nr_hubs=3):
    assert nr_variables >= nr_hubs, 'There cannot be more hubs than nodes.'

    graph = np.zeros((nr_variables, nr_variables))

    # Select which nodes will become the hubs.
    hubs = np.random.choice(range(nr_variables), nr_hubs, replace=False)

    # Construct the edges of the graph.
    for node in range(nr_variables):
        if node in hubs:
            continue
        assigned_hub = np.random.choice(hubs, 1, replace=True)[0]
        graph[node][assigned_hub] = 1
        graph[assigned_hub][node] = 1

    return graph

def generate_random_nm_graph(nr_variables, nr_edges=0):
    G = nx.gnm_random_graph(nr_variables, nr_edges, seed=np.random)
    G = nx.Graph(G)
    return nx.to_numpy_array(G)

if __name__ == '__main__':
    print(generate_random_nm_graph(10, 3))


