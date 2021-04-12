import networkx as nx
import numpy as np

# TODO: Must initialise np.random.seed before using these classes.
class GraphGenerator():
    def __init__(self):
        pass

    def generate_graph(self, nr_variables):
        pass

class ScaleFreeGraphGenerator(GraphGenerator):
    def __init__(self, alpha=0.41, beta=0.54, gamma=0.05):
        super().__init__()
        # Parameter explanations taken from the NetworkX documentation:
        # https://networkx.org/documentation/stable//reference/generated/networkx.generators.directed.scale_free_graph.html

        # "Probability for adding a new node connected to an existing node chosen randomly according to the in-degree
        # distribution."
        self.alpha = alpha

        # "Probability for adding an edge between two existing nodes."
        self.beta = beta

        # "Probability for adding a new node connected to an existing node chosen randomly according to the
        # out-degree distribution."
        self.gamma = gamma

        assert np.isclose(alpha + beta + gamma, 1), "ScaleFreeGraphGenerator: alpha + beta + gamma must sum up to 1"

    def generate_graph(self, nr_variables):
        G = nx.scale_free_graph(nr_variables, self.alpha, self.beta, self.gamma, seed=np.random)
        G = self._remove_self_loop_edges(G)

        np_G = nx.to_numpy_array(G)
        del G

        return np_G


    def _remove_self_loop_edges(self, G):
        sle = list(nx.selfloop_edges(G))

        for u, v in sle:
            G.remove_edge(u, v)

        return G

class LatticeGraphGenerator(GraphGenerator):
    # TODO: Seems like a fake parameter - its effect is negligible.
    def __init__(self, sparsity_level=0):
        # A sparsity level of 0 corresponds to selecting the two divisors m, n of nr_variables such that the m x n grid
        # graph contains the maximum number of edges, subject to m * n = nr_variables.
        # A sparsity level of 1 = selecting the next two divisors m, n of nr_variables such that m x n grid contains the
        # next possible highest number of edges (lower than the number of edges for sparsity_level = 0).
        super().__init__()
        self.sparsity_level = sparsity_level

    @staticmethod
    def _calculate_divisors(number):
        divisors = []
        for ii in range(1, number+1):
            if number % ii == 0:
                divisors.append(ii)
        return divisors

    @staticmethod
    def _calculate_start_index(divisors):
        if len(divisors) % 2 == 1:
            start_index = (len(divisors) - 1) // 2
        else:
            start_index = (len(divisors) // 2) - 1
        return start_index

    def generate_graph(self, nr_variables):
        nr_variables_divisors = LatticeGraphGenerator._calculate_divisors(nr_variables)
        start_index = LatticeGraphGenerator._calculate_start_index(nr_variables_divisors)

        for ii in range(self.sparsity_level):
            start_index = start_index - 1
            if start_index == 0:
                break

        side_1 = nr_variables_divisors[start_index]
        side_2 = nr_variables_divisors[len(nr_variables_divisors) - 1 - start_index]
        G = nx.grid_2d_graph(side_1, side_2)

        np_G = nx.to_numpy_array(G)
        del G

        return np_G

class HubGraphGenerator(GraphGenerator):
    def __init__(self, nr_hubs):
        super().__init__()
        self.nr_hubs = nr_hubs


    def generate_graph(self, nr_variables):
        assert nr_variables >= self.nr_hubs, 'HubGraphGenerator: There cannot be more hubs than nodes.'

        graph = np.zeros((nr_variables, nr_variables))

        # Select the nodes which will become the hubs.
        hubs = np.random.choice(range(nr_variables), self.nr_hubs, replace=False)

        # Construct the edges of the graph.
        for node in range(nr_variables):
            if node in hubs:
                continue

            assigned_hub = np.random.choice(hubs, 1, replace=True)[0]
            graph[node][assigned_hub] = 1
            graph[assigned_hub][node] = 1

        return graph

class RandomNmGraphGenerator(GraphGenerator):
    def __init__(self, nr_edges):
        super().__init__()
        self.nr_edges = nr_edges

    def generate_graph(self, nr_variables):
        G = nx.gnm_random_graph(nr_variables, self.nr_edges, seed=np.random)
        G = nx.Graph(G)
        np_G = nx.to_numpy_array(G)
        return np_G



