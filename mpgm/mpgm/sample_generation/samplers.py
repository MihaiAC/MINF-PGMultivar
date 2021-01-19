import numpy as np
from mpgm.mpgm.models.TPGM import TPGM
from mpgm.mpgm.models.SPGM import SPGM
from mpgm.mpgm.models.Model import Model
from tqdm import trange
from scipy.special import logsumexp

class Sampler():
    def generate_samples(self, model:Model, init:np.ndarray, nr_samples:int) -> np.ndarray:
        pass

class SIPRVSampler(Sampler):
    def __init__(self, lambda_true:float, lambda_noise:float):
        self.lambda_true = lambda_true
        self.lambda_noise = lambda_noise

    def generate_samples(self, model:Model, init:np.ndarray, nr_samples:int) -> np.ndarray:
        """
        :param model: Must be a default model; model.theta must be the adjacency matrix.
        :param init: Only interested in its length, which must be the nr of variables.
        :param nr_samples: Number of samples we want to generate.
        :return: Generated samples.
        """
        A = model.theta
        n = nr_samples
        p = A.shape[0]

        B = SIPRVSampler.create_B(A)

        C = self.lambda_true * A
        upper_tri = C[np.triu_indices(p, 1)]
        nonzero_tri = upper_tri[upper_tri != 0]
        Y_lambdas = np.append(np.repeat(self.lambda_true, p), nonzero_tri)
        Y = np.zeros((len(Y_lambdas), n))
        for ii, ylam in enumerate(Y_lambdas):
            Y[ii, :] = np.random.poisson(ylam, (1, n))

        X = np.matmul(B, Y)  # X is p x n
        X = np.transpose(X)  # X is now n x p

        # Add noise.
        E = np.random.poisson(self.lambda_noise, (n, p))
        X = X + E
        return X


    @staticmethod
    def create_B(adj_matrix: np.ndarray) -> np.ndarray:
        A = adj_matrix
        nrows = A.shape[0]
        ncols = A.shape[1]
        if nrows != ncols:
            raise Exception("create_B: Adjacency matrix is not square.")

        B = np.identity(nrows)
        for ii in range(nrows):
            for jj in range(ii + 1, ncols):
                if A[ii][jj] == 1:
                    new_col = np.zeros((nrows, 1))
                    new_col[ii] = 1
                    new_col[jj] = 1
                    B = np.hstack((B, new_col))

        return B

class GibbsSampler(Sampler):
    def __init__(self, burn_in:int, thinning_nr:int):
        self.burn_in = burn_in
        self.thinning_nr = thinning_nr

    def __setattr__(self, key, value):
        if key == 'thinning_nr':
            assert type(value) is int and value > 0, "The thinning number must be a positive integer."
        super(GibbsSampler, self).__setattr__(key, value)


    def generate_node_sample(self, model:Model, node:int, nodes_values:np.ndarray) -> int:
        pass

    def generate_samples(self, model:Model, init:np.ndarray, nr_samples:int) -> np.ndarray:
        """
        :param model: model to generate samples from;
        :param init: (N, ) numpy array; = starting value of the sampling process
        :param nr_samples: number of samples we want to generate
        :return:  (nr_samples x N) matrix containing one sample per row.
        """
        nr_variables = len(init)
        samples = np.zeros((nr_samples, nr_variables))
        nodes_values = np.array(init)

        for sample_nr in trange(self.burn_in + (nr_samples-1) * self.thinning_nr + 1):
            # Generate one sample.
            for node in range(nr_variables):
                node_sample = self.generate_node_sample(model, node, nodes_values)
                nodes_values[node] = node_sample

            # Check if this sample should be kept.
            if sample_nr >= self.burn_in and (sample_nr - self.burn_in) % self.thinning_nr == 0:
                samples[(sample_nr - self.burn_in) // self.thinning_nr, :] = nodes_values

        return samples

class TPGMGibbsSampler(GibbsSampler):
    def __init__(self, burn_in:int, thinning_nr:int):
        super().__init__(burn_in, thinning_nr)

    def generate_node_sample(self, model:TPGM, node:int, nodes_values:np.ndarray) -> int:
        uu = np.random.uniform(0, 1)

        return_vals = model.node_cond_prob(node, 0, nodes_values)
        cdf = return_vals[0]
        aux_params = return_vals[1:]

        for node_value in range(1, model.R+1):
            if uu < cdf:
                return node_value-1
            return_vals = model.node_cond_prob(node, node_value, nodes_values, *aux_params)
            cond_prob = return_vals[0]
            aux_params = return_vals[1:]

            cdf += cond_prob

        return model.R

class SPGMGibbsSampler(GibbsSampler):
    def __init__(self, burn_in:int, thinning_nr:int):
        super().__init__(burn_in, thinning_nr)

    def generate_node_sample(self, model:SPGM, node:int, nodes_values:np.ndarray):
        nodes_values_suffst = []
        for node_value in nodes_values:
            nodes_values_suffst.append(model.sufficient_statistics(node_value))
        nodes_values_suffst = np.array(nodes_values_suffst)

        uu = np.random.uniform(0, 1, 1)[0]

        # # TODO: remove debug.
        # print("Uniform shit is: " + str(uu))

        return_vals = model.node_cond_prob(node, 0, nodes_values_suffst)
        cdf = return_vals[0]
        aux_params = return_vals[1:]

        node_value = 1
        while (True):
            if uu < cdf:
                # # TODO: remove debug
                # print("generate_node_sample: node=" + str(node) + "node_value=" + str(node_value-1))
                return node_value - 1
            return_vals = model.node_cond_prob(node, node_value, nodes_values_suffst, *aux_params)
            cond_prob = return_vals[0]
            aux_params = return_vals[1:]

            cdf += cond_prob
            node_value += 1

            # There's an infinite loop here. Why?


            # # TODO: remove debug
            # print(return_vals)
            # print(cdf)
