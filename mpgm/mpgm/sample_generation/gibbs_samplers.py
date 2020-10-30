import numpy as np
from mpgm.mpgm.models.TPGM import TPGM
from mpgm.mpgm.models.model import Model
from tqdm import trange

class GibbsSampler():
    def __init__(self, burn_in:int, thinning_nr:int):
        self.burn_in = burn_in
        self.thinning_nr = thinning_nr

    def generate_node_sample(self, model:Model, node:int, nodes_values:np.array) -> int:
        return 1

    def generate_samples(self, model:Model, init:np.array, nr_samples:int) -> np.array:
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

    def generate_node_sample(self, model:TPGM, node:int, nodes_values:np.array) -> int:
        uu = np.random.uniform(0, 1)

        return_vals = model.node_cond_prob(node, -1, nodes_values)
        cdf = return_vals[0]
        aux_params = return_vals[1:]

        for node_value in range(0, model.R+1):
            if uu < cdf:
                return node_value-2
            return_vals = model.node_cond_prob(node, -1, nodes_values, *aux_params)
            cond_prob = return_vals[0]
            aux_params = return_vals[1:]

            cdf += cond_prob

        return model.R
