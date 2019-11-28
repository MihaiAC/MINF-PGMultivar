import numpy as np
from tqdm import trange


def generate_samples_gibbs(model, init, nr_samples=50, burn_in=700, thinning_nr=100):
    """
    Gibbs sampler for this implementation of the TPGM distribution.

    :param model: model to draw samples from, must implement "generate_sample_cond_prob(node, data)"
    :param init: 1xN np array = starting value for the sampling process;
    :param nr_samples: number of samples we want to generate.
    :param burn_in: number of initial samples to be discarded.
    :param thinning_nr: we select every thinning_nr samples after the burn-in period;

    :return: (nr_samples X N) matrix containing one sample per row.
    """
    N = np.shape(init)[0]
    samples = np.zeros((nr_samples, N))
    data = np.array(init)

    for sample_nr in trange(burn_in + (nr_samples - 1) * thinning_nr + 1):
        # Generate one sample.
        for node in range(N):
            node_sample = model.generate_sample_cond_prob(node, data)
            data[node] = node_sample

        # Check if we should keep this sample.
        if sample_nr >= burn_in and (sample_nr - burn_in) % thinning_nr == 0:
            samples[(sample_nr - burn_in) // thinning_nr, :] = data

    return samples
