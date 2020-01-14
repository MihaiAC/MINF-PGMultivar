import networkx as nx
import numpy as np
from mpgm.mpgm.auxiliary_methods import generate_scale_free_graph
from mpgm.mpgm.samplers import generate_samples_gibbs
from mpgm.mpgm.tpgm import Tpgm

# TODO: Add references to models' docstring.
# TODO: Draw theta ss too for tpgm parameters.
# TODO: Derive TPGM log likelihood derivative in dissertation.
# TODO: Need a general testing script (used to set up parameters + save/load the results). Doing it by command line
#   would be useful.
# TODO: Make sure that np.seed is initialised only once for the whole program (HERE).



def load_params_as_dict():
    pass

def save_model():
    pass

# TODO: Need to compare both graph structure and actual parameter values; recreate the graphs used by the papers.
def test_tpgm(params_file_path, model_save_path):

    # Placeholder for load_params:
    nr_variables = 50
    seed = 200

    weak_pos_mean = 0.04
    weak_neg_mean = -0.04
    weak_std = 0.03

    np.random.seed(seed)

    alpha = 0.15
    beta = 0.7
    gamma = 0.15

    generate_scale_free_graph(nr_variables, seed, weak_pos_mean, weak_neg_mean, weak_std, alpha, beta, gamma)

    model = Tpgm(theta=A, seed=seed, R=100)
    X = generate_samples_gibbs(model, np.random.randint(0, 3, A.shape[0]), nr_samples=200, burn_in=400, thinning_nr=70)
    np.save('tpgm_sample1.npy', X)

    return