import numpy as np
import pickle
import os
from mpgm.mpgm.auxiliary_methods import generate_scale_free_graph
from mpgm.mpgm.samplers import generate_samples_gibbs
from mpgm.mpgm.tpgm import Tpgm
from mpgm.mpgm.solvers import prox_grad


class Experiment:
    def __init__(self, graph_generator, graph_generator_params, generator_model, generator_model_params, sampler,
                 sampler_params, model_to_fit, fit_params, random_seed, experiment_name, subexperiment_name,
                 samples_not_generated_before=True):

        self.graph_generator = graph_generator
        self.graph_generator_params = graph_generator_params

        self.generator_model = generator_model
        self.generator_model_params = generator_model_params

        self.sampler = sampler
        self.sampler_params = sampler_params

        self.model_to_fit = model_to_fit
        self.fit_params = fit_params

        self.experiment_name = experiment_name
        self.subexperiment_name = subexperiment_name

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.samples_not_generated_before = samples_not_generated_before

    def run_experiment(self):
        if self.samples_not_generated_before:
            os.mkdir(self.experiment_name)
        folder_path = self.experiment_name + '/'

        # Save experiment parameters.
        if self.samples_not_generated_before:
            with open(folder_path + 'graph_sample_parameters.pkl', 'wb') as f:
                pickle.dump(self.graph_generator.__name__, f)
                pickle.dump(self.graph_generator_params, f)
                pickle.dump(self.generator_model.__name__, f)
                pickle.dump(self.generator_model_params, f)
                pickle.dump(self.sampler.__name__, f)
                pickle.dump(self.sampler_params, f)

            generator_model_theta = self.graph_generator(*self.graph_generator_params)
            np.save(folder_path + 'generator_model_theta.npy', generator_model_theta)
            model = self.generator_model(generator_model_theta, *self.generator_model_params)
            generated_data = self.sampler(model, *self.sampler_params)
            np.save(folder_path + 'generated_data.npy', generated_data)

        else:
            generated_data = np.load(folder_path + 'generated_data.npy')

        fit_data = self.model_to_fit.fit(generated_data, *self.fit_params)
        with open(folder_path + self.subexperiment_name + '_fit_data.pkl', 'wb') as f:
            pickle.dump(self.fit_params, f)
            pickle.dump(fit_data, f)

        print('Experiment ' + self.experiment_name + ', subexperiment ' + self.subexperiment_name + ' done!')

# TODO: Add references to models' docstring.
# TODO: Draw theta ss too for tpgm parameters.
# TODO: Derive TPGM log likelihood derivative in dissertation.
# TODO: Need a general testing script (used to set up parameters + save/load the results). Doing it by command line
#   would be useful.
# TODO: Make sure that np.seed is initialised only once for the whole program (HERE).

# TODO: Need to compare both graph structure and actual parameter values; recreate the graphs used by the papers.

def test_tpgm():
    random_seed = 200

    nr_variables = 50
    weak_pos_mean = 0.04
    weak_neg_mean = -0.04
    weak_std = 0.3
    alpha = 0.05
    beta = 0.9
    gamma = 0.05
    graph_generator_params = (nr_variables, weak_pos_mean, weak_neg_mean, weak_std, alpha, beta, gamma)
    graph_generator = generate_scale_free_graph
    print(graph_generator.__name__)

    R = 50
    generator_model = Tpgm
    generator_model_params = [R]

    sampler_init = np.random.randint(1, 3, nr_variables)
    nr_samples = 200
    burn_in = 100
    thinning_nr = 70
    sampler_params = (sampler_init, nr_samples, burn_in, thinning_nr)
    sampler = generate_samples_gibbs

    theta_init = np.zeros(nr_variables)
    alpha = np.sqrt(np.log(nr_variables)/nr_samples)
    max_iter = 5000
    max_line_search_iter = 50
    lambda_p = 1.0
    beta = 0.5
    rel_tol = 1e-3
    abs_tol = 1e-6
    model_to_fit = Tpgm
    fit_params = (theta_init, alpha, R, max_iter, max_line_search_iter, lambda_p, beta, rel_tol, abs_tol)

    experiment_name = 'TPGM_experiment_1'
    subexperiment_name = 'run_2'

    exp = Experiment(graph_generator, graph_generator_params, generator_model, generator_model_params, sampler,
                     sampler_params, model_to_fit, fit_params, random_seed, experiment_name, subexperiment_name, False)
    exp.run_experiment()


#test_tpgm()
