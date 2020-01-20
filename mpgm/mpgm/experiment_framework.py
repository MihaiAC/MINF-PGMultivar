import numpy as np
import pickle
import os
from mpgm.mpgm.samplers import Sampler
from mpgm.mpgm.tpgm import Tpgm
generate_gibbs_samples = Sampler.generate_samples_gibbs()

class Experiment:
    def __init__(self, data_file_path, model_to_fit, fit_params, random_seed, experiment_name, subexperiment_name):
        '''
        :param data_file_path: Location of the file containing the data we train our model on.
        :param model_to_fit: The model we fit.
        :param fit_params: Dictionary containing the arguments that must be passed to the model_fit function.
        :param random_seed: Seed for np.random for reproducibility.
        :param experiment_name: Name of the experiment. Two experiments are different if they use different sampled data.
        :param subexperiment_name: Subexperiments within an experiment.
        # TODO: Get rid of this variable. Check if there is a folder with the experiment name instead.
        :param samples_not_generated_before:
        '''
        self.data_file_path = data_file_path

        self.model_to_fit = model_to_fit
        self.fit_params = fit_params

        self.experiment_name = experiment_name
        self.subexperiment_name = subexperiment_name

        self.random_seed = random_seed
        np.random.seed(random_seed)

    def run_experiment(self):
        if(not os.path.isdir(self.experiment_name)):
            os.mkdir(self.experiment_name)

        folder_path = self.experiment_name + '/'


        data = np.load(self.data_file_path + '/generated_samples.npy')

        fit_data = self.model_to_fit.fit(data, **self.fit_params)
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
    random_seed = 195

    data_file_path = 'Samples/TPGM_scale_free/Sample_1/'
    with open(data_file_path + 'graph_sample_parameters.pkl', 'rb') as f:
        a = pickle.load(f)
        graph_generator_params = pickle.load(f)
        a = pickle.load(f)
        generator_model_params = pickle.load(f)
        a = pickle.load(f)
        sampling_method_params = pickle.load(f)

    nr_variables = graph_generator_params['nr_variables']
    nr_samples = sampling_method_params['nr_samples']
    R = generator_model_params['R']

    theta_init = np.zeros(nr_variables)
    alpha = np.sqrt(np.log(nr_variables)/nr_samples)
    max_iter = 7000
    max_line_search_iter = 50
    lambda_p = 1.0
    beta = 0.5
    rel_tol = 1e-6
    abs_tol = 1e-9
    condition = None
    model_to_fit = Tpgm
    fit_params = dict({'theta_init': theta_init, 'alpha': alpha, 'R': R, 'max_iter': max_iter,
                       'max_line_search_iter': max_line_search_iter, 'lambda_p': lambda_p, 'beta': beta,
                       'rel_tol': rel_tol, 'abs_tol': abs_tol, 'condition': condition})

    experiment_name = 'TPGM_experiment_2'
    subexperiment_name = 'run_5'


    exp = Experiment(model_to_fit, fit_params, random_seed, experiment_name, subexperiment_name)
    exp.run_experiment()
    #alpha_vals = [1/4 * alpha, 1/2 * alpha, alpha, alpha * 2, alpha * 4]
    #bools = [True, False, False, False, False]
    #for kk, alpha_val in enumerate(alpha_vals):
    #    subexperiment_name = 'run_' + str(kk)
    #    fit_params['alpha'] = alpha_val
    #    exp = Experiment(graph_generator, graph_generator_params, generator_model, generator_model_params, sampler,
    #                     sampler_params, model_to_fit, fit_params, random_seed, experiment_name, subexperiment_name,
    #                     bools[kk])
    #    exp.run_experiment()


test_tpgm()
