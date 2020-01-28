import numpy as np
import pickle
import os
from mpgm.mpgm.samplers import Sampler
from mpgm.mpgm.TPGM import TPGM
from mpgm.mpgm.QPGM import QPGM
from mpgm.mpgm.SPGM import SPGM
from mpgm.mpgm.LPGM import LPGM
generate_gibbs_samples = Sampler.generate_samples_gibbs()

class Experiment:
    def __init__(self, data_file_path, model_to_fit_name, experiment_name, subexperiment_name, random_seed=200, R=10,
                 R0 = 5, theta_init=None, theta_quad_init=None, alpha=None, lpgm_m=None, lpgm_B = 10, lpgm_beta=0.05,
                 nr_alphas=100, max_iter=1000, max_line_search_iter=100, prox_grad_lambda_p=1.0, prox_grad_beta=0.5,
                 rel_tol=1e-6, abs_tol=1e-9):
        '''
        :param data_file_path: Location of the file containing the data we train our model on.
        :param model_to_fit: The model we fit.
        :param fit_params: Dictionary containing the arguments that must be passed to the model_fit function.
        :param random_seed: Seed for np.random for reproducibility.
        :param experiment_name: Name of the experiment. Two experiments are different if they use different sampled data.
        :param subexperiment_name: Subexperiments within an experiment.
        '''
        self.data_file_path = data_file_path
        self.experiment_name = experiment_name
        self.subexperiment_name = subexperiment_name

        np.random.seed(random_seed)

        self.data = np.load(self.data_file_path + '/generated_samples.npy')
        nr_samples, nr_variables = self.data.shape

        if(theta_init == None):
            theta_init = np.random.normal(0, 0.0001, nr_variables)

        if(theta_quad_init == None):
            theta_quad_init = -0.5 * np.ones((nr_variables, ))

        if(alpha == None):
            alpha = np.sqrt(np.log(nr_variables)/nr_samples)

        self.prox_grad_params = dict({'alpha': alpha, 'max_iter': max_iter, 'max_line_search_iter': max_line_search_iter,
                                 'prox_grad_lambda_p': prox_grad_lambda_p, 'prox_grad_beta': prox_grad_beta,
                                 'rel_tol': rel_tol, 'abs_tol': abs_tol})

        if(model_to_fit_name == 'TPGM'):
            self.model_to_fit = TPGM(theta_init, R)
        elif(model_to_fit_name == 'QPGM'):
            self.model_to_fit = QPGM(theta_init, theta_quad_init)
        elif(model_to_fit_name == 'SPGM'):
            self.model_to_fit = SPGM(theta_init, R, R0)
        elif(model_to_fit_name == 'LPGM'):
            seeds = np.random.choice(list(range(nr_variables * 10000)), size=nr_variables, replace=False)
            if(lpgm_m == None):
                lpgm_m = int(np.floor(10 * np.sqrt(nr_variables)))
            self.prox_grad_params.pop('alpha')
            self.model_to_fit = LPGM(lpgm_m, lpgm_B, lpgm_beta, nr_alphas, seeds)

        self.experiment_name = experiment_name
        self.subexperiment_name = subexperiment_name

    def run_experiment(self):
        if(not os.path.isdir(self.experiment_name)):
            os.mkdir(self.experiment_name)

        folder_path = self.experiment_name + '/'

        fit_data = self.model_to_fit.fit(self.data, **self.prox_grad_params)
        with open(folder_path + self.subexperiment_name + '_fit_data.pkl', 'wb') as f:
            pickle.dump(self.model_to_fit, f)
            pickle.dump(self.prox_grad_params, f)
            pickle.dump(fit_data, f)

        print('Experiment ' + self.experiment_name + ', subexperiment ' + self.subexperiment_name + ' done!')
