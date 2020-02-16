import numpy as np
import pickle
import os
from mpgm.mpgm.TPGM import TPGM
from mpgm.mpgm.QPGM import QPGM
from mpgm.mpgm.SPGM import SPGM
from mpgm.mpgm.LPGM import LPGM

class Experiment:
    def __init__(self, samples_name, model_to_fit_name, experiment_name, random_seed=200, R=10,
                 R0 = 5, theta_init=None, theta_quad_init=None, alpha=None, lpgm_m=None, lpgm_B = 10, lpgm_beta=0.05,
                 nr_alphas=100, prox_grad_accelerated=False, max_iter=1000, max_line_search_iter=100,
                 prox_grad_lambda_p=1.0, prox_grad_beta=0.5, rel_tol=1e-6, abs_tol=1e-9):
        '''
        :param data_file_path: Location of the file containing the data we train our model on.
        :param model_to_fit: The model we fit.
        :param fit_params: Dictionary containing the arguments that must be passed to the model_fit function.
        :param random_seed: Seed for np.random for reproducibility.
        :param experiment_name: Name of the experiment. Two experiments are different if they use different sampled data.
        :param subexperiment_name: Subexperiments within an experiment.
        '''
        self.data_file_path = 'Samples/' + samples_name + '/'
        self.experiment_name = experiment_name

        np.random.seed(random_seed)

        self.data = np.load(self.data_file_path + 'samples.npy')
        nr_samples, nr_variables = self.data.shape

        if(theta_init is None):
            theta_init = np.zeros((nr_variables, nr_variables))

        if(theta_quad_init is None):
            theta_quad_init = -0.01 * np.ones((nr_variables, ))

        if(alpha == None):
            alpha = np.sqrt(np.log(nr_variables)/nr_samples)

        self.prox_grad_params = dict({'alpha': alpha, 'prox_grad_accelerated': prox_grad_accelerated,
                                      'max_iter': max_iter, 'max_line_search_iter': max_line_search_iter,
                                      'prox_grad_lambda_p': prox_grad_lambda_p, 'prox_grad_beta': prox_grad_beta,
                                       'rel_tol': rel_tol, 'abs_tol': abs_tol})

        self.model_to_fit_name = model_to_fit_name
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

    def run_experiment(self):
        fit_data = self.model_to_fit.fit(self.data, **self.prox_grad_params)

        experiment_folder = self.data_file_path + self.experiment_name
        if(not os.path.isdir(experiment_folder)):
            os.mkdir(experiment_folder)
        experiment_folder = experiment_folder + '/'

        self.save_experiment(experiment_folder, fit_data)

        print('Experiment ' + self.experiment_name + ' done!')

    def save_experiment(self, experiment_folder, fit_data):
        if(self.model_to_fit_name in ['TPGM', 'SPGM', 'QPGM']):
            fit_model_theta, likelihoods, conditions, converged = self.reshape_fit_data_TQSPGM(fit_data)
            np.save(experiment_folder + 'fit_model_theta.npy', fit_model_theta)

            with open(experiment_folder + 'likelihoods.pkl', 'wb') as f:
                pickle.dump(likelihoods, f)

            with open(experiment_folder + 'converged.pkl', 'wb') as f:
                pickle.dump(converged, f)

            if(self.model_to_fit_name == 'TPGM'):
                return

            with open(experiment_folder + 'conditions.pkl', 'wb') as f:
                pickle.dump(conditions, f)

        elif(self.model_to_fit_name == 'LPGM'):
            np.save(experiment_folder + 'optimal_alpha.npy', np.array([fit_data[0]]))
            np.save(experiment_folder + 'fit_model_theta.npy', fit_data[1])

        else:
            print('Invalid value for model_to_fit_name.')

    def reshape_fit_data_TQSPGM(self, fit_data):
        nr_variables = len(fit_data)

        fit_model_theta = np.zeros((nr_variables, nr_variables))
        likelihoods = []
        conditions = []
        converged = []

        for ii in range(len(fit_data)):
            fit_model_theta[ii, :] = fit_data[ii][0]
            likelihoods.append(fit_data[ii][1])
            conditions.append(fit_data[ii][2])
            converged.append(fit_data[ii][3])

        return fit_model_theta, likelihoods, conditions, converged

def run_test_tpgm_experiment():
    exp = Experiment('lattice_test', 'TPGM', 'TPGM_fit_test', alpha=0.1, theta_init=np.zeros((10, 10)), random_seed=1337,
                     prox_grad_accelerated=False)
    exp.run_experiment()

def run_test_qpgm_experiment():
    exp = Experiment('QPGM_test', 'QPGM', 'QPGM_fit_test', alpha=0.1, random_seed=123)
    exp.run_experiment()

def run_test_lpgm_experiment():
    exp = Experiment('test', 'LPGM', 'LPGM_fit_test', lpgm_m=24)
    exp.run_experiment()

if __name__ == '__main__':
    run_test_tpgm_experiment()