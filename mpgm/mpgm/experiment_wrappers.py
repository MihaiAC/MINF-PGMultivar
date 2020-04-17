import numpy as np
import pickle
import os
from mpgm.mpgm.TPGM import TPGM
from mpgm.mpgm.QPGM import QPGM
from mpgm.mpgm.SPGM import SPGM
from mpgm.mpgm.LPGM import LPGM

class Fit_Wrapper:
    def __init__(self, experiment_name, samples_name, random_seed, **prox_grad_params):
        self.data_file_path = 'Samples/' + samples_name + '/'
        self.experiment_name = experiment_name
        self.model_to_fit = None
        self.data = np.load(self.data_file_path + 'samples.npy')

        self.prox_grad_params = Fit_Wrapper.generate_model_prox_grad_params_dict(**prox_grad_params)
        self.prox_grad_params['data'] = self.data

        if random_seed is not None:
            np.random.seed(random_seed)

    def run_experiment(self):
        fit_data = self.model_to_fit.fit(self.prox_grad_params)

        experiment_folder = self.data_file_path + self.experiment_name
        if(not os.path.isdir(experiment_folder)):
            os.mkdir(experiment_folder)
        experiment_folder = experiment_folder + '/'

        self.save_experiment(experiment_folder, fit_data)

        print('Fit_Wrapper ' + self.experiment_name + ' done!')

    def save_experiment(self, experiment_folder, fit_data):
        if isinstance(self.model_to_fit, TPGM) or isinstance(self.model_to_fit, QPGM) or isinstance(self.model_to_fit, SPGM):
            fit_model_theta, likelihoods, conditions, converged = self.reshape_fit_data_TQSPGM(fit_data)
            np.save(experiment_folder + 'fit_model_theta.npy', fit_model_theta)

            with open(experiment_folder + 'likelihoods.pkl', 'wb') as f:
                pickle.dump(likelihoods, f)

            with open(experiment_folder + 'converged.pkl', 'wb') as f:
                pickle.dump(converged, f)

            if(isinstance(self.model_to_fit, TPGM)):
                return

            with open(experiment_folder + 'conditions.pkl', 'wb') as f:
                pickle.dump(conditions, f)

        elif isinstance(self.model_to_fit, LPGM):
            np.save(experiment_folder + 'optimal_alpha.npy', np.array([fit_data[0]]))
            np.save(experiment_folder + 'fit_model_theta.npy', fit_data[1])
        else:
            print('Invalid value for model_to_fit_name.')

    def reshape_fit_data_TQSPGM(self, fit_data):
        nr_variables = len(fit_data)

        fit_model_theta = np.zeros((nr_variables, len(fit_data[0][0])))
        likelihoods = []
        conditions = []
        converged = []

        for ii in range(len(fit_data)):
            fit_model_theta[ii, :] = fit_data[ii][0]
            likelihoods.append(fit_data[ii][1])
            conditions.append(fit_data[ii][2])
            converged.append(fit_data[ii][3])

        return fit_model_theta, likelihoods, conditions, converged

    @staticmethod
    def generate_model_prox_grad_params_dict(alpha=0.01, accelerated=True, max_iter=1000, line_search_rel_tol=1e-4,
                                            max_line_search_iter=100, lambda_p=1.0, beta=0.5, rel_tol=1e-6, abs_tol=1e-9):

        model_prox_grad_params = dict({'alpha': alpha, 'accelerated': accelerated, 'max_iter': max_iter,
                                 'max_line_search_iter': max_line_search_iter, 'line_search_rel_tol': line_search_rel_tol,
                                  'lambda_p': lambda_p, 'beta': beta, 'rel_tol': rel_tol, 'abs_tol': abs_tol})

        return model_prox_grad_params



class TPGM_FitWrapper(Fit_Wrapper):
    def __init__(self, experiment_name, samples_name, random_seed, R, theta_init=None, **model_prox_grad_params):
        super(TPGM_FitWrapper, self).__init__(experiment_name, samples_name, random_seed, **model_prox_grad_params)

        nr_samples, nr_variables = self.data.shape

        if theta_init is None:
            theta_init = np.zeros((nr_variables, nr_variables))
        self.model_to_fit = TPGM(theta_init, R)

class QPGM_FitWrapper(Fit_Wrapper):
    def __init__(self, experiment_name, samples_name, random_seed, theta_init=None, theta_quad_init=None, qtp_c=1e4,
                 **model_prox_grad_params):
        super(QPGM_FitWrapper, self).__init__(experiment_name, samples_name, random_seed, **model_prox_grad_params)
        self.prox_grad_params['qtp_c'] = qtp_c

        nr_samples, nr_variables = self.data.shape

        if theta_init is None:
            theta_init = np.zeros((nr_variables, nr_variables))

        if theta_quad_init is None:
            theta_quad_init = -0.01 * np.ones((nr_variables, ))

        self.model_to_fit = QPGM(theta_init, theta_quad_init)

class SPGM_FitWrapper(Fit_Wrapper):
    def __init__(self, experiment_name, samples_name, random_seed, R, R0, theta_init=None, **model_prox_grad_params):
        super(SPGM_FitWrapper, self).__init__(experiment_name, samples_name, random_seed, **model_prox_grad_params)

        nr_samples, nr_variables = self.data.shape

        if theta_init is None:
            theta_init = np.zeros((nr_variables, nr_variables))

        self.model_to_fit = SPGM(theta_init, R, R0)

class LPGM_FitWrapper(Fit_Wrapper):
    pass
'''
                 alpha=None, lpgm_m=None, lpgm_B = 10, lpgm_beta=0.05,


        elif(model_to_fit_name == 'LPGM'):
            seeds = np.random.choice(list(range(nr_variables * 10000)), size=nr_variables, replace=False)
            if(lpgm_m == None):
                lpgm_m = int(np.floor(10 * np.sqrt(nr_variables)))
            self.prox_grad_params.pop('alpha')
            self.model_to_fit = LPGM(lpgm_m, lpgm_B, lpgm_beta, nr_alphas, seeds)'''


def run_test_tpgm_experiment(samples_name, experiment_name):
    exp = TPGM_FitWrapper(experiment_name, samples_name, random_seed=1337, R=10, alpha=0.2, accelerated=True,
                          max_line_search_iter=400, line_search_rel_tol=1e-3)
    exp.run_experiment()

def run_test_qpgm_experiment():
    exp = Fit_Wrapper('QPGM_test', 'QPGM', 'QPGM_fit_test', alpha=0.1, random_seed=123, prox_grad_accelerated=True)
    exp.run_experiment()

def run_test_lpgm_experiment():
    exp = Fit_Wrapper('test', 'LPGM', 'LPGM_fit_test', lpgm_m=24)
    exp.run_experiment()

def tpgm_node_cond_prob_sum_check():
    exp = TPGM_FitWrapper('TPGM_fit_test', 'TPGM_burn_in_200_thinning_nr_6400', random_seed=1337, R=10, alpha=0.8, accelerated=True,
                          max_line_search_iter=200, line_search_rel_tol=1e-1)

    sample = exp.data[0, :]
    print(sample)
    node = 0
    R = 10

    sum_cond_prob = 0
    for node_value in range(R+1):
        sum_cond_prob += exp.model_to_fit.node_cond_prob(node, node_value, sample, None, None, None)[0]

    print(sum_cond_prob)
    # Sanity check successful.

def run_test_tpgm_vary_alpha():
    alphas = np.linspace(1e-4, 1, 50)
    samples_name = 'TPGM_test'

    for idx, alpha in enumerate(alphas):
        experiment_name = 'TPGM_fit_test_alpha_' + str(idx)
        exp = TPGM_FitWrapper(experiment_name, samples_name, random_seed=1337, R=10, alpha=alpha,
                              accelerated=True, max_line_search_iter=400, line_search_rel_tol=1e-3)
        exp.run_experiment()


if __name__ == '__main__':
    run_test_tpgm_experiment('TPGM_test_moderate_params_sparse', 'TPGM_fit_test')
