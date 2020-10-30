import numpy as np
from multiprocessing import Pool
from mpgm.mpgm.models.model import Model

class LPGM(Model):
    def __init__(self, lpgm_m, lpgm_B, lpgm_beta, nr_alphas, seeds):
        self.lpgm_m = lpgm_m
        self.lpgm_B = lpgm_B
        self.lpgm_beta = lpgm_beta
        self.nr_alphas = nr_alphas
        self.seeds = seeds
        self.condition = None


    def calculate_nll_and_grad_nll(self, node, data, theta_curr):
        '''
        :param Y: Predictor variable, 1 x N.
        :param W: X.
        :param beta: The parameters we are learning. 1 x p. beta[ii] is ignored.
        :return:
        '''
        Y = data[:, node]
        W = data
        beta = theta_curr
        theta_curr[node] = 0

        nll = 0
        N = len(Y)

        dots_Wi_beta = np.dot(data, theta_curr)

        nll = -np.dot(Y, dots_Wi_beta) + np.sum(np.exp(dots_Wi_beta))
        grad_nll = -np.dot(Y - np.exp(dots_Wi_beta), W)

        nll = (nll/N) - 1 # exp(W[ii, node] * theta[node]) = exp(0) = 1; removing this.
        grad_nll = grad_nll/N
        grad_nll[node] = 0

        return nll, grad_nll

    def calculate_nll(self, node, data, theta_curr):
        Y = data[:, node]
        W = data
        beta = theta_curr
        theta_curr[node] = 0

        nll = 0
        N = len(Y)

        dots_Wi_beta = np.dot(data, theta_curr)

        nll -= np.dot(Y, dots_Wi_beta) + np.sum(np.exp(dots_Wi_beta))
        nll = nll/N - 1 #

        return nll

    def generate_alpha_values(self, data):
        alpha_min = 0.0001

        alpha_max = 0
        n_rows, n_cols = data.shape
        for ii in range(1, n_cols):
            for jj in range(0, ii):
                value = np.sum(data[:, ii] * data[:, jj])
                if(value > alpha_max):
                    alpha_max = value

        alpha_vals = np.logspace(np.log10(alpha_max), np.log10(alpha_min), self.nr_alphas)
        return alpha_vals

    def fit_node(self, node, data, alpha_values, max_iter, max_line_search_iter, lambda_p, beta, rel_tol, abs_tol):

        prox_grad_unmutable_params = [max_iter, max_line_search_iter, lambda_p, beta, rel_tol, abs_tol]

        thetas_main = np.zeros((len(alpha_values), data.shape[1]))
        thetas_subsamples = np.zeros((self.lpgm_B, len(alpha_values), data.shape[1]))

        local_rng = np.random.RandomState(self.seeds[node])

        theta_init = local_rng.normal(0, 0.0001, data.shape[1])
        self.theta = theta_init
        for ii, alpha in enumerate(alpha_values):
            theta_warm = prox_grad(node, self, data, alpha, *prox_grad_unmutable_params)[0]
            thetas_main[ii, :] = theta_warm
            self.theta = theta_warm

        for bb in range(self.lpgm_B):
            theta_init = local_rng.normal(0, 0.0001, data.shape[1])
            self.theta = theta_init
            subsample_indices = local_rng.choice(data.shape[0], self.lpgm_m, replace=False)
            subsampled_data = data[subsample_indices, :]
            for ii, alpha in enumerate(alpha_values):
                theta_warm = prox_grad(node, self, subsampled_data, alpha, *prox_grad_unmutable_params)[0]
                thetas_subsamples[bb, ii, :] = theta_warm
                self.theta = theta_warm

        print('Node ' + str(node) + ' done.')
        return thetas_main, thetas_subsamples

    @staticmethod
    def make_matrices(all_thetas, p, M, B):
        thetas_main = np.zeros((M, p, p))
        thetas_subsamples = np.zeros((M, B, p, p))

        for ii in range(len(all_thetas)):
            for alpha_index in range(M):
                thetas_main[alpha_index, ii, :] = all_thetas[ii][0][alpha_index, :]
                for b in range(B):
                    thetas_subsamples[alpha_index, b, ii, :] = all_thetas[ii][1][b, alpha_index, :]

        return thetas_main, thetas_subsamples

    @staticmethod
    def make_ahat(theta):
        p = theta.shape[0]
        ahat = np.zeros(theta.shape)
        for ii in range(1, p):
            for jj in range(ii):
                ahat[ii][jj] = max(np.abs(np.sign(theta[ii, jj])), np.abs(np.sign(theta[jj, ii])))
        return ahat

    @staticmethod
    def make_ahats(thetas_main, thetas_subsamples):
        ahats_main = np.zeros(thetas_main.shape)
        ahats_subsamples = np.zeros(thetas_subsamples.shape)

        M = thetas_main.shape[0]
        B = thetas_subsamples.shape[1]

        for m in range(M):
            ahats_main[m, :, :] = LPGM.make_ahat(thetas_main[m, :, :])
            for b in range(B):
                ahats_subsamples[m, b, :, :] = LPGM.make_ahat(thetas_subsamples[m, b, :, :])

        return ahats_main, ahats_subsamples

    @staticmethod
    def make_abars_subsamples(ahats_subsamples):
        M, B, p, _ = ahats_subsamples.shape
        abars_subsamples = np.zeros((M, p, p))

        for m in range(M):
            for b in range(B):
                abars_subsamples[m, :, :] += ahats_subsamples[m, b, :, :]
            abars_subsamples[m, :, :] = abars_subsamples[m, :, :] / B
        return abars_subsamples

    @staticmethod
    def calculate_stability(abar_subsample):
        stability_measure = 0
        p, _ = abar_subsample.shape

        for ii in range(1, p):
            for jj in range(ii):
                stability_measure += abar_subsample[ii, jj] * (1 - abar_subsample[ii, jj])
        stability_measure = 2 * stability_measure
        stability_measure = stability_measure / (p * (p-1)/2)
        return stability_measure

    def select_optimal_alpha(self, abars_subsamples):
        M, p, _ = abars_subsamples.shape
        for m in range(M-1, -1, -1):
            stab_measure = LPGM.calculate_stability(abars_subsamples[m, :, :])
            if(stab_measure <= self.lpgm_beta):
                return m
        return 0

    def fit(self, data, max_iter=5000, max_line_search_iter=50, prox_grad_lambda_p=1.0, prox_grad_beta=0.5,
            rel_tol=1e-3, abs_tol=1e-6):
        #nr_alphas = model_specific_params['nr_alphas']
        #lpgm_beta = model_specific_params['beta']
        #lpgm_m = model_specific_params['m']
        #lpgm_m = floor(lpgm_m) # Ensure that lpgm_m is an integer.
        #lpgm_B = model_specific_params['B']
        #seeds = model_specific_params['seeds']

        alpha_values = self.generate_alpha_values(data)
        tail_args = [data, alpha_values, max_iter, max_line_search_iter, prox_grad_lambda_p, prox_grad_beta,
                     rel_tol, abs_tol]
        nr_nodes = data.shape[1]

        with Pool(processes=4) as pool:
            all_thetas =  pool.starmap(self.fit_node, Model.provide_args(nr_nodes, tail_args))

        thetas_main, thetas_subsamples = LPGM.make_matrices(all_thetas, p=data.shape[1], M=self.nr_alphas, B=self.lpgm_B)

        ahats_main, ahats_subsamples = LPGM.make_ahats(thetas_main, thetas_subsamples)

        abars_subsamples = LPGM.make_abars_subsamples(ahats_subsamples)

        alpha_opt_index = self.select_optimal_alpha(abars_subsamples)

        return alpha_values[alpha_opt_index], ahats_main[alpha_opt_index, :, :]
