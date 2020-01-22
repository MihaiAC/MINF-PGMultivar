import numpy as np
from math import floor
from functools import partial
from multiprocessing import Pool
from scipy.special import gammaln
from mpgm.mpgm.solvers import prox_grad

def calculate_nll_and_grad_nll(node, data, theta):
    '''
    :param Y: Predictor variable, 1 x N.
    :param W: X.
    :param beta: The parameters we are learning. 1 x p. beta[ii] is ignored.
    :return:
    '''
    Y = data[:, node]
    W = data
    beta = theta
    theta[node] = 0

    nll = 0
    N = len(Y)

    dots_Wi_beta = np.dot(data, theta)

    nll = -np.dot(Y, dots_Wi_beta) + np.sum(np.exp(dots_Wi_beta))
    grad_nll = -np.sum(W * (Y - np.exp(dots_Wi_beta)), axis=0)

    nll = (nll/N) - 1 # exp(W[ii, node] * theta[node]) = exp(0) = 1; removing this.
    grad_nll = grad_nll/N
    grad_nll[node] = 0

    return nll, grad_nll

def calculate_nll(node, data, theta):
    Y = data[:, node]
    W = data
    beta = theta
    theta[node] = 0

    nll = 0
    N = len(Y)

    dots_Wi_beta = np.dot(data, theta)

    nll -= np.dot(Y, dots_Wi_beta) + np.sum(np.exp(dots_Wi_beta))
    nll = nll/N - 1 #

    return nll

def generate_alpha_values(data, nr_vals=100):
    alpha_min = 0.0001

    alpha_max = 0
    n_rows, n_cols = data.shape
    for ii in range(1, n_cols):
        for jj in range(0, ii):
            value = np.sum(data[:, ii] * data[:, jj])
            if(value > alpha_max):
                alpha_max = value

    alpha_vals = np.logspace(np.log10(alpha_max), np.log10(alpha_min), nr_vals)
    return alpha_vals

def provide_args(nr_nodes, other_args):
    for ii in range(nr_nodes):
        yield [ii] + other_args
    return

def fit_node(node, alpha_values, lpgm_beta, lpgm_m, lpgm_B, seeds, theta_init, alpha, data, f, f_and_grad_f, f_gf_params,
             condition, max_iter, max_line_search_iter, lambda_p, beta, rel_tol, abs_tol):
    f = calculate_nll
    f_and_grad_f = calculate_nll_and_grad_nll
    f_gf_params = []

    prox_grad_unmutable_params = [f, f_and_grad_f, f_gf_params, None, max_iter, max_line_search_iter,
                                  lambda_p, beta, rel_tol, abs_tol]

    thetas_main = np.zeros((len(alpha_values), data.shape[1]))
    thetas_subsamples = np.zeros(lpgm_B, len(alpha_values), data.shape[1])

    local_rng = np.random.RandomState(seeds[node])

    theta_init = local_rng.normal(0, 0.0001, data.shape[1])
    for ii, alpha in enumerate(alpha_values):
        theta_warm = prox_grad(node, theta_init, alpha, data, *prox_grad_unmutable_params)[0]
        thetas_main[ii, :] = theta_warm
        theta_init = theta_warm

    for bb in range(lpgm_B):
        theta_init = local_rng.normal(0, 0.0001, data.shape[1])
        subsample_indices = local_rng.choice(data.shape[0], lpgm_m, replace=False)
        subsampled_data = data[subsample_indices, :]
        for ii, alpha in enumerate(alpha_values):
            theta_warm = prox_grad(node, theta_init, alpha, subsampled_data, *prox_grad_unmutable_params)[0]
            thetas_subsamples[bb, ii, :] = theta_warm
            theta_init = theta_warm

    return thetas_main, thetas_subsamples

def fit(model_specific_params, prox_grad_args):
    """
    :param data: N X P matrix; each row is a datapoint;
    :param theta_init: starting parameter values;
    :param alpha: regularization parameter;
    :param max_iter:
    :param max_line_search_iter:
    :param lambda_p: step size for the proximal gradient descent method;
    :param beta:
    :param rel_tol:
    :param abs_tol:
    :return:
    """
    data = prox_grad_args['data']
    alpha_values = generate_alpha_values(data)

    lpgm_beta = model_specific_params['beta']
    lpgm_m = model_specific_params['m']
    lpgm_m = floor(lpgm_m) # Ensure that lpgm_m is an integer.
    lpgm_B = model_specific_params['B']
    seeds = model_specific_params['seeds']

    tail_args = [alpha_values, lpgm_beta, lpgm_m, lpgm_B, seeds] + prox_grad_args
    nr_nodes = data.shape[1]

    with Pool(processes=4) as pool:
        all_thetas =  pool.starmap(fit_node, provide_args(nr_nodes, tail_args))