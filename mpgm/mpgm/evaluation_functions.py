from mpgm.mpgm.generating_samples import SampleParamsSave, SampleParamsWrapper
from mpgm.mpgm.fitting_models import FitParamsSave, FitParamsWrapper

import numpy as np
from scipy.stats import norm

from mpgm.mpgm.models.Model import Model
from mpgm.mpgm.models.TPGM import TPGM
from mpgm.mpgm.models.QPGM import QPGM
from mpgm.mpgm.models.SPGM import SPGM

from typing import Optional, Tuple, List

from mpgm.mpgm.sample_generation.gibbs_samplers import *

def node_cond_prob_KL_divergence(model_P: Model, model_Q:Model, sampler:GibbsSampler, alpha:Optional[float]=0.05,
                                 mse_tol: Optional[float]=1e-2, nr_samples_per_iter:Optional[int]=100,
                                 iter_limit:Optional[int]=20) -> Tuple[np.ndarray, np.ndarray]:
    conf = norm.ppf(1 - alpha)
    nr_nodes = model_P.theta.shape[1]
    mse_nodes = np.inf * np.ones((nr_nodes, ))
    kl_div_nodes = np.zeros((nr_nodes, ))
    var_sum_nodes = np.zeros((nr_nodes, ))
    iter = 0

    init_sample = np.zeros((nr_nodes, ))
    while any(mse_nodes > mse_tol) and iter < iter_limit:
        if iter == 0:
            samples = sampler.generate_samples(model_P, init_sample, nr_samples_per_iter)
            sampler.burn_in = 0
        else:
            samples = sampler.generate_samples(model_P, init_sample, nr_samples_per_iter)
            init_sample = np.array(samples[nr_samples_per_iter - 1, :])
            samples = sampler.generate_samples(model_P, init_sample, nr_samples_per_iter)

        init_sample = np.array(samples[nr_samples_per_iter - 1, :])

        iter += 1
        for node in range(nr_nodes):
            if mse_nodes[node] < mse_tol:
                continue

            P_node_cond_prob = np.zeros((nr_samples_per_iter,))
            Q_node_cond_prob = np.zeros((nr_samples_per_iter,))
            kl_terms = np.zeros((nr_samples_per_iter, ))

            for ii in range(nr_samples_per_iter):
                P_node_cond_prob[ii] = model_P.node_cond_prob(node, samples[ii, node], samples[ii, :])[0]
                Q_node_cond_prob[ii] = model_Q.node_cond_prob(node, samples[ii, node], samples[ii, :])[0]
                kl_terms[ii] = P_node_cond_prob[ii] * np.log2(P_node_cond_prob[ii]) / np.log2(Q_node_cond_prob[ii])

            kl_div_nodes[node] += (np.mean(kl_terms) - kl_div_nodes[node]) / iter
            var_sum_nodes[node] += np.sum((kl_terms - kl_div_nodes[node]) ** 2)
            mse_nodes[node] = conf * np.sqrt(var_sum_nodes[node] / (iter * nr_samples_per_iter * (iter * nr_samples_per_iter - 1)))

    return kl_div_nodes, mse_nodes


def calculate_tpr_fpr_acc(theta_orig:np.ndarray, theta_fit:np.ndarray, threshold:Optional[float]=0.0):
    nr_variables = theta_orig.shape[0]

    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for ii in range(1, nr_variables):
        for kk in range(ii):
            real_edge = theta_orig[ii][kk] != 0
            inferred_edge = abs(theta_fit[ii][kk]) >= threshold or abs(theta_fit[kk][ii]) >= threshold

            if real_edge and inferred_edge:
                TP += 1
            elif real_edge and not inferred_edge:
                FN += 1
            elif not real_edge and not inferred_edge:
                TN += 1
            else:
                FP += 1
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    ACC = (TP + TN) / (TP + FP + TN + FN)
    return TPR, FPR, ACC

def is_symmetric_values(matrix:np.ndarray) -> bool:
    for ii in range(1, len(matrix)):
        for jj in range(ii):
            if matrix[ii][jj] != matrix[jj][ii]:
                return False
    return True

def is_symmetric_neighbourhood(matrix:np.ndarray) -> bool:
    for ii in range(1, len(matrix)):
        for jj in range(ii):
            if (matrix[ii][jj] == 0 and matrix[jj][ii] != 0) or (matrix[ii][jj] != 0 and matrix[jj][ii] == 0):
                return False
    return True


def calculate_param_sparsity(matrix:np.ndarray) -> float:
    total_params = 0
    zero_params = 0
    for ii in range(1, len(matrix)):
        for jj in range(ii):
            total_params += 2
            if matrix[ii][jj] == 0:
                zero_params += 1
            if matrix[jj][ii] == 0:
                zero_params += 1

    return 100 * zero_params / total_params

if __name__ == "__main__":
    sqlite_file_name = "samples.sqlite"
    SPS = SampleParamsWrapper.load_samples("PleaseWork", sqlite_file_name)

    fit_id = "debugProxGrad"
    fit_file_name = "fit_models.sqlite"

    FPS = FitParamsWrapper.load_fit(fit_id, fit_file_name)

    model_P = globals()[SPS.model_name](**SPS.model_params)
    model_Q = globals()[FPS.model_name](R=FPS.model_params['R'], theta=FPS.theta_final)
    sampler = TPGMGibbsSampler(60, 90)

    print(node_cond_prob_KL_divergence(model_P, model_Q, sampler))

