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
        samples = sampler.generate_samples(model_P, init_sample, nr_samples_per_iter)

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

if __name__ == "__main__":
    sqlite_file_name = "samples.sqlite"
    SPS = SampleParamsWrapper.load_samples("PleaseWork", sqlite_file_name)

    fit_id = "debugProxGrad"
    fit_file_name = "fit_models.sqlite"

    FPS = FitParamsWrapper.load_fit(fit_id, fit_file_name)

    model_P = globals()[SPS.model_name](**SPS.model_params)
    model_Q = globals()[FPS.model_name](R=FPS.model_params['R'], theta=FPS.theta_final)
    sampler = TPGMGibbsSampler(60, 90)

    print(node_cond_prob_KL_divergence(model_P, model_P, sampler))

