from mpgm.mpgm.evaluation.generating_samples import SampleParamsWrapper
from mpgm.mpgm.evaluation.fitting_models import FitParamsWrapper

from scipy.stats import norm
from math import ceil

from typing import Optional, Tuple

from mpgm.mpgm.sample_generation.gibbs_samplers import *

# TODO: need to symmetrise theta_fit.
class EvalFns():

    @staticmethod
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

    # TODO: Need to account for the rule used (AND, OR).
    @staticmethod
    def calculate_tpr_fpr_acc(theta_orig:np.ndarray, theta_fit:np.ndarray, threshold:Optional[float]=1e-3):
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

    @staticmethod
    def calculate_percentage_symmetric_values(matrix:np.ndarray, threshold:Optional[float]=1e-3) -> float:
        N, M = matrix.shape
        assert N == M, str("calculate_percentage_symmetric_values: input matrix should be square; has dimensions " +
                           str((N,M)) + " instead")

        nr_values = N * (N-1) / 2
        nr_symmetric_values = 0

        for ii in range(1, len(matrix)):
            for jj in range(ii):
                if abs(matrix[ii][jj] - matrix[jj][ii]) <= threshold:
                    nr_symmetric_values += 1
        return nr_symmetric_values/nr_values

    @staticmethod
    def calculate_percentage_symmetric_neighbourhood(matrix:np.ndarray, threshold:Optional[float]=1e-6) -> float:
        N, M = matrix.shape
        assert N == M, str("calculate_percentage_symmetric_values: input matrix should be square; has dimensions " +
                           str((N, M)) + " instead")

        nr_values = N * (N-1) / 2
        nr_symmetric_binary_values = 0

        for ii in range(1, len(matrix)):
            for jj in range(ii):
                x_is_zero = abs(matrix[ii][jj]) <= threshold
                x_T_is_zero = abs(matrix[jj][ii]) <= threshold

                if x_is_zero == x_T_is_zero:
                    nr_symmetric_binary_values += 1
        return nr_symmetric_binary_values/nr_values

    @staticmethod
    def calculate_percentage_sparsity(matrix:np.ndarray, threshold:Optional[float]=1e-6) -> float:
        N, M = matrix.shape
        assert N == M, str("calculate_percentage_symmetric_values: input matrix should be square; has dimensions " +
                           str((N, M)) + " instead")

        nr_params = N * (N - 1)
        nr_zero_params = 0
        for ii in range(1, len(matrix)):
            for jj in range(ii):
                if abs(matrix[ii][jj]) <= threshold:
                    nr_zero_params += 1
                if abs(matrix[jj][ii]) <= threshold:
                    nr_zero_params += 1
        return nr_zero_params / nr_params

    # Helper functions.
    @staticmethod
    def get_average_degree(graph:np.ndarray) -> float:
        nr_variables = graph.shape[0]
        avg_degree = 0
        for ii in range(nr_variables):
            for jj in range(nr_variables):
                if graph[ii][jj] != 0:
                    avg_degree += 1
        return avg_degree / nr_variables

    @staticmethod
    def get_min_nr_edges(graph:np.ndarray, ct:float=1) -> int:
        d = EvalFns.get_average_degree(graph)
        p = graph.shape[0]
        return ceil(ct * np.log(p) * d ** 2)

if __name__ == "__main__":
    sqlite_file_name = "samples.sqlite"
    SPS = SampleParamsWrapper.load_samples("PleaseWork", sqlite_file_name)

    fit_id = "debugProxGrad"
    fit_file_name = "fit_models.sqlite"

    FPS = FitParamsWrapper.load_fit(fit_id, fit_file_name)

    model_P = globals()[SPS.model_name](**SPS.model_params)
    model_Q = globals()[FPS.model_name](R=FPS.model_params['R'], theta=FPS.theta_final)
    sampler = TPGMGibbsSampler(60, 90)

    # print(node_cond_prob_KL_divergence(model_P, model_Q, sampler))

