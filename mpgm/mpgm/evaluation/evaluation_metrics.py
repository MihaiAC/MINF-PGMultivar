from mpgm.mpgm.evaluation.generating_samples import SampleParamsWrapper
from mpgm.mpgm.evaluation.fitting_models import FitParamsWrapper

from scipy.stats import norm
from math import ceil

from typing import Optional, Tuple
from enum import Enum

from mpgm.mpgm.sample_generation.samplers import *

class EvalMetrics():

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

    class SymmModes(Enum):
        WW_MIN = "ww_min"
        WW_MAX = "ww_max"
        NONE = "none"

    @staticmethod
    def copy_and_symmetrize_matrix(matrix:np.ndarray, symm_mode:SymmModes) -> np.ndarray:
        symm_matrix = np.copy(matrix)
        nr_variables = symm_matrix.shape[0]

        if symm_mode == EvalMetrics.SymmModes.NONE:
            return symm_matrix

        for ii in range(1, nr_variables):
            for jj in range(ii):
                val_1 = matrix[ii][jj]
                val_2 = matrix[jj][ii]
                is_smaller = np.abs(val_1) < np.abs(val_2)

                if symm_mode == EvalMetrics.SymmModes.WW_MAX:
                    if is_smaller:
                        symm_matrix[ii][jj] = val_2
                        symm_matrix[jj][ii] = val_2
                    else:
                        symm_matrix[ii][jj] = val_1
                        symm_matrix[jj][ii] = val_1
                elif symm_mode == EvalMetrics.SymmModes.WW_MIN:
                    if is_smaller:
                        symm_matrix[ii][jj] = val_1
                        symm_matrix[jj][ii] = val_1
                    else:
                        symm_matrix[ii][jj] = val_2
                        symm_matrix[jj][ii] = val_2

        return symm_matrix


    @staticmethod
    def calculate_tpr_fpr_acc_nonzero(theta_orig:np.ndarray,
                                      theta_fit:np.ndarray,
                                      symm_mode:SymmModes,
                                      threshold:Optional[float]=1e-6) -> Tuple[float, float, float]:
        nr_variables = theta_orig.shape[0]
        symm_theta_fit = EvalMetrics.copy_and_symmetrize_matrix(theta_fit, symm_mode)

        TN, TP, FP, FN = 0, 0, 0, 0
        for ii in range(nr_variables):
            for kk in range(nr_variables):
                if ii == kk:
                    continue

                real_edge = theta_orig[ii][kk] != 0
                inferred_edge = np.abs(symm_theta_fit[ii][kk]) > threshold

                if real_edge and inferred_edge:
                    TP += 1
                elif real_edge and not inferred_edge:
                    FN += 1
                elif not real_edge and not inferred_edge:
                    TN += 1
                else:
                    FP += 1
        if TP+FN == 0:
            TPR = 0
        else:
            TPR = TP / (TP + FN)

        if FP+TN == 0:
            FPR = 0
        else:
            FPR = FP / (FP + TN)

        ACC = (TP + TN) / (TP + FP + TN + FN)
        return TPR, FPR, ACC

    @staticmethod
    def calculate_edge_sign_recall(theta_orig:np.ndarray, theta_fit:np.ndarray, symm_mode:SymmModes,
                                   threshold:Optional[float]=1e-6) -> float:
        nr_variables = theta_orig.shape[0]
        symm_theta_fit = EvalMetrics.copy_and_symmetrize_matrix(theta_fit, symm_mode)
        signed_TP = 0
        true_edges = 0

        for ii in range(nr_variables):
            for kk in range(nr_variables):
                if ii == kk:
                    continue

                real_edge_sign = np.sign(theta_orig[ii][kk])
                if real_edge_sign != 0:
                    true_edges += 1

                if np.abs(symm_theta_fit[ii][kk]) <= threshold:
                    inferred_edge_sign = 0
                else:
                    inferred_edge_sign = np.sign(symm_theta_fit[ii][kk])

                if inferred_edge_sign != 0 and inferred_edge_sign == real_edge_sign:
                    signed_TP += 1

        if true_edges == 0:
            return 0
        else:
            return signed_TP/true_edges

    @staticmethod
    def calculate_MSEs(theta_orig:np.ndarray, theta_fit:np.ndarray, symm_mode:SymmModes) -> Tuple[float, float]:
        nr_variables = theta_orig.shape[0]
        symm_theta_fit = EvalMetrics.copy_and_symmetrize_matrix(theta_fit, symm_mode)

        MSE, diag_MSE = 0, 0
        N, diag_N = 0, 0

        for ii in range(nr_variables):
            for kk in range(nr_variables):
                real_value = theta_orig[ii][kk]
                inferred_value = symm_theta_fit[ii][kk]

                if ii == kk:
                    diag_N += 1
                    diag_MSE += (real_value - inferred_value) ** 2
                else:
                    N += 1
                    MSE += (real_value - inferred_value) ** 2

        return MSE/N, diag_MSE/diag_N

    @staticmethod
    def calculate_percentage_symmetric_signs(input_matrix:np.ndarray) -> float:
        N, M = input_matrix.shape
        if M == N+1:
            matrix = np.copy(input_matrix)
            matrix = matrix[0:N, 0:N]
        else:
            assert N == M, str("calculate_percentage_symmetric_signs: input matrix should be square; has dimensions " +
                               str((N,M)) + " instead")
            matrix = np.copy(input_matrix)

        nr_values = N * (N-1) / 2
        nr_symmetric_signs = 0

        for ii in range(1, len(matrix)):
            for jj in range(ii):
                if np.sign(matrix[ii][jj]) == np.sign(matrix[jj][ii]):
                    nr_symmetric_signs += 1
        return nr_symmetric_signs/nr_values

    @staticmethod
    def calculate_percentage_symmetric_values(input_matrix:np.ndarray, threshold:Optional[float]=1e-6) -> float:
        N, M = input_matrix.shape
        if M == N + 1:
            matrix = np.copy(input_matrix)
            matrix = matrix[0:N, 0:N]
        else:
            assert N == M, str("calculate_percentage_symmetric_signs: input matrix should be square; has dimensions " +
                               str((N, M)) + " instead")
            matrix = np.copy(input_matrix)

        nr_values = N * (N-1) / 2
        nr_symmetric_values = 0

        for ii in range(1, len(matrix)):
            for jj in range(ii):
                if abs(matrix[ii][jj] - matrix[jj][ii]) <= threshold:
                    nr_symmetric_values += 1
        return nr_symmetric_values/nr_values

    @staticmethod
    def calculate_percentage_symmetric_nonzero(input_matrix:np.ndarray, threshold:Optional[float]=0) -> float:
        N, M = input_matrix.shape
        if M == N + 1:
            matrix = np.copy(input_matrix)
            matrix = matrix[0:N, 0:N]
        else:
            assert N == M, str("calculate_percentage_symmetric_signs: input matrix should be square; has dimensions " +
                               str((N, M)) + " instead")
            matrix = np.copy(input_matrix)

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
    def calculate_percentage_sparsity(input_matrix:np.ndarray, threshold:Optional[float]=0) -> float:
        N, M = input_matrix.shape
        if M == N + 1:
            matrix = np.copy(input_matrix)
            matrix = matrix[0:N, 0:N]
        else:
            assert N == M, str("calculate_percentage_symmetric_signs: input matrix should be square; has dimensions " +
                               str((N, M)) + " instead")
            matrix = np.copy(input_matrix)

        nr_params = N * (N - 1)
        nr_zero_params = 0
        for ii in range(1, len(matrix)):
            for jj in range(ii):
                if abs(matrix[ii][jj]) <= threshold:
                    nr_zero_params += 1
                if abs(matrix[jj][ii]) <= threshold:
                    nr_zero_params += 1
        percentage_sparsity = nr_zero_params / nr_params
        return round(percentage_sparsity, 4)

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
    def get_min_nr_samples(graph:np.ndarray, ct:float=1) -> int:
        d = EvalMetrics.get_average_degree(graph)
        p = graph.shape[0]
        return ceil(ct * np.log(p) * d ** 2)

    @staticmethod
    def get_percentage_negative_edges(graph:np.ndarray) -> float:
        nr_edges = 0
        nr_negative_edges = 0
        nr_variables = graph.shape[0]
        for ii in range(nr_variables):
            for jj in range(nr_variables):
                if ii == jj:
                    continue

                edge_sign = np.sign(graph[ii][jj])
                if edge_sign != 0:
                    nr_edges += 1
                    if edge_sign < 0:
                        nr_negative_edges += 1

        if nr_edges == 0:
            return 0
        else:
            return nr_negative_edges/nr_edges

if __name__ == "__main__":
    sqlite_file_name = "samples.sqlite"
    SPS = SampleParamsWrapper.load_samples("PleaseWork", sqlite_file_name)

    fit_id = "debugProxGrad"
    fit_file_name = "fit_models.sqlite"

    FPS = FitParamsWrapper.load_fit(fit_id, fit_file_name)

    model_P = globals()[SPS.model_name](**SPS.model_params)
    model_Q = globals()[FPS.model_name](theta=FPS.theta_fit, **FPS.model_params)
    sampler = TPGMGibbsSampler(60, 90)
