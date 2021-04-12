import time
import numpy as np
import multiprocessing
import math
from mpgm.mpgm.model_fitters.prox_operators import SoftThreshold, QuadProgOperator, AdmmOperator, QPGM_SoftThreshold
from typing import Callable, Optional, Iterator, Any, Tuple, List

class Prox_Grad_Fitter():
    def __init__(self,
                 alpha,
                 accelerated=True,
                 max_iter=5000,
                 max_line_search_iter=50,
                 line_search_rel_tol=1e-4,
                 init_step_size=1.0,
                 beta=0.5,
                 rel_tol=1e-3,
                 abs_tol=1e-6,
                 early_stop_criterion='weight',
                 minimum_iterations_until_early_stop=2,
                 save_regularization_paths=False,
                 keep_diag_zero=False):
        """
        Proximal gradient descent for solving the l1-regularized node-wise regressions required to fit some models in this
            package.
        :param alpha: L1 regularization parameter.
        :param accelerated: Whether to use the accelerated prox grad descent or not.
        :param max_iter: Maximum iterations allowed for the algorithm.
        :param max_line_search_iter: Maximum iterations allowed for the line search performed at every iteration step.
        :param lambda_p: Initial step-size.
        :param beta: Line search parameter.
        :param rel_tol: Relative tolerance value for early stopping.
        :param abs_tol: Absolute tolerance value for early stopping. Should be 1e-3 * rel_tol.
        :param early_stop_criterion: If equal to 'weight', training stops when the weight values have converged.
            If equal to 'likelihood', training stops when the value of the NLL has converged.
        :return: (parameters, likelihood_values, converged) - tuple containing parameters and the NLL value for each
            iteration;
        """
        self.alpha = alpha
        self.accelerated = accelerated
        self.max_iter = max_iter
        self.max_line_search_iter = max_line_search_iter
        self.line_search_rel_tol = line_search_rel_tol
        self.init_step_size = init_step_size
        self.beta = beta
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.early_stop_criterion = early_stop_criterion
        self.minimum_iterations_until_early_stop = minimum_iterations_until_early_stop
        self.save_regularization_paths = save_regularization_paths
        self.keep_diag_zero = keep_diag_zero

        self.prox_operator = SoftThreshold()
        assert minimum_iterations_until_early_stop > 0, "At least one iteration must pass until convergence is checked."

    def call_fit_node(self, nll:Callable[[int, np.ndarray, np.ndarray], float],
                      grad_nll:Callable[[int, np.ndarray, np.ndarray], np.ndarray],
                      data_points:np.ndarray,
                      theta_init:np.ndarray,
                      parallelize:Optional[bool]=True) -> \
            Tuple[np.ndarray, List[np.ndarray], List[bool], List[Any], List[np.ndarray], float]:

        nr_nodes = data_points.shape[1]
        theta_fit = np.zeros(theta_init.shape)
        likelihoods = []
        converged = []
        conditions = []
        regularization_paths = []
        avg_node_fit_time = 0

        if parallelize == False:
            for node in range(nr_nodes):
                theta_fit_node, likelihoods_node, converged_node, conditions_node, theta_evolution_node, \
                elapsed_time_node = self.fit_node(node, nll, grad_nll, data_points, theta_init)
                theta_fit[node, :] = theta_fit_node
                likelihoods.append(likelihoods_node)
                converged.append(converged_node)
                conditions.append(conditions_node)
                regularization_paths.append(theta_evolution_node)
                avg_node_fit_time += elapsed_time_node / nr_nodes

        else:
            nr_cpus = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=nr_cpus-1) as pool:
                results = pool.starmap(self.fit_node, self.fit_node_parameter_generator(nr_nodes, nll, grad_nll, data_points, theta_init))

            for node, node_result in enumerate(results):
                theta_fit[node, :] = node_result[0]
                likelihoods.append(node_result[1])
                converged.append(node_result[2])
                conditions.append(node_result[3])
                regularization_paths.append(node_result[4])
                avg_node_fit_time += node_result[5] / nr_nodes

        return theta_fit, likelihoods, converged, conditions, regularization_paths, avg_node_fit_time

    def update_fit_params(self, iteration:int, prev_params:np.ndarray, penultimate_params:np.ndarray, node:int,
                          reg_param:float, data_points:np.ndarray):
        if self.accelerated:
            w_k = iteration / (iteration + 3)
            params = prev_params + w_k * (prev_params - penultimate_params)
        else:
            params = prev_params
        return params

    def check_line_search_condition_simple(self, f_z, f_tilde):
        return f_z <= f_tilde

    def check_line_search_condition_closeto(self, f_z, f_tilde):
        return f_z <= f_tilde or np.isclose(f_z, f_tilde, rtol=self.line_search_rel_tol)

    def fit_node_parameter_generator(self, nr_nodes:int, nll:Callable[[int, np.ndarray, np.ndarray], float],
                      grad_nll:Callable[[int, np.ndarray, np.ndarray], np.ndarray],
                      data_points:np.ndarray, theta_init:np.ndarray) -> Iterator[Any]:
        for node in range(nr_nodes):
            yield (node, nll, grad_nll, data_points, theta_init)

    def check_params_convergence(self, iteration_nr, theta_k_1, theta_k_2):
        if iteration_nr < self.minimum_iterations_until_early_stop:
            return False
        converged_params = (theta_k_1 - theta_k_2) ** 2 <= (self.rel_tol ** 2) * (theta_k_1 ** 2)
        converged = all(converged_params)
        return converged

    def check_likelihood_convergence(self, iteration_nr, neg_log_likelihoods):
        if iteration_nr < self.minimum_iterations_until_early_stop:
            return False

        current_nll = neg_log_likelihoods[iteration_nr]
        prev_nll = neg_log_likelihoods[iteration_nr-1]
        converged = current_nll > prev_nll or np.isclose(current_nll, prev_nll, self.rel_tol, self.abs_tol)
        return converged

    def fit_node(self, node, f_nll, f_grad_nll, data_points, theta_init):
        """

        :param node: Index of the node to fit.
        :param f_nll: calculates negative ll of node value given the other data_points.
        :param f_grad_nll: calculates gradient of negative ll.
        :param data_points: N x m matrix, N = number of points and m = number of nodes in the graph.
        :param theta_init: m x m matrix; theta_init[node, :] must contain the initial guesses for the parameters fit here.
        :return: (parameters which resulted from the method, list of lists of line search likelihoods,
        bool which indicated if the method converged, not sure what this was)
        """
        start_time = time.time()

        neg_log_likelihoods = []
        conditions = []
        converged = False

        theta_node_init = theta_init[node, :]
        theta_k_2 = np.zeros(theta_node_init.shape)
        theta_k_1 = np.copy(theta_node_init)
        step_size_k = self.init_step_size

        regularization_paths = []

        z = np.zeros(np.size(theta_node_init))
        f_z = 0

        for k in range(self.max_iter):
            theta_k = self.update_fit_params(k, theta_k_1, theta_k_2, node, step_size_k * self.alpha, data_points)

            if self.save_regularization_paths and self.accelerated:
                regularization_paths.append(list(theta_k))

            f_theta_k = f_nll(node, data_points, theta_k)
            grad_f_theta_k = f_grad_nll(node, data_points, theta_k)

            if self.keep_diag_zero:
                grad_f_theta_k[node] = 0

            found_step_size = False

            for _ in range(self.max_line_search_iter):
                candidate_new_theta_k = theta_k - step_size_k * grad_f_theta_k
                threshold = step_size_k * self.alpha
                z = self.prox_operator.prox(objective=candidate_new_theta_k,
                                            reg_parameter=threshold,
                                            node=node,
                                            data_points=data_points,
                                            keep_diag_zero=self.keep_diag_zero)

                # TODO: Could use this as a gradient alert of some sort?
                # if self.save_regularization_paths:
                #     regularization_paths.append(list(z))
                #     if np.sum(z)/len(z) >= 0.5:
                #         print("Node: " + str(node) + "; Iteration: " + str(k) + "; Gradients: " + str(grad_f_theta_k))

                f_tilde = f_theta_k + np.dot(grad_f_theta_k, z-theta_k) + (1/(2 * step_size_k)) * np.sum((z-theta_k) ** 2)
                f_z = f_nll(node, data_points, z)

                if self.check_line_search_condition_closeto(f_z, f_tilde):
                    found_step_size = True
                    break
                else:
                    step_size_k = step_size_k * self.beta

            if found_step_size:
                theta_k_2 = theta_k_1
                theta_k_1 = z
                neg_log_likelihoods.append(f_z)

                if self.save_regularization_paths:
                    regularization_paths.append(list(z))

                converged = False
                if self.early_stop_criterion == 'weight':
                    converged = self.check_params_convergence(k, theta_k_1, theta_k_2)
                elif self.early_stop_criterion == 'likelihood':
                    converged = self.check_likelihood_convergence(k, neg_log_likelihoods)

                if converged:
                    # print('\nParameters for node ' + str(node) + ' converged in ' + str(k) + ' iterations.')
                    # In this case, last parameters and likelihood were not the best.
                    if self.early_stop_criterion == "likelihood":
                        theta_k_1 = theta_k_2
                        neg_log_likelihoods = neg_log_likelihoods[:-1]

                    break

            else:
                converged = False
                print('\nLine search failed for node: ' + str(node))
                break

        self.data_points = None
        regularization_paths = np.array(regularization_paths)

        elapsed_time = time.time() - start_time
        return theta_k_1, np.array(neg_log_likelihoods), converged, np.array(conditions), regularization_paths, elapsed_time

class Constrained_Prox_Grad_Fitter(Prox_Grad_Fitter):
    def __init__(self,
                 alpha,
                 accelerated=True,
                 constraint_solver='admm',
                 max_iter=5000,
                 max_line_search_iter=50,
                 line_search_rel_tol=1e-4,
                 init_step_size=1.0,
                 beta=0.5,
                 rel_tol=1e-3,
                 abs_tol=1e-6,
                 early_stop_criterion='weight',
                 minimum_iterations_until_early_stop=2,
                 save_regularization_paths=False,
                 keep_diag_zero=False,
                 admm_tau: Optional[float] = 0.1,
                 admm_min_iter: Optional[int] = 100,
                 admm_max_iter: Optional[int] = 1000,
                 qpgm_qtc: Optional[float] = 1e4):
        super().__init__(alpha, accelerated, max_iter, max_line_search_iter, line_search_rel_tol, init_step_size,
                         beta, rel_tol, abs_tol, early_stop_criterion, minimum_iterations_until_early_stop,
                         save_regularization_paths, keep_diag_zero)
        if constraint_solver in ['qpoases', 'osqp']:
            self.prox_operator = QuadProgOperator(constraint_solver)
        elif constraint_solver == 'admm':
            self.prox_operator = AdmmOperator(tau=admm_tau,
                                              min_iter=admm_min_iter,
                                              max_iter=admm_max_iter)
        elif constraint_solver == 'qpgm_soft':
            self.prox_operator = QPGM_SoftThreshold(qpgm_qtc)
        else:
            self.prox_operator = SoftThreshold()

    # A matrix' row rank is equal to its column rank.
    # This is why in this case, the max rank will be 10 (for a 150x10 matrix).
    # E.g. the max number of linearly independent rows will be 10.

    # @staticmethod
    # def is_pos_semidef(x):
    #     return np.all(np.linalg.eigvals(x) >= 0)

    def update_fit_params(self, iteration:int, prev_params:np.ndarray, penultimate_params:np.ndarray, node:int,
                          reg_parameter:float, data_points:np.ndarray):
        if self.accelerated:
            w_k = iteration / (iteration + 3)
            params = prev_params + w_k * (prev_params - penultimate_params)
            # Need to project this to a point; sparsity is not as important here.
            if iteration != 0 and self.prox_operator.constraint_solver != 'soft':
                params = self.prox_operator.prox(params, reg_parameter, node, data_points, self.keep_diag_zero)
        else:
            params = prev_params
        return params

class Pseudo_Likelihood_Prox_Grad_Fitter(Constrained_Prox_Grad_Fitter):
    def __init__(self,
                 alpha,
                 accelerated=True,
                 constraint_solver='qpoases',
                 max_iter=5000,
                 max_line_search_iter=50,
                 line_search_rel_tol=1e-4,
                 init_step_size=1.0,
                 beta=0.5,
                 rel_tol=1e-3,
                 abs_tol=1e-6,
                 early_stop_criterion='weight',
                 minimum_iterations_until_early_stop=1,
                 save_regularization_paths=False,
                 keep_diag_zero=False,
                 admm_tau: Optional[float] = 0.1,
                 admm_min_iter: Optional[int] = 100,
                 admm_max_iter: Optional[int] = 1000
                 ):
        super().__init__(alpha, accelerated, constraint_solver, max_iter, max_line_search_iter, line_search_rel_tol,
                         init_step_size, beta, rel_tol, abs_tol, early_stop_criterion,
                         minimum_iterations_until_early_stop, save_regularization_paths, keep_diag_zero, admm_tau,
                         admm_min_iter, admm_max_iter)

    def call_fit_node(self, nll:Callable[[int, np.ndarray, np.ndarray], float],
                      grad_nll:Callable[[int, np.ndarray, np.ndarray], np.ndarray],
                      data_points:np.ndarray,
                      theta_init:np.ndarray,
                      parallelize:Optional[bool]=True) -> \
            Tuple[np.ndarray, List[np.ndarray], List[bool], List[Any], List[np.ndarray], float]:

        # Returned parameters differ from the parameters returned by the normal Prox_Grad.
        # Since the first param is the same and the types match, it shouldn't pose a problem to the StatsGenerator
        # class.
        theta_fit, likelihoods, converged, conditions, regularization_path, elapsed_time = self.fit_all_nodes(nll, grad_nll, data_points, theta_init)
        return theta_fit, [likelihoods], [converged], [conditions], [regularization_path], elapsed_time

    def update_fit_params(self, iteration:int, prev_params:np.ndarray, penultimate_params:np.ndarray, node:int,
                          reg_parameter:float, data_points:np.ndarray):
        if self.accelerated:
            w_k = iteration / (iteration + 3)
            params = prev_params + w_k * (prev_params - penultimate_params)
            # Need to project this to a point; sparsity is not as important here.
            if iteration != 0 and self.prox_operator.constraint_solver != 'soft':
                return_params = np.zeros(params.shape)
                nr_nodes = params.shape[0]
                for node in range(nr_nodes):
                    return_params[node, :] = self.prox_operator.prox(params[node, :], reg_parameter, node, data_points, self.keep_diag_zero)
                if self.prox_operator.constraint_solver == 'qpgm_soft':
                    return_params[0:nr_nodes, 0:nr_nodes] = (return_params[0:nr_nodes, 0:nr_nodes] +
                                                             return_params[0:nr_nodes, 0:nr_nodes].T)/2
                else:
                    return_params = (return_params + return_params.T)/2
                return return_params
        else:
            params = prev_params
        return params

    def check_params_convergence(self, iteration_nr, theta_k_1, theta_k_2):
        if iteration_nr < self.minimum_iterations_until_early_stop:
            return False
        converged_params = (theta_k_1 - theta_k_2) ** 2 <= (self.rel_tol ** 2) * (theta_k_1 ** 2)
        converged = converged_params.all()
        return converged

    # Method not used in this class, since we are fitting all the nodes at the same time.
    def fit_node(self, node, f_nll, f_grad_nll, data_points, theta_init):
        pass

    def fit_all_nodes(self, f_nll, f_grad_nll, data_points, theta_init):
        """
        :param f_nll: calculates negative ll of node value given the other data_points.
        :param f_grad_nll: calculates gradient of negative ll.
        :param data_points: N x m matrix, N = number of points and m = number of nodes in the graph.
        :param theta_init: m x m matrix; theta_init[node, :] must contain the initial guesses for the parameters fit here.
        :return: (parameters which resulted from the method, list of lists of line search likelihoods,
        bool which indicated if the method converged, not sure what this was)
        """
        start_time = time.time()

        neg_log_likelihoods = []
        conditions = []
        converged = False

        nr_nodes = data_points.shape[1]

        theta_k_2 = np.zeros(theta_init.shape)
        theta_k_1 = np.copy(theta_init)
        step_size_k = self.init_step_size

        regularization_paths = []

        z = np.zeros(theta_init.shape)
        f_z = 0

        for k in range(self.max_iter):
            theta_k = self.update_fit_params(k, theta_k_1, theta_k_2, -1, step_size_k * self.alpha, data_points)

            if self.save_regularization_paths and self.accelerated:
                regularization_paths.append(theta_k)

            f_theta_k = 0
            grad_f_theta_k = np.zeros(theta_init.shape)
            for node in range(nr_nodes):
                f_theta_k += f_nll(node, data_points, theta_k[node, :])
                grad_f_theta_k[node, :] = f_grad_nll(node, data_points, theta_k[node, :])

                if self.keep_diag_zero:
                    grad_f_theta_k[node, node] = 0

            found_step_size = False

            # Make grad_f_theta_k symmetric.
            grad_f_theta_k[0:nr_nodes, 0:nr_nodes] = grad_f_theta_k[0:nr_nodes, 0:nr_nodes] + grad_f_theta_k[0:nr_nodes, 0:nr_nodes].T

            # Only prox operator it works with at the moment is Soft Thresholding.
            for _ in range(self.max_line_search_iter):
                candidate_new_theta_k = theta_k - step_size_k * grad_f_theta_k
                threshold = step_size_k * self.alpha

                z = np.zeros(theta_init.shape)
                for node in range(nr_nodes):
                    z[node, :] = self.prox_operator.prox(objective=candidate_new_theta_k[node, :],
                                                         reg_parameter=threshold,
                                                         node=node,
                                                         data_points=data_points,
                                                         keep_diag_zero=self.keep_diag_zero)
                z[0:nr_nodes, 0:nr_nodes] = (z[0:nr_nodes, 0:nr_nodes] + z[0:nr_nodes, 0:nr_nodes].T)/2

                # Need to correct how these are calculated yo.
                first_term = f_theta_k
                second_term = np.sum((grad_f_theta_k * (z-theta_k))[np.triu_indices(nr_nodes)])
                third_term = (1/(2 * step_size_k)) * np.sum((z-theta_k)[np.triu_indices(nr_nodes)]**2)
                f_tilde = first_term + second_term + third_term

                f_z = 0
                for node in range(nr_nodes):
                    f_z += f_nll(node, data_points, z[node, :])

                if self.check_line_search_condition_closeto(f_z, f_tilde):
                    found_step_size = True
                    break
                else:
                    step_size_k = step_size_k * self.beta

            if found_step_size:
                theta_k_2 = theta_k_1
                theta_k_1 = z
                neg_log_likelihoods.append(f_z)

                if self.save_regularization_paths:
                    regularization_paths.append(z)

                converged = False
                if self.early_stop_criterion == 'weight':
                    converged = self.check_params_convergence(k, theta_k_1, theta_k_2)
                elif self.early_stop_criterion == 'likelihood':
                    converged = self.check_likelihood_convergence(k, neg_log_likelihoods)

                if converged:
                    # print('\nParameters for node ' + str(node) + ' converged in ' + str(k) + ' iterations.')
                    # In this case, last parameters and likelihood were not the best.
                    if self.early_stop_criterion == "likelihood":
                        theta_k_1 = theta_k_2
                        neg_log_likelihoods = neg_log_likelihoods[:-1]

                    break

            else:
                converged = False
                print('\nLine search failed.')
                break

        self.data_points = None
        regularization_paths = np.array(regularization_paths)

        elapsed_time = time.time() - start_time
        return theta_k_1, np.array(neg_log_likelihoods), converged, np.array(conditions), regularization_paths, elapsed_time

class LPGM_Fitter(Constrained_Prox_Grad_Fitter):
    def __init__(self,
                 accelerated=True,
                 constraint_solver='qpoases',
                 max_iter=5000,
                 max_line_search_iter=50,
                 line_search_rel_tol=1e-4,
                 init_step_size=1.0,
                 beta=0.5,
                 rel_tol=1e-3,
                 abs_tol=1e-6,
                 early_stop_criterion='weight',
                 minimum_iterations_until_early_stop=1,
                 save_regularization_paths=False,
                 keep_diag_zero=False,
                 admm_tau: Optional[float] = 0.1,
                 admm_min_iter: Optional[int] = 100,
                 admm_max_iter: Optional[int] = 1000,
                 nr_alphas: Optional[int] = 100,
                 lpgm_B: Optional[int] = 10,
                 lpgm_beta: Optional[float] = 0.1
                 ):
        super().__init__(-1, accelerated, constraint_solver, max_iter, max_line_search_iter, line_search_rel_tol,
                         init_step_size, beta, rel_tol, abs_tol, early_stop_criterion,
                         minimum_iterations_until_early_stop, save_regularization_paths, keep_diag_zero, admm_tau,
                         admm_min_iter, admm_max_iter)
        self.nr_alphas = nr_alphas
        self.lpgm_B = lpgm_B
        self.lpgm_beta = lpgm_beta

        # These parameters will be initialised when the data_points are given.
        self.lpgm_m = None
        self.alpha_values = np.array([])

    def generate_alpha_values(self, data_points:np.ndarray) -> np.ndarray:
        alpha_min = 0.0001

        alpha_max = 0
        n_rows, n_cols = data_points.shape
        for ii in range(1, n_cols):
            for jj in range(0, ii):
                value = np.sum(data_points[:, ii] * data_points[:, jj])
                if(value > alpha_max):
                    alpha_max = value

        alpha_vals = np.logspace(np.log10(alpha_max), np.log10(alpha_min), self.nr_alphas)
        return alpha_vals

    def make_matrices(self, results):
        M = len(self.alpha_values)
        p = len(results)

        thetas_main = np.zeros((M, p, p))
        thetas_subsamples = np.zeros((M, self.lpgm_B, p, p))

        for ii in range(p):
            node_thetas_main = results[ii][0]
            node_thetas_subsamples = results[ii][1]
            for alpha_index in range(M):
                thetas_main[alpha_index, ii, :] = node_thetas_main[ii, :]
                for b in range(self.lpgm_B):
                    thetas_subsamples[alpha_index, b, ii, :] = node_thetas_subsamples[b, alpha_index, :]

        return thetas_main, thetas_subsamples

    # TODO: Can replace max by min for fewer edges?
    @staticmethod
    def make_ahat(theta):
        p = theta.shape[0]
        ahat = np.zeros(theta.shape)
        for ii in range(1, p):
            for jj in range(ii):
                ahat[ii][jj] = min(np.abs(np.sign(theta[ii, jj])), np.abs(np.sign(theta[jj, ii])))
                ahat[jj][ii] = ahat[ii][jj]
        return ahat

    @staticmethod
    def make_ahats(thetas_main, thetas_subsamples):
        ahats_main = np.zeros(thetas_main.shape)
        ahats_subsamples = np.zeros(thetas_subsamples.shape)

        M = thetas_main.shape[0]
        B = thetas_subsamples.shape[1]

        for m in range(M):
            ahats_main[m, :, :] = LPGM_Fitter.make_ahat(thetas_main[m, :, :])
            for b in range(B):
                ahats_subsamples[m, b, :, :] = LPGM_Fitter.make_ahat(thetas_subsamples[m, b, :, :])

        return ahats_main, ahats_subsamples

    @staticmethod
    def make_abars_subsamples(ahats_subsamples):
        M, B, p, _ = ahats_subsamples.shape
        abars_subsamples = np.zeros((M, p, p))

        for m in range(M):
            for b in range(B):
                abars_subsamples[m, :, :] += ahats_subsamples[m, b, :, :]
            abars_subsamples[m, :, :] = (1/B) * abars_subsamples[m, :, :]
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
            stab_measure = LPGM_Fitter.calculate_stability(abars_subsamples[m, :, :])
            if (stab_measure <= self.lpgm_beta):
                return m
        return 0


    def call_fit_node(self,
                      nll:Callable[[int, np.ndarray, np.ndarray], float],
                      grad_nll:Callable[[int, np.ndarray, np.ndarray], np.ndarray],
                      data_points:np.ndarray,
                      theta_init:np.ndarray,
                      parallelize:Optional[bool]=True) -> \
            Tuple[np.ndarray, List[np.ndarray], List[bool], List[Any], List[np.ndarray], float]:

        nr_datapoints, nr_nodes = data_points.shape
        start_time = time.time()

        self.lpgm_m = math.floor(10*math.sqrt(nr_datapoints))
        self.alpha_values = self.generate_alpha_values(data_points)

        nr_cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=nr_cpus-1) as pool:
            results = pool.starmap(self.fit_node_wrapper, self.fit_node_parameter_generator(nr_nodes, nll, grad_nll, data_points, theta_init))

        thetas_main, thetas_subsamples = self.make_matrices(results)
        ahats_main, ahats_subsamples = LPGM_Fitter.make_ahats(thetas_main, thetas_subsamples)
        abars_subsamples = LPGM_Fitter.make_abars_subsamples(ahats_subsamples)
        alpha_opt_index = self.select_optimal_alpha(abars_subsamples)

        print('Optimal alpha value: ' + str(self.alpha_values[alpha_opt_index]))
        theta_fit = ahats_main[alpha_opt_index, :, :]

        total_fit_time = time.time() - start_time
        return ahats_main, [], [], [], [], total_fit_time


    def fit_node_wrapper(self, node, f_nll, f_grad_nll, data_points, theta_init):
        nr_datapoints, nr_nodes = data_points.shape
        # rho x p
        thetas_main = np.zeros((len(self.alpha_values), nr_nodes))
        # B x rho x p
        thetas_subsamples = np.zeros((self.lpgm_B, len(self.alpha_values), nr_nodes))

        theta_warm = np.copy(theta_init)
        for ii, alpha in enumerate(self.alpha_values):
            self.alpha = alpha
            theta_result = self.fit_node(node, f_nll, f_grad_nll, data_points, theta_warm)[0]
            # theta_warm = np.copy(theta_init)
            if max(theta_result[0:nr_nodes]) > 2 or min(theta_result[0:nr_nodes] < -2):
                theta_warm[node, 0:nr_nodes] = np.copy(theta_init)
            else:
                theta_warm[node, 0:nr_nodes] = theta_result[0:nr_nodes] + 0.01 * theta_init[node, 0:nr_nodes]
            thetas_main[ii, 0:nr_nodes] = theta_result[0:nr_nodes]

        for bb in range(self.lpgm_B):
            theta_warm = np.copy(theta_init)
            subsample_indices = np.random.choice(list(range(nr_datapoints)), self.lpgm_m, replace=False)
            subsampled_data_points = data_points[subsample_indices, :]
            for ii, alpha in enumerate(self.alpha_values):
                self.alpha = alpha
                theta_result = self.fit_node(node, f_nll, f_grad_nll, subsampled_data_points, theta_warm)[0]
                # theta_warm = np.copy(theta_init)
                if max(theta_result[0:nr_nodes]) > 2 or min(theta_result[0:nr_nodes] < -2):
                    theta_warm[node, 0:nr_nodes] = np.copy(theta_init)
                else:
                    theta_warm[node, 0:nr_nodes] = theta_result[0:nr_nodes] + 0.01 * theta_init[node, 0:nr_nodes]
                thetas_subsamples[bb, ii, 0:nr_nodes] = theta_result[0:nr_nodes]
        return thetas_main, thetas_subsamples
