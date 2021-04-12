from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter, Constrained_Prox_Grad_Fitter, Pseudo_Likelihood_Prox_Grad_Fitter
from mpgm.mpgm.evaluation.evaluation_metrics import *

from mpgm.mpgm.sample_generation.samplers import *
from mpgm.mpgm.sample_generation.graph_generators import *
from mpgm.mpgm.sample_generation.weight_assigners import *

# from typing import List, Any, Callable
from mpgm.mpgm.evaluation.evaluation import Experiment
from mpgm.mpgm.evaluation.evaluation import StatsGenerator
from mpgm.mpgm.evaluation.evaluation_metrics import EvalMetrics

from mpgm.mpgm.evaluation.preprocessing import ClampMax
from mpgm.mpgm.models.PGM import PGM

import matplotlib.pyplot as plt

def vary_alpha(FPW: FitParamsWrapper, alpha:int):
    FPW_fitter = FPW.fitter
    FPW_fitter.alpha = alpha
    FPW.fitter = FPW_fitter


samples_file_name = "../samples.sqlite"
fit_file_name = "../fit_models.sqlite"
# experiment_base_name = Experiment.generate_experiment_name('SPGM',
#                                                            'random_graph',
#                                                            'TPGM',
#                                                            'vary_alpha',
#                                                            'likelihood',
#                                                            'pseudo'
#                                                            )


# Gibbs sampling experiments:
# samples_name1 = "lattice_vary_alpha_Gibbs_weight"
# experiment_name1 = "lattice_vary_alpha_Gibbs_weight"

# SIPRV sampling experiments:
# samples_name = "lattice_vary_alpha_SIPRV_weight"
# experiment_name = "lattice_vary_alpha_SIPRV_weight"



# Pseudo likelihood experiments:
# experiment_name1 = "lattice_vary_alpha_Gibbs_weight"
# experiment_name = "lattice_vary_alpha_Gibbs_weight_pseudo_likelihood"

# SIPRV experiments:
# experiment_name1 = "lattice_vary_alpha_SIPRV_weight"
# experiment_name = "lattice_vary_alpha_SIPRV_weight_pseudo_likelihood"

# ADMM experiments:

experiment_name = 'SPGM_random_graph_TPGM_vary_alpha_likelihood_pseudo_admm'
samples_name = 'SPGM_random_graph_TPGM_vary_alpha_likelihood_qpoases_positive_weights'

# experiment_name1 = 'SPGM_random_graph_SIPRV_vary_alpha_likelihood_admm_10_variables'

# Gibbs:
# experiment_name = 'SPGM_random_graph_TPGM_vary_alpha_likelihood_admm'
# samples_name = 'SPGM_random_graph_TPGM_vary_alpha_likelihood_qpoases_positive_weights'
#
# experiment_name1 = 'SPGM_random_graph_SIPRV_vary_alpha_likelihood_admm'
# samples_name1 = 'SPGM_random_graph_SIPRV_vary_alpha_likelihood_none_10_variables'

# experiment_name = 'SPGM_random_graph_TPGM_vary_alpha_likelihood_pseudo_admm'
# samples_name = 'SPGM_random_graph_TPGM_vary_alpha_likelihood_pseudo_10_variables'
#
experiment_name1 = 'SPGM_random_graph_SIPRV_vary_alpha_likelihood_pseudo_admm'
samples_name1 = 'SPGM_random_graph_SIPRV_vary_alpha_likelihood_pseudo'

# experiment_name = 'SPGM_random_graph_TPGM_vary_alpha_likelihood_qpoases_positive_weights'

nr_variables = 10
nr_samples = 160
alphas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.6, 25.2, 50.4]
# alphas = list(np.linspace(1.6, 3.2, 10))

if __name__ == '__main__':
    SGW = SampleParamsWrapper(nr_variables=nr_variables,
                              sample_init=np.zeros((nr_variables, )),
                              nr_samples=nr_samples,
                              random_seed=-1)

    SGW.graph_generator = RandomNmGraphGenerator(nr_edges=10)
    # SGW.weight_assigner = Bimodal_Gaussian_Weight_Assigner(mean_1=0.05,
    #                                                        std_1=0,
    #                                                        mean_2=-0.05,
    #                                                        std_2=0,
    #                                                        split=0.7)
    SGW.weight_assigner = Constant_Weight_Assigner(0.1)
    # SGW.weight_assigner = Dummy_Weight_Assigner()
    SGW.model = TPGM(R=10)
    # SGW.model = Model(theta=np.zeros((nr_samples, nr_variables)))
    SGW.sampler = TPGMGibbsSampler(burn_in=200, thinning_nr=150)
    # SGW.sampler = SIPRVSampler(lambda_true=1, lambda_noise=0.5)

    nr_batches = len(alphas)

    FPW = FitParamsWrapper(random_seed=0,
                           samples_file_name=samples_file_name)
    FPW.model = TPGM(R=10)

    # FPW.fitter = Constrained_Prox_Grad_Fitter(alpha=-1,
    #                                           accelerated=True,
    #                                           constraint_solver='admm',
    #                                           save_regularization_paths=False,
    #                                           init_step_size=0.1,
    #                                           early_stop_criterion='likelihood',
    #                                           keep_diag_zero=True)
    # FPW.fitter = Prox_Grad_Fitter(alpha=-1,
    #                               accelerated=True,
    #                               save_regularization_paths=False,
    #                               early_stop_criterion='weight',
    #                               keep_diag_zero=False)
    FPW.fitter = Pseudo_Likelihood_Prox_Grad_Fitter(alpha=-1,
                                                    accelerated=True,
                                                    constraint_solver='admm',
                                                    save_regularization_paths=False,
                                                    early_stop_criterion='likelihood',
                                                    init_step_size=0.1,
                                                    keep_diag_zero=False)
    FPW.preprocessor = ClampMax(10)

    theta_init = np.random.normal(0, 0.05, (nr_variables, nr_variables))
    theta_init[np.tril_indices(nr_variables)] = 0
    theta_init = theta_init + theta_init.T

    experiment = Experiment(experiment_name=experiment_name,
                            random_seeds=list(range(5)),
                            SPW=SGW,
                            samples_file_name=samples_file_name,
                            FPW=FPW,
                            fit_file_name=fit_file_name,
                            vary_fit=True,
                            fit_theta_init=theta_init,
                            fit_parallelize=True,
                            samples_name=samples_name
                            )

    experiment1 = Experiment(experiment_name=experiment_name1,
                            random_seeds=list(range(5)),
                            SPW=SGW,
                            samples_file_name=samples_file_name,
                            FPW=FPW,
                            fit_file_name=fit_file_name,
                            vary_fit=True,
                            fit_theta_init=theta_init,
                            fit_parallelize=True,
                            samples_name=samples_name1
                            )

    # experiment.generate_single_batch_of_samples()
    # experiment.vary_x_fit_samples(alphas, vary_alpha)
    # experiment1.vary_x_fit_samples(alphas, vary_alpha)

    stats_gen = StatsGenerator(experiments=[experiment, experiment1],
                               experiment_labels=['TPGM', 'SIPRV'],
                               var_name="alpha",
                               var_values=alphas,
                               folder_name=experiment_name)

    stats_gen.plot_ALL("SPGM_admm_pseudo_vary_alpha", "alpha", alphas, x_log_scale=True,
                       SIPRV_active=False)

