from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter, Pseudo_Likelihood_Prox_Grad_Fitter
from mpgm.mpgm.evaluation.evaluation_metrics import *

from mpgm.mpgm.sample_generation.samplers import *
from mpgm.mpgm.sample_generation.graph_generators import *
from mpgm.mpgm.sample_generation.weight_assigners import *

# from typing import List, Any, Callable
from mpgm.mpgm.evaluation.evaluation import Experiment
from mpgm.mpgm.evaluation.evaluation import StatsGenerator
from mpgm.mpgm.evaluation.evaluation_metrics import EvalMetrics

from mpgm.mpgm.evaluation.preprocessing import ClampMax

import matplotlib.pyplot as plt


def vary_nr_samples(SPW: SampleParamsWrapper, nr_samples: int):
    SPW.nr_samples = nr_samples


samples_file_name = "../samples.sqlite"
fit_file_name = "../fit_models.sqlite"
# experiment_base_name = "lattice_vary_nr_samples_SIPRV_weight_normal_alpha"
# experiment_base_name = "lattice_vary_nr_samples_SIPRV_likelihood_normal_alpha"
# experiment_base_name = "lattice_vary_nr_samples_Gibbs_likelihood_normal_alpha"

# Gibbs sampling experiments:
experiment_name1 = "lattice_vary_nr_samples_Gibbs_weight_normal_alpha_10_variables"
# experiment_name = "lattice_vary_nr_samples_Gibbs_weight_pseudo_likelihood_10_variables"

# SIPRV sampling experiments:
experiment_name = "lattice_vary_nr_samples_SIPRV_weight_normal"

nr_variables = 10
samples_numbers = [10, 20, 40, 80, 160, 320, 640]

if __name__ == '__main__':
    SGW = SampleParamsWrapper(nr_variables=nr_variables,
                              sample_init=np.zeros((nr_variables, )),
                              nr_samples=-1,
                              random_seed=-1)

    SGW.graph_generator = LatticeGraphGenerator(sparsity_level=0)
    SGW.weight_assigner = Dummy_Weight_Assigner()
    # SGW.weight_assigner = Constant_Weight_Assigner(ct_weight=-0.5)
    SGW.model = TPGM(R=10)
    # SGW.sampler = TPGMGibbsSampler(burn_in=200, thinning_nr=150)
    SGW.sampler = SIPRVSampler(lambda_true=1, lambda_noise=0.5)

    nr_batches = len(samples_numbers)

    FPW = FitParamsWrapper(random_seed=0,
                           samples_file_name=samples_file_name)
    FPW.model = TPGM(R=10)
    FPW.fitter = Prox_Grad_Fitter(alpha=0.1,
                                  early_stop_criterion='weight',
                                  accelerated=True)

    # FPW.fitter = Pseudo_Likelihood_Prox_Grad_Fitter(alpha=-1,
    #                                                 accelerated=True,
    #                                                 constraint_solver='soft',
    #                                                 save_regularization_paths=False,
    #                                                 early_stop_criterion='weight',
    #                                                 init_step_size=1,
    #                                                 keep_diag_zero=True)

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
                            fit_theta_init=theta_init,
                            vary_fit=False,
                            samples_name=experiment_name
                            )

    experiment1 = Experiment(experiment_name=experiment_name1,
                            random_seeds=list(range(5)),
                            SPW=SGW,
                            samples_file_name=samples_file_name,
                            FPW=FPW,
                            fit_file_name=fit_file_name,
                            fit_theta_init=theta_init,
                            vary_fit=False,
                            samples_name=experiment_name1
                            )

    # experiment.vary_x_generate_samples(samples_numbers, vary_nr_samples)
    # experiment.fit_all_samples_same_FPW(len(samples_numbers))

    stats_gen = StatsGenerator(experiments=[experiment, experiment1],
                               experiment_labels=['SIPRV', 'Gibbs'],
                               var_name="# samples",
                               var_values=samples_numbers,
                               folder_name=experiment_name)

    stats_gen.plot_ALL("vary_nr_samples", "# samples", samples_numbers, x_log_scale=False,
                       SIPRV_active=False)
