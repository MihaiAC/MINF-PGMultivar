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

def vary_alpha(FPW: FitParamsWrapper, alpha:int):
    FPW_fitter = FPW.fitter
    FPW_fitter.alpha = alpha
    FPW.fitter = FPW_fitter


samples_file_name = "../samples.sqlite"
fit_file_name = "../fit_models.sqlite"
# experiment_base_name = "lattice_vary_alpha_Gibbs"
# experiment_base_name = "lattice_vary_alpha_SIPRV"
# experiment_base_name = "lattice_vary_alpha_SIPRV_likelihood"
# experiment_base_name = "lattice_vary_alpha_Gibbs_likelihood"
experiment_base_name = "random_vary_alpha_Gibbs_pseudo_likelihood"
nr_variables = 10
nr_samples = 150
alphas = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.6, 25.2, 50.4]
experiment_name = experiment_base_name + '_' + str(nr_variables) + '_variables'

if __name__ == '__main__':
    SGW = SampleParamsWrapper(nr_variables=nr_variables,
                              sample_init=np.zeros((nr_variables, )),
                              nr_samples=nr_samples,
                              random_seed=-1)

    SGW.graph_generator = RandomNmGraphGenerator(nr_edges=10)
    SGW.weight_assigner = Constant_Weight_Assigner(ct_weight=-0.5)
    # SGW.weight_assigner = Dummy_Weight_Assigner()
    SGW.model = TPGM(R=10)
    SGW.sampler = TPGMGibbsSampler(burn_in=200, thinning_nr=150)
    # SGW.sampler = SIPRVSampler(lambda_true=1, lambda_noise=0.5)

    nr_batches = len(alphas)

    FPW = FitParamsWrapper(random_seed=0,
                           samples_file_name=samples_file_name)
    FPW.model = TPGM(R=10)
    # FPW.fitter = Prox_Grad_Fitter(alpha=0.5,
    #                               early_stop_criterion='likelihood')
    FPW.fitter = Pseudo_Likelihood_Prox_Grad_Fitter(alpha=-1,
                                                    accelerated=True,
                                                    save_regularization_paths=False,
                                                    early_stop_criterion='likelihood',
                                                    init_step_size=1,
                                                    keep_diag_zero=True)
    FPW.preprocessor = ClampMax(10)

    # This is not symmetric, which is an issue.
    theta_init = np.random.normal(0, 0.1, (nr_variables, nr_variables))
    theta_init[np.tril_indices(nr_variables)] = 0
    theta_init = theta_init + theta_init.T

    # Now that the starting parameters are symmetric, let's see if the final parameters are also symmetric.
    # The initial results suggested that is not the case.

    experiment = Experiment(experiment_name=experiment_name,
                            random_seeds=list(range(5)),
                            SPW=SGW,
                            samples_file_name=samples_file_name,
                            FPW=FPW,
                            fit_file_name=fit_file_name,
                            vary_fit=True,
                            fit_theta_init=theta_init,
                            fit_parallelize=False
                            )

    # experiment.generate_single_batch_of_samples()
    experiment.vary_x_fit_samples(alphas, vary_alpha)

    stats_gen = StatsGenerator(experiments=[experiment],
                               experiment_labels=[''],
                               var_name='alpha',
                               var_values=alphas,
                               folder_name=experiment_name)
    stats_gen.plot_ALL("vary_alpha", "alpha", alphas, x_log_scale=True,
                       SIPRV_active=False)


