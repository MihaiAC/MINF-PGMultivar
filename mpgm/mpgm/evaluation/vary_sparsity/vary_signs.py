from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter
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

def vary_signs(SPW:SampleParamsWrapper, split:float):
    weight_assigner = SPW.weight_assigner
    weight_assigner.split = split
    SPW.weight_assigner = weight_assigner

splits = [0, 0.2, 0.4, 0.6, 0.8, 1]

samples_file_name = "../samples.sqlite"
fit_file_name = "../fit_models.sqlite"

# Sparsity names:
experiment_name = "lattice_vary_signs_Gibbs_weight_normal"

nr_variables = 10
nr_samples = 150
alpha_Gibbs = 0.2

if __name__ == '__main__':
    SGW = SampleParamsWrapper(nr_variables=nr_variables,
                              sample_init=np.zeros((nr_variables, )),
                              nr_samples=nr_samples,
                              random_seed=-1)

    SGW.graph_generator = LatticeGraphGenerator(sparsity_level=0)

    # SGW.weight_assigner = Dummy_Weight_Assigner()
    # SGW.sampler = SIPRVSampler(lambda_true=1, lambda_noise=0.5)

    SGW.weight_assigner = Bimodal_Gaussian_Weight_Assigner(mean_1=-0.2,
                                                           std_1=0,
                                                           mean_2=0.2,
                                                           std_2=0,
                                                           split=0)
    SGW.sampler = TPGMGibbsSampler(burn_in=200, thinning_nr=150)

    SGW.model = TPGM(R=10)

    nr_batches = len(splits)

    FPW = FitParamsWrapper(random_seed=0,
                           samples_file_name=samples_file_name)
    FPW.model = TPGM(R=10)
    FPW.fitter = Prox_Grad_Fitter(alpha=alpha_Gibbs,
                                  accelerated=True,
                                  save_regularization_paths=False,
                                  early_stop_criterion='weight',
                                  keep_diag_zero=False)
    FPW.preprocessor = ClampMax(10)

    theta_init = np.random.normal(0, 0.05, (nr_variables, nr_variables))
    theta_init[np.tril_indices(nr_variables)] = 0
    theta_init = theta_init + theta_init.T

    experiment1 = Experiment(experiment_name=experiment_name,
                            random_seeds=list(range(5)),
                            SPW=SGW,
                            samples_file_name=samples_file_name,
                            FPW=FPW,
                            fit_file_name=fit_file_name,
                            vary_fit=False,
                            fit_theta_init=theta_init,
                            samples_name=experiment_name
                            )

    experiment1.vary_x_generate_samples(splits, vary_signs)
    experiment1.fit_all_samples_same_FPW(len(splits))

    stats_gen = StatsGenerator(experiments=[experiment1],
                               experiment_labels=[''],
                               var_name="% negative weights",
                               var_values=splits,
                               folder_name=experiment_name)

    stats_gen.plot_ALL("vary_signs", "% negative weights", splits, x_log_scale=False,
                       SIPRV_active=False)
