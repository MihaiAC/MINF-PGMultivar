from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter
from mpgm.mpgm.evaluation.evaluation_metrics import *

from mpgm.mpgm.sample_generation.samplers import *
from mpgm.mpgm.sample_generation.graph_generators import *
from mpgm.mpgm.sample_generation.weight_assigners import *

# from typing import List, Any, Callable
from mpgm.mpgm.evaluation.evaluation import Experiment
from mpgm.mpgm.evaluation.evaluation import StatsGenerator
from mpgm.mpgm.evaluation.evaluation_metrics import EvalMetrics

import matplotlib.pyplot as plt

def vary_alpha(FPW: FitParamsWrapper, alpha:int):
    FPW.fitter.alpha = alpha


samples_file_name = "../samples.sqlite"
fit_file_name = "../fit_models.sqlite"
experiment_base_name = "lattice_vary_alpha_SIPRV"
nr_variables = 10
nr_samples = 150
alphas = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
experiment_name = experiment_base_name + '_' + str(nr_variables) + '_variables'

if __name__ == '__main__':
    SGW = SampleParamsWrapper(nr_variables=nr_variables,
                              sample_init=np.zeros((nr_variables, )),
                              nr_samples=nr_samples,
                              random_seed=-1)

    SGW.graph_generator = LatticeGraphGenerator(sparsity_level=0)
    SGW.weight_assigner = Dummy_Weight_Assigner()
    SGW.model = TPGM(R=10)
    SGW.sampler = SIPRVSampler(lambda_true=1, lambda_noise=0.5)

    nr_batches = len(alphas)

    FPW = FitParamsWrapper(random_seed=0,
                           samples_file_name=samples_file_name)
    FPW.model = TPGM(R=10)
    FPW.fitter = Prox_Grad_Fitter(alpha=0.5, early_stop_criterion='weight')

    experiment = Experiment(experiment_name=experiment_name,
                            random_seeds=list(range(5)),
                            SPW=SGW,
                            samples_file_name=samples_file_name,
                            FPW=FPW,
                            fit_file_name=fit_file_name,
                            vary_fit=True
                            )

    experiment.generate_one_samples_batch()
    experiment.vary_x_fit_samples(alphas, vary_alpha)

    # stats_gen = StatsGenerator(experiment, "samples", samples_numbers)
    # stats_gen.plot_ys_common_xs_ALL("nr_variables=10, alpha=0.5, gibbs, weight", "nr_samples", samples_numbers)



# TPRs, FPRs, ACCs, symm_vals, symm_neighbourhoods, sparsities, min_nr_samples, node_KL_divergences = experiment.generate_stats(samples_numbers, 'nr_samples')
# fit_name = experiment.get_fit_name(10, 0)
# FPS = FitParamsWrapper.load_fit(fit_name, fit_file_name)
# print(FPS.theta_fit)
