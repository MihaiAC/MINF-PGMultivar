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


def vary_nr_samples(SPW: SampleParamsWrapper, nr_samples: int):
    SPW.nr_samples = nr_samples


samples_file_name = "../samples.sqlite"
fit_file_name = "../fit_models.sqlite"
experiment_base_name = "lattice_vary_nr_samples"
nr_variables = 10
experiment_name = experiment_base_name + '_' + str(nr_variables) + '_variables'

if __name__ == '__main__':
    SGW = SampleParamsWrapper(nr_variables=nr_variables,
                              sample_init=np.zeros((nr_variables, )),
                              nr_samples=-1,
                              random_seed=-1)

    SGW.graph_generator = LatticeGraphGenerator(sparsity_level=0)
    # First, keep only negative edges (don't want to concern myself with another confounding variable).
    SGW.weight_assigner = Constant_Weight_Assigner(-0.1)
    SGW.model = TPGM(R=10)
    SGW.sampler = TPGMGibbsSampler(burn_in=500,
                                   thinning_nr=150)

    samples_numbers = [20, 40, 80, 160, 320]
    nr_batches = len(samples_numbers)

    FPW = FitParamsWrapper(random_seed=0,
                           samples_file_name=samples_file_name)
    FPW.model = TPGM(R=10)
    FPW.fitter = Prox_Grad_Fitter(alpha=0.5, early_stop_criterion='weight')

    experiment = Experiment(experiment_name=experiment_name,
                            random_seeds=list(range(5)),
                            SPW=SGW,
                            samples_file_name=samples_file_name,
                            FPW=FPW,
                            fit_file_name=fit_file_name
                            )

    # experiment.vary_x_generate_samples(samples_numbers, vary_nr_samples)
    # experiment.fit_all_samples_same_FPW(len(samples_numbers))

    # TODO: Once the experiment has stopped running, we need to generate stats.
    #  Do it once + see what improvements you can do. Need a script + to get past this mental block.

    stats_gen = StatsGenerator(experiment, "samples", samples_numbers)
    stats_gen.plot_ys_common_xs_ALL("nr_variables=10, alpha=0.5, gibbs, weight", "nr_samples", samples_numbers)



# TPRs, FPRs, ACCs, symm_vals, symm_neighbourhoods, sparsities, min_nr_samples, node_KL_divergences = experiment.generate_stats(samples_numbers, 'nr_samples')
# fit_name = experiment.get_fit_name(10, 0)
# FPS = FitParamsWrapper.load_fit(fit_name, fit_file_name)
# print(FPS.theta_fit)
