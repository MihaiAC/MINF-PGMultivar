from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter
from mpgm.mpgm.evaluation.evaluation_metrics import *

from mpgm.mpgm.sample_generation.gibbs_samplers import *
from mpgm.mpgm.sample_generation.graph_generators import *
from mpgm.mpgm.sample_generation.weight_assigners import *

# from typing import List, Any, Callable
from mpgm.mpgm.evaluation.evaluation import Experiment
from mpgm.mpgm.evaluation.evaluation import StatsGenerator

from mpgm.mpgm.evaluation.evaluation_metrics import EvalMetrics

import matplotlib.pyplot as plt

def vary_nr_samples(SPW:SampleParamsWrapper, nr_samples:int):
    SPW.nr_samples = nr_samples

samples_file_name = "../samples.sqlite"
fit_file_name = "../fit_models.sqlite"
experiment_base_name = "lattice_vary_nr_samples"
experiment_names = []
nr_variables = [40, 80, 120, 240]
for nr in nr_variables:
    experiment_names.append(experiment_base_name + '_' + str(nr) + '_variables')

if __name__ == '__main__':

    for ii, nr_vars in [(0, 40)]:
        SGW = SampleParamsWrapper(nr_variables=nr_vars,
                                  sample_init=np.zeros((nr_vars, )),
                                  nr_samples=-1,
                                  random_seed=-1)

        SGW.graph_generator = LatticeGraphGenerator(sparsity_level=0)
        SGW.weight_assigner = Bimodal_Distr_Weight_Assigner(neg_mean=-0.1, pos_mean=0.1, std=0, threshold=0.5)
        SGW.model = TPGM(R=10)
        SGW.sampler = TPGMGibbsSampler(burn_in = 100,
                                       thinning_nr = 50)


        samples_numbers = [100, 200, 400, 800, 1600]
        nr_batches = len(samples_numbers)

        FPW = FitParamsWrapper(random_seed=2,
                               samples_file_name=samples_file_name)
        FPW.model = TPGM(R=10)
        FPW.fitter = Prox_Grad_Fitter(alpha=0.065, early_stop_criterion='weight')

        experiment = Experiment(experiment_name=experiment_names[ii],
                                random_seeds=list(range(5)),
                                SPW=SGW,
                                samples_file_name=samples_file_name,
                                FPW=FPW,
                                fit_file_name=fit_file_name
                                )

        # experiment.vary_x_generate_samples(samples_numbers, vary_nr_samples)
        # experiment.fit_all_samples_same_FPW(len(samples_numbers))
        stats_gen = StatsGenerator(experiment, "samples", samples_numbers)
        print(stats_gen.get_theta_orig(0, 0))
        print(stats_gen.get_theta_fit(0, 0))


    # TPRs, FPRs, ACCs, symm_vals, symm_neighbourhoods, sparsities, min_nr_samples, node_KL_divergences = experiment.generate_stats(samples_numbers, 'nr_samples')
    # fit_name = experiment.get_fit_name(10, 0)
    # FPS = FitParamsWrapper.load_fit(fit_name, fit_file_name)
    # print(FPS.theta_fit)