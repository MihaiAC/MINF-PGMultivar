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

def vary_sparsity(SPW:SampleParamsWrapper, nr_edges:int):
    SPW.graph_generator = RandomNmGraphGenerator(nr_edges=nr_edges)

sparsities = []
edges_nrs = list(range(46))
for nr_edges in edges_nrs:
    gg = RandomNmGraphGenerator(nr_edges)
    G = gg.generate_graph(10)
    sparsity = EvalMetrics.calculate_percentage_sparsity(G) * 100
    sparsities.append(sparsity)

samples_file_name = "../samples.sqlite"
fit_file_name = "../fit_models.sqlite"
experiment_base_name = "lattice_vary_sparsity_SIPRV_weight_normal_alpha"
# experiment_base_name = "lattice_vary_sparsity_Gibbs_weight"
nr_variables = 10
nr_samples = 150
alpha_SIPRV = 3.2
alpha_Gibbs = 0.4

experiment_name = experiment_base_name + '_' + str(nr_variables) + '_variables'

if __name__ == '__main__':
    SGW = SampleParamsWrapper(nr_variables=nr_variables,
                              sample_init=np.zeros((nr_variables, )),
                              nr_samples=nr_samples,
                              random_seed=-1)

    SGW.graph_generator = LatticeGraphGenerator(sparsity_level=0)

    SGW.weight_assigner = Dummy_Weight_Assigner()
    SGW.sampler = SIPRVSampler(lambda_true=1, lambda_noise=0.5)

    # SGW.weight_assigner = Constant_Weight_Assigner(ct_weight=-0.1)
    # SGW.sampler = TPGMGibbsSampler(burn_in=200, thinning_nr=150)

    SGW.model = TPGM(R=10)

    nr_batches = len(edges_nrs)

    FPW = FitParamsWrapper(random_seed=0,
                           samples_file_name=samples_file_name)
    FPW.model = TPGM(R=10)
    FPW.fitter = Prox_Grad_Fitter(alpha=alpha_SIPRV, early_stop_criterion='weight')
    FPW.preprocessor = ClampMax(10)

    experiment = Experiment(experiment_name=experiment_name,
                            random_seeds=list(range(5)),
                            SPW=SGW,
                            samples_file_name=samples_file_name,
                            FPW=FPW,
                            fit_file_name=fit_file_name,
                            vary_fit=False
                            )

    experiment.vary_x_generate_samples(edges_nrs, vary_sparsity)
    experiment.fit_all_samples_same_FPW(len(edges_nrs))

    stats_gen = StatsGenerator(experiment, "sample_sparsity", sparsities)
    stats_gen.plot_ALL("nr_variables=10, alpha=3.2, SIPRV, weight", "sample_sparsity", sparsities)
    # Don't forget to upgrade the title of the graph.
