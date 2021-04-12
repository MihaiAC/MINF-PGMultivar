from mpgm.mpgm.model_fitters.prox_grad_fitters import LPGM_Fitter
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
experiment_base_name = Experiment.generate_experiment_name('TPGM',
                                                           'random_graph',
                                                           'TPGM',
                                                           'LPGM',
                                                           'likelihood',
                                                           )
nr_variables = 10
nr_samples = 150

# TODO: Include nr_variables to generate experiment name?
experiment_name = experiment_base_name + '_' + str(nr_variables) + '_variables'

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

    FPW = FitParamsWrapper(random_seed=0,
                           samples_file_name=samples_file_name)
    FPW.model = TPGM(R=10)

    FPW.fitter = LPGM_Fitter(init_step_size=1,
                             early_stop_criterion='likelihood',
                             nr_alphas=50,
                             beta=0.1)
    # FPW.preprocessor = ClampMax(10)

    theta_init = np.random.normal(0, 0.1, (nr_variables, nr_variables))
    theta_init[np.tril_indices(nr_variables)] = 0
    theta_init = theta_init + theta_init.T

    experiment = Experiment(experiment_name=experiment_name,
                            random_seeds=[0],
                            SPW=SGW,
                            samples_file_name=samples_file_name,
                            FPW=FPW,
                            fit_file_name=fit_file_name,
                            vary_fit=True,
                            fit_theta_init=theta_init,
                            fit_parallelize=False
                            )

    experiment.generate_single_batch_of_samples()
    experiment.fit_LPGM(50)

    # Getting the alphas.
    # TODO: Do overflows mess up the fit?
    sample_name = experiment.get_sample_name(0, 0)
    SPS = SampleParamsWrapper.load_samples(sample_name, samples_file_name)
    samples = SPS.samples
    alphas = FPW.fitter.generate_alpha_values(samples)
    alphas = list(alphas)

    stats_gen = StatsGenerator(experiments=[experiment],
                               experiment_labels=[''],
                               var_name="alpha",
                               var_values=alphas,
                               folder_name=experiment_base_name)
    stats_gen.plot_ALL("vary_alpha", "alpha", alphas, x_log_scale=True,
                       SIPRV_active=False)

