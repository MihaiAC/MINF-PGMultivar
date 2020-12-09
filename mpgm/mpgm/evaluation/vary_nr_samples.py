from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter
from mpgm.mpgm.evaluation.evaluation_metrics import *

from mpgm.mpgm.sample_generation.gibbs_samplers import *
from mpgm.mpgm.sample_generation.graph_generators import *
from mpgm.mpgm.sample_generation.weight_assigners import *

# from typing import List, Any, Callable
from evaluation_generation import Experiment

def vary_nr_samples(SPW:SampleParamsWrapper, nr_samples:int):
    SPW.nr_samples = nr_samples

samples_file_name = "samples.sqlite"
fit_file_name = "fit_models.sqlite"
experiment_name = "hub_vary_nr_samples"


SGW = SampleParamsWrapper(nr_variables=4,
                          sample_init=np.zeros((4, )),
                          nr_samples=-1,
                          random_seed=-1)

SGW.graph_generator = HubGraphGenerator(nr_hubs=1)
SGW.weight_assigner = Bimodal_Distr_Weight_Assigner(threshold=0.9)
SGW.model = TPGM(R=10)
SGW.sampler = TPGMGibbsSampler(burn_in = 200,
                               thinning_nr = 300)


samples_numbers = [10, 30, 100, 300, 600]
nr_batches = len(samples_numbers)

FPW = FitParamsWrapper(random_seed=2,
                       samples_file_name=samples_file_name)
FPW.model = TPGM(R=10)
FPW.fitter = Prox_Grad_Fitter(alpha=0.065, early_stop_criterion='weight')

experiment = Experiment(experiment_name=experiment_name,
                        random_seeds=list(range(5)),
                        SPW=SGW,
                        samples_file_name=samples_file_name,
                        FPW=FPW,
                        fit_file_name=fit_file_name
                        )

if __name__ == '__main__':
    # experiment.vary_x_generate_samples(samples_numbers, vary_nr_samples)
    experiment.fit_all_samples_same_FPW(len(samples_numbers))