from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter
from mpgm.mpgm.evaluation.evaluation_metrics import *

from mpgm.mpgm.sample_generation.gibbs_samplers import *
from mpgm.mpgm.sample_generation.graph_generators import *
from mpgm.mpgm.sample_generation.weight_assigners import *

from typing import List, Any, Callable

class Experiment():
    def __init__(self, experiment_name: str, random_seeds: List[int], SPW: SampleParamsWrapper,
                 samples_file_name:str, FPW:FitParamsWrapper, fit_file_name:str):
        self.experiment_name = experiment_name
        self.random_seeds = random_seeds
        self.samples_per_batch = len(random_seeds)
        self.SPW = SPW
        self.samples_file_name = samples_file_name
        self.FPW = FPW
        self.fit_file_name = fit_file_name

    def generate_batch_samples_vary_seed(self, batch_nr:int):
        for sample_nr in range(self.samples_per_batch):
            self.SPW.random_seed = self.random_seeds[sample_nr]
            sample_name = self.get_sample_name(batch_nr, sample_nr)

            self.SPW.generate_samples_and_save(sample_name, self.samples_file_name)
            print("Sampling " + sample_name + " finished.")

    def get_sample_name(self, batch_nr: int, sample_nr: int) -> str:
        sample_name = self.experiment_name + "_batch_" + str(batch_nr) + "_sample_" + str(sample_nr)
        return sample_name

    def get_fit_name(self, batch_nr:int, sample_nr:int) -> str:
        return "fit_" + self.get_sample_name(batch_nr, sample_nr)

    def vary_x_generate_samples(self, xs:List[Any], f_vary:Callable[[SampleParamsWrapper, Any], None]):
        for batch_nr, x in enumerate(xs):
            f_vary(self.SPW, x)
            self.generate_batch_samples_vary_seed(batch_nr)

    def fit_all_samples_same_FPW(self, nr_batches:int):
        for batch_nr in range(nr_batches):
            for sample_nr in range(self.samples_per_batch):
                sample_name = self.get_sample_name(batch_nr, sample_nr)
                fit_name = self.get_fit_name(batch_nr, sample_nr)
                self.FPW.fit_model_and_save(fit_id=fit_name,
                                            fit_file_name=self.fit_file_name,
                                            parallelize=True,
                                            samples_file_name=self.samples_file_name,
                                            samples_id=sample_name)
