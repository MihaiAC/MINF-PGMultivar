# Import sample generation functions.
from mpgm.mpgm.sample_generation.gibbs_samplers import *
from mpgm.mpgm.sample_generation.graph_generators import *
from mpgm.mpgm.sample_generation.weight_assigners import *
from typing import Dict, Any, Type
from sqlitedict import SqliteDict
import numpy as np

class SampleParamsSave():
    def __init__(self):
        self.nr_variables = None
        self.nr_samples = None
        self.random_seed = None
        self.samples_init = None

        self.graph_generator = (None, None)
        self.weight_assigner = (None, None)
        self.model = (None, None)
        self.sampler = (None, None)

        self.samples = None

    @property
    def graph_generator_name(self):
        return self.graph_generator[0]

    @property
    def graph_generator_params(self):
        return self.graph_generator[1]

    @property
    def weight_assigner_name(self):
        return self.weight_assigner[0]

    @property
    def weight_assigner_params(self):
        return self.weight_assigner[1]

    @property
    def model_name(self):
        return self.model[0]

    @property
    def model_params(self):
        return self.model[1]

    @property
    def sampler_name(self):
        return self.sampler[0]

    @property
    def sampler_params(self):
        return self.sampler[1]

class SampleParamsWrapper():
    def __init__(self, nr_variables:int, nr_samples:int, random_seed:int, sample_init:np.array):
        self.SPS = SampleParamsSave()
        self.SPS.nr_variables = nr_variables
        self.SPS.nr_samples = nr_samples
        self.SPS.random_seed = random_seed
        self.SPS.samples_init = sample_init

        self._graph_generator = None
        self._weight_assigner = None
        self._model = None
        self._sampler = None

    @property
    def nr_variables(self):
        return self.SPS.nr_variables

    @property
    def nr_samples(self):
        return self.SPS.nr_samples

    @property
    def random_seed(self):
        return self.SPS.random_seed

    @property
    def samples_init(self):
        return self.SPS.samples_init

    @property
    def graph_generator(self):
        return self._graph_generator

    @graph_generator.setter
    def graph_generator(self, value:GraphGenerator):
        self._graph_generator = value
        self.SPS.graph_generator = (type(value).__name__, vars(value))

    @property
    def weight_assigner(self):
        return self._weight_assigner

    @weight_assigner.setter
    def weight_assigner(self, value:Weight_Assigner):
        self._weight_assigner = value
        self.SPS.weight_assigner = (type(value).__name__, vars(value))

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value:Model):
        self._model = value
        self.SPS.model = (type(value).__name__, vars(value))

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, value:GibbsSampler):
        self._sampler = value
        self.SPS.sampler = (type(value).__name__, vars(value))

    def generate_samples_and_save(self, sample_id:str, sqlite_file_name:str):
        np.random.seed(self.random_seed)
        graph = self.graph_generator.generate_graph(self.SPS.nr_variables)
        self.weight_assigner.assign_weights(graph)

        self.model.theta = graph
        self.model = self.model

        self.SPS.samples = self.sampler.generate_samples(self.model, self.SPS.samples_init, self.SPS.nr_samples)

        samples_dict = SqliteDict('./' + sqlite_file_name, autocommit=True)
        samples_dict[sample_id] = self.SPS
        samples_dict.close()

        return self.SPS.samples

    @staticmethod
    def load_samples(sample_id:str, sqlite_file_name:str) -> SampleParamsSave:
        samples_dict = SqliteDict('./' + sqlite_file_name, autocommit=True)
        return samples_dict[sample_id]

if __name__ == "__main__":
    sqlite_file_name = "samples.sqlite"

    SGW = SampleParamsWrapper(nr_variables=4,
                              nr_samples=10,
                              random_seed=0,
                              sample_init=np.zeros((4, )))

    SGW.graph_generator = HubGraphGenerator(nr_hubs=1)
    SGW.weight_assigner = Bimodal_Distr_Weight_Assigner(threshold=0.9)
    SGW.model = TPGM(R=10)
    SGW.sampler = TPGMGibbsSampler(burn_in = 20,
                                   thinning_nr = 10)

    #print(SGW.generate_samples_and_save("PleaseWork", sqlite_file_name))
    SPS = SampleParamsWrapper.load_samples("PleaseWork", sqlite_file_name)
