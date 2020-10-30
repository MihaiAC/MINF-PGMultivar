# Import sample generation functions.
from mpgm.mpgm.sample_generation.gibbs_samplers import *
from mpgm.mpgm.sample_generation.graph_generators import *
from mpgm.mpgm.sample_generation.weight_assigners import *
from sqlitedict import SqliteDict
import numpy as np

class SampleParamsSave():
    def __init__(self):
        self.nr_variables = None
        self.seed = None

        self.graph_generator_name = None
        self.graph_generator_params = None

        self.weight_assigner_name = None
        self.weight_assigner_params = None

        self.gibbs_sampler_name = None
        self.gibbs_sampler_params = None

        self.model_name = None
        self.model_params = None

        self.sample = None

def save_params_to_sqlite(sample_id:str, params:SampleParamsSave, sqlite_file_name:str):
    samples_dict = SqliteDict('./' + sqlite_file_name, autocommit=True)
    samples_dict[sample_id] = params
    samples_dict.close()

# Sample generation begins here.

# General parameters.
sqlite_file_name = "samples.sqlite"
SPS = SampleParamsSave()
sample_params.nr_variables = 5
sample_params.seed = 1